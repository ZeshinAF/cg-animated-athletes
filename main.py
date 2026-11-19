"""Animated, textured athletes - Computer Graphics course project.

Run:  python main.py
Controls are listed in README.md and on screen (H toggles the help).
"""
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pygame
from OpenGL.GL import (
    GL_AMBIENT, GL_COLOR_BUFFER_BIT, GL_COLOR_MATERIAL, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_DIFFUSE, GL_FOG, GL_FOG_COLOR, GL_FOG_END, GL_FOG_MODE, GL_FOG_START,
    GL_FRONT_AND_BACK, GL_LIGHT0, GL_LIGHTING, GL_LIGHT_MODEL_AMBIENT, GL_LINE_SMOOTH,
    GL_LINEAR, GL_MODELVIEW, GL_MULTISAMPLE, GL_NORMALIZE, GL_ONE_MINUS_SRC_ALPHA, GL_POSITION,
    GL_PROJECTION, GL_QUADS, GL_RGB, GL_SHININESS, GL_SMOOTH, GL_SPECULAR,
    GL_SRC_ALPHA, GL_STENCIL_BUFFER_BIT, GL_TEXTURE_2D, GL_UNSIGNED_BYTE,
    GL_AMBIENT_AND_DIFFUSE, glBegin, glBlendFunc, glClear, glClearStencil, glColor3f,
    glColorMaterial, glDepthMask, glDisable, glEnable, glEnd, glFogf, glFogfv, glFogi,
    glLightModelfv, glLightfv, glLoadIdentity, glMaterialf, glMaterialfv, glMatrixMode,
    glReadPixels, glShadeModel, glVertex2f, glViewport,
)
from OpenGL.GLU import gluPerspective

from core.camera import OrbitCamera
from core.hud import Hud
from core.scene import LIGHT_DIR
from sports.basketball import BasketballScene
from sports.football import FootballScene

SKY_TOP = (0.36, 0.58, 0.88)
SKY_HORIZON = (0.78, 0.86, 0.95)


@dataclass
class Controls:
    move: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world-space direction
    sprint: bool = False
    action: bool = False      # edge-triggered
    alt: bool = False
    reset: bool = False


class App:
    def __init__(self, width=1280, height=720, title="Animated Athletes - OpenGL"):
        pygame.init()
        for attr, val in ((pygame.GL_DEPTH_SIZE, 24), (pygame.GL_STENCIL_SIZE, 8),
                          (pygame.GL_MULTISAMPLEBUFFERS, 1), (pygame.GL_MULTISAMPLESAMPLES, 4)):
            pygame.display.gl_set_attribute(attr, val)
        flags = pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE
        try:
            pygame.display.set_mode((width, height), flags)
        except pygame.error:   # no multisampling available
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
            pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption(title)
        self.width, self.height = width, height
        self._init_gl()
        self.hud = Hud()
        self.scenes = [FootballScene(), BasketballScene()]
        self.cameras = [OrbitCamera(yaw=s.camera_yaw) for s in self.scenes]
        for s, c in zip(self.scenes, self.cameras):
            c.target = s.focus_point().copy()
        self.active = 0
        self.show_help = True
        self.fps = 0.0
        self.dragging = False
        self.want_screenshot = False

    @property
    def scene(self):
        return self.scenes[self.active]

    @property
    def camera(self):
        return self.cameras[self.active]

    # ------------------------------------------------------------- OpenGL
    def _init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_NORMALIZE)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.97, 0.9, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.6, 0.6, 0.6, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.42, 0.44, 0.5, 1.0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.25, 0.25, 0.25, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 24.0)
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogfv(GL_FOG_COLOR, (*SKY_HORIZON, 1.0))
        glFogf(GL_FOG_START, 45.0)
        glFogf(GL_FOG_END, 150.0)
        glClearStencil(0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _sky(self):
        """Vertical gradient drawn as a full-screen quad behind everything."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_FOG)
        glDepthMask(False)
        glBegin(GL_QUADS)
        glColor3f(*SKY_HORIZON)
        glVertex2f(-1, -1)
        glVertex2f(1, -1)
        glColor3f(*SKY_TOP)
        glVertex2f(1, 1)
        glVertex2f(-1, 1)
        glEnd()
        glDepthMask(True)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_FOG)

    # --------------------------------------------------------------- input
    def read_controls(self, events):
        ctl = Controls()
        for e in events:
            if e.type == pygame.QUIT:
                return None
            if e.type == pygame.VIDEORESIZE:
                self.width, self.height = max(e.w, 320), max(e.h, 240)
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return None
                if e.key == pygame.K_SPACE:
                    ctl.action = True
                elif e.key == pygame.K_q:
                    ctl.alt = True
                elif e.key == pygame.K_r:
                    ctl.reset = True
                elif e.key == pygame.K_TAB:
                    self.active = (self.active + 1) % len(self.scenes)
                elif e.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif e.key == pygame.K_F12:
                    self.want_screenshot = True
            if e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    self.dragging = True
                elif e.button == 3:
                    ctl.action = True
            if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self.dragging = False
            if e.type == pygame.MOUSEMOTION and self.dragging:
                self.camera.drag(*e.rel)
            if e.type == pygame.MOUSEWHEEL:
                self.camera.zoom(e.y)

        keys = pygame.key.get_pressed()
        fwd, right = self.camera.ground_axes()
        f = (keys[pygame.K_w] or keys[pygame.K_UP]) - (keys[pygame.K_s] or keys[pygame.K_DOWN])
        r = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        move = fwd * f + right * r
        n = np.linalg.norm(move)
        ctl.move = move / n if n > 0 else move
        ctl.sprint = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        return ctl

    # ------------------------------------------------------------- frame
    def step(self, dt, ctl):
        self.scene.update(dt, ctl)
        self.camera.follow(self.scene.focus_point(), dt)

    def render(self):
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        self._sky()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(55.0, self.width / float(self.height), 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.camera.apply()
        glLightfv(GL_LIGHT0, GL_POSITION, (*LIGHT_DIR, 0.0))  # directional, set in eye space
        self.scene.render()
        self._hud()

    def _hud(self):
        s, h = self.scene, self.height
        self.hud.begin()
        y = 10
        _, lh = self.hud.text(12, y, f"{s.title}   |   {s.status()}", h, (255, 255, 120))
        y += lh + 2
        fps = f"FPS {self.fps:5.1f}   " if self.fps > 0 else ""
        self.hud.text(12, y, f"{fps}Tab: switch athlete   H: help", h, (220, 220, 220))
        y += lh + 8
        if self.show_help:
            for line in s.help_lines() + ["Left mouse drag  orbit camera,  wheel  zoom",
                                          "F12  screenshot,  Esc  quit"]:
                _, lh = self.hud.text(12, y, line, h)
                y += lh
        if s.message:
            self.hud.centered(int(h * 0.18), s.message, self.width, h)
        self.hud.end()

    def screenshot(self, path=None):
        os.makedirs("screenshots", exist_ok=True)
        path = path or time.strftime("screenshots/shot_%Y%m%d_%H%M%S.png")
        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        surf = pygame.image.fromstring(data, (self.width, self.height), "RGB", True)
        pygame.image.save(surf, path)
        return path

    def run(self):
        clock = pygame.time.Clock()
        while True:
            dt = min(clock.tick(120) / 1000.0, 1 / 20)
            ctl = self.read_controls(pygame.event.get())
            if ctl is None:
                break
            self.step(dt, ctl)
            self.render()
            if self.want_screenshot:      # read the back buffer before it is swapped
                self.want_screenshot = False
                print("saved", self.screenshot())
            pygame.display.flip()
            self.fps = clock.get_fps()
        pygame.quit()


def main():
    App().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
