"""Base class for a sport scene: ground, static props, an athlete and a ball.

Rendering order (per frame):
  1. ground          - written to the stencil buffer (value 1)
  2. planar shadows  - casters are flattened onto y = 0 with the shadow
                       matrix and drawn in translucent black only where the
                       stencil is 1; the stencil is then reset to 0 so
                       overlapping shadows do not get darker (no "double blending")
  3. props, athlete and ball with full lighting and textures
"""
import numpy as np
from OpenGL.GL import (
    GL_ALWAYS, GL_BLEND, GL_DEPTH_TEST, GL_EQUAL, GL_KEEP, GL_LIGHTING,
    GL_ONE_MINUS_SRC_ALPHA, GL_REPLACE, GL_SRC_ALPHA, GL_STENCIL_TEST, GL_TEXTURE_2D, GL_ZERO,
    glBlendFunc, glColor4f, glDisable, glEnable, glMultMatrixf, glPopMatrix, glPushMatrix,
    glStencilFunc, glStencilOp,
)

from . import primitives as P
from .mathutil import shadow_matrix, to_gl

LIGHT_DIR = (0.45, 1.0, 0.35)   # direction *towards* the sun


class Scene:
    title = "scene"

    def __init__(self):
        self.athlete = None
        self.ball = None
        self.score = 0
        self.message = ""
        self.message_t = 0.0
        self._shadow = to_gl(shadow_matrix(LIGHT_DIR, plane_y=0.003))

    # ---- to be provided by subclasses ------------------------------------
    def draw_ground(self):
        raise NotImplementedError

    def draw_props(self):
        pass

    def draw_prop_shadows(self):
        pass

    def update(self, dt, controls):
        raise NotImplementedError

    def help_lines(self):
        return []

    def status(self):
        return ""

    # ---- shared -----------------------------------------------------------
    def flash(self, text, seconds=2.0):
        self.message, self.message_t = text, seconds

    def tick_message(self, dt):
        if self.message_t > 0:
            self.message_t -= dt
            if self.message_t <= 0:
                self.message = ""

    def focus_point(self):
        return self.athlete.pos + np.array([0.0, 1.15, 0.0])

    def render(self):
        glEnable(GL_STENCIL_TEST)
        glStencilFunc(GL_ALWAYS, 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)
        self.draw_ground()

        glStencilFunc(GL_EQUAL, 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.0, 0.0, 0.0, 0.38)
        P.RenderState.shadow_pass = True
        glPushMatrix()
        glMultMatrixf(self._shadow)
        self.athlete.render()
        if self.ball.pos[1] < 6.0:
            self.ball.render()
        self.draw_prop_shadows()
        glPopMatrix()
        P.RenderState.shadow_pass = False
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glDisable(GL_STENCIL_TEST)

        self.draw_props()
        self.athlete.render()
        self.ball.render()
