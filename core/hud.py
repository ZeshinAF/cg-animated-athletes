"""2D text overlay: pygame renders text to a surface, OpenGL copies the
pixels into the frame buffer with glDrawPixels."""
import pygame
from OpenGL.GL import (
    GL_BLEND, GL_DEPTH_TEST, GL_FOG, GL_LIGHTING, GL_ONE_MINUS_SRC_ALPHA, GL_RGBA,
    GL_SRC_ALPHA, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBlendFunc, glDisable, glDrawPixels,
    glEnable, glWindowPos2i,
)


class Hud:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 16)
        self.big = pygame.font.SysFont("arial,dejavusans,sans", 64, bold=True)
        self._cache = {}

    def _image(self, text, font, color):
        key = (text, id(font), color)
        img = self._cache.get(key)
        if img is None:
            fg = font.render(text, True, color)
            shadow = font.render(text, True, (0, 0, 0))
            surf = pygame.Surface((fg.get_width() + 2, fg.get_height() + 2), pygame.SRCALPHA)
            surf.blit(shadow, (2, 2))
            surf.blit(fg, (0, 0))
            data = pygame.image.tostring(surf, "RGBA", True)  # flipped for OpenGL
            img = (surf.get_width(), surf.get_height(), data)
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = img
        return img

    def begin(self):
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_FOG)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def end(self):
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_FOG)

    def text(self, x, y_from_top, text, screen_h, color=(255, 255, 255), big=False):
        w, h, data = self._image(text, self.big if big else self.font, color)
        glWindowPos2i(int(x), int(screen_h - y_from_top - h))
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)
        return w, h

    def centered(self, y_from_top, text, screen_w, screen_h, color=(255, 230, 80)):
        w, _, _ = self._image(text, self.big, color)
        self.text((screen_w - w) // 2, y_from_top, text, screen_h, color, big=True)
