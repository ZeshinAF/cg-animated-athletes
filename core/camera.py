"""Third-person orbit camera controlled with the mouse."""
import math

import numpy as np
from OpenGL.GLU import gluLookAt

from .mathutil import clamp


class OrbitCamera:
    def __init__(self, yaw=200.0, pitch=18.0, distance=7.0):
        self.yaw = yaw            # degrees, rotation around the target (Y axis)
        self.pitch = pitch        # degrees above the horizon
        self.distance = distance  # metres from the target
        self.target = np.array([0.0, 1.2, 0.0])
        self.follow_speed = 6.0

    # --- mouse input -------------------------------------------------------
    def drag(self, dx, dy, sensitivity=0.3):
        self.yaw -= dx * sensitivity
        self.pitch = clamp(self.pitch + dy * sensitivity, -5.0, 80.0)

    def zoom(self, steps):
        self.distance = clamp(self.distance * (0.9 ** steps), 2.5, 25.0)

    # --- per frame -----------------------------------------------------------
    def follow(self, point, dt):
        """Exponential smoothing towards the followed point (frame-rate independent)."""
        k = 1.0 - math.exp(-self.follow_speed * dt)
        self.target += (np.asarray(point, dtype=float) - self.target) * k

    def eye(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        offset = np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])
        return self.target + offset * self.distance

    def ground_axes(self):
        """Forward/right vectors on the ground plane, used for camera-relative movement."""
        y = math.radians(self.yaw)
        forward = np.array([-math.sin(y), 0.0, -math.cos(y)])
        right = np.array([math.cos(y), 0.0, -math.sin(y)])
        return forward, right

    def apply(self):
        e, t = self.eye(), self.target
        gluLookAt(e[0], e[1], e[2], t[0], t[1], t[2], 0.0, 1.0, 0.0)
