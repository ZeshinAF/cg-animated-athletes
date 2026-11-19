"""Very small ball simulation: gravity, bounces, rolling friction and the
rotation of the ball (so its texture visibly rolls and spins)."""
import math

import numpy as np
from OpenGL.GL import glMultMatrixf, glPopMatrix, glPushMatrix, glTranslatef

from . import primitives as P
from .mathutil import axis_angle, to_gl

GRAVITY = 9.81
UP = np.array([0.0, 1.0, 0.0])


class Ball:
    def __init__(self, radius, tex, restitution=0.6, rolling_friction=0.8, air_drag=0.05):
        self.r = radius
        self.tex = tex
        self.restitution = restitution
        self.rolling_friction = rolling_friction
        self.air_drag = air_drag
        self.pos = np.array([0.0, radius, 0.0])
        self.vel = np.zeros(3)
        self.omega = np.zeros(3)                 # angular velocity (rad/s)
        self.rot = np.identity(3, dtype=np.float32)
        self.held = False                         # position driven by an athlete
        self.bounds = None                        # (xmin, xmax, zmin, zmax) walls

    @property
    def on_ground(self):
        return self.pos[1] <= self.r + 1e-3

    def place(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.vel[:] = 0.0
        self.omega[:] = 0.0

    def carry_to(self, new_pos, dt, spin=None):
        """Move a held ball; its velocity is derived from the displacement so
        that it rolls naturally when dribbled on the ground."""
        new_pos = np.asarray(new_pos, dtype=float)
        if dt > 0:
            self.vel = (new_pos - self.pos) / dt
        self.pos = new_pos
        if spin is not None:
            self.omega = spin
        elif self.on_ground:
            self.omega = np.cross(UP, self.vel) / self.r
        self._integrate_rotation(dt)

    def launch(self, velocity, spin=(0.0, 0.0, 0.0)):
        self.held = False
        self.vel = np.asarray(velocity, dtype=float)
        self.omega = np.asarray(spin, dtype=float)

    def _integrate_rotation(self, dt):
        w = float(np.linalg.norm(self.omega))
        if w > 1e-6:
            self.rot = axis_angle(self.omega / w, w * dt) @ self.rot

    def update(self, dt):
        if self.held:
            return
        self.vel[1] -= GRAVITY * dt
        self.vel *= max(0.0, 1.0 - self.air_drag * dt)
        self.pos += self.vel * dt
        if self.pos[1] < self.r:                       # ground contact
            self.pos[1] = self.r
            if self.vel[1] < -0.8:
                self.vel[1] = -self.vel[1] * self.restitution
            else:
                self.vel[1] = 0.0
        if self.on_ground:                             # rolling without slipping
            k = max(0.0, 1.0 - self.rolling_friction * dt)
            self.vel[0] *= k
            self.vel[2] *= k
            self.omega = np.cross(UP, self.vel) / self.r
        if self.bounds:
            x0, x1, z0, z1 = self.bounds
            for i, lo, hi in ((0, x0, x1), (2, z0, z1)):
                if self.pos[i] < lo + self.r:
                    self.pos[i] = lo + self.r
                    self.vel[i] = abs(self.vel[i]) * 0.5
                elif self.pos[i] > hi - self.r:
                    self.pos[i] = hi - self.r
                    self.vel[i] = -abs(self.vel[i]) * 0.5
        self._integrate_rotation(dt)

    def bounce_off_point(self, point, extra_radius, restitution=0.6):
        """Collision with a thin object approximated by its closest point
        (goal posts, the basketball rim). Returns True on contact."""
        d = self.pos - point
        dist = float(np.linalg.norm(d))
        lim = self.r + extra_radius
        if 1e-6 < dist < lim:
            n = d / dist
            self.pos = point + n * lim
            vn = float(self.vel @ n)
            if vn < 0:
                self.vel -= (1.0 + restitution) * vn * n
            return True
        return False

    def speed(self):
        return float(np.linalg.norm(self.vel))

    def render(self):
        glPushMatrix()
        glTranslatef(*self.pos)
        m = np.identity(4, dtype=np.float32)
        m[:3, :3] = self.rot
        glMultMatrixf(to_gl(m))
        P.material((1, 1, 1), self.tex)
        P.sphere(self.r, slices=32, stacks=20)
        glPopMatrix()


def ballistic_velocity(p0, target, flight_time):
    """Initial velocity so a projectile starting at p0 reaches target after
    flight_time seconds:  p(t) = p0 + v t + 1/2 g t^2  ->  v = (target - p0 - 1/2 g T^2) / T."""
    p0, target = np.asarray(p0, float), np.asarray(target, float)
    g = np.array([0.0, -GRAVITY, 0.0])
    return (target - p0 - 0.5 * g * flight_time ** 2) / flight_time


def heading_vector(heading_deg):
    a = math.radians(heading_deg)
    return np.array([math.sin(a), 0.0, math.cos(a)])
