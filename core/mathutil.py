"""Small linear-algebra helpers (4x4 homogeneous matrices, interpolation).

All matrices are row-major numpy arrays that act on column vectors
(p' = M @ p).  OpenGL expects column-major order, so they are transposed
with :func:`to_gl` before being passed to glMultMatrixf.
"""
import math

import numpy as np


def identity():
    return np.identity(4, dtype=np.float32)


def translate(x, y, z):
    m = identity()
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def rotate(axis, deg):
    """Right-handed rotation around a principal axis ('x', 'y' or 'z')."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = identity()
    if axis == "x":
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif axis == "y":
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def axis_angle(axis, angle):
    """3x3 rotation matrix from a unit axis and an angle in radians (Rodrigues)."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ], dtype=np.float32)


def to_gl(m):
    """Row-major numpy matrix -> column-major float array for OpenGL."""
    return np.ascontiguousarray(m.T, dtype=np.float32)


def shadow_matrix(light_dir, plane_y=0.0):
    """Planar projection matrix that flattens geometry onto the plane y = plane_y
    along a directional light (Blinn, 'Me and my (fake) shadow', 1988).

    M = (P . L) I - L P^T, where P is the plane and L the homogeneous light."""
    lx, ly, lz = light_dir
    light = np.array([lx, ly, lz, 0.0], dtype=np.float32)
    plane = np.array([0.0, 1.0, 0.0, -plane_y], dtype=np.float32)
    return float(plane @ light) * np.identity(4, dtype=np.float32) - np.outer(light, plane)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def smoothstep(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_cos(t):
    """Cosine ease-in/ease-out used between animation keyframes."""
    return 0.5 - 0.5 * math.cos(math.pi * clamp(t, 0.0, 1.0))


def angle_diff(a, b):
    """Shortest signed difference b - a between two angles in degrees."""
    return (b - a + 180.0) % 360.0 - 180.0


def approach(value, target, max_step):
    if value < target:
        return min(value + max_step, target)
    return max(value - max_step, target)
