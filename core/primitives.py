"""Procedural meshes with explicit normals and texture coordinates.

Every mesh is generated once and compiled into an OpenGL display list, so
the per-frame cost is a single glCallList per body part.

UV conventions
--------------
* sphere / limb:  u goes around the Y axis, u = 0.5 points to +Z (the
  "front" of a character), v goes from 0 (bottom) to 1 (top).
* box:           every face gets its own (u0, v0, u1, v1) rectangle so a
  single texture atlas can hold front, back and side images.
"""
import math

from OpenGL.GL import (
    GL_COMPILE, GL_LINES, GL_QUAD_STRIP, GL_QUADS, GL_TEXTURE_2D, GL_TRIANGLE_FAN,
    glBegin, glBindTexture, glCallList, glColor4f, glDisable, glEnable, glEnd,
    glEndList, glGenLists, glNewList, glNormal3f, glTexCoord2f, glVertex3f,
)


class RenderState:
    """Global flag used by the planar-shadow pass: while it is set, materials
    are ignored so everything is drawn in the flat shadow colour."""
    shadow_pass = False


def material(color=(1.0, 1.0, 1.0), tex=None, alpha=1.0):
    """Select a colour (used by GL_COLOR_MATERIAL) and optionally a texture."""
    if RenderState.shadow_pass:
        return
    if tex is None:
        glDisable(GL_TEXTURE_2D)
    else:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
    glColor4f(color[0], color[1], color[2], alpha)


_lists = {}


def _cached(key, build):
    lst = _lists.get(key)
    if lst is None:
        lst = glGenLists(1)
        glNewList(lst, GL_COMPILE)
        build()
        glEndList()
        _lists[key] = lst
    glCallList(lst)


def _sphere_point(u, v):
    phi = (u - 0.5) * 2.0 * math.pi
    theta = math.pi * (1.0 - v)
    return (math.sin(theta) * math.sin(phi), math.cos(theta), math.sin(theta) * math.cos(phi))


def sphere(rx, ry=None, rz=None, slices=28, stacks=18):
    """Ellipsoid centred at the origin with spherical UV mapping."""
    ry = rx if ry is None else ry
    rz = rx if rz is None else rz

    def build():
        for i in range(stacks):
            v0, v1 = i / stacks, (i + 1) / stacks
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                u = j / slices
                for v in (v1, v0):
                    x, y, z = _sphere_point(u, v)
                    nx, ny, nz = x / rx, y / ry, z / rz
                    ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                    glNormal3f(nx / ln, ny / ln, nz / ln)
                    glTexCoord2f(u, v)
                    glVertex3f(x * rx, y * ry, z * rz)
            glEnd()

    _cached(("sphere", rx, ry, rz, slices, stacks), build)


def limb(length, r_top, r_bottom, slices=18, caps=True, sx=1.0):
    """Tapered cylinder hanging from the origin down to y = -length.
    `sx` squashes it along X (used for flatter body parts)."""

    def build():
        dy = (r_top - r_bottom)
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            u = j / slices
            phi = (u - 0.5) * 2.0 * math.pi
            x, z = math.sin(phi), math.cos(phi)
            n = (x * length, -dy, z * length)
            ln = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            glNormal3f(n[0] / ln, n[1] / ln, n[2] / ln)
            glTexCoord2f(u, 1.0)
            glVertex3f(x * r_top * sx, 0.0, z * r_top)
            glTexCoord2f(u, 0.0)
            glVertex3f(x * r_bottom * sx, -length, z * r_bottom)
        glEnd()
        if caps:
            for y, r, ny in ((0.0, r_top, 1.0), (-length, r_bottom, -1.0)):
                glBegin(GL_TRIANGLE_FAN)
                glNormal3f(0.0, ny, 0.0)
                glTexCoord2f(0.5, 0.5)
                glVertex3f(0.0, y, 0.0)
                rng = range(slices + 1) if ny < 0 else range(slices, -1, -1)
                for j in rng:
                    phi = j / slices * 2.0 * math.pi
                    glTexCoord2f(0.5 + 0.5 * math.sin(phi), 0.5 + 0.5 * math.cos(phi))
                    glVertex3f(math.sin(phi) * r * sx, y, math.cos(phi) * r)
                glEnd()

    _cached(("limb", length, r_top, r_bottom, slices, caps, sx), build)


def cylinder_between(p0, p1, r, slices=12):
    """Straight cylinder between two points (goal posts, hoop arm...)."""
    _cached(("cyl", tuple(p0), tuple(p1), r, slices), lambda: _cylinder_between(p0, p1, r, slices))


def _cylinder_between(p0, p1, r, slices):
    ax = [p1[i] - p0[i] for i in range(3)]
    ln = math.sqrt(sum(a * a for a in ax))
    ax = [a / ln for a in ax]
    # any vector not parallel to the axis
    ref = (0.0, 1.0, 0.0) if abs(ax[1]) < 0.9 else (1.0, 0.0, 0.0)
    b1 = _norm(_cross(ax, ref))
    b2 = _cross(ax, b1)
    glBegin(GL_QUAD_STRIP)
    for j in range(slices + 1):
        a = j / slices * 2.0 * math.pi
        n = [math.cos(a) * b1[i] + math.sin(a) * b2[i] for i in range(3)]
        glNormal3f(*n)
        glTexCoord2f(j / slices, 0.0)
        glVertex3f(*[p0[i] + n[i] * r for i in range(3)])
        glTexCoord2f(j / slices, 1.0)
        glVertex3f(*[p1[i] + n[i] * r for i in range(3)])
    glEnd()


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    ln = math.sqrt(sum(x * x for x in a)) or 1.0
    return tuple(x / ln for x in a)


# Corners of each face in the order bottom-left, bottom-right, top-right,
# top-left *as seen from outside*, so textures are never mirrored.
_BOX_FACES = {
    "pz": ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
    "nz": ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]),
    "px": ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)]),
    "nx": ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
    "py": ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)]),
    "ny": ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
}


def box(w, h, d, center=(0.0, 0.0, 0.0), uv=None):
    """Axis-aligned box. `uv` maps face names (pz=front, nz=back, px, nx,
    py=top, ny=bottom) to (u0, v0, u1, v1) rectangles of a texture atlas."""
    uv = uv or {}
    key = ("box", w, h, d, tuple(center), tuple(sorted(uv.items())))

    def build():
        hx, hy, hz = w / 2.0, h / 2.0, d / 2.0
        cx, cy, cz = center
        glBegin(GL_QUADS)
        for name, (normal, corners) in _BOX_FACES.items():
            u0, v0, u1, v1 = uv.get(name, (0.0, 0.0, 1.0, 1.0))
            tcs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            glNormal3f(*normal)
            for (sx, sy, sz), (tu, tv) in zip(corners, tcs):
                glTexCoord2f(tu, tv)
                glVertex3f(cx + sx * hx, cy + sy * hy, cz + sz * hz)
        glEnd()

    _cached(key, build)


def torus(major, minor, seg=40, side=10):
    """Ring lying in the XZ plane (basketball rim)."""

    def build():
        for i in range(seg):
            a0, a1 = i / seg * 2 * math.pi, (i + 1) / seg * 2 * math.pi
            glBegin(GL_QUAD_STRIP)
            for j in range(side + 1):
                b = j / side * 2 * math.pi
                for a in (a0, a1):
                    nx, ny, nz = math.cos(b) * math.cos(a), math.sin(b), math.cos(b) * math.sin(a)
                    glNormal3f(nx, ny, nz)
                    glVertex3f((major + minor * math.cos(b)) * math.cos(a), minor * math.sin(b),
                               (major + minor * math.cos(b)) * math.sin(a))
            glEnd()

    _cached(("torus", major, minor, seg, side), build)


def ground_quad(x0, z0, x1, z1, y=0.0, repeat=(1.0, 1.0)):
    """Horizontal textured rectangle. The texture's top edge maps to z0."""

    def build():
        ru, rv = repeat
        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(x0, y, z1)
        glTexCoord2f(ru, 0.0)
        glVertex3f(x1, y, z1)
        glTexCoord2f(ru, rv)
        glVertex3f(x1, y, z0)
        glTexCoord2f(0.0, rv)
        glVertex3f(x0, y, z0)
        glEnd()

    _cached(("ground", x0, z0, x1, z1, y, repeat), build)


def lines(segments):
    """Immediate-mode line list (nets); segments = [(p0, p1), ...]."""
    key = ("lines", id(segments))

    def build():
        glBegin(GL_LINES)
        for p0, p1 in segments:
            glVertex3f(*p0)
            glVertex3f(*p1)
        glEnd()

    _cached(key, build)
