"""Procedural texture generation.

All textures are produced in code at start-up (no image files are needed):
numpy is used for per-pixel work (noise, ball patterns) and pygame's 2D
drawing / font rendering for shapes and text.  Generator functions return
(H, W, 3|4) uint8 arrays, which keeps them independent of OpenGL; the
:func:`upload` function turns an array into a mip-mapped GL texture.
"""
import math

import numpy as np
import pygame

# --------------------------------------------------------------------------
# OpenGL upload
# --------------------------------------------------------------------------


def upload(img, repeat=True):
    from OpenGL.GL import (
        GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT, GL_RGB, GL_RGBA, GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
        GL_UNSIGNED_BYTE, glBindTexture, glGenTextures, glTexParameteri, GL_CLAMP_TO_EDGE,
        glPixelStorei, GL_UNPACK_ALIGNMENT,
    )
    from OpenGL.GLU import gluBuild2DMipmaps

    img = np.ascontiguousarray(np.flipud(img))  # image row 0 = top = v 1
    h, w = img.shape[:2]
    fmt = GL_RGBA if img.shape[2] == 4 else GL_RGB
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    wrap = GL_REPEAT if repeat else GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    gluBuild2DMipmaps(GL_TEXTURE_2D, fmt, w, h, fmt, GL_UNSIGNED_BYTE, img)
    return tex


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _rgb(c):
    return np.array(c, dtype=np.float32)


def surface_to_array(surf):
    return np.ascontiguousarray(pygame.surfarray.array3d(surf).swapaxes(0, 1))


def array_to_surface(arr):
    return pygame.surfarray.make_surface(np.ascontiguousarray(arr[:, :, :3].swapaxes(0, 1)))


def value_noise(h, w, cells, rng, tile=True):
    """Smoothly interpolated random lattice ('value noise')."""
    cy, cx = cells if isinstance(cells, tuple) else (cells, cells)
    grid = rng.random((cy + 1, cx + 1)).astype(np.float32)
    if tile:
        grid[-1, :] = grid[0, :]
        grid[:, -1] = grid[:, 0]
    ys = np.linspace(0, cy, h, endpoint=False)
    xs = np.linspace(0, cx, w, endpoint=False)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = ys - y0, xs - x0
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    a = grid[y0][:, x0]
    b = grid[y0][:, x0 + 1]
    c = grid[y0 + 1][:, x0]
    d = grid[y0 + 1][:, x0 + 1]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    return top + (bot - top) * fy[:, None]


def fractal_noise(h, w, base_cells=4, octaves=4, seed=0):
    """Sum of value-noise octaves (fBm), normalised to [0, 1]."""
    rng = np.random.default_rng(seed)
    total = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    by, bx = base_cells if isinstance(base_cells, tuple) else (base_cells, base_cells)
    for o in range(octaves):
        total += amp * value_noise(h, w, (by * 2 ** o, bx * 2 ** o), rng)
        norm += amp
        amp *= 0.5
    return total / norm


def _font(size, bold=True):
    pygame.font.init()
    return pygame.font.SysFont("arial,dejavusans,sans", size, bold=bold)


def _sphere_dirs(h, w):
    """Unit direction for every texel of a sphere texture (same mapping as
    primitives.sphere: u=0.5 -> +Z, v=1 -> top)."""
    u = (np.arange(w) + 0.5) / w
    v = 1.0 - (np.arange(h) + 0.5) / h
    phi = (u - 0.5) * 2 * np.pi
    theta = np.pi * (1.0 - v)
    st = np.sin(theta)[:, None]
    x = st * np.sin(phi)[None, :]
    y = np.cos(theta)[:, None] * np.ones((1, w))
    z = st * np.cos(phi)[None, :]
    return np.stack([x, y, z], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------
# balls
# --------------------------------------------------------------------------


def football(h=256, w=512):
    """Classic 32-panel football (truncated icosahedron).

    The 12 icosahedron vertices (pentagon centres) and the 20 face centres of
    the icosahedron (hexagon centres) are used as sites of a weighted
    spherical Voronoi diagram: each texel takes the colour of its nearest
    site, and texels that are almost equally close to two sites become the
    seams."""
    phi = (1 + 5 ** 0.5) / 2
    ico = []
    for a in (-1, 1):
        for b in (-phi, phi):
            ico += [(0, a, b), (a, b, 0), (b, 0, a)]
    ico = np.array(ico, np.float32)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    dod = []
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):
                dod.append((a, b, c))
    for a in (-1, 1):
        for b in (-1, 1):
            dod += [(a / phi, 0, b * phi), (b * phi, a / phi, 0), (0, b * phi, a / phi)]
    dod = np.array(dod, np.float32)
    dod /= np.linalg.norm(dod, axis=1, keepdims=True)

    sites = np.concatenate([ico, dod])
    # A ray from the centre hits the face whose plane maximises dot(d, n_i) / r_i,
    # where r_i is the distance of the face plane from the centre.  For a
    # truncated icosahedron (edge 1) r_pentagon = 2.327 and r_hexagon = 2.267.
    weights = np.concatenate([np.full(12, 2.267 / 2.327), np.full(20, 1.0)])
    dirs = _sphere_dirs(h, w)
    dots = (dirs.reshape(-1, 3) @ sites.T) * weights
    order = np.argsort(-dots, axis=1)
    best, second = order[:, 0], order[:, 1]
    d1 = np.take_along_axis(dots, best[:, None], 1)[:, 0]
    d2 = np.take_along_axis(dots, second[:, None], 1)[:, 0]

    img = np.where((best < 12)[:, None], _rgb((25, 25, 30)), _rgb((245, 245, 245)))
    seam = (d1 - d2) < 0.010
    img[seam] = _rgb((90, 90, 95))
    img = img.reshape(h, w, 3)
    img *= (0.94 + 0.06 * fractal_noise(h, w, 8, 2, seed=3))[..., None]
    return img.clip(0, 255).astype(np.uint8)


def basketball(h=256, w=512):
    """Orange pebbled leather with the eight-panel seam pattern."""
    d = _sphere_dirs(h, w)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    pebble = fractal_noise(h, w, 48, 2, seed=5)
    base = _rgb((205, 92, 30))[None, None, :] * (0.82 + 0.25 * pebble)[..., None]
    lw = 0.022
    seams = (np.abs(y) < lw) | (np.abs(z) < lw) | (np.abs(np.abs(x) - 0.72) < lw)
    base[seams] = _rgb((30, 20, 15))
    return base.clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# characters
# --------------------------------------------------------------------------


def jersey(primary, secondary, number, name, stripes=False, w=512, h=352):
    """Shirt atlas: left half = front, right half = back.
    The halves have the same aspect ratio as the torso faces (0.40 x 0.55 m)."""
    surf = pygame.Surface((w, h))
    surf.fill(primary)
    half = w // 2
    if stripes:
        sw = half // 7
        for i in range(1, 7, 2):
            pygame.draw.rect(surf, secondary, (i * sw, 0, sw, h))
            pygame.draw.rect(surf, secondary, (half + i * sw, 0, sw, h))
    # side panels and hem in the secondary colour
    for x0 in (0, half - 14, half, w - 14):
        pygame.draw.rect(surf, secondary, (x0, 0, 14, h))
    pygame.draw.rect(surf, secondary, (0, h - 16, w, 16))
    # V-neck collar on the front, round collar on the back
    pygame.draw.polygon(surf, secondary, [(half // 2 - 40, 0), (half // 2 + 40, 0), (half // 2, 46)])
    pygame.draw.polygon(surf, primary, [(half // 2 - 28, 0), (half // 2 + 28, 0), (half // 2, 32)])
    pygame.draw.ellipse(surf, secondary, (half + half // 2 - 46, -24, 92, 44), 8)

    num_color = (255, 255, 255) if sum(primary) < 450 else (20, 20, 20)
    outline = secondary
    # front: small number (football) or big number (basketball)
    big = _font(180)
    small = _font(64)
    if stripes:
        dark = tuple(int(c * 0.55) for c in primary)
        _text_outlined(surf, str(number), small, dark, (255, 255, 255), (half // 2 + 52, 112))
        pygame.draw.circle(surf, (230, 190, 40), (half // 2 - 58, 100), 22)  # crest
        pygame.draw.circle(surf, outline, (half // 2 - 58, 100), 22, 4)
    else:
        _text_outlined(surf, str(number), big, num_color, outline, (half // 2, 200))
    # back: name above a big number (on a plain panel when the shirt is striped)
    if stripes:
        pygame.draw.rect(surf, primary, (half + 40, 40, half - 80, 270), border_radius=18)
    _text_outlined(surf, name.upper(), _font(40), num_color, outline, (half + half // 2, 70))
    _text_outlined(surf, str(number), big, num_color, outline, (half + half // 2, 210))
    img = surface_to_array(surf).astype(np.float32)
    img *= (0.93 + 0.07 * fractal_noise(h, w, 32, 2, seed=11))[..., None]  # fabric
    return img.clip(0, 255).astype(np.uint8)


def _text_outlined(surf, text, font, color, outline, center):
    for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)):
        img = font.render(text, True, outline)
        surf.blit(img, img.get_rect(center=(center[0] + dx, center[1] + dy)))
    img = font.render(text, True, color)
    surf.blit(img, img.get_rect(center=center))


def face(skin, hair, eye=(70, 45, 25), headband=None, beard=False, w=512, h=256):
    """Head texture for an ellipsoid with spherical UVs (front at u = 0.5)."""
    surf = pygame.Surface((w, h))
    surf.fill(skin)
    cx = w // 2
    dark = tuple(int(c * 0.82) for c in skin)
    # ears at +-90 degrees
    for ex in (cx - w // 4, cx + w // 4):
        pygame.draw.ellipse(surf, dark, (ex - 10, 112, 20, 34))
    # hair: hairline is high at the front and low at the back of the head
    for x in range(w):
        a = (x - cx) / w * 2 * math.pi
        line = int(78 + 74 * (1 - math.cos(a)) / 2)
        pygame.draw.line(surf, hair, (x, 0), (x, line))
    if headband:
        pygame.draw.rect(surf, headband, (0, 70, w, 16))
    # eyebrows, eyes, nose, mouth
    for s in (-1, 1):
        ex = cx + s * 28
        pygame.draw.line(surf, hair, (ex - 12, 101), (ex + 12, 99), 4)
        pygame.draw.ellipse(surf, (250, 250, 250), (ex - 11, 108, 22, 12))
        pygame.draw.circle(surf, eye, (ex, 114), 5)
        pygame.draw.circle(surf, (10, 10, 10), (ex, 114), 2)
    pygame.draw.polygon(surf, dark, [(cx, 118), (cx - 7, 146), (cx + 7, 146)])
    pygame.draw.arc(surf, (150, 60, 55), (cx - 18, 150, 36, 18), math.pi * 1.1, math.pi * 1.9, 4)
    if beard:
        for x in range(cx - 60, cx + 61):
            top = 158 + int(8 * math.cos((x - cx) / 60 * math.pi / 2))
            pygame.draw.line(surf, hair, (x, top + (12 if abs(x - cx) < 22 else 0)), (x, 205))
        pygame.draw.arc(surf, (150, 60, 55), (cx - 18, 150, 36, 18), math.pi * 1.1, math.pi * 1.9, 4)
    img = surface_to_array(surf).astype(np.float32)
    img *= (0.95 + 0.05 * fractal_noise(h, w, 16, 3, seed=7))[..., None]
    return img.clip(0, 255).astype(np.uint8)


def socks(color, stripe, w=64, h=256):
    """Long football socks: plain colour with two stripes near the top."""
    img = np.empty((h, w, 3), np.float32)
    img[:] = color
    for y0 in (20, 52):
        img[y0:y0 + 14] = stripe
    img *= (0.92 + 0.08 * fractal_noise(h, w, (16, 4), 2, seed=9))[..., None]
    return img.clip(0, 255).astype(np.uint8)


def skin_tex(skin, w=64, h=64):
    img = np.empty((h, w, 3), np.float32)
    img[:] = skin
    img *= (0.95 + 0.05 * fractal_noise(h, w, 8, 2, seed=1))[..., None]
    return img.clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# environments
# --------------------------------------------------------------------------


def grass_tile(size=256, seed=21):
    n = fractal_noise(size, size, 8, 5, seed=seed)
    base = _rgb((58, 120, 48))[None, None, :] * (0.75 + 0.45 * n)[..., None]
    return base.clip(0, 255).astype(np.uint8)


def pitch(length=30.0, width=20.0, margin=2.0, ppm=32):
    """Five-a-side style pitch with mowing stripes and white markings."""
    W, H = int((length + 2 * margin) * ppm), int((width + 2 * margin) * ppm)
    n = fractal_noise(H, W, 24, 5, seed=4)
    xs = (np.arange(W) / ppm - margin)
    stripe = ((np.floor(xs / 2.5) % 2) == 0).astype(np.float32)
    shade = 0.88 + 0.12 * stripe
    img = _rgb((60, 135, 50))[None, None, :] * (shade[None, :] * (0.8 + 0.35 * n))[..., None]
    surf = array_to_surface(img.clip(0, 255).astype(np.uint8))

    def p(x, z):  # metres (pitch centred at 0) -> pixels
        return int((x + length / 2 + margin) * ppm), int((z + width / 2 + margin) * ppm)

    white = (240, 240, 235)
    lw = 4
    hl, hw = length / 2, width / 2
    pygame.draw.rect(surf, white, (*p(-hl, -hw), int(length * ppm), int(width * ppm)), lw)
    pygame.draw.line(surf, white, p(0, -hw), p(0, hw), lw)
    pygame.draw.circle(surf, white, p(0, 0), int(3.0 * ppm), lw)
    pygame.draw.circle(surf, white, p(0, 0), 6)
    for s in (-1, 1):
        # penalty area (5 m deep, 12 m wide) and goal area (2 m x 7 m)
        x_line = s * hl
        for depth, half_w in ((5.0, 6.0), (2.0, 3.5)):
            x0 = min(x_line, x_line - s * depth)
            pygame.draw.rect(surf, white, (*p(x0, -half_w), int(depth * ppm), int(2 * half_w * ppm)), lw)
        pygame.draw.circle(surf, white, p(s * (hl - 4.0), 0), 6)
    return surface_to_array(surf)


def court(width=15.0, depth=14.0, margin=1.0, ppm=48):
    """Outdoor half-court (FIBA dimensions). Baseline at z = -depth/2."""
    W, H = int((width + 2 * margin) * ppm), int((depth + 2 * margin) * ppm)
    n = fractal_noise(H, W, 32, 4, seed=8)
    xs = np.arange(W) / ppm - margin - width / 2
    zs = np.arange(H) / ppm - margin - depth / 2
    X, Z = np.meshgrid(xs, zs)
    inside = (np.abs(X) <= width / 2) & (np.abs(Z) <= depth / 2)
    base_z = -depth / 2
    key = (np.abs(X) <= 2.45) & (Z >= base_z) & (Z <= base_z + 5.8)
    img = np.empty((H, W, 3), np.float32)
    img[:] = (70, 110, 80)
    img[inside] = (45, 92, 150)
    img[key] = (185, 75, 50)
    img *= (0.85 + 0.25 * n)[..., None]
    surf = array_to_surface(img.clip(0, 255).astype(np.uint8))

    def p(x, z):
        return int((x + width / 2 + margin) * ppm), int((z + depth / 2 + margin) * ppm)

    white = (240, 240, 240)
    lw = 5
    pygame.draw.rect(surf, white, (*p(-width / 2, base_z), int(width * ppm), int(depth * ppm)), lw)
    pygame.draw.rect(surf, white, (*p(-2.45, base_z), int(4.9 * ppm), int(5.8 * ppm)), lw)
    hoop = (0.0, base_z + 1.575)
    ft = p(0, base_z + 5.8)
    pygame.draw.circle(surf, white, ft, int(1.8 * ppm), lw)
    # three-point line: two straight lines + an arc of radius 6.75 m around the hoop
    r3 = 6.75
    zc = hoop[1] + math.sqrt(r3 ** 2 - 6.6 ** 2)
    for s in (-1, 1):
        pygame.draw.line(surf, white, p(s * 6.6, base_z), p(s * 6.6, zc), lw)
    pts = []
    a0 = math.asin(6.6 / r3)
    for i in range(81):
        a = -a0 + 2 * a0 * i / 80
        pts.append(p(r3 * math.sin(a), hoop[1] + r3 * math.cos(a)))
    pygame.draw.lines(surf, white, False, pts, lw)
    # restricted area arc and the centre circle cut by the half-way line
    pygame.draw.arc(surf, white, (*p(-1.25, hoop[1] - 1.25), int(2.5 * ppm), int(2.5 * ppm)),
                    math.pi, 2 * math.pi, lw)
    cc = p(0, depth / 2)
    pygame.draw.circle(surf, white, cc, int(1.8 * ppm), lw)
    return surface_to_array(surf)


def backboard(w=360, h=210):
    surf = pygame.Surface((w, h))
    surf.fill((245, 245, 245))
    red = (200, 40, 30)
    pygame.draw.rect(surf, red, (0, 0, w, h), 10)
    # inner target square: 59 x 45 cm on a 180 x 105 cm board
    sq_w, sq_h = int(w * 0.59 / 1.8), int(h * 0.45 / 1.05)
    pygame.draw.rect(surf, red, ((w - sq_w) // 2, h - sq_h - int(h * 0.15 / 1.05), sq_w, sq_h), 7)
    return surface_to_array(surf)


def boards(text="CG 2025  *  TU SOFIA  *  OPENGL", w=1024, h=64):
    """Advertising boards around the football pitch."""
    surf = pygame.Surface((w, h))
    surf.fill((20, 50, 120))
    img = _font(40).render(text, True, (255, 255, 255))
    for x in range(0, w, img.get_width() + 60):
        surf.blit(img, (x + 20, (h - img.get_height()) // 2))
    return surface_to_array(surf)
