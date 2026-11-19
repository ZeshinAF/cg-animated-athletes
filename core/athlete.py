"""Common articulated athlete: skeleton, appearance, locomotion.

The footballer and the basketballer only differ in their appearance
(textures, kit) and in their sport-specific clips and ball handling, which
live in sports/football.py and sports/basketball.py.

Joint conventions (degrees, 0 = standing straight):
  *_hip, *_shoulder  > 0 swings the limb forward
  *_knee             > 0 bends the shin backwards
  *_elbow            > 0 bends the forearm forwards
  *_shoulder_roll, *_hip_roll > 0 lifts the limb sideways (outwards)
  torso_pitch        > 0 leans forward,  head_pitch > 0 looks down
  root_y             vertical offset of the pelvis in metres (jumps, crouches)
"""
import math

import numpy as np
from OpenGL.GL import glMultMatrixf, glPopMatrix, glPushMatrix, glTranslatef

from . import primitives as P
from .animation import Animator, Clip, mirror
from .mathutil import angle_diff, approach, clamp, rotate, to_gl, translate
from .physics import heading_vector
from .scenegraph import Node

HIP_H = 0.98          # pelvis height when standing (metres)
THIGH, SHIN = 0.44, 0.42
UPPER_ARM, FOREARM = 0.29, 0.26
TORSO_H = 0.52

# Torso atlas: left half = front, right half = back, thin edge strip for the sides.
JERSEY_UV = {
    "pz": (0.0, 0.0, 0.5, 1.0),
    "nz": (0.5, 0.0, 1.0, 1.0),
    "px": (0.032, 0.06, 0.06, 0.85),
    "nx": (0.032, 0.06, 0.06, 0.85),
    "py": (0.1, 0.9, 0.12, 0.92),
    "ny": (0.1, 0.02, 0.12, 0.04),
}


def c255(rgb):
    return tuple(c / 255.0 for c in rgb)


def at(x, y, z, fn):
    """Wrap a draw call so it is executed at a local offset."""
    def draw():
        glPushMatrix()
        glTranslatef(x, y, z)
        fn()
        glPopMatrix()
    return draw


def base_clips():
    idle_a = {"l_shoulder_roll": 6, "r_shoulder_roll": 6, "l_elbow": 12, "r_elbow": 12,
              "l_hip_roll": 3, "r_hip_roll": 3}
    idle_b = dict(idle_a, root_y=-0.012, torso_pitch=2.5, head_pitch=-2,
                  l_shoulder_roll=8, r_shoulder_roll=8, l_elbow=16, r_elbow=16)

    walk_contact = {"l_hip": 22, "l_knee": 4, "l_ankle": 8, "r_hip": -18, "r_knee": 18,
                    "r_ankle": -12, "l_shoulder": -18, "r_shoulder": 18, "l_elbow": 15,
                    "r_elbow": 28, "l_shoulder_roll": 5, "r_shoulder_roll": 5,
                    "torso_pitch": 4, "torso_yaw": 5, "root_y": -0.02}
    walk_pass = {"l_hip": 0, "l_knee": 6, "r_hip": 8, "r_knee": 42, "r_ankle": 10,
                 "l_elbow": 20, "r_elbow": 20, "l_shoulder_roll": 5, "r_shoulder_roll": 5,
                 "torso_pitch": 4, "root_y": 0.015}

    run_contact = {"l_hip": 48, "l_knee": 22, "l_ankle": 10, "r_hip": -28, "r_knee": 60,
                   "r_ankle": -20, "l_shoulder": -40, "r_shoulder": 48, "l_elbow": 80,
                   "r_elbow": 95, "l_shoulder_roll": 8, "r_shoulder_roll": 8,
                   "torso_pitch": 12, "torso_yaw": 9, "root_y": -0.04}
    run_flight = {"l_hip": -12, "l_knee": 30, "l_ankle": -10, "r_hip": 30, "r_knee": 105,
                  "l_shoulder": 0, "r_shoulder": 5, "l_elbow": 85, "r_elbow": 85,
                  "l_shoulder_roll": 8, "r_shoulder_roll": 8, "torso_pitch": 12, "root_y": 0.06}

    celebrate = [
        (0.0, {}),
        (0.18, {"root_y": -0.14, "l_knee": 50, "r_knee": 50, "l_hip": 30, "r_hip": 30,
                "l_shoulder": 30, "r_shoulder": 30, "torso_pitch": 12}),
        (0.38, {"root_y": 0.32, "l_shoulder": 170, "r_shoulder": 170, "l_shoulder_roll": 18,
                "r_shoulder_roll": 18, "l_knee": 25, "r_knee": 70, "r_hip": 35, "head_pitch": -18}),
        (0.6, {"root_y": -0.04, "l_shoulder": 165, "r_shoulder": 165, "l_shoulder_roll": 18,
               "r_shoulder_roll": 18, "l_knee": 20, "r_knee": 20, "l_hip": 10, "r_hip": 10,
               "head_pitch": -10}),
        (0.78, {"root_y": 0.22, "l_shoulder": 172, "r_shoulder": 172, "l_shoulder_roll": 12,
                "r_shoulder_roll": 12, "l_knee": 60, "r_knee": 20, "l_hip": 35, "head_pitch": -18}),
        (1.0, {}),
    ]
    return [
        Clip("idle", 2.4, [(0.0, idle_a), (0.5, idle_b)]),
        Clip("walk", 1.1, [(0.0, walk_contact), (0.25, walk_pass),
                           (0.5, mirror(walk_contact)), (0.75, mirror(walk_pass))]),
        Clip("run", 0.72, [(0.0, run_contact), (0.25, run_flight),
                           (0.5, mirror(run_contact)), (0.75, mirror(run_flight))]),
        Clip("celebrate", 1.8, celebrate, loop=False),
    ]


class Athlete:
    WALK_SPEED = 2.0
    RUN_SPEED = 5.5
    ACCEL = 9.0
    TURN_RATE = 540.0       # degrees per second

    def __init__(self, look, clips, bounds):
        self.look = look
        self.pos = np.array([0.0, 0.0, 0.0])
        self.heading = 0.0
        self.speed = 0.0
        self.bounds = bounds                      # (xmin, xmax, zmin, zmax)
        self.root = self._build(look)
        self.animator = Animator(base_clips() + clips)
        self.animator.play("idle", fade=0.0)
        self.action = None                        # active one-shot clip
        self.r_hand = self.root.find("r_hand")
        self.l_hand = self.root.find("l_hand")
        self.r_foot = self.root.find("r_ankle")

    # ------------------------------------------------------------------ body
    def _build(self, look):
        skin = c255(look["skin"])
        skin_tex = look.get("skin_tex")
        shirt = c255(look["shirt_color"])
        shorts = c255(look["shorts_color"])
        shoe = c255(look["shoe_color"])
        sole = c255(look.get("sole_color", (30, 30, 30)))

        def limb_drawer(length, r0, r1, color, tex=None, sx=1.0):
            return lambda: (P.material(color, tex), P.limb(length, r0, r1, sx=sx))

        def arm(side):
            s = 1 if side == "l" else -1
            hand = Node(side + "_hand", (0.0, -FOREARM - 0.02, 0.0))
            draws_upper = [
                lambda: (P.material(shirt), P.sphere(0.072)),
                limb_drawer(UPPER_ARM, 0.058, 0.046, skin, skin_tex),
            ]
            if look.get("sleeves"):
                draws_upper.append(limb_drawer(0.15, 0.072, 0.064, shirt))
            if look.get("wristband"):
                wb = c255(look["wristband"])
                wrist = [at(0, -FOREARM + 0.07, 0, limb_drawer(0.06, 0.047, 0.045, wb))]
            else:
                wrist = []
            elbow = Node(side + "_elbow", (0.0, -UPPER_ARM, 0.0), [("x", side + "_elbow", -1)],
                         draw=[lambda: (P.material(skin, skin_tex), P.sphere(0.047)),
                               limb_drawer(FOREARM, 0.046, 0.036, skin, skin_tex),
                               at(0, -FOREARM - 0.035, 0.0,
                                  lambda: (P.material(skin, skin_tex), P.sphere(0.042, 0.06, 0.03)))] + wrist,
                         children=[hand])
            return Node(side + "_shoulder", (s * 0.245, 0.47, 0.0),
                        [("x", side + "_shoulder", -1), ("z", side + "_shoulder_roll", s)],
                        draw=draws_upper, children=[elbow])

        def leg(side):
            s = 1 if side == "l" else -1
            sock_tex = look.get("sock_tex")
            shin_draw = [lambda: (P.material(skin, skin_tex), P.sphere(0.06))]
            if sock_tex is not None:
                shin_draw.append(limb_drawer(SHIN, 0.062, 0.044, (1, 1, 1), sock_tex))
            else:
                shin_draw.append(limb_drawer(SHIN, 0.058, 0.042, skin, skin_tex))
                sock = c255(look.get("sock_color", (240, 240, 240)))
                shin_draw.append(at(0, -SHIN + 0.12, 0, limb_drawer(0.12, 0.047, 0.045, sock)))
            shoe_parts = [
                lambda: (P.material(shoe), P.box(0.105, 0.07, 0.27, center=(0.0, -0.035, 0.05))),
                lambda: (P.material(sole), P.box(0.11, 0.02, 0.28, center=(0.0, -0.07, 0.05))),
            ]
            if look.get("high_tops"):
                shoe_parts.append(at(0, 0.06, 0, lambda: (P.material(shoe), P.limb(0.1, 0.056, 0.056))))
            ankle = Node(side + "_ankle", (0.0, -SHIN, 0.0), [("x", side + "_ankle", -1)], draw=shoe_parts)
            knee = Node(side + "_knee", (0.0, -THIGH, 0.0), [("x", side + "_knee", 1)],
                        draw=shin_draw, children=[ankle])
            thigh_draw = [limb_drawer(THIGH, 0.085, 0.062, skin, skin_tex),
                          limb_drawer(look["shorts_len"], 0.1, 0.092, shorts)]
            return Node(side + "_hip", (s * 0.1, -0.04, 0.0),
                        [("x", side + "_hip", -1), ("z", side + "_hip_roll", s)],
                        draw=thigh_draw, children=[knee])

        head = Node("head", (0.0, TORSO_H, 0.0), [("x", "head_pitch", 1), ("y", "head_yaw", 1)],
                    draw=[at(0, 0.1, 0, lambda: (P.material(skin, skin_tex), P.limb(0.1, 0.052, 0.058))),
                          at(0, 0.22, 0.005, lambda: (P.material((1, 1, 1), look["face_tex"]),
                                                      P.sphere(0.105, 0.13, 0.115)))])
        torso = Node("torso", (0.0, 0.06, 0.0),
                     [("x", "torso_pitch", 1), ("y", "torso_yaw", 1), ("z", "torso_roll", 1)],
                     draw=[lambda: (P.material((1, 1, 1), look["jersey_tex"]),
                                    P.box(0.4, TORSO_H, 0.22, center=(0.0, TORSO_H / 2, 0.0), uv=JERSEY_UV))],
                     children=[head, arm("l"), arm("r")])
        pelvis = Node("pelvis", lambda pose: (0.0, HIP_H + pose.get("root_y", 0.0), 0.0),
                      draw=[lambda: (P.material(shorts), P.box(0.37, 0.2, 0.23, center=(0.0, -0.01, 0.0)))],
                      children=[torso, leg("l"), leg("r")])
        return pelvis

    # ------------------------------------------------------------ behaviour
    def forward(self):
        return heading_vector(self.heading)

    def base_matrix(self):
        return translate(*self.pos) @ rotate("y", self.heading)

    def start_action(self, name, fade=0.12):
        self.action = name
        self.animator.play(name, restart=True, fade=fade)

    def action_move_factor(self):
        """How much the athlete may move while a one-shot clip is playing."""
        return 0.0

    def post_pose(self, pose, dt):
        """Hook for sport-specific animation layers (e.g. the dribbling arm)."""
        return pose

    def update(self, dt, move_dir, sprint):
        move_dir = np.asarray(move_dir, dtype=float)
        mag = min(1.0, float(np.linalg.norm(move_dir)))
        factor = self.action_move_factor() if self.action else 1.0
        target = mag * (self.RUN_SPEED if sprint else self.WALK_SPEED) * factor
        self.speed = approach(self.speed, target, self.ACCEL * dt)
        if mag > 0.05 and factor > 0:
            desired = math.degrees(math.atan2(move_dir[0], move_dir[2]))
            turn = clamp(angle_diff(self.heading, desired), -self.TURN_RATE * dt, self.TURN_RATE * dt)
            self.heading = (self.heading + turn) % 360.0
        self.pos += self.forward() * self.speed * dt
        x0, x1, z0, z1 = self.bounds
        self.pos[0] = clamp(self.pos[0], x0, x1)
        self.pos[2] = clamp(self.pos[2], z0, z1)

        if self.action and self.animator.finished:
            self.action = None
        if not self.action:
            if self.speed < 0.15:
                self.animator.play("idle", fade=0.3)
            elif self.speed < 3.2:
                self.animator.play("walk", speed=self.speed / 1.4, fade=0.25, keep_phase=True)
            else:
                self.animator.play("run", speed=self.speed / 5.0, fade=0.25, keep_phase=True)

        events = self.animator.update(dt)
        pose = self.post_pose(self.animator.pose, dt)
        self.pose = pose
        self.root.update(self.base_matrix(), pose)
        return events

    def render(self):
        glPushMatrix()
        glMultMatrixf(to_gl(self.base_matrix()))
        self.root.render()
        glPopMatrix()

    def locomotion_phase(self):
        if self.animator.current in ("walk", "run"):
            return self.animator.phase()
        return 0.0
