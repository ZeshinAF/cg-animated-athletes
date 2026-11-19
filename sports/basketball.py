"""Basketballer and the outdoor half-court scene.

The basketballer dribbles with the right hand: the arm motion is a
procedural layer added on top of the walk/run keyframe clips, and the ball
height is a parabola in the dribble phase so it touches the hand at phase 0
and the floor at phase 0.5.  Space starts a jump shot; at the 'release'
event the ball gets the ballistic velocity that would take it to the rim
centre, plus a random error that grows with the distance."""
import math

import numpy as np
from OpenGL.GL import GL_LIGHTING, glDisable, glEnable, glLineWidth, glPopMatrix, glPushMatrix, glTranslatef

from core import primitives as P
from core import textures as T
from core.animation import Clip
from core.athlete import Athlete
from core.physics import Ball, UP, ballistic_velocity
from core.scene import Scene

BASKETBALLER_LOOK = {
    "jersey": dict(primary=(35, 70, 170), secondary=(250, 250, 250), number=23,
                   name="Petrov", stripes=False),
    "face": dict(skin=(150, 98, 62), hair=(25, 20, 18), eye=(45, 30, 20),
                 headband=(250, 250, 250), beard=True),
    "skin": (150, 98, 62),
    "shirt_color": (35, 70, 170),
    "shorts_color": (35, 70, 170),
    "shorts_len": 0.34,
    "shoe_color": (245, 245, 245),
    "sole_color": (200, 40, 40),
    "sock_color": (245, 245, 245),
    "wristband": (250, 250, 250),
    "high_tops": True,
    "sleeves": False,
}

COURT_W, COURT_D = 15.0, 14.0
BASELINE_Z = -COURT_D / 2
BOARD_Z = BASELINE_Z + 1.2               # front face of the backboard
RIM = np.array([0.0, 3.05, BASELINE_Z + 1.575])
RIM_R, RIM_TUBE = 0.225, 0.012
POLE_Z = BASELINE_Z - 1.3
THREE_PT = 6.75


def shoot_clip():
    hold = {"l_shoulder": 45, "r_shoulder": 45, "l_elbow": 95, "r_elbow": 95,
            "l_shoulder_roll": -8, "r_shoulder_roll": -8}
    crouch = {"root_y": -0.16, "l_hip": 40, "r_hip": 40, "l_knee": 70, "r_knee": 70,
              "l_ankle": 30, "r_ankle": 30, "torso_pitch": 14, "l_shoulder": 55, "r_shoulder": 55,
              "l_elbow": 110, "r_elbow": 110, "l_shoulder_roll": -10, "r_shoulder_roll": -10}
    jump = {"root_y": 0.38, "l_knee": 8, "r_knee": 8, "l_ankle": -30, "r_ankle": -30,
            "l_shoulder": 160, "r_shoulder": 160, "l_elbow": 45, "r_elbow": 35,
            "l_shoulder_roll": -6, "r_shoulder_roll": -4, "head_pitch": -12}
    release = {"root_y": 0.36, "l_knee": 10, "r_knee": 10, "l_ankle": -25, "r_ankle": -25,
               "l_shoulder": 150, "r_shoulder": 165, "l_elbow": 35, "r_elbow": 0,
               "l_shoulder_roll": 5, "head_pitch": -12}
    land = {"root_y": -0.08, "l_hip": 20, "r_hip": 20, "l_knee": 35, "r_knee": 35,
            "l_ankle": 15, "r_ankle": 15, "torso_pitch": 6, "l_shoulder": 40, "r_shoulder": 100,
            "l_elbow": 30, "r_elbow": 10}
    rest = {"l_shoulder": 20, "r_shoulder": 20, "l_elbow": 30, "r_elbow": 30}
    return Clip("shoot", 1.05, [(0.0, hold), (0.25, crouch), (0.5, jump), (0.6, release),
                                (0.85, land), (1.0, rest)], loop=False, events={"release": 0.55})


class Basketballer(Athlete):
    def __init__(self, bounds):
        look = dict(BASKETBALLER_LOOK)
        look["jersey_tex"] = T.upload(T.jersey(**look["jersey"]))
        look["face_tex"] = T.upload(T.face(**look["face"]))
        look["skin_tex"] = T.upload(T.skin_tex(look["skin"]))
        super().__init__(look, [shoot_clip()], bounds)
        self.dribbling = False
        self.dribble_phase = 0.0

    def action_move_factor(self):
        return 0.0

    def post_pose(self, pose, dt):
        """Animation layer: replace the arm channels with a procedural dribble."""
        if not self.dribbling or self.action:
            return pose
        freq = 1.6 + 0.22 * self.speed
        self.dribble_phase = (self.dribble_phase + freq * dt) % 1.0
        c = math.cos(2 * math.pi * self.dribble_phase)
        pose = dict(pose)
        pose["r_shoulder"] = 22 + 10 * c
        pose["r_shoulder_roll"] = 14
        pose["r_elbow"] = 55 + 30 * c
        pose["l_shoulder"] = 35
        pose["l_shoulder_roll"] = 28
        pose["l_elbow"] = 70
        pose["torso_pitch"] = pose.get("torso_pitch", 0) + 8
        pose["root_y"] = pose.get("root_y", 0) - 0.04
        return pose


class BasketballScene(Scene):
    title = "Basketball"

    def __init__(self):
        super().__init__()
        self.court_tex = T.upload(T.court(COURT_W, COURT_D), repeat=False)
        self.grass_tex = T.upload(T.grass_tile(seed=33))
        self.board_tex = T.upload(T.backboard(), repeat=False)
        hw, hd = COURT_W / 2 + 1.0, COURT_D / 2 + 1.0
        self.athlete = Basketballer(bounds=(-hw + 0.4, hw - 0.4, BASELINE_Z - 0.6, hd - 0.4))
        self.athlete.pos[:] = (1.5, 0.0, 1.0)
        self.athlete.heading = 180.0
        self.ball = Ball(0.12, T.upload(T.basketball()), restitution=0.78, rolling_friction=0.5)
        self.ball.bounds = (-hw - 0.5, hw + 0.5, -hd - 0.5, hd + 0.5)
        self.holding = True
        self.athlete.dribbling = True
        self.ball.held = True
        self.cooldown = 0.0
        self.shot = None             # info about the ball in flight
        self.pending_celebrate = False
        self.net = self._net()
        self.camera_yaw = 0.0
        self.rng = np.random.default_rng()
        self.attempts = 0

    # ----------------------------------------------------------- geometry
    def _net(self):
        segs, n = [], 12
        top = [(RIM[0] + RIM_R * math.cos(2 * math.pi * i / n), RIM[1],
                RIM[2] + RIM_R * math.sin(2 * math.pi * i / n)) for i in range(n)]
        bot = [(RIM[0] + 0.14 * math.cos(2 * math.pi * (i + 0.5) / n), RIM[1] - 0.42,
                RIM[2] + 0.14 * math.sin(2 * math.pi * (i + 0.5) / n)) for i in range(n)]
        mid = [(RIM[0] + 0.18 * math.cos(2 * math.pi * i / n), RIM[1] - 0.21,
                RIM[2] + 0.18 * math.sin(2 * math.pi * i / n)) for i in range(n)]
        for i in range(n):
            j = (i + 1) % n
            segs += [(top[i], bot[i]), (top[j], bot[i]), (mid[i], mid[j]), (bot[i], bot[j])]
        return segs

    def _hoop(self):
        P.material((0.25, 0.27, 0.3))
        P.cylinder_between((0, 0, POLE_Z), (0, 3.5, POLE_Z), 0.09)
        P.cylinder_between((0, 3.35, POLE_Z), (0, 3.35, BOARD_Z - 0.05), 0.05)
        P.material((1, 1, 1), self.board_tex)
        P.box(1.8, 1.05, 0.04, center=(0, 2.9 + 0.525, BOARD_Z - 0.02),
              uv={"px": (0, 0, 0.02, 0.02), "nx": (0, 0, 0.02, 0.02),
                  "py": (0, 0, 0.02, 0.02), "ny": (0, 0, 0.02, 0.02)})
        P.material((0.95, 0.4, 0.1))
        P.box(0.12, 0.04, 0.16, center=(0, RIM[1], BOARD_Z + 0.08))
        glPushMatrix()
        glTranslatef(*RIM)
        P.torus(RIM_R, RIM_TUBE)
        glPopMatrix()

    def draw_ground(self):
        P.material((0.9, 0.95, 0.9), self.grass_tex)
        P.ground_quad(-80, -80, 80, 80, y=-0.01, repeat=(40, 40))
        P.material((1, 1, 1), self.court_tex)
        hw, hd = COURT_W / 2 + 1.0, COURT_D / 2 + 1.0
        P.ground_quad(-hw, -hd, hw, hd)

    def draw_props(self):
        self._hoop()
        glDisable(GL_LIGHTING)
        P.material((0.95, 0.95, 0.95))
        glLineWidth(1.5)
        P.lines(self.net)
        glEnable(GL_LIGHTING)

    def draw_prop_shadows(self):
        self._hoop()

    # ------------------------------------------------------------ gameplay
    def update(self, dt, ctl):
        a, ball = self.athlete, self.ball
        if ctl.reset:
            self._give_ball()
        if ctl.action and self.holding and a.action is None:
            to_rim = RIM - a.pos
            a.heading = math.degrees(math.atan2(to_rim[0], to_rim[2]))   # face the basket
            a.dribbling = False
            a.start_action("shoot", fade=0.15)

        events = a.update(dt, ctl.move, ctl.sprint)

        if self.holding:
            if a.action == "shoot":
                if "release" in events:
                    self._release()
                else:   # ball between the hands
                    mid = (a.l_hand.world_pos() + a.r_hand.world_pos()) / 2
                    ball.carry_to(mid + a.forward() * 0.12 + UP * 0.06, dt, spin=np.zeros(3))
            else:
                self._dribble(dt)
        else:
            self.cooldown -= dt
            prev = ball.pos.copy()
            ball.update(dt)
            self._hoop_physics(prev)
            near = np.hypot(*(ball.pos - a.pos)[[0, 2]]) < 0.75
            if self.cooldown <= 0 and near and ball.pos[1] < 1.4 and a.action is None:
                self._give_ball(keep_position=True)

        if self.pending_celebrate and a.action is None:
            self.pending_celebrate = False
            a.start_action("celebrate")
        self.tick_message(dt)

    def _dribble(self, dt):
        a, ball = self.athlete, self.ball
        p = a.dribble_phase
        hand = a.r_hand.world_pos()
        top = hand[1] - 0.07 - ball.r
        y = ball.r + max(0.0, top - ball.r) * (2 * p - 1) ** 2
        pos = hand + a.forward() * 0.06
        pos[1] = y
        spin = np.cross(UP, a.forward()) * (a.speed / ball.r) * 0.5
        ball.carry_to(pos, dt, spin=spin)

    def _release(self):
        a, ball = self.athlete, self.ball
        p0 = ball.pos.copy()
        dist = float(np.hypot(*(RIM - a.pos)[[0, 2]]))
        flight = 0.75 + 0.065 * dist
        sigma = 0.02 + 0.011 * dist
        target = RIM + np.array([self.rng.normal(0, sigma), 0.0, self.rng.normal(0, sigma) + 0.02])
        v = ballistic_velocity(p0, target, flight)
        horiz = np.array([v[0], 0.0, v[2]])
        backspin = np.cross(horiz / (np.linalg.norm(horiz) + 1e-6), UP) * 15.0
        ball.launch(v, spin=backspin)
        self.holding = False
        self.cooldown = 0.8
        self.attempts += 1
        self.shot = {"points": 3 if dist > THREE_PT else 2, "live": True}

    def _hoop_physics(self, prev):
        ball, r = self.ball, self.ball.r
        # backboard (front face only)
        if abs(ball.pos[0]) < 0.9 + r and 2.9 - r < ball.pos[1] < 3.95 + r \
                and ball.pos[2] - r < BOARD_Z <= prev[2] - r + 1e-6:
            ball.pos[2] = BOARD_Z + r
            ball.vel[2] = abs(ball.vel[2]) * 0.55
        # rim = ring of points; collide with the closest one
        d = ball.pos - RIM
        dh = np.array([d[0], 0.0, d[2]])
        ln = float(np.linalg.norm(dh))
        q = RIM + (dh / ln if ln > 1e-6 else np.array([1.0, 0.0, 0.0])) * RIM_R
        ball.bounce_off_point(q, RIM_TUBE, restitution=0.55)
        # pole
        ball.bounce_off_point(np.array([0.0, min(max(ball.pos[1], 0), 3.5), POLE_Z]), 0.09, 0.4)
        # basket made: the ball centre passes downwards through the ring
        if self.shot and self.shot["live"]:
            if prev[1] >= RIM[1] > ball.pos[1] and ln < RIM_R - 0.03:
                pts = self.shot["points"]
                self.score += pts
                self.shot["live"] = False
                self.flash(f"+{pts}  {'THREE!' if pts == 3 else 'SCORE!'}", 2.0)
                self.pending_celebrate = True
            elif ball.on_ground:
                self.shot["live"] = False

    def _give_ball(self, keep_position=False):
        a = self.athlete
        self.holding = True
        self.ball.held = True
        a.dribbling = True
        a.dribble_phase = 0.6
        self.shot = None
        if not keep_position:
            p = a.pos + a.forward() * 0.4
            self.ball.place((p[0], 0.8, p[2]))

    # ------------------------------------------------------------------ HUD
    def status(self):
        return f"Points: {self.score}    Shots: {self.attempts}"

    def help_lines(self):
        return ["WASD / arrows  move (relative to camera)", "Shift  sprint",
                "Space / right click  jump shot (auto-aims at the hoop)",
                "R  get the ball back"]
