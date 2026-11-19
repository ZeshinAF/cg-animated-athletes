"""Footballer and the football pitch scene.

Ball control: when the ball is close to the feet the player takes
possession and dribbles it (the ball is pushed ahead with every step and
rolls according to its displacement).  Space shoots, Q passes; the ball is
released at the 'contact' event of the kick animation."""
import math

import numpy as np
from OpenGL.GL import glLineWidth, glDisable, glEnable, GL_LIGHTING

from core import primitives as P
from core import textures as T
from core.animation import Clip
from core.athlete import Athlete
from core.physics import Ball, UP
from core.scene import Scene

FOOTBALLER_LOOK = {
    "jersey": dict(primary=(200, 30, 45), secondary=(250, 250, 250), number=10,
                   name="Ivanov", stripes=True),
    "face": dict(skin=(226, 182, 150), hair=(70, 45, 25), eye=(60, 90, 50)),
    "socks": dict(color=(200, 30, 45), stripe=(250, 250, 250)),
    "skin": (226, 182, 150),
    "shirt_color": (200, 30, 45),
    "shorts_color": (245, 245, 245),
    "shorts_len": 0.2,
    "shoe_color": (25, 25, 30),
    "sole_color": (235, 200, 40),
    "sleeves": True,
}

PITCH_L, PITCH_W = 30.0, 20.0
GOAL_HALF_W, GOAL_H, GOAL_DEPTH = 2.5, 2.0, 1.2
POST_R = 0.06


def kick_clip():
    rest = {"torso_pitch": 6, "l_shoulder_roll": 10, "r_shoulder_roll": 10,
            "l_elbow": 20, "r_elbow": 20}
    windup = {"r_hip": -40, "r_knee": 95, "r_ankle": -25, "l_hip": 10, "l_knee": 18,
              "torso_pitch": -4, "torso_yaw": -10, "l_shoulder": 25, "l_shoulder_roll": 40,
              "r_shoulder": -25, "r_shoulder_roll": 15, "l_elbow": 20, "r_elbow": 20,
              "root_y": -0.03}
    strike = {"r_hip": 38, "r_knee": 12, "r_ankle": -30, "l_hip": 8, "l_knee": 15,
              "torso_pitch": -10, "torso_yaw": 12, "l_shoulder": 10, "l_shoulder_roll": 45,
              "r_shoulder": 20, "r_shoulder_roll": 20, "l_elbow": 15, "r_elbow": 25,
              "root_y": -0.02}
    follow = {"r_hip": 78, "r_knee": 10, "r_ankle": -20, "l_knee": 10, "torso_pitch": -14,
              "torso_yaw": 15, "l_shoulder_roll": 35, "r_shoulder": 35, "r_shoulder_roll": 15,
              "l_elbow": 15, "r_elbow": 25, "root_y": 0.02}
    return Clip("kick", 0.75, [(0.0, rest), (0.28, windup), (0.42, strike), (0.65, follow),
                               (1.0, rest)], loop=False, events={"contact": 0.42})


class Footballer(Athlete):
    def __init__(self, bounds):
        look = dict(FOOTBALLER_LOOK)
        look["jersey_tex"] = T.upload(T.jersey(**look["jersey"]))
        look["face_tex"] = T.upload(T.face(**look["face"]))
        look["sock_tex"] = T.upload(T.socks(**look["socks"]))
        look["skin_tex"] = T.upload(T.skin_tex(look["skin"]))
        super().__init__(look, [kick_clip()], bounds)

    def action_move_factor(self):
        return 0.25 if self.action == "kick" else 0.0


class FootballScene(Scene):
    title = "Football"

    def __init__(self):
        super().__init__()
        self.pitch_tex = T.upload(T.pitch(PITCH_L, PITCH_W), repeat=False)
        self.grass_tex = T.upload(T.grass_tile())
        self.boards_tex = T.upload(T.boards())
        hl, hw = PITCH_L / 2 + 2.0, PITCH_W / 2 + 2.0
        self.athlete = Footballer(bounds=(-hl + 0.5, hl - 0.5, -hw + 0.5, hw - 0.5))
        self.athlete.pos[:] = (-3.0, 0.0, 0.0)
        self.athlete.heading = 90.0
        self.ball = Ball(0.11, T.upload(T.football()), restitution=0.65, rolling_friction=0.55)
        self.ball.bounds = (-hl, hl, -hw, hw)
        self.ball.place((-2.4, 0.11, 0.0))
        self.has_ball = False
        self.cooldown = 0.0
        self.kick_type = "shot"
        self.reset_timer = 0.0
        self.nets = [self._net(s) for s in (-1, 1)]
        self.camera_yaw = 270.0

    # ----------------------------------------------------------- geometry
    def _net(self, side):
        x0, x1 = side * PITCH_L / 2, side * (PITCH_L / 2 + GOAL_DEPTH)
        segs = []
        n = 12
        for i in range(n + 1):             # back net: vertical + horizontal lines
            z = -GOAL_HALF_W + 2 * GOAL_HALF_W * i / n
            segs.append(((x1, 0, z), (x1, GOAL_H, z)))
            segs.append(((x0, GOAL_H, z), (x1, GOAL_H, z)))
        for j in range(7):
            y = GOAL_H * j / 6
            segs.append(((x1, y, -GOAL_HALF_W), (x1, y, GOAL_HALF_W)))
            for s in (-1, 1):
                segs.append(((x0, y, s * GOAL_HALF_W), (x1, y, s * GOAL_HALF_W)))
        for k in range(5):                 # side nets and roof depth lines
            x = x0 + (x1 - x0) * k / 4
            segs.append(((x, GOAL_H, -GOAL_HALF_W), (x, GOAL_H, GOAL_HALF_W)))
            for s in (-1, 1):
                segs.append(((x, 0, s * GOAL_HALF_W), (x, GOAL_H, s * GOAL_HALF_W)))
        return segs

    def _goal_frames(self):
        for side in (-1, 1):
            x = side * PITCH_L / 2
            for s in (-1, 1):
                P.cylinder_between((x, 0, s * GOAL_HALF_W), (x, GOAL_H, s * GOAL_HALF_W), POST_R)
            P.cylinder_between((x, GOAL_H, -GOAL_HALF_W - POST_R), (x, GOAL_H, GOAL_HALF_W + POST_R), POST_R)

    def draw_ground(self):
        P.material((0.9, 0.95, 0.9), self.grass_tex)
        P.ground_quad(-80, -80, 80, 80, y=-0.01, repeat=(40, 40))
        P.material((1, 1, 1), self.pitch_tex)
        hl, hw = PITCH_L / 2 + 2.0, PITCH_W / 2 + 2.0
        P.ground_quad(-hl, -hw, hl, hw)

    def draw_props(self):
        P.material((0.97, 0.97, 0.97))
        self._goal_frames()
        glDisable(GL_LIGHTING)
        P.material((0.85, 0.85, 0.85))
        glLineWidth(1.5)
        for net in self.nets:
            P.lines(net)
        glEnable(GL_LIGHTING)
        # advertising boards around the pitch
        P.material((1, 1, 1), self.boards_tex)
        hl, hw = PITCH_L / 2 + 2.05, PITCH_W / 2 + 2.05
        uv_long = {"pz": (0, 0, 3, 1), "nz": (0, 0, 3, 1), "px": (0, 0, 2, 1), "nx": (0, 0, 2, 1)}
        P.box(2 * hl, 0.7, 0.08, center=(0, 0.35, -hw), uv=uv_long)
        P.box(2 * hl, 0.7, 0.08, center=(0, 0.35, hw), uv=uv_long)
        P.box(0.08, 0.7, 2 * hw, center=(-hl, 0.35, 0), uv=uv_long)
        P.box(0.08, 0.7, 2 * hw, center=(hl, 0.35, 0), uv=uv_long)

    def draw_prop_shadows(self):
        self._goal_frames()

    # ------------------------------------------------------------ gameplay
    def update(self, dt, ctl):
        a, ball = self.athlete, self.ball
        if ctl.reset:
            self._reset_ball(in_front=True)
        if (ctl.action or ctl.alt) and a.action is None:
            self.kick_type = "shot" if ctl.action else "pass"
            a.start_action("kick")
        events = a.update(dt, ctl.move, ctl.sprint)
        fwd = a.forward()

        if self.has_ball:
            if "contact" in events:
                if self.kick_type == "shot":
                    v = fwd * 17.0 + UP * 5.0
                else:
                    v = fwd * 10.0 + UP * 0.6
                ball.launch(v)
                self.has_ball = False
                self.cooldown = 0.5
            else:
                # the ball is pushed ahead with every step (two touches per gait cycle)
                touch = 0.1 * math.sin(4 * math.pi * a.locomotion_phase()) if a.speed > 0.3 else 0.0
                target = a.pos + fwd * (0.45 + 0.04 * a.speed + touch)
                target[1] = ball.r
                k = 1.0 - math.exp(-12.0 * dt)
                ball.carry_to(ball.pos + (target - ball.pos) * k, dt)
        else:
            self.cooldown -= dt
            prev = ball.pos.copy()
            ball.update(dt)
            self._goal_physics(prev)
            feet = a.pos + fwd * 0.3
            near = np.hypot(*(ball.pos - feet)[[0, 2]]) < 0.55
            if self.cooldown <= 0 and near and ball.pos[1] < 0.5 and self.reset_timer <= 0:
                self.has_ball = True
                ball.held = True

        if self.reset_timer > 0:
            self.reset_timer -= dt
            if self.reset_timer <= 0:
                self._reset_ball(in_front=False)
        self.tick_message(dt)

    def _goal_physics(self, prev):
        ball = self.ball
        r = ball.r
        for side in (-1, 1):
            gx = PITCH_L / 2
            sx, psx = ball.pos[0] * side, prev[0] * side
            # goal frame: closest points on the posts and the crossbar
            for s in (-1, 1):
                ball.bounce_off_point(np.array([side * gx, min(max(ball.pos[1], 0), GOAL_H), s * GOAL_HALF_W]),
                                      POST_R, 0.5)
            ball.bounce_off_point(np.array([side * gx, GOAL_H, min(max(ball.pos[2], -GOAL_HALF_W), GOAL_HALF_W)]),
                                  POST_R, 0.5)
            inside = gx < sx < gx + GOAL_DEPTH + r and abs(ball.pos[2]) < GOAL_HALF_W + r and ball.pos[1] < GOAL_H + r
            if not inside:
                continue
            if psx >= gx + GOAL_DEPTH + r:                       # hit the back net from behind
                ball.pos[0] = side * (gx + GOAL_DEPTH + r)
                ball.vel[0] *= -0.2
                continue
            if abs(prev[2]) >= GOAL_HALF_W + r or prev[1] >= GOAL_H + r:   # from the side or the top
                ball.pos[2] = np.sign(prev[2]) * (GOAL_HALF_W + r) if abs(prev[2]) >= GOAL_HALF_W + r else ball.pos[2]
                if prev[1] >= GOAL_H + r:
                    ball.pos[1] = GOAL_H + r
                    ball.vel[1] = abs(ball.vel[1]) * 0.2
                else:
                    ball.vel[2] *= -0.2
                continue
            if psx < gx <= sx and self.reset_timer <= 0:        # crossed the goal line
                self.score += 1
                self.flash("GOAL!", 2.5)
                self.reset_timer = 3.0
                if self.athlete.action is None:
                    self.athlete.start_action("celebrate")
            # the net catches the ball
            if sx > gx + GOAL_DEPTH - r:
                ball.pos[0] = side * (gx + GOAL_DEPTH - r)
                ball.vel[0] *= -0.15
                ball.vel[2] *= 0.5
            if abs(ball.pos[2]) > GOAL_HALF_W - r:
                ball.pos[2] = np.sign(ball.pos[2]) * (GOAL_HALF_W - r)
                ball.vel[2] *= -0.2
            if ball.pos[1] > GOAL_H - r:
                ball.pos[1] = GOAL_H - r
                ball.vel[1] = -abs(ball.vel[1]) * 0.2

    def _reset_ball(self, in_front):
        a = self.athlete
        self.has_ball = False
        self.ball.held = False
        self.reset_timer = 0.0
        if in_front:
            p = a.pos + a.forward() * 0.6
            self.ball.place((p[0], self.ball.r, p[2]))
        else:
            self.ball.place((0.0, self.ball.r, 0.0))

    # ------------------------------------------------------------------ HUD
    def status(self):
        return f"Goals: {self.score}    Ball: {'at feet' if self.has_ball else 'free'}"

    def help_lines(self):
        return ["WASD / arrows  move (relative to camera)", "Shift  sprint",
                "Space / right click  shoot     Q  pass", "R  reset ball"]

