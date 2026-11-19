"""Scripted run of the application that saves screenshots (used for the
report and as a quick smoke test).

    python tools/capture.py [output_dir]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from main import App, Controls  # noqa: E402

DT = 1 / 60


def run(app, seconds, move=(0, 0, 0), sprint=False, action_at=None, alt_at=None, reset=False):
    frames = int(seconds / DT)
    for i in range(frames):
        pygame.event.pump()
        ctl = Controls(move=np.array(move, float), sprint=sprint)
        ctl.reset = reset and i == 0
        ctl.action = action_at is not None and i == int(action_at / DT)
        ctl.alt = alt_at is not None and i == int(alt_at / DT)
        app.step(DT, ctl)
    app.render()


def shot(app, out, name):
    path = app.screenshot(os.path.join(out, name + ".png"))
    pygame.display.flip()
    print("saved", path)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "screenshots"
    os.makedirs(out, exist_ok=True)
    app = App(1280, 720)
    app.show_help = False

    # ---------------- football
    app.active = 0
    cam = app.camera
    cam.yaw, cam.pitch, cam.distance = 235.0, 14.0, 5.5
    run(app, 1.0)
    shot(app, out, "football_idle")
    run(app, 1.2, move=(1, 0, 0))
    shot(app, out, "football_walk")
    cam.yaw, cam.pitch, cam.distance = 300.0, 10.0, 6.0
    run(app, 0.9, move=(1, 0, 0), sprint=True)
    shot(app, out, "football_run")
    cam.yaw, cam.pitch, cam.distance = 190.0, 8.0, 4.5
    run(app, 0.30, action_at=0.0)
    shot(app, out, "football_kick")
    run(app, 0.35)
    cam.yaw, cam.pitch, cam.distance = 250.0, 18.0, 11.0
    run(app, 0.6)
    shot(app, out, "football_goal")
    run(app, 2.5)
    app.show_help = True
    cam.yaw, cam.pitch, cam.distance = 225.0, 30.0, 16.0
    run(app, 0.5)
    shot(app, out, "football_overview")
    app.show_help = False

    # ---------------- basketball
    app.active = 1
    cam = app.camera
    cam.yaw, cam.pitch, cam.distance = 120.0, 10.0, 4.5
    run(app, 1.3)
    shot(app, out, "basketball_dribble")
    cam.yaw, cam.pitch, cam.distance = 60.0, 12.0, 6.0
    run(app, 1.0, move=(0, 0, -1), sprint=True)
    shot(app, out, "basketball_run")
    run(app, 0.6, move=(0, 0, 0))
    cam.yaw, cam.pitch, cam.distance = 90.0, 8.0, 5.0
    run(app, 0.52, action_at=0.0)
    shot(app, out, "basketball_jump")
    cam.yaw, cam.pitch, cam.distance = 70.0, 12.0, 8.0
    run(app, 0.45)
    shot(app, out, "basketball_shot_flight")
    run(app, 3.0)
    app.show_help = True
    cam.yaw, cam.pitch, cam.distance = 30.0, 28.0, 13.0
    run(app, 0.5)
    shot(app, out, "basketball_overview")

    # ---------------- close-up of the heads (texture mapping on ellipsoids)
    for idx in (0, 1):
        app.active = idx
        a = app.scene.athlete
        cam = app.camera
        cam.yaw = (a.heading + 20.0) % 360.0          # slightly to the side of the face
        cam.pitch, cam.distance = 5.0, 2.6
        cam.target = a.pos + np.array([0.0, 1.45, 0.0])
        app.render()
        shot(app, out, f"closeup_{app.scene.title.lower()}")

    # ---------------- simple benchmark: update + render without vsync
    import time
    app.active = 0
    t0 = time.perf_counter()
    n = 300
    for _ in range(n):
        pygame.event.pump()
        app.step(DT, Controls())
        app.render()
        pygame.display.flip()
    print(f"benchmark: {n / (time.perf_counter() - t0):.1f} frames/s")
    pygame.quit()


if __name__ == "__main__":
    main()
