"""Keyframe animation with easing, cross-fade blending and phase sync.

A *pose* is a dict joint-name -> angle in degrees (plus a few non-angle
channels such as 'root_y', the vertical offset of the pelvis).  Joints that
are missing from a pose are treated as 0 (the neutral, standing pose).
"""
from .mathutil import ease_cos, lerp, smoothstep


def mirror(pose):
    """Swap left and right joints - used to build the second half of gait cycles."""
    out = {}
    for k, v in pose.items():
        if k.startswith("l_"):
            out["r_" + k[2:]] = v
        elif k.startswith("r_"):
            out["l_" + k[2:]] = v
        elif k.endswith("_yaw") or k.endswith("_roll"):
            out[k] = -v          # twisting / leaning of the body changes sign
        else:
            out[k] = v
    return out


def blend(a, b, w):
    keys = set(a) | set(b)
    return {k: lerp(a.get(k, 0.0), b.get(k, 0.0), w) for k in keys}


class Clip:
    """Sequence of keyframes [(normalised_time, pose), ...] over `duration` seconds."""

    def __init__(self, name, duration, keys, loop=True, events=None):
        self.name = name
        self.duration = duration
        self.loop = loop
        self.keys = sorted(keys, key=lambda k: k[0])
        if loop and self.keys[-1][0] < 1.0:
            self.keys.append((1.0, self.keys[0][1]))  # close the cycle
        self.events = events or {}  # name -> normalised time (e.g. ball release)

    def sample(self, t):
        u = (t % self.duration) / self.duration if self.loop else min(t / self.duration, 1.0)
        keys = self.keys
        if u <= keys[0][0]:
            return dict(keys[0][1])
        for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
            if u <= t1:
                s = ease_cos((u - t0) / (t1 - t0)) if t1 > t0 else 1.0
                return blend(p0, p1, s)
        return dict(keys[-1][1])


class Animator:
    """Plays one clip at a time and cross-fades from the previous pose."""

    def __init__(self, clips):
        self.clips = {c.name: c for c in clips}
        self.current = None
        self.time = 0.0
        self.speed = 1.0
        self.pose = {}
        self._from = {}
        self._blend_t = 1.0
        self._blend_dur = 0.0
        self._fired = set()

    @property
    def clip(self):
        return self.clips[self.current]

    def play(self, name, speed=1.0, fade=0.2, restart=False, keep_phase=False):
        self.speed = speed
        if name == self.current and not restart:
            return
        phase = self.phase() if keep_phase and self.current else 0.0
        self._from = dict(self.pose)
        self._blend_t, self._blend_dur = 0.0, fade
        self.current = name
        self.time = phase * self.clips[name].duration
        self._fired = set()

    def phase(self):
        c = self.clip
        if c.loop:
            return (self.time % c.duration) / c.duration
        return min(self.time / c.duration, 1.0)

    @property
    def finished(self):
        return not self.clip.loop and self.time >= self.clip.duration

    def update(self, dt):
        """Advance time, return the list of clip events that fired this frame."""
        self.time += dt * self.speed
        self._blend_t += dt
        target = self.clip.sample(self.time)
        if self._blend_dur > 0 and self._blend_t < self._blend_dur:
            target = blend(self._from, target, smoothstep(self._blend_t / self._blend_dur))
        self.pose = target
        fired = []
        u = self.phase()
        for ev, at in self.clip.events.items():
            if ev not in self._fired and u >= at:
                self._fired.add(ev)
                fired.append(ev)
        return fired
