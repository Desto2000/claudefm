"""What plays when: scene scheduling, dissolves and the music-driven context the scenes react to (`Show`,
`GShow` prepares the next scene on a worker thread), and the status bar text."""

from __future__ import annotations

import concurrent.futures
import copy
import math
import sys

import numpy as np

from kor.music import SR
from kor.scene import Ctx as _Ctx
from kor.scene import in_background, prepare

from . import scenes as cf
from .gpu_scenes import GPU_OF
from .scenes import SCENES, War


class Ctx(_Ctx):
    """What a claude fm scene sees each frame. Besides time and size (kor.scene.Ctx):
      kick, snare      decaying pulses on drum hits          level   overall loudness (0..1)
      lead             the lead instrument's loudness, 0..1 relative to its recent peak
      lead_pitch       the last lead note, 0 (low) .. 1 (high)
      bass             the low layers' loudness, 0..1 relative to their recent peak
      gust             the wind gust you hear, 0..~1.2 (the same curve the wind sound follows)
      pop              a pulse on every audible fire crackle, scaled by its size
      pluck            a pulse on every plucked ember note
      chord            a slow pulse (3 s) after each chord change
      energy           the section's energy, 0..1;  weather  'clear' | 'snowfall' | 'storm'"""
    __slots__ = ("lead", "lead_pitch", "bass", "gust", "pop", "pluck", "chord", "energy", "weather")


class Show:
    """Picks scenes, dissolves between them and builds the context they react to.
    Used by the window and by offline preview rendering alike."""

    FADE = 1.6

    def __init__(self, rng):
        self.rng = rng
        self.war = False            # a war track is playing: the battlefield, and it stays until the war ends
        self.scene = self._new_scene(None)
        self.prev = None
        self.fade_t = 0.0
        self.switch_at = self._next_switch(0)
        self.level = 0.0
        self.track = None
        self.mask = None

    def _new_scene(self, current):
        if self.war:
            return War(self.rng)
        choices = [s for s in SCENES if current is None or s is not type(current)]
        return choices[int(self.rng.integers(len(choices)))](self.rng)

    def _next_switch(self, t):
        return math.inf if self.war else t + self.rng.uniform(22, 45)

    def user_next(self, t):
        """The 'n' key: another scene — except during a war, whose battlefield stays."""
        if not self.war:
            self.next_scene(t)

    def next_scene(self, t):
        self.prev, self.scene = self.scene, self._new_scene(self.scene)
        self.fade_t = t
        self.mask = None
        self.switch_at = self._next_switch(t)

    def context(self, t, dt, W, H, s, timeline, a=2.0, sx=1, sy=1):
        info, lk, ls = timeline.at(s)
        c = Ctx()
        c.t, c.dt, c.W, c.H, c.info = t, dt, W, H, info
        c.a, c.sx, c.sy = a, sx, sy
        c.kick = math.exp(-(s - lk) / (0.11 * SR)) if lk is not None and s >= lk else 0.0
        c.snare = math.exp(-(s - ls) / (0.09 * SR)) if ls is not None and s >= ls else 0.0
        lvl, c.beat = 0.0, 0.0
        if info:
            rel = s - info["start"]
            c.beat = info.get("beat0", info["bar"] * info.get("beats", 4)) + rel / info["beat_len"]
            env = info["env"]
            if len(env):
                lvl = float(env[int(np.clip(rel // 1024, 0, len(env) - 1))])
        self.level += (min(1.0, lvl / 0.2) - self.level) * min(1.0, dt * 8)
        c.level = self.level
        self._features(c, s, info, timeline, dt)
        war = bool(info and info.get("war"))
        if war != self.war:                         # war begins (^w) or ends: in or out of the battlefield
            self.war = war
            self.track = info["track_no"] if info else self.track
            self.next_scene(t)
        elif info and info["track_no"] != self.track:
            if self.track is not None and not war:  # one war track after another: the same battlefield
                self.next_scene(t)
            self.track = info["track_no"]
        elif t >= self.switch_at:
            self.next_scene(t)
        return c

    def _features(self, c, s, info, timeline, dt):
        """Fill the context with what the music is doing (see Ctx). Envelopes are normalised against their own
        slowly decaying peak, so quiet and loud tracks both move the scene."""
        st = self.__dict__.setdefault("_feat", {"lead": 0.0, "bass": 0.0, "lmax": 0.05, "bmax": 0.05,
                                                 "gust": 0.5, "pitch": 0.5})

        def block(key):
            if not info or key not in info or not len(info[key]):
                return 0.0
            a = info[key]
            return float(a[int(np.clip((s - info["start"]) // 1024, 0, len(a) - 1))])
        k = min(1.0, dt * 6)
        lead, bass = block("lead_env"), block("bass_env")
        st["lmax"] = max(lead, st["lmax"] * (1 - dt * 0.05), 0.02)
        st["bmax"] = max(bass, st["bmax"] * (1 - dt * 0.05), 0.02)
        st["lead"] += (lead / st["lmax"] - st["lead"]) * k
        st["bass"] += (bass / st["bmax"] - st["bass"]) * k
        st["gust"] += ((block("gust") if info and "gust" in info else 0.5) - st["gust"]) * min(1.0, dt * 4)
        c.lead, c.bass, c.gust = st["lead"], st["bass"], st["gust"]
        get = getattr(timeline, "last", None)
        ev = (lambda kind: get(kind, s)) if get else (lambda kind: None)
        ln = ev("leads")
        if ln is not None:
            st["pitch"] += (np.clip((ln[2] - 48) / 36, 0, 1) - st["pitch"]) * min(1.0, dt * 2)
        c.lead_pitch = st["pitch"]
        pp = ev("pops")
        c.pop = pp[1] / 4 * math.exp(-(s - pp[0]) / (0.12 * SR)) if pp is not None else 0.0
        pk = ev("plucks")
        c.pluck = math.exp(-(s - pk) / (0.35 * SR)) if pk is not None else 0.0
        cc = ev("chords")
        c.chord = math.exp(-(s - cc) / (3.0 * SR)) if cc is not None else 0.0
        c.energy = float(info.get("energy", 0.5)) if info else 0.5
        c.weather = info.get("weather", "snowfall") if info else "snowfall"
        self.weather = c.weather


RESET = "\x1b[0m"


ESC_CACHE: dict = {}


def esc(rgb_):
    s = ESC_CACHE.get(rgb_)
    if s is None:
        s = ESC_CACHE[rgb_] = "\x1b[38;2;%d;%d;%dm" % rgb_
    return s


ORANGE = esc((217, 119, 87))


DIM = esc((110, 110, 120))


LIGHT = esc((225, 222, 214))


def hud(c, engine, show, W, mode="ascii"):
    info = c.info
    segs = [(ORANGE, " ✻ claude fm  ")]
    if info:
        nb = info.get("beats", 4)
        beat = int(c.beat - info.get("beat0", info["bar"] * nb)) % nb
        dots = "".join("●" if i == beat else "·" for i in range(nb))
        segs += [(LIGHT, f"{info['track_no']:02d} · {info['track']}"),
                 (DIM, f"   {info['bpm']} bpm · {info['key']} · "), (LIGHT, info["chord"]),
                 (ORANGE, f"  {dots}")]
    else:
        segs += [(DIM, "warming up the tape…")]
    state = "paused" if engine.paused else f"vol {round(engine.volume * 100)}%"
    used = sum(len(t) for _, t in segs)
    right = f"{show.scene.name} · {mode} · {state} "
    for extra in (f"· {engine.backend}  n scene · t track · ^w war · c time · r look · space pause · ↑↓ vol · h · q ",
                  "  n · t · ^w · c · r · space · ↑↓ · q "):
        if used + len(right) + len(extra) + 2 <= W:
            right += extra
            break
    if used + len(right) + 2 > W:
        right = f"{state} "
    pad = max(1, W - used - len(right))
    line, n = [], 0
    for col, txt in segs + [(DIM, " " * pad + right)]:
        txt = txt[: max(0, W - n)]
        line.append(col + txt)
        n += len(txt)
    return "".join(line)


class GShow(Show):
    """Show, with each scene prepared ahead on a worker thread.

    A scene's numpy setup takes 40–400 ms at 4K, so the next scene is prepared on a worker thread while the
    current one plays; a switch (timer, new track, 'n') then costs only the host→device copies. A switch that
    comes before the next scene is ready (e.g. 'n' pressed twice quickly) waits for it rather than stalling a
    frame. 'c' (new time of day) works the same way on a copy of the current scene.
    With prefetch=False everything runs synchronously, as before (used by --bench)."""

    def __init__(self, rng, dev, prefetch=True):
        self.dev, self.prefetch = dev, prefetch
        if prefetch:
            # setup is largely Python-level numpy loops; at CPython's default 5 ms switch interval the render
            # loop waits 13–21 ms each time it takes the GIL back from the worker (measured on Windows,
            # Python 3.12); at 0.5 ms the worst wait is ~3 ms. Process-wide; also helps the audio callback.
            sys.setswitchinterval(min(sys.getswitchinterval(), 0.0005))
        self.size = None            # (W, H, a, sx, sy) of the sample grid, known from the first context()
        self.ready = None           # Future → the next scene, set up and packed
        self.want = False           # a switch is due and happens as soon as `ready` is done
        self.recoloring = None      # (scene it was made from, Future → recoloured copy)
        super().__init__(rng)

    def _pick(self, exclude, war=False):
        choices = [s for s in cf.SCENES if s is not exclude]
        S = cf.War if war else choices[int(self.rng.integers(len(choices)))]
        # each scene gets its own generator: its setup runs on another thread than the show's draws
        return GPU_OF[S](S(np.random.default_rng(int(self.rng.integers(1 << 62)))), self.dev)

    def _new_scene(self, current):      # Show.__init__: the first scene (set up on the first frame)
        return self._pick(None if current is None else type(current.cpu))

    def _run(self, fn, *args):
        if self.prefetch:
            return in_background(fn, *args)
        fut = concurrent.futures.Future()
        fut.set_result(fn(*args))
        return fut

    def _prepare_next(self, war=False):
        self.ready = self._run(prepare, self._pick(type(self.scene.cpu), war), self.size)
        self.ready_war = war

    def next_scene(self, t):            # Show.context (timer, new track, war begins or ends) and the 'n' key
        if self.size is not None and self.war != getattr(self, "ready_war", False):
            self._prepare_next(self.war)    # the battlefield (or, after the war, a normal scene) instead
        self.want = True
        self.switch_at = math.inf

    def recolor(self):                  # the 'c' key
        if self.size is None:
            return self.scene.recolor()
        cur = self.scene
        twin = type(cur)(copy.deepcopy(cur.cpu), self.dev)
        self.recoloring = (cur, self._run(prepare, twin, self.size, True))

    def context(self, t, dt, W, H, s, timeline, a=2.0, sx=1, sy=1):
        if (W, H, a, sx, sy) != self.size:
            self.size = (W, H, a, sx, sy)
            self.recoloring = None
            self._prepare_next()        # (re)start for this grid size
        c = super().context(t, dt, W, H, s, timeline, a, sx, sy)
        if self.want and self.ready.done():
            gs = self.ready.result()    # re-raises anything that failed on the worker
            gs.commit()
            self.prev, self.scene = self.scene, gs
            self.fade_t, self.mask, self.want = t, None, False
            self.switch_at = self._next_switch(t)
            self.recoloring = None
            self._prepare_next()
        if self.recoloring and self.recoloring[1].done():
            src, fut = self.recoloring
            self.recoloring = None
            gs = fut.result()
            if src is self.scene:       # dropped if the scene changed meanwhile
                gs.commit()
                gs.seed = src.seed
                self.scene = gs
        return c
