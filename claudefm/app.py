"""claude fm — generative music with animated landscapes, in a GPU window. Built on the kor engine.

    uv run python -m claudefm                               # 4K canvas, 8×16 cells (480×133 characters)
    uv run python -m claudefm --canvas 2560x1440 --cell 6x12
    uv run python -m claudefm --bench 120                   # time every stage on your GPU, no window
    uv run python -m claudefm --render out.wav --minutes 2  # music only, to a file

keys: n scene · t track · Ctrl+W war · c time of day · r look (ascii/blocks/braille)
      space pause · ↑↓ volume · f fullscreen · h hide bar · q quit
"""

from __future__ import annotations

import argparse
import functools
import time
import wave

import numpy as np

from kor.audio import Engine, Timeline
from kor.music import SR, Composer
from kor.renderer import LOOKS, Renderer

from . import scenes as cf
from .gpu_scenes import GPU_OF
from .show import GShow, hud

# A renderer whose show is claude fm's: the landscapes in rotation, the battlefield while war plays.
make_renderer = functools.partial(Renderer, show=GShow)


def render_wav(path, minutes, seed, style=None):
    composer = Composer(np.random.default_rng(seed), style=style)
    total = int(minutes * 60 * SR)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        done = 0
        while done < total:
            audio, info = composer.next_bar()
            if info["bar"] == 0:
                print(f"  track {info['track_no']:02d} · {info['track']} · {info['bpm']} bpm · {info['key']}")
            elif style == "war" and info["bar"] % 16 == 0:
                print(f"    bar {info['bar']:3d} · {info['bpm']} bpm · {info['key']} · {info['chord']}")
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
            done += len(audio)
    print(f"wrote {path} ({done / SR / 60:.1f} min)")


def bench(args):
    import warp as wp
    wp.init()
    dev = wp.get_preferred_device()
    r = make_renderer(args.canvas, args.cell, dev, args.seed, prefetch=False)   # no worker thread skewing timings
    composer = Composer(np.random.default_rng(1))
    timeline = Timeline()
    for _ in range(12):
        timeline.add(composer.next_bar()[1])
    pix = wp.zeros(r.OW * r.OH, dtype=wp.uint32, device=dev)
    print(f"device {dev} · canvas {r.OW}×{r.OH} · {r.cols}×{r.rows} characters · {r.SW}×{r.SH} samples")
    for S in cf.SCENES + [cf.War]:
        for look in LOOKS:
            r.look = look
            r.show.scene = GPU_OF[S](S(np.random.default_rng(3)), dev)
            r.show.prev = None
            r.timings = {}
            n = args.bench
            for i in range(n):
                c = r.show.context(i / 60, 1 / 60, r.SW, r.SH, int((3 + i / 60) * SR), timeline, 1.0, 2, 4)
                r.frame(c, pix, timed=i >= 3)
            ms = {k: 1000 * float(np.median(v)) for k, v in r.timings.items()}
            tot = sum(ms.values())
            print(f"{S.__name__:11s} {look:8s} scene {ms['scene']:7.2f} ms · cells {ms['cells']:6.2f} ms · "
                  f"raster {ms['raster']:6.2f} ms · total {tot:7.2f} ms ≈ {1000 / tot:6.0f} fps")


def run_window(args):
    import warp as wp

    from kor.window import run
    wp.init()
    dev = wp.get_preferred_device()
    r = make_renderer(args.canvas, args.cell, dev, args.seed, graphs=not args.no_graph)
    r.look = args.look
    composer = Composer(np.random.default_rng(r.rng.integers(1 << 32)))
    if args.war:                                  # start at war (as if ^w was pressed)
        composer.toggle_war()
        composer.restart_at(0)
    timeline = Timeline()
    engine = Engine(composer, timeline, args.volume)
    while not engine.ready():
        time.sleep(0.02)
    engine.start_output(silent=args.silent)

    def on_key(sym, mod, t, k):                   # kor handles q, r, space, ↑↓, f, h
        if sym == k.N:
            r.show.user_next(t)
        elif sym == k.W and mod & k.MOD_CTRL:     # ^w: war, until pressed again
            composer.toggle_war()
            engine.skip = True
        elif sym == k.T:
            engine.skip = True
        elif sym == k.C:
            r.show.recolor()

    def status(c, look, fps):
        return hud(c, engine, r.show, r.hud_n, f"{look} · {fps:.0f} fps")

    run(r, engine, timeline, title="claude fm", vsync=args.vsync, present=args.present, debug_gl=args.debug_gl,
        seconds=args.seconds, screenshot=args.screenshot, on_key=on_key, hud=status)


def main():
    p = argparse.ArgumentParser(description="claude fm: generative music and animated landscapes (kor engine).")
    size = lambda s: tuple(int(v) for v in s.lower().split("x"))
    p.add_argument("--canvas", type=size, default=(3840, 2160), help="fixed render size, e.g. 3840x2160 or 2560x1440")
    p.add_argument("--cell", type=size, default=(8, 16), help="character cell in canvas pixels, e.g. 8x16, 6x12")
    p.add_argument("--look", choices=LOOKS, default="ascii")
    p.add_argument("--seed", type=int, help="seed for a reproducible session")
    p.add_argument("--volume", type=int, default=70, help="starting volume 0-100")
    p.add_argument("--silent", action="store_true", help="visuals only, no audio")
    p.add_argument("--war", action="store_true", help="start at war (Ctrl+W toggles it)")
    p.add_argument("--vsync", action="store_true", help="cap at the display's refresh rate")
    p.add_argument("--debug-gl", action="store_true", help="check every OpenGL call for errors (slow)")
    p.add_argument("--no-graph", action="store_true", help="launch kernels one by one instead of as a CUDA graph")
    p.add_argument("--present", choices=("area", "mip"), default="area",
                   help="how the canvas is scaled into the window: exact box filter (default) or mipmaps (old)")
    p.add_argument("--bench", type=int, metavar="FRAMES", help="time each stage per scene and look, no window")
    p.add_argument("--render", metavar="WAV", help="render music to a .wav instead of playing")
    p.add_argument("--minutes", type=float, default=3.0, help="length for --render")
    p.add_argument("--style", help="--render only this style (e.g. war, hearth, ritual)")
    p.add_argument("--seconds", type=float, help=argparse.SUPPRESS)
    p.add_argument("--screenshot", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.render:
        return render_wav(args.render, args.minutes, args.seed, args.style)
    if args.bench:
        return bench(args)
    run_window(args)
