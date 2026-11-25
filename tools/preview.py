"""Render a short landscape preview video (1920×1080, with its music) of a scene reacting to the music.

    uv run python tools/preview.py --out summit_calm.mp4 --weather clear
    uv run python tools/preview.py --out summit_storm.mp4 --weather storm --seconds 20

The track starts at its warm middle section (the lead is playing). Uses ffmpeg from imageio-ffmpeg."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warp as wp  # noqa: E402

from claudefm import gpu_scenes as g  # noqa: E402
from claudefm import scenes as cf  # noqa: E402
from claudefm.app import make_renderer  # noqa: E402
from kor.audio import Timeline  # noqa: E402
from kor.music import SR, Composer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weather", choices=["clear", "snowfall", "storm"])
    ap.add_argument("--scene", default="Summit")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--canvas", default="3840x2160")
    args = ap.parse_args()
    W, H = (int(v) for v in args.canvas.split("x"))

    comp = Composer(np.random.default_rng(args.seed), style="hearth")
    tr = comp.tr
    if args.weather:
        tr.weather = args.weather
    start = next(s for s in tr.sections if s.label == "warmth").start * tr.bar_len
    timeline, chunks = Timeline(), []
    while comp.pos < (start + args.seconds + 1) * SR:
        y, info = comp.next_bar()
        timeline.add(info)
        chunks.append(y)
    audio = np.concatenate(chunks)[int(start * SR): int((start + args.seconds) * SR)].astype(np.float64)
    n = len(audio)
    audio *= np.minimum(1, np.minimum(np.arange(n) / (1.0 * SR), (n - np.arange(n)) / (1.5 * SR)))[:, None]

    wp.config.quiet = True
    wp.init()
    dev = wp.get_preferred_device()
    r = make_renderer((W, H), (8, 16), dev, seed=7, hud=False, prefetch=False)
    S = getattr(cf, args.scene)
    r.show.scene = g.GPU_OF[S](S(np.random.default_rng(5)), dev)
    pix = wp.zeros(W * H, dtype=wp.uint32, device=dev)
    fps = 30

    def frame(i):
        t = i / fps
        r.show.switch_at = 1e9
        c = r.show.context(t, 1 / fps, r.SW, r.SH, int((start + t) * SR), timeline, 1.0, 2, 4)
        r.frame(c, pix)
        img = pix.numpy().view(np.uint8).reshape(H, W, 4)[::-1, :, :3]
        k = float(np.clip(min(t / 0.8, (args.seconds - t) / 1.0), 0, 1))
        return (img * k).astype(np.uint8) if k < 1 else np.ascontiguousarray(img)

    for i in range(90):                                     # let snow, fire and the weather settle in
        frame(i)
    r.show.scene.cpu.look = dict(r.show.scene.cpu.look)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    wav = args.out + ".wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(args.seconds * fps)):
        p.stdin.write(frame(i).tobytes())
    p.stdin.close()
    p.wait()
    os.remove(wav)
    print("video", args.out, "exit", p.returncode, "weather", tr.weather)


if __name__ == "__main__":
    main()
