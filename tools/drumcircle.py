"""Render drum-circle clips, one per Asian palette, through the full mix: ab/drums_<palette>.wav

    uv run python tools/drumcircle.py [seconds]

Each clip runs from the start of the 'trance' section into the 'storm'."""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import kor.music.compose as mc  # noqa: E402
from kor.music import SR, Composer  # noqa: E402


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    for pal in ("japan", "korea", "china", "india"):
        mc.FORCE_PALETTE = pal
        c = Composer(np.random.default_rng(7), style="ritual")
        tr = c.tr
        start = next(s for s in tr.sections if s.label == "trance").start * tr.bar_len - 3
        out = []
        while c.pos < (start + secs) * SR:
            out.append(c.next_bar()[0])
        y = np.concatenate(out)[int(start * SR): int((start + secs) * SR)].astype(np.float64)
        n = len(y)
        y *= np.minimum(1, np.minimum(np.arange(n) / (0.5 * SR), (n - np.arange(n)) / (2.0 * SR)))[:, None]
        path = f"ab/drums_{pal}.wav"
        with wave.open(path, "wb") as w:
            w.setnchannels(2), w.setsampwidth(2), w.setframerate(SR)
            w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
        rms = 10 * np.log10((y ** 2).mean())
        print(f"{path}: {tr.meter}, {round(60 * len(tr.groups) / tr.bar_len)} bpm, {tr.mode}, RMS {rms:.1f} dBFS, "
              f"peak {20 * np.log10(np.abs(y).max()):.1f} dBFS")
    mc.FORCE_PALETTE = None


if __name__ == "__main__":
    main()
