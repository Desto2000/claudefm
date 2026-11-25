"""Solo recordings of single instruments through the real mix (body resonance + mountain reverb).

    uv run python tools/solo.py ab/solo        # writes ab/solo_<instrument>.wav

Each plays a short idiomatic passage in D minor (Aeolian)."""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kor.music.core import SR, Note, fft_conv  # noqa: E402
from kor.music.mix import BODY_OF, Orchestra, master_eq  # noqa: E402

N = Note
D2, D3 = 38, 50


def passages():
    """(file name, instrument key, notes). Times in seconds."""
    bass = [  # solo, bowed over the fingerboard
        N(0.0, "basses", D2, 2.6, 0.55, tasto=True, players=1, swell=0.5),
        N(2.7, "basses", D2 + 3, 1.2, 0.5, tasto=True, players=1),
        N(3.9, "basses", D2 + 5, 1.2, 0.55, tasto=True, players=1, **{"from": D2 + 3, "glide": 0.15}),
        N(5.1, "basses", D2 + 7, 2.4, 0.6, tasto=True, players=1),
        N(7.6, "basses", D2 + 5, 1.0, 0.5, tasto=True, players=1),
        N(8.6, "basses", D2 + 3, 1.0, 0.5, tasto=True, players=1),
        N(9.6, "basses", D2, 3.4, 0.55, tasto=True, players=1, swell=0.6),
    ]
    section = [N(13.8 + 3.2 * k, "basses", m, 3.1, 0.6, swell=0.5) for k, m in enumerate((D2, D2 - 2, D2 - 4, D2 - 5))]
    walk = [N(27.0 + 0.6 * k, "basses", m, 0.55, 0.75, art="pizz")
            for k, m in enumerate((D2, D2 + 3, D2 + 5, D2 + 7, D2 + 8, D2 + 7, D2 + 5, D2 + 3,
                                   D2 - 2, D2, D2 + 3, D2 + 2, D2))]
    octo = [N(0.0, "octobass", D2 - 12, 6.0, 0.7, tasto=True, swell=0.4),
            N(6.2, "octobass", D2 - 12 + 3, 2.6, 0.65, tasto=True),
            N(8.9, "octobass", D2 - 12 + 5, 2.6, 0.7, tasto=True),
            N(11.6, "octobass", D2 - 12 + 7, 3.0, 0.75),
            N(14.8, "octobass", D2 - 12, 6.5, 0.7, tasto=True, swell=0.5)]
    khuur = [N(0.0, "khuur", 69, 1.4, 0.75),
             N(1.4, "khuur", 74, 1.6, 0.85, **{"from": 69, "glide": 0.13}, vib=32),
             N(3.0, "khuur", 72, 0.7, 0.75), N(3.7, "khuur", 70, 0.7, 0.75),
             N(4.4, "khuur", 69, 2.2, 0.8, vib=34),
             N(6.9, "khuur", 65, 0.9, 0.7), N(7.8, "khuur", 67, 0.9, 0.75, **{"from": 65, "glide": 0.1}),
             N(8.7, "khuur", 69, 1.2, 0.8), N(9.9, "khuur", 77, 1.8, 0.9, **{"from": 69, "glide": 0.16}, vib=36),
             N(11.7, "khuur", 74, 0.8, 0.8), N(12.5, "khuur", 72, 0.8, 0.75),
             N(13.3, "khuur", 74, 3.2, 0.8, trill=(1.0, 7.5, 1.4), vib=30)]
    serp = [N(0.0, "serpent", D2, 3.0, 0.5, swell=0.4),
            N(3.1, "serpent", D2 + 3, 1.3, 0.55), N(4.4, "serpent", D2 + 5, 1.3, 0.55),
            N(5.7, "serpent", D2 + 7, 2.2, 0.62), N(8.0, "serpent", D2 + 10, 1.5, 0.68),
            N(9.5, "serpent", D2 + 8, 1.2, 0.6), N(10.7, "serpent", D2 + 7, 1.2, 0.58),
            N(11.9, "serpent", D2 + 5, 1.2, 0.55), N(13.1, "serpent", D2 + 3, 1.2, 0.52),
            N(14.3, "serpent", D2, 4.5, 0.5, swell=0.5)]
    return [("double_bass", bass + section + walk), ("octobass", octo), ("morin_khuur", khuur),
            ("contrabass_serpent", serp)]


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "ab/solo"
    orch = Orchestra(np.random.default_rng(21))
    ir = orch.irs["mountain"]
    for name, notes in passages():
        end = max(n.t + n.dur for n in notes) + 4.5
        buf = np.zeros((int(end * SR), 2))
        for n in notes:
            x = orch.render(n)
            i = int(n.t * SR)
            e = min(len(buf), i + len(x))
            buf[i:e] += x[: e - i]
        body = orch.bodies[BODY_OF[notes[0].inst]]
        buf = np.stack([fft_conv(buf[:, c], body)[: len(buf)] for c in range(2)], 1)
        wet = np.stack([fft_conv(buf[:, c] * 0.45, ir[:, c])[: len(buf)] for c in range(2)], 1)
        y = buf + 0.55 * wet
        y = np.stack([fft_conv(y[:, c], master_eq(11000, low=1.2, fl=80.0))[: len(y)] for c in range(2)], 1)
        y *= 10 ** (-16 / 20) / (np.sqrt((y ** 2).mean()) + 1e-9)
        y = np.tanh(y * 1.1) / 1.1
        fade = np.minimum(1, (len(y) - np.arange(len(y))) / (1.5 * SR))
        y *= fade[:, None]
        path = f"{prefix}_{name}.wav"
        with wave.open(path, "wb") as w:
            w.setnchannels(2), w.setsampwidth(2), w.setframerate(SR)
            w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
        print(f"{path}  {len(y) / SR:.1f} s  peak {20 * np.log10(np.abs(y).max()):.1f} dBFS")


if __name__ == "__main__":
    main()
