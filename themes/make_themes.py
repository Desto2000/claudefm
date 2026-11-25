"""Three standalone pieces played by claude fm's music engine (instruments, bodies, rooms), each scored by hand
here and mixed offline. Nothing in the radio uses this; `kor.music` is only imported, never changed.

    uv run python themes/make_themes.py                  # all three → themes/*.wav
    uv run python themes/make_themes.py modern           # one of: prehistoric medieval modern

  prehistoric  "Ochre"            A minor pentatonic, 75 bpm. A cave: fire, bone flutes, throat drone, log and
                                  frame drums, rattles, stones, a men's chant that turns into a trance.
  medieval     "Lantern Road"     D dorian. Plainchant in organum under a bell, then an estampie in 6/8:
                                  hurdy-gurdy with its buzzing trompette, recorder, lyre, tabor; Amen.
  modern       "Glass Skyline"    E minor, 120 bpm, i–VI–III–VII. Cinematic minimalism with a pop pulse:
                                  harp ostinato, string spiccato, a beat, sidechain pump, riser, drop.
"""

from __future__ import annotations

import math
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kor.music.core import F32, SR, Compressor, Note, fft_conv, fft_filter, pan_gains  # noqa: E402
from kor.music.mix import BODY_OF, STAGE, Mixer, Orchestra, make_ir, master_eq  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


class Score:
    """A note list with a little human timing."""

    def __init__(self, seed, sd=0.004):
        self.r = np.random.default_rng(seed)
        self.notes = []
        self.sd = sd

    def add(self, t, inst, m, dur, vel, sd=None, **art):
        j = self.r.normal(0, self.sd if sd is None else sd)
        self.notes.append(Note(max(0.0, t + j), inst, float(m), dur, float(np.clip(vel, 0.02, 1.0)), **art))

    def hit(self, t, sound, vel, pan=0.0, m=0.0, sd=None):
        self.add(t, "drum", m, 1.0, vel, sd=sd, sound=sound, pan=pan, rr=True)

    def line(self, t0, beat, inst, phrase, vel, legato=0.95, **art):
        """phrase: [(midi or None, beats, extra art dict or None), …] → notes; returns the end time."""
        t, prev = t0, None
        for item in phrase:
            m, b = item[0], item[1]
            extra = item[2] if len(item) > 2 and item[2] else {}
            if m is not None:
                a = dict(art, **extra)
                if prev is not None and a.pop("slur", False):
                    a.update({"from": prev, "glide": a.get("glide", 0.08)})
                a.pop("slur", None)
                self.add(t, inst, m, b * beat * legato, vel * a.pop("acc", 1.0), **a)
                prev = m
            t += b * beat
        return t


# ════════════════════════════════════════════════════════════════════════════
#  mixdown: buses → instrument bodies → room → (sidechain pump) → mono lows → glue → EQ → (filter sweep)
# ════════════════════════════════════════════════════════════════════════════

def sweep_filter(x, points, n_fft=4096, hop=1024):
    """Time-varying low-pass (12 dB/oct) by STFT overlap-add; points = [(t s, cutoff Hz), …], log-interpolated."""
    n = len(x)
    win = np.sqrt(np.hanning(n_fft + 1)[:-1])
    pad = np.concatenate([np.zeros((n_fft, 2)), x, np.zeros((n_fft, 2))])
    out = np.zeros_like(pad)
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    ts, fc = np.array([p[0] for p in points]), np.log(np.array([p[1] for p in points]))
    for s in range(0, len(pad) - n_fft, hop):
        t = (s + n_fft / 2 - n_fft) / SR
        c = math.exp(float(np.interp(t, ts, fc)))
        g = 1 / np.sqrt(1 + (f / c) ** 4)
        seg = pad[s:s + n_fft] * win[:, None]
        out[s:s + n_fft] += np.fft.irfft(np.fft.rfft(seg, axis=0) * g[:, None], n_fft, axis=0) * win[:, None]
    return out[n_fft:n_fft + n] / (n_fft / (2 * hop))        # sqrt-hann² (= hann) at 75% overlap sums to 2


def pump(n, kicks, depth, release=0.17, attack=0.006):
    """Sidechain gain: dips on every kick, recovers over `release`."""
    g = np.ones(n, F32)
    L = int(release * 5 * SR)
    t = np.arange(L) / SR
    shape = (1 - depth * np.exp(-t / release) * np.clip(t / attack, 0, 1) ** 0.5).astype(F32)
    shape[: int(attack * SR)] = np.linspace(1, shape[int(attack * SR)], int(attack * SR))
    for k in kicks:
        i = int(k * SR)
        if i < n:
            e = min(n, i + L)
            g[i:e] = np.minimum(g[i:e], shape[: e - i])
    return g


def mixdown(orch, notes, seconds, ir, *, stage=None, verb=0.5, eq=None, comp=(-20.0, 2.0), ambience=None,
            kicks=None, duck=0.0, sweep=None, target_db=-15.0, fade_out=3.0):
    stage = dict(STAGE, **(stage or {}))
    N = int(seconds * SR)
    rng = np.random.default_rng(99)
    buses, t0 = {}, time.perf_counter()
    for k, note in enumerate(sorted(notes, key=lambda z: z.t)):
        x = orch.render(note)
        pan, gain, _ = stage[note.inst]
        pan = float(np.clip(note.art.get("pan", pan) + rng.normal(0, 0.03), -1, 1))
        gl, gr = pan_gains(pan)
        i = int(round(note.t * SR))
        if i >= N:
            continue
        e = min(N, i + len(x))
        b = buses.setdefault(note.inst, np.zeros((N, 2), F32))
        b[i:e] += (x[: e - i] * (np.array([gl, gr]) * math.sqrt(2) * gain)).astype(F32)
    t_notes = time.perf_counter() - t0
    dry, drums, send = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2))
    fams, stems = {}, {}
    for inst, x in buses.items():
        s = stage[inst][2]
        stems[inst] = float(np.sqrt((x.astype(float) ** 2).mean()))
        fam = BODY_OF.get(inst)
        if fam:
            acc = fams.setdefault(fam, np.zeros((N, 4), F32))
            acc[:, :2] += x
            acc[:, 2:] += s * x
        else:
            (drums if inst in ("drum", "timpani") else dry)[:] += x
            send += s * x
    for fam, acc in fams.items():
        y = fft_conv(acc, orch.bodies[fam])[:N]
        dry += y[:, :2]
        send += y[:, 2:]
    del fams, buses
    wet = np.stack([fft_conv(send[:, c], ir[:, c])[:N] for c in range(2)], 1)
    g = pump(N, kicks, duck)[:, None] if kicks and duck else 1.0
    mix = (dry + verb * wet) * g + drums
    if ambience is not None:
        mix += ambience[:N]
    mid, side = (mix[:, 0] + mix[:, 1]) / 2, (mix[:, 0] - mix[:, 1]) / 2
    side = fft_filter(side, 120, None, 4)                      # lows in mono
    mix = np.stack([mid + side, mid - side], 1)
    th, ratio = comp
    mix *= 10 ** (-18 / 20) / (np.sqrt((mix ** 2).mean()) + 1e-9)   # into the compressor at a known level
    mix = Compressor(thresh_db=th, ratio=ratio, attack=0.02, release=0.3, knee=8)(mix)
    eqp = {**dict(lp=15000, air=1.0, low=1.0, fl=90.0), **(eq or {})}
    h = master_eq(eqp.pop("lp"), **eqp)
    mix = fft_conv(mix, h)[len(h) // 2: len(h) // 2 + N]
    if sweep:
        mix = sweep_filter(mix, sweep)
    mix *= 10 ** (target_db / 20) / (np.sqrt((mix ** 2).mean()) + 1e-9)
    y = np.tanh(mix * 1.05) / 1.05
    n = len(y)
    y *= np.minimum(1, np.minimum(np.arange(n) / (0.02 * SR), (n - np.arange(n)) / (fade_out * SR)))[:, None]
    return y, dict(t_notes=t_notes, stems=stems)


def write_wav(path, y):
    with wave.open(path, "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


# ════════════════════════════════════════════════════════════════════════════
#  1 · prehistoric — "Ochre"
# ════════════════════════════════════════════════════════════════════════════

PENT = [0, 3, 5, 7, 10]                 # minor pentatonic


def pent(deg, base):
    return base + PENT[deg % 5] + 12 * (deg // 5)


def cave_ir(rng):
    """A deep limestone cave: dense early reflections, a long dark tail and two slap-backs off the far wall."""
    ir = make_ir(rng, 5.5, (4.6, 3.4, 1.8, 0.7), 0.03, 30)
    for d, g in ((0.19, 0.5), (0.37, 0.3), (0.71, 0.15)):
        i = int(d * SR)
        L = int(0.12 * SR)
        e = fft_filter(rng.standard_normal((L, 2)), 250, 2500) * np.exp(-np.arange(L) / (0.02 * SR))[:, None]
        ir[i:i + L] += e * g / (np.sqrt((e ** 2).sum(0)) + 1e-9)
    return ir / np.sqrt((ir ** 2).sum(0))


def prehistoric():
    s = Score(101, sd=0.008)
    B = 0.8                                  # beat (s); bar = 4 beats
    bar = lambda k: k * 4 * B
    A2, A4 = 45, 69
    P = lambda d: pent(d, A4)                # flute register

    # ── throat drone: one breath per 2 bars, kargyraa (an octave below) in the trance
    for k in range(2, 30, 2):
        style = "kargyraa" if 14 <= k < 26 else "khoomei"
        hs = [(0, 8), (2.2, 6), (4.0, 8)] if k % 4 else [(0, 6), (2.6, 7), (4.4, 6)]
        v = 0.5 if k < 6 else 0.62 if k < 26 else 0.45
        s.add(bar(k) + 0.1, "voice", A2, 2 * 4 * B - 0.7, v, style=style, harmonics=hs,
              vowels=["o", "a", "o"] if style == "kargyraa" else ["ö", "u"])

    # ── bone flute calls (willow: overtone flute, breathy), opening and closing
    call_a = [(P(3), 2.5, {"scoop": 60}), (P(2), 0.5), (P(1), 0.5, {"slur": True}), (P(0), 3.5, {"vib": 22})]
    call_b = [(P(0), 0.5), (P(1), 0.5), (P(2), 1.0), (P(4), 1.5, {"vib": 18}), (P(3), 0.5),
              (P(2), 1.0, {"slur": True}), (P(0), 4.0, {"vib": 24})]
    s.line(bar(0) + 1.2, B, "willow", call_a, 0.62, pan=-0.3)
    s.line(bar(2) + 0.4, B, "willow", call_b, 0.6, pan=0.35)            # an answer from deeper in the cave
    s.line(bar(27), B, "willow", call_a, 0.55, pan=-0.3)
    s.line(bar(29) + 0.4, B, "willow", [(P(2), 1.0), (P(1), 1.0, {"slur": True}), (P(0), 6.0, {"vib": 20})],
           0.5, pan=0.35)

    # ── ocarina song (clay vessel flute), verse 1 over the heartbeat
    O = lambda d: pent(d, 69)
    song = [(O(0), 1), (O(1), 0.5), (O(2), 0.5), (O(3), 2.0, {"vib": 15}), (O(2), 1), (O(1), 1, {"slur": True}),
            (O(0), 2), (None, 0),
            (O(3), 1), (O(4), 0.5), (O(5), 1.5, {"vib": 18}), (O(4), 0.5), (O(3), 0.5), (O(2), 1),
            (O(3), 3.0, {"vib": 15}), (None, 1),
            (O(5), 1.5), (O(4), 0.5), (O(3), 1), (O(2), 1), (O(1), 1), (O(2), 0.5), (O(1), 0.5, {"slur": True}),
            (O(0), 4.0, {"vib": 20})]
    s.line(bar(6), B, "ocarina", song, 0.6)

    # ── drums
    for k in range(4, 27):
        t = bar(k)
        e = 0.0 if k < 12 else min(1.0, (k - 12) / 10)        # the circle grows
        trance = 22 <= k < 26
        # heartbeat on the shaman drum: DUM-dum
        for q in range(0, 4, 2):
            s.hit(t + q * B, "shaman", 0.85 if q == 0 else 0.7, pan=-0.05)
            s.hit(t + q * B + 0.28, "shaman", 0.45, pan=-0.05)
        # stones knocked together on the off-beats
        for q in range(4):
            s.hit(t + q * B + B / 2, "stick", 0.25 + 0.15 * e, pan=0.45)
        if k >= 8:                                            # rattle shakes, long-short
            for q in range(4):
                s.hit(t + q * B, "rattle", 0.3, pan=-0.5)
                s.hit(t + q * B + B * 2 / 3, "rattle", 0.2, pan=-0.5)
        if k >= 12:                                           # log drum + frame drums
            s.hit(t, "taiko", 0.7 + 0.25 * e)
            s.hit(t + 2.5 * B, "taiko", 0.5 + 0.2 * e)
            for q in (1, 3):
                s.hit(t + q * B, "frame", 0.45 + 0.25 * e, pan=0.25)
                s.hit(t + q * B + B * 2 / 3, "frame_rim", 0.3 + 0.2 * e, pan=0.3)
        if k >= 16:                                           # hooves of stone, 3 over 4
            for q in range(6):
                s.hit(t + q * 4 * B / 6, "hoof", 0.35 + 0.15 * e, pan=0.55)
        if trance:                                            # frame drums in running triplets
            for q in range(12):
                s.hit(t + q * B / 3, "frame", 0.3 + 0.25 * (q % 3 == 0) + 0.1 * (k - 22) / 3, pan=-0.3)
        if k in (11, 15, 21, 25):                             # fills into the next phrase
            for q in range(6):
                s.hit(t + 3 * B + q * B / 6, "taiko_mid", 0.4 + 0.08 * q)
    for k in range(26, 30):                                   # embers: the heartbeat slows and fades
        v = 0.7 * (1 - (k - 26) / 4.5)
        s.hit(bar(k), "shaman", v)
        s.hit(bar(k) + 0.33, "shaman", v * 0.6)
        s.hit(bar(k) + 2 * B + 0.2 * (k - 26), "shaman", v * 0.7)
    s.hit(bar(30), "taiko", 0.6)
    for k in (0, 1, 3, 27, 29):                               # a rattle stirs in the dark
        s.hit(bar(k) + 2.1, "rattle", 0.35, pan=0.6)

    # ── the men's chant: call (tenors) and response (all), then a sung trance on 'ho'
    T3 = pent(0, 57)
    call = [(T3, 1.0, {"vowel": "a"}), (pent(1, 57), 0.5, {"vowel": "e"}), (pent(2, 57), 1.5, {"vowel": "a"}),
            (pent(1, 57), 1.0, {"vowel": "o"})]
    resp_t = [(pent(2, 57), 0.5, {"vowel": "o"}), (pent(1, 57), 0.5, {"vowel": "o"}), (T3, 3.0, {"vowel": "u"})]
    for k in (12, 14, 16, 18):
        t = bar(k)
        s.line(t, B, "choir_t", call, 0.62, players=3, legato=0.9)
        tr = t + 4 * B
        s.line(tr, B, "choir_t", resp_t, 0.7, legato=0.92)
        s.line(tr, B, "choir_b", [(m - 12 if m != T3 else A2, b, a) for m, b, a in resp_t], 0.72, legato=0.92)
    for k in range(20, 26):                                   # trance: short pulsed 'HO - ha' on the beats
        t = bar(k)
        for q in range(4):
            v = 0.55 + 0.25 * (q == 0) + 0.04 * (k - 20)
            m = [A2, A2, pent(1, 45), pent(4, 33)][q] if k % 2 else [A2, pent(2, 45), A2, pent(3, 33)][q]
            s.add(t + q * B, "choir_b", m, B * 0.55, v, vowel="o")
            s.add(t + q * B, "choir_t", m + 12, B * 0.55, v * 0.85, vowel="a")
    # ── throat whistle melody (sygyt) and jaw harp over the trance
    for k in (18, 22):
        hs, tt = [], 0.4
        for h in [8, 7, 6, 8, 12, 14, 12, 8, 7, 8, 6, 8]:
            hs.append((tt, h))
            tt += B * (1.0 if len(hs) % 3 else 1.5)
        s.add(bar(k) + 0.1, "voice", A2, 4 * 4 * B - 0.6, 0.55, style="sygyt", harmonics=hs, vowels=["ö", "u"],
              pan=0.2)
    for k in range(16, 26):
        t = bar(k)
        hits = [q * B / 2 for q in range(8)]
        riff = [8, 6, 8, 9, 10, 8, 6, 7] if k % 2 else [8, 6, 12, 9, 8, 6, 7, 6]
        s.add(t, "jaw", A2, 4 * B, 0.42 if k < 20 else 0.5, plucks=[(h, 1.0 if i % 2 == 0 else 0.6)
                                                                       for i, h in enumerate(hits)],
              mouth=[(h, riff[i]) for i, h in enumerate(hits)], pan=0.35)
    # ── ney: a high wail over the peak
    N4 = lambda d: pent(d, 69)
    s.line(bar(22), B, "ney", [(N4(5), 3, {"scoop": 80, "vib": 30}), (N4(4), 1), (N4(3), 2, {"slur": True}),
                               (N4(4), 1), (N4(5), 1), (N4(7), 4, {"vib": 34, "slur": True}), (N4(5), 2),
                               (N4(4), 1), (N4(3), 1), (N4(2), 2), (N4(0), 4, {"vib": 28, "slur": True})],
           0.6, pan=-0.2)

    seconds = bar(31) + 2.0
    return s, seconds, dict(
        name="prehistoric_ochre", title="Ochre", ir=cave_ir, verb=0.75,
        stage={"willow": (-0.3, 1.0, 0.8), "ocarina": (0.25, 0.75, 0.75), "voice": (0.0, 0.45, 0.5),
               "choir_b": (0.15, 0.5, 0.55), "choir_t": (-0.15, 0.5, 0.6), "ney": (-0.2, 0.6, 0.7),
               "drum": (0.0, 1.6, 0.45), "jaw": (0.35, 0.7, 0.35)},
        eq=dict(lp=9000, low=1.5, fl=85.0), comp=(-21.0, 2.2), ambience=("fire", 1.0, 0.5),
        target_db=-16.0, fade_out=4.0)


# ════════════════════════════════════════════════════════════════════════════
#  2 · medieval — "Lantern Road"
# ════════════════════════════════════════════════════════════════════════════

def medieval():
    s = Score(202, sd=0.006)
    D2, D3, D4 = 38, 50, 62

    # ── I. plainchant at dusk: tenors with basses in organum a fourth below, the organ's open fifth, a bell
    s.hit(0.0, "bell", 0.6, pan=0.4)
    s.hit(4.8, "bell", 0.45, pan=0.4)
    chant = [(D3, 1.3, "a"), (53, 0.7, "e"), (55, 0.7, "e"), (57, 1.5, "a"), (57, 0.7, "i"), (60, 0.8, "o"),
             (57, 0.7, "a"), (55, 1.0, "e"), (57, 2.0, "u"), (None, 0.5, ""),
             (57, 0.7, "a"), (55, 0.7, "e"), (53, 0.7, "i"), (52, 0.7, "a"), (53, 1.0, "o"), (55, 0.8, "a"),
             (52, 0.8, "e"), (D3, 3.0, "u")]
    t = 1.5
    for i, (m, d, v) in enumerate(chant):
        if m is not None:
            s.add(t, "choir_t", m, d * 0.97, 0.6 + 0.08 * (m - D3) / 10, vowel=v)
            org = max(m - 5, 45) if 0 < i < len(chant) - 1 else m - 12     # occursus: settle on the octave
            s.add(t + 0.01, "choir_b", org, d * 0.97, 0.55, vowel=v)
        t += d
    chant_end = t
    s.add(1.0, "organ", D3, chant_end - 0.5, 0.3, swell=0.5)
    s.add(1.4, "organ", D3 + 7, chant_end - 1.0, 0.22, swell=0.5)

    # ── II. estampie, 6/8 (eighth = 0.22 s, bar = 6 eighths)
    E = 0.22
    t0 = chant_end + 0.6
    bt = lambda k: t0 + k * 6 * E
    p1 = [(7, 2), (5, 1), (3, 2), (2, 1), (0, 2), (2, 1), (3, 1), (5, 1), (7, 1), (10, 2), (7, 1), (9, 2), (5, 1)]
    p2 = [(12, 3), (10, 2), (7, 1), (10, 2), (12, 1), (14, 2), (12, 1), (10, 1), (7, 1), (5, 1), (7, 2), (3, 1)]
    ouvert = [(7, 3), (2, 3)]
    clos = [(2, 1), (3, 1), (2, 1), (0, 3)]
    clos2 = [(3, 1), (2, 1), (-2, 1), (0, 3)]
    chords = {0: (D3, 57), 1: (D3, 57), 2: (48, 55), 3: (D3, 57)}       # per bar of a 4-bar phrase: D5 D5 C5 D5

    def melody(k, phrase, ending, inst, vel, octave=0, **art):
        tt = bt(k)
        for i, (o, n8) in enumerate(phrase + ending):
            a = dict(art)
            if inst == "hurdy" and n8 >= 2 and i % 3 == 1:
                a["grace"] = [(2, 0.05)]                                       # a flick from above
            s.add(tt, inst, D4 + o + 12 * octave, n8 * E * (0.92 if n8 < 3 else 0.97), vel * (1.08 if i == 0 else 1),
                  **a)
            tt += n8 * E

    puncta = [(p1, ouvert), (p1, clos), (p2, ouvert), (p2, clos2)]
    plan = [(2, puncta[0], {"hurdy"}), (6, puncta[1], {"hurdy"}),
            (10, puncta[2], {"hurdy", "flute"}), (14, puncta[3], {"hurdy", "flute"}),
            (22, puncta[0], {"hurdy", "flute", "choir"}), (26, puncta[1], {"hurdy", "flute", "choir"}),
            (30, puncta[2], {"hurdy", "flute", "choir", "lyre"}), (34, puncta[3], {"hurdy", "flute", "choir", "lyre"})]
    for k, (ph, end), who in plan:
        melody(k, ph, end, "hurdy", 0.7, part="melody")
        if "flute" in who:
            melody(k, ph, end, "flute", 0.55, octave=1, pan=0.35)
        if "choir" in who:                                                 # the townsfolk sing along on 'a'
            melody(k, ph, end, "choir_t", 0.3, octave=-1, vowel="a")
        if "lyre" in who:
            melody(k, ph, end, "lyre", 0.5, octave=0)
    # interlude (bars 18–21): lyre and recorder trade a little round over the drone
    rnd = [(D4 + 7, 2), (D4 + 9, 1), (D4 + 10, 2), (D4 + 9, 1), (D4 + 7, 3), (D4 + 5, 3),
           (D4 + 3, 2), (D4 + 5, 1), (D4 + 7, 2), (D4 + 5, 1), (D4 + 2, 3), (D4, 3)]
    tt = bt(18)
    for m, n8 in rnd:
        s.add(tt, "lyre", m - 12, n8 * E, 0.6)
        if tt + 6 * E < bt(22) - 0.1:
            s.add(tt + 6 * E, "flute", m + 12, n8 * E * 0.95, 0.45, pan=0.35)
        tt += n8 * E
    END = 38
    # drones: hurdy-gurdy bourdon (D3 + A3) and a low cello bourdon, re-bowed every 2 bars
    for k in range(0, END, 2):
        s.add(bt(k), "hurdy", D3, 12 * E - 0.05, 0.42, part="drone", swell=0.2)
        s.add(bt(k) + 0.02, "hurdy", D3 + 7, 12 * E - 0.05, 0.3, part="drone", swell=0.2)
        if k >= 2:
            s.add(bt(k), "cellos", D2 + 12 if k < 10 else D2, 12 * E - 0.05, 0.45, tasto=True, players=2)
    # the trompette: the buzzing bridge on the wheel's strokes (dotted: strong on 1 and 4)
    for k in range(2, END):
        strokes = [(0, 1.0), (2 * E, 0.4), (3 * E, 0.85), (5 * E, 0.4)]
        if k in (18, 19, 20, 21):
            strokes = [(0, 0.7), (3 * E, 0.6)]
        s.add(bt(k), "buzz", D3, 6 * E, 0.55 if k < 22 else 0.65, strokes=strokes)
    # lyre and harp: open fifths on the downbeats, strummed on the 4
    for k in range(2, END):
        lo, hi = chords[(k - 2) % 4]
        s.add(bt(k), "harp", lo - 12, 6 * E, 0.55, rr=True)
        for j, m in enumerate((lo, hi, lo + 12)):
            s.add(bt(k) + 3 * E + 0.025 * j, "lyre", m, 3 * E, 0.4, rr=True)
    # tabor (frame drum): DUM . da DUM . da, a bigger drum from the second dance on
    for k in range(0, END):
        t = bt(k)
        v = 0.55 if k < 10 else 0.7
        s.hit(t, "frame", v, pan=-0.1)
        s.hit(t + 2 * E, "frame_rim", v * 0.55, pan=0.15)
        s.hit(t + 3 * E, "frame", v * 0.8, pan=-0.1)
        s.hit(t + 5 * E, "frame_rim", v * 0.5, pan=0.15)
        if k >= 22:
            s.hit(t, "doum", 0.55, pan=0.0)
            for q in (1, 4):
                s.hit(t + q * E, "rattle", 0.22, pan=0.5)
        if k in (17, 21, 29, 37):
            for q in range(6):
                s.hit(t + q * E, "frame", 0.35 + 0.08 * q, pan=-0.2)
    # ── III. the last chord, a bell, and Amen
    tf = bt(END)
    s.add(tf, "hurdy", D4, 3.5, 0.7, part="melody")
    s.add(tf, "hurdy", D3, 4.5, 0.45, part="drone")
    s.add(tf, "hurdy", D3 + 7, 4.5, 0.32, part="drone")
    s.add(tf, "flute", D4 + 12, 3.0, 0.5, pan=0.35)
    s.add(tf, "buzz", D3, 1.0, 0.6, strokes=[(0, 1.0)])
    s.add(tf, "harp", D2, 3.0, 0.7)
    for j, m in enumerate((D3, 57, D4, 69)):
        s.add(tf + 0.04 * j, "lyre", m, 3.0, 0.55)
    s.hit(tf, "frame", 0.8)
    s.hit(tf, "doum", 0.7)
    s.hit(tf + 0.2, "bell", 0.55, pan=0.4)
    ta = tf + 3.8
    s.add(ta, "choir_t", 55, 1.3, 0.55, vowel="a")
    s.add(ta, "choir_b", 43, 1.3, 0.55, vowel="a")
    s.add(ta + 1.35, "choir_t", 57, 4.0, 0.55, vowels=["e", "e", "m"])
    s.add(ta + 1.35, "choir_b", D3, 4.0, 0.55, vowels=["e", "e", "m"])
    s.add(ta + 1.35, "choir_b", D2, 4.0, 0.4, vowels=["e", "e", "m"])
    s.add(ta + 1.0, "organ", D3, 5.0, 0.22, swell=0.6)
    s.hit(ta + 1.35, "bell", 0.4, pan=0.4)
    seconds = ta + 8.5
    return s, seconds, dict(
        name="medieval_lantern_road", title="Lantern Road", ir=lambda r: make_ir(r, 4.8, (3.8, 3.2, 2.2, 1.1), 0.03, 16),
        verb=0.6,
        stage={"hurdy": (0.0, 0.55, 0.35), "buzz": (0.1, 1.6, 0.25), "flute": (0.35, 0.95, 0.45),
               "lyre": (-0.35, 0.8, 0.35), "harp": (-0.5, 1.0, 0.35), "choir_t": (-0.15, 1.0, 0.6),
               "choir_b": (0.15, 1.0, 0.55), "organ": (0.0, 0.8, 0.7), "cellos": (0.4, 1.1, 0.3),
               "drum": (0.0, 1.2, 0.3)},
        eq=dict(lp=13000, air=1.3, low=1.2, fl=90.0), comp=(-20.0, 2.0), target_db=-15.5, fade_out=3.0)


# ════════════════════════════════════════════════════════════════════════════
#  3 · modern — "Glass Skyline"
# ════════════════════════════════════════════════════════════════════════════

def modern():
    s = Score(303, sd=0.002)
    Q = 0.5                                           # beat at 120 bpm; bar = 2 s
    bar = lambda k: k * 4 * Q
    # i – VI – III – VII in E minor: roots, pad voicings (voice-led), arpeggio tones
    ROOT = [40, 36, 43, 38]                           # E2 C2 G2 D2
    PAD = [[52, 59, 64, 67], [52, 55, 60, 67], [50, 55, 62, 67], [50, 57, 62, 66]]
    ARP = [[52, 59, 64, 66, 67, 64, 59, 66],          # Em(add9)
           [48, 55, 64, 62, 60, 64, 55, 62],          # Cadd9
           [43, 50, 59, 57, 55, 59, 50, 57],          # G6
           [50, 57, 64, 62, 66, 62, 57, 64]]          # Dsus2→D
    INTRO, VERSE, CHORUS, BREAK, DROP, OUTRO = 0, 4, 12, 20, 24, 36
    END = 44
    kicks = []

    for k in range(END):
        c = k % 4
        t = bar(k)
        sec = ("intro" if k < VERSE else "verse" if k < CHORUS else "chorus" if k < BREAK else "break" if k < DROP
               else "drop" if k < OUTRO else "outro")
        # harp ostinato in eighths (all the way through; it is the piece's clock)
        hv = 0.5 if sec in ("intro", "break", "outro") else 0.4
        for q, m in enumerate(ARP[c]):
            if sec == "outro" and k >= END - 2 and q > 3:
                break
            s.add(t + q * Q / 2, "harp", m + 12, Q / 2 * 1.5, hv * (1.15 if q % 4 == 0 else 0.9),
                  rr=True, bright=1.2)
        # low end: octobass sustained root, cello spiccato octaves in eighths (the pulse), basses doubling
        if sec in ("verse", "chorus", "drop"):
            s.add(t, "octobass", ROOT[c] - 12 if ROOT[c] - 12 >= 21 else ROOT[c], 4 * Q - 0.05, 0.55)
            for q in range(8):
                m = ROOT[c] + (12 if q % 2 else 0)
                s.add(t + q * Q / 2, "cellos", m, Q / 2 * 0.8, 0.55 if q % 2 == 0 else 0.42, art="spic", rr=True)
        if sec in ("chorus", "drop"):
            s.add(t, "basses", ROOT[c], 4 * Q - 0.05, 0.5)
        # violas: syncopated spiccato on the pad's inner voices (3+3+2)
        if sec in ("verse", "chorus", "drop"):
            for q in (0, 3, 6):
                for m in PAD[c][1:3]:
                    s.add(t + q * Q / 2, "violas", m, Q * 0.45, 0.45, art="spic", rr=True)
        # violins 16th spiccato in the drop
        if sec == "drop":
            for q in range(16):
                m = PAD[c][3] + 12 if q % 4 in (0, 3) else PAD[c][2] + 12
                s.add(t + q * Q / 4, "violins", m, Q / 4 * 0.8, 0.4 + 0.12 * (q % 4 == 0), art="spic", rr=True)
        # pads: horns in the choruses, choir 'aa' + sustained strings in the drop and break
        if sec in ("chorus", "drop"):
            for m in PAD[c]:
                s.add(t, "horns", m - 12 if m > 60 else m, 4 * Q - 0.1, 0.42 if sec == "chorus" else 0.5, swell=0.5)
        if sec in ("break", "drop", "outro"):
            for m, inst in zip(PAD[c][1:], ("choir_a", "choir_s", "choir_s")):
                s.add(t, inst, m + 12 if inst == "choir_s" else m + 12 if m < 55 else m,
                      4 * Q - 0.1, 0.4 if sec != "drop" else 0.48, vowel="a" if sec == "drop" else "u")
            if sec != "drop":
                for m in PAD[c]:
                    s.add(t, "violins" if m >= 55 else "violas", m + 12 if m >= 55 else m, 4 * Q - 0.05, 0.32,
                          tasto=True)
        # the beat
        if sec in ("verse", "chorus", "drop"):
            big = sec == "drop"
            for q in ((0, 3, 4) if sec != "verse" or k % 2 == 0 else (0, 4, 7)):
                s.hit(t + q * Q / 2, "taiko", 0.85 if q == 0 else 0.7, sd=0.0)
                s.hit(t + q * Q / 2, "odaiko", 0.5 + 0.2 * big, sd=0.0)
                kicks.append(t + q * Q / 2)
            for q in (2, 6):                                         # backbeat: rim + stick + tek, a "clap"
                s.hit(t + q * Q / 2, "taiko_rim", 0.6, pan=0.05, sd=0.0)
                s.hit(t + q * Q / 2 + 0.004, "stick", 0.45, pan=-0.1, sd=0.0)
                s.hit(t + q * Q / 2 + 0.009, "tek", 0.5, pan=0.15, sd=0.0)
            for q in range(16 if sec != "verse" else 8):              # shaker
                step = Q / 4 if sec != "verse" else Q / 2
                s.hit(t + q * step, "rattle", (0.28 if q % 2 else 0.18) + 0.05 * big, pan=0.45)
            if big:
                for q in (1, 5):
                    s.hit(t + q * Q / 2 + Q / 4, "hyoshigi", 0.22, pan=-0.45)
            if c == 3:                                                # fill into the next phrase
                for q in range(4):
                    s.hit(t + 3 * Q + q * Q / 4, "taiko_mid", 0.45 + 0.1 * q)
        if sec == "break" and k >= BREAK + 2:                          # half-time heartbeat creeping back
            s.hit(t, "taiko", 0.5)
            kicks.append(t)
    # big moments: tam-tam + timpani on the drop, riser and timpani roll into it, tam-tam at the chorus
    s.add(bar(DROP - 2), "drum", 0, bar(2), 0.9, sound="swell")
    s.add(bar(DROP) - 1.6, "timpani", 40, 1.6, 0.7, roll=1.5)
    for k in (CHORUS, DROP, DROP + 8):
        s.hit(bar(k), "tamtam", 0.55)
        s.add(bar(k), "timpani", 40, 3.0, 0.8)
    # ── lead melodies
    verse = [[(59, 1.5), (64, 0.5), (66, 1), (67, 1)], [(67, 1.5), (66, 0.5), (64, 2)],
             [(62, 1), (64, 0.5), (62, 0.5), (59, 2)], [(57, 1), (59, 1), (62, 1), (66, 1)],
             [(59, 1.5), (64, 0.5), (66, 1), (67, 1)], [(71, 1.5), (69, 0.5), (67, 1), (66, 1)],
             [(67, 1), (66, 0.5), (64, 0.5), (62, 2)], [(57, 1.5), (59, 2.5)]]
    chorus = [[(76, 1), (74, 0.5), (76, 0.5), (79, 2)], [(79, 0.5), (78, 0.5), (76, 1), (74, 1), (76, 1)],
              [(74, 1.5), (71, 0.5), (74, 1), (79, 1)], [(78, 3), (81, 1)],
              [(76, 1), (74, 0.5), (76, 0.5), (79, 2)], [(79, 0.5), (81, 0.5), (83, 1), (81, 1), (79, 1)],
              [(78, 1.5), (76, 0.5), (74, 1), (71, 1)], [(78, 2), (76, 2)]]

    def lead(k0, bars_, inst, vel, shift=0, **art):
        for i, ph in enumerate(bars_):
            s.line(bar(k0 + i), Q, inst, [(m + shift, b, {"slur": j > 0 and b <= 0.5}) for j, (m, b) in enumerate(ph)],
                   vel, legato=0.97, sd=0.004, **art)

    lead(VERSE, verse, "cellos", 0.72, players=1, vib=18)
    lead(CHORUS, chorus, "violins", 0.6, vib=14)
    lead(CHORUS, chorus, "flute", 0.38, shift=-12)
    lead(BREAK, verse[4:], "khuur", 0.6, shift=12, vib=24)
    lead(DROP, chorus, "violins", 0.66, vib=14)
    lead(DROP, chorus, "cellos", 0.6, shift=-24, players=4)
    lead(DROP + 8, chorus[:4], "violins", 0.7, vib=16)
    lead(DROP + 8, chorus[:4], "khuur", 0.55, shift=-12, vib=20)
    lead(OUTRO, verse[:4], "cellos", 0.55, players=1, vib=16, tasto=True)
    # last chord
    tf = bar(END)
    s.add(tf, "harp", 40, 4.0, 0.6)
    for j, m in enumerate((52, 59, 66, 67, 71)):
        s.add(tf + 0.05 * j, "harp", m + 12, 4.0, 0.5)
    for m in (52, 59, 66, 67):
        s.add(tf, "violins" if m >= 55 else "violas", m + 12 if m >= 55 else m, 4.5, 0.3, tasto=True, swell=0.3)
    s.add(tf, "octobass", 28, 4.5, 0.45)
    seconds = tf + 6.0
    sweep = [(0.0, 700.0), (bar(VERSE) - 0.1, 2500.0), (bar(VERSE), 18000.0), (bar(BREAK), 18000.0),
             (bar(BREAK) + 0.5, 1400.0), (bar(DROP) - 2.0, 2500.0), (bar(DROP) - 0.05, 9000.0), (bar(DROP), 18000.0),
             (bar(OUTRO + 4), 18000.0), (seconds, 900.0)]
    return s, seconds, dict(
        name="modern_glass_skyline", title="Glass Skyline", ir=lambda r: make_ir(r, 3.4, (2.6, 2.1, 1.5, 0.8), 0.02, 12),
        verb=0.42, kicks=kicks, duck=0.35, sweep=sweep,
        stage={"harp": (-0.35, 0.6, 0.35), "cellos": (0.3, 1.0, 0.3), "violas": (0.2, 0.55, 0.35),
               "violins": (-0.35, 0.75, 0.4), "horns": (-0.2, 0.55, 0.5), "khuur": (0.0, 1.0, 0.4),
               "flute": (0.25, 0.7, 0.45), "octobass": (0.0, 1.3, 0.05), "basses": (0.3, 0.45, 0.15),
               "choir_s": (-0.3, 0.4, 0.55), "choir_a": (0.3, 0.4, 0.55), "drum": (0.0, 2.3, 0.15),
               "timpani": (0.0, 1.0, 0.4)},
        eq=dict(lp=17000, air=2.5, low=1.4, fl=80.0), comp=(-19.0, 2.5), target_db=-13.5, fade_out=3.0)


THEMES = {"prehistoric": prehistoric, "medieval": medieval, "modern": modern}


def render(key):
    t0 = time.perf_counter()
    score, seconds, cfg = THEMES[key]()
    orch = Orchestra(np.random.default_rng(7))
    ir = cfg["ir"](np.random.default_rng(8))
    amb = None
    if cfg.get("ambience"):
        _, fire, wind = cfg["ambience"]
        mx = Mixer(orch)
        N = int(seconds * SR)
        amb = mx.fire(N, fire) * 0.8 + mx.wind(N, wind)
    y, st = mixdown(orch, score.notes, seconds, ir, stage=cfg.get("stage"), verb=cfg["verb"], eq=cfg.get("eq"),
                    comp=cfg["comp"], ambience=amb, kicks=cfg.get("kicks"), duck=cfg.get("duck", 0.0),
                    sweep=cfg.get("sweep"), target_db=cfg["target_db"], fade_out=cfg["fade_out"])
    path = os.path.join(OUT, cfg["name"] + ".wav")
    write_wav(path, y)
    el = time.perf_counter() - t0
    rms = 20 * np.log10(np.sqrt((y ** 2).mean()))
    pk = 20 * np.log10(np.abs(y).max())
    print(f"{path}: \"{cfg['title']}\" {len(y) / SR:.1f} s, {len(score.notes)} notes, RMS {rms:.1f} dBFS, "
          f"peak {pk:.1f} dBFS, render {el:.0f} s (notes {st['t_notes']:.0f} s)")
    tot = sum(v ** 2 for v in st["stems"].values())
    print("   stems (energy share before bodies/room): " + ", ".join(
        f"{k} {100 * v ** 2 / tot:.0f}%" for k, v in sorted(st["stems"].items(), key=lambda kv: -kv[1]) if v ** 2 / tot > 0.01))
    return path, y


if __name__ == "__main__":
    for key in (sys.argv[1:] or list(THEMES)):
        render(key)
