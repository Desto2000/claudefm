"""Make a vertical (1080×1920) Instagram story: the GPU campfire scene at night in falling snow, framed by live
data from the engine that is making the music — the arrangement, every instrument's level, the score as a
piano roll (past and upcoming notes), a spectrum, and measured costs (synthesis per bar, GPU stages).

    uv run python tools/story.py --out ig_story.mp4 --poster ig_story_cover.png
    uv run python tools/story.py --poster story_poster.png            # a still only

Everything shown is real: levels are the instrument buses' RMS, notes come from the score, timings are
measured while this video renders. Music: a "by the fire" track (seed 4) from its warm middle section."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warp as wp  # noqa: E402

from claudefm import gpu_scenes as g  # noqa: E402
from claudefm import scenes as cf  # noqa: E402
from claudefm.app import make_renderer  # noqa: E402
from kor.audio import Timeline  # noqa: E402
from kor.music.core import SR, NOTE_NAMES, harmonics_for, mtof  # noqa: E402
from kor.music.mix import Composer, Mixer  # noqa: E402
W, H = 1080, 1920
FPS = 30
SCENE_Y, SCENE_H = 708, 640
BG = (9, 9, 12)
LINE = (38, 38, 46)
DIM, TEXT, ACCENT = (118, 116, 128), (226, 221, 210), (217, 119, 87)
ROLE = {"lead": (226, 132, 88), "second": (214, 170, 110), "ember": (236, 200, 120), "pad": (128, 140, 172),
        "voice": (150, 128, 170), "bass": (176, 96, 64), "sub": (130, 62, 46), "drum": (110, 110, 118)}


def font(size):
    for n in ("CascadiaMono.ttf", "consola.ttf"):
        p = os.path.join("C:/Windows/Fonts", n)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def role_of(k, lead, second, ember):
    if k == lead:
        return "lead"
    if k == second:
        return "second"
    if k == ember:
        return "ember"
    if k in ("octobass",):
        return "sub"
    if k in ("basses", "cbcl", "serpent", "tuba"):
        return "bass"
    if k in ("choir_b", "voice"):
        return "voice"
    if k == "drum":
        return "drum"
    return "pad"


def render_music(seconds):
    """Render the track through the warm section, capturing per-bar costs and every bus's level."""
    captured = {}
    orig = Mixer.process

    def process(self, bus, n, tr, lp):
        captured["bus"] = {k: np.sqrt((x ** 2).mean(1)) for k, x in bus.items()}
        return orig(self, bus, n, tr, lp)

    Mixer.process = process
    comp = Composer(np.random.default_rng(4), style="hearth")
    tr = comp.tr
    start = next(s for s in tr.sections if s.label == "warmth").start * tr.bar_len
    end = start + seconds + 1
    nf = int(end * FPS) + 2
    levels, bars, chunks = {}, [], []
    timeline = Timeline()
    while comp.pos < end * SR:
        p0 = comp.pos
        t0 = time.perf_counter()
        y, info = comp.next_bar()
        cost = time.perf_counter() - t0
        timeline.add(info)
        chunks.append(y)
        bars.append((p0 / SR, len(y) / SR, cost, info))
        for k, e in captured["bus"].items():                     # RMS per video frame
            arr = levels.setdefault(k, np.zeros(nf))
            f0 = int(p0 / SR * FPS)
            hop = SR // FPS
            for j in range(len(e) // hop):
                if f0 + j < nf:
                    arr[f0 + j] = max(arr[f0 + j], float(np.sqrt((e[j * hop:(j + 1) * hop] ** 2).mean())))
    Mixer.process = orig
    audio = np.concatenate(chunks).astype(np.float64)
    return comp, tr, start, audio, levels, bars, timeline


class Story:
    def __init__(self, comp, tr, start, audio, levels, bars, dur):
        self.comp, self.tr, self.start, self.audio, self.levels, self.bars, self.dur = \
            comp, tr, start, audio, levels, bars, dur
        lead = tr.lead
        second = {"hearth": "cellos", "snowfall": "willow", "soul": "cellos"}.get(tr.style)
        ember = {"hearth": "lyre", "snowfall": "harp", "soul": "lyre"}.get(tr.style)
        used = {e.inst for e in tr.events}
        order = [lead, second, ember, "choir_b", "voice", "cellos", "basses", "cbcl", "serpent", "tuba", "octobass",
                 "drum"]
        self.rows = [k for k in dict.fromkeys(order) if k in used and k in levels][:9]
        self.roles = {k: role_of(k, lead, second, ember) for k in used}
        short = {"drum": "percussion", "cbcl": "c.bass clarinet", "serpent": "c.bass serpent", "choir_b": "low voices",
                 "voice": "throat singer", "octobass": "octobass"}
        self.names = {k: short.get(k, comp.orch.inst[k].name if k in comp.orch.inst else k) for k in used}
        ms = [e.m for e in tr.events if e.inst in comp.orch.inst]
        self.lo_m, self.hi_m = min(ms) - 2, max(ms) + 2
        self.f_big, self.f, self.f_data, self.f_lab = font(46), font(28), font(26), font(24)
        self.peak = {k: -60.0 for k in self.rows}
        self.spec = np.full(96, -90.0)
        self.edges = np.geomspace(30, 16000, 97)
        self.bg = self.background()

    # layout (px). Instagram covers roughly the top ~200 px and the bottom ~220 px with its own UI.
    NOW_Y, INFO_Y, ARR_Y, ARR_H = 200, 266, 318, 40
    ROWS_Y, ROW_H = 392, 34
    SPEC_Y0, SPEC_Y1 = 1360, 1460
    ROLL_Y0, ROLL_Y1 = 1484, 1700
    SPLIT = 548

    def background(self):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for y in (self.ROWS_Y - 14, SCENE_Y - 6, self.SPEC_Y1 + 10):
            d.line([(40, y), (W - 40, y)], fill=LINE)
        d.line([(self.SPLIT, self.ROWS_Y), (self.SPLIT, self.ROWS_Y + 9 * self.ROW_H - 8)], fill=LINE)
        return img

    def active(self, T):
        return [e for e in self.tr.events if e.t <= T < e.t + e.dur + 0.25]

    def frame(self, scene_img, t, gpu):
        tr, T = self.tr, self.start + t
        img = self.bg.copy()
        img.paste(scene_img, (0, SCENE_Y))
        d = ImageDraw.Draw(img, "RGBA")
        bar = min(int(T // tr.bar_len), tr.nbars - 1)
        beat = (T - bar * tr.bar_len) / (tr.bar_len / len(tr.groups))

        # now playing + music state
        y = self.NOW_Y
        d.polygon([(42, y + 12), (42, y + 44), (68, y + 28)], fill=ACCENT)
        d.text((86, y), f"{tr.name} · {self.names[tr.lead]}", font=self.f_big, fill=TEXT)
        deg = tr.chords[bar]
        root = tr.scale.pitch(deg)
        third = [(tr.scale.pitch(c) - root) % 12 for c in tr.scale.chord(deg)[1:]]
        third = [x for x in third if 2.5 < x < 4.5]
        chord = NOTE_NAMES[round(root) % 12] + (("m" if third[0] < 3.5 else "") if third else "5")
        mode = {"minor pentatonic": "minor pent.", "suspended pentatonic": "sus pent."}.get(tr.mode, tr.mode)
        bpm = round(60 * len(tr.groups) / tr.bar_len)
        info = f"{NOTE_NAMES[round(tr.scale.tonic) % 12]} {mode}  ·  {tr.meter}  ·  {bpm} bpm  ·  " \
               f"bar {bar + 1}.{int(beat) + 1}  ·  {chord}"
        d.text((42, self.INFO_Y), info, font=self.f, fill=DIM)

        # arrangement strip with playhead
        x0, x1, y0, hh = 40, W - 40, self.ARR_Y, self.ARR_H
        total = tr.nbars
        for s in tr.sections:
            a, b = x0 + (x1 - x0) * s.start / total, x0 + (x1 - x0) * s.end / total
            now = s.start <= bar < s.end
            d.rectangle([a + 2, y0, b - 2, y0 + hh], fill=(46, 36, 32) if now else (24, 24, 30))
            if b - a > d.textlength(s.label, font=self.f_lab) + 14:
                d.text((a + 10, y0 + 7), s.label, font=self.f_lab, fill=TEXT if now else DIM)
        px = x0 + (x1 - x0) * (T / tr.bar_len) / total
        d.line([(px, y0 - 6), (px, y0 + hh + 6)], fill=ACCENT, width=3)

        # instrument meters
        fi = min(int(T * FPS), len(next(iter(self.levels.values()))) - 1)
        act = self.active(T)
        y = self.ROWS_Y
        mx0, mx1 = 282, self.SPLIT - 24
        for k in self.rows:
            lv = self.levels[k][max(0, fi - 1):fi + 2].max()
            db = 20 * math.log10(lv + 1e-6)
            self.peak[k] = max(db, self.peak[k] - 0.6)
            col = ROLE[self.roles[k]]
            d.text((40, y), self.names[k], font=self.f_data, fill=DIM)
            d.rectangle([mx0, y + 9, mx1, y + 23], fill=(24, 24, 30))
            fill = mx0 + (mx1 - mx0) * np.clip((db + 54) / 54, 0, 1)
            d.rectangle([mx0, y + 9, fill, y + 23], fill=col)
            pk = mx0 + (mx1 - mx0) * np.clip((self.peak[k] + 54) / 54, 0, 1)
            d.line([(pk, y + 5), (pk, y + 27)], fill=TEXT, width=2)
            y += self.ROW_H

        # engine metrics (measured)
        cur = next((b for b in self.bars if b[0] <= T < b[0] + b[1]), self.bars[-1])
        players = sum(e.art.get("players", getattr(self.comp.orch.inst.get(e.inst), "players", 1))
                      for e in act if e.inst in self.comp.orch.inst)
        partials = sum(harmonics_for(float(mtof(e.m))) * e.art.get("players", self.comp.orch.inst[e.inst].players)
                       for e in act if e.inst in self.comp.orch.inst)
        tonal = sum(1 for e in act if e.inst in self.comp.orch.inst)
        rows = [("synthesis", f"{100 * cur[2] / cur[1]:.0f}% real time"),
                ("this bar", f"{cur[2]:.2f} s / {cur[1]:.2f} s"),
                ("notes", f"{len(act)}"),
                ("players", f"{players}"),
                ("partials", f"{partials:,}"),
                ("tables/s", f"{tonal * 100:,}"),
                ("gpu scene", f"{gpu.get('scene', 0):.2f} ms"),
                ("gpu cells", f"{gpu.get('cells', 0):.2f} ms"),
                ("gpu raster", f"{gpu.get('raster', 0):.2f} ms")]
        y = self.ROWS_Y
        for label, val in rows:
            d.text((self.SPLIT + 24, y), label, font=self.f_data, fill=DIM)
            d.text((W - 40 - d.textlength(val, font=self.f_data), y), val, font=self.f_data, fill=TEXT)
            y += self.ROW_H

        # spectrum
        s0 = int(T * SR)
        seg = self.audio[max(0, s0 - 4096):s0].mean(1)
        if len(seg) == 4096:
            X = np.abs(np.fft.rfft(seg * np.hanning(4096))) / 1024
            fr = np.fft.rfftfreq(4096, 1 / SR)
            idx = np.searchsorted(fr, self.edges)
            val = np.array([X[idx[i]:max(idx[i] + 1, idx[i + 1])].max() for i in range(96)])
            dbs = 20 * np.log10(val + 1e-9)
            self.spec = np.where(dbs > self.spec, dbs, self.spec - 1.2)
        sy0, sy1 = self.SPEC_Y0, self.SPEC_Y1
        bw = (W - 80) / 96
        for i, v in enumerate(self.spec):
            h = (sy1 - sy0) * np.clip((v + 80) / 70, 0, 1)
            c = tuple(int(a + (b - a) * i / 96) for a, b in zip((176, 96, 64), (236, 200, 120)))
            d.rectangle([40 + i * bw + 1, sy1 - h, 40 + (i + 1) * bw - 2, sy1], fill=c)

        # the score as a piano roll: 6 s back, 6 s ahead
        ry0, ry1, span = self.ROLL_Y0, self.ROLL_Y1, 6.0
        cx = W // 2
        ppm = (ry1 - ry0) / (self.hi_m - self.lo_m)
        for e in self.tr.events:
            if e.inst not in self.comp.orch.inst or e.t > T + span or e.t + e.dur < T - span:
                continue
            xa = cx + (e.t - T) / span * (W / 2 - 40)
            xb = cx + (e.t + e.dur - T) / span * (W / 2 - 40)
            yy = ry1 - (e.m - self.lo_m) * ppm
            role = self.roles.get(e.inst, "pad")
            col = ROLE[role]
            sounding = e.t <= T < e.t + e.dur
            alpha = 255 if sounding else (150 if e.t > T else 70)
            hh = 6 if role in ("lead", "second") else (4 if role == "ember" else 2)
            xb = max(xa + 3, xb - 4)                                   # a gap so repeated notes read apart
            if xb >= 40 and xa <= W - 40:
                d.rectangle([max(40, xa), yy - hh, min(W - 40, xb), yy + hh], fill=col + (alpha,))
        d.line([(cx, ry0 - 6), (cx, ry1 + 6)], fill=ACCENT + (220,), width=3)

        out = img
        pic = float(np.clip(min(t / 0.6, (self.dur - t) / 0.8), 0, 1))           # fade from / to black
        if pic < 1:
            out = Image.fromarray((np.asarray(out, np.float32) * pic).astype(np.uint8))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poster")
    ap.add_argument("--out")
    ap.add_argument("--ffmpeg")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    comp, tr, start, audio_full, levels, bars, timeline = render_music(args.seconds)
    story = Story(comp, tr, start, audio_full, levels, bars, args.seconds)

    wp.config.quiet = True
    wp.init()
    dev = wp.get_preferred_device()
    r = make_renderer((W, SCENE_H), (8, 16), dev, seed=7, hud=False, prefetch=False)
    rng = np.random.default_rng(11)
    while True:
        cpu = cf.Campfire(rng)
        if cpu.tod == "night" and cpu.wkind == "snow":
            break
    cpu.frame_pines = False
    r.show.scene = g.GPU_OF[cf.Campfire](cpu, dev)
    pix = wp.zeros(W * SCENE_H, dtype=wp.uint32, device=dev)

    def frame(i):
        t = i / FPS
        r.show.switch_at = 1e9
        c = r.show.context(t, 1 / FPS, r.SW, r.SH, int((start + t) * SR), timeline, 1.0, 2, 4)
        r.timings = {}
        r.frame(c, pix, timed=True)
        gpu = {k: 1000 * v[-1] for k, v in r.timings.items()}
        scene = Image.fromarray(np.ascontiguousarray(pix.numpy().view(np.uint8).reshape(SCENE_H, W, 4)[::-1, :, :3]))
        return story.frame(scene, t, gpu)

    if args.poster:
        for i in range(int(2 * FPS)):
            frame(i)
        frame(int(8 * FPS)).save(args.poster)
        print("poster", args.poster)
    if args.out:
        if not args.ffmpeg:
            import imageio_ffmpeg                                                 # dev dependency
            args.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        audio = audio_full[int(start * SR): int((start + args.seconds) * SR)].copy()
        n = len(audio)
        audio *= np.minimum(1, np.minimum(np.arange(n) / (1.0 * SR), (n - np.arange(n)) / (2.0 * SR)))[:, None]
        wav = args.out + ".wav"
        with wave.open(wav, "wb") as w:
            w.setnchannels(2), w.setsampwidth(2), w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        cmd = [args.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "18",
               "-pix_fmt", "yuv420p", "-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
               "-shortest", "-movflags", "+faststart", args.out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        story.peak = {k: -60.0 for k in story.rows}
        story.spec = np.full(96, -90.0)
        for i in range(int(args.seconds * FPS)):
            p.stdin.write(np.asarray(frame(i)).tobytes())
        p.stdin.close()
        p.wait()
        os.remove(wav)
        print("video", args.out, "exit", p.returncode)


if __name__ == "__main__":
    main()
