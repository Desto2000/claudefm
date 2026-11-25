"""claude fm's landscapes, set up on the host: a time of day, the land, static glyph/colour layers, and the
small simulations the kernels draw each frame (particles, the boat, the battlefield). Drawing happens in
gpu_scenes.py."""

from __future__ import annotations

import math

import numpy as np

from kor.glyphs import SPACE, codes
from kor.scene import Scene
from kor.util import F32, TAU, fbm1d, rgb, stops, value_noise2


# Each time of day sets the whole mood: sky gradient, sun or moon, the light
# falling on the land, cloud colour, stars, fog and how water looks.
TOD = {
    "dawn":   dict(sky=[(0, (46, 64, 124)), (0.5, (150, 140, 198)), (0.82, (248, 180, 186)), (1, (255, 222, 196))],
                   body="sun", body_col=(255, 240, 214), glow=(255, 206, 190), sun_h=0.10, light=(255, 214, 196),
                   lum=0.85, cloud=(255, 214, 222), stars=0.0, fog=(222, 200, 212), water=(118, 120, 170),
                   deep=(30, 38, 74), refl=(255, 234, 220), birds=True),
    "day":    dict(sky=[(0, (52, 108, 196)), (0.6, (118, 170, 228)), (1, (206, 228, 246))],
                   body="sun", body_col=(255, 252, 232), glow=(255, 248, 220), sun_h=0.75, light=(255, 250, 236),
                   lum=1.0, cloud=(250, 250, 255), stars=0.0, fog=(214, 228, 242), water=(70, 130, 178),
                   deep=(18, 52, 96), refl=(255, 255, 240), birds=True),
    "golden": dict(sky=[(0, (64, 92, 164)), (0.55, (172, 158, 172)), (0.85, (250, 198, 132)), (1, (255, 216, 152))],
                   body="sun", body_col=(255, 236, 172), glow=(255, 190, 110), sun_h=0.28, light=(255, 208, 142),
                   lum=0.95, cloud=(255, 226, 192), stars=0.0, fog=(240, 204, 164), water=(122, 112, 124),
                   deep=(30, 32, 64), refl=(255, 214, 142), birds=True),
    "sunset": dict(sky=[(0, (28, 18, 68)), (0.45, (122, 52, 112)), (0.78, (242, 112, 90)), (1, (255, 192, 112))],
                   body="sun", body_col=(255, 228, 150), glow=(255, 150, 80), sun_h=0.05, light=(255, 156, 104),
                   lum=0.75, cloud=(255, 142, 122), stars=0.1, fog=(200, 112, 122), water=(150, 70, 82),
                   deep=(26, 14, 44), refl=(255, 202, 122), birds=True),
    "dusk":   dict(sky=[(0, (14, 16, 46)), (0.5, (54, 44, 100)), (0.85, (170, 96, 118)), (1, (224, 140, 120))],
                   body=None, body_col=None, glow=(210, 120, 110), sun_h=0.0, light=(180, 126, 156),
                   lum=0.5, cloud=(140, 96, 126), stars=0.45, fog=(110, 92, 130), water=(72, 62, 112),
                   deep=(10, 10, 30), refl=(224, 150, 140), birds=False),
    "night":  dict(sky=[(0, (4, 6, 20)), (0.6, (12, 22, 58)), (1, (40, 52, 96))],
                   body="moon", body_col=(246, 242, 218), glow=(110, 125, 170), sun_h=0.6, light=(150, 166, 212),
                   lum=0.44, cloud=(58, 64, 98), stars=1.0, fog=(50, 60, 92), water=(28, 40, 82),
                   deep=(3, 6, 18), refl=(232, 228, 204), birds=False),
}


FLAMES = {
    "fire": [(0, (0, 0, 0)), (0.18, (62, 6, 2)), (0.38, (168, 28, 6)), (0.58, (240, 94, 16)),
             (0.78, (255, 182, 52)), (0.92, (255, 238, 152)), (1, (255, 255, 236))],
}


def draw_pine(ch, fg, bg, x, base, h, hw, col, sundir, rng, snow=None, trunk=None):
    """A layered pine: '^' tip, '/' and '\\' edges, needle texture, optional snow on each tier."""
    H, W = ch.shape
    h = max(2, int(h))
    base = int(base)
    for k in range(h):
        y = base - h + k
        if y < 0 or y >= H:
            continue
        tier = (k * 4 / h) % 1                      # four branch tiers, each flaring out
        half = max(0.5, hw * (k + 1) / h * (0.70 + 0.30 * tier))
        x0, x1 = int(round(x - half)), int(round(x + half))
        a0, a1 = max(0, x0), min(W - 1, x1)
        if a0 > a1:
            continue
        xs = np.arange(a0, a1 + 1)
        litv = 0.78 + 0.22 * (xs - x) / half * sundir
        cc = col[None, :] * litv[:, None]
        if snow is not None and tier < 0.35:
            cc = cc * 0.3 + snow[None, :] * 0.7 * litv[:, None]
        bg[y, a0:a1 + 1] = cc
        ch[y, a0:a1 + 1] = np.where(rng.random(len(xs)) < 0.22, ord(":"), SPACE)
        fg[y, a0:a1 + 1] = cc * 0.7
        if x0 >= 0:
            ch[y, x0], fg[y, x0] = ord("/"), np.clip(cc[0] * 1.35, 0, 1)
        if x1 <= W - 1 and x1 != x0:
            ch[y, x1], fg[y, x1] = ord("\\"), np.clip(cc[-1] * 1.35, 0, 1)
    xi = int(round(x))
    if 0 <= xi < W:
        if 0 <= base - h < H:
            ch[base - h, xi], fg[base - h, xi] = ord("^"), np.clip(col * 1.3, 0, 1)
        if trunk is not None and 0 <= base < H:
            ch[base, xi], fg[base, xi], bg[base, xi] = ord("|"), trunk * 1.3, trunk * 0.7


def draw_round(ch, fg, bg, x, base, h, a, pal, trunk, sundir, rng):
    """A leafy broadleaf tree: lumpy canopy with leaf texture over a trunk. Returns canopy x-radius."""
    H, W = ch.shape
    ry = h * 0.36
    rx = ry * a * 0.95
    cy = base - h + ry
    xi = int(round(x))
    tw = max(1, int(round(0.06 * h * a)))
    for y in range(max(0, int(cy + ry * 0.4)), min(H, int(base) + 1)):
        for xx in range(max(0, xi - tw // 2), min(W, xi - tw // 2 + tw)):
            ch[y, xx], fg[y, xx], bg[y, xx] = ord("|"), trunk * 1.25, trunk * 0.75
    y0, y1 = max(0, int(cy - ry)), min(H, int(cy + ry) + 1)
    x0, x1 = max(0, int(x - rx)), min(W, int(x + rx) + 1)
    if y0 >= y1 or x0 >= x1:
        return rx
    yy, xx = np.mgrid[y0:y1, x0:x1]
    nx, ny = (xx - x) / rx, (yy - cy) / ry
    inside = np.hypot(nx, ny) < 0.86 + 0.14 * np.sin(np.arctan2(ny, nx) * 7 + rng.uniform(0, 6))
    if not inside.any():
        return rx
    litv = np.clip(0.75 + 0.3 * (nx * sundir * 0.6 - ny * 0.6), 0.45, 1.15)
    colr = np.array(pal, F32)[rng.integers(len(pal), size=inside.shape)] * litv[..., None]
    sb, sc, sf = bg[y0:y1, x0:x1], ch[y0:y1, x0:x1], fg[y0:y1, x0:x1]
    sb[inside] = colr[inside]
    leaf = codes("@&%8")[rng.integers(4, size=inside.shape)]
    sc[inside] = np.where(rng.random(inside.shape) < 0.45, leaf, SPACE)[inside]
    sf[inside] = np.clip(colr[inside] * np.where(rng.random(int(inside.sum())) < 0.5, 1.3, 0.7)[:, None], 0, 1)
    return rx


class Particles:
    """Weather and small life drawn over a scene: rain, snow, leaves, petals, fireflies."""

    PER = {"rain": 32, "snow": 45, "leaves": 420, "petals": 420, "fireflies": 650}

    def __init__(self, kind, rng, W, H, sx, sy, a, dens):
        self.kind, self.rng, self.W, self.H, self.sx, self.sy, self.a = kind, rng, W, H, sx, sy, a
        n = max(6, int(W * H / self.PER[kind] * dens))
        self.x, self.y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        if kind == "fireflies":
            self.y = rng.uniform(H * 0.45, H, n)
        self.v, self.ph, self.ang = rng.uniform(0.7, 1.3, n), rng.uniform(0, TAU, n), rng.uniform(0, TAU, n)
        self.wind = rng.uniform(-0.4, 0.4)
        if kind == "leaves":
            self.col = stops(rng.random(n), [(0, (200, 70, 30)), (0.4, (232, 122, 40)), (0.75, (242, 182, 62)), (1, (150, 92, 42))])
        elif kind == "petals":
            self.col = stops(rng.random(n), [(0, (255, 186, 210)), (1, (255, 246, 246))])


class Nature(Scene):
    """Base for the landscapes: sky, sun/moon, clouds, stars, birds, aurora,
    open sea, mirrored water, fog and weather, all driven by a time of day."""

    tods = tuple(TOD)
    base_name = "nature"
    aurora_p = 0.0
    body_range = (0.2, 0.8)     # where across the sky the sun/moon may sit

    def recolor(self):  # 'c': a new time of day (and season / weather) for the same place
        self.tod = self.pick(self.tods)
        self.T = TOD[self.tod]
        self.variant()
        self.name = self.label()
        if self.size:
            self.setup(*self.size[:2])

    def variant(self):
        pass

    def label(self):
        return f"{self.base_name} · {self.tod}"

    def lit(self, col, k=1.0):
        """A surface colour as it looks under this time of day's light."""
        T = self.T
        return np.asarray(col, F32) * (rgb(*T["light"]) * 0.55 + 0.45) * (T["lum"] * k)

    @staticmethod
    def blank(H, W):
        return np.full((H, W), SPACE, "<u4"), np.zeros((H, W, 3), F32), np.zeros((H, W, 3), F32)

    # ── sky ──
    def sky_setup(self, W, H, hr):
        r, T = self.rng, self.T
        self.hr = hr
        tt = np.clip(self.rows[:hr, :1] / max(1, hr - 1), 0, 1)
        sky = np.broadcast_to(stops(tt, T["sky"]), (hr, W, 3)).copy()
        self.horizon_col = stops(np.array([1.0]), T["sky"])[0]
        self.mr = max(2.0, H * (0.07 if T["body"] == "moon" else 0.085))
        self.bx = r.uniform(W * self.body_range[0], W * self.body_range[1])
        self.by = hr - 1 - T["sun_h"] * hr * 0.85
        self.bd = np.hypot((self.cols - self.bx) / self.a, self.rows - self.by) / self.mr
        if T["body"]:
            sun = T["body"] == "sun"
            g = np.exp(-np.maximum(self.bd[:hr] - 1, 0) * (0.55 if sun else 0.8)) * (0.55 if sun else 0.35)
        else:  # the sun has just set: a broad glow along the horizon
            g = (np.exp(-np.abs(self.cols[:hr] - self.bx) / (W * 0.35))
                 * np.exp(-(hr - 1 - self.rows[:hr]) / (hr * 0.3)) * 0.45)
        sky += g[..., None] * rgb(*T["glow"])
        self.sky_bg = np.clip(sky, 0, 1).astype(F32)
        n = int(W * hr / 40 * self.dens * T["stars"])
        self.stx, self.sty = r.integers(0, W, n), r.integers(0, max(1, int(hr * 0.85)), n)
        self.sph, self.sbr = r.uniform(0, TAU, n), r.uniform(0.4, 1.0, n)
        self.cl, self.cloudy = self.u(0.7, 2.0, 2), self.u(0.5, 0.8)
        self.crater = r.random((H, W)) < 0.2
        nb = int(r.integers(2, 6)) if T["birds"] else 0
        self.birx, self.biry = r.uniform(0, W, nb), r.uniform(hr * 0.15, hr * 0.55, nb)
        self.birv, self.birp = r.uniform(0.12, 0.3, nb) * r.choice([-1, 1], nb), r.uniform(0, TAU, nb)
        self.aurora = self.tod == "night" and r.random() < self.aurora_p

    # ── open sea below the horizon ──
    def sea_setup(self, W, H):
        hr, T = self.hr, self.T
        R = np.arange(hr, H)[:, None].astype(F32)
        self.sdep = (R - hr + 1) / (H - hr)
        self.sRc = R / self.sy
        self.wk = self.u(6, 12, 3)
        water, deep = rgb(*T["water"]), rgb(*T["deep"])
        far = self.horizon_col * 0.45 + water * 0.55
        d = (self.sdep ** 0.8)[..., None]
        sea = np.broadcast_to(far * (1 - d) + deep * d, (H - hr, W, 3)).copy()
        if T["body"]:
            path = np.exp(-np.abs(self.cols[hr:] - self.bx) / (self.a * self.mr * (0.6 + 2.5 * self.sdep)))
            sea += (path * (0.32 - 0.15 * self.sdep))[..., None] * rgb(*T["refl"])
        else:
            sea += np.exp(-self.sdep * 4)[..., None] * 0.15 * rgb(*T["glow"])
        self.sea_bg = np.clip(sea, 0, 1).astype(F32)

    def stamp(self, ch, fg, top, left, lines, color, bg=None):
        """Draw a small ASCII sprite; each character fills one character cell's worth of pixels
        (solid in the pixel looks, where thin glyphs would barely show)."""
        sx, sy = self.sx, self.sy
        H, W = ch.shape
        for i, line in enumerate(lines):
            for j, g in enumerate(line):
                if g == " ":
                    continue
                y0, x0 = top + i * sy, left + j * sx
                if 0 <= y0 and y0 + sy <= H and 0 <= x0 and x0 + sx <= W:
                    ch[y0:y0 + sy, x0:x0 + sx] = ord(g)
                    fg[y0:y0 + sy, x0:x0 + sx] = color(i, g)
                    if bg is not None and sx * sy > 1:
                        bg[y0:y0 + sy, x0:x0 + sx] = color(i, g)


class Seaside(Nature):
    """A beach: waves wash up the sand and drain back, breakers roll in, a boat drifts by."""

    base_name = "seaside"

    def variant(self):
        self.boat = self.rng.random() < 0.75
        self.wkind = "rain" if self.tod in ("dawn", "dusk", "night") and self.rng.random() < 0.2 else None

    def setup(self, W, H):
        r, T = self.rng, self.T
        hr = int(H * self.u(0.38, 0.46))
        self.sky_setup(W, H, hr)
        self.sea_setup(W, H)
        self.xp = xp = (np.arange(W) - W / 2) / (H * self.a / 2)
        span = H - hr
        self.shore = (hr + span * self.u(0.42, 0.55) + span * 0.12 * np.sin(xp * 0.8 + self.u(0, 6))
                      + span * 0.04 * fbm1d(xp * 2, r, 3))
        self.swash = span * 0.08
        self.sand = self.lit(rgb(226, 200, 152))
        self.stex = r.random((H, W))
        shallow = rgb(80, 182, 178) if self.tod in ("day", "golden") else rgb(*T["water"]) * 1.25
        near = np.clip(1 - (self.shore[None, :] - self.rows[hr:]) / (span * 0.22), 0, 1)
        self.sea_bg += (shallow * (0.4 + 0.6 * T["lum"]) - self.sea_bg) * (near * 0.55)[..., None]
        self.boat_x, self.boat_v = self.u(0.1, 0.9) * W, self.u(0.03, 0.08) * self.pick([-1, 1])
        self.boat_row = int(hr + span * 0.07)
        self.weather = Particles(self.wkind, r, W, H, self.sx, self.sy, self.a, self.dens) if self.wkind else None

    def _boat(self, c, ch, fg, bg):
        self.boat_x = (self.boat_x + self.boat_v * c.dt * 30 * self.sx) % c.W
        bob = int(round(math.sin(c.t * 1.3) * 0.4 * self.sy))
        sail = np.clip(self.lit(rgb(246, 242, 232), 1.3), 0, 1)
        hull = np.clip(self.lit(rgb(130, 80, 46), 1.4), 0, 1)
        lines = [" |\\", " |_\\", "\\___/"] if self.boat_v > 0 else ["  /|", " /_|", "\\___/"]
        self.stamp(ch, fg, self.boat_row - 3 * self.sy + bob, int(self.boat_x), lines,
                   lambda i, g: hull if i == 2 else (hull * 0.8 if g == "|" else sail), bg)
        if self.T["lum"] < 0.6:  # a lantern at the masthead after dark
            y, x = self.boat_row - 3 * self.sy + bob, int(self.boat_x) + (1 if self.boat_v > 0 else 3) * self.sx
            if 0 <= y < c.H and 0 <= x < c.W:
                ch[y, x], fg[y, x] = ord("*"), rgb(255, 214, 120)


class Lighthouse(Nature):
    """A striped lighthouse on a headland; its beam sweeps the sky, surf breaks on the rocks."""

    base_name = "lighthouse"
    tods = ("night", "night", "dusk", "sunset", "dawn")

    def variant(self):
        self.side = self.pick([-1, 1])
        self.body_range = (0.55, 0.85) if self.side < 0 else (0.15, 0.45)   # sun/moon over open sea
        self.stripes = self.rng.random() < 0.7
        self.spin = self.u(0.45, 0.7)
        dark = self.tod in ("night", "dusk")
        self.wkind = self.pick([None, None, "rain", "fog"]) if dark else None

    def setup(self, W, H):
        r, T, sx, sy = self.rng, self.T, self.sx, self.sy
        hr = int(H * self.u(0.44, 0.52))
        self.sky_setup(W, H, hr)
        self.sea_setup(W, H)
        xp = (np.arange(W) - W / 2) / (H * self.a / 2)
        rows = self.rows
        ch, fg, bg = self.blank(H, W)
        width = W * self.u(0.34, 0.44)
        dist = np.arange(W) if self.side < 0 else (W - 1 - np.arange(W))
        u_ = dist / width
        shape = np.clip((1.0 - u_) / 0.30, 0, 1) ** 0.7               # plateau, then a craggy fall to the sea
        cliff = hr * self.u(0.16, 0.24)
        drop = np.clip((u_ - 0.7) / 0.5, 0, 1)                         # the face slopes out under the water
        top = hr - cliff * shape + hr * (0.05 + 0.08 * drop) * fbm1d(xp * 4, r, 4) * (shape > 0)
        top = top + (H - hr) * drop ** 1.5 * (u_ < 1.2)
        top = np.where((shape > 0.02) | ((u_ < 1.2) & (top < H)), top, H + 5).astype(int)
        land = rows >= top[None, :]
        dr = np.clip((rows - top[None, :]) / max(1, H - hr * 0.5), 0, 1)
        steep = (np.abs(np.gradient(top.astype(F32))) > 0.8 * sy)[None, :]
        rock = self.lit(rgb(150, 140, 134))
        strata = 0.84 + 0.16 * np.sin(rows * 0.9 / sy + 3 * fbm1d(xp * 2.5, r, 3)[None, :])   # layered rock
        rockc = (rock[None, None, :] * (strata * (0.95 - 0.35 * dr))[..., None]
                 * np.where(steep, 0.78, 1.0)[..., None])
        bg[land] = np.broadcast_to(rockc, (H, W, 3))[land]
        tex = r.random((H, W))
        rt = land & (tex < 0.16)
        ch[rt] = np.where(np.broadcast_to(steep, (H, W))[rt], ord("|"), ord(":"))
        fg[rt] = np.clip(bg[rt] * 1.3, 0, 1)
        grass = self.lit(rgb(72, 116, 60))
        gm = land & ((rows - top[None, :]) < max(1, int(hr * 0.045))) & ~steep
        bg[gm] = grass
        gt = gm & (tex < 0.4)
        ch[gt] = codes("\"',")[r.integers(0, 3, int(gt.sum()))]
        fg[gt] = np.clip(grass * 1.4, 0, 1)
        # the lighthouse
        lx = int(width * 0.45) if self.side < 0 else int(W - 1 - width * 0.45)
        base = int(top[lx])
        th = int(hr * 0.42)
        hwb = max(1.2 * sx, th * 0.09 * self.a)
        red = np.clip(self.lit(rgb(200, 48, 46), 1.6), 0, 1)
        white = np.clip(self.lit(rgb(238, 236, 228), 1.6), 0, 1)
        for k in range(th):
            y = base - 1 - k
            if y < 0:
                break
            f = k / th
            hw = hwb * (1 - 0.38 * f)
            x0, x1 = int(round(lx - hw)), int(round(lx + hw))
            xs = np.arange(x0, x1 + 1)
            band = self.stripes and int(f * 5) % 2 == 1
            shade = 0.78 + 0.22 * (xs - lx) / max(hw, 0.5) * (-self.side)
            bg[y, x0:x1 + 1] = (red if band else white)[None, :] * shade[:, None]
            ch[y, x0:x1 + 1], fg[y, x0:x1 + 1] = SPACE, 0
            land[y, x0:x1 + 1] = True
        win = rgb(255, 210, 120) if T["lum"] < 0.6 else rgb(40, 40, 52)
        for f in (0.3, 0.6):
            y = base - 1 - int(th * f)
            if y >= 0:
                ch[y, lx], fg[y, lx] = ord(":"), win
        gy = base - 1 - th
        gw = int(round(hwb * 0.62 + sx))
        rail = np.clip(self.lit(rgb(120, 120, 130), 1.5), 0, 1)
        lantern_rows = max(2, 2 * sy)
        if gy - lantern_rows - sy >= 0:
            ch[gy, lx - gw:lx + gw + 1], fg[gy, lx - gw:lx + gw + 1] = ord("="), rail
            bg[gy, lx - gw:lx + gw + 1], land[gy, lx - gw:lx + gw + 1] = rail * 0.5, True
            lw = max(1, int(round(hwb * 0.45)))
            for j in range(1, lantern_rows + 1):
                y = gy - j
                ch[y, lx - lw:lx + lw + 1], fg[y, lx - lw:lx + lw + 1] = ord("#"), rgb(255, 248, 214)
                bg[y, lx - lw:lx + lw + 1], land[y, lx - lw:lx + lw + 1] = rgb(255, 222, 148), True
            roof = np.clip(self.lit(rgb(110, 40, 36), 1.4), 0, 1)
            for j in range(max(1, sy)):
                y = gy - lantern_rows - 1 - j
                hw = max(0, lw - j)
                ch[y, lx - hw:lx + hw + 1], fg[y, lx - hw:lx + hw + 1] = ord("^"), roof
                bg[y, lx - hw:lx + hw + 1], land[y, lx - hw:lx + hw + 1] = roof * 0.7, True
        self.lx, self.ly = lx, gy - lantern_rows / 2
        # where the sea meets the rock: surf
        coast = np.zeros((H, W), bool)
        for j in range(1, 2 * sx + 1):
            coast |= np.roll(land, j * -self.side, axis=1)
        coast &= ~land & (rows >= hr)
        self.cy, self.cx = np.nonzero(coast)
        self.cph = r.uniform(0, TAU, len(self.cy))
        n = 80
        self.spx, self.spy, self.svx, self.svy, self.slife = (np.zeros(n) for _ in range(5))
        self.kick_prev = 0.0
        self.t_ch, self.t_fg, self.t_bg, self.t_mask = ch, fg, bg, land
        self.weather = Particles("rain", r, W, H, sx, sy, self.a, self.dens) if self.wkind == "rain" else None


class Mountains(Nature):
    """Layered ranges fading into haze: snow caps, rim-lit ridges, valley mist, pines, aurora at night."""

    base_name = "mountains"
    aurora_p = 0.6
    lake = False

    def variant(self):
        r = self.rng
        self.layers = int(r.integers(4, 7))
        self.snowy = r.random() < 0.7
        self.mist = self.u(0.35, 0.8)
        self.freq = self.u(0.7, 1.1)
        self.wkind = "snow" if (self.snowy and r.random() < 0.25) else ("fog" if r.random() < 0.15 else None)

    def setup(self, W, H):
        r, T = self.rng, self.T
        land = int(H * self.u(0.66, 0.74)) if self.lake else H
        self.wl = land
        self.sky_setup(W, H, land)
        xp = (np.arange(W) - W / 2) / (H * self.a / 2)
        rows, cx = self.rows, np.arange(W)
        ch, fg, bg = self.blank(H, W)
        mask = np.zeros((H, W), bool)
        fogm = np.zeros((H, W), F32)
        light, hz = rgb(*T["light"]), self.horizon_col
        fogc = (rgb(*T["fog"]) * (0.5 + 0.5 * T["lum"]) + hz) / 2
        sundir = 1.0 if self.bx > W / 2 else -1.0
        tex = r.random((H, W))
        tree = self.lit(rgb(40, 84, 58))
        L = self.layers
        for i in range(L):
            k = i / (L - 1)                                   # 0 = farthest range, 1 = nearest
            # a range = overlapping peaks (each a sharp ridge falling away on both sides) + fractal detail
            span = xp[-1] - xp[0]
            npk = int(span * self.freq * (1.6 + 1.6 * k)) + 3
            px_ = r.uniform(xp[0] - 0.3, xp[-1] + 0.3, npk)
            ph_ = r.uniform(0.45, 1.0, npk)
            ps_ = r.uniform(0.9, 1.8, npk) * (1 + 0.8 * k)
            prof = np.max(ph_[:, None] - np.abs(xp[None, :] - px_[:, None]) * ps_[:, None], axis=0)
            prof = np.clip(prof + 0.07 * fbm1d(xp * 6 + i * 3.1, r, 4, 0.55), 0, 1)
            top = np.clip(land * (0.22 + 0.40 * k) + (1 - prof) * land * (0.34 - 0.12 * k), 0, land - 1).astype(int)
            m = (rows >= top[None, :]) & (rows < land)
            haze = 0.80 * (1 - k) ** 1.3 + 0.05              # far ranges melt into the sky
            rock = self.lit(rgb(70, 74, 92) * (1.0 - 0.35 * k))
            base = rock + (hz - rock) * haze
            grad = np.gradient(top.astype(F32)) / max(1e-6, xp[1] - xp[0])
            grad = np.convolve(grad, np.ones(5) / 5, "same")
            litv = 0.5 + 0.5 * np.tanh(grad * sundir / (land * 0.8))   # slopes facing the sun are lit
            sunk = 0.25 if T["body"] is None else 0.45
            col = (base[None, None, :] * (0.75 + sunk * litv)[None, :, None]
                   + light * (0.10 * (1 - haze)) * litv[None, :, None])
            col = np.broadcast_to(col, (H, W, 3)).copy()
            tmean = float(np.percentile(top, 35))
            dr = np.clip((rows - tmean) / (land * 0.28), 0, 1) * np.ones((1, W))
            mf = (dr ** 1.4) * self.mist * (0.55 + 0.45 * (1 - k))          # mist pooling in the valleys
            col += (fogc - col) * mf[..., None]
            if self.snowy and k < 0.75:
                p = np.clip((land * 0.40 - top) / (land * 0.22), 0, 1)
                sd = p * land * 0.22 * (0.55 + 0.45 * (fbm1d(xp * 4 + i, r, 3) + 1) / 2)
                sm = m & ((rows - top[None, :]) < sd[None, :])
                snow = self.lit(rgb(236, 240, 250)) * (0.7 + 0.45 * litv)[None, :, None]
                snow = np.broadcast_to(snow + (hz - snow) * haze * 0.7, (H, W, 3))
                col[sm] = snow[sm]
            bg[m] = col[m]
            ch[m], fg[m] = SPACE, 0
            tm = m & (tex < 0.03 + 0.10 * k)
            ch[tm] = codes(".:'")[r.integers(0, 3, int(tm.sum()))]
            fg[tm] = np.clip(col[tm] * 1.25, 0, 1)
            if k > 0.55 and not (self.snowy and k < 0.75):     # pine forest on the near slopes
                tr = m & ((rows - top[None, :]) > land * 0.03) & (tex > 1 - 0.30 * k)
                ch[tr] = ord("^")
                fg[tr] = np.clip(tree + (fogc - tree) * (mf[tr] * 0.8)[:, None], 0, 1)
            nxt, prv = np.r_[top[1:], top[-1]], np.r_[top[0], top[:-1]]
            rg = np.where(nxt < top, ord("/"), np.where(nxt > top, ord("\\"), ord("_")))
            rg = np.where((prv > top) & (nxt > top), ord("^"), rg)
            ch[top, cx], fg[top, cx] = rg, np.clip(col[top, cx] * 1.35 + light * 0.08, 0, 1)
            mask |= m
            fogm[m] = mf[m]
        self.t_ch, self.t_fg, self.t_bg, self.t_mask = ch, fg, bg, mask
        self.fogm, self.fogc = fogm, fogc
        self.weather = Particles("snow", r, W, H, self.sx, self.sy, self.a, self.dens) if self.wkind == "snow" else None


class Lake(Mountains):
    """The same ranges, mirrored in a rippling lake."""

    base_name = "mountain lake"
    lake = True


class Forest(Nature):
    """Wooded hills over a meadow, through the seasons; grass sways on the beat."""

    base_name = "forest"
    aurora_p = 0.25
    SEASONS = {
        "spring": dict(leaf=[(120, 200, 90), (160, 222, 112), (255, 190, 212)], grass=(110, 180, 80),
                       ground=(70, 120, 50), pine=(40, 92, 62), flowers=0.05, weather=[None, "petals"]),
        "summer": dict(leaf=[(50, 122, 50), (72, 152, 62), (40, 102, 46)], grass=(82, 152, 62),
                       ground=(50, 100, 40), pine=(30, 80, 50), flowers=0.025, weather=[None, None, "rain"]),
        "autumn": dict(leaf=[(212, 92, 30), (232, 152, 42), (182, 52, 30), (202, 172, 62)], grass=(152, 140, 72),
                       ground=(112, 92, 52), pine=(40, 80, 55), flowers=0.004, weather=["leaves", "leaves", None]),
        "winter": dict(leaf=None, grass=(226, 232, 242), ground=(212, 222, 236), pine=(30, 70, 55),
                       flowers=0.0, weather=["snow", "snow", None]),
    }
    FLOWERS = [(240, 60, 70), (255, 214, 60), (250, 250, 255), (180, 110, 255), (100, 160, 255)]

    def variant(self):
        r = self.rng
        self.season = self.pick(list(self.SEASONS))
        S = self.SEASONS[self.season]
        self.wkind = self.pick(S["weather"])
        if self.tod in ("dusk", "night") and self.season in ("spring", "summer") and r.random() < 0.7:
            self.wkind = "fireflies"
        self.pine_ratio = 1.0 if self.season == "winter" else self.u(0.3, 0.8)

    def label(self):
        return f"forest · {self.season} {self.tod}"

    def setup(self, W, H):
        r, T, S = self.rng, self.T, self.SEASONS[self.season]
        hz = int(H * 0.56)
        self.sky_setup(W, H, hz)
        xp = (np.arange(W) - W / 2) / (H * self.a / 2)
        rows = self.rows
        ch, fg, bg = self.blank(H, W)
        hzc = self.horizon_col
        sundir = 1.0 if self.bx > W / 2 else -1.0
        winter = self.season == "winter"
        snow = self.lit(rgb(236, 240, 250)) if winter else None
        for i, (tm, amp, haze, tree_h) in enumerate([(0.47, 0.05, 0.62, 0.07), (0.55, 0.05, 0.38, 0.11),
                                                      (0.63, 0.04, 0.16, 0.17)]):
            top = ((tm + amp * fbm1d(xp * 0.9 + i * 5.1, r, 4, 0.5)) * H).astype(int)
            if i == 0:
                top = np.minimum(top, hz - 1)
            m = rows >= top[None, :]
            g = self.lit(rgb(*S["ground"]) if i < 2 else (rgb(*S["ground"]) + rgb(*S["grass"])) / 2)
            g = g + (hzc - g) * haze
            dr = np.clip((rows - top[None, :]) / (H * 0.25), 0, 1)
            bg[m] = np.broadcast_to(g[None, None, :] * (1 - 0.25 * dr[..., None]), (H, W, 3))[m]
            ch[m], fg[m] = SPACE, 0
            x = r.uniform(-3, 0) * self.sx
            while x < W + 3 * self.sx:
                h = int(H * tree_h * self.u(0.6, 1.25))
                base = top[int(np.clip(x, 0, W - 1))] + int(h * 0.05)
                if r.random() < self.pine_ratio:
                    pc = self.lit(rgb(*S["pine"]))
                    pc = pc + (hzc - pc) * haze
                    hw = h * 0.24 * self.a
                    sn = None if snow is None else snow * (1 - haze) + hzc * haze
                    draw_pine(ch, fg, bg, x, base, h, hw, pc, sundir, r, snow=sn)
                    x += hw * self.u(0.9, 1.9)
                else:
                    pal = [self.lit(rgb(*cc)) for cc in S["leaf"]]
                    pal = [p + (hzc - p) * haze for p in pal]
                    tr = self.lit(rgb(92, 64, 44))
                    rx = draw_round(ch, fg, bg, x, base, h, self.a, pal, tr + (hzc - tr) * haze, sundir, r)
                    x += rx * self.u(1.0, 2.0)
        # the meadow in front
        mt = ((0.74 + 0.02 * fbm1d(xp * 1.4 + 9, r, 3)) * H).astype(int)
        mm = rows >= mt[None, :]
        gy = np.clip((rows - mt[None, :]) / max(1, H - int(mt.min())), 0, 1)
        gc = self.lit(rgb(*S["grass"]))
        col = np.broadcast_to(gc[None, None, :] * (0.95 - 0.35 * gy[..., None]), (H, W, 3))
        edge = np.clip((rows - mt[None, :]) / max(1.0, 1.5 * self.sy), 0, 1)[..., None]   # soft meadow edge
        bg[mm] = (bg * (1 - edge) * 0.5 + col * (0.5 + 0.5 * edge))[mm]
        ch[mm], fg[mm] = SPACE, 0
        self.gmask = mm & (r.random((H, W)) < (0.06 if winter else 0.55))
        stalk = self.lit(rgb(150, 130, 90)) if winter else None
        self.gcol = np.clip(col * (1.25 + 0.35 * r.random((H, W, 1))) if stalk is None
                            else np.broadcast_to(stalk, (H, W, 3)), 0, 1).astype(F32)
        self.gph = r.uniform(0, 1.2, (H, W))
        fm = mm & ~self.gmask & (r.random((H, W)) < S["flowers"])
        ch[fm] = np.where(r.random(int(fm.sum())) < 0.5, ord("*"), ord("o"))
        fg[fm] = np.clip(np.array([self.lit(rgb(*f), 1.3) for f in self.FLOWERS])[r.integers(5, size=int(fm.sum()))], 0, 1)
        self.t_ch, self.t_fg, self.t_bg = ch, fg, bg
        self.t_mask = rows >= hz - int(H * 0.2)
        self.t_mask &= (bg.sum(-1) > 0) | (ch != SPACE)
        self.weather = Particles(self.wkind, r, W, H, self.sx, self.sy, self.a, self.dens) if self.wkind else None


class Campfire(Nature):
    """A fire in a clearing at night: it lights the tent and trees, sparks and smoke rise."""

    base_name = "campfire"
    frame_pines = True          # the two big foreground pines; tools/story.py turns them off for portrait video
    tods = ("night", "night", "dusk")

    def variant(self):
        self.tent_side = self.pick([-1, 1])
        self.wkind = self.pick([None, None, "fireflies", "snow"])

    def setup(self, W, H):
        r, T, sx, sy, a = self.rng, self.T, self.sx, self.sy, self.a
        hz = int(H * self.u(0.50, 0.56))
        self.sky_setup(W, H, hz)
        xp = (np.arange(W) - W / 2) / (H * a / 2)
        rows = self.rows
        ch, fg, bg = self.blank(H, W)
        snowy = self.wkind == "snow"
        gcol = self.lit(rgb(222, 228, 240) if snowy else rgb(56, 70, 46), 1.2)
        gtop = hz + (H * 0.015 * fbm1d(xp * 2, r, 3)).astype(int)
        gm = rows >= gtop[None, :]
        dr = np.clip((rows - hz) / max(1, H - hz), 0, 1)
        bg[gm] = np.broadcast_to(gcol[None, None, :] * (0.75 + 0.35 * dr[..., None]), (H, W, 3))[gm]
        tex = r.random((H, W))
        gt = gm & (tex < 0.12)
        ch[gt] = codes(",'.")[r.integers(0, 3, int(gt.sum()))]
        fg[gt] = np.clip(bg[gt] * 1.4, 0, 1)
        pine = self.lit(rgb(26, 54, 40), 1.2)
        trunk = self.lit(rgb(70, 50, 36), 1.2)
        sn = self.lit(rgb(230, 236, 248)) if snowy else None
        x = r.uniform(-2, 0) * sx
        while x < W + 2 * sx:                           # the treeline around the clearing
            h = int(H * self.u(0.10, 0.24))
            hw = h * 0.24 * a
            draw_pine(ch, fg, bg, x, hz + int(h * 0.05), h, hw, pine + (self.horizon_col - pine) * 0.2, 1.0, r, snow=sn)
            x += hw * self.u(0.7, 1.4)
        for side in ((-1, 1) if self.frame_pines else ()):   # two big pines framing the view
            h = int(H * 0.8)
            draw_pine(ch, fg, bg, W / 2 + side * W * 0.52, H - 1, h, h * 0.24 * a, pine * 0.8, -side, r,
                      snow=sn, trunk=trunk)
        self.fx = W / 2 - self.tent_side * W * 0.06
        self.fb = int(hz + (H - hz) * 0.62)
        self.fw = max(6, int(0.34 * H * a / 2))
        self.fh = max(6, int(H * 0.30))
        self.heat = np.zeros((self.fh + 1, self.fw), F32)
        xs = np.arange(self.fw)
        self.fprof = np.exp(-((xs - self.fw / 2) / (self.fw * 0.26)) ** 2)
        self.fcols, self.frows = xs, np.arange(self.fh)[:, None]
        wood = self.lit(rgb(110, 70, 40), 1.6)
        lx0, lx1 = int(self.fx - self.fw * 0.55), int(self.fx + self.fw * 0.55)
        for j in range(max(1, sy)):
            y = self.fb + j
            if y < H:
                ch[y, max(0, lx0):lx1], fg[y, max(0, lx0):lx1], bg[y, max(0, lx0):lx1] = ord("="), wood, wood * 0.5
        stone = self.lit(rgb(120, 116, 112), 1.4)
        for ang in np.linspace(0.15, math.pi - 0.15, 9):
            y = int(self.fb + sy * 0.5 + math.sin(ang) * sy * 1.2)
            x_ = int(self.fx + math.cos(ang) * self.fw * 0.78)
            if 0 <= y < H and 0 <= x_ < W:
                ch[y, x_:x_ + sx], fg[y, x_:x_ + sx] = ord("o"), stone
        tx = self.fx + self.tent_side * W * 0.27
        tb = self.fb + int(sy * 0.5)
        th = int(H * 0.24)
        thw = th * 0.55 * a
        canvas = self.lit(rgb(196, 122, 62), 1.3)
        for k in range(th):
            y = tb - th + k
            half = thw * (k + 1) / th
            x0, x1 = int(round(tx - half)), int(round(tx + half))
            a0, a1 = max(0, x0), min(W - 1, x1)
            if a0 > a1 or y < 0 or y >= H:
                continue
            xs_ = np.arange(a0, a1 + 1)
            facing = 0.8 + 0.25 * np.sign(self.fx - tx) * (xs_ - tx) / max(half, 1)
            bg[y, a0:a1 + 1] = canvas[None, :] * facing[:, None]
            ch[y, a0:a1 + 1], fg[y, a0:a1 + 1] = SPACE, 0
            if k > th * 0.45:                          # the door flap
                hd = thw * 0.32 * (k - th * 0.45) / (th * 0.55)
                d0, d1 = max(0, int(round(tx - hd))), min(W - 1, int(round(tx + hd)))
                bg[y, d0:d1 + 1] = rgb(26, 18, 14)
            if x0 >= 0:
                ch[y, x0], fg[y, x0] = ord("/"), np.clip(canvas * 1.4, 0, 1)
            if x1 <= W - 1:
                ch[y, x1], fg[y, x1] = ord("\\"), np.clip(canvas * 1.4, 0, 1)
        dx = (self.cols - self.fx) / (H * a / 2)
        dy = (rows - (self.fb - self.fh * 0.3)) / (H / 2)
        dist = np.hypot(dx, dy)
        self.Lmap = np.exp(-dist / 0.45).astype(F32)
        self.Lair = np.exp(-dist / 0.18).astype(F32)
        self.t_ch, self.t_fg, self.t_bg = ch, fg, bg
        self.t_mask = (bg.sum(-1) > 0) | (ch != SPACE)
        self.t_mask[:max(0, hz - int(H * 0.3))] &= ch[:max(0, hz - int(H * 0.3))] != SPACE
        n = 50
        self.ex, self.ey, self.evy, self.life = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        self.eph = r.uniform(0, TAU, n)
        self.weather = Particles(self.wkind, r, W, H, sx, sy, a, self.dens) if self.wkind else None


WEATHER_LOOK = {  # cloud cover, snow (share of the storm's flakes), storm haze, wind strength
    "clear": (0.22, 0.12, 0.0, 0.5),
    "snowfall": (0.5, 0.42, 0.08, 0.9),
    "storm": (0.82, 1.0, 0.45, 2.6),
}


class Summit(Nature):
    """Alone on a high ledge at night: a fire, a few flowers bending in the wind, snowy peaks falling away
    into a sea of cloud, the Milky Way (and a faint distant galaxy) behind drifting clouds, a stray ray of
    light through a gap. The music moves it: the wind you hear bends the flowers and drives the snow, the
    lead's phrases open the ray of light, the low layers breathe in the firelight, every audible crackle
    throws sparks, plucked notes make the stars glint, chord changes part the clouds, and the weather
    (clear, snowfall, storm) follows the track."""

    base_name = "summit"
    tods = ("night",)
    aurora_p = 0.0

    def variant(self):
        r = self.rng
        self.layers = int(r.integers(4, 6))
        self.fire_side = self.pick([-1, 1])
        self.nflowers = int(r.integers(6, 11))
        self.look = {"cover": 0.5, "snow": 0.4, "haze": 0.05, "wind": 0.9}

    def label(self):
        return "summit · night"

    def setup(self, W, H):
        r, T, sx, sy, a = self.rng, self.T, self.sx, self.sy, self.a
        hr = int(H * 0.8)
        self.sky_setup(W, H, hr)
        X, Y = self.X, self.Y
        rows, cx = self.rows, np.arange(W)

        # ── the Milky Way: a band of light with dark dust lanes, warm at its core, blue at its edges ──
        ang = r.uniform(0.35, 0.95) * self.pick([-1, 1])
        c0x, c0y = r.uniform(-0.3, 0.3), r.uniform(-0.75, -0.35)
        d = (np.cos(ang) * (Y[:hr] - c0y) - np.sin(ang) * (X[:hr] - c0x))
        n1 = value_noise2(r, hr, W, 5, 4)
        n2 = value_noise2(r, hr, W, 14, 3)
        band = np.exp(-(d / 0.30) ** 2) * (0.5 + 0.5 * n1)
        lanes = np.clip((n2 - 0.4) * 3.0, 0, 1) * np.exp(-(d / 0.12) ** 2)
        gal = np.clip(band * (1 - 0.8 * lanes), 0, 1)
        tint = stops(np.clip(np.abs(d) / 0.35, 0, 1), [(0, (255, 226, 196)), (0.5, (200, 196, 255)), (1, (120, 150, 255))])
        fade = np.clip(1.15 - rows[:hr, :1] / hr, 0, 1) ** 1.5           # thinner towards the horizon
        self.galaxy = (tint * (gal * fade)[..., None] * 0.6).astype(F32)
        # a faint spiral galaxy, far away, off the band
        gx, gy = r.uniform(-0.9, 0.9), r.uniform(-0.85, -0.55)
        if abs(np.cos(ang) * (gy - c0y) - np.sin(ang) * (gx - c0x)) > 0.35:
            dx, dy = (X[:hr] - gx) / 0.09, (Y[:hr] - gy) / 0.045
            rr, th = np.hypot(dx, dy), np.arctan2(dy, dx)
            arms = 0.5 + 0.5 * np.cos(2 * th - 3.0 * np.log(rr + 0.2))
            spiral = np.exp(-rr * 1.6) * (0.55 + 0.45 * arms) + 0.8 * np.exp(-(rr / 0.25) ** 2)
            self.galaxy += (rgb(220, 210, 255) * (spiral * 0.22)[..., None]).astype(F32)
        # many more faint stars inside the band
        cand = r.random((hr, W)) < gal * 0.06 * self.dens
        gy_, gx_ = np.nonzero(cand)
        self.stx, self.sty = np.r_[self.stx, gx_], np.r_[self.sty, gy_]
        self.sph = np.r_[self.sph, r.uniform(0, TAU, len(gx_))]
        self.sbr = np.r_[self.sbr, r.uniform(0.25, 0.7, len(gx_))]

        # clouds: two seamless noise sheets that scroll at different speeds
        self.cl_hi = value_noise2(r, hr, W, 3, 5, 0.6)
        self.cl_lo = value_noise2(r, H, W, 4, 4, 0.55)

        # ── peaks: dark rock and moonlit snow, ranges fading into the distance, cloud pooled in the valleys ──
        xp = (np.arange(W) - W / 2) / (H * a / 2)
        ch, fg, bg = self.blank(H, W)
        mask = np.zeros((H, W), bool)
        fogm = np.zeros((H, W), F32)
        light, hz = rgb(*T["light"]), self.horizon_col
        sundir = 1.0 if self.bx > W / 2 else -1.0
        tex = r.random((H, W))
        streak = value_noise2(r, H, W, 60, 3, 0.5)                          # couloirs: snow runs down gullies
        facet = value_noise2(r, H, W, 18, 3, 0.6)                           # faceted faces catch the moon
        snow_lit, snow_dark = rgb(196, 208, 238), rgb(40, 50, 84)
        rock_lit, rock_dark = rgb(64, 66, 82), rgb(14, 15, 24)
        far = hz * 0.85 + rgb(30, 36, 64) * 0.15
        L = self.layers
        for i in range(L):
            k = i / (L - 1)
            npk = int((xp[-1] - xp[0]) * (1.1 + 1.4 * k)) + 3
            px_ = r.uniform(xp[0] - 0.3, xp[-1] + 0.3, npk)
            ph_ = r.uniform(0.5, 1.0, npk)
            ps_ = r.uniform(1.3, 2.6, npk) * (1 + 0.5 * k)                 # steep, jagged
            prof = np.max(ph_[:, None] - np.abs(xp[None, :] - px_[:, None]) * ps_[:, None], axis=0)
            prof = np.clip(prof + 0.09 * fbm1d(xp * 7 + i * 3.1, r, 5, 0.6), 0, 1)
            top = np.clip(H * (0.30 + 0.30 * k) + (1 - prof) * H * (0.30 - 0.08 * k), 0, H - 1).astype(int)
            m = rows >= top[None, :]
            depth = np.clip((rows - top[None, :]) / (H * 0.35), 0, 1)
            grad = np.convolve(np.gradient(top.astype(F32)), np.ones(5) / 5, "same")
            litv = 0.5 + 0.5 * np.tanh(grad * sundir / (H * 0.004))
            lit = np.clip(litv[None, :] + 0.45 * (facet - 0.5) - 0.25 * depth, 0, 1)
            # snow: a cap near the summit, then streaks down the gullies, thinning lower down
            cap = H * (0.04 + 0.08 * (fbm1d(xp * 5 + i, r, 3) + 1) / 2)
            is_snow = ((rows - top[None, :]) < cap[None, :]) | (streak > 0.47 + 0.4 * depth)
            col = np.where(is_snow[..., None],
                           snow_dark + (snow_lit - snow_dark) * lit[..., None] ** 1.4,
                           rock_dark + (rock_lit - rock_dark) * (lit[..., None] * 0.7))
            haze = 0.82 * (1 - k) ** 1.5                                    # far ranges melt into the night air
            col = col + (far - col) * haze
            bg[m] = col[m]
            ch[m], fg[m] = SPACE, 0
            tm = m & (tex < 0.02 + 0.05 * k)
            ch[tm] = codes(".:'")[r.integers(0, 3, int(tm.sum()))]
            fg[tm] = np.clip(col[tm] * 1.3, 0, 1)
            nxt, prv = np.r_[top[1:], top[-1]], np.r_[top[0], top[:-1]]
            rg = np.where(nxt < top, ord("/"), np.where(nxt > top, ord("\\"), ord("_")))
            rg = np.where((prv > top) & (nxt > top), ord("^"), rg)
            ridge = np.clip(col[top, cx] * 1.25 + snow_lit * 0.12 * litv[:, None], 0, 1)
            ch[top, cx], fg[top, cx] = rg, ridge
            # cloud pools in the low ground in front of each far range (a band, not a blanket)
            if k < 0.9:
                p = np.percentile(top, 70)
                band = np.clip((rows - p) / (H * 0.06), 0, 1) * np.clip(1 - (rows - p - H * 0.16) / (H * 0.08), 0, 1)
                fogm = np.where(m, np.maximum(fogm, band * (1 - k) * 0.75), fogm)
            mask |= m

        # ── the ledge: a snow shelf with dark boulders, at the bottom, where the fire burns ──
        ly = int(H * 0.86)
        ltop = (ly + H * 0.025 * fbm1d(xp * 3, r, 3)).astype(int)
        lm = rows >= ltop[None, :]
        dr = np.clip((rows - ly) / max(1, H - ly), 0, 1)
        sh = value_noise2(r, H, W, 25, 3, 0.6)
        ledge = snow_dark + (snow_lit - snow_dark) * np.clip(0.35 + 0.5 * sh - 0.2 * dr, 0, 1)[..., None]
        bg[lm] = (ledge * 0.85)[lm]
        ch[lm], fg[lm] = SPACE, 0
        lt = lm & (tex < 0.06)
        ch[lt] = codes(".,'")[r.integers(0, 3, int(lt.sum()))]
        fg[lt] = np.clip(bg[lt] * 1.2, 0, 1)
        ch[ltop, cx], fg[ltop, cx] = ord("_"), np.clip(snow_lit * 0.9, 0, 1)
        for _ in range(int(r.integers(3, 6))):                              # boulders, snow piled on their tops
            bxp, byp = r.uniform(0, W), r.uniform(ly + (H - ly) * 0.35, H + (H - ly) * 0.2)
            rx, ry = r.uniform(0.03, 0.06) * H * a / 2, r.uniform(0.025, 0.045) * H
            u, v = (self.cols - bxp) / rx, (rows - byp) / ry
            e = (u ** 2 + v ** 2 < 1) & lm & (v < 0.5)                    # the lower part is buried
            capm = e & (v < -0.45 + 0.15 * np.sin(u * 5))
            side = np.clip(0.5 + 0.5 * u * sundir, 0, 1)
            rockc = rock_dark + (rock_lit - rock_dark) * (side * 0.8)[..., None]
            bg[e] = rockc[e]
            bg[capm] = (snow_dark + (snow_lit - snow_dark) * (0.55 + 0.45 * side)[..., None])[capm]
            ch[e] = SPACE
        fogm[lm] = 0
        mask |= lm
        self.ledge_top = ltop

        # ── the fire ──
        self.fx = W / 2 + self.fire_side * W * self.u(0.08, 0.18)
        self.fb = int(min(H - 2, ly + (H - ly) * 0.55))
        self.fw = max(6, int(0.26 * H * a / 2))
        self.fh = max(6, int(H * 0.24))
        self.heat = np.zeros((self.fh + 1, self.fw), F32)
        xs = np.arange(self.fw)
        self.fprof = np.exp(-((xs - self.fw / 2) / (self.fw * 0.26)) ** 2)
        self.fcols, self.frows = xs, np.arange(self.fh)[:, None]
        wood = self.lit(rgb(110, 70, 40), 1.6)
        for j in range(max(1, sy)):
            y = self.fb + j
            if y < H:
                x0, x1 = max(0, int(self.fx - self.fw * 0.55)), min(W, int(self.fx + self.fw * 0.55))
                ch[y, x0:x1], fg[y, x0:x1], bg[y, x0:x1] = ord("="), wood, wood * 0.5
        stone = self.lit(rgb(120, 116, 112), 1.4)
        for t_ in np.linspace(0.15, math.pi - 0.15, 9):
            y = int(self.fb + sy * 0.5 + math.sin(t_) * sy * 1.2)
            x_ = int(self.fx + math.cos(t_) * self.fw * 0.78)
            if 0 <= y < H and 0 <= x_ < W:
                ch[y, x_:x_ + sx], fg[y, x_:x_ + sx] = ord("o"), stone
        dx = (self.cols - self.fx) / (H * a / 2)
        dy = (rows - (self.fb - self.fh * 0.3)) / (H / 2)
        dist = np.hypot(dx, dy)
        self.Lmap = np.exp(-dist / 0.42).astype(F32)
        self.Lair = np.exp(-dist / 0.16).astype(F32)
        n = 60
        self.ex, self.ey, self.evy, self.life = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        self.eph = r.uniform(0, TAU, n)

        # ── flowers: two or three small clusters growing from cracks of bare rock in the firelight ──
        #    edelweiss (white stars) and gentians (deep blue) need dark ground behind them to be seen
        fl = []
        earth = rgb(30, 26, 26)
        for ci in range(int(r.integers(2, 4))):
            side = self.pick([-1, 1])
            cxp = self.fx + side * self.fw * r.uniform(1.6, 3.6)
            if not (W * 0.04 < cxp < W * 0.96):
                continue
            cyp = int(ltop[int(cxp)] + (H - ltop[int(cxp)]) * r.uniform(0.3, 0.5))
            prx, pry = r.uniform(3.0, 4.2) * sx * 3, sy * r.uniform(5.0, 6.0)          # tall enough to back the heads
            u, v = (self.cols - cxp) / prx, (rows - cyp) / pry
            rock = (u ** 2 + (v * 1.15) ** 2 < 1 + 0.12 * np.sin(u * 5 + ci)) & lm & (v < 0.55)
            side = np.clip(0.5 + 0.5 * u * sundir - 0.3 * v, 0, 1)
            bg[rock] = (rock_dark + (rock_lit - rock_dark) * (side * 0.9)[..., None])[rock]
            ch[rock] = SPACE
            capm = rock & (v < -0.55 + 0.12 * np.sin(u * 6 + ci))          # snow piled on top
            bg[capm] = (snow_dark + (snow_lit - snow_dark) * (0.5 + 0.5 * side)[..., None])[capm]
            soil = lm & (np.abs(u) < 0.95) & (v >= 0.35) & (v < 0.75 + 0.1 * np.sin(u * 7))   # bare earth at its foot
            bg[soil] = (earth * (1.1 - 0.3 * np.abs(u)[..., None]))[soil]
            ch[soil] = np.where(r.random(int(soil.sum())) < 0.3, ord(","), SPACE)
            fg[soil] = rgb(64, 80, 56)
            for _ in range(int(r.integers(3, 6))):
                x = cxp + r.uniform(-0.8, 0.8) * prx
                kind = int(r.random() < 0.55)                              # 1 = edelweiss, 0 = gentian
                fl.append((x, int(cyp + r.uniform(0.5, 0.7) * pry), int(sy * r.uniform(2.6, 4.4)), kind,
                           r.uniform(0, TAU)))
        self.flowers = fl

        # ── the stray ray: from a gap in the clouds, slanting down across the peaks ──
        tgt_x = (self.fx - W / 2) / (H * a / 2)
        tgt_y = (self.fb - H / 2) / (H / 2)
        sxp, syp = tgt_x + self.pick([-1, 1]) * r.uniform(0.9, 1.4), -1.1
        dxr, dyr = tgt_x - sxp, tgt_y - syp
        nrm = math.hypot(dxr, dyr)
        dxr, dyr = dxr / nrm, dyr / nrm
        along = (X - sxp) * dxr + (Y - syp) * dyr
        perp = np.abs((X - sxp) * dyr - (Y - syp) * dxr)
        soft = np.exp(-(perp / (0.10 + along * 0.10)) ** 2) * 0.6 + np.exp(-(perp / (0.04 + along * 0.05)) ** 2) * 0.4
        self.beam = (soft * (0.35 + 0.65 * np.exp(-np.maximum(along, 0) / 4.0)) * (along > 0)).astype(F32)
        self.motes = np.nonzero((r.random((H, W)) < 0.004) & (self.beam > 0.35))

        # snow: as many flakes as a storm needs; calmer weather uses a share of them
        ns = max(20, int(W * H / 18 * self.dens))
        self.sx_, self.sy_ = r.uniform(0, W, ns), r.uniform(0, H, ns)
        self.sv, self.sph_ = r.uniform(0.6, 1.4, ns), r.uniform(0, TAU, ns)
        self.t_ch, self.t_fg, self.t_bg, self.t_mask = ch, fg, bg, mask
        self.fogm = fogm
        self.weather = None

    def _look(self, c):
        """Ease the weather towards the track's (cover, snow, haze, wind), over a few seconds."""
        tgt = WEATHER_LOOK.get(getattr(c, "weather", "snowfall"), WEATHER_LOOK["snowfall"])
        k = min(1.0, c.dt * 0.4)
        for i, key in enumerate(("cover", "snow", "haze", "wind")):
            self.look[key] += (tgt[i] - self.look[key]) * k
        return self.look


class SnowCamp(Campfire):
    """The campfire, always at night and in the snow (the mood scenes' second place)."""

    tods = ("night",)

    def variant(self):
        self.tent_side = self.pick([-1, 1])
        self.wkind = "snow"


def _cap(yy, xx, a, b, r):
    """Pixels within r of the segment a→b ((y, x) points)."""
    (ay, ax), (by, bx) = a, b
    dy, dx = by - ay, bx - ax
    L2 = dy * dy + dx * dx + 1e-9
    u = np.clip(((yy - ay) * dy + (xx - ax) * dx) / L2, 0, 1)
    return (yy - ay - u * dy) ** 2 + (xx - ax - u * dx) ** 2 <= r * r


# a gallop, key poses (degrees from straight down, + = forward): front upper, front lower, rear upper,
# rear lower, body lift (sprite pixels)
GALLOP = [(58, 85, -48, -15, -1.5), (25, 5, -20, -45, 0.0), (-15, -55, 18, -5, 1.0), (-35, -80, 40, 20, 0.0)]


SPR_H, SPR_W, SPR_F = 56, 64, 8


def horse_sprites(nf=SPR_F, kind=0):
    """A galloping horse and rider seen from the side, facing right, one silhouette per phase of the stride.
    uint8 (nf, SPR_H, SPR_W): 0 empty, 1 the near side, 2 the far legs (darker), 3 the top edge (lit by the sky).
    kind 0: a lance; 1: a sword held high; 2: a banner streaming back."""
    yy, xx = np.mgrid[0:SPR_H, 0:SPR_W].astype(F32)
    out = np.zeros((nf, SPR_H, SPR_W), np.uint8)
    K = len(GALLOP)

    def pose(f):
        q = f / nf * K
        i, w = int(q) % K, q - int(q)
        a, b = GALLOP[i], GALLOP[(i + 1) % K]
        return [x + (y - x) * w for x, y in zip(a, b)]

    def leg(hip, up, lo, r):
        knee = (hip[0] + 10 * math.cos(math.radians(up)), hip[1] + 10 * math.sin(math.radians(up)))
        hoof = (knee[0] + 10 * math.cos(math.radians(lo)), knee[1] + 10 * math.sin(math.radians(lo)))
        return _cap(yy, xx, hip, knee, r) | _cap(yy, xx, knee, hoof, r * 0.75)

    for f in range(nf):
        fu, fl, ru, rl, lift = pose(f)
        far = pose((f + 1) % nf)
        o = 8 + lift
        near = np.zeros((SPR_H, SPR_W), bool)
        near |= ((xx - 33) / 14.5) ** 2 + ((yy - (27 + o)) / 7.5) ** 2 <= 1          # body
        near |= _cap(yy, xx, (25 + o, 44), (14 + o, 51), 4.2)                         # neck
        near |= _cap(yy, xx, (13 + o, 51), (17 + o, 60), 3.0)                         # head
        near |= _cap(yy, xx, (11 + o, 50), (7 + o, 49), 1.0)                          # ear
        for k in range(3):                                                            # the mane, streaming
            near |= _cap(yy, xx, (15 + 3 * k + o, 47 - 2 * k), (12 + 3 * k + o + math.sin(f + k), 41 - 3 * k), 0.8)
        wave = 2.2 * math.sin(TAU * f / nf)
        near |= _cap(yy, xx, (24 + o, 19), (20 + o + wave, 6), 2.3)                   # tail
        near |= leg((30 + o, 43), fu, fl, 2.3) | leg((29 + o, 23), ru, rl, 2.6)
        farm = leg((30 + o, 41), far[0], far[1], 2.0) | leg((29 + o, 25), far[2], far[3], 2.3)
        # the rider: leaning into the charge
        near |= _cap(yy, xx, (21 + o, 31), (9 + o, 35), 2.8)                          # torso
        near |= ((xx - 37) ** 2 + (yy - (5.5 + o)) ** 2) <= 2.6 ** 2                 # head
        near |= _cap(yy, xx, (3 + o, 37), (0 + o, 38), 0.8)                           # helmet spike
        near |= _cap(yy, xx, (20 + o, 32), (26 + o, 37), 1.6)                         # leg on the flank
        if kind == 0:
            near |= _cap(yy, xx, (16 + o, 22), (8 + o, 63), 0.7)                      # lance, couched
        elif kind == 1:
            near |= _cap(yy, xx, (12 + o, 34), (6 + o, 41), 1.1)                      # arm
            near |= _cap(yy, xx, (6 + o, 41), (-6 + o + 2 * math.sin(f), 47), 0.8)    # sword raised
        else:
            near |= _cap(yy, xx, (20 + o, 30), (-4 + o, 27), 0.7)                     # banner pole
            flap = np.sin(xx * 0.5 + f * 1.6) * 1.5
            near |= (xx > 13) & (xx < 27) & (yy > 1 + o + flap) & (yy < 9 + o + flap - (27 - xx) * 0.15)
        spr = np.where(near, 1, np.where(farm, 2, 0)).astype(np.uint8)
        rim = near & ~np.r_[np.zeros((1, SPR_W), bool), near[:-1]]
        spr[rim] = 3
        out[f] = spr
    return out


class War(Scene):
    """A battle under a blood-red sky: mountains on the horizon with walls and towers along their ridges, a
    plain where riders charge at the gallop, volleys of arrows arcing over, flaming catapult shots bursting on
    the walls. The music drives it: the riders' stride follows the beat, volleys fly with the beats, chord
    changes and big blows fire the catapults, and everything grows with the battle (info['x'], 0 → 1).

    The moving parts are simulated in step() on the host as small arrays of launch parameters; arrows and
    projectiles fly on closed-form parabolas."""

    base_name = "war"
    NH, NA, NB, NE = 18, 320, 24, 32          # riders, arrows, catapult shots, explosions (fixed sizes)

    def init(self):
        self.name = "battlefield"

    def setup(self, W, H):
        r, a = self.rng, self.a
        rows, cols = self.rows, self.cols
        X = self.X
        hr = self.hr = int(H * 0.62)                          # where the plain meets the mountains' feet
        self.G = H * 0.9                                      # gravity, rows/s² (physics in row units: x/a)
        # ── the sky: black-red overhead to burning orange at the horizon, a swollen sun sinking ──
        tt = np.clip(rows[:, :1] / max(1, hr), 0, 1.2)
        sky = stops(np.clip(tt, 0, 1), [(0, (22, 2, 4)), (0.35, (78, 8, 10)), (0.7, (168, 26, 16)),
                                        (0.9, (226, 72, 28)), (1, (250, 120, 48))])
        sky = np.broadcast_to(sky, (H, W, 3)).copy()
        self.sun = (r.uniform(W * 0.25, W * 0.75), hr * r.uniform(0.42, 0.55), H * 0.11)
        sd = np.hypot((cols - self.sun[0]) / a, rows - self.sun[1]) / self.sun[2]
        sky += rgb(255, 60, 30) * (np.exp(-np.maximum(sd - 1, 0) * 0.9) * 0.45)[..., None]
        disc = sd < 1
        sky[disc] = (rgb(255, 92, 52) * (0.8 + 0.2 * np.clip(1 - sd, 0, 1))[..., None])[disc]
        self.sky_bg = np.clip(sky, 0, 1).astype(F32)
        self.smoke = value_noise2(r, H, W, 3, 5, 0.6)          # drifting smoke, scrolls sideways
        self.smoke2 = value_noise2(r, H, W, 7, 3, 0.5)
        # ── the land: mountain ranges, the nearest with a wall and towers along its ridge; the plain ──
        xp = (np.arange(W) - W / 2) / (H * a / 2)
        ch, fg, bg = np.full((H, W), SPACE, "<u4"), np.zeros((H, W, 3), F32), np.zeros((H, W, 3), F32)
        mask = np.zeros((H, W), bool)
        tex = r.random((H, W))
        rim_c = rgb(210, 60, 30)
        ranges = [((132, 40, 36), 0.30, 0.18, 0.55), ((70, 20, 18), 0.38, 0.17, 0.3), ((34, 10, 10), 0.44, 0.16, 0.0)]
        tops = []
        for i, (col, y0, amp, haze) in enumerate(ranges):
            npk = int((xp[-1] - xp[0]) * (1.2 + 0.6 * i)) + 3
            px_, ph_ = r.uniform(xp[0] - 0.3, xp[-1] + 0.3, npk), r.uniform(0.4, 1.0, npk)
            ps_ = r.uniform(1.0, 2.0, npk)
            prof = np.max(ph_[:, None] - np.abs(xp[None, :] - px_[:, None]) * ps_[:, None], axis=0)
            prof = np.clip(prof + 0.06 * fbm1d(xp * 6 + i * 2.3, r, 5, 0.55), 0, 1)
            top = np.clip(H * y0 + (1 - prof) * H * amp, 0, hr - 2).astype(int)
            m = (rows >= top[None, :]) & (rows < hr + 2)
            depth = np.clip((rows - top[None, :]) / (H * 0.2), 0, 1)
            c3 = rgb(*col) * (1 - 0.35 * depth[..., None])
            bg[m], ch[m], fg[m] = c3[m], SPACE, 0
            tm = m & (tex < 0.03 + 0.03 * i)
            ch[tm] = codes(".:'")[r.integers(0, 3, int(tm.sum()))]
            fg[tm] = np.clip(c3[tm] * 1.5, 0, 1)
            cx = np.arange(W)
            ridge = np.clip(rgb(*col) + rim_c * (0.35 - 0.1 * i), 0, 1)
            ch[top, cx], fg[top, cx] = ord("_"), ridge
            bg[top, cx] = bg[top, cx] * 0.6 + ridge * 0.4
            mask |= m
            tops.append(top)
        # the wall: along the nearest ridge, a crenellated band; towers at intervals; lit windows
        top = tops[-1]
        hw = max(2, int(H * 0.022))
        stone, stone_lit = rgb(26, 9, 9), rgb(92, 30, 22)
        merl = max(1, int(round(a * hw * 0.6)))
        walls = np.zeros((H, W), bool)
        wy = np.convolve(np.r_[[top[0]] * 6, top, [top[-1]] * 6], np.ones(13) / 13, "valid")[:W].astype(int)
        for x in range(W):
            y1 = wy[x] + 1
            y0 = wy[x] - hw - (1 if (x // merl) % 2 == 0 else 0)
            walls[max(0, y0):max(0, y1 + 2), x] = True
        self.towers = []
        nt = int(r.integers(5, 8))
        for k in range(nt):
            tx = int((k + r.uniform(0.25, 0.75)) * W / nt)
            th, tw = int(H * r.uniform(0.08, 0.14)), max(3, int(a * H * r.uniform(0.016, 0.026)))
            ty = wy[min(W - 1, tx)]
            x0, x1 = max(0, tx - tw), min(W, tx + tw)
            walls[max(0, ty - th):ty + 2, x0:x1] = True
            for x in range(x0, x1):                                                    # crenellations
                if ((x - x0) // max(1, merl)) % 2 == 0:
                    walls[max(0, ty - th - 1), x] = True
            for _ in range(int(r.integers(1, 4))):                                     # windows burning inside
                wy_, wx_ = ty - int(th * r.uniform(0.3, 0.8)), int(r.uniform(x0 + 1, max(x0 + 2, x1 - 1)))
                self.towers.append(("win", wy_, wx_))
            self.towers.append(("top", ty - th, tx))
        wm = walls & (rows < hr + 2)
        litw = np.clip(1.2 - (rows - (wy[None, :] - hw * 4)) / (hw * 6), 0, 1)
        wc = stone + (stone_lit - stone) * litw[..., None] * 0.5
        bg[wm], ch[wm] = wc[wm], SPACE
        brick = wm & ((rows + (cols // max(1, int(a * 2))) % 2) % 2 == 0) & (tex < 0.25)
        ch[brick], fg[brick] = ord("="), np.clip(wc[brick] * 1.8, 0, 1)
        edge = wm & ~np.r_[np.zeros((1, W), bool), wm[:-1]]
        bg[edge] = np.clip(bg[edge] + rim_c * 0.45, 0, 1)
        for kind, y, x in self.towers:
            if kind == "win" and 0 <= y < H:
                ch[y, x], fg[y, x], bg[y, x] = ord("'"), rgb(255, 190, 80), rgb(150, 50, 10)
        mask |= wm
        self.walls = wm
        self.wall_y = wy
        # the plain: scorched ground, darker towards you, littered with spears and broken things
        pm = rows >= hr
        dp = np.clip((rows - hr) / max(1, H - hr), 0, 1)
        n = value_noise2(r, H, W, 9, 3, 0.6)
        ground = stops(dp, [(0, (26, 8, 7)), (0.2, (60, 20, 14)), (0.6, (48, 16, 12)), (1, (26, 9, 7))])
        ground = ground * (0.75 + 0.5 * n)[..., None]
        bg[pm], ch[pm], fg[pm] = ground[pm], SPACE, 0
        gt = pm & (tex < 0.08)
        ch[gt] = codes(".,'`")[r.integers(0, 4, int(gt.sum()))]
        fg[gt] = np.clip(ground[gt] * 1.6, 0, 1)
        for _ in range(int(W * 0.06)):                                                # spears stuck in the ground
            x, y = int(r.uniform(0, W)), int(r.uniform(hr + (H - hr) * 0.2, H))
            L = int((H - hr) * 0.06 * (0.5 + (y - hr) / max(1, H - hr)))
            for j in range(L):
                yy = y - j
                xx = int(x + j * 0.3 * a)
                if 0 <= yy < H and 0 <= xx < W:
                    ch[yy, xx], fg[yy, xx] = ord("/"), rgb(40, 22, 16)
        mask |= pm
        dust = np.exp(-np.abs(rows - hr) / (H * 0.035)) * (0.6 + 0.4 * n)        # dust and smoke where they meet
        bg += (rgb(120, 40, 24) - bg) * (dust * 0.55)[..., None]
        self.t_ch, self.t_fg, self.t_bg, self.t_mask = ch, fg, bg, mask
        # fires: behind the walls (the city burning) and scattered over the plain
        fl = []
        for _ in range(int(r.integers(3, 6))):
            x = r.uniform(0, W)
            fl.append((x / a, wy[int(x)] - hw, H * r.uniform(0.05, 0.09)))
        for _ in range(int(r.integers(4, 8))):
            fl.append((r.uniform(0, W) / a, r.uniform(hr + (H - hr) * 0.15, H * 0.97), H * r.uniform(0.03, 0.06)))
        self.fires = np.array(fl, F32)                         # (x phys, y, radius)
        self.glow = np.zeros((H, W), F32)
        for fx, fy, fr in self.fires:
            self.glow += np.exp(-np.hypot(cols / a - fx, (rows - fy) * 1.4) / fr)
        self.glow = np.clip(self.glow, 0, 2).astype(F32)
        self.fire_ph = r.uniform(0, TAU, len(self.fires)).astype(F32)
        # what the catapults aim at, and where the defenders shoot from
        self.targets = np.array([(x / a, wy[min(W - 1, max(0, int(x)))] - hw * r.uniform(0, 1.5))
                                 for x in r.uniform(W * 0.05, W * 0.95, 24)], F32)
        self.sprites = np.stack([horse_sprites(kind=k) for k in range(3)])      # (3, F, SPR_H, SPR_W)
        # summed-area tables (any ink, near side): a scaled rider samples its exact coverage, not one pixel
        sat = np.zeros((3, SPR_F, 2, SPR_H + 1, SPR_W + 1), F32)
        sat[:, :, 0, 1:, 1:] = (self.sprites > 0).cumsum(2).cumsum(3)
        sat[:, :, 1, 1:, 1:] = ((self.sprites == 1) | (self.sprites == 3)).cumsum(2).cumsum(3)
        self.spr_sat = sat
        # moving parts: launch parameters, simulated in step()
        self.hor = np.zeros((self.NH, 8), F32)      # x (phys), base row, height, frame, dir, depth, kind, active
        self.hor[:, 0] = -1e4
        self.arr_ = np.zeros((self.NA, 8), F32)     # x0, y0, vx, vy, t0, T, fire, active
        self.bal = np.zeros((self.NB, 8), F32)      # x0, y0, vx, vy, t0, T, size, active
        self.exp_ = np.zeros((self.NE, 8), F32)     # x, y, t0, size, active, burn time
        self.last_beat = None
        self.last_chord = 0.0
        self.ash = np.stack([r.uniform(0, W, 600), r.uniform(0, H, 600), r.uniform(0.4, 1.4, 600),
                             r.uniform(0, TAU, 600)], 1).astype(F32)
        self.t_last = None

    # ── simulation (host side, shared with the GPU version) ───────────────────
    def intensity(self, c):
        info = getattr(c, "info", None)
        if info and info.get("war"):
            return float(info.get("x", 0.0))
        return 0.5

    def step(self, c):
        r, H, W, a, t = self.rng, c.H, c.W, self.a, c.t
        x = self.intensity(c)
        info = getattr(c, "info", None) or {}
        bpm = float(info.get("bpm", 160))
        Wp = W / a
        # riders: more of them as the battle grows; the stride follows the beat
        want = 3 + int(round(15 * x))
        for i in range(self.NH):
            h = self.hor[i]
            if h[7] and (h[0] < -0.2 * Wp or h[0] > 1.2 * Wp):
                h[7] = 0
            if not h[7] and i < want and r.random() < c.dt * 1.5:
                d = r.uniform(0, 1) ** 0.7
                dirn = 1.0 if r.random() < 0.85 - 0.3 * x else -1.0
                h[:] = (-0.15 * Wp if dirn > 0 else 1.15 * Wp, self.hr + (H - self.hr) * (0.08 + 0.85 * d),
                        H * (0.06 + 0.16 * d), 0, dirn, d, int(r.integers(3)), 1)
                h[3] = r.uniform(0, 1)                         # stride phase offset
            if h[7]:
                h[0] += h[4] * c.dt * H * (0.25 + 0.55 * h[5]) * bpm / 160
        # arrows: a volley on new beats, thicker and more often as it goes; defenders shoot back from the walls
        beat = float(getattr(c, "beat", t * 2.5))
        nb = int(beat)
        if self.last_beat is None:
            self.last_beat = nb
        if nb != self.last_beat:
            self.last_beat = nb
            if r.random() < 0.35 + 0.55 * x:
                self.volley(c, x, 3 + int(26 * x * r.uniform(0.5, 1.0)), from_walls=r.random() < 0.15 + 0.4 * x)
        live = self.arr_[:, 7] > 0
        self.arr_[live & (t - self.arr_[:, 4] > self.arr_[:, 5]), 7] = 0
        # catapults: on chord changes, big blows, and at a rate that grows with the battle
        ch_ = float(getattr(c, "chord", 0.0))
        big = ch_ > self.last_chord + 0.3
        self.last_chord = ch_
        if big or r.random() < c.dt * (0.15 + 0.9 * x) or (c.kick > 0.97 and r.random() < 0.05 + 0.2 * x):
            self.launch_ball(c, x)
        for b in self.bal:
            if b[7] and t - b[4] >= b[5]:                      # it lands: an explosion where it hits
                b[7] = 0
                tau = b[5]
                self.explode(b[0] + b[2] * tau, b[1] + b[3] * tau + 0.5 * self.G * tau * tau, t, b[6])
        e = self.exp_
        e[(e[:, 4] > 0) & (t - e[:, 2] > e[:, 5]), 4] = 0
        # ash drifting down, embers drifting up
        dt = c.dt
        s = self.ash
        s[:, 1] += s[:, 2] * dt * H * 0.03 * (1 - 2 * (s[:, 2] > 1.25))
        s[:, 0] += (0.4 + 0.6 * x) * dt * H * 0.08 * s[:, 2] + np.sin(t + s[:, 3]) * 0.3
        s[:, 0] %= W
        s[:, 1] %= H

    def volley(self, c, x, n, from_walls=False):
        r, t, H, W, a = self.rng, c.t, c.H, c.W, self.a
        free = np.nonzero(self.arr_[:, 7] == 0)[0][:n]
        Wp = W / a
        side = 1.0 if r.random() < 0.6 else -1.0
        for i in free:
            T = r.uniform(1.4, 2.3)
            if from_walls:                                     # down from the towers onto the plain
                k = int(r.integers(len(self.targets)))
                x0, y0 = self.targets[k] + (r.uniform(-1, 1) * H * 0.02, -H * 0.02)
                x1, y1 = r.uniform(0.05, 0.95) * Wp, r.uniform(self.hr, H)
                T *= 0.8
            else:                                              # from one army over the field at the other
                x0 = (r.uniform(-0.1, 0.25) if side > 0 else r.uniform(0.75, 1.1)) * Wp
                y0 = r.uniform(self.hr + (H - self.hr) * 0.3, H * 1.05)
                x1 = x0 + side * r.uniform(0.35, 0.75) * Wp
                y1 = r.uniform(self.hr, H)
            vx = (x1 - x0) / T
            vy = (y1 - y0 - 0.5 * self.G * T * T) / T
            fire = 1.0 if r.random() < 0.1 + 0.35 * x else 0.0
            self.arr_[i] = (x0, y0, vx, vy, t + r.uniform(0, 0.25), T, fire, 1)

    def launch_ball(self, c, x):
        r, t, H, W, a = self.rng, c.t, c.H, c.W, self.a
        free = np.nonzero(self.bal[:, 7] == 0)[0]
        if not len(free):
            return
        Wp = W / a
        if r.random() < 0.75:                                  # from the siege lines at the walls
            k = int(r.integers(len(self.targets)))
            x1, y1 = self.targets[k]
            x0 = x1 + r.choice([-1, 1]) * r.uniform(0.3, 0.7) * Wp
            y0 = H * 1.05
        else:                                                  # from the walls down onto the plain
            k = int(r.integers(len(self.targets)))
            x0, y0 = self.targets[k]
            x1, y1 = r.uniform(0.05, 0.95) * Wp, r.uniform(self.hr + (H - self.hr) * 0.1, H * 0.95)
        T = r.uniform(2.2, 3.4)
        vx = (x1 - x0) / T
        vy = (y1 - y0 - 0.5 * self.G * T * T) / T
        self.bal[free[0]] = (x0, y0, vx, vy, t, T, r.uniform(0.7, 1.3), 1)

    def explode(self, x, y, t, size):
        e = self.exp_
        free = np.nonzero(e[:, 4] == 0)[0]
        i = free[0] if len(free) else int(np.argmin(e[:, 2]))
        e[i] = (x, y, t, size, 1, self.rng.uniform(4.0, 9.0), self.rng.uniform(0, 1000), 0)


ALL_SCENES = [Seaside, Lighthouse, Mountains, Lake, Forest, Campfire, Summit, SnowCamp, War]


SCENES = [Summit, SnowCamp]          # the rotation: the mood is a lonely fire high on a snowy mountain
