"""claude fm's landscapes on the GPU: each `G*` class wraps a host scene from scenes.py, uploads its static
layers once and draws it every frame with kernels (its own below, shared ones from kor.fx)."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from kor.fx import (k_embers, k_fire_base, k_fire_draw, k_fire_step, k_fog, k_particles, k_reflect, k_smoke,
                    k_smoke_wind, k_snow)
from kor.glyphs import INK as GLYPH_INK
from kor.gpu import clip3, comp, fmodp, hash01, k_dots, ramp
from kor.launch import launch
from kor.scene import GScene as _GScene
from kor.util import F32, rgb

from . import scenes as cf

SPACE = 32


# ink of the glyphs scenes use (kor.glyphs.INK, × 1.7, clipped)
INK = {g: float(min(1.0, GLYPH_INK[ord(g)] * 1.7)) for g in " .,:;'`\"-_~=+*^/\\|()<>oO@#%$&"}


@wp.kernel
def k_sky(out: wp.array2d(dtype=wp.vec3), sky_bg: wp.array2d(dtype=wp.vec3),
          bd: wp.array2d(dtype=float), crater: wp.array2d(dtype=wp.int32),
          X: wp.array2d(dtype=float), Y: wp.array2d(dtype=float),
          star_id: wp.array2d(dtype=wp.int32), sph: wp.array(dtype=float), sbr: wp.array(dtype=float),
          hr: int, t: float, cl0: float, cl1: float, cloudy: float, cloud_col: wp.vec3,
          body: int, body_col: wp.vec3, stars: float, aurora: int,
          aur_x: wp.array(dtype=float), aur_c: wp.array(dtype=wp.vec3)):
    i, j = wp.tid()
    bg = sky_bg[i, j]
    fg = bg
    a = float(0.0)
    x = X[i, j]
    y = Y[i, j]
    d = bd[i, j]
    hrf = float(hr)
    if aurora == 1:                                       # curtains of light, folds drifting
        yc = hrf * (0.28 + 0.10 * wp.sin(x * 1.1 + t * 0.08) + 0.05 * wp.sin(x * 3.3 - t * 0.13))
        dd = (float(i) - yc) / hrf
        band = float(0.0)
        if dd < 0.0:
            band = wp.exp(-(dd / 0.05) * (dd / 0.05))
        else:
            band = wp.exp(-dd / 0.22)
        rays = 0.72 + 0.28 * wp.sin(x * 9.0 + wp.sin(x * 3.0 + t * 0.3) * 2.5 + t * 0.5)
        inten = wp.clamp(band * rays * (0.75 + 0.25 * wp.sin(t * 0.4 + x * 1.7)), 0.0, 1.0)
        ac = ramp(aur_x, aur_c, wp.clamp(-dd * 4.0 + 0.4, 0.0, 1.0))
        bg = bg + ac * inten * 0.5
        fg = bg
        if inten > 0.55 and rays > 0.97:
            fg = clip3(ac * (0.5 + 0.6 * inten))
            a = 0.51
    sid = star_id[i, j]
    if sid >= 0 and stars > 0.0:                          # twinkling stars add light
        tw = 0.55 + 0.45 * wp.sin(t * 1.8 + sph[sid])
        fade = wp.clamp(1.0 - float(i) / (hrf * 0.9), 0.15, 1.0)
        fg = clip3(bg + wp.vec3(0.88, 0.9, 1.0) * (tw * sbr[sid] * fade * stars))
        if tw > 0.88:
            a = 0.61
        else:
            a = 0.17
    if body > 0 and d < 1.0:                              # sun or moon
        bg = body_col * 0.92
        fg = body_col
        a = 1.0
        if body == 2 and crater[i, j] == 1:
            fg = body_col * 0.8
            a = 0.765
    n = 0.5 + 0.25 * (wp.sin(x * cl0 + t * 0.02 + wp.sin(y * 5.0) * 0.5) + wp.sin(x * cl1 * 1.9 - t * 0.03 + y * 7.0))
    yy = float(i) / wp.max(hrf - 1.0, 1.0)
    cloud = wp.clamp((n - cloudy) / 0.22, 0.0, 1.0) * wp.clamp(1.25 - yy, 0.0, 1.0) * 0.85
    if cloud > 0.0:                                       # clouds drift in front, lit from the sun/moon side
        ccol = cloud_col * (0.62 + 0.38 * wp.exp(-wp.max(d - 1.0, 0.0) * 0.25))
        bg = bg + (ccol - bg) * cloud
        if cloud > 0.5:
            fg = clip3(ccol * 1.12)
            a = 0.34
    out[i, j] = comp(bg, fg, a)


@wp.kernel
def k_birds(out: wp.array2d(dtype=wp.vec3), bx0: wp.array(dtype=float), by0: wp.array(dtype=float),
            bv: wp.array(dtype=float), bp: wp.array(dtype=float), t: float, W: int, hr: int, sx: int, sy: int):
    b, k = wp.tid()
    x = fmodp(bx0[b] + bv[b] * 30.0 * float(sx) * t, float(W))
    y = by0[b] + wp.sin(t * 0.5 + bp[b]) * float(sy)
    flap = wp.sin(t * 7.0 + bp[b]) > 0.0
    dx = (k - 1) * sx
    yy = int(y)
    if flap and k != 1:
        yy = yy - 1
    xx = int(x) + dx
    if yy >= 0 and yy < hr and xx >= 0 and xx < W:
        out[yy, xx] = wp.vec3(0.14, 0.125, 0.165)


@wp.kernel
def k_sea(out: wp.array2d(dtype=wp.vec3), sea_bg: wp.array2d(dtype=wp.vec3), X: wp.array2d(dtype=float),
          sdep: wp.array(dtype=float), sky_line: wp.array(dtype=wp.vec3), hr: int, sy: int, sx: int,
          k0: float, k1: float, k2: float, t: float, level: float, kick: float,
          body: int, bx: float, mr: float, a: float, refl: wp.vec3):
    r, j = wp.tid()
    i = r + hr
    bg = sea_bg[r, j]
    dp = sdep[r]
    R = float(i) / float(sy)
    x = X[i, j]
    w = (wp.sin(x * k0 / (0.12 + dp) + t * 1.1 + R * 0.9)
         + 0.6 * wp.sin(x * k1 / (0.12 + dp) - t * 1.6 + R * 1.7)
         + 0.3 * wp.sin(x * k2 * 2.0 / (0.12 + dp) + t * 2.3))
    w = w * (0.85 + 0.3 * level + 0.2 * kick)
    ink = float(0.0)
    if w > 1.1:
        ink = 0.476
    elif w > 0.45:
        ink = 0.34
    elif w > 0.0:
        ink = 0.17
    crest = wp.clamp(w / 1.6, 0.0, 1.0)
    fg = bg * (1.3 + 0.9 * crest) + sky_line[j] * 0.25 * crest
    if body > 0:                                          # glittering path under the sun/moon
        width = a * mr * (0.35 + 1.5 * dp)
        wob = 0.6 + 0.4 * wp.sin(t * 3.0 + R * 2.1 + float(j) / float(sx) * 0.3)
        if wp.abs(float(j) - bx) < width * wob and w > 0.2:
            fg = refl * (0.75 + 0.25 * crest)
            bg = bg + refl * 0.18
            if w > 0.9:
                ink = 0.646
            else:
                ink = 0.34
    out[i, j] = comp(bg, clip3(fg), ink)


@wp.kernel
def k_beach(out: wp.array2d(dtype=wp.vec3), shore: wp.array(dtype=float), xp: wp.array(dtype=float),
            stex: wp.array2d(dtype=float), hr: int, H: int, sx: int, sy: int, swash0: float, t: float,
            sand: wp.vec3, horizon: wp.vec3, lum: float, white: wp.vec3):
    r, j = wp.tid()
    i = r + hr
    fi = float(i)
    span = float(H - hr)
    sw = swash0 * (0.5 + 0.5 * wp.sin(t * 0.45 + 0.7 * wp.sin(xp[j] * 1.1)))
    edge = shore[j] + sw                                  # how far up the beach the water reaches now
    wetline = shore[j] + swash0
    c = out[i, j]
    if fi >= edge:
        bg = sand
        if fi < wetline:
            wf = wp.clamp((wetline - fi) / wp.max(swash0, 1.0), 0.0, 1.0)
            bg = sand * (1.0 - 0.28 * wf) + horizon * (0.12 * lum * wf)
        tx = stex[i, j]
        c = bg
        if tx < 0.12:
            m = float(1.2)
            if tx < 0.08:
                m = 0.75
            ink = float(0.17)
            if tx < 0.05:
                ink = 0.374
            c = comp(bg, clip3(bg * m), ink)
    else:
        if fi >= edge - 1.5 * float(sy):                  # foam at the water's edge
            bg = c * 0.5 + white * 0.5
            c = bg
            if wp.sin(float(j) * 0.7 / float(sx) + t * 3.0) > -0.3:
                c = comp(bg, white, 0.476)
        ph = fmodp(t * 0.11 + 0.15 * wp.sin(xp[j] * 0.7), 1.0)
        brow = edge - (1.0 - ph) * span * 0.18             # a breaker rolling in
        if wp.abs(fi - brow) < 0.6 * float(sy) and ph > 0.15:
            bg2 = c * 0.75 + white * 0.25
            c = comp(bg2, white * 0.85, 0.476)
    out[i, j] = c


@wp.kernel
def k_land(out: wp.array2d(dtype=wp.vec3), mask: wp.array2d(dtype=wp.int32), t_bg: wp.array2d(dtype=wp.vec3),
           t_fg: wp.array2d(dtype=wp.vec3), t_ink: wp.array2d(dtype=float)):
    i, j = wp.tid()
    if mask[i, j] == 1:
        out[i, j] = comp(t_bg[i, j], t_fg[i, j], t_ink[i, j])


@wp.kernel
def k_mountains(out: wp.array2d(dtype=wp.vec3), mask: wp.array2d(dtype=wp.int32), t_bg: wp.array2d(dtype=wp.vec3),
                t_fg: wp.array2d(dtype=wp.vec3), t_ink: wp.array2d(dtype=float), fogm: wp.array2d(dtype=float),
                X: wp.array2d(dtype=float), Y: wp.array2d(dtype=float), fogc: wp.vec3, t: float):
    i, j = wp.tid()
    if mask[i, j] == 1:                                   # valley mist slowly breathing
        breathe = 0.5 + 0.5 * wp.sin(t * 0.15 + X[i, j] * 1.4 + Y[i, j] * 2.0)
        b = t_bg[i, j]
        tb = b + (fogc - b) * (fogm[i, j] * breathe * 0.35)
        out[i, j] = comp(tb, t_fg[i, j], t_ink[i, j])


@wp.kernel
def k_forest(out: wp.array2d(dtype=wp.vec3), mask: wp.array2d(dtype=wp.int32), t_bg: wp.array2d(dtype=wp.vec3),
             t_fg: wp.array2d(dtype=wp.vec3), t_ink: wp.array2d(dtype=float), grass: wp.array2d(dtype=wp.int32),
             gcol: wp.array2d(dtype=wp.vec3), gph: wp.array2d(dtype=float), X: wp.array2d(dtype=float),
             t: float, kick: float):
    i, j = wp.tid()
    if mask[i, j] == 1:
        if grass[i, j] == 1:                              # blades catch the light as gusts roll through
            sway = wp.sin(t * 1.5 + X[i, j] * 2.0 + gph[i, j]) + 0.9 * kick
            out[i, j] = comp(t_bg[i, j], clip3(gcol[i, j] * (0.82 + 0.22 * sway)), 0.51)
        else:
            out[i, j] = comp(t_bg[i, j], t_fg[i, j], t_ink[i, j])


@wp.kernel
def k_campfire_land(out: wp.array2d(dtype=wp.vec3), mask: wp.array2d(dtype=wp.int32), t_bg: wp.array2d(dtype=wp.vec3),
                    t_fg: wp.array2d(dtype=wp.vec3), t_ink: wp.array2d(dtype=float), Lmap: wp.array2d(dtype=float),
                    Lair: wp.array2d(dtype=float), flick: float, warm: wp.vec3):
    i, j = wp.tid()
    c = out[i, j]
    if mask[i, j] == 1:                                   # everything near the fire is lit by it
        L = Lmap[i, j] * flick
        fg = clip3(t_fg[i, j] * (1.0 + L * 1.5) + warm * (L * 0.15))
        bg = t_bg[i, j] * (1.0 + L * 1.2) + warm * (L * 0.5)
        c = comp(bg, fg, t_ink[i, j])
    out[i, j] = c + warm * (Lair[i, j] * flick * 0.3)


@wp.kernel
def k_spray(out: wp.array2d(dtype=wp.vec3), px: wp.array(dtype=float), py: wp.array(dtype=float),
            vx: wp.array(dtype=float), vy: wp.array(dtype=float), life: wp.array(dtype=float),
            cy: wp.array(dtype=wp.int32), cx: wp.array(dtype=wp.int32), spawn: int, seed: int,
            dt: float, sx: int, sy: int, W: int, H: int, white: wp.vec3):
    p = wp.tid()
    st = wp.rand_init(seed, p)
    n = cy.shape[0]
    if life[p] <= 0.0:
        if spawn == 1 and n > 0 and wp.randf(st) < 0.2:   # a wave slams the rocks on the kick
            k = int(wp.randf(st) * float(n)) % n
            px[p] = float(cx[k])
            py[p] = float(cy[k])
            vx[p] = wp.randn(st) * 0.35 * float(sx)
            vy[p] = -(0.5 + 0.8 * wp.randf(st)) * float(sy)
            life[p] = 0.6 + 0.4 * wp.randf(st)
        return
    step = dt * 30.0
    vy[p] = vy[p] + 0.06 * float(sy) * step
    px[p] = px[p] + vx[p] * step
    py[p] = py[p] + vy[p] * step
    life[p] = life[p] - dt * 0.9
    x = int(px[p])
    y = int(py[p])
    if life[p] > 0.0 and x >= 0 and x < W and y >= 0 and y < H:
        ink = float(0.17)
        if life[p] > 0.5:
            ink = 0.765
        out[y, x] = comp(out[y, x], white * life[p], ink)


@wp.kernel
def k_coast(out: wp.array2d(dtype=wp.vec3), cy: wp.array(dtype=wp.int32), cx: wp.array(dtype=wp.int32),
            cph: wp.array(dtype=float), t: float, kick: float, white: wp.vec3):
    k = wp.tid()
    f = wp.sin(t * 2.2 + cph[k]) * 0.5 + 0.5 + 0.6 * kick
    if f > 0.75:                                          # surf breaking on the rocks
        y = cy[k]
        x = cx[k]
        bg = out[y, x] * 0.6 + white * 0.4
        ink = float(0.476)
        if f > 1.05:
            ink = 0.765
        out[y, x] = comp(bg, white, ink)


@wp.kernel
def k_beam(out: wp.array2d(dtype=wp.vec3), lx: float, ly: float, a: float, sy: int, t: float, spin: float,
           W: int, H: int, strength: float):
    i, j = wp.tid()
    dx = (float(j) - lx) / a
    dy = float(i) - ly
    span = float(W) / a * 1.1
    tot = float(0.0)
    glare = float(0.0)
    for k in range(2):                                    # two opposite beams on one rotating lamp
        ang = t * spin + float(k) * 3.14159265
        cx = wp.cos(ang)
        g = wp.max(0.0, -wp.sin(ang))
        glare = wp.max(glare, g * g * g * g * g * g * g * g)
        along = dx
        if cx < 0.0:
            along = -dx
        if along > 0.0:
            hw = 0.45 * float(sy) + along * 0.075
            perp = (dy - along * 0.02) / hw
            tot = tot + wp.exp(-perp * perp) * wp.exp(-along / (wp.abs(cx) * span * 0.6 + 1.0)) * (0.3 + 0.7 * wp.abs(cx))
    rr = wp.sqrt(dx * dx + dy * dy)
    flare = wp.exp(-rr / (float(H) * 0.022)) * (0.6 + 1.4 * glare) + wp.exp(-rr / (float(H) * 0.10)) * 0.14 * (0.3 + glare)
    fk = float(0.35)
    if strength > 0.5:
        fk = 0.6
    out[i, j] = out[i, j] + wp.vec3(1.0, 0.94, 0.78) * (tot * 0.45 * strength + flare * fk)


class GScene(_GScene):
    """kor's GScene plus what claude fm's landscapes share: the sky, weather particles and the static
    land layers."""

    def upload_sky(self):
        s, T = self.cpu, self.cpu.T
        H, W = s.size[1], s.size[0]
        self.H, self.W = H, W
        self.X, self.Y = self.arr(s.X.astype(F32)), self.arr(s.Y.astype(F32))
        self.sky_bg = self.arr(s.sky_bg, wp.vec3)
        self.bd = self.arr(s.bd.astype(F32))
        self.crater = self.arr(s.crater.astype(np.int32), wp.int32)
        sid = np.full((H, W), -1, np.int32)
        sid[s.sty, s.stx] = np.arange(len(s.stx), dtype=np.int32)
        self.star_id = self.arr(sid, wp.int32)
        self.sph = self.arr(np.r_[s.sph, 0].astype(F32))
        self.sbr = self.arr(np.r_[s.sbr, 0].astype(F32))
        self.body = {"sun": 1, "moon": 2}.get(T["body"], 0)
        self.body_col = wp.vec3(*(rgb(*T["body_col"]) if T["body"] else rgb(0, 0, 0)))
        self.cloud_col = wp.vec3(*rgb(*T["cloud"]))
        self.aur = self.stops([(0, (60, 255, 150)), (0.5, (60, 220, 200)), (1, (190, 90, 255))])
        nb = len(s.birx)
        self.birds = [self.arr(np.r_[v, 0].astype(F32)) for v in (s.birx, s.biry, s.birv, s.birp)] + [nb]

    def draw_sky(self, c, out):
        s, T = self.cpu, self.cpu.T
        launch(self, k_sky, dim=(s.hr, self.W), device=self.dev,
                  inputs=[out, self.sky_bg, self.bd, self.crater, self.X, self.Y, self.star_id, self.sph, self.sbr,
                          s.hr, c.t, float(s.cl[0]), float(s.cl[1]), float(s.cloudy), self.cloud_col,
                          self.body, self.body_col, float(T["stars"]), int(s.aurora), self.aur[0], self.aur[1]])
        if self.birds[-1]:
            bx, by, bv, bp, nb = self.birds
            launch(self, k_birds, dim=(nb, 3), device=self.dev,
                      inputs=[out, bx, by, bv, bp, c.t, self.W, s.hr, s.sx, s.sy])

    def upload_weather(self):
        wthr = getattr(self.cpu, "weather", None)
        self.part = None
        if wthr is not None:
            kind = {"rain": 0, "snow": 1, "leaves": 2, "petals": 2, "fireflies": 3}[wthr.kind]
            col = getattr(wthr, "col", np.zeros((len(wthr.x), 3), F32))
            self.part = dict(kind=kind, wind=float(wthr.wind), x=self.arr(wthr.x.astype(F32)),
                             y=self.arr(wthr.y.astype(F32)), v=self.arr(wthr.v.astype(F32)),
                             ph=self.arr(wthr.ph.astype(F32)), ang=self.arr(wthr.ang.astype(F32)),
                             col=self.arr(col, wp.vec3), n=len(wthr.x))

    def draw_weather(self, c, out):
        p = self.part
        if p is None:
            return
        self.seed += 1
        s = self.cpu
        launch(self, k_particles, dim=p["n"], device=self.dev,
                  inputs=[out, p["x"], p["y"], p["v"], p["ph"], p["ang"], p["col"], p["kind"], p["wind"], c.t, c.dt,
                          c.kick, float(s.T["lum"]), self.W, self.H, s.sx, s.sy, float(s.a), self.seed])

    def static_layers(self):
        s = self.cpu
        ink = np.clip(GLYPH_INK[np.minimum(s.t_ch, 255)] * 1.7, 0, 1).astype(F32)
        self.mask = self.arr(s.t_mask.astype(np.int32), wp.int32)
        self.t_bg, self.t_fg = self.arr(s.t_bg, wp.vec3), self.arr(s.t_fg, wp.vec3)
        self.t_ink = self.arr(ink)


class GMountains(GScene):
    def upload(self):
        s = self.cpu
        self.upload_sky()
        self.static_layers()
        self.fogm = self.arr(s.fogm.astype(F32))
        self.fogc = wp.vec3(*np.asarray(s.fogc, F32))
        self.water = wp.vec3(*rgb(*s.T["water"]))
        self.fog_col = wp.vec3(*(rgb(*s.T["fog"]) * (0.5 + 0.5 * s.T["lum"])))
        self.upload_weather()

    def draw(self, c, out):
        s = self.cpu
        self.draw_sky(c, out)
        launch(self, k_mountains, dim=(self.H, self.W), device=self.dev,
                  inputs=[out, self.mask, self.t_bg, self.t_fg, self.t_ink, self.fogm, self.X, self.Y, self.fogc, c.t])
        if s.lake:
            launch(self, k_reflect, dim=(self.H - s.wl, self.W), device=self.dev,
                      inputs=[out, s.wl, self.H, self.W, s.sx, s.sy, c.t, self.water])
        if s.wkind == "fog":
            launch(self, k_fog, dim=(self.H, self.W), device=self.dev, inputs=[out, self.X, self.Y, self.H, c.t, 0.35, self.fog_col])
        self.draw_weather(c, out)


class GSea(GScene):
    def upload_sea(self):
        s, T = self.cpu, self.cpu.T
        self.sea_bg = self.arr(s.sea_bg, wp.vec3)
        self.sdep = self.arr(s.sdep[:, 0].astype(F32))
        self.sky_line = self.arr(s.sky_bg[-1], wp.vec3)
        self.refl = wp.vec3(*rgb(*T["refl"]))

    def draw_sea(self, c, out):
        s = self.cpu
        k = s.wk
        launch(self, k_sea, dim=(self.H - s.hr, self.W), device=self.dev,
                  inputs=[out, self.sea_bg, self.X, self.sdep, self.sky_line, s.hr, s.sy, s.sx, float(k[0]), float(k[1]),
                          float(k[2]), c.t, c.level, c.kick, self.body, float(s.bx), float(s.mr), float(s.a), self.refl])


class GSeaside(GSea):
    def upload(self):
        s = self.cpu
        self.upload_sky()
        self.upload_sea()
        self.shore = self.arr(s.shore.astype(F32))
        self.xp = self.arr(s.xp.astype(F32))
        self.stex = self.arr(s.stex.astype(F32))
        self.white = wp.vec3(*np.clip(s.lit(rgb(250, 250, 255), 1.15), 0, 1))
        self.upload_weather()

    def draw(self, c, out):
        s = self.cpu
        self.draw_sky(c, out)
        self.draw_sea(c, out)
        if s.boat:
            self.draw_boat(c, out)
        launch(self, k_beach, dim=(self.H - s.hr, self.W), device=self.dev,
                  inputs=[out, self.shore, self.xp, self.stex, s.hr, self.H, s.sx, s.sy, float(s.swash), c.t,
                          wp.vec3(*s.sand), wp.vec3(*s.horizon_col), float(s.T["lum"]), self.white])
        self.draw_weather(c, out)

    def draw_boat(self, c, out):
        s = self.cpu   # the sprite is tiny: build its pixels on the CPU (which also moves it), write them in one launch
        if getattr(self, "_scratch", None) is None or self._scratch[0].shape != (c.H, c.W):
            self._scratch = (np.full((c.H, c.W), SPACE, "<u4"), np.zeros((c.H, c.W, 3), F32), np.zeros((c.H, c.W, 3), F32))
        ch, fg, bg = self._scratch                       # reused every frame; only the boat's rows are touched
        sub = slice(max(0, s.boat_row - 4 * s.sy), min(c.H, s.boat_row + s.sy))
        s._boat(c, ch, fg, bg)
        ys, xs = np.nonzero(ch[sub] != SPACE)
        if len(ys):
            ys = ys + sub.start
            col = np.where(bg[ys, xs].any(-1)[:, None], bg[ys, xs], fg[ys, xs])   # the masthead lantern has no bg
            self.dots(out, ys, xs, col)
        ch[sub], fg[sub], bg[sub] = SPACE, 0, 0

    def dots(self, out, ys, xs, cols):
        launch(self, k_dots, dim=len(ys), device=self.dev,
                  inputs=[out, self.arr(ys.astype(np.int32), wp.int32), self.arr(xs.astype(np.int32), wp.int32),
                          self.arr(cols.astype(F32), wp.vec3), self.H, self.W])


class GLighthouse(GSea):
    def upload(self):
        s = self.cpu
        self.upload_sky()
        self.upload_sea()
        self.static_layers()
        self.cy, self.cx = self.arr(s.cy.astype(np.int32), wp.int32), self.arr(s.cx.astype(np.int32), wp.int32)
        self.cph = self.arr(s.cph.astype(F32))
        n = 120
        self.spray = [self.arr(np.zeros(n, F32)) for _ in range(5)]
        self.white = wp.vec3(*np.clip(s.lit(rgb(245, 248, 255), 1.8), 0, 1))
        self.fog_col = wp.vec3(*(rgb(*s.T["fog"]) * (0.5 + 0.5 * s.T["lum"])))
        self.kick_prev = 0.0
        self.upload_weather()

    def draw(self, c, out):
        s = self.cpu
        self.draw_sky(c, out)
        self.draw_sea(c, out)
        launch(self, k_land, dim=(self.H, self.W), device=self.dev, inputs=[out, self.mask, self.t_bg, self.t_fg, self.t_ink])
        if len(s.cy):
            launch(self, k_coast, dim=len(s.cy), device=self.dev, inputs=[out, self.cy, self.cx, self.cph, c.t, c.kick, self.white])
        spawn = int(c.kick > 0.9 and self.kick_prev <= 0.9)
        self.kick_prev = c.kick
        self.seed += 1
        px, py, vx, vy, life = self.spray
        launch(self, k_spray, dim=px.shape[0], device=self.dev,
                  inputs=[out, px, py, vx, vy, life, self.cy, self.cx, spawn, self.seed, c.dt, s.sx, s.sy, self.W, self.H, self.white])
        strength = {"night": 1.0, "dusk": 0.75, "sunset": 0.35, "dawn": 0.3}.get(s.tod, 0.3) * (1.5 if s.wkind == "fog" else 1.0)
        launch(self, k_beam, dim=(self.H, self.W), device=self.dev,
                  inputs=[out, float(s.lx), float(s.ly), float(s.a), s.sy, c.t, float(s.spin), self.W, self.H, strength])
        if s.wkind == "fog":
            launch(self, k_fog, dim=(self.H, self.W), device=self.dev, inputs=[out, self.X, self.Y, self.H, c.t, 0.4, self.fog_col])
        self.draw_weather(c, out)


class GForest(GScene):
    def upload(self):
        s = self.cpu
        self.upload_sky()
        self.static_layers()
        self.grass = self.arr(s.gmask.astype(np.int32), wp.int32)
        self.gcol = self.arr(np.broadcast_to(s.gcol, (self.H, self.W, 3)), wp.vec3)
        self.gph = self.arr(s.gph.astype(F32))
        self.upload_weather()

    def draw(self, c, out):
        self.draw_sky(c, out)
        launch(self, k_forest, dim=(self.H, self.W), device=self.dev,
                  inputs=[out, self.mask, self.t_bg, self.t_fg, self.t_ink, self.grass, self.gcol, self.gph, self.X, c.t, c.kick])
        self.draw_weather(c, out)


class GCampfire(GScene):
    def upload(self):
        s = self.cpu
        self.upload_sky()
        self.static_layers()
        self.Lmap, self.Lair = self.arr(s.Lmap.astype(F32)), self.arr(s.Lair.astype(F32))
        self.heat = [self.arr(np.zeros((s.fh + 1, s.fw), F32)) for _ in range(2)]
        self.fprof = self.arr(s.fprof.astype(F32))
        self.flame = self.stops(cf.FLAMES["fire"])
        self.ramp_ink = self.arr(np.array([INK.get(g, 1.0) for g in " .,:;=+*#%@"], F32))
        n = 60
        self.emb = [self.arr(np.zeros(n, F32)) for _ in range(4)] + [self.arr(s.rng.uniform(0, 6.3, n).astype(F32))]
        self.upload_weather()

    def draw(self, c, out):
        s, r = self.cpu, self.cpu.rng
        self.draw_sky(c, out)
        pop, pluck = float(getattr(c, "pop", 0.0)), float(getattr(c, "pluck", 0.0))
        flick = (0.72 + 0.10 * math.sin(c.t * 11.3) * math.sin(c.t * 6.1) + 0.22 * c.kick + 0.08 * c.level
                 + 0.18 * getattr(c, "bass", 0.5) + 0.3 * pop)
        warm, smoke = self.colours(lambda: (wp.vec3(1.0, 150 / 255, 60 / 255),
                                            wp.vec3(*(rgb(120, 118, 126) * s.T["lum"] * 1.4))))
        launch(self, k_campfire_land, dim=(self.H, self.W), device=self.dev,
                  inputs=[out, self.mask, self.t_bg, self.t_fg, self.t_ink, self.Lmap, self.Lair, flick, warm])
        for _ in range(s.sy):                                 # the fire simulation, ping-ponging two grids
            self.seed += 1
            a, b = self.heat
            launch(self, k_fire_base, dim=s.fw, device=self.dev, inputs=[a, self.fprof, s.fh, c.level, max(c.kick, pop), self.seed])
            launch(self, k_fire_step, dim=(s.fh + 1, s.fw), device=self.dev, inputs=[a, b, s.fh, s.fw, self.seed + 7])
            self.heat = [b, a]
        x0 = int(s.fx - s.fw / 2)
        launch(self, k_fire_draw, dim=(s.fh, s.fw), device=self.dev,
                  inputs=[out, self.heat[0], s.fb - s.fh, x0, self.H, self.W, self.flame[0], self.flame[1], self.ramp_ink])
        launch(self, k_smoke, dim=(self.H, self.W), device=self.dev,
                  inputs=[out, float(s.fx), float(s.fb - s.fh * 0.6), float(s.fw), self.H, s.sy, c.t, smoke])
        self.seed += 1
        ex, ey, evy, life, eph = self.emb
        launch(self, k_embers, dim=ex.shape[0], device=self.dev,
                  inputs=[out, ex, ey, evy, life, eph, float(s.fx), float(s.fb), float(s.fw), float(s.fh), c.t, c.dt,
                          float(c.kick + (0.8 * pop + 0.3 * pluck - 0.04) / 0.4), self.seed, s.sx, s.sy, self.W,
                          self.H, self.flame[0], self.flame[1]])
        self.draw_weather(c, out)


@wp.kernel
def k_summit_sky(out: wp.array2d(dtype=wp.vec3), sky_bg: wp.array2d(dtype=wp.vec3),
                 galaxy: wp.array2d(dtype=wp.vec3), bd: wp.array2d(dtype=float), crater: wp.array2d(dtype=wp.int32),
                 star_id: wp.array2d(dtype=wp.int32), sph: wp.array(dtype=float), sbr: wp.array(dtype=float),
                 cl_hi: wp.array2d(dtype=float), hr: int, W: int, t: float, off: int, cover: float,
                 cloud_col: wp.vec3, body_col: wp.vec3, pluck: float):
    i, j = wp.tid()
    bg = sky_bg[i, j] + galaxy[i, j]
    fg = bg
    a = float(0.0)
    sid = star_id[i, j]
    if sid >= 0:                                          # stars; some glint on plucked notes
        tw = 0.55 + 0.45 * wp.sin(t * 1.8 + sph[sid])
        g = float(1.0)
        if fmodp(sph[sid], 1.0) > 0.7:
            g = 1.0 + 0.8 * pluck
        sb = wp.min(tw * sbr[sid] * g, 1.4)
        fg = clip3(bg + wp.vec3(0.88, 0.9, 1.0) * sb)
        if sb > 0.85:
            a = 0.61
        else:
            a = 0.17
    d = bd[i, j]
    if d < 1.0:                                           # the moon
        bg = body_col * 0.92
        fg = body_col
        a = 1.0
        if crater[i, j] == 1:
            fg = body_col * 0.8
            a = 0.765
    n = cl_hi[i, (j + off) % W]                           # clouds scroll with the wind
    yy = float(i) / wp.max(float(hr), 1.0)
    cl = wp.clamp((n - (1.0 - cover)) / 0.22, 0.0, 1.0) * wp.clamp(1.3 - yy, 0.0, 1.0)
    if cl > 0.0:
        ccol = cloud_col * (0.7 + 0.5 * wp.exp(-wp.max(d - 1.0, 0.0) * 0.3))
        bg = bg + (ccol - bg) * cl
        if cl > 0.55:
            fg = clip3(ccol * 1.15)
            a = 0.3
        else:
            fg = fg + (ccol - fg) * cl
    out[i, j] = comp(bg, fg, a)


@wp.kernel
def k_summit_land(out: wp.array2d(dtype=wp.vec3), mask: wp.array2d(dtype=wp.int32), t_bg: wp.array2d(dtype=wp.vec3),
                  t_fg: wp.array2d(dtype=wp.vec3), t_ink: wp.array2d(dtype=float), fogm: wp.array2d(dtype=float),
                  cl_lo: wp.array2d(dtype=float), off: int, W: int, cover: float, fogc: wp.vec3,
                  Lmap: wp.array2d(dtype=float), Lair: wp.array2d(dtype=float), flick: float, warm: wp.vec3,
                  beam: wp.array2d(dtype=float), ray: float, bcol: wp.vec3, motes: wp.array2d(dtype=wp.int32),
                  t: float):
    i, j = wp.tid()
    c = out[i, j]
    if mask[i, j] == 1:
        sea = cl_lo[i, (j + off) % W]
        f = fogm[i, j] * wp.clamp(0.35 + 0.9 * sea * (0.6 + 0.6 * cover), 0.0, 1.0)
        L = Lmap[i, j] * flick
        tb = t_bg[i, j] + (fogc - t_bg[i, j]) * f
        bg = tb * (1.0 + L * 1.2) + warm * (L * 0.5)
        fg = clip3(t_fg[i, j] * (1.0 + L * 1.5) + warm * (L * 0.15) + fogc * (f * 0.6))
        ink = t_ink[i, j] * (1.0 - f)
        if f > 0.6 and sea > 0.62:                        # the cloud sea's own texture
            ink = 0.28
        c = comp(bg, fg, ink)
    c = c + warm * (Lair[i, j] * flick * 0.28)
    b = beam[i, j] * ray                                  # the stray ray of light
    c = c + bcol * (b * 0.55)
    if motes[i, j] == 1 and wp.sin(t * 2.3 + float(j) * 0.7 + float(i) * 1.3) > 0.6:
        c = comp(c, clip3(bcol * (0.4 + 0.8 * ray)), 0.1)
    out[i, j] = c


@wp.kernel
def k_flowers(out: wp.array2d(dtype=wp.vec3), fx: wp.array(dtype=float), fb: wp.array(dtype=float),
              fh: wp.array(dtype=float), kind: wp.array(dtype=wp.int32), ph: wp.array(dtype=float),
              t: float, gust: float, wind: float, sx: int, H: int, W: int, stem: wp.vec3, white: wp.vec3,
              blue: wp.vec3, Lmap: wp.array2d(dtype=float), flick: float):
    f, j = wp.tid()
    h = int(fh[f])
    if j > h:
        return
    bend = 0.35 * wp.sin(t * 1.7 + ph[f]) * (0.4 + gust) + wind * 1.4
    q = float(j) / wp.max(float(h), 1.0)
    x = int(fx[f] + bend * q * q * float(sx) * 2.2)
    y = int(fb[f]) - j
    if y < 0 or y >= H or x < 0 or x >= W:
        return
    L = 1.0 + Lmap[y, x] * flick * 1.5
    if j < h:                                             # the stem: a thin line, light enough to read
        out[y, x] = comp(out[y, x], clip3(stem * L), 0.8)
        return
    if kind[f] == 1:                                      # edelweiss: a white star of woolly petals
        c = clip3(white * L)
        for dy in range(-2, 3):
            for dx in range(-2 * sx, 2 * sx + 1):
                ax = wp.abs(dx) / wp.max(sx, 1)
                ay = wp.abs(dy)
                if (ax == 0 or ay == 0) or (ax == 1 and ay == 1):
                    xx = wp.clamp(x + dx, 0, W - 1)
                    yy = wp.clamp(y + dy, 0, H - 1)
                    a = 0.95
                    if ax + ay >= 2:
                        a = 0.6
                    out[yy, xx] = comp(out[yy, xx], c, a)
    else:                                                 # gentian: a deep blue cup, open at the top
        c = clip3(blue * L)
        for dy in range(-3, 1):
            w = sx + (3 + dy) * sx // 2
            for dx in range(-w, w + 1):
                xx = wp.clamp(x + dx, 0, W - 1)
                yy = wp.clamp(y + dy, 0, H - 1)
                rim = float(1.0)
                if dy == -3:
                    rim = 1.35
                out[yy, xx] = comp(out[yy, xx], clip3(c * rim), 0.9)


class GSummit(GScene):
    def upload(self):
        s = self.cpu
        self.upload_sky()
        self.static_layers()
        self.galaxy = self.arr(s.galaxy, wp.vec3)
        self.cl_hi, self.cl_lo = self.arr(s.cl_hi.astype(F32)), self.arr(s.cl_lo.astype(F32))
        self.fogm = self.arr(s.fogm.astype(F32))
        self.Lmap, self.Lair = self.arr(s.Lmap.astype(F32)), self.arr(s.Lair.astype(F32))
        self.beam = self.arr(s.beam.astype(F32))
        mot = np.zeros(s.beam.shape, np.int32)
        mot[s.motes] = 1
        self.motes = self.arr(mot, wp.int32)
        self.heat = [self.arr(np.zeros((s.fh + 1, s.fw), F32)) for _ in range(2)]
        self.fprof = self.arr(s.fprof.astype(F32))
        self.flame = self.stops(cf.FLAMES["fire"])
        self.ramp_ink = self.arr(np.array([INK.get(g, 1.0) for g in " .,:;=+*#%@"], F32))
        n = 60
        self.emb = [self.arr(np.zeros(n, F32)) for _ in range(4)] + [self.arr(s.eph.astype(F32))]
        fl = s.flowers
        self.nfl = len(fl)
        self.flw = [self.arr(np.array([f[0] for f in fl], F32)), self.arr(np.array([f[1] for f in fl], F32)),
                    self.arr(np.array([f[2] for f in fl], F32)), self.arr(np.array([f[3] for f in fl], np.int32), wp.int32),
                    self.arr(np.array([f[4] for f in fl], F32))]
        self.fmax = max([f[2] for f in fl] + [1]) + 1
        self.snow = [self.arr(s.sx_.astype(F32)), self.arr(s.sy_.astype(F32)), self.arr(s.sv.astype(F32)),
                     self.arr(s.sph_.astype(F32))]
        self.fogc = wp.vec3(*(rgb(*s.T["fog"]) * 1.05))
        self.drift = 0.0

    def draw(self, c, out):
        s, T, dev = self.cpu, self.cpu.T, self.dev
        lk = s._look(c)
        gust = getattr(c, "gust", 0.5)
        wind = (gust - 0.3) * lk["wind"]
        cover = float(np.clip(lk["cover"] + 0.12 * (getattr(c, "energy", 0.5) - 0.5) - 0.2 * getattr(c, "chord", 0.0), 0, 1))
        self.drift += c.dt * (0.6 + 1.2 * max(wind, 0)) * s.sx
        off = int(self.drift) % self.W
        launch(self, k_summit_sky, dim=(s.hr, self.W), device=dev,
                  inputs=[out, self.sky_bg, self.galaxy, self.bd, self.crater, self.star_id, self.sph, self.sbr,
                          self.cl_hi, s.hr, self.W, c.t, off, cover, self.cloud_col, self.body_col,
                          float(getattr(c, "pluck", 0.0))])
        pop = float(getattr(c, "pop", 0.0))
        flick = (0.7 + 0.08 * math.sin(c.t * 11.3) * math.sin(c.t * 6.1) + 0.28 * getattr(c, "bass", 0.5)
                 + 0.35 * pop + 0.25 * c.kick)
        ray = (0.08 + 0.75 * getattr(c, "lead", 0.0)) * (1 - 0.7 * lk["haze"] / 0.45) * (0.55 + 0.45 * (1 - cover))
        K = self.colours(lambda: dict(
            warm=wp.vec3(1.0, 150 / 255, 60 / 255), bcol=wp.vec3(1.0, 238 / 255, 205 / 255),
            smoke=wp.vec3(*(rgb(96, 94, 104) * T["lum"] * 1.4)),
            stem=wp.vec3(*s.lit(rgb(120, 160, 110), 1.8)), white=wp.vec3(*s.lit(rgb(240, 242, 232), 2.1)),
            blue=wp.vec3(*s.lit(rgb(60, 92, 235), 2.0)),
            snow=wp.vec3(*(rgb(240, 244, 255) * (0.55 + 0.45 * T["lum"]))),
            haze=wp.vec3(*(rgb(150, 160, 185) * (0.6 + 0.4 * T["lum"])))))
        warm, bcol = K["warm"], K["bcol"]
        launch(self, k_summit_land, dim=(self.H, self.W), device=dev,
                  inputs=[out, self.mask, self.t_bg, self.t_fg, self.t_ink, self.fogm, self.cl_lo,
                          int(self.drift * 0.4) % self.W, self.W, cover, self.fogc, self.Lmap, self.Lair, flick, warm,
                          self.beam, float(ray), bcol, self.motes, c.t])
        for _ in range(s.sy):                                  # the fire
            self.seed += 1
            a, b = self.heat
            launch(self, k_fire_base, dim=s.fw, device=dev, inputs=[a, self.fprof, s.fh, flick - 0.6, pop, self.seed])
            launch(self, k_fire_step, dim=(s.fh + 1, s.fw), device=dev, inputs=[a, b, s.fh, s.fw, self.seed + 7])
            self.heat = [b, a]
        x0 = int(s.fx - s.fw / 2)
        launch(self, k_fire_draw, dim=(s.fh, s.fw), device=dev,
                  inputs=[out, self.heat[0], s.fb - s.fh, x0, self.H, self.W, self.flame[0], self.flame[1], self.ramp_ink])
        smoke = K["smoke"]
        launch(self, k_smoke_wind, dim=(self.H, self.W), device=dev,
                  inputs=[out, float(s.fx), float(s.fb - s.fh * 0.6), float(s.fw), self.H, self.W, s.sy, c.t,
                          float(wind), smoke])
        self.seed += 1
        p_spark = 0.03 + 0.9 * pop + 0.35 * getattr(c, "pluck", 0.0)
        ex, ey, evy, life, eph = self.emb
        launch(self, k_embers, dim=ex.shape[0], device=dev,
                  inputs=[out, ex, ey, evy, life, eph, float(s.fx), float(s.fb), float(s.fw), float(s.fh), c.t, c.dt,
                          float((p_spark - 0.08) / 0.4), self.seed, s.sx, s.sy, self.W, self.H, self.flame[0],
                          self.flame[1]])
        if self.nfl:
            stem, white, blue = K["stem"], K["white"], K["blue"]
            fx, fb, fh, kind, ph = self.flw
            launch(self, k_flowers, dim=(self.nfl, self.fmax), device=dev,
                      inputs=[out, fx, fb, fh, kind, ph, c.t, float(gust), float(wind), s.sx, self.H, self.W, stem,
                              white, blue, self.Lmap, flick])
        nact = int(self.snow[0].shape[0] * lk["snow"])
        if nact:
            x, y, v, ph = self.snow
            col = K["snow"]
            launch(self, k_snow, dim=x.shape[0], device=dev,
                   inputs=[out, x, y, v, ph, c.t, c.dt, float(wind * (1.5 if lk["haze"] > 0.2 else 1.0)),
                           0.16 + 0.12 * lk["haze"], self.W, self.H, s.sx, s.sy, col, nact])
        haze = lk["haze"] * (0.5 + 0.8 * gust)
        if haze > 0.01:
            hcol = K["haze"]
            launch(self, k_fog, dim=(self.H, self.W), device=dev, inputs=[out, self.X, self.Y, self.H, c.t,
                                                                         float(min(0.85, haze)), hcol])


@wp.func
def war_smoke_col(y: float):
    """Smoke colour by height (0 = top of the sky, 1 = the horizon): black-red above, lit red from below."""
    if y < 0.7:
        return wp.vec3(20.0, 6.0, 6.0) / 255.0 + (wp.vec3(60.0, 16.0, 12.0) - wp.vec3(20.0, 6.0, 6.0)) / 255.0 * (y / 0.7)
    q = wp.min((y - 0.7) / 0.3, 1.0)
    return wp.vec3(60.0, 16.0, 12.0) / 255.0 + (wp.vec3(110.0, 34.0, 18.0) - wp.vec3(60.0, 16.0, 12.0)) / 255.0 * q


@wp.kernel
def k_war_bg(out: wp.array2d(dtype=wp.vec3), sky_bg: wp.array2d(dtype=wp.vec3), mask: wp.array2d(dtype=wp.int32),
             t_bg: wp.array2d(dtype=wp.vec3), t_fg: wp.array2d(dtype=wp.vec3), t_ink: wp.array2d(dtype=float),
             smoke: wp.array2d(dtype=float), smoke2: wp.array2d(dtype=float), off1: int, off2: int, W: int, H: int,
             hr: int, cover: float, glow: wp.array2d(dtype=float), flick: float, warmth: float,
             fires: wp.array2d(dtype=float), nf: int, ex: wp.array2d(dtype=float), bal: wp.array2d(dtype=float),
             t: float, a: float, G: float, flame_x: wp.array(dtype=float), flame_c: wp.array(dtype=wp.vec3)):
    """Sky, land, smoke, firelight, burning fires, the light of explosions and of the flying fireballs."""
    i, j = wp.tid()
    fi, fj = float(i), float(j)
    c = wp.vec3(0.0)
    if mask[i, j] == 1:
        c = comp(t_bg[i, j], t_fg[i, j], t_ink[i, j])
    else:
        c = sky_bg[i, j]
    n = 0.7 * smoke[i, (j + off1) % W] + 0.3 * smoke2[i, (j + off2) % W]
    sm = wp.clamp((n - (1.0 - cover)) / 0.3, 0.0, 1.0) * wp.clamp(1.25 - fi / (float(hr) * 1.1), 0.0, 1.0)
    c = c + (war_smoke_col(wp.min(fi / float(hr), 1.0)) - c) * (sm * 0.8)
    warm = wp.vec3(1.0, 110.0 / 255.0, 40.0 / 255.0)
    c = c + warm * (glow[i, j] * flick * warmth)
    px = fj / a
    for k in range(nf):                                       # flames licking up from each fire
        fx, fy, fr, ph = fires[k, 0], fires[k, 1], fires[k, 2], fires[k, 3]
        dy = fy - fi
        hgt = fr * 0.5
        if dy >= 0.0 and dy < hgt:
            q = dy / hgt
            w = fr * 0.09 * (1.0 - q) * (0.7 + 0.3 * wp.sin(t * 11.0 + ph * 3.0 + dy * 0.6))
            dx = px - fx - fr * 0.12 * wp.sin(t * 7.0 + ph + dy * 0.3) * q
            if wp.abs(dx) < w:
                c = comp(c, ramp(flame_x, flame_c, 1.0 - q * (0.6 + 0.4 * wp.abs(dx) / w)), 0.85)
    for k in range(ex.shape[0]):                              # explosions: a flash, then the place burns
        if ex[k, 4] > 0.0:
            tau = t - ex[k, 2]
            d = wp.sqrt((px - ex[k, 0]) * (px - ex[k, 0]) + (fi - ex[k, 1]) * (fi - ex[k, 1])) / (float(H) * 0.05 * ex[k, 3])
            fl = wp.exp(-tau / 0.25) * 2.5 + 0.45 * (0.7 + 0.3 * wp.sin(t * 13.0 + ex[k, 6])) * wp.min(1.0, (ex[k, 5] - tau) / 2.0)
            c = c + wp.vec3(1.0, 150.0 / 255.0, 60.0 / 255.0) * (wp.exp(-d) * fl)
    for k in range(bal.shape[0]):                             # the light of each fireball in flight
        if bal[k, 7] > 0.0:
            tau = t - bal[k, 4]
            bx = bal[k, 0] + bal[k, 2] * tau
            by = bal[k, 1] + bal[k, 3] * tau + 0.5 * G * tau * tau
            d = wp.sqrt((px - bx) * (px - bx) + (fi - by) * (fi - by)) / (float(H) * 0.03 * bal[k, 6])
            c = c + wp.vec3(1.0, 130.0 / 255.0, 40.0 / 255.0) * (wp.exp(-d) * 0.8)
    out[i, j] = c


@wp.kernel
def k_riders(out: wp.array2d(dtype=wp.vec3), hor: wp.array2d(dtype=float), sat: wp.array3d(dtype=float),
             y0: int, W: int, beat: float, a: float, sh: int, sw: int, nfr: int):
    """Every rider over every pixel of the plain, nearest last (the host sorts them by depth): the sprite's exact
    coverage of the pixel from its summed-area tables, solid above 40%; the top edge lit by the sky."""
    ii, j = wp.tid()
    i = ii + y0
    hit = int(0)
    col = wp.vec3(0.0)
    for h in range(hor.shape[0]):
        if hor[h, 7] <= 0.0:
            continue
        rows_n = wp.max(3, int(hor[h, 2]))
        cols_n = wp.max(3, int(hor[h, 2] * float(sw) / float(sh) * a))
        top = int(hor[h, 1]) - rows_n
        left = int(hor[h, 0] * a - float(cols_n) / 2.0)
        ry, rx = i - top, j - left
        if ry < 0 or ry >= rows_n or rx < 0 or rx >= cols_n:
            continue
        frame = int(fmodp(beat + hor[h, 3], 1.0) * float(nfr)) % nfr
        base = (int(hor[h, 6]) * nfr + frame) * 2
        xa = wp.clamp(rx * sw // cols_n, 0, sw - 1)
        xb = wp.clamp(wp.max((rx + 1) * sw // cols_n, xa + 1), 1, sw)
        if hor[h, 4] < 0.0:
            t0 = sw - xb
            xb = sw - xa
            xa = t0
        ya = wp.clamp(ry * sh // rows_n, 0, sh - 1)
        yb = wp.clamp(wp.max((ry + 1) * sh // rows_n, ya + 1), 1, sh)
        cov = (sat[base, yb, xb] - sat[base, ya, xb] - sat[base, yb, xa] + sat[base, ya, xa]) / float((yb - ya) * (xb - xa))
        if cov <= 0.4:
            continue
        near = (sat[base + 1, yb, xb] - sat[base + 1, ya, xb] - sat[base + 1, yb, xa] + sat[base + 1, ya, xa]) / \
            wp.max(cov * float((yb - ya) * (xb - xa)), 1.0e-6)
        c = wp.vec3(12.0, 4.0, 4.0) / 255.0 + (wp.vec3(26.0, 9.0, 8.0) - wp.vec3(12.0, 4.0, 4.0)) / 255.0 * near
        if rows_n >= 14:                                      # lit edge: the pixel above is not covered
            up = float(0.0)
            if ry > 0:
                ya2 = wp.clamp((ry - 1) * sh // rows_n, 0, sh - 1)
                yb2 = wp.clamp(wp.max(ry * sh // rows_n, ya2 + 1), 1, sh)
                up = (sat[base, yb2, xb] - sat[base, ya2, xb] - sat[base, yb2, xa] + sat[base, ya2, xa]) / \
                    float((yb2 - ya2) * (xb - xa))
            if up <= 0.4:
                c = wp.vec3(160.0, 46.0, 28.0) / 255.0
        haze = (1.0 - hor[h, 5]) * 0.55
        col = c + (wp.vec3(150.0, 44.0, 30.0) / 255.0 - c) * haze
        hit = 1
    if hit == 1:
        out[i, j] = col


@wp.kernel
def k_arrows(out: wp.array2d(dtype=wp.vec3), arr: wp.array2d(dtype=float), t: float, G: float, L: float,
             W: int, H: int, a: float):
    p, k = wp.tid()
    if arr[p, 7] <= 0.0 or t < arr[p, 4]:
        return
    tau = t - arr[p, 4]
    vx = arr[p, 2]
    vy = arr[p, 3] + G * tau
    sp = wp.sqrt(vx * vx + vy * vy) + 1.0e-6
    q = float(k) / 11.0
    x = (arr[p, 0] + arr[p, 2] * tau - vx / sp * L * q) * a
    y = arr[p, 1] + arr[p, 3] * tau + 0.5 * G * tau * tau - vy / sp * L * q
    ix, iy = int(x), int(y)
    if ix < 0 or ix >= W or iy < 0 or iy >= H:
        return
    c = wp.vec3(0.05, 0.02, 0.015)
    if arr[p, 6] > 0.0 and k < 3:                             # a burning head
        c = wp.vec3(1.0, 0.75, 0.32)
    out[iy, ix] = c                                           # two samples wide, or the glyph pass loses it
    if ix + 1 < W:
        out[iy, ix + 1] = comp(out[iy, ix + 1], c, 0.8)


@wp.kernel
def k_fireballs(out: wp.array2d(dtype=wp.vec3), bal: wp.array2d(dtype=float), t: float, G: float, W: int, H: int,
                a: float, sc: float):
    """A burning ball and behind it its trail: white-hot, then flame, then smoke (the same parabola, earlier)."""
    p, k = wp.tid()
    if bal[p, 7] <= 0.0:
        return
    tk = t - bal[p, 4] - float(k) * 0.035
    if tk < 0.0:
        return
    x = (bal[p, 0] + bal[p, 2] * tk) * a
    y = bal[p, 1] + bal[p, 3] * tk + 0.5 * G * tk * tk
    q = float(k) / 15.0
    c = wp.vec3(0.0)
    if q < 0.15:
        c = wp.vec3(1.0, 0.98, 0.82)
    elif q < 0.5:
        c = wp.vec3(1.0, 0.67, 0.2) + (wp.vec3(0.78, 0.2, 0.04) - wp.vec3(1.0, 0.67, 0.2)) * ((q - 0.15) / 0.35)
    else:
        c = wp.vec3(0.78, 0.2, 0.04) + (wp.vec3(0.2, 0.12, 0.12) - wp.vec3(0.78, 0.2, 0.04)) * ((q - 0.5) / 0.5)
    rad = sc * bal[p, 6] * (1.0 - 0.6 * q)
    R = int(rad) + 1
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            d = wp.sqrt(float(dx * dx + dy * dy)) / wp.max(rad, 0.5)
            if d <= 1.0:
                ix, iy = int(x) + dx, int(y) + dy
                if ix >= 0 and ix < W and iy >= 0 and iy < H:
                    out[iy, ix] = comp(out[iy, ix], c, (1.0 - d * 0.5) * (1.0 - 0.5 * q))


@wp.kernel
def k_debris(out: wp.array2d(dtype=wp.vec3), ex: wp.array2d(dtype=float), t: float, G: float, W: int, H: int,
             a: float):
    """Burning debris thrown up by each blast, for its first 0.6 s."""
    e, k = wp.tid()
    if ex[e, 4] <= 0.0:
        return
    tau = t - ex[e, 2]
    if tau > 0.6 or tau < 0.0:
        return
    s = int(ex[e, 6])
    ang = -3.14159 * hash01(e, k, s)
    sp = (0.3 + 0.7 * hash01(k, e, s + 1)) * float(H) * 0.5 * ex[e, 3]
    x = (ex[e, 0] + wp.cos(ang) * sp * tau) * a
    y = ex[e, 1] + wp.sin(ang) * sp * tau + 0.5 * G * tau * tau
    ix, iy = int(x), int(y)
    if ix >= 0 and ix < W and iy >= 0 and iy < H:
        out[iy, ix] = comp(out[iy, ix], wp.vec3(1.0, 0.82, 0.47), 0.9)


@wp.kernel
def k_ash(out: wp.array2d(dtype=wp.vec3), ash: wp.array2d(dtype=float), n: int, W: int, H: int):
    p = wp.tid()
    if p >= n:
        return
    ix = int(ash[p, 0]) % W
    iy = int(ash[p, 1]) % H
    if ash[p, 2] > 1.25:                                      # an ember rising
        out[iy, ix] = comp(out[iy, ix], wp.vec3(1.0, 0.55, 0.2), 0.6)
    else:
        out[iy, ix] = comp(out[iy, ix], wp.vec3(0.27, 0.24, 0.24), 0.35)


class GWar(GScene):
    def upload(self):
        s = self.cpu
        H, W = s.size[1], s.size[0]
        self.H, self.W = H, W
        self.static_layers()
        self.sky_bg = self.arr(s.sky_bg, wp.vec3)
        self.smoke, self.smoke2 = self.arr(s.smoke.astype(F32)), self.arr(s.smoke2.astype(F32))
        self.glow = self.arr(s.glow.astype(F32))
        fires = np.zeros((max(1, len(s.fires)), 4), F32)
        fires[: len(s.fires), :3] = s.fires
        fires[: len(s.fires), 3] = s.fire_ph
        self.fires, self.nf = self.arr(fires), len(s.fires)
        sat = s.spr_sat.reshape(-1, cf.SPR_H + 1, cf.SPR_W + 1)       # (kind, frame, any|near) flattened
        self.sat = self.arr(np.ascontiguousarray(sat, F32))
        self.flame = self.stops(cf.FLAMES["fire"])
        # the moving parts: simulated on the host (cf.War.step), copied each frame (a few KB)
        self.d_hor = self.arr(np.zeros_like(s.hor))
        self.d_arr = self.arr(np.zeros_like(s.arr_))
        self.d_bal = self.arr(np.zeros_like(s.bal))
        self.d_exp = self.arr(np.zeros_like(s.exp_))
        self.d_ash = self.arr(np.zeros_like(s.ash))
        self.ry0 = max(0, int(s.hr - H * 0.25))                    # riders live below this row

    def draw(self, c, out):
        s, dev = self.cpu, self.dev
        s.step(c)
        x = s.intensity(c)
        order = np.argsort(s.hor[:, 5], kind="stable")             # far riders first, nearest painted last
        self.d_hor.assign(s.hor[order])
        self.d_arr.assign(s.arr_)
        self.d_bal.assign(s.bal)
        self.d_exp.assign(s.exp_)
        self.d_ash.assign(s.ash)
        drift = c.t * (3 + 6 * x) * s.sx
        flick = 0.7 + 0.15 * math.sin(c.t * 9.1) * math.sin(c.t * 5.3) + 0.35 * c.kick + 0.2 * getattr(c, "level", 0.5)
        a = float(s.a)
        launch(self, k_war_bg, dim=(self.H, self.W), device=dev,
               inputs=[out, self.sky_bg, self.mask, self.t_bg, self.t_fg, self.t_ink, self.smoke, self.smoke2,
                       int(drift) % self.W, int(drift * 1.7) % self.W, self.W, self.H, int(s.hr), float(0.3 + 0.45 * x),
                       self.glow, float(flick), float(0.25 + 0.2 * x), self.fires, self.nf, self.d_exp, self.d_bal,
                       c.t, a, float(s.G), self.flame[0], self.flame[1]])
        launch(self, k_riders, dim=(self.H - self.ry0, self.W), device=dev,
               inputs=[out, self.d_hor, self.sat, self.ry0, self.W, float(getattr(c, "beat", 0.0)), a,
                       cf.SPR_H, cf.SPR_W, cf.SPR_F])
        launch(self, k_arrows, dim=(s.NA, 12), device=dev,
               inputs=[out, self.d_arr, c.t, float(s.G), float(self.H * 0.05), self.W, self.H, a])
        launch(self, k_fireballs, dim=(s.NB, 16), device=dev,
               inputs=[out, self.d_bal, c.t, float(s.G), self.W, self.H, a, float(self.H * 0.008)])
        launch(self, k_debris, dim=(s.NE, 14), device=dev, inputs=[out, self.d_exp, c.t, float(s.G), self.W, self.H, a])
        launch(self, k_ash, dim=len(s.ash), device=dev,
               inputs=[out, self.d_ash, int(150 + 450 * x), self.W, self.H])


GPU_OF = {cf.Seaside: GSeaside, cf.Lighthouse: GLighthouse, cf.Mountains: GMountains, cf.Lake: GMountains,
          cf.Forest: GForest, cf.Campfire: GCampfire, cf.Summit: GSummit, cf.SnowCamp: GCampfire, cf.War: GWar}
