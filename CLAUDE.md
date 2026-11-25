# claude fm — handoff

Generative lo-fi radio with animated ASCII nature scenes in a GPU window, built on the **kor** engine.

## Layout (since 2026-11-25)

The engine was split out into **kor**: its own uv project, checked out next to this one as `../kor`
(https://github.com/Desto2000/Kor, branch main), with its own CLAUDE.md. This project depends on it as an editable
path dependency (`[tool.uv.sources] kor = { path = "../kor", editable = true }`): edits in kor apply here at once.
The terminal front end was dropped (by request) along with every scene's CPU per-frame `draw()`.

| here: `claudefm/` package (`python -m claudefm`) | in kor (`kor.*`) |
| --- | --- |
| `scenes.py` host setup + host sims of the landscapes (Nature, Seaside … Summit, War, `horse_sprites`, `TOD`) | `music/` (was `./music`), `audio.py` (`Engine`, `Timeline`) |
| `gpu_scenes.py` `G*` wrappers, landscape kernels (`k_sky`, `k_war_bg`, `k_riders` …), `GPU_OF` | `scene.py` `Scene`, `GScene`, base `Ctx`, `prepare`, `in_background` |
| `show.py` `Show`, `GShow`, `Ctx` (adds the music-feature slots), `hud` | `renderer.py` `Renderer(show=…)`, `LOOKS`; `window.py` `run()`, `Presenter`, `TitleBar` |
| `app.py` `make_renderer`, `run_window`, `bench`, `render_wav`, `main` | `launch.py`, `cells.py`, `raster.py`, `gpu.py`, `fx.py` (fire, smoke, embers, reflect, fog, particles, snow), `glyphs.py`, `util.py` |

The `Renderer` takes `show=` (a factory `(rng, dev, prefetch)`); `make_renderer` binds `GShow`. `kor.window.run`
handles q/Esc, r, space, ↑↓, f, h itself; the app passes `on_key` (n, t, Ctrl+W, c) and `hud`. Older notes below
name `claudefm.py` / `claudefm_gpu.py`: read them as the modules above. The split was verified with
`tools/launch_trace.py` (all 9,228 launches, host state and music identical to the pre-split code).
Git: initialised 2026-11-25, history squashed to a single commit the same day; large outputs (wav, mp4, `ab/`,
`profiles/`) are ignored.

## Environment and constraints

- **Use uv only. No global installs, ever.** `pyproject.toml` is a non-package uv project;
  `uv sync` makes `.venv` here (and installs kor from `../kor`, editable). Run everything with `uv run ...`.
- Test machine: Windows 11, NVIDIA RTX 3060 Ti. Python 3.14 (`.python-version` = `3.14+gil`, uv-managed 3.14.7; pyproject
  allows ≥3.10 <3.16). 3.15rc1 and free-threaded 3.14t also run (same fps within noise, 2026-11-24); uv 0.12.3
  only knows 3.15.0rc1. Note `3.14` alone made uv pick the free-threaded build once a 3.14t was installed.
- Working style: be direct, give real numbers, ground claims in official
  docs (Warp: https://nvidia.github.io/warp/) rather than memory. Flag unverified assumptions.

## Commands

```
uv sync
uv run python -m claudefm --bench 120                                      # time every scene × look, no window
uv run python -m claudefm --bench 120 --canvas 2560x1440 --cell 6x12
uv run python -m claudefm --seconds 25 --screenshot window.png             # window test, auto-closes
uv run python -m claudefm                                                  # normal run (or claudefm.cmd)
uv run python -m claudefm --render out.wav --minutes 2                     # music only (--style war …)
uv run python tools/story.py --out ig_story.mp4 --poster ig_story_cover.png   # 20 s vertical IG story
uv run python tools/launch_trace.py record before/   …   compare before/ after/   # refactor check (see the tool)
```
`run_bench.cmd` / `run_window_test.cmd` wrap the above and log to `bench.log` / `window.log`.
Set `PYTHONUTF8=1` when redirecting output on Windows (the logs contain `·` and `×`).

GPU-window keys: n scene · t track · Ctrl+W war on/off · c time of day · r look (ascii/blocks/braille) · space pause ·
↑↓ volume · f fullscreen · h status bar · q quit.

## Architecture (GPU version)

Per frame: `Renderer.frame()` →
1. **scene** → sample buffer `(rows*4, cols*2)` of `vec3` (2×4 samples per character cell).
   `GScene` subclasses wrap the CPU scene: `ensure()` runs the CPU `setup()` once per size,
   `upload()` copies static arrays to the device, `draw()` launches kernels
   (`k_sky`, `k_sea`, `k_beach`, `k_reflect`, `k_mountains`, `k_forest`, `k_campfire_land`,
   fire sim `k_fire_base/k_fire_step/k_fire_draw`, `k_smoke`, `k_embers`, `k_spray`, `k_coast`,
   `k_beam`, `k_fog`, `k_particles`, `k_birds`, `k_dots`). Scene changes dissolve via `k_dissolve`.
2. **cells** — one thread per character cell: `k_cells_ascii` (brightness ramp + edge glyphs
   from 2×4 ink templates), `k_cells_blocks` (`▀`, fg=top, bg=bottom), `k_cells_braille` (dithered dots).
   Auto-exposure reads cell luminance back asynchronously every 15 frames (after the frame's kernels).
3. **raster** — `raster_kernel(cw, ch, hcw)` builds kernels with the cell size compiled in. `k_raster`: one thread
   per canvas pixel, samples the glyph atlas (`Atlas`, built with Pillow from Cascadia Mono / Consolas, symbols
   from Segoe UI Symbol), writes RGBA8 — used by `--bench`, `--present mip` and tools. `k_raster_view`: one
   thread per *window* pixel, the exact box filter of the canvas computed straight from the cells via per-glyph
   summed-area tables (`Atlas.sat`); ≤1 LSB from box-filtering k_raster's canvas. `frame(c, pix, view=(w, h))`.

All kernels go through `launch(owner, kernel, dim, inputs, device)`: a cached `wp.Launch` per call site that
re-packs only changed args (numbers/vectors written in place). In the window, `Renderer.frame` runs inside
`FrameGraph`: the frame's launches are recorded and submitted as one CUDA graph (node params updated with
`cuGraphExecKernelNodeSetParams` via ctypes on nvcuda.dll; a sequence is captured once stable for 8 frames;
`--no-graph` turns it off). Keep launch sizes fixed per scene (k_snow takes the active count as a param) or
the graph is re-captured.

Presentation (`Presenter`, `PixelBuffer`): CUDA writes straight into a registered OpenGL buffer (zero copy);
default `--present area`: the buffer is window-sized (re-allocated on resize), drawn 1:1 through a buffer
texture by a one-triangle shader. `--present mip` (old look): 4K canvas → `glTexSubImage2D` → mipmaps (only
the levels the scale samples) → trilinear. pyglet's `debug_gl` is off (`--debug-gl` turns it on).
The canvas is fixed (default 3840×2160, 8×16 cells = 480×133 chars); the window only scales it.
Title bar (`TitleBar`, Windows only, 2026-11-25): the native caption is replaced by a 24 px (× DPI) black bar drawn in
GL above the picture; window style stays WS_OVERLAPPEDWINDOW, subclassed WM_NCCALCSIZE/WM_NCHITTEST keep drag, snap,
double-click maximise, system menu and resize native; min/max/close are GL-drawn and clicked via pyglet mouse events.
Hidden in fullscreen. Verified by hit-test probes (normal/maximised/restored); ~1300 fps unchanged.
Without CUDA it falls back to rendering in host memory and `glBufferSubData`.

Music: `cf.Composer` renders bars ahead on a thread; `cf.Engine` outputs via sounddevice
(fallback: pipe to mpv/ffplay, else silent). Visuals sync to `engine.position()`.
- All music lives in `kor.music` (was `./music`; the files moved unchanged):
  `core.py` DSP + engine, `instruments.py`, `compose.py`, `mix.py`. Engine render-ahead is 8 s.
- Engine: a note = pitch curve per 10 ms frame + spectrum per frame → wavetables (batched irfft) read by
  1..N *players* (`Tonal.players`), each with own detune, vibrato, onset spread, loudness wobble and pan —
  ensembles (string sections 3–6, choir 5, brass 3–4) are the main realism lever. ~0.3–3% RT per note.
- Instruments: violins/violas/cellos/basses (sus, trem, spic, pizz), octobass, horns/trombones/tuba, SATB
  choir (formants, vowels, breath), organ, hurdy-gurdy (+ HurdyBuzz trompette), tagelharpa, morin khuur,
  flute/willow flute/ocarina/ney, contrabass clarinet/serpent, harp/lyre/yatga/dombra/saz, throat singer
  (khoomei/sygyt/kargyraa), jaw harp, drums (taiko ensemble, frame, shaman, tam-tam, anvil, bells, swell).
- The mood brief (2026-11-24): NOT epic orchestra. "A warm environment on a snowy mountain, a bonfire in
  front of you, dark and chill, lonely; deep bass." Earlier epic styles (cathedral/march) were rejected and
  removed; the old engine was also heard as "8-bit synths".
- Mood variants (`compose.STYLES`, never the same twice in a row): hearth (tagelharpa/hurdy, lyre embers,
  willow flute), snowfall (solo cello `tasto`, flute, contrabass clarinet), soul (morin khuur, khöömei,
  serpent). `Lonely` writer: tonic pedal (octobass + low reed) that never stops, chords every 6–12 s,
  sparse 2-bar phrases + 2 bars of silence, embers, hummed low voices, faint heartbeat; sections
  dusk/alone/warmth/night/ash (~3 min). Ostinato/groove tools remain in `Writer`, unused by this mood.
- Realism levers in `Tonal`: `jitter` (fast cents), `shimmer`, `living` (per-harmonic slow drift),
  `noise_lock` (breath/bow noise pulsing with the waveform); `tasto` articulation for bowed strings.
  Contrabass clarinet no longer odd-only (that was a square wave = the 8-bit sound).
- Mix: per-family dense modal body IR (`mix.make_body`, flat average response — an earlier version tilted
  ~3.6 dB/oct and made everything mud); mountain IR (dark, valley echoes); wind + fire; warm master EQ.
- Feedback 2 (same day): mood right, but "too empty; a bit more bass, more lead so it touches; more bass
  options in parallel". Now: low end = octobass sub pedal + walking bowed basses (passing notes into chord
  changes) + low reed countermelody (cbcl/serpent) + second reed pedal + pizz heartbeat + tuba or low choir;
  lead = `lyric_theme` (3 singing bars + 1 breath, leap to a peak, stepwise fall, "sigh" appoggiatura at
  the cadence), doubled an octave below by a second instrument in "warmth", last phrase climbs an octave;
  mid fill = early cello pads, rolled harp/lyre chords per chord change.
- Master compressor detector is high-passed at 150 Hz (`Compressor.sc_hpf`); without it, bass boosts were
  simply compressed away. Low shelf +3.5 dB below 80 Hz for the mood. Engine render-ahead is 12 s.
- Measured (seed 4): ~37% RT; worst bar 4.4 s (bar 3.6 s, absorbed by the 12 s buffer); GIL stall ≤ 4.4 ms;
  RMS ≈ −10 dBFS, peaks −0.4…−1.1. Vs the version approved for mood: bass body (60–250) +2…+4 dB
  relative to the lead band, sub kept pace with a louder lead.
- Listening material in `ab/`: `low_before/after.wav` (engine A/B), `clip_<variant>.wav`, `full_<variant>.wav`.
  Balance is by measurement; it needs ears.

## Status

Verified (in a Linux sandbox, Warp on **CPU**, Xvfb + Mesa for the window):
- All 6 scenes × 3 looks render correctly through the kernels and match the (since removed) terminal version.
- Window path works (letterbox, status bar, music, clean exit) via the CPU fallback.
- `--bench` runs. CPU-device numbers are meaningless for GPU speed.

Verified on Windows 11 / RTX 3060 Ti (2026-11-24; warp-lang 1.19.0, CUDA 13.4 driver, Python 3.12):
- `uv sync` clean (numpy 2.5.5, pyglet 2.1.19, pillow 13.0.0, sounddevice 0.5.6).
- `--bench` (wall-clock per stage, synced): 4K/8×16 total 0.56–1.19 ms; 1440p/6×12 0.36–0.83 ms.
  GPU-only kernel time (`wp.timing_begin/end`) ≈ pipelined throughput ≈ host launch time,
  0.40–0.58 ms (4K) / 0.24–0.41 ms (1440p) → not launch-bound yet; `k_raster` is the biggest
  kernel (~0.30 ms at 4K, ~0.16 ms at 1440p). CUDA graphs not worth it at this scale.
- Zero-copy `RegisteredGLBuffer` path works; window ~920–1040 fps uncapped, audio via sounddevice.
- Font: Cascadia not installed on this machine → Consolas; symbols from seguisym.ttf.
  Fixed: symbol glyphs are now baseline-aligned, centred and shrunk to fit (`●` was clipped).

Scene-switch hitch: fixed. CPU `setup()` (40–400 ms at 4K) used to run on the first frame of a new scene
(worst frame 477 ms). Now `GShow` prepares the next scene on a daemon worker thread while the current one plays:
`prepare()` runs setup and `GScene.pack()` (= `upload()` with `arr()` returning host-side `Packed`
placeholders, **no Warp calls on the worker** — Warp's docs don't cover multi-threaded host use), then
`commit()` on the main thread does only the `wp.array` H2D copies (~3–4 ms). A switch requested before the
next scene is ready waits for it instead of stalling. 'c' recolours a `deepcopy` of the scene the same way.
`sys.setswitchinterval(0.0005)` is essential: at the default 5 ms, GIL handoffs to/from the (Python-loop-heavy)
setup cost the render loop 13–21 ms each. Result (switch or recolour every 1.5 s): worst frame 12.6 ms at 4K,
10.3 ms at 1440p. Worker-built scenes render bit-identically to synchronously built ones.
`--bench` uses `prefetch=False` (synchronous, no worker thread skewing timings).

Known nondeterminism (pre-existing): `k_particles` snow in Forest — overlapping particles write the same
sample in nondeterministic order (a few samples differ between identical runs). Measured 2026-11-24: it is wider
than Forest — particle/snow/ember kernels make ~25% of frames differ run to run in most scenes (tens to hundreds
of pixels at 1080p). Frame hashes therefore can't prove a refactor is output-identical; compare launch params.

Render-loop CPU cost (2026-11-24, nsys Python sampling): the render thread was Python-bound — `wp.launch` argument
packing ~48%, `set_hud` 11%, flip 13%. Now: every kernel goes through `launch(owner, kernel, dim, inputs, device)`,
which records a `wp.Launch` per call site (`record_cmd=True`, cached on the scene/renderer in `_cmds`) and re-packs
only changed args (verified: 13,740 replayed launches byte-identical to fresh `wp.launch` packing, all scenes × looks).
HUD is rebuilt only when its text changes; auto-exposure reads `lum` back asynchronously into pinned memory and uses
it 15 frames later (no sync). Window, unprofiled, 4K: 522–559 → 854–883 fps, mean frame 1.85 → 1.15 ms, p99 4.0 →
2.4 ms. `--bench` scene stage 0.51–0.75 → 0.29–0.47 ms. Remaining per-frame CPU: Warp's native launch call
(~25%), `flip` 18%, pyglet `tex.blit` 12% (builds a vertex list every call), `Show.context` 7%.
Profiling captures: `profiles/` (clean1 = before, clean2 = after; not in git). A recipe that works without admin,
run from PowerShell (from Git Bash nsys printed nothing): `nsys profile --trace=cuda,opengl,nvtx,python-gil
--sample=none --cpuctxsw=none --python-sampling=true --delay=15 --duration=30 -o profiles/X uv run python -m claudefm
--seconds 50`, then `nsys export --type=sqlite --include-json=true` (Python backtraces: NVTX_EVENTS.jsonText, domain
"Python Periodic Sampling"; GIL ranges: domain "GIL Trace"). `--sample` / `--cpuctxsw` need an elevated shell.

Round 2 (2026-11-24, Python 3.14). The window had become GPU-bound (86% util, ~170 W): per frame the GPU spent
~270 µs on k_raster, 193 µs copying the 33 MB canvas into a texture and 116–180 µs on mipmaps (GL timer queries).
Now: compile-time cell size in k_raster (330 → 268 µs, bit-identical), `k_raster_view` renders only the window's
pixels (no canvas, no copy, no mipmaps; box filter instead of trilinear: PSNR ~31–32 dB vs the old look, a bit
crisper), Presenter replaces `tex.blit`, pyglet debug_gl off, CUDA graph per frame, k_snow fixed launch size,
in-place params, per-scene colour cache. Verified: graph vs direct bit-identical for the deterministic scenes
(Seaside, Lighthouse; all looks, with dissolves) and within run-to-run noise elsewhere; launch params byte-identical.
Window, 4K canvas → 1600×900, unprofiled, same seeds: original code 396–444 fps → 1153–1161 fps (p99 4.0–5.3 →
1.6–1.8 ms). Uncapped, that is ~150 W. `--vsync` used to do nothing in a window: pyglet 2.1 on Windows 8+ forces the
swap interval to 0 when windowed (and skips its DwmFlush fallback); `run_window` now sets it on the GL context
itself (`apply_vsync`, re-applied after the fullscreen toggle) → 360.0 fps on the 360 Hz screen, 114 W vs 152 W. Remaining per frame (~0.86 ms): scene Python +
launch bookkeeping ~0.3 ms, flip ~0.18, CUDA-GL map/unmap ~0.11, `Show.context` ~0.08, graph submit ~0.07.
Open: a rare single long frame (66–240 ms) seen in a few runs of both the original and new code; not reproduced
under the stack-dump watchdog (5×40 s + 24 s).

- Feedback 3: bring every instrument back in the new style; a percussion-led style. Each mood track now
  draws 3–4 *colours* from its variant's pool (`write_track`: horns as distant calls in the lead's breath bar,
  violins `air` (high tremolo, tasto), violas, trombones, organ hum, `ghosts` (choir_s/a on 'oo'), dombra
  strum, jaw harp, saz/yatga embers, ocarina/ney/flute seconds, hurdy buzz). Over 160 tracks all 30
  instruments and all 14 drum sounds (incl. new `rattle`, `stick`) appear. `tr.colours` records the picks.
- New style `ritual` ("drum circle", `Ritual` writer): locked layers per meter (heart, circle, shaman, rattle,
  sticks 3-over-2, hands = darbuka, hooves, taiko, rims), fills into every 4th bar; sections
  gathering/circle/trance/storm/embers; octobass + throat chant + jaw + distant horns; drums ×1.4.
- Cost fixes: long notes split at ~9 s (`Lonely.runs`), jitter/shimmer/wobble at frame rate in `Tonal.render`,
  bus bodies summed per family (one 4-column convolution each). Mood ≈ 50–60% RT on one core (mixer ~20%),
  worst bars ~4–6 s at section starts (12 s buffer); ritual ≈ 23% RT, worst bar < bar length.
- `tools/solo.py` renders solo instrument passages through body + mountain reverb.

- Feedback 4: Asian percussion on top. `Drums` adds Japan (shime, kane/kane_d, hyoshigi), Korea (janggu
  kung/deok, kkwaeng/kkwaeng_d, jing), China (bangu, daluo with falling pitch, xiaoluo rising, bo, temple
  blocks), India/Himalaya (tabla ge with wrist bend, ke, te; dayan na/tin, singing bowl). Tuned sounds
  (`Drums.TUNED`: na, tin, block, bowl) render per pitch from `note.m` and are cached.
- Each drum-circle track layers one palette (`Ritual.ASIA[meter][palette]` light/main + `asia_big`) over its own
  drums (None 12%, others 22% each); `compose.FORCE_PALETTE` forces one for tools. Mood tracks can draw a
  singing bowl colour (on every other chord in dusk/night/ash). `tools/drumcircle.py` renders one clip per
  palette (same track, only the palette differs).

- War (^w, 2026-11-24): `compose.write_war` / `War(Ritual)`. Hidden from the radio (`compose.HIDDEN`); `Composer.toggle_war()`
  + `engine.skip` (Ctrl+W). No sections: everything is a function of x = track position —
  tempo map (`Track.starts/ps/bar_groups`, 120–180 → ~195–210 bpm), odd bars (7/8, 5/8, 11/8…), torn-out beats + tutti,
  chromatic roots/clusters, riff in canon and a tritone apart; every instrument and all 36 drum sounds (verified seeds 0–5).
  Ends in one last blow + failing heartbeat. `mix.Orchestra.render` caches notes tagged `rr=True` (2 variants, 400 MB cap).
  `python -m claudefm --render x.wav --style war` renders it; `--war` starts at war.
- War feedback 1 (2026-11-25): "more strings and percussion — bassy war drums, beopgo; too much keys". Measured, the
  organ was only ~1% of the mix; the keyboard-like layers were plucked tremolo (saz/dombra), harp sweeps, winds, buzz
  and tuned metal. Now: leads are bowed only (`WAR_LEADS`); `string_drive` (violins 16th spiccato + violas),
  `string_runs` (scales into the bar line), strings on every brass stab, the riff doubled by violins/cellos
  (`unison`); `war_drums`: new `Drums` sounds `beopgo` (~44 Hz temple barrel drum + rolls), `odaiko` (~35 Hz),
  `davul`/`davul_tek` (marching war drum), own RNG so other sounds' variants are unchanged; organ only in the last
  blow; plucks/buzz/chimes/winds turned down. Family energy share, last third (seed 3): strings 36 → 47%,
  drums 27 → 36%, plucked 9 → 1%, winds+reeds 9 → 4%. All 30 instruments still appear. 34% RT.
  `ab/war_seed3.wav` (full track), `ab/war_drums.wav` (the four new drums alone).
- War feedback 2: "I don't hear the drums at all" (the solo clip was audible: a mix problem, not the
  sub-bass) → "make them punchier". Measured (last 40%, seed 3): the drums owned 52% of < 150 Hz but only 9% of
  1–5 kHz (violins + khuur 50%); hits were long soft booms (taiko 89% < 80 Hz, no attack) masked by the octobass drone
  (37% of the low band). Now: war drums re-voiced with a kick-like pitch drop (drop ≈ 1, drop_t ≈ 20–30 ms),
  shell knock 100–200 Hz, stick/mallet crack 1–6 kHz, 30–50% shorter ring; new `wtaiko` (punchy taiko, war only);
  `STYLES['war']['drum_gain']` 1.5 (applied in `Composer.next_bar`); octobass pedal 0.7–1.0 → 0.42–0.6; heartbeat
  frame lowered. Result: drums 19% of 1–5 kHz, 65% of < 150 Hz (octobass 17%). `ab/war_climax_clip.wav` (150–185 s).
  Timing measured while the machine was busy: 45–59% RT (was 34%); the score's cost didn't change (drums are cached).
- War feedback 3: "still too quiet". Energy shares were the wrong measure: ears discount lows heavily. Metric now
  (scratchpad `drum_audible.py`, worth keeping as a tool): **A-weighted drums vs the rest of the orchestra, after
  ducking, before the master compressor**, plus loudness with vs without drums. At drum_gain 1.0 the drums were
  −8.8 dB under the orchestra (inaudible); removing all drums changed the mix loudness by 0.5 dB.
  Now: big drums re-voiced again (`big_drum` in `Drums`: strong upper head modes, shell 100–180 Hz, thwack
  150 Hz–1.2 kHz, crack 1.5–6 kHz; 64–75% of the energy above 80 Hz, was 20–40%); big drums (`mix.BIG_DRUMS`:
  wtaiko, beopgo, odaiko, davul) on their own bus `bigdrum`; `STYLES['war']`: drum_gain 2.5 and
  duck=(6 dB, lo 0.4, hi 1.6): `Mixer.duck_under_drums` pulls everything but drums down by 0–6 dB following the big
  drums' level (60 ms release; mean cut 3.2 dB at the climax, 0.9 dB at the start). Result: drums vs orchestra
  −0.8 dB at the start → +2.3 dB at the climax. 42% RT; peaks ≤ −0.3 dBFS.
  `ab/war_opening_clip.wav` (15–45 s), `ab/war_climax_clip.wav` (150–185 s).
- War feedback 4: "change our bass to double bass; drums still quiet — they should be like our lead; more throat
  singing and chorus". Tagged buses: a note's `art['bus']` routes it to `"<inst>|<tag>"` (same stage/body as the
  instrument); the riff is on `|lead`, so levels can be measured against the lead line (scratchpad `war_levels.py`:
  A-weighted per group after ducking, before the master compressor, relative to the lead). Finding: "drums" had
  measured loud only because of small percussion — ringing metal above all (anvil 20%, tam-tam 16%, alarm bell 13%
  of it) — while the big war drums were ~6 dB under the lead. Now: octobass → bowed double basses (`War.low`,
  players 3; the octobass no longer plays in war, by request); metal/hooves/sticks/shime/chaos lowered;
  `STYLES['war']`: big_gain 2.2 (bigdrum bus only), duck=(6, 0.85, 3.2), trim 0.62. Voices: throat-singer drone
  from the start and louder, kargyraa `grunts` on the beats, a `sygyt` overtone line, `war_choir` (men's choir
  holding open fifths per chord, women above later), shouts earlier/every beat, the choir takes up the riff's long
  notes late. Result (seed 3, vs the lead line): big drums +2.6 dB at 20–50 s, +0.9 dB at 150–185 s (was −6);
  voices +2.9 / +0.7 dB (was absent / −4.4). 48% RT overall, worst fifth 68% (the choir's long notes).
  Old `taiko` no longer used in war (`wtaiko` replaces it).
- War feedback 5: "chorus and throat singers are quiet — take them to the front". Their level came from their low
  body; in 1–4 kHz (presence) they were −4.5 / −2.9 dB under the lead, sent 40–60% into the mountain reverb, and ducked
  with the orchestra. New per-style mixer options: `stage` (pan, gain, reverb send per instrument, via
  `mix.stage_of`), `duck_skip` (buses the drums don't duck), `presence` (dB, Hz, octaves: a linear-phase bell on the
  summed voices, `mix.presence_fir`, run like a body family). War: voices gain ≈ +1.5 dB over STAGE, sends 0.12–0.18,
  centred, not ducked, +7.5 dB at 2.8 kHz. Result (vs the lead line, A-weighted / 1–4 kHz): voices +5.9 / +0.9 dB at
  20–50 s, +6.0 / +4.4 dB at 150–185 s; big drums unchanged (+2.6 / +0.9). 48% RT, worst fifth 67%.
- Render cost (2026-11-25), engine-wide changes made for war, output statistically the same:
  `core.band_noise` (per-band filtered noise bank, 2^19 samples, random slices; was fresh fft_filter per note ≈30% of
  render), piecewise `env_adsr`, drums added mono to buses (`Bus.add_mono`), cached spectra of body/crossover/EQ
  filters (`Mixer.H`), and `Composer.next_bar` mixes bars shorter than 1.6 s in pairs (`SHORT_BAR`; per-bar info is
  split back out). War seed 3: 111% → 42% RT overall, worst fifth 160% → 61%; hearth 37%, ritual 18%.
  Live 4K window at war (vsync): 0 audio underruns (`Engine.underruns`, printed with `--seconds`).
- War scene (`cf.War`, GPU `GWar`): blood-red sky with a swollen sun, scrolling smoke, three mountain ranges with a
  crenellated wall and towers (lit windows) along the nearest ridge, a scorched plain with fires; riders at the gallop
  (procedural 8-frame silhouettes, 3 kinds: lance / sword / banner, `horse_sprites`; sampled by exact coverage from
  summed-area tables), arrow volleys on the beats (also down from the walls), flaming catapult shots on chord changes /
  big hits that burst on the walls and leave them burning. Simulated on the host in `War.step` (closed-form
  parabolas); GPU kernels `k_war_bg`, `k_riders`, `k_arrows`, `k_fireballs`, `k_debris`, `k_ash`.
  Intensity = `info['x']`. `Show.war`: the battlefield replaces the rotation while war plays, stays across war tracks,
  `n` is ignored (`Show.user_next`); not in `SCENES`. GPU bench: 0.61 ms scene, 0.94 ms total at 4K.
  Not yet: dust behind riders, stuck arrows, bodies, a closer look at the far plain (dark), ears on the music.

## Visual mood (2026-11-24)

- Rotation (`cf.SCENES`) is now the mood: `Summit` (a fire on a snowy ledge, peaks, Milky Way + a faint spiral
  galaxy behind scrolling clouds, a stray light ray aimed at the fire, edelweiss/gentians at snow-capped
  boulders, snow/blizzard) and `SnowCamp` (the campfire, night + snow). Old scenes stay in `ALL_SCENES`.
  `Campfire.frame_pines` (default True) turns off the framing pines (tools/story.py uses it).
- The music drives the visuals through new per-bar `info` fields → `Timeline` event tracks → `Ctx` fields
  (`Show._features`): lead (envelope + note pitch) opens the ray; bass breathes the firelight; `gust` is the
  same curve as the wind audio (snow, flowers, smoke, storm haze); `pop` = audible fire crackles → sparks;
  `pluck` = ember notes → sparks + star glints; `chord` changes part the clouds; `energy`; `weather`.
- Each track has a weather (`compose.WEATHER_ODDS`: clear / snowfall / storm). Audio: storms howl
  (narrowband noise ring-modulated onto a carrier that glides with the gust) + blizzard hiss; calm nights
  rarely get a distant wolf (into the valley reverb). Scenes ease to the weather live (`Summit._look`).
- Summit cost at 4K: ~1.1 ms/frame total (scene 0.7 ms).
- `tools/preview.py` renders a 1080p preview video with music (`--weather clear|snowfall|storm`).

## Likely next steps (discuss before big changes)

- CUDA graph per frame: done (`FrameGraph`, see Status). Next CPU-side candidates: scene Python per frame,
  `Show.context` (~80 µs), CUDA-GL map/unmap (~110 µs).
- Startup still sets up the first scene synchronously (~100 ms on the first frame).
- Per-frame host allocations to avoid: boat sprite (already uses a reused scratch buffer),
  HUD `assign` (small), exposure readback every 15 frames.
- Longer-term goal: **extremely realistic yet cheap scenes** using the cell grid as a
  budget — one thread per cell, expensive shading once per cell (physically based sky,
  ray-marched clouds/fog, accumulated GI), cheap sub-cell coverage to pick the glyph,
  temporal accumulation, only re-shading cells that change. Measure ms/frame and samples/cell,
  not just FPS. Don't over-commit to one architecture; realism + appeal + performance together.

## Don't

- Don't install anything globally or outside `.venv`. (`imageio-ffmpeg` is a dev dependency, added on
  purpose, only for tools/story.py — there is no system ffmpeg.)
- Don't change the music engine (`kor.music`) unless asked.
- Don't put new claude fm specifics into kor (scenes, the show, branding): kor is meant to carry the next app
  too (a side-scrolling base-defence game, design brief in progress). App code stays in `claudefm/`. (Exception
  as of the split: `kor.music` still holds claude fm's styles — the music moved whole, by request.)
