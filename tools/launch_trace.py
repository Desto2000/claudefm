"""Prove a refactor didn't change what reaches the GPU. Pixel hashes can't: the particle kernels race, so frames
differ run to run. This records, with fixed seeds, for every scene × look (and a dissolve + recolour):
  - every kernel launch: kernel, launch size, scalar params, and each array param's shape, dtype and content hash
    (taken the first time that array is seen; seen arrays are kept alive so an id() is never reused),
  - the host-side scene state (every numpy array on the CPU scene) after setup and after the frames,
  - the final sample buffer and canvas, and a hash of 12 bars of music.

    uv run python tools/launch_trace.py record before/      # before the change
    uv run python tools/launch_trace.py record after/       # after it
    uv run python tools/launch_trace.py compare before/ after/

Expected noise (measured 2026-11-25 between two runs of the same code): the auto-exposure value passed to the
cell kernels differs in ~15 launches (Mountains, Lake: |Δ| ≤ 2e-5, it is read back from noisy frames), and frames
differ in a few rows of Mountains, Lake, Summit and SnowCamp. Everything else must be identical."""

from __future__ import annotations

import ast
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

H = lambda b: hashlib.blake2b(b, digest_size=12).hexdigest()  # noqa: E731


def record(out):
    import warp as wp

    import kor.launch as kl
    from claudefm import scenes as cf
    from claudefm.app import make_renderer
    from claudefm.gpu_scenes import GPU_OF
    from kor.audio import Timeline
    from kor.music import SR, Composer
    from kor.renderer import LOOKS

    os.makedirs(out, exist_ok=True)
    trace, seen = [], {}
    real = kl.launch

    def tag(v):
        if isinstance(v, wp.array):
            k = id(v)
            if k not in seen:
                seen[k] = (v, (str(v.shape), str(getattr(v.dtype, "__name__", v.dtype)),
                               H(np.ascontiguousarray(v.numpy()).tobytes())))
            return ("arr",) + seen[k][1]
        if isinstance(v, (bool, int, float, np.integer, np.floating)):
            return repr(v)
        try:
            return tuple(float(x) for x in v)
        except TypeError:
            return repr(type(v))

    def traced(owner, kernel, dim, inputs, device):
        name = getattr(getattr(kernel, "func", None), "__name__", None) or kernel.key
        trace.append((name, repr(dim), tuple(tag(v) for v in inputs)))
        return real(owner, kernel, dim, inputs, device)

    for m in list(sys.modules.values()):           # every module that imported launch by name
        if m is not None and getattr(m, "launch", None) is real:
            m.launch = traced

    def host_state(cpu):
        return H("|".join(k + ":" + H(np.ascontiguousarray(v).tobytes()) for k in sorted(vars(cpu))
                          if isinstance(v := getattr(cpu, k), np.ndarray)).encode())

    comp, tl = Composer(np.random.default_rng(1)), Timeline()
    mh = hashlib.blake2b(digest_size=12)
    for _ in range(12):
        audio, info = comp.next_bar()
        mh.update(np.ascontiguousarray(audio).tobytes())
        mh.update(repr(sorted((k, v if np.isscalar(v) else H(np.ascontiguousarray(v).tobytes()))
                              for k, v in info.items() if not isinstance(v, (list, dict)))).encode())
        tl.add(info)

    wp.init()
    dev = wp.get_preferred_device()
    r = make_renderer((3840, 2160), (8, 16), dev, 7, prefetch=False)
    pix = wp.zeros(r.OW * r.OH, dtype=wp.uint32, device=dev)
    results, images = {"music": mh.hexdigest()}, {}
    for S in cf.ALL_SCENES:
        for look in LOOKS:
            r.look = look
            cpu = S(np.random.default_rng(3))
            r.show.scene, r.show.prev = GPU_OF[S](cpu, dev), None
            n0 = len(trace)
            for i in range(30):
                c = r.show.context(i / 60, 1 / 60, r.SW, r.SH, int((3 + i / 60) * SR), tl, 1.0, 2, 4)
                if i == 1:
                    setup = host_state(cpu)
                r.frame(c, pix)
            wp.synchronize_device(dev)
            key = f"{S.__name__}/{look}"
            results[key] = dict(n=len(trace) - n0, setup=setup, end=host_state(cpu))
            images[key + "/buf"], images[key + "/pix"] = r.buf.numpy(), pix.numpy()
            print(key, results[key]["n"], "launches", flush=True)
    r.look = "ascii"                               # the show: a dissolve into the next scene, then a recolour
    for i in range(60):
        t = 10 + i / 60
        c = r.show.context(t, 1 / 60, r.SW, r.SH, int((3 + i / 60) * SR), tl, 1.0, 2, 4)
        if i == 5:
            r.show.next_scene(t)
        if i == 40:
            r.show.recolor()
        r.frame(c, pix, view=(1600, 900))
    wp.synchronize_device(dev)
    images["show/pix"] = pix.numpy()[: 1600 * 900]
    np.savez_compressed(f"{out}/images.npz", **{k.replace("/", "__"): v for k, v in images.items()})
    with open(f"{out}/state.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{k} {v}\n" for k, v in results.items())
    with open(f"{out}/launches.txt", "w", encoding="utf-8") as f:
        f.writelines(repr(t) + "\n" for t in trace)
    print(f"{len(trace)} launches · music {results['music']} → {out}")


def compare(a, b):
    sa, sb = open(f"{a}/state.txt", encoding="utf-8").read(), open(f"{b}/state.txt", encoding="utf-8").read()
    print("host state + music:", "identical" if sa == sb else "DIFFERENT")
    la = open(f"{a}/launches.txt", encoding="utf-8").read().splitlines()
    lb = open(f"{b}/launches.txt", encoding="utf-8").read().splitlines()
    if len(la) != len(lb):
        print(f"launch count differs: {len(la)} vs {len(lb)}")
    kinds = {}
    for x, y in zip(la, lb):
        if x != y:
            X, Y = ast.literal_eval(x), ast.literal_eval(y)
            idx = tuple(i for i, (p, q) in enumerate(zip(X[2], Y[2])) if p != q)
            kinds.setdefault((X[0], X[0] == Y[0] and X[1] == Y[1], idx), []).append(1)
    print(f"{len(la)} launches: {sum(len(v) for v in kinds.values())} differ")
    for (k, same, idx), v in kinds.items():
        print(f"  {k}: {len(v)} launches, " + (f"params {idx} differ" if same else "kernel/size differ"))
    ia, ib = np.load(f"{a}/images.npz"), np.load(f"{b}/images.npz")
    for k in ia.files:
        d = ia[k] != ib[k]
        n = int(d.reshape(d.shape[0], -1).any(1).sum()) if d.ndim > 1 else int(d.sum())
        if n:
            print(f"  frames {k}: {n} {'rows' if d.ndim > 1 else 'pixels'} differ")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "record":
        record(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
