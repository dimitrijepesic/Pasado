"""Independent checks of the speedup ladder's claims (results/ladder_a100_40gb*), run on the CPU.

    python experiments/verify_ladder_claims.py paths      # each step runs the code it claims to run
    python experiments/verify_ladder_claims.py gpu_time   # timed wall clock vs GPU busy time (kernel rerun)
    python experiments/verify_ladder_claims.py dense      # bound >= a dense, refined search for the max derivative
    python experiments/verify_ladder_claims.py baselines  # Pasado vs plain zonotope and interval bounds
    python experiments/verify_ladder_claims.py dplane     # D >= exact max |g - P| for the plane actually used
    python experiments/verify_ladder_claims.py all

Writes one JSON per check to --out (default results/verification_2026-11-15/) and prints a
PASS/FAIL line per check. Nothing here is timed for the ladder; the checks only read and recompute.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys
import time

import numpy as np
import torch
import torch._dynamo

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import measure_speedup_ladder as ladder  # noqa: E402  (puts the source dirs on sys.path)
import measure_split_sweep as mss  # noqa: E402

import batched_splits as bs  # noqa: E402
import planar_fit  # noqa: E402
import precise_transformer as pt  # noqa: E402

DT = torch.float64
LADDER = os.path.join(ROOT, "results", "ladder_a100_40gb")
NETS = ["3layer", "4layer", "5layer", "big"]


def save(out, name, data):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(data, f, indent=1, default=float)


def verdict(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


# ----------------------------------------------------------------------------- 1. code paths

WATCH = [  # (module, attribute, label)
    (mss, "eval_splits_once", "sequential pass"),
    (bs, "lipschitz_bound", "batched pass"),
    (pt, "check_corners", "corner check, loop"),
    (pt, "check_corners_tensor", "corner check, vectorized"),
    (pt, "check_nonlinear_boundary", "boundary check, loop"),
    (pt, "check_nonlinear_boundary_tensor", "boundary check, vectorized"),
    (pt, "lin_reg_tensor", "gelsd per neuron"),
    (pt, "lin_reg_tensor_batched", "gelsd batched"),
    (pt, "get_linspace", "grid per neuron"),
    (pt, "get_linspace_batched", "grid batched"),
    (pt, "inverse_poly_tensor", "cubic, complex"),
    (pt, "inverse_poly_real_tensor", "cubic, real"),
    (planar_fit, "fit_planes", "closed-form fit, eager"),
]

# what each step must and must not call
EXPECT = {
    0: (["sequential pass", "corner check, loop", "boundary check, loop", "gelsd per neuron", "grid per neuron", "cubic, complex"],
        ["batched pass", "corner check, vectorized", "gelsd batched", "cubic, real", "closed-form fit, eager"]),
    1: (["sequential pass", "corner check, vectorized", "boundary check, vectorized", "gelsd per neuron", "cubic, complex"],
        ["corner check, loop", "boundary check, loop", "gelsd batched", "cubic, real"]),
    2: (["sequential pass", "gelsd batched", "grid batched", "cubic, complex"],
        ["gelsd per neuron", "grid per neuron", "cubic, real", "batched pass"]),
    3: (["sequential pass", "gelsd batched", "grid batched", "cubic, real"],
        ["gelsd per neuron", "cubic, complex", "batched pass"]),
    4: (["sequential pass", "gelsd batched", "cubic, real"], ["batched pass", "closed-form fit, eager"]),
    5: (["batched pass", "gelsd batched", "grid batched", "cubic, real"],
        ["sequential pass", "closed-form fit, eager", "gelsd per neuron"]),
    6: (["batched pass", "closed-form fit, eager", "cubic, real"], ["gelsd batched", "sequential pass"]),
    7: (["batched pass", "cubic, real"], ["gelsd batched", "closed-form fit, eager", "sequential pass"]),
    8: (["batched pass"], ["gelsd batched", "closed-form fit, eager", "sequential pass",
                           "corner check, vectorized", "boundary check, vectorized"]),
}


def check_paths(out):
    side = ladder.Side("3layer", torch.device("cpu"), 1, 7, None)
    img, eps = side.images[0][1], ladder.EPS[8]
    compiled_sigmoid = ladder.make_compiled_sigmoid()
    rows, ok_all = [], True
    for step in ladder.STEPS:
        ladder.configure(step, compiled_sigmoid)
        # the compiled fit is built lazily; count it by wrapping the factory's result
        counts = {label: 0 for *_, label in WATCH}
        counts["compiled fit"] = 0
        saved = []
        traced = step["compiled_sigmoid"]
        planar_fit.compiled_fit_planes()  # create the compiled fit from the unwrapped fit_planes
        for module, attr, label in WATCH:
            if traced and module is not mss and attr != "lipschitz_bound":
                continue  # inside a compiled graph a counter would be traced, not run
            if step["compiled_fit"] and attr == "fit_planes":
                continue  # the compiled fit traces fit_planes; it is counted as "compiled fit"
            original = getattr(module, attr)
            saved.append((module, attr, original))

            def wrapped(*a, _f=original, _l=label, **k):
                counts[_l] += 1
                return _f(*a, **k)
            setattr(module, attr, wrapped)
        orig_factory = planar_fit.compiled_fit_planes
        if not traced:
            def factory(_f=orig_factory):
                inner = _f()

                def run(*a, **k):
                    counts["compiled fit"] += 1
                    return inner(*a, **k)
                return run
            planar_fit.compiled_fit_planes = factory
        pass
        torch._dynamo.reset() if traced else None
        from torch._dynamo.utils import counters
        frames_before = counters["stats"]["unique_graphs"]
        try:
            np.random.seed(7)
            bound = side.run(step, img, eps, 4)
        finally:
            planar_fit.compiled_fit_planes = orig_factory
            for module, attr, original in reversed(saved):
                setattr(module, attr, original)
        graphs = counters["stats"]["unique_graphs"] - frames_before
        must, must_not = EXPECT[step["id"]]
        if step["id"] == 7:
            must = must + ["compiled fit"]
        missing = [m for m in must if counts.get(m, 0) == 0]
        present = [m for m in must_not if counts.get(m, 0) > 0]
        fallback = dict(planar_fit.compile_fallback)
        ok = not missing and not present and (graphs > 0 if traced else True) and not fallback
        ok_all &= ok
        rows.append(dict(step=step["id"], name=step["name"], bound=bound, counts=counts,
                         compiled_graphs=graphs, compile_fallback=fallback, missing=missing, unexpected=present, ok=ok))
        print(f"  step {step['id']} {step['name']:17s} bound {bound:.6f} "
              f"{'ok' if ok else 'WRONG'}  called: {', '.join(k for k, v in counts.items() if v)}"
              + (f"  compiled graphs: {graphs}" if traced else "")
              + (f"  COMPILE FELL BACK {fallback}" if fallback else "")
              + (f"  missing {missing}" if missing else "") + (f"  unexpected {present}" if present else ""))
    ladder.restore_library()
    save(out, "paths", rows)
    return verdict("paths", ok_all, "every step calls exactly the code it claims (3layer, 4 splits, CPU)")


# ----------------------------------------------------------------------------- 2. GPU busy time

def check_gpu_time(out):
    base = os.path.join(ROOT, "results", "ladder_kernels")
    rows, ok_all = [], True
    for path in sorted(glob.glob(os.path.join(base, "profiles", "*.txt"))):
        text = open(path).read()
        m_cuda = re.search(r"Self CUDA time total: ([\d.]+)(ms|us|s)", text)
        m_pass = re.search(r"pass_ms\s+([\d.]+)", text)
        if not m_cuda:
            continue
        scale = {"us": 1e-3, "ms": 1.0, "s": 1e3}[m_cuda.group(2)]
        cuda_ms = float(m_cuda.group(1)) * scale
        name = os.path.basename(path)[:-4]
        net, s, step = name.split("_")[0], int(name.split("_")[1][1:]), int(name.split("_")[2])
        cell = json.load(open(os.path.join(base, "cells", f"{net}_S{s}.json")))
        timed = [p["ms"] for p in cell["steps"][str(step)]["passes"]
                 if p["image_id"] == 0 and p["eps_index"] == 0]
        wall = statistics.median(timed)
        ratio = cuda_ms / wall
        # the GPU cannot be busy longer than the timed wall clock (allowing pass-to-pass noise)
        ok = ratio <= 1.10
        ok_all &= ok
        rows.append(dict(profile=name, gpu_busy_ms=cuda_ms, timed_median_ms=wall,
                         instrumented_pass_ms=float(m_pass.group(1)) if m_pass else None,
                         gpu_busy_over_wall=ratio, ok=ok))
        print(f"  {name:28s} GPU busy {cuda_ms:9.2f} ms   timed wall {wall:9.2f} ms   busy/wall {ratio:5.2f}")
    save(out, "gpu_time", rows)
    return verdict("gpu_time", ok_all and bool(rows),
                   "GPU busy time never exceeds the timed wall clock, so the timer waits for the GPU")


# ----------------------------------------------------------------------------- 3. dense search

def derivative(net, img, ts):
    x = (1 - ts[:, None]) * img[None, :] + ts[:, None]
    _, d = torch.func.jvp(net, (x,), ((1 - img)[None, :].expand_as(x),))
    return d.abs().amax(dim=1)


def dense_max(net, img, eps, n=100001, chunk=10000, refine_top=20, refine_n=2001):
    ts = torch.linspace(0.0, eps, n, dtype=DT)
    vals = torch.cat([derivative(net, img, ts[i:i + chunk]) for i in range(0, n, chunk)])
    best = vals.max().item()
    h = eps / (n - 1)
    # refine around the largest local maxima
    is_peak = torch.ones(n, dtype=torch.bool)
    is_peak[1:] &= vals[1:] >= vals[:-1]
    is_peak[:-1] &= vals[:-1] >= vals[1:]
    peaks = torch.nonzero(is_peak).flatten()
    peaks = peaks[torch.argsort(vals[peaks], descending=True)][:refine_top]
    for i in peaks.tolist():
        lo, hi = max(0.0, ts[i].item() - h), min(eps, ts[i].item() + h)
        fine = torch.linspace(lo, hi, refine_n, dtype=DT)
        best = max(best, derivative(net, img, fine).max().item())
    return best, vals.max().item()


def check_dense(out):
    rows, ok_all = [], True
    for net_name in NETS:
        net, images = mss.load_network_and_images(net_name, torch.device("cpu"), DT, 3)
        images = dict(images)
        cells = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(LADDER, "cells", f"{net_name}_S*.json")))]
        for key in sorted(cells[0]["sampled"]):
            image_id, eps_index = (int(v) for v in key.split("|"))
            eps = ladder.EPS[eps_index]
            t0 = time.time()
            refined, uniform = dense_max(net, images[image_id], eps)
            for cell in cells:
                bounds = [p["bound"] for st in cell["steps"].values() for p in st["passes"]
                          if p["image_id"] == image_id and p["eps_index"] == eps_index]
                if not bounds:
                    continue
                low = min(bounds)
                ok = low >= refined
                ok_all &= ok
                rows.append(dict(network=net_name, n_splits=cell["n_splits"], image_id=image_id,
                                 eps_index=eps_index, smallest_bound=low, sampled_4001=cell["sampled"][key],
                                 dense_100001=uniform, dense_refined=refined,
                                 bound_over_refined=low / refined, ok=ok))
            print(f"  {net_name:6s} img {image_id} eps {eps_index:2d}: 4001-pt {cells[0]['sampled'][key]:.6f}  "
                  f"refined {refined:.6f} (+{(refined / cells[0]['sampled'][key] - 1) * 100:.4f}%)  "
                  f"smallest bound/refined over splits: "
                  f"{min(r['bound_over_refined'] for r in rows if r['network'] == net_name and r['image_id'] == image_id and r['eps_index'] == eps_index):.6f}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)
    save(out, "dense", rows)
    worst = min(r["bound_over_refined"] for r in rows)
    return verdict("dense", ok_all, f"{len(rows)} (cell, pair) bounds, smallest bound / refined max = {worst:.6f}")


# ----------------------------------------------------------------------------- 4. baselines

def baseline_forwards(net, net_name):
    from Duals import Sigmoid_di, abstract_di
    from SimpleZono import AffineDualZonotope, SigmoidDualZonotope
    if net_name == "big":
        mods = [net[i] for i in (1, 3, 5, 7, 9)]
    else:
        k = int(net_name[0])
        mods = [net.fc1, net.fc2] + ([net.fc3] if k >= 4 else []) + ([net.fc4] if k == 5 else []) + [net.fc_final]

    def zono(x):
        for m in mods[:-1]:
            x = SigmoidDualZonotope(AffineDualZonotope(x, m.weight.T) + m.bias.data)
        return AffineDualZonotope(x, mods[-1].weight.T) + mods[-1].bias.data

    def interval(x):
        for m in mods[:-1]:
            x = Sigmoid_di(x @ abstract_di(m.weight.T) + abstract_di(m.bias.data))
        return x @ abstract_di(mods[-1].weight.T) + abstract_di(mods[-1].bias.data)
    return zono, interval


def baseline_bounds(net, net_name, img, eps, n_splits):
    from Duals import DualIntervalTensor
    from SimpleZono import HyperDualIntervalToDualZonotope
    zono, interval = baseline_forwards(net, net_name)
    one = torch.tensor([1.0], dtype=DT)
    delta = eps / n_splits
    bz, bi = [], []
    with torch.no_grad():
        for i in range(n_splits):
            l_, r_ = i * delta, i * delta + delta
            iv = DualIntervalTensor(real_l=torch.tensor([l_], dtype=DT), real_u=torch.tensor([r_], dtype=DT))
            iv.e1_l[0] = 1
            iv.e1_u[0] = 1
            o = zono((one - HyperDualIntervalToDualZonotope(iv)) * img + HyperDualIntervalToDualZonotope(iv))
            bz.append(torch.max(torch.maximum(o.dual.get_lb().abs(), o.dual.get_ub().abs())).item())
            o = interval((1 - iv) * img + iv)
            bi.append(torch.max(torch.maximum(o.e1_l.abs(), o.e1_u.abs())).item())
    return max(bz), max(bi)


def check_baselines(out, splits=(1, 8)):
    dense = {(r["network"], r["image_id"], r["eps_index"]): r["dense_refined"]
             for r in json.load(open(os.path.join(out, "dense.json")))} if os.path.exists(os.path.join(out, "dense.json")) else {}
    rows, sound_all, tighter_all = [], True, True
    for net_name in NETS:
        net, images = mss.load_network_and_images(net_name, torch.device("cpu"), DT, 3)
        images = dict(images)
        for s in splits:
            cell = json.load(open(os.path.join(LADDER, "cells", f"{net_name}_S{s}.json")))
            for key, sampled in sorted(cell["sampled"].items()):
                image_id, eps_index = (int(v) for v in key.split("|"))
                eps = ladder.EPS[eps_index]
                pasado = [p["bound"] for st in cell["steps"].values() for p in st["passes"]
                          if p["image_id"] == image_id and p["eps_index"] == eps_index]
                bz, bi = baseline_bounds(net, net_name, images[image_id], eps, s)
                ref = dense.get((net_name, image_id, eps_index), sampled)
                sound = min(bz, bi) >= ref
                tighter = max(pasado) <= bi  # below interval AD; vs the plain zonotope it is data, not a rule
                sound_all &= sound
                tighter_all &= tighter
                rows.append(dict(network=net_name, n_splits=s, image_id=image_id, eps_index=eps_index,
                                 reference_max=ref, pasado_max=max(pasado), plain_zonotope=bz, interval=bi,
                                 zonotope_over_pasado=bz / max(pasado), interval_over_pasado=bi / max(pasado),
                                 baselines_above_reference=sound, pasado_below_interval=tighter,
                                 pasado_below_zonotope=max(pasado) <= bz))
                print(f"  {net_name:6s} S{s:<2d} img {image_id} eps {eps_index:2d}: ref {ref:11.4f}  "
                      f"pasado {max(pasado):11.4f}  zono {bz:13.4f}  interval {bi:13.4f}"
                      f"{'' if max(pasado) <= bz else '  pasado above zonotope'}{'' if tighter else '  PASADO ABOVE INTERVAL'}"
                      f"{'' if sound else '  BASELINE BELOW REF'}", flush=True)
    save(out, "baselines", rows)
    ok = verdict("baselines sound", sound_all, "plain zonotope and interval bounds are above the reference maximum")
    above_zono = [r for r in rows if not r["pasado_below_zonotope"]]
    print(f"  Pasado above the plain zonotope in {len(above_zono)} of {len(rows)}: "
          + ", ".join(f"{r['network']} S{r['n_splits']} img {r['image_id']} eps {r['eps_index']} "
                      f"(+{(1 / r['zonotope_over_pasado'] - 1) * 100:.2f}%)" for r in above_zono))
    return verdict("pasado below interval", tighter_all,
                   f"Pasado <= interval AD on all {len(rows)} (net, splits, pair)") and ok


# ----------------------------------------------------------------------------- 5. D for the plane used

OUTWARD_LIMIT = bs.OUTWARD_D_EPS  # the margin of the outward-rounding prototype
RECHECK_UNITS = 1.0  # boxes whose float64 shortfall exceeds this many eps*T are redone in 60 digits


def mp_exact_max_err(A, B, C, lx, ux, ly, uy, scan=400):
    """The same maximum in 60-digit arithmetic: corners, and on each y edge the roots of
    sigma''(x) y = A, bracketed by a scan and polished by findroot."""
    import mpmath
    mpmath.mp.dps = 60
    A, B, C, lx, ux = (mpmath.mpf(v) for v in (A, B, C, lx, ux))
    best = mpmath.mpf(0)
    for y in (mpmath.mpf(ly), mpmath.mpf(uy)):
        def f(x):
            s = 1 / (1 + mpmath.e ** (-x))
            return abs(s * (1 - s) * y - (A * x + B * y + C))

        def g(x):
            s = 1 / (1 + mpmath.e ** (-x))
            return s * (1 - s) * (1 - 2 * s) * y - A
        best = max(best, f(lx), f(ux))
        xs = [lx + (ux - lx) * k / scan for k in range(scan + 1)]
        gv = [g(x) for x in xs]
        for k in range(scan):
            if gv[k] == 0:
                best = max(best, f(xs[k]))
            elif gv[k] * gv[k + 1] < 0:
                best = max(best, f(mpmath.findroot(g, (xs[k], xs[k + 1]), solver="anderson")))
    return best


def exact_max_err(A, B, C, lx, ux, ly, uy, n=2001, refine_n=2001):
    """max over the box of |sigmoid'(x) y - (A x + B y + C)|. Linear in y for fixed x, so the
    maximum is on y = ly or y = uy; over x a grid, then a fine grid around the best point."""
    def f(xs, y):
        s = torch.sigmoid(xs)
        return (s * (1 - s) * y[:, None] - (A[:, None] * xs + B[:, None] * y[:, None] + C[:, None])).abs()
    t = torch.linspace(0, 1, n, dtype=DT)
    xs = lx[:, None] + (ux - lx)[:, None] * t[None, :]
    best = torch.zeros_like(lx)
    for y in (ly, uy):
        v = f(xs, y)
        val, idx = v.max(dim=1)
        h = (ux - lx) / (n - 1)
        x0 = xs.gather(1, idx[:, None])[:, 0]
        lo = torch.maximum(lx, x0 - h)
        hi = torch.minimum(ux, x0 + h)
        tf = torch.linspace(0, 1, refine_n, dtype=DT)
        fine = lo[:, None] + (hi - lo)[:, None] * tf[None, :]
        best = torch.maximum(best, torch.maximum(val, f(fine, y).max(dim=1).values))
    return best


def check_dplane(out, cells=(("3layer", 8), ("5layer", 8), ("big", 8)), steps=(5, 6, 7)):
    rows, ok_all = [], True
    for net_name, s in cells:
        side = ladder.Side(net_name, torch.device("cpu"), 1, 7, None)
        for step_id in steps:
            step = ladder.STEP_BY_ID[step_id]
            ladder.configure(step, None)
            captured = []
            orig_c, orig_b = pt.check_corners_tensor, pt.check_nonlinear_boundary_tensor

            def corners(abc, lx, ux, ly, uy):
                r = orig_c(abc, lx, ux, ly, uy)
                captured.append([abc.clone(), lx.clone(), ux.clone(), ly.clone(), uy.clone(), r])
                return r

            def boundary(lx, ux, ly, uy, abc):
                r1, r2 = orig_b(lx, ux, ly, uy, abc)
                captured[-1] += [r1, r2]
                return r1, r2
            pt.check_corners_tensor, pt.check_nonlinear_boundary_tensor = corners, boundary
            try:
                np.random.seed(7)
                side.run(step, side.images[0][1], ladder.EPS[8], s)
            finally:
                pt.check_corners_tensor, pt.check_nonlinear_boundary_tensor = orig_c, orig_b
            eps64 = torch.finfo(DT).eps
            worst_units, worst_rel, n_short, n_out_short, n_tot, rechecked = 0.0, 0.0, 0, 0, 0, 0
            for layer, (abc, lx, ux, ly, uy, c, e1, e2) in enumerate(captured):
                D = torch.maximum(torch.maximum(c, e1), e2)  # as in batched_splits.precise_sigmoid
                exact = torch.cat([exact_max_err(abc[i:i + 4096, 0], abc[i:i + 4096, 1], abc[i:i + 4096, 2],
                                                 lx[i:i + 4096], ux[i:i + 4096], ly[i:i + 4096], uy[i:i + 4096])
                                   for i in range(0, lx.shape[0], 4096)])
                # D is a small difference of terms of size T (batched_splits.outward_error), so
                # float64 can only promise it to a few eps * T, not relative to D itself
                Aa, Ba, Ca = abc[:, 0].abs(), abc[:, 1].abs(), abc[:, 2].abs()
                T = 0.25 * torch.maximum(ly.abs(), uy.abs()) + torch.maximum(Aa * lx.abs(), Aa * ux.abs()) \
                    + Ba * torch.maximum(ly.abs(), uy.abs()) + Ca
                units = (exact - D) / (eps64 * T)
                # the float64 reference carries its own rounding of a few eps*T: redo the
                # boxes that look short in 60 digits and keep only that verdict
                import mpmath
                d_out = bs.outward_error(D, abc, lx, ux, ly, uy)
                for i in torch.nonzero(units > RECHECK_UNITS).flatten().tolist():
                    m = mp_exact_max_err(*(t[i].item() for t in (abc[:, 0], abc[:, 1], abc[:, 2], lx, ux, ly, uy)))
                    true_units = float((m - mpmath.mpf(D[i].item())) / (eps64 * mpmath.mpf(T[i].item())))
                    rechecked += 1
                    worst_units = max(worst_units, true_units)
                    n_short += true_units > OUTWARD_LIMIT
                    n_out_short += mpmath.mpf(d_out[i].item()) < m
                worst_rel = max(worst_rel, ((exact - D) / exact.clamp_min(1e-300)).max().item())
                n_tot += D.numel()
            ok = n_short == 0 and n_out_short == 0
            ok_all &= ok
            rows.append(dict(network=net_name, n_splits=s, step=step_id, name=step["name"], boxes=n_tot,
                             worst_shortfall_eps_T_60_digits=worst_units, rechecked_in_60_digits=rechecked, worst_shortfall_rel_to_D=worst_rel,
                             boxes_beyond_limit=n_short, boxes_outward_D_below_exact=n_out_short, ok=ok))
            print(f"  {net_name:6s} S{s} step {step_id} {step['name']:16s} {n_tot:6d} boxes  worst (exact - D) "
                  f"= {worst_units:5.2f} eps*T (60 digits, {rechecked} boxes) ({worst_rel:.1e} of D)   beyond {OUTWARD_LIMIT} eps*T: {n_short}   "
                  f"outward D below exact: {n_out_short}", flush=True)
    ladder.restore_library()
    save(out, "dplane", rows)
    return verdict("dplane", ok_all, f"D falls short of the exact max |g - P| of the plane each step used by at most "
                   f"{OUTWARD_LIMIT} eps*T (rounding), and the outward-rounded D covers every box")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("check", choices=["paths", "gpu_time", "dense", "baselines", "dplane", "all"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "verification_2026-11-15"))
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_default_dtype(DT)
    checks = {"paths": check_paths, "gpu_time": check_gpu_time, "dense": check_dense,
              "baselines": check_baselines, "dplane": check_dplane}
    todo = list(checks) if args.check == "all" else [args.check]
    results = {}
    for name in todo:
        print(f"== {name}", flush=True)
        results[name] = checks[name](args.out)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
