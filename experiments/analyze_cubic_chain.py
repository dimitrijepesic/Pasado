"""Analysis of the "cubic chain" in the precise path.

The cubic chain is the part of check_nonlinear_boundary_tensor that solves a
cubic equation per neuron (inverse_sigmoid_2nd_deriv and its helpers) and then
filters and evaluates the roots. This script measures where its time goes and
tests possible rewrites.

It does not modify precise_transformer.py. The rewrites are prototypes that
live only in this file, so they can be checked for bit-identity and timed
before anyone changes the real code.

Sections:
  1. capture   Record the real inputs of check_nonlinear_boundary_tensor during
               actual forward passes.
  2. stats     Measure how much of the work is done on values that end up NaN.
  3. stages    Time each stage of the chain on those real inputs.
  4. ops       Count the PyTorch operations in one call (torch.profiler).
  5. variants  Compare the prototypes with the original: bit-identity on real
               and synthetic inputs, plus speed per call and end to end.

Usage (from the Pasado repo root, venv active):
    python experiments/analyze_cubic_chain.py
    python experiments/analyze_cubic_chain.py --section stats stages
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import profile_precise_only as pp            # noqa: E402  (also sets sys.path and float64)
import precise_transformer as pt             # noqa: E402

NAN = float("nan")
# Bounds of the image of sigmoid''(x): its maximum is 1 / (6 * sqrt(3)). A / y
# outside [L, U] has no real solution, so the original code sets it to NaN.
L = -1.0 / (6.0 * np.sqrt(3.0))
U = -L


# --------------------------------------------------------------------------
# 1. capture
# --------------------------------------------------------------------------
def capture(n_images=2, eps_indices=(0, 4, 8, 12)):
    """Run real precise forwards and record every call's inputs.

    check_nonlinear_boundary_tensor is temporarily wrapped with a function that
    stores clones of its arguments (so later in-place edits cannot change them)
    and then calls the original. Returns the recorded calls, the network, the
    images and the epsilons used.
    """
    pp.set_flags("best")
    net = pp.ce.build_net()
    imgs = pp.load_images(n_images)
    test_range = [10 ** (-k / 4) * 2 for k in range(2, 18)]
    epsilons = [test_range[i] for i in eps_indices]

    rec = []
    orig = pt.check_nonlinear_boundary_tensor

    def spy(lx, ux, ly, uy, ABCs_tensor):
        rec.append(dict(lx=lx.clone(), ux=ux.clone(), ly=ly.clone(),
                        uy=uy.clone(), ABC=ABCs_tensor.clone(),
                        layer=len(rec) % 2, eps=pp_eps[0]))
        return orig(lx, ux, ly, uy, ABCs_tensor)

    pt.check_nonlinear_boundary_tensor = spy
    pp_eps = [None]
    try:
        for img in imgs:
            for e in epsilons:
                pp_eps[0] = e
                pp.one_forward(net, img, e)
    finally:
        pt.check_nonlinear_boundary_tensor = orig
    return rec, net, imgs, epsilons


# --------------------------------------------------------------------------
# 2. stats: where does the chain compute values that are discarded?
# --------------------------------------------------------------------------
def stats(rec):
    """Print how many neurons and root slots survive each filtering step.

    For each layer and each of the two branches (ly and uy) it reports the
    share of neurons whose A / y is inside the valid range, the share of the
    three root slots that are valid (inside (0, 1) before inverting the
    sigmoid), the share that also fall inside [lx, ux], and the share of
    neurons that keep at least one candidate.
    """
    print("\n=== 2. STATS on captured real inputs ===")
    print(f"captured calls: {len(rec)}  (neurons per call: {rec[0]['lx'].numel()})")
    agg = {}
    for c in rec:
        for tag, y in (("ly", c["ly"]), ("uy", c["uy"])):
            A = c["ABC"][:, 0]
            n = A.numel()
            ratio = A / y
            n_inf = int(torch.isinf(ratio).sum())
            n_zero_y = int((y == 0).sum())
            r = pt.inf_to_nan(ratio.clone())
            r = pt.filter_range(L, U, r)
            in_range = int((~torch.isnan(r)).sum())
            roots = pt.inverse_sigmoid_2nd_deriv(r.clone())
            valid_roots = [int((~torch.isnan(t)).sum()) for t in roots]
            filt = [pt.filter_range(c["lx"], c["ux"], t.clone()) for t in roots]
            surv = [int((~torch.isnan(t)).sum()) for t in filt]
            has_any = int((~torch.isnan(torch.stack(filt, 1))).any(1).sum())
            k = (c["layer"], tag)
            a = agg.setdefault(k, dict(n=0, inf=0, zero_y=0, in_range=0,
                                       valid=[0, 0, 0], surv=[0, 0, 0], any=0))
            a["n"] += n; a["inf"] += n_inf; a["zero_y"] += n_zero_y
            a["in_range"] += in_range; a["any"] += has_any
            for i in range(3):
                a["valid"][i] += valid_roots[i]; a["surv"][i] += surv[i]
    print(f"{'layer/y':9s} {'neurons':>8s} {'y==0':>6s} {'A/y in range':>13s} "
          f"{'roots in (0,1)':>16s} {'survive [lx,ux]':>17s} {'>=1 candidate':>14s}")
    for (layer, tag), a in sorted(agg.items()):
        n = a["n"]
        print(f"L{layer}/{tag}     {n:8d} {a['zero_y']:6d} "
              f"{100 * a['in_range'] / n:12.1f}% "
              f"{100 * sum(a['valid']) / (3 * n):15.1f}% "
              f"{100 * sum(a['surv']) / (3 * n):16.1f}% "
              f"{100 * a['any'] / n:13.1f}%")
    print("(percent of neurons, resp. of the 3n root slots; the whole chain is run on "
          "all n regardless)")


# --------------------------------------------------------------------------
# 3. stage timing
# --------------------------------------------------------------------------
def med_us(fn, reps=400, warm=30):
    """Median wall-clock time of fn() in microseconds, after warm-up calls."""
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def stages(rec):
    """Time each stage of one branch of the chain on a real captured call.

    The helper functions used here are idempotent (applying them twice gives the
    same result), so they can be timed repeatedly on the same input without
    copying it each time.
    """
    print("\n=== 3. STAGE TIMING (median us, one ly branch, real inputs) ===")
    c = rec[len(rec) // 2]
    lx, ux, ly = c["lx"], c["ux"], c["ly"]
    A = c["ABC"][:, 0]; B = c["ABC"][:, 1]; C = c["ABC"][:, 2]
    n = A.numel()

    ratio = A / ly
    r_in = pt.filter_range(L, U, pt.inf_to_nan(ratio.clone()))
    roots = pt.inverse_sigmoid_2nd_deriv(r_in.clone())
    r_f = [pt.filter_range(lx, ux, t.clone()) for t in roots]
    roots_x = torch.stack(r_f, 1).to(ly.dtype)
    ys = ly.unsqueeze(1)

    rows = [
        ("a  A/ly + inf_to_nan + filter_range(l,u)",
         lambda: pt.filter_range(L, U, pt.inf_to_nan(A / ly))),
        ("b  inverse_poly_tensor (complex64 Cardano)",
         lambda: pt.inverse_poly_tensor(r_in)),
        ("c  inverse_sigmoid_2nd_deriv (b + bool_to_nan x3 + inv_sigmoid x3)",
         lambda: pt.inverse_sigmoid_2nd_deriv(r_in)),
        ("d  filter_range(lx,ux) x3 on roots",
         lambda: [pt.filter_range(lx, ux, t) for t in roots]),
        ("e  stack + to(dtype) + max objective",
         lambda: pt._max_objective_over_x_candidates(
             A, B, C, torch.stack(r_f, 1).to(ly.dtype), ys)),
        ("   whole check_nonlinear_boundary_tensor (both branches)",
         lambda: pt.check_nonlinear_boundary_tensor(lx, ux, ly, c["uy"], c["ABC"])),
    ]
    res = {}
    for name, fn in rows:
        res[name] = med_us(fn)
        print(f"  {name:70s} {res[name]:9.1f} us")
    one_branch = sum(res[r[0]] for r in rows[:1]) + res[rows[2][0]] + res[rows[3][0]] + res[rows[4][0]]
    print(f"  a+c+d+e (one branch)={one_branch:.1f} us, x2 branches={2 * one_branch:.1f} us "
          f"vs whole={res[rows[5][0]]:.1f} us   [n={n}]")


# --------------------------------------------------------------------------
# 4. torch.profiler op counts
# --------------------------------------------------------------------------
def ops(rec):
    """Profile one call with torch.profiler and print operation counts.

    Counts include nested operations, so the total is larger than the number of
    top-level calls. Also lists the operations behind boolean-mask assignment,
    to show whether it uses nonzero (which would force a device sync on GPU).
    """
    print("\n=== 4. OP COUNT / TOP OPS for one check_nonlinear_boundary_tensor call ===")
    from torch.profiler import profile, ProfilerActivity
    c = rec[len(rec) // 2]
    args = (c["lx"], c["ux"], c["ly"], c["uy"], c["ABC"])
    for _ in range(20):
        pt.check_nonlinear_boundary_tensor(*args)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        pt.check_nonlinear_boundary_tensor(*args)
    ev = [e for e in prof.key_averages() if e.key.startswith("aten::")]
    total_calls = sum(e.count for e in ev)
    total_self = sum(e.self_cpu_time_total for e in ev)
    print(f"  distinct aten ops: {len(ev)}   total aten calls (incl. nested): {total_calls}   "
          f"sum self CPU: {total_self:.0f} us")
    ev.sort(key=lambda e: e.self_cpu_time_total, reverse=True)
    print(f"  {'op':32s} {'calls':>6s} {'self us':>9s}")
    for e in ev[:14]:
        print(f"  {e.key:32s} {e.count:6d} {e.self_cpu_time_total:9.0f}")
    idx = [e for e in ev if "index" in e.key or "nonzero" in e.key or "masked" in e.key]
    print("  mask-assignment related (index_put_/nonzero/masked*): "
          + ", ".join(f"{e.key} x{e.count}" for e in idx))


# --------------------------------------------------------------------------
# 5. variants (prototypes, live only here)
# --------------------------------------------------------------------------
def inf_to_nan_w(b):
    """inf_to_nan written with torch.where instead of in-place mask assignment."""
    return torch.where(torch.isinf(b), NAN, b)


def filter_range_w(l, u, x):
    """filter_range written with torch.where; also works for broadcast bounds."""
    guard = torch.logical_not(torch.logical_and(x <= u, x >= l))
    return torch.where(guard, NAN, x)


def bool_to_nan_w(x):
    """bool_to_nan with torch.where: 1.0 where x is real, NaN otherwise (float32)."""
    b1 = torch.isreal(x).float()
    return torch.where(b1 < 0.5, NAN, b1)


def roots_stacked(y, use_where):
    """[3, m] float32, equal to stack(inverse_sigmoid_2nd_deriv(y)) element-wise."""
    x1, x2, x3 = pt.inverse_poly_tensor(y)
    x = torch.stack((x1, x2, x3))                 # complex64 [3, m]
    b = bool_to_nan_w(x) if use_where else _bool_to_nan_orig(x)
    return pt.inv_sigmoid(x.real * b)


def _bool_to_nan_orig(x):
    """The original bool_to_nan, wrapped so roots_stacked can switch between versions."""
    return pt.bool_to_nan(x)


def cnb_variant(lx, ux, ly, uy, ABCs_tensor, merge, use_where):
    """check_nonlinear_boundary_tensor with: roots handled as one stacked tensor
    (always), ly/uy branches merged into one pass (merge), and mask assignment
    replaced by torch.where (use_where)."""
    A = ABCs_tensor[:, 0]; B = ABCs_tensor[:, 1]; C = ABCs_tensor[:, 2]
    i2n = inf_to_nan_w if use_where else pt.inf_to_nan
    flt = filter_range_w if use_where else pt.filter_range

    def branch_inputs(y):
        return flt(L, U, i2n(A / y))

    if merge:
        y_all = torch.cat((branch_inputs(ly), branch_inputs(uy)))
        roots = roots_stacked(y_all, use_where)                  # [3, 2n]
        lx2, ux2 = torch.cat((lx, lx)), torch.cat((ux, ux))
        roots = flt(lx2, ux2, roots)
        xs = roots.t().to(ly.dtype)                              # [2n, 3]
        ys = torch.cat((ly, uy)).unsqueeze(1)
        A2, B2, C2 = torch.cat((A, A)), torch.cat((B, B)), torch.cat((C, C))
        m = pt._max_objective_over_x_candidates(A2, B2, C2, xs, ys)
        n = A.numel()
        return m[:n], m[n:]

    outs = []
    for y in (ly, uy):
        roots = roots_stacked(branch_inputs(y), use_where)       # [3, n]
        roots = flt(lx, ux, roots)
        xs = roots.t().to(ly.dtype)
        outs.append(pt._max_objective_over_x_candidates(A, B, C, xs, y.unsqueeze(1)))
    return outs[0], outs[1]


def patched_helpers_where():
    """Swap the mask-assignment helpers in precise_transformer for the where versions.

    The structure of the original function is unchanged; only the three helpers
    are replaced. Returns the originals so restore_helpers can undo the swap.
    """
    saved = (pt.inf_to_nan, pt.filter_range, pt.bool_to_nan)
    pt.inf_to_nan, pt.filter_range, pt.bool_to_nan = inf_to_nan_w, filter_range_w, bool_to_nan_w
    return saved


def restore_helpers(saved):
    """Undo patched_helpers_where using the tuple it returned."""
    pt.inf_to_nan, pt.filter_range, pt.bool_to_nan = saved


def edge_cases(dtype=torch.float64):
    """Synthetic inputs aimed at the places where a rewrite could diverge."""
    g = torch.Generator().manual_seed(1)
    n = 64
    lx = torch.rand(n, generator=g) * 6 - 3
    ux = lx + torch.rand(n, generator=g) * 2
    ly = torch.rand(n, generator=g) * 4 - 2
    uy = ly + torch.rand(n, generator=g) * 2
    A = (torch.rand(n, generator=g) - 0.5) * 0.4
    B = torch.randn(n, generator=g)
    C = torch.randn(n, generator=g)
    cases = {}
    cases["random"] = (lx, ux, ly, uy, torch.stack((A, B, C), 1))
    # ly == 0 / uy == 0 -> A/y = inf -> NaN path
    ly0, uy0 = ly.clone(), uy.clone(); ly0[::3] = 0; uy0[1::3] = 0
    cases["y==0"] = (lx, ux, ly0, uy0, torch.stack((A, B, C), 1))
    # lx == ux (point in x)
    cases["lx==ux"] = (lx, lx.clone(), ly, uy, torch.stack((A, B, C), 1))
    # A/y exactly at +-1/(6 sqrt 3), the boundary of the discriminant
    edge = 1.0 / (6.0 * np.sqrt(3.0))
    Ae = torch.full((n,), 1.0) * edge * ly.abs().clamp_min(0.5) * torch.sign(ly)
    cases["A/ly at boundary"] = (lx, ux, ly.clone(), uy, torch.stack((Ae, B, C), 1))
    # A/y just outside / just inside the boundary
    for tag, f in (("just inside", 1 - 1e-7), ("just outside", 1 + 1e-7)):
        Ab = Ae * f
        cases[f"A/ly {tag}"] = (lx, ux, ly.clone(), uy, torch.stack((Ab, B, C), 1))
    # NaN in the data
    An = A.clone(); An[::5] = float("nan")
    cases["NaN in A"] = (lx, ux, ly, uy, torch.stack((An, B, C), 1))
    # huge / tiny A
    cases["tiny A"] = (lx, ux, ly, uy, torch.stack((A * 1e-12, B, C), 1))
    cases["wide x range"] = (lx - 50, ux + 50, ly, uy, torch.stack((A, B, C), 1))
    return cases


def variants(rec, net, imgs, epsilons):
    """Compare the prototypes with the original function.

    Three checks per variant: bit-identity on the captured real inputs and on
    the synthetic edge cases, speed of the function alone (n = 100), and speed
    of the whole 3layer precise forward pass (interleaved runs, best of 5),
    followed by bit-identity of the final output zonotope.
    """
    print("\n=== 5. VARIANTS (prototypes) ===")
    orig = pt.check_nonlinear_boundary_tensor

    def where_only(*a):
        s = patched_helpers_where()
        try:
            return orig(*a)
        finally:
            restore_helpers(s)

    cands = {
        "orig": orig,
        "orig+where(helpers)": where_only,
        "stack": lambda *a: cnb_variant(*a, merge=False, use_where=False),
        "stack+where": lambda *a: cnb_variant(*a, merge=False, use_where=True),
        "stack+merge": lambda *a: cnb_variant(*a, merge=True, use_where=False),
        "stack+merge+where": lambda *a: cnb_variant(*a, merge=True, use_where=True),
    }

    # --- bit identity: captured real inputs + edge cases
    print("-- bit-identity vs original (torch.equal on both outputs)")
    inputs = [(f"real#{i}", (c["lx"], c["ux"], c["ly"], c["uy"], c["ABC"]))
              for i, c in enumerate(rec)]
    inputs += list(edge_cases().items())
    # keep it readable: show all edge cases, summarize the real ones
    n_real = len(rec)
    for name, fn in cands.items():
        if name == "orig":
            continue
        bad_real, bad_edge = 0, []
        for tag, a in inputs:
            ref = orig(*[t.clone() for t in a])
            got = fn(*[t.clone() for t in a])
            same = (torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
                    and ref[0].dtype == got[0].dtype)
            # torch.equal treats NaN != NaN; outputs here are -inf/finite after nan_to_num
            if not same:
                if tag.startswith("real#"):
                    bad_real += 1
                else:
                    bad_edge.append(tag)
        print(f"  {name:22s} real inputs: {n_real - bad_real}/{n_real} identical   "
              f"edge cases differing: {bad_edge if bad_edge else 'none'}")

    # --- micro timing on real inputs of both layers
    print("-- micro timing, median us per call (n=100)")
    c = rec[len(rec) // 2]
    args = (c["lx"], c["ux"], c["ly"], c["uy"], c["ABC"])
    base = None
    for name, fn in cands.items():
        t = med_us(lambda: fn(*args), reps=300)
        base = base or t
        print(f"  {name:22s} {t:9.1f} us   x{base / t:5.2f} vs orig")

    # --- e2e: swap the function into the real transformer
    print("-- e2e: 3layer precise forward wall-clock (interleaved, min of 5 rounds)")
    n_fwd = len(imgs) * len(epsilons)

    def run_with(fn):
        pt.check_nonlinear_boundary_tensor = fn
        try:
            t0 = time.perf_counter()
            for _ in range(10):
                pp.run_all(net, imgs, epsilons)
            return (time.perf_counter() - t0) / (10 * n_fwd) * 1e3
        finally:
            pt.check_nonlinear_boundary_tensor = orig

    names = ["orig", "stack+where", "stack+merge+where"]
    best = {k: float("inf") for k in names}
    for _ in range(5):
        for k in names:
            best[k] = min(best[k], run_with(cands[k]))
    for k in names:
        print(f"  {k:22s} {best[k]:7.3f} ms/forward   x{best['orig'] / best[k]:5.3f} vs orig")

    print("-- e2e bit-identity of the final dual zonotope (same seed)")
    img, eps = imgs[0], epsilons[0]
    ref = pp.one_forward(net, img, eps)
    for k in names[1:]:
        pt.check_nonlinear_boundary_tensor = cands[k]
        try:
            out = pp.one_forward(net, img, eps)
        finally:
            pt.check_nonlinear_boundary_tensor = orig
        same = (torch.equal(ref.dual.centers, out.dual.centers)
                and torch.equal(ref.dual.generators, out.dual.generators))
        print(f"  {k:22s} dual centers+generators identical: {same}")


def main():
    """Capture real inputs once, then run the requested sections."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", nargs="+",
                    choices=["stats", "stages", "ops", "variants"],
                    default=["stats", "stages", "ops", "variants"])
    args = ap.parse_args()

    rec, net, imgs, epsilons = capture()
    print(f"torch={torch.__version__} threads={torch.get_num_threads()} "
          f"captured={len(rec)} calls")
    if "stats" in args.section:
        stats(rec)
    if "stages" in args.section:
        stages(rec)
    if "ops" in args.section:
        ops(rec)
    if "variants" in args.section:
        variants(rec, net, imgs, epsilons)


if __name__ == "__main__":
    main()
