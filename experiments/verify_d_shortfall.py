"""Checks the claim that the error term D falls short of the exact maximum by at most 2.4 ulp
on 51% of the boxes (results/d_shortfall_verification_2026-11/PROTOCOL.md).

  python experiments/verify_d_shortfall.py [--csv results/compiled_sigmoid_v3_cuda_2026-10-29/gate_d_contract.csv]
      [--out results/d_shortfall_verification_2026-11]
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys

import mpmath
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, HERE)
EPS64 = 2.0 ** -52


def term_size(A, B, C, lx, ux, ly, uy):
    """max over the box of 0.25|y| + |A x| + |B y| + |C| (convex, so at a corner)."""
    return max(0.25 * abs(y) + abs(A * x) + abs(B * y) + abs(C) for x in (lx, ux) for y in (ly, uy))


def check1(rows):
    out = {}
    for side in ("eager", "compiled"):
        short = [r for r in rows if float(r[f"{side}_shortfall_abs"]) > 0]
        ulps_eps, ulps_ulp = [], []
        for r in rows:
            g = float(r[f"{side}_shortfall_abs"])
            T = term_size(*(float(r[f"{side}_{k}"]) for k in ("A", "B", "C", "lx", "ux", "ly", "uy")))
            ulps_eps.append(g / (EPS64 * T))
            ulps_ulp.append(g / math.ulp(T))
        pos_eps = [u for u in ulps_eps if u > 0]
        pos_ulp = [u for u in ulps_ulp if u > 0]
        out[side] = dict(boxes=len(rows), below=len(short), below_pct=100 * len(short) / len(rows),
                         max_ulp_eps_def=max(ulps_eps), median_ulp_eps_def_all=statistics.median(ulps_eps),
                         median_ulp_eps_def_below=statistics.median(pos_eps) if pos_eps else 0.0,
                         max_ulp_ulp_def=max(ulps_ulp), median_ulp_ulp_def_all=statistics.median(ulps_ulp),
                         median_ulp_ulp_def_below=statistics.median(pos_ulp) if pos_ulp else 0.0,
                         max_shortfall_abs=max(float(r[f"{side}_shortfall_abs"]) for r in rows),
                         max_shortfall_over_W=max(float(r[f"{side}_shortfall_over_W"]) for r in rows))
    return out


def edge_max_independent(A, B, C, lx, ux, y0, grid=2001, dps=50):
    """max |sigma'(x) y0 - (A x + B y0 + C)| on [lx, ux]: float64 grid for brackets, then
    golden-section refinement in mpmath of the best three brackets (no cubic roots)."""
    xs = np.linspace(lx, ux, grid)
    s = 1.0 / (1.0 + np.exp(-xs))
    f = np.abs(s * (1 - s) * y0 - (A * xs + B * y0 + C))
    with mpmath.workdps(dps):
        Am, Bm, Cm, y = (mpmath.mpf(v) for v in (A, B, C, y0))

        def err(x):
            sg = 1 / (1 + mpmath.exp(-x))
            return abs(sg * (1 - sg) * y - (Am * x + Bm * y + Cm))
        best = max(err(mpmath.mpf(lx)), err(mpmath.mpf(ux)))
        if grid > 2 and ux > lx:
            order = np.argsort(-f)
            for i in order[:3]:
                a, b = mpmath.mpf(xs[max(i - 1, 0)]), mpmath.mpf(xs[min(i + 1, grid - 1)])
                gr = (mpmath.sqrt(5) - 1) / 2
                c, d = b - gr * (b - a), a + gr * (b - a)
                fc, fd = err(c), err(d)
                for _ in range(160):
                    if fc > fd:
                        b, d, fd = d, c, fc
                        c = b - gr * (b - a); fc = err(c)
                    else:
                        a, c, fc = c, d, fd
                        d = a + gr * (b - a); fd = err(d)
                best = max(best, fc, fd, err(mpmath.mpf(xs[i])))
        return best


def check2(rows, n_top=200, n_random=200, seed=2026):
    idx_top = sorted(range(len(rows)), key=lambda i: -max(float(rows[i]["eager_shortfall_over_W"]),
                                                          float(rows[i]["compiled_shortfall_over_W"])))[:n_top]
    rest = [i for i in range(len(rows)) if i not in set(idx_top)]
    rng = np.random.default_rng(seed)
    idx = idx_top + sorted(rng.choice(rest, size=min(n_random, len(rest)), replace=False).tolist())
    worst_excess, mismatches = 0.0, []
    for i in idx:
        r = rows[i]
        for side in ("eager", "compiled"):
            A, B, C, lx, ux, ly, uy = (float(r[f"{side}_{k}"]) for k in ("A", "B", "C", "lx", "ux", "ly", "uy"))
            indep = max(edge_max_independent(A, B, C, lx, ux, ly), edge_max_independent(A, B, C, lx, ux, uy))
            d_true = mpmath.mpf(r[f"{side}_D_true"])
            if d_true > 0:
                worst_excess = max(worst_excess, float((indep - d_true) / d_true))
            short_csv = float(r[f"{side}_shortfall_abs"]) > 0
            short_new = float(indep) > float(r[f"{side}_D"])
            if short_csv != short_new:
                mismatches.append(dict(row=i, side=side, D=float(r[f"{side}_D"]), D_true=float(d_true),
                                       independent=float(indep)))
    return dict(boxes_checked=len(idx), largest_excess_of_independent_over_D_true_rel=worst_excess,
                classification_mismatches=len(mismatches), mismatch_examples=mismatches[:5])


def check3(rows):
    import precise_transformer as pt
    pt.USE_VECTORIZED_PRECISE = pt.USE_VECTORIZED_BOUNDARY = pt.USE_REAL_CUBIC = True
    torch.set_default_dtype(torch.float64)
    col = lambda k: torch.tensor([float(r[f"eager_{k}"]) for r in rows], dtype=torch.float64)  # noqa: E731
    A, B, C, lx, ux, ly, uy = (col(k) for k in ("A", "B", "C", "lx", "ux", "ly", "uy"))
    abc = torch.stack((A, B, C), dim=1)
    corners = pt.check_corners_tensor(abc, lx, ux, ly, uy)
    e_ly, e_uy = pt.check_nonlinear_boundary_tensor(lx, ux, ly, uy, abc)
    D_cpu = torch.maximum(torch.maximum(corners, e_ly), e_uy)
    D_cuda = col("D")
    D_true = [float(r["eager_D_true"]) for r in rows]
    below = sum(1 for d, t in zip(D_cpu.tolist(), D_true) if t > d)
    ulps = []
    for r, d, t in zip(rows, D_cpu.tolist(), D_true):
        T = term_size(*(float(r[f"eager_{k}"]) for k in ("A", "B", "C", "lx", "ux", "ly", "uy")))
        ulps.append(max(0.0, t - d) / (EPS64 * T))
    return dict(boxes=len(rows), cpu_equal_to_cuda_bitwise=int((D_cpu == D_cuda).sum()),
                cpu_max_rel_diff_to_cuda=float(((D_cpu - D_cuda).abs() / D_cuda.abs().clamp_min(1e-300)).max()),
                cpu_below_D_true=below, cpu_below_pct=100 * below / len(rows), cpu_max_ulp_eps_def=max(ulps),
                cpu_median_ulp_eps_def_all=statistics.median(ulps))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "compiled_sigmoid_v3_cuda_2026-10-29",
                                                  "gate_d_contract.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "d_shortfall_verification_2026-11"))
    ap.add_argument("--n-top", type=int, default=200)
    ap.add_argument("--n-random", type=int, default=200)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    result = dict(source=os.path.relpath(args.csv, ROOT), torch=torch.__version__, mpmath=mpmath.__version__)
    result["check1_recount"] = check1(rows)
    print(json.dumps(result["check1_recount"], indent=1), flush=True)
    result["check3_cpu_recompute"] = check3(rows)
    print(json.dumps(result["check3_cpu_recompute"], indent=1), flush=True)
    result["check2_independent_oracle"] = check2(rows, args.n_top, args.n_random)
    print(json.dumps(result["check2_independent_oracle"], indent=1), flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "result.json"), "w") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
