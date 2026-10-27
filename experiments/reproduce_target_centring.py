"""The plane fit used to depend on the order of summation on narrow boxes.

planar_fit.fit_planes (and candidates A1, A2, H) formed the slopes from the targets
centred on their computed mean, dz = z - fl(mean(z)). On a narrow axis the variation
of z along it (about 1e-13) sits next to an offset of about 0.1 that cancels in the
sum, so one ulp of the offset is 1e-4 of the signal: the result depends on the order
in which the reduction adds. Eager PyTorch on the CPU happened to add in a lucky
order on many boxes; TorchInductor adds differently and exposed it. The fix centres
z on a data point along each axis first (z_ij - z_0j, z_ij - z_i0), which is exact
and small, as is already done for x and y.

For every box whose rank the fit keeps exact, the distance to the exact
minimum-norm solution (lstsq_oracle.exact_min_norm) is compared with
32 * eps * (1 + max|x| + max|y|): a few ulps of the size of the data, which bounds how
rounding in the means and the slopes enters the intercept (with |xbar| + |ybar| in place
of the maxima the bound is too tight when a wide y range is centred near 0). The old
targets exceed it by many orders of magnitude, the new ones do not.

Run:
  python experiments/reproduce_target_centring.py
  python experiments/reproduce_target_centring.py --device cuda
  python experiments/reproduce_target_centring.py --no-compile
"""
import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))
sys.path.insert(0, HERE)

import planar_fit as pf  # noqa: E402
import compare_lstsq_solvers as h  # noqa: E402
import lstsq_candidates as lc  # noqa: E402
import lstsq_oracle as orc  # noqa: E402

BOUND = 32
FAMILIES = ["offset_narrow", "cutoff_band_x", "cutoff_band_y", "mixed_scale"]
SEED = 5150


def _fit_targets_mean_centred(st, z, xdeg, ydeg):
    """The targets as before the fix: centred on fl(mean(z))."""
    n, s = z.shape[0], st.s
    z3 = z.reshape(n, s, s)
    zbar = z.mean(1)
    dz = z3 - zbar[:, None, None]
    az = pf._pow2_scale(dz.reshape(n, s * s))
    zx = dz.mean(2) / az[:, None]
    zy = dz.mean(1) / az[:, None]
    zero = torch.zeros_like(zbar)
    a0 = torch.where(xdeg, zero,
                     (st.dxn * zx).sum(1) / pf._safe(st.nxx, xdeg) * az / pf._safe(st.ax, xdeg))
    b0 = torch.where(ydeg, zero,
                     (st.dyn * zy).sum(1) / pf._safe(st.nyy, ydeg) * az / pf._safe(st.ay, ydeg))
    k = zbar - a0 * st.xbar - b0 * st.ybar
    return pf._min_norm(k, a0, b0, st.xbar, st.ybar, xdeg, ydeg)


def fit_before(grid, z):
    st = pf._stats(grid)
    cx, cy = pf.gelsd_cutoff_axes(st, pf.default_rcond(grid.dtype, grid.shape[1]))
    return _fit_targets_mean_centred(st, z, st.xdeg | cx, st.ydeg | cy)


def worst_ratio(solve, grid, z, keep, scale, eps, exact):
    coef = solve(grid, z).cpu()
    ratios = [orc.relative_error(coef[i].tolist(), exact[i]) / (eps * scale[i])
              for i in range(grid.shape[0]) if keep[i]]
    return max(ratios), sum(r > BOUND for r in ratios), len(ratios)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-compile", action="store_true", help="eager only")
    args = ap.parse_args()
    device = torch.device(args.device)
    h.set_production_flags()

    solvers = {"before, eager": fit_before, "after, eager": pf.fit_planes}
    if not args.no_compile:
        solvers["before, compiled"] = torch.compile(fit_before, fullgraph=True)
        solvers["after, compiled"] = torch.compile(pf.fit_planes, fullgraph=True)

    exceeded = {name: 0 for name in solvers}
    for dtype_name in ("float64", "float32"):
        if device.type == "mps" and dtype_name == "float64":
            continue
        dtype = getattr(torch, dtype_name)
        eps = torch.finfo(dtype).eps
        for k, family in enumerate(FAMILIES):
            b = h.offset_narrow_bounds(dtype_name) if family == "offset_narrow" else \
                h.synthetic_bounds(family, 300, SEED + k, dtype_name)
            grid, z = h.grid_and_targets(*(t.to(device=device, dtype=dtype) for t in b))
            keep = (lc.chosen_rank("A2", grid, z) == lc.exact_rank(grid).cpu()).tolist()
            if not any(keep):
                print(f"{dtype_name:8s} {family:14s} no box keeps its exact rank")
                continue
            s = math.isqrt(grid.shape[1])
            xs, ys, zl = grid[:, ::s, 0].tolist(), grid[:, :s, 1].tolist(), z.tolist()
            exact = {i: orc.exact_min_norm(xs[i], ys[i], zl[i])
                     for i in range(grid.shape[0]) if keep[i]}
            scale = (1 + grid[..., 0].abs().amax(1) + grid[..., 1].abs().amax(1)).tolist()
            line = f"{dtype_name:8s} {family:14s}"
            for name, solve in solvers.items():
                worst, over, n = worst_ratio(solve, grid, z, keep, scale, eps, exact)
                exceeded[name] += over
                line += f" | {name}: worst {worst:8.1e}, {over:3d}/{n} over"
            print(line, flush=True)

    print(f"\nboxes over the bound {BOUND} eps (1 + max|x| + max|y|): {exceeded}")
    before = [v for name, v in exceeded.items() if name.startswith("before")]
    after = [v for name, v in exceeded.items() if name.startswith("after")]
    reproduced = all(v > 0 for v in before) and all(v == 0 for v in after)
    print("REPRODUCED: the mean-centred targets exceed the bound, the pivot-centred ones do not"
          if reproduced else "NOT REPRODUCED")
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
