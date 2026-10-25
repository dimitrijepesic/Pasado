"""Pre-registered comparison of replacements for the batched gelsd plane fit.

The selection rule below was fixed before the first measurement and committed on
its own; the results are read against it, not the other way round.

--smoke runs every level on a handful of cases with other seeds to check that the
four code paths (box, network, timing, decide) execute. It is not a measurement
and cannot decide: its timing runs on the local device, R4 needs the A100, so a
smoke run ends with "Undecided at R4". The same holds for any run without the
Colab timing file.

Candidates (experiments/lstsq_candidates.py)
  G   gelsd on CPU: the current code, and the baseline of G1 and R1
  A1  closed form; degenerate axis = equal endpoints
  A2  closed form; degenerate axis = gelsd's cutoff, computed analytically. It
      matches gelsd's rank decision almost always but not bit for bit at the
      threshold, and treating the dropped axis as constant is close to, not
      identical with, gelsd's truncated-SVD projection.
  A3  closed form on the s values of sigmoid'(x); never reads z (separable targets)
  H   A1, plus CPU gelsd on the boxes gelsd truncates beyond A1 (one host sync)
  P   torch.linalg.pinv with rtol = gelsd's rcond (one host sync per call)
  On a Mac, P runs LAPACK gesdd on the CPU, not cuSOLVER gesvdj, so every result
  for P is provisional until the box level has run on CUDA.

Case set (fixed)
  Box level, float64 and float32, master seed 20261025, family k uses seed
  20261025 + k, 500 boxes per synthetic family:
     0 normal           lx, ly ~ U(-4, 4), widths ~ U(1e-3, 4)
     1 degenerate_x     ux = lx
     2 degenerate_y     uy = ly (every first-layer box is like this)
     3 point            ux = lx and uy = ly
     4 near_degenerate  both widths 1e-12
     5 saturated        lx ~ U(25, 35)
     6 mixed_scale      x near 0 with width 1e-8, ly ~ U(-1e6, 1e6), y width ~ U(1, 1e6)
     7 cutoff_band_x    x width = r * (gelsd cutoff width predicted for the box),
                        r log-uniform in [0.1, 10], x centre ~ U(-20, 20)
     8 cutoff_band_y    the same for y
     9 large_magnitude  |centre| of x, y or both in {1e4, 1e8, 1e16, 1e100, 1e155, 1e300}
                        (float32: {1e4, 1e8, 1e16, 1e19, 1e20, 1e30}), width 0 or
                        1e-6 * |centre|, random sign
    10 real             boxes recorded from the batched forward pass (batched_splits.py,
                        seed 7, n_splits 4) on 3layer, 4layer, 5layer and big, the first
                        2 correctly classified images, eps_idx 0, 8 and 15, every
                        sigmoid layer. Per network 500 drawn at random plus up to 500
                        of those gelsd truncates beyond their exact rank. float32
                        casts the recorded float64 bounds.
    11 nonfinite        8 fixed boxes with NaN, +-inf or lower > upper; informational
  Grid, targets and the max-error check are the production code
  (get_linspace_batched, the targets of batched_splits.precise_sigmoid,
  check_corners_tensor and check_nonlinear_boundary_tensor with the real cubic).
  Every candidate gets the same seeded N(0, 1e-4) noise on A before the error
  check, as in production.

  Network level, float64 on the CPU (Colab repeats it on CUDA): 3layer, 4layer,
  5layer and big; the first 3 correctly classified images; eps_idx 0, 4, 8, 12
  and 15; n_splits 1, 4 and 16 (chunk 4); seed 7, so every candidate sees the same
  noise on A. It runs only the candidates that pass G0-G2 (plus G), read from the
  box-level results in the same --out directory. It refuses to run without them,
  if they were produced by other code (hash of the source files) or in the other
  smoke mode, or if a candidate has no result there. --force-candidates overrides
  the selection; the run is then marked as outside the rule.

Gates: a candidate that fails one is out.
  G0 implementation checks C1-C7 of experiments/lstsq_contract.py (output contract,
     batch independence, z scaling, axis swap, normal equations where the
     candidate keeps the exact rank, minimum norm on degenerate axes, exact
     oracle), on boxes where no rank decision is ambiguous. Before any of this
     counts, the self-test runs the planted bugs S1-S10 of lstsq_contract.py
     through C1-C7, G1, G2 and a reduced box level (families 0-9, 50 boxes each,
     seeds 20261025 + 200 + k): each must fail C1-C7, G1 or G2, or rank strictly
     below its parent on R2a or R3. If one survives, the box level stops before
     the measurement. (R2b is not used there: its order-of-magnitude buckets can
     split two equal planes by rounding alone.)
  G1 numerical safety regression against gelsd. This is not a soundness proof:
     Pasado's error term is computed in floating point, and the float32 pipeline
     falls short of the true maximum with gelsd itself (see
     experiments/reproduce_f32_root_shortfall.py). Per box, the shortfall of the
     computed max error D below the true maximum of
     |sigmoid'(x) y - (A x + B y + C)| over the box for the emitted (A, B, C),
     evaluated with mpmath at 50 digits (corners plus the exact roots on both
     y-edges), relative to W (defined under R2): the missing part of the neuron's
     radius. Pass if, per family and dtype, the worst shortfall is at most
     10 * max(tau, worst shortfall of G); tau = 1e-12 (float64), 1e-5 (float32):
     not an order of magnitude less safe than the current code. Relative to D
     itself the shortfall is rounding noise whenever D is far below the size of
     the terms (D ~ 1e-11 next to A x ~ 1e5). Reported but not judged: the number
     of boxes with any shortfall, the worst absolute shortfall, and the worst
     shortfall relative to D. Boxes whose true maximum cannot be computed (root
     finder does not converge) are listed, not counted.
  G2 finite. No NaN/inf coefficient or D and no A == 0 after the noise on
     families 0-8 and 10; on family 9 no more non-finite boxes than G.
  G3 network soundness. Every per-split maximum >= the sampled autograd maximum of
     the derivative along the haze path (1001 points), at every n_splits.
  G0-G3 filter out errors; they do not rank. Every plane is sound in real
  arithmetic once D is exact, so they cannot tell correct candidates apart.

Ranking of the survivors: lexicographic; a criterion decides only when the ones
before it tie.
  R1 tightness. Worst relative excess of a per-split bound over G's, across the
     network level. |excess| <= 1e-9 (the project's numerical contract) counts as 0.
  R2 robustness on families 4, 6, 7, 8 and 9, both dtypes:
     a) W = |A| r_x + |B| r_y + D is the radius a neuron's box adds. Per box,
        W / W_best, where W_best is the smallest W any surviving candidate reaches
        on that box; an excess W - W_best within 16 ulps of the size of the terms
        (0.25 |y| + |A x| + |B y| + |C| of the best plane) counts as none, since E
        is evaluated with about 10 roundings of terms of that size. Worst 99th
        percentile over these families and dtypes, rounded to 2 decimals.
        (Not W / W_G: that would make G win by definition.)
     b) worst shortfall of D as in G1, by order of magnitude.
  R3 accuracy. Median distance to the exact minimum-norm solution (Fraction
     oracle, rank decided exactly) on the boxes where gelsd's rank equals the
     exact rank, worst over families and dtypes, by order of magnitude.
  R4 speed. Median solver time on the A100, n = 65536, float64, in steps of 20%.
     Pending until the Colab run.
  R5 simplicity. No host sync and no CPU work (A1, A2, A3) before one host sync
     (H, P) before the CPU solve (G).

Expected outcome, written down so that a tie is not read as "nothing found": the
noise N(0, 1e-4) on A is about eight orders of magnitude larger than the
differences between solvers on real boxes (<= 1.2e-13 in the final bound), so R1
will almost certainly tie unless a candidate breaks (NaN, huge coefficients).
The winner is then decided by R2 (hard boxes and float32) and after that by R4.

Run:
  python experiments/compare_lstsq_solvers.py --smoke
  python experiments/compare_lstsq_solvers.py --level box --out DIR
  python experiments/compare_lstsq_solvers.py --level network --out DIR
  python experiments/compare_lstsq_solvers.py --level timing --device cuda --out DIR
  python experiments/compare_lstsq_solvers.py --level decide --out DIR
"""
import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))
sys.path.insert(0, HERE)

import precise_transformer as pt  # noqa: E402
import batched_splits as bs  # noqa: E402
import lstsq_candidates as lc  # noqa: E402
import lstsq_contract as ct  # noqa: E402
import lstsq_oracle as orc  # noqa: E402

MASTER_SEED = 20261025
EPS_SCHEDULE = [10 ** (-k / 4) * 2 for k in range(2, 18)]
FAMILIES = ["normal", "degenerate_x", "degenerate_y", "point", "near_degenerate",
            "saturated", "mixed_scale", "cutoff_band_x", "cutoff_band_y",
            "large_magnitude", "real", "nonfinite"]
MUST_BE_FINITE = set(FAMILIES[:9]) | {"real"}
HARD = ["near_degenerate", "mixed_scale", "cutoff_band_x", "cutoff_band_y", "large_magnitude"]
INFORMATIONAL = {"nonfinite"}
TAU = {"float64": 1e-12, "float32": 1e-5}
CONTRACT = 1e-9
LARGE = {"float64": [1e4, 1e8, 1e16, 1e100, 1e155, 1e300],
         "float32": [1e4, 1e8, 1e16, 1e19, 1e20, 1e30]}
SOURCES = [os.path.join(HERE, f) for f in ("compare_lstsq_solvers.py", "lstsq_candidates.py",
                                           "lstsq_contract.py", "lstsq_oracle.py")] + \
          [os.path.join(ROOT, "forward_mode_tensorized_src", f)
           for f in ("precise_transformer.py", "batched_splits.py", "SimpleZono.py")]

CONFIG = dict(
    box_count=500,
    seed=MASTER_SEED,
    g0=dict(boxes=ct.G0_BOXES, seed=ct.G0_SEED),
    selftest=dict(box_count=50, seed=MASTER_SEED + 200),
    real=dict(networks=["3layer", "4layer", "5layer", "big"], num_images=2,
              eps_indices=[0, 8, 15], n_splits=4, seed=7, per_network=500),
    network=dict(networks=["3layer", "4layer", "5layer", "big"], num_images=3,
                 eps_indices=[0, 4, 8, 12, 15], splits=[1, 4, 16], chunk=4, seed=7,
                 autograd_samples=1001),
    timing=dict(sizes=[1024, 8192, 32768, 65536], warmup=3, repeats=10),
)
SMOKE = dict(
    box_count=4,
    seed=MASTER_SEED + 1000,
    g0=dict(boxes=8, seed=MASTER_SEED + 1100),
    selftest=dict(box_count=6, seed=MASTER_SEED + 1200),
    real=dict(networks=["3layer"], num_images=1, eps_indices=[0], n_splits=2, seed=7,
              per_network=4),
    network=dict(networks=["3layer"], num_images=1, eps_indices=[0], splits=[1, 2],
                 chunk=4, seed=7, autograd_samples=101),
    timing=dict(sizes=[256], warmup=1, repeats=2),
)


# ---------------------------------------------------------------- shared pieces

@contextlib.contextmanager
def swapped_solver(fn):
    """batched_splits looks up pt.lin_reg_tensor_batched at call time."""
    original = pt.lin_reg_tensor_batched
    pt.lin_reg_tensor_batched = fn
    try:
        yield
    finally:
        pt.lin_reg_tensor_batched = original


def set_production_flags():
    pt.USE_VECTORIZED_PRECISE = True
    pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = True
    pt.USE_BATCHED_GRID = True
    pt.USE_REAL_CUBIC = True


def grid_and_targets(lx, ux, ly, uy):
    grid = pt.get_linspace_batched(lx, ux, ly, uy)
    gx, gy = grid[..., 0], grid[..., 1]
    # as in batched_splits.precise_sigmoid
    return grid, (1.0 - torch.sigmoid(gx)) * (torch.sigmoid(gx) * gy)


def max_error(A, B, C, lx, ux, ly, uy):
    """D as compute_max_error computes it on the vectorized path."""
    abc = torch.stack((A, B, C), dim=1)
    corners = pt.check_corners_tensor(abc, lx, ux, ly, uy)
    e_ly, e_uy = pt.check_nonlinear_boundary_tensor(lx, ux, ly, uy, abc)
    return torch.maximum(torch.maximum(corners, e_ly), e_uy)


def bounds_from_grid(grid):
    s = math.isqrt(grid.shape[1])
    return grid[:, 0, 0], grid[:, -1, 0], grid[:, 0, 1], grid[:, s - 1, 1]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def code_hash():
    h = hashlib.sha256()
    for path in SOURCES:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def environment(device):
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=False).stdout.strip()
    except OSError:
        commit = ""
    return dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                platform=platform.platform(), device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "",
                cuda=torch.version.cuda or "", commit=commit, code_sha256=code_hash(),
                time=time.strftime("%Y-%m-%d %H:%M:%S"))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    """A CSV cell or an in-memory value as a float (nan when empty)."""
    return float(x) if x not in ("", None) else float("nan")


# ---------------------------------------------------------------- box families

def synthetic_bounds(family, n, seed, dtype_name):
    """(lx, ux, ly, uy) in float64 on the CPU for one synthetic family."""
    g = torch.Generator().manual_seed(seed)
    f64 = torch.float64

    def u(lo, hi):
        return torch.rand(n, generator=g, dtype=f64) * (hi - lo) + lo

    def log_u(lo, hi):
        return torch.exp(u(math.log(lo), math.log(hi)))

    zero = torch.zeros(n, dtype=f64)
    rcond = lc.default_rcond(getattr(torch, dtype_name), 25)
    lx, wx, ly, wy = u(-4, 4), u(1e-3, 4), u(-4, 4), u(1e-3, 4)
    if family == "degenerate_x":
        wx = zero
    elif family == "degenerate_y":
        wy = zero
    elif family == "point":
        wx, wy = zero, zero
    elif family == "near_degenerate":
        wx, wy = zero + 1e-12, zero + 1e-12
    elif family == "saturated":
        lx = u(25, 35)
    elif family == "mixed_scale":
        lx, wx, ly, wy = u(-1e-8, 1e-8), zero + 1e-8, u(-1e6, 1e6), u(1.0, 1e6)
    elif family in ("cutoff_band_x", "cutoff_band_y"):
        cx, cy = u(-20, 20), u(-4, 4)
        if family == "cutoff_band_y":
            cx, cy = cy, cx
        # gelsd drops an axis at about w = 2 sqrt(2) rcond sqrt(1 + c^2)
        # sqrt(1 + cx^2 + cy^2 + sigma_other^2), sigma = w / (2 sqrt(2)) for 5 points
        r = log_u(0.1, 10.0)
        if family == "cutoff_band_x":
            so = wy / (2 * math.sqrt(2))
            wx = r * 2 * math.sqrt(2) * rcond * torch.sqrt(1 + cx ** 2) * torch.sqrt(
                1 + cx ** 2 + cy ** 2 + so ** 2)
        else:
            so = wx / (2 * math.sqrt(2))
            wy = r * 2 * math.sqrt(2) * rcond * torch.sqrt(1 + cy ** 2) * torch.sqrt(
                1 + cx ** 2 + cy ** 2 + so ** 2)
        lx, ly = cx - wx / 2, cy - wy / 2
    elif family == "large_magnitude":
        mags = torch.tensor(LARGE[dtype_name], dtype=f64)
        mag = mags[torch.randint(len(mags), (n,), generator=g)]
        sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(f64)
        which = torch.randint(3, (n,), generator=g)  # 0: x, 1: y, 2: both
        width = torch.where(torch.rand(n, generator=g) < 0.5, zero, 1e-6 * mag)
        big_x, big_y = which != 1, which != 0
        lx = torch.where(big_x, sign * mag, lx)
        wx = torch.where(big_x, width, wx)
        ly = torch.where(big_y, sign * mag, ly)
        wy = torch.where(big_y, width, wy)
    elif family != "normal":
        raise ValueError(family)
    return lx, lx + wx, ly, ly + wy


def nonfinite_bounds():
    nan, inf = float("nan"), float("inf")
    one_minus = math.nextafter(1.0, 0.0)
    rows = [(nan, 1.0, 0.0, 1.0), (0.0, nan, 0.0, 1.0), (0.0, 1.0, nan, 1.0),
            (inf, inf, 0.0, 1.0), (-inf, 0.0, 0.0, 1.0), (0.0, 1.0, -inf, inf),
            (1.0, one_minus, 0.0, 1.0), (0.0, 1.0, 1.0, one_minus)]
    t = torch.tensor(rows, dtype=torch.float64)
    return t[:, 0], t[:, 1], t[:, 2], t[:, 3]


def capture_real_boxes(cfg, device):
    """Boxes the batched forward pass hands to the solver, plus per-layer rank stats."""
    from measure_split_sweep import load_network_and_images

    calls = []

    def recorder(grid, z):
        calls.append(grid.detach().to("cpu", torch.float64))
        return lc.solve_G(grid, z)

    rng = np.random.default_rng(cfg["seed_family"])
    picked, sources, stats = [], [], []
    with swapped_solver(recorder):
        for network in cfg["networks"]:
            net, images = load_network_and_images(network, device, torch.float64,
                                                  cfg["num_images"])
            layers = bs.extract_affine_layers(net, network)
            pool, pool_src, pool_trunc = [], [], []
            for image_id, img in images:
                for eps_idx in cfg["eps_indices"]:
                    calls.clear()
                    bs.split_bounds(layers, img, EPS_SCHEDULE[eps_idx], cfg["n_splits"],
                                    seed=cfg["seed"])
                    for layer, grid in enumerate(calls):
                        z = grid_and_targets(*bounds_from_grid(grid))[1]
                        rg, re = lc.gelsd_rank(grid, z), lc.exact_rank(grid)
                        r2, r3 = lc.singular_value_ratios(grid)
                        rcond = lc.default_rcond(torch.float64, grid.shape[1])
                        band = ((r3 > rcond / 2) & (r3 < 2 * rcond)) | \
                               ((r2 > rcond / 2) & (r2 < 2 * rcond))
                        trunc = rg < re
                        st = lc._stats(grid)
                        stats.append(dict(network=network, image_id=image_id, eps_idx=eps_idx,
                                          layer=layer, n_boxes=grid.shape[0],
                                          n_xdeg=int(st.xdeg.sum()), n_ydeg=int(st.ydeg.sum()),
                                          n_gelsd_truncated=int(trunc.sum()),
                                          n_cutoff_band=int(band.sum()),
                                          min_s3_over_s1=float(r3.min())))
                        pool.append(torch.stack(bounds_from_grid(grid), dim=1))
                        pool_src += [f"{network}/img{image_id}/eps{eps_idx}/layer{layer}"] * grid.shape[0]
                        pool_trunc.append(trunc)
            pool = torch.cat(pool)
            pool_trunc = torch.cat(pool_trunc).numpy()
            k = min(cfg["per_network"], pool.shape[0])
            idx = rng.choice(pool.shape[0], size=k, replace=False)
            rest = np.setdiff1d(np.flatnonzero(pool_trunc), idx)
            extra = rng.choice(rest, size=min(cfg["per_network"], rest.size), replace=False) \
                if rest.size else np.array([], dtype=int)
            for i in np.concatenate((idx, extra)).astype(int):
                picked.append(pool[i])
                sources.append(pool_src[i])
    b = torch.stack(picked)
    return (b[:, 0], b[:, 1], b[:, 2], b[:, 3]), sources, stats


# ---------------------------------------------------------------- box level

def evaluate_family(family, dtype_name, bounds, sources, seed, solvers, device):
    """Summary rows and per-box rows of one family for every solver in `solvers`."""
    dtype = getattr(torch, dtype_name)
    lx, ux, ly, uy = (t.to(device=device, dtype=dtype) for t in bounds)
    n = lx.shape[0]
    grid, z = grid_and_targets(lx, ux, ly, uy)
    noise = torch.as_tensor(np.random.default_rng(seed).normal(0, 1e-4, n), dtype=dtype,
                            device=device)
    info = family in INFORMATIONAL
    rank_g = lc.gelsd_rank(grid, z).tolist() if not info else [None] * n
    rank_e = lc.exact_rank(grid).cpu().tolist()
    s = math.isqrt(grid.shape[1])
    xs_l = grid[:, ::s, 0].cpu().tolist()
    ys_l = grid[:, :s, 1].cpu().tolist()
    z_l = z.cpu().tolist()
    b_l = [t.cpu().tolist() for t in (lx, ux, ly, uy)]
    exact = [orc.exact_min_norm(xs_l[i], ys_l[i], z_l[i]) for i in range(n)] if not info else None
    rx, ry = (ux - lx) / 2, (uy - ly) / 2

    per_cand, d_true_cache = {}, {}
    for name, solve in solvers.items():
        res = dict(status="ok")
        try:
            sync(device)
            t0 = time.perf_counter()
            coef = solve(grid, z)
            sync(device)
            res["time_ms"] = (time.perf_counter() - t0) * 1000.0
        except Exception as e:  # recorded, then judged by the gates
            res["status"] = f"error: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
            per_cand[name] = res
            continue
        C, A, B = coef[:, 0], coef[:, 1] + noise, coef[:, 2]
        D = max_error(A, B, C, lx, ux, ly, uy)
        W = A.abs() * rx + B.abs() * ry + D
        finite = torch.isfinite(coef).all(dim=1) & torch.isfinite(D)
        res.update(coef=coef.cpu().tolist(), A=A.cpu().tolist(), D=D.cpu().tolist(),
                   W=W.cpu().double().numpy(), finite=finite.cpu().numpy(),
                   n_nonfinite=int((~finite).sum()), n_A_zero=int((A == 0).sum()))
        d_true, gaps, sf_w, sf_d, err = [], [], [], [], []
        for i in range(n):
            if info or not res["finite"][i]:
                for lst in (d_true, gaps, sf_w, sf_d):
                    lst.append(float("nan"))
                err.append(float("nan") if info else float("inf"))
                continue
            Ci, _, Bi = res["coef"][i]
            Ai = res["A"][i]
            key = (Ai, Bi, Ci, b_l[0][i], b_l[1][i], b_l[2][i], b_l[3][i])
            if key not in d_true_cache:
                d_true_cache[key] = orc.true_max_error(Ai, Bi, Ci, *key[3:])
            dt = d_true_cache[key]
            dc, wi = res["D"][i], float(res["W"][i])
            d_true.append(dt)
            if math.isnan(dt):
                gaps.append(float("nan"))
                sf_w.append(float("nan"))
                sf_d.append(float("nan"))
            else:
                gap = max(0.0, dt - dc)
                gaps.append(gap)
                # the missing part of the neuron's radius W (G1)
                sf_w.append(gap / wi if wi > 0 else gap)
                sf_d.append(gap / dt if dt > 0 else 0.0)
            err.append(orc.relative_error(res["coef"][i], exact[i]))
        res.update(d_true=d_true, gap=np.array(gaps), shortfall=np.array(sf_w),
                   shortfall_rel_D=np.array(sf_d), err=np.array(err))
        per_cand[name] = res

    g = per_cand.get("G")
    summaries, box_rows = [], []
    agree = np.array([rg == re for rg, re in zip(rank_g, rank_e)])
    n_trunc = sum(1 for rg, re in zip(rank_g, rank_e) if rg is not None and rg < re)

    def nanmax(a):
        return float(np.nanmax(a)) if np.isfinite(a).any() or np.isinf(a).any() else float("nan")

    for name in solvers:
        res = per_cand[name]
        row = dict(dtype=dtype_name, family=family, candidate=name, status=res["status"],
                   n_boxes=n, n_gelsd_truncated=n_trunc, device=str(device))
        if res["status"] != "ok":
            summaries.append(row)
            continue
        ratio = np.full(n, np.nan)
        diff = [float("nan")] * n
        if g is not None and g["status"] == "ok":
            wg, w = g["W"], res["W"]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(wg > 0, w / wg, np.where(w == wg, 1.0, np.inf))
            diff = [max(abs(a - b) for a, b in zip(ca, cg)) / max(1e-300, max(abs(b) for b in cg))
                    if all(map(math.isfinite, ca + cg)) else float("inf")
                    for ca, cg in zip(res["coef"], g["coef"])]
        sf = res["shortfall"]
        er = res["err"][agree] if not info else np.array([])
        fin_coef = [abs(c) for cs in res["coef"] for c in cs if math.isfinite(c)]
        has_ratio = not np.isnan(ratio).all()
        row.update(
            time_ms=res["time_ms"],
            n_nonfinite=res["n_nonfinite"], n_A_zero=res["n_A_zero"],
            n_unverified=int(np.isnan(sf).sum()) - (n if info else res["n_nonfinite"]),
            worst_shortfall=nanmax(sf),
            n_any_shortfall=int((res["gap"] > 0).sum()),
            worst_abs_shortfall=nanmax(res["gap"]),
            worst_shortfall_rel_D=nanmax(res["shortfall_rel_D"]),
            w_ratio_vs_G_p50=float(np.nanpercentile(ratio, 50, method="higher")) if has_ratio else float("nan"),
            w_ratio_vs_G_p99=float(np.nanpercentile(ratio, 99, method="higher")) if has_ratio else float("nan"),
            w_ratio_vs_G_max=float(np.nanmax(ratio)) if has_ratio else float("nan"),
            frac_w_better_than_G=float((ratio < 1 - 1e-12).mean()),
            frac_w_worse_than_G=float((ratio > 1 + 1e-12).mean()),
            max_abs_coef=max(fin_coef) if fin_coef else float("nan"),
            n_rank_agree=int(agree.sum()) if not info else 0,
            oracle_err_median=float(np.median(er)) if er.size else float("nan"),
            oracle_err_max=float(np.max(er)) if er.size else float("nan"),
            max_rel_diff_vs_G=float(np.nanmax(diff)) if not all(map(math.isnan, diff)) else float("nan"),
        )
        summaries.append(row)
        for i in range(n):
            Ci, _, Bi = res["coef"][i]
            box_rows.append(dict(dtype=dtype_name, family=family, box=i,
                                 source=sources[i] if sources else "",
                                 candidate=name, lx=b_l[0][i], ux=b_l[1][i], ly=b_l[2][i],
                                 uy=b_l[3][i], C=Ci, A=res["A"][i], B=Bi,
                                 noise=float(noise[i]), D=res["D"][i], D_true=res["d_true"][i],
                                 abs_shortfall=res["gap"][i], shortfall=sf[i],
                                 shortfall_rel_D=res["shortfall_rel_D"][i], W=res["W"][i],
                                 w_ratio_vs_G=ratio[i], oracle_err=res["err"][i],
                                 rank_gelsd=rank_g[i], rank_exact=rank_e[i]))
    return summaries, box_rows


def run_g0(cfg, dtypes, device, out, smoke):
    """Contract checks for the candidates, then the self-test with planted bugs.
    Returns True when every planted bug is caught."""
    print("G0 contract checks ...", flush=True)
    contract = []
    for dtype_name in dtypes:
        for name, solve in lc.CANDIDATES.items():
            contract += ct.contract_checks(name, solve, ct.contract_rank_rule(name),
                                           lc.READS_Z[name], dtype_name, cfg["g0"]["boxes"],
                                           cfg["g0"]["seed"], device)
    write_csv(os.path.join(out, "g0_contract.csv"), contract)
    for name in lc.CANDIDATES:
        bad = [r for r in contract if r["candidate"] == name and r["n_failed"]]
        print(f"  {name:3s} {'PASS' if not bad else 'FAIL'} "
              + "; ".join(f"{r['check']} {r['family']}/{r['dtype']} {r['n_failed']}/{r['n_checked']}"
                          for r in bad[:4]), flush=True)

    print("G0 self-test with planted bugs ...", flush=True)
    st_cfg = cfg["selftest"]
    m_contract = []
    for mid, parent, _, solve in ct.MUTANTS:
        for dtype_name in dtypes:
            m_contract += ct.contract_checks(mid, solve, ct.contract_rank_rule(parent),
                                             lc.READS_Z[parent], dtype_name, cfg["g0"]["boxes"],
                                             cfg["g0"]["seed"], device)
    solvers = dict(lc.CANDIDATES)
    solvers.update({mid: solve for mid, _, _, solve in ct.MUTANTS})
    summaries, per_box = [], []
    for dtype_name in dtypes:
        for k, family in enumerate(FAMILIES[:10]):
            seed = st_cfg["seed"] + k
            bounds = synthetic_bounds(family, st_cfg["box_count"], seed, dtype_name)
            s, b = evaluate_family(family, dtype_name, bounds, None, seed, solvers, device)
            summaries += s
            per_box += b

    rows, all_caught = [], True
    for mid, parent, desc, _ in ct.MUTANTS:
        how = []
        failed = sorted({f"{r['check']} {r['family']}/{r['dtype']}" for r in m_contract
                         if r["candidate"] == mid and r["n_failed"]})
        if failed:
            how.append("contract: " + ", ".join(failed[:3]))
        group = [r for r in summaries if r["candidate"] in ("G", parent, mid)]
        reasons = gate_failures(group, [], None, [mid])[mid]
        if reasons:
            how.append("gate: " + reasons[0])
        keys = ranking_keys(group, [r for r in per_box if r["candidate"] in ("G", parent, mid)],
                            [], [], ["G", parent, mid])
        for label, i in (("R2a", 1), ("R3", 3)):
            km, kp = keys[mid][i], keys[parent][i]
            if km is not None and kp is not None and km > kp:
                how.append(f"{label}: {km} > {kp} of {parent}")
        caught = bool(how)
        all_caught &= caught
        rows.append(dict(mutant=mid, parent=parent, description=desc, caught=caught,
                         caught_by=" | ".join(how)))
        print(f"  {mid:4s} in {parent:3s} {'caught' if caught else 'SURVIVED'}: "
              f"{desc}{'  <- ' + how[0] if how else ''}", flush=True)
    write_csv(os.path.join(out, "g0_selftest.csv"), rows)
    write_csv(os.path.join(out, "g0_selftest_contract.csv"), m_contract)
    write_csv(os.path.join(out, "g0_selftest_box_summary.csv"), summaries)
    if not all_caught:
        print("SELF-TEST FAILED: the harness misses a planted bug"
              + (" (smoke run: continuing)" if smoke else "; the measurement does not start"))
    return all_caught


def run_box_level(args, cfg, device, out):
    dtypes = ["float64", "float32"] if args.dtype == "both" else [args.dtype]
    meta = dict(environment(device), config=cfg, candidates=list(lc.CANDIDATES), dtypes=dtypes,
                smoke=args.smoke)
    if not run_g0(cfg, dtypes, device, out, args.smoke):
        meta["selftest_passed"] = False
        with open(os.path.join(out, "box_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if not args.smoke:
            raise SystemExit(2)
    else:
        meta["selftest_passed"] = True

    real_cfg = dict(cfg["real"], seed_family=cfg["seed"] + FAMILIES.index("real"))
    print("recording real boxes ...", flush=True)
    real_bounds, real_sources, rank_stats = capture_real_boxes(real_cfg, device)
    write_csv(os.path.join(out, "real_rank_stats.csv"), rank_stats)
    print(f"  {sum(r['n_boxes'] for r in rank_stats)} boxes recorded, gelsd truncates "
          f"{sum(r['n_gelsd_truncated'] for r in rank_stats)} beyond the exact rank, "
          f"{sum(r['n_cutoff_band'] for r in rank_stats)} lie within 2x of the cutoff; "
          f"{len(real_sources)} kept", flush=True)

    summaries, box_rows = [], []
    for dtype_name in dtypes:
        for k, family in enumerate(FAMILIES):
            seed = cfg["seed"] + k
            if family == "real":
                bounds, sources = real_bounds, real_sources
            elif family == "nonfinite":
                bounds, sources = nonfinite_bounds(), None
            else:
                bounds, sources = synthetic_bounds(family, cfg["box_count"], seed, dtype_name), None
            t0 = time.perf_counter()
            s, b = evaluate_family(family, dtype_name, bounds, sources, seed, lc.CANDIDATES,
                                   device)
            summaries += s
            box_rows += b
            print(f"  {dtype_name:8s} {family:16s} {bounds[0].shape[0]:5d} boxes "
                  f"({time.perf_counter() - t0:.1f} s)", flush=True)
    write_csv(os.path.join(out, "box_summary.csv"), summaries)
    write_csv(os.path.join(out, "box_per_box.csv"), box_rows)
    with open(os.path.join(out, "box_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print_box_summary(summaries)


def print_box_summary(rows):
    print()
    print(f"{'dtype':8s} {'family':16s} {'cand':4s} {'status':6s} {'nonfin':>6s} "
          f"{'A==0':>4s} {'worst_sf':>9s} {'W/G p99':>8s} {'W/G max':>9s} {'err med':>9s} "
          f"{'|coef|max':>9s}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['dtype']:8s} {r['family']:16s} {r['candidate']:4s} {r['status']}")
            continue
        print(f"{r['dtype']:8s} {r['family']:16s} {r['candidate']:4s} {'ok':6s} "
              f"{r['n_nonfinite']:6d} {r['n_A_zero']:4d} {r['worst_shortfall']:9.2e} "
              f"{r['w_ratio_vs_G_p99']:8.4f} {r['w_ratio_vs_G_max']:9.3e} "
              f"{r['oracle_err_median']:9.2e} {r['max_abs_coef']:9.2e}")


# ---------------------------------------------------------------- the rule

def gate_failures(box_rows, g0_rows, net_rows, candidates):
    """Reasons each candidate fails G0, G1, G2 and (when net_rows is given) G3."""
    by = {(r["dtype"], r["family"], r["candidate"]): r for r in box_rows}
    keys = sorted({(r["dtype"], r["family"]) for r in box_rows})
    failed = {c: [] for c in candidates}
    for c in candidates:
        for r in g0_rows:
            if r["candidate"] == c and int(num(r["n_failed"])):
                failed[c].append(f"G0 {r['check']} {r['family']}/{r['dtype']}: "
                                 f"{r['n_failed']} of {r['n_checked']}")
        for dt, fam in keys:
            r, g = by.get((dt, fam, c)), by.get((dt, fam, "G"))
            if fam in INFORMATIONAL:
                continue
            if r is None:
                failed[c].append(f"{fam}/{dt}: no result")
                continue
            if r["status"] != "ok":
                failed[c].append(f"{fam}/{dt}: {r['status']}")
                continue
            g_ok = g is not None and g["status"] == "ok"
            g_sf = num(g["worst_shortfall"]) if g_ok else float("nan")
            limit = 10 * (max(TAU[dt], g_sf) if not math.isnan(g_sf) else TAU[dt])
            if num(r["worst_shortfall"]) > limit:
                failed[c].append(f"G1 {fam}/{dt}: shortfall {num(r['worst_shortfall']):.2e} "
                                 f"> {limit:.2e}")
            if fam in MUST_BE_FINITE and (int(num(r["n_nonfinite"])) or int(num(r["n_A_zero"]))):
                failed[c].append(f"G2 {fam}/{dt}: {r['n_nonfinite']} non-finite, "
                                 f"{r['n_A_zero']} A == 0")
            if fam == "large_magnitude" and g_ok and \
                    int(num(r["n_nonfinite"])) > int(num(g["n_nonfinite"])):
                failed[c].append(f"G2 {fam}/{dt}: {r['n_nonfinite']} non-finite "
                                 f"> G's {g['n_nonfinite']}")
        if net_rows is not None:
            mine = [r for r in net_rows if r["candidate"] == c]
            if any(r["status"] != "ok" or str(r["sound"]) != "True" for r in mine):
                failed[c].append("G3: a network-level case failed or fell below the "
                                 "autograd maximum")
    return failed


def ranking_keys(box_rows, per_box_rows, net_rows, timing_rows, survivors):
    """[R1, R2a, R2b, R3, R4, R5] per survivor; None where not measured."""
    by = {(r["dtype"], r["family"], r["candidate"]): r for r in box_rows}
    keys = sorted({(r["dtype"], r["family"]) for r in box_rows})

    def k1(c):
        mine = [num(r["max_excess"]) for r in net_rows if r["candidate"] == c and r["status"] == "ok"]
        if not mine:
            return None
        e = max(mine)
        return 0.0 if abs(e) <= CONTRACT else e

    boxes = {}
    for r in per_box_rows:
        if r["family"] in HARD and r["candidate"] in survivors:
            boxes.setdefault((r["dtype"], r["family"], str(r["box"])), {})[r["candidate"]] = r
    ratios = {}
    for (dt, fam, _), rs in boxes.items():
        finite = {c: r for c, r in rs.items() if math.isfinite(num(r["W"]))}
        if not finite:
            continue
        best = min(finite.values(), key=lambda r: num(r["W"]))
        mx = max(abs(num(best["lx"])), abs(num(best["ux"])))
        my = max(abs(num(best["ly"])), abs(num(best["uy"])))
        size = 0.25 * my + abs(num(best["A"])) * mx + abs(num(best["B"])) * my + abs(num(best["C"]))
        band = 16 * torch.finfo(getattr(torch, dt)).eps * size
        w_best = num(best["W"])
        for c in survivors:
            w = num(rs[c]["W"]) if c in rs else float("nan")
            if not math.isfinite(w):
                ratio = float("inf")
            elif w - w_best <= band:
                ratio = 1.0
            else:
                ratio = w / w_best if w_best > 0 else float("inf")
            ratios.setdefault((dt, fam, c), []).append(ratio)

    def k2a(c):
        p99 = [float(np.percentile(v, 99, method="higher")) for (_, _, cc), v in ratios.items() if cc == c]
        return round(max(p99), 2) if p99 else None

    def hard_rows(c):
        return [by[(dt, fam, c)] for dt, fam in keys if fam in HARD and (dt, fam, c) in by
                and by[(dt, fam, c)]["status"] == "ok"]

    def k2b(c):
        vals = [num(r["worst_shortfall"]) for r in hard_rows(c)]
        vals = [v for v in vals if not math.isnan(v)]
        return math.floor(math.log10(max(max(vals), 1e-300))) if vals else None

    def k3(c):
        meds = [num(by[(dt, fam, c)]["oracle_err_median"]) for dt, fam in keys
                if fam not in INFORMATIONAL and (dt, fam, c) in by
                and by[(dt, fam, c)]["status"] == "ok"]
        meds = [m for m in meds if not math.isnan(m)]
        return math.floor(math.log10(max(max(meds), 1e-300))) if meds else None

    t = {r["candidate"]: num(r["median_ms"]) for r in timing_rows
         if r.get("status") == "ok" and r["dtype"] == "float64" and int(num(r["n"])) == 65536}
    t_min = min((t[c] for c in survivors if c in t), default=None)

    def k4(c):
        if c not in t or not t_min:
            return None
        return round(math.log(t[c] / t_min) / math.log(1.2))

    return {c: [k1(c), k2a(c), k2b(c), k3(c), k4(c), lc.SIMPLICITY.get(c)] for c in survivors}


def box_survivors(out, smoke):
    """Candidates that pass G0-G2 on the box-level results in out, plus G; refuses
    results it cannot trust."""
    meta_path = os.path.join(out, "box_meta.json")
    box = read_csv(os.path.join(out, "box_summary.csv"))
    g0 = read_csv(os.path.join(out, "g0_contract.csv"))
    if not (os.path.exists(meta_path) and box and g0):
        raise SystemExit(f"no box-level results in {out}: run --level box first")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("smoke") != smoke:
        raise SystemExit("box-level results come from the other smoke mode")
    if meta.get("code_sha256") != code_hash():
        raise SystemExit("box-level results were produced by different code (source hash differs)")
    if not meta.get("selftest_passed") and not smoke:
        raise SystemExit("the box-level self-test failed; the measurement is not valid")
    have = {r["candidate"] for r in box}
    missing = [c for c in lc.CANDIDATES if c not in have]
    if missing:
        raise SystemExit(f"no box-level result for {missing}")
    failed = gate_failures(box, g0, None, list(lc.CANDIDATES))
    return ["G"] + [c for c in lc.CANDIDATES if c != "G" and not failed[c]], failed


# ---------------------------------------------------------------- network level

def run_network_level(args, cfg, device, out):
    from measure_split_sweep import load_network_and_images
    from check_sweep_soundness import sampled_max_derivative

    forced = args.force_candidates is not None
    if forced:
        candidates = ["G"] + [c for c in args.force_candidates if c != "G"]
        print(f"  --force-candidates: running {candidates} OUTSIDE the pre-registered rule")
    else:
        candidates, failed = box_survivors(out, args.smoke)
        print(f"  survivors of G0-G2: {candidates}; out: "
              f"{[c for c in lc.CANDIDATES if failed[c]]}")
    c = cfg["network"]
    rows = []
    for network in c["networks"]:
        net, images = load_network_and_images(network, device, torch.float64, c["num_images"])
        layers = bs.extract_affine_layers(net, network)
        net64 = copy.deepcopy(net).to("cpu", torch.float64)
        for image_id, img in images:
            for eps_idx in c["eps_indices"]:
                eps = EPS_SCHEDULE[eps_idx]
                sampled = sampled_max_derivative(net64, img.cpu().double(), eps,
                                                 c["autograd_samples"])
                for n_splits in c["splits"]:
                    ref = bs.split_bounds(layers, img, eps, n_splits, seed=c["seed"],
                                          chunk=c["chunk"]).cpu().double()
                    for name in candidates:
                        row = dict(network=network, image_id=image_id, eps_idx=eps_idx,
                                   n_splits=n_splits, candidate=name, sampled_max=sampled,
                                   device=str(device), forced=forced)
                        try:
                            if name == "G":
                                got = ref
                            else:
                                with swapped_solver(lc.CANDIDATES[name]):
                                    got = bs.split_bounds(layers, img, eps, n_splits,
                                                          seed=c["seed"],
                                                          chunk=c["chunk"]).cpu().double()
                        except Exception as e:
                            row.update(status=f"error: {type(e).__name__}: "
                                              f"{str(e).splitlines()[0][:120]}", sound=False)
                            rows.append(row)
                            continue
                        rel = got / ref - 1
                        bound = got.max().item()
                        row.update(status="ok", bound=bound, max_excess=rel.max().item(),
                                   max_abs_rel=rel.abs().max().item(),
                                   sound=bool(math.isfinite(bound) and bound >= sampled))
                        rows.append(row)
                print(f"  {network} image {image_id} eps_idx {eps_idx} done", flush=True)
    write_csv(os.path.join(out, "network_summary.csv"), rows)
    with open(os.path.join(out, "network_meta.json"), "w") as f:
        json.dump(dict(environment(device), config=c, candidates=candidates, forced=forced,
                       smoke=args.smoke), f, indent=2)
    for name in candidates:
        mine = [r for r in rows if r["candidate"] == name]
        ok = [r for r in mine if r["status"] == "ok"]
        worst = max((r["max_abs_rel"] for r in ok), default=float("nan"))
        print(f"  {name:3s}: {len(ok)}/{len(mine)} ok, all sound: "
              f"{all(r['sound'] for r in mine)}, max |rel diff| vs G {worst:.2e}")


# ---------------------------------------------------------------- timing

def run_timing(args, cfg, device, out):
    c = cfg["timing"]
    dtypes = ["float64", "float32"] if args.dtype == "both" else [args.dtype]
    rows = []
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        for n in c["sizes"]:
            bounds = synthetic_bounds("normal", n, cfg["seed"], dtype_name)
            grid, z = grid_and_targets(*(t.to(device=device, dtype=dtype) for t in bounds))
            for name, fn in lc.CANDIDATES.items():
                try:
                    for _ in range(c["warmup"]):
                        fn(grid, z)
                    times = []
                    for _ in range(c["repeats"]):
                        sync(device)
                        t0 = time.perf_counter()
                        fn(grid, z)
                        sync(device)
                        times.append((time.perf_counter() - t0) * 1000.0)
                    rows.append(dict(device=str(device), dtype=dtype_name, candidate=name, n=n,
                                     median_ms=float(np.median(times)), min_ms=min(times),
                                     status="ok"))
                except Exception as e:
                    rows.append(dict(device=str(device), dtype=dtype_name, candidate=name, n=n,
                                     status=f"error: {type(e).__name__}"))
                r = rows[-1]
                print(f"  {dtype_name:8s} n={n:6d} {name:3s} "
                      f"{r.get('median_ms', float('nan')):9.3f} ms  {r['status']}", flush=True)
    write_csv(os.path.join(out, f"timing_{device.type}.csv"), rows)
    with open(os.path.join(out, f"timing_{device.type}_meta.json"), "w") as f:
        json.dump(dict(environment(device), config=c, smoke=args.smoke), f, indent=2)


# ---------------------------------------------------------------- decision

def decide(out):
    """Applies the gates and the ranking of the module docstring to the files in out."""
    box = read_csv(os.path.join(out, "box_summary.csv"))
    g0 = read_csv(os.path.join(out, "g0_contract.csv"))
    selftest = read_csv(os.path.join(out, "g0_selftest.csv"))
    net = read_csv(os.path.join(out, "network_summary.csv"))
    timing = read_csv(os.path.join(out, "timing_cuda.csv"))
    if not box:
        print("no box_summary.csv: run --level box first")
        return 1
    survived = [r["mutant"] for r in selftest if r["caught"] != "True"]
    if survived or not selftest:
        print(f"Self-test: {'missing' if not selftest else 'planted bugs not caught: ' + str(survived)}"
              " -> the comparison is not valid")
    if any(r.get("forced") == "True" for r in net):
        print("The network level was run with --force-candidates, outside the rule.")

    cands = list(lc.CANDIDATES)
    failed = gate_failures(box, g0, net if net else None, cands)
    print("Gates")
    for c in cands:
        print(f"  {c:3s} {'PASS' if not failed[c] else 'FAIL'}")
        for reason in failed[c][:8]:
            print(f"        {reason}")
    if not net:
        print("  (G3 pending: no network-level results)")
    survivors = [c for c in cands if not failed[c]]
    if not survivors:
        print("no candidate passes the gates")
        return 0

    table = ranking_keys(box, read_csv(os.path.join(out, "box_per_box.csv")), net, timing,
                         survivors)
    names = ["R1 tightness", "R2a W/W_best p99", "R2b shortfall", "R3 accuracy", "R4 speed",
             "R5 simplicity"]
    print("\nRanking keys (lower is better; None = not measured yet)")
    print("      " + "  ".join(f"{n:>16s}" for n in names))
    for c in survivors:
        print(f"  {c:3s} " + "  ".join(f"{str(v):>16s}" for v in table[c]))

    order = list(survivors)
    for i, name in enumerate(names):
        vals = [table[c][i] for c in order]
        if any(v is None for v in vals):
            print(f"\nUndecided at {name}: not measured for "
                  f"{[c for c in order if table[c][i] is None]}. Tied so far: {order}")
            break
        best = min(vals)
        order = [c for c in order if table[c][i] == best]
        if len(order) == 1:
            print(f"\nWinner: {order[0]} (decided by {name})")
            break
    else:
        print(f"\nTied after every criterion: {order}")
    if any(r.get("device") == "cpu" for r in box if r["candidate"] == "P"):
        print("P was evaluated on the CPU (gesdd); its place is provisional until the box "
              "level runs on CUDA.")
    return 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", choices=["box", "network", "timing", "decide", "all"], default="box")
    ap.add_argument("--force-candidates", nargs="+", choices=list(lc.CANDIDATES),
                    help="network level: run these instead of the box-level survivors "
                         "(outside the pre-registered rule; recorded as such)")
    ap.add_argument("--dtype", choices=["float64", "float32", "both"], default="both",
                    help="box and timing levels; the network level is float64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", help="output directory (required unless --smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny cases with other seeds through every level; not a measurement")
    args = ap.parse_args()

    if args.out is None:
        if not args.smoke:
            ap.error("--out is required unless --smoke")
        args.out = tempfile.mkdtemp(prefix="lstsq_smoke_")
    os.makedirs(args.out, exist_ok=True)
    cfg = SMOKE if args.smoke else CONFIG
    device = torch.device(args.device)
    set_production_flags()
    print(f"out: {args.out}  device: {device}  smoke: {args.smoke}", flush=True)

    levels = ["box", "network", "timing", "decide"] if (args.level == "all" or args.smoke) \
        else [args.level]
    for level in levels:
        print(f"\n=== {level} ===", flush=True)
        if level == "box":
            run_box_level(args, cfg, device, args.out)
        elif level == "network":
            run_network_level(args, cfg, device, args.out)
        elif level == "timing":
            run_timing(args, cfg, device, args.out)
        else:
            decide(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
