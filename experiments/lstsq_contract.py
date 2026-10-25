"""Gate G0 of experiments/compare_lstsq_solvers.py: implementation checks for the
candidate plane solvers, and planted bugs the harness must catch.

Contract checks C1-C7, per candidate, float64 and float32, on four families on
which no rank decision is ambiguous (200 boxes each, seed 20261025 + 100 + k):
well_conditioned (centres U(-4, 4), widths U(0.5, 4)), degenerate_x,
degenerate_y and point. Thresholds are in ulps of the dtype.

  C1 output: shape [n, 3], the dtype and device of the input, all finite
  C2 batch independence: a shuffled batch gives the shuffled result, and the two
     halves give the whole result, bit for bit
  C3 z scaling (candidates that read z): b(2^k z) == 2^k b(z) bit for bit, k = 5, -3
  C4 axis swap (candidates that read z): swapping the grid axes, with z
     transposed so that x still runs slowest, gives (C, B, A) within 1e4 ulps of
     max |b|. Not a symmetry of A3: there the sigmoid belongs to x.
  C5 normal equations, on the boxes where the candidate's own rank rule keeps the
     exact rank (a deliberately truncated solution does not solve them):
     ||X^T (X b - z)||_inf / (||X||_F^2 ||b||_inf + ||X||_F ||z||_inf) <= 1e3 ulps,
     the residual computed exactly (Fractions) for the floating-point b
  C6 minimum norm: on an exactly degenerate axis, b is orthogonal to the exact
     null vector of X, (x0, -1, 0) or (y0, 0, -1):
     |b . v| / (||b|| ||v||) <= 1e3 ulps. Holds for any rank the candidate
     chooses: dropping more singular values keeps b in the span of the retained
     right singular vectors, all orthogonal to the null space.
  C7 exact oracle: relative distance to the exact minimum-norm solution
     (lstsq_oracle.exact_min_norm) <= 1e4 ulps

A3 never reads z (it rebuilds the separable targets from the grid), so C3 and C4
do not apply to it and its z is always the production target.

Self-test: the planted bugs S1-S10 below run through the same pipeline. Each must
be caught by C1-C7, by gate G1 or G2, or rank strictly below the candidate it was
planted in on R2a, R2b or R3. One that survives means the harness is blind to
that bug, and the measurement does not start.
"""
import contextlib
import math
from fractions import Fraction

import numpy as np
import torch

import lstsq_candidates as lc
import lstsq_oracle as orc
import precise_transformer as pt

G0_FAMILIES = ["well_conditioned", "degenerate_x", "degenerate_y", "point"]
G0_BOXES = 200
G0_SEED = 20261025 + 100
ULPS = dict(C4=1e4, C5=1e3, C6=1e3, C7=1e4)


def g0_bounds(family, n, seed):
    g = torch.Generator().manual_seed(seed)

    def u(lo, hi):
        return torch.rand(n, generator=g, dtype=torch.float64) * (hi - lo) + lo

    lx, wx, ly, wy = u(-4, 4), u(0.5, 4), u(-4, 4), u(0.5, 4)
    zero = torch.zeros(n, dtype=torch.float64)
    if family in ("degenerate_x", "point"):
        wx = zero
    if family in ("degenerate_y", "point"):
        wy = zero
    return lx, lx + wx, ly, ly + wy


def _targets(grid):
    gx, gy = grid[..., 0], grid[..., 1]
    return (1.0 - torch.sigmoid(gx)) * (torch.sigmoid(gx) * gy)


def _exact_normal_ratio(xs, ys, z, b):
    X = [(Fraction(1), Fraction(x), Fraction(y)) for x in xs for y in ys]
    B = [Fraction(v) for v in b]
    r = [sum(a * c for a, c in zip(row, B)) - Fraction(zv) for row, zv in zip(X, z)]
    xtr = max(abs(sum(row[i] * ri for row, ri in zip(X, r))) for i in range(3))
    fro2 = float(sum(a * a for row in X for a in row))
    den = fro2 * max(abs(v) for v in b) + math.sqrt(fro2) * max(abs(v) for v in z)
    return float(xtr) / den if den > 0 else float(xtr)


def contract_checks(name, solve, rank_rule, reads_z, dtype_name, n=G0_BOXES, seed=G0_SEED,
                    device="cpu"):
    """Rows (check, family, n_checked, n_failed, worst, threshold) for one solver."""
    dtype = getattr(torch, dtype_name)
    eps = torch.finfo(dtype).eps
    rows = []

    def add(check, family, values, threshold):
        values = [v for v in values if v is not None]
        bad = [v for v in values if not (v <= threshold)]
        rows.append(dict(candidate=name, dtype=dtype_name, family=family, check=check,
                         n_checked=len(values), n_failed=len(bad),
                         worst=max(values) if values else float("nan"), threshold=threshold))

    for k, family in enumerate(G0_FAMILIES):
        bounds = [t.to(device=device, dtype=dtype) for t in g0_bounds(family, n, seed + k)]
        grid = pt.get_linspace_batched(*bounds)
        z = _targets(grid)
        try:
            b = solve(grid, z)
        except Exception as e:
            rows.append(dict(candidate=name, dtype=dtype_name, family=family, check="C1",
                             n_checked=n, n_failed=n, worst=float("inf"), threshold=0,
                             note=f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"))
            continue
        ok = b.shape == (n, 3) and b.dtype == dtype and b.device == grid.device
        fin = torch.isfinite(b).all(dim=1).cpu().tolist()
        add("C1", family, [0.0 if (ok and f) else 1.0 for f in fin], 0.0)

        gen = torch.Generator().manual_seed(seed + 50 + k)
        perm = torch.randperm(n, generator=gen).to(grid.device)
        shuffled = torch.equal(solve(grid[perm], z[perm]), b[perm])
        h = n // 2
        halves = torch.equal(torch.cat((solve(grid[:h], z[:h]), solve(grid[h:], z[h:]))), b)
        add("C2", family, [0.0 if (shuffled and halves) else 1.0], 0.0)

        if reads_z:
            same = all(torch.equal(solve(grid, z * 2.0 ** e), b * 2.0 ** e) for e in (5, -3))
            add("C3", family, [0.0 if same else 1.0], 0.0)
            s = math.isqrt(grid.shape[1])
            lx, ux, ly, uy = bounds
            grid_sw = pt.get_linspace_batched(ly, uy, lx, ux)
            z_sw = z.reshape(n, s, s).transpose(1, 2).reshape(n, s * s)
            got = solve(grid_sw, z_sw)
            want = b[:, [0, 2, 1]]
            scale = b.abs().max(dim=1).values.clamp_min(torch.finfo(dtype).tiny)
            add("C4", family, ((got - want).abs().max(dim=1).values / scale / eps).cpu().tolist(),
                ULPS["C4"])

        s = math.isqrt(grid.shape[1])
        xs_l = grid[:, ::s, 0].cpu().tolist()
        ys_l = grid[:, :s, 1].cpu().tolist()
        z_l = z.cpu().tolist()
        b_l = b.cpu().tolist()
        keep = (rank_rule(grid, z) == lc.exact_rank(grid).cpu()).tolist()
        add("C5", family, [_exact_normal_ratio(xs_l[i], ys_l[i], z_l[i], b_l[i]) / eps
                           if keep[i] and fin[i] else None for i in range(n)], ULPS["C5"])

        st = lc._stats(grid)
        xdeg, ydeg = st.xdeg.cpu().tolist(), st.ydeg.cpu().tolist()
        c6 = []
        for i in range(n):
            C, A, B = (Fraction(v) for v in b_l[i]) if fin[i] else (None,) * 3
            nb = math.sqrt(sum(v * v for v in b_l[i])) if fin[i] else 0.0
            for deg, v0, coef in ((xdeg[i], xs_l[i][0], A), (ydeg[i], ys_l[i][0], B)):
                if not deg:
                    continue
                if not fin[i]:
                    c6.append(float("inf"))
                    continue
                dot = abs(C * Fraction(v0) - coef)
                nv = math.sqrt(v0 * v0 + 1.0)
                c6.append(float(dot) / (nb * nv) / eps if nb > 0 else 0.0)
        add("C6", family, c6, ULPS["C6"])

        add("C7", family, [orc.relative_error(b_l[i], orc.exact_min_norm(xs_l[i], ys_l[i], z_l[i]))
                           / eps for i in range(n)], ULPS["C7"])
    return rows


def contract_rank_rule(parent):
    return lambda grid, z: lc.chosen_rank(parent, grid, z)


# ---------------------------------------------------------------- planted bugs

@contextlib.contextmanager
def _patched(name, fn):
    old = getattr(lc, name)
    setattr(lc, name, fn)
    try:
        yield
    finally:
        setattr(lc, name, old)


def _with(name, fn, solve):
    def mutant(grid, z):
        with _patched(name, fn):
            return solve(grid, z)
    return mutant


def _min_norm_c_is_k(k, a0, b0, xbar, ybar, xdeg, ydeg):
    zero = torch.zeros_like(k)
    A = torch.where(xdeg, k * torch.where(xdeg, xbar, zero) / (1 + xbar * xbar), a0)
    B = torch.where(ydeg, k * torch.where(ydeg, ybar, zero) / (1 + ybar * ybar), b0)
    return torch.stack((k, A, B), dim=1)


def _min_norm_zero_share(k, a0, b0, xbar, ybar, xdeg, ydeg):
    return torch.stack((k, a0, b0), dim=1)


def _centred_mean(v):
    m = v.mean(dim=1)
    return v - m[:, None], m


def _stats_variance_test(grid):
    xs, ys = lc._grid_axes(grid)
    dx, xbar = _centred_mean(xs)
    dy, ybar = _centred_mean(ys)
    sxx, syy = (dx * dx).sum(1), (dy * dy).sum(1)
    return lc._Stats(xs.shape[1], xs, ys, dx, dy, xbar, ybar, sxx, syy, sxx == 0, syy == 0)


def _axes_y_slowest(grid):
    s = math.isqrt(grid.shape[1])
    return grid[:, :s, 0], grid[:, ::s, 1]


def _cutoff_other_axis(st, rcond):
    cx, cy = _orig_cutoff(st, rcond)
    one = cx ^ cy
    return torch.where(one, cy, cx), torch.where(one, cx, cy)


_orig_cutoff = lc.gelsd_cutoff_axes


def _sigmoid_squared(x):
    sig = torch.sigmoid(x)
    return sig * sig


def _swap_ab(solve):
    return lambda grid, z: solve(grid, z)[:, [0, 2, 1]]


def _negate_a(solve):
    def mutant(grid, z):
        b = solve(grid, z).clone()
        b[:, 1] = -b[:, 1]
        return b
    return mutant


# (id, candidate it is planted in, description, solver)
MUTANTS = [
    ("S1", "A1", "slope A with the wrong sign", _negate_a(lc.solve_A1)),
    ("S2", "A1", "degenerate axis: C = k instead of k / (1 + x0^2)",
     _with("_min_norm", _min_norm_c_is_k, lc.solve_A1)),
    ("S3", "A1", "degeneracy from a variance about fl(mean), not the endpoints",
     _with("_stats", _stats_variance_test, lc.solve_A1)),
    ("S4", "A1", "centred on fl(mean) instead of a data point",
     _with("_centred", _centred_mean, lc.solve_A1)),
    ("S5", "A1", "A and B swapped", _swap_ab(lc.solve_A1)),
    ("S6", "A1", "grid read as y running slowest", _with("_grid_axes", _axes_y_slowest, lc.solve_A1)),
    ("S7", "A2", "cutoff drops the other axis", _with("gelsd_cutoff_axes", _cutoff_other_axis,
                                                       lc.solve_A2)),
    ("S8", "A1", "degenerate axis gets coefficient 0 and C = k",
     _with("_min_norm", _min_norm_zero_share, lc.solve_A1)),
    ("S9", "A3", "h = sigmoid^2 instead of sigmoid'", _with("_sigmoid_prime", _sigmoid_squared,
                                                            lc.solve_A3)),
    ("S10", "P", "pinv cutoff rtol = 0.1", lambda grid, z: lc.solve_P(grid, z, rcond=0.1)),
]
