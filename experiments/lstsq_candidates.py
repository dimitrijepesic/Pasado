"""Candidate replacements for the batched gelsd plane fit of the precise transformer.

Each solver maps the sampling grid [n, s*s, 2] built by
precise_transformer.get_linspace_batched (x runs slowest, y fastest) and the
targets z [n, s*s] to coefficients [n, 3] in the order (C, A, B), like
precise_transformer.lin_reg_tensor_batched, on the device of the input.

  G   gelsd on CPU: the current code and the baseline
  A1  closed form; an axis is degenerate when its endpoints are equal
  A2  closed form; an axis is degenerate when gelsd's cutoff would drop it. The
      analytic cutoff matches gelsd's rank decision almost always, but not bit
      for bit at the threshold itself, and treating the dropped axis as constant
      is close to, not identical with, gelsd's truncated-SVD projection.
  A3  closed form fitted to the s values of sigmoid'(x); assumes z = sigmoid'(x) * y
  H   A1, plus gelsd on CPU for the boxes gelsd truncates beyond A1 (one host sync)
  P   torch.linalg.pinv with gelsd's cutoff (one host sync per call)

The closed form: on a Cartesian grid the centred columns of X = [1, x, y] are
orthogonal, so X = Q (s K) with K = [[1, xbar, ybar], [0, sx, 0], [0, 0, sy]]
(sx, sy the standard deviations of the s values of each axis) and the normal
equations decouple. A degenerate axis makes its column collinear with the
intercept; the minimum-norm solution then splits the intercept between C and
that axis' coefficient.

The sums of squares and of products are formed from centred values divided by a
power of two near their maximum, and the power is multiplied back afterwards.
Scaling by a power of two is exact, so the result has the same bits as the
unscaled sums wherever those are finite, and stays finite where they overflow
(run v1 of the comparison: centred y of 1e294 times centred targets of 1e293).

Compared by experiments/compare_lstsq_solvers.py. Nothing in the analysis uses
these solvers yet.
"""
import math
import os
import sys
from collections import namedtuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "forward_mode_tensorized_src"))

import precise_transformer as pt  # noqa: E402

# Bound at import time, so code that swaps pt.lin_reg_tensor_batched for a
# candidate still reaches the CPU solve through G and H.
_gelsd = pt.lin_reg_tensor_batched


def default_rcond(dtype, m):
    """torch's default cutoff for gelsd: eps * max(m, n), with n = 3 unknowns."""
    return torch.finfo(dtype).eps * max(m, 3)


# dxn = dx / ax with ax a power of two, nxx = sum(dxn^2), so sum(dx^2) = ax^2 nxx
_Stats = namedtuple("_Stats", "s xs ys dxn dyn xbar ybar ax ay nxx nyy xdeg ydeg")


def _grid_axes(grid):
    """The s distinct x values and the s distinct y values of each grid, [n, s] each."""
    s = math.isqrt(grid.shape[1])
    assert s * s == grid.shape[1], f"grid has {grid.shape[1]} points, not s * s"
    return grid[:, ::s, 0], grid[:, :s, 1]


def _centred(v):
    """(v - v0) minus its mean, and the mean of v computed as v0 + mean(v - v0).

    Shifting by a data point first keeps the differences exact for a narrow box
    away from 0 (Sterbenz), whereas v - fl(mean(v)) carries an error of
    eps * |mean| into every point (Chan, Golub and LeVeque 1983).
    """
    r = v - v[:, :1]
    rm = r.mean(dim=1, keepdim=True)
    return r - rm, v[:, 0] + rm[:, 0]


def _pow2_scale(v):
    """2^e per row, where max |v| = f 2^e with 0.5 <= f < 1 (1 for a zero row)."""
    _, e = torch.frexp(v.abs().amax(dim=1))
    return torch.ldexp(torch.ones_like(v[:, 0]), e)


def _stats(grid):
    xs, ys = _grid_axes(grid)
    dx, xbar = _centred(xs)
    dy, ybar = _centred(ys)
    ax, ay = _pow2_scale(dx), _pow2_scale(dy)
    dxn, dyn = dx / ax[:, None], dy / ay[:, None]
    nxx = (dxn * dxn).sum(1)
    nyy = (dyn * dyn).sum(1)
    # Degeneracy from the exact endpoints: the mean of s equal floats is not always
    # that float, so a computed variance can be 1e-34 instead of 0. A zero sum of
    # the unscaled squares also catches widths so small that the squares underflow.
    xdeg = (xs[:, 0] == xs[:, -1]) | ((dx * dx).sum(1) == 0)
    ydeg = (ys[:, 0] == ys[:, -1]) | ((dy * dy).sum(1) == 0)
    return _Stats(xs.shape[1], xs, ys, dxn, dyn, xbar, ybar, ax, ay, nxx, nyy, xdeg, ydeg)


def _spreads(st):
    """Standard deviations sx, sy of the s values of each axis, without squaring them."""
    return st.ax * (st.nxx / st.s).sqrt(), st.ay * (st.nyy / st.s).sqrt()


def _safe(den, deg):
    """den where the axis is not degenerate, 1 elsewhere: torch.where evaluates both
    branches, and 0/0 in the unused one would still poison gradients."""
    return torch.where(deg, torch.ones_like(den), den)


def _min_norm(k, a0, b0, xbar, ybar, xdeg, ydeg):
    """(C, A, B) given the fit k = mean(z) - a0 xbar - b0 ybar of the intercept.

    On a degenerate axis the solutions form a line, and the minimum-norm point on it
    is k * (1, x0) / (1 + x0^2) (both axes: k * (1, x0, y0) / (1 + x0^2 + y0^2)).
    Written with m = max(1, |x0|, |y0|) so that x0^2 cannot overflow.
    """
    zero = torch.zeros_like(k)
    u = torch.where(xdeg, xbar, zero)
    v = torch.where(ydeg, ybar, zero)
    m = torch.maximum(torch.ones_like(k), torch.maximum(u.abs(), v.abs()))
    inv, un, vn = 1.0 / m, u / m, v / m
    denom = inv * inv + un * un + vn * vn
    km = k / m
    C = km * inv / denom
    A = torch.where(xdeg, km * un / denom, a0)
    B = torch.where(ydeg, km * vn / denom, b0)
    return torch.stack((C, A, B), dim=1)


def _fit_targets(st, z, xdeg, ydeg):
    """Closed form on the s*s targets."""
    n, s = z.shape[0], st.s
    z3 = z.reshape(n, s, s)  # [n, x index, y index]
    zbar = z.mean(1)
    dz = z3 - zbar[:, None, None]
    az = _pow2_scale(dz.reshape(n, s * s))
    zx = dz.mean(2) / az[:, None]
    zy = dz.mean(1) / az[:, None]
    zero = torch.zeros_like(zbar)
    # Elementwise products and sums, not matmul: no TF32 on CUDA in float32. The
    # scaled slope is multiplied by az and divided by ax in two exact steps.
    a0 = torch.where(xdeg, zero,
                     (st.dxn * zx).sum(1) / _safe(st.nxx, xdeg) * az / _safe(st.ax, xdeg))
    b0 = torch.where(ydeg, zero,
                     (st.dyn * zy).sum(1) / _safe(st.nyy, ydeg) * az / _safe(st.ay, ydeg))
    k = zbar - a0 * st.xbar - b0 * st.ybar
    return _min_norm(k, a0, b0, st.xbar, st.ybar, xdeg, ydeg)


def gram_eigenvalues(xbar, ybar, sx, sy, steps=8):
    """Eigenvalues l1 >= l2 >= l3 of K^T K / m^2, m = max(1, |xbar|, |ybar|, sx, sy).

    sx, sy are the standard deviations; dividing every entry of K by m before
    squaring keeps the invariants finite for any finite box. Uses the invariants
    e1 (trace), e2 (sum of the principal 2x2 minors) and e3 (determinant), which
    are sums of non-negative
    terms and so accurate to a few ulps; forming X^T X and calling an eigensolver
    resolves l3 only down to about eps * l1. l1 is the largest root of
    l^3 - e1 l^2 + e2 l - e3; Newton from l = e1 >= l1 descends monotonically
    because the cubic is convex and increasing on [l1, e1]. Then l2 + l3 =
    (e2 - l2 l3) / l1 and l2 l3 = e3 / l1, with l3 from the product to avoid
    cancellation.
    """
    one = torch.ones_like(xbar)
    zero = torch.zeros_like(xbar)
    m = torch.maximum(torch.maximum(one, torch.maximum(xbar.abs(), ybar.abs())),
                      torch.maximum(sx, sy))
    c2 = (one / m) ** 2
    a2 = (xbar / m) ** 2
    b2 = (ybar / m) ** 2
    p2 = (sx / m) ** 2
    q2 = (sy / m) ** 2
    e1 = c2 + a2 + b2 + p2 + q2
    e2 = p2 * (c2 + b2) + q2 * (c2 + a2) + p2 * q2
    e3 = c2 * p2 * q2
    l1 = e1
    for _ in range(steps):
        f = ((l1 - e1) * l1 + e2) * l1 - e3
        df = (3.0 * l1 - 2.0 * e1) * l1 + e2
        l1 = l1 - torch.where(df > 0, f / _safe(df, df <= 0), zero)
    prod = e3 / l1
    total = ((e2 - prod) / l1).clamp_min(0)
    disc = (total * total - 4.0 * prod).clamp_min(0)
    l2 = 0.5 * (total + disc.sqrt())
    l3 = torch.where(l2 > 0, prod / _safe(l2, l2 <= 0), zero)
    # degeneracy measure of each axis: its spread relative to its offset
    tx = p2 / (c2 + a2)
    ty = q2 / (c2 + b2)
    return l1, l2, l3, tx, ty


def gelsd_cutoff_axes(st, rcond):
    """Axes gelsd's rule s_i <= rcond * s_1 drops, as two [n] masks.

    Both axes when two singular values fall under the cutoff; with one, the axis
    whose spread is smaller relative to its offset from the origin (an exactly
    degenerate axis always has spread 0, so it is the one chosen).
    """
    l1, l2, l3, tx, ty = gram_eigenvalues(st.xbar, st.ybar, *_spreads(st))
    r2 = rcond * rcond
    drop2 = l2 <= r2 * l1
    drop3 = l3 <= r2 * l1
    pick_x = tx <= ty
    return drop2 | (drop3 & pick_x), drop2 | (drop3 & ~pick_x)


def singular_value_ratios(grid):
    """s2/s1 and s3/s1 of X = [1, x, y] for each grid, from gram_eigenvalues."""
    st = _stats(grid)
    l1, l2, l3, _, _ = gram_eigenvalues(st.xbar, st.ybar, *_spreads(st))
    return (l2 / l1).sqrt(), (l3 / l1).sqrt()


def _flat_targets(z):
    return z.squeeze(-1) if z.dim() == 3 else z


def solve_G(grid, z):
    return _gelsd(grid, z)


def solve_A1(grid, z):
    st = _stats(grid)
    return _fit_targets(st, _flat_targets(z), st.xdeg, st.ydeg)


def solve_A2(grid, z, rcond=None):
    st = _stats(grid)
    rcond = default_rcond(grid.dtype, grid.shape[1]) if rcond is None else rcond
    cx, cy = gelsd_cutoff_axes(st, rcond)
    return _fit_targets(st, _flat_targets(z), st.xdeg | cx, st.ydeg | cy)


def _sigmoid_prime(x):
    sig = torch.sigmoid(x)
    return (1.0 - sig) * sig


def solve_A3(grid, z):
    """Closed form for z = h(x) * y with h = sigmoid', from the s values h(x_i).

    With separable targets the decoupled slopes reduce to B = mean(h) and
    A = ybar * cov(x, h) / var(x), and mean(z) = mean(h) * ybar: s evaluations of
    h per box instead of s*s targets. z is not read. h uses the association of the
    production targets, (1 - sigmoid) * sigmoid.
    """
    st = _stats(grid)
    h = _sigmoid_prime(st.xs)
    hbar = h.mean(1)
    zero = torch.zeros_like(hbar)
    # same operation order as the unscaled ybar * sum(dx (h - hbar)) / sum(dx^2)
    a0 = torch.where(st.xdeg, zero,
                     st.ybar * (st.dxn * (h - hbar[:, None])).sum(1) / _safe(st.nxx, st.xdeg)
                     / _safe(st.ax, st.xdeg))
    b0 = torch.where(st.ydeg, zero, hbar)
    k = hbar * st.ybar - a0 * st.xbar - b0 * st.ybar
    return _min_norm(k, a0, b0, st.xbar, st.ybar, st.xdeg, st.ydeg)


def solve_H(grid, z, rcond=None):
    z = _flat_targets(z)
    st = _stats(grid)
    coef = _fit_targets(st, z, st.xdeg, st.ydeg)
    rcond = default_rcond(grid.dtype, grid.shape[1]) if rcond is None else rcond
    cx, cy = gelsd_cutoff_axes(st, rcond)
    flag = (cx & ~st.xdeg) | (cy & ~st.ydeg)
    if bool(flag.any()):  # host sync
        idx = flag.nonzero(as_tuple=True)[0]
        coef[idx] = _gelsd(grid[idx], z[idx])
    return coef


def solve_P(grid, z, rcond=None):
    z = _flat_targets(z)
    n, m, _ = grid.shape
    X = torch.cat((grid.new_ones(n, m, 1), grid), dim=2)
    rcond = default_rcond(grid.dtype, m) if rcond is None else rcond
    return (torch.linalg.pinv(X, rtol=rcond) @ z.unsqueeze(-1)).squeeze(-1)


def gelsd_rank(grid, z):
    """The rank gelsd decides for each box (CPU), with torch's default rcond."""
    g, t = grid.detach().cpu(), _flat_targets(z).detach().cpu()
    X = torch.cat((g.new_ones(g.shape[0], g.shape[1], 1), g), dim=2)
    return torch.linalg.lstsq(X, t.unsqueeze(-1), driver="gelsd").rank


def exact_rank(grid):
    """1 + number of axes whose endpoints differ."""
    st = _stats(grid)
    return 3 - st.xdeg.long() - st.ydeg.long()


def chosen_rank(name, grid, z, rcond=None):
    """The rank each candidate's own rule decides, on the CPU."""
    rcond = default_rcond(grid.dtype, grid.shape[1]) if rcond is None else rcond
    if name in ("A1", "A3"):
        return exact_rank(grid).cpu()
    if name == "G":
        return gelsd_rank(grid, z)
    st = _stats(grid)
    cx, cy = gelsd_cutoff_axes(st, rcond)
    if name == "A2":
        return (3 - (st.xdeg | cx).long() - (st.ydeg | cy).long()).cpu()
    if name == "H":
        flag = ((cx & ~st.xdeg) | (cy & ~st.ydeg)).cpu()
        return torch.where(flag, gelsd_rank(grid, z), exact_rank(grid).cpu())
    if name == "P":
        X = torch.cat((grid.new_ones(grid.shape[0], grid.shape[1], 1), grid), dim=2)
        s = torch.linalg.svdvals(X).cpu()
        return (s > rcond * s[:, :1]).sum(1)
    raise ValueError(name)


CANDIDATES = {
    "G": solve_G,
    "A1": solve_A1,
    "A2": solve_A2,
    "A3": solve_A3,
    "H": solve_H,
    "P": solve_P,
}

# Last tie-break of the comparison: 0 = elementwise only, 1 = one host sync per
# call, 2 = the solve itself runs on the CPU.
SIMPLICITY = {"A1": 0, "A2": 0, "A3": 0, "H": 1, "P": 1, "G": 2}

# A3 rebuilds the separable targets from the grid and never reads z.
READS_Z = {"G": True, "A1": True, "A2": True, "A3": False, "H": True, "P": True}
