"""Closed-form plane fit for the precise transformer, a GPU replacement for gelsd.

fit_planes(grid, z) returns what lin_reg_tensor_batched returns: for every box the
minimum-norm least-squares coefficients (C, A, B) of z ~ C + A x + B y on its
s x s sampling grid (get_linspace_batched: x runs slowest, y fastest), with the
rank decided like gelsd's rule s_i <= rcond * s_1. It is elementwise over the
boxes: no LAPACK call, no host sync, no transfer to the CPU.

On a Cartesian grid the centred columns of X = [1, x, y] are orthogonal, so
X = Q (s K) with K = [[1, xbar, ybar], [0, sx, 0], [0, 0, sy]] (sx, sy the standard
deviations of the s values of each axis), the normal equations decouple and the
singular values of X follow from the 3 x 3 matrix K. A degenerate axis (equal
endpoints, or dropped by the cutoff) makes its column collinear with the
intercept, and the minimum-norm solution splits the intercept between C and that
axis' coefficient.

This is candidate A2 of experiments/lstsq_candidates.py, chosen by the
pre-registered comparison in experiments/compare_lstsq_solvers.py (see
results/lstsq_comparison_cuda_2026-10-27/README.md); experiments/test_planar_fit.py
checks that both give the same bits. Known limits: the analytic cutoff matches
gelsd's rank decision except within a narrow band at the threshold, where the
solution jumps by O(1) as it must at a rank switch, and treating a dropped axis
as constant is close to, not identical with, gelsd's truncated-SVD projection.
"""
import math
from collections import namedtuple

import torch


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
    """2^e per row, where max |v| = f 2^e with 0.5 <= f < 1 (1 for a zero row).

    Sums of squares and products are formed from values divided by this power of
    two and multiplied back afterwards. That is exact while no scaled intermediate
    is subnormal, so the result then has the bits of the unscaled sums wherever
    those are finite, and stays finite where they overflow.
    """
    m = v.abs().amax(dim=1)
    if m.device.type == "mps":
        # frexp is not implemented on MPS: take the n exponents on the CPU. One host
        # sync per call, and a fit scales x, y and z, so up to three per fit.
        _, e = torch.frexp(m.cpu())
        return torch.ldexp(torch.ones_like(e, dtype=m.dtype), e).to(m.device)
    _, e = torch.frexp(m)
    return torch.ldexp(torch.ones_like(m), e)


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
    are sums of non-negative terms and so accurate to a few ulps; forming X^T X
    and calling an eigensolver resolves l3 only down to about eps * l1. l1 is the
    largest root of l^3 - e1 l^2 + e2 l - e3; Newton from l = e1 >= l1 descends
    monotonically because the cubic is convex and increasing on [l1, e1]. Then
    l2 + l3 = (e2 - l2 l3) / l1 and l2 l3 = e3 / l1, with l3 from the product to
    avoid cancellation.
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


def fit_planes(grid, z, rcond=None):
    """[n, 3] coefficients (C, A, B) for grid [n, s*s, 2] and targets z [n, s*s]
    (or [n, s*s, 1]), on the device and in the dtype of the input."""
    if z.dim() == 3:
        z = z.squeeze(-1)
    st = _stats(grid)
    rcond = default_rcond(grid.dtype, grid.shape[1]) if rcond is None else rcond
    cx, cy = gelsd_cutoff_axes(st, rcond)
    return _fit_targets(st, z, st.xdeg | cx, st.ydeg | cy)
