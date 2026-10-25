"""Exact references for experiments/compare_lstsq_solvers.py.

exact_min_norm: the minimum-norm least-squares coefficients (C, A, B) of one box,
computed with fractions.Fraction on the very same floating-point grid values and
targets, so any error left in a solver's answer is the solver's own. It solves
the normal equations generically and decides the rank exactly; it uses nothing
of the Cartesian structure, so it shares no derivation with the closed forms it
checks.

true_max_error: max |sigmoid'(x) y - (A x + B y + C)| over the box for given
floating-point (A, B, C), in 50-digit arithmetic. For fixed x the error is
affine in y, so the maximum lies on the edges y = ly and y = uy; on each edge
the candidates are the two corners and the roots in [lx, ux] of
sigmoid''(x) y0 = A, i.e. 2 s^3 - 3 s^2 + s - A / y0 = 0 with s = sigmoid(x).
"""
import math
from fractions import Fraction

import mpmath
from mpmath.libmp import NoConvergence


def _independent_rows(M):
    """Indices of a maximal set of linearly independent rows of M, kept in order."""
    basis, chosen = [], []
    for i, row in enumerate(M):
        v = list(row)
        for piv, brow in basis:
            if v[piv] != 0:
                f = v[piv] / brow[piv]
                v = [a - f * b for a, b in zip(v, brow)]
        piv = next((k for k, a in enumerate(v) if a != 0), None)
        if piv is not None:
            basis.append((piv, v))
            chosen.append(i)
    return chosen


def _solve(M, b):
    """Gauss-Jordan elimination for a nonsingular square system, in Fractions."""
    k = len(M)
    a = [list(M[i]) + [b[i]] for i in range(k)]
    for col in range(k):
        piv = next(r for r in range(col, k) if a[r][col] != 0)
        a[col], a[piv] = a[piv], a[col]
        for r in range(k):
            if r != col and a[r][col] != 0:
                f = a[r][col] / a[col][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [a[i][k] / a[i][i] for i in range(k)]


def exact_min_norm(xs, ys, z):
    """(C, A, B) as Fractions.

    xs, ys: the s grid values of each axis (floats); z: the s*s targets with x
    running slowest. The solutions of the normal equations G beta = X^T z are the
    least-squares solutions; with R the independent rows of G, the minimum-norm
    one is R^T (R R^T)^-1 r.
    """
    X = [(Fraction(1), Fraction(x), Fraction(y)) for x in xs for y in ys]
    Z = [Fraction(v) for v in z]
    G = [[sum(r[i] * r[j] for r in X) for j in range(3)] for i in range(3)]
    rhs = [sum(r[i] * zv for r, zv in zip(X, Z)) for i in range(3)]
    rows = _independent_rows(G)
    if not rows:
        return (Fraction(0),) * 3
    R = [G[i] for i in rows]
    rb = [rhs[i] for i in rows]
    RRt = [[sum(R[i][c] * R[j][c] for c in range(3)) for j in range(len(R))]
           for i in range(len(R))]
    w = _solve(RRt, rb)
    return tuple(sum(R[i][c] * w[i] for i in range(len(R))) for c in range(3))


def relative_error(coef, exact):
    """max_i |coef_i - exact_i| / max_i |exact_i| (absolute when exact is 0); inf
    for a non-finite coefficient."""
    if not all(math.isfinite(c) for c in coef):
        return float("inf")
    diff = max(abs(Fraction(c) - e) for c, e in zip(coef, exact))
    scale = max(abs(e) for e in exact)
    return float(diff / scale) if scale != 0 else float(diff)


def true_max_error(A, B, C, lx, ux, ly, uy, dps=50):
    """The exact maximum error of the plane over the box, as a float (nan if the
    root finder does not converge)."""
    with mpmath.workdps(dps):
        A, B, C = mpmath.mpf(A), mpmath.mpf(B), mpmath.mpf(C)
        lo, hi = mpmath.mpf(lx), mpmath.mpf(ux)
        tol = mpmath.mpf(10) ** (-(dps // 2))
        best = mpmath.mpf(0)
        for y0f in (ly, uy):
            y0 = mpmath.mpf(y0f)
            xs = [lo, hi]
            if y0 != 0:
                c = A / y0
                roots = None
                for steps, extra in ((100, 2 * dps), (500, 8 * dps)):
                    try:
                        roots = mpmath.polyroots([2, -3, 1, -c], maxsteps=steps, extraprec=extra)
                        break
                    except NoConvergence:
                        continue
                if roots is None:
                    return float("nan")
                for r in roots:
                    r = mpmath.mpc(r)
                    s = r.real
                    # A double root of sigma'' = c is an inflection of the edge
                    # function, not an extremum, so dropping one is harmless.
                    if abs(r.imag) <= tol * (1 + abs(s)) and 0 < s < 1:
                        x = mpmath.log(s / (1 - s))
                        if lo <= x <= hi:
                            xs.append(x)
            for x in xs:
                sg = 1 / (1 + mpmath.exp(-x))
                e = abs(sg * (1 - sg) * y0 - (A * x + B * y0 + C))
                if e > best:
                    best = e
        return float(best)
