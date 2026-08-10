import math
import os
import torch
from sklearn import linear_model
from SimpleZono import *


# ---------------------------------------------------------------------------
# Experimental implementation selectors (Phase 3 / Phase 6 of the 5.4 sprint).
#
# PASADO_VECTORIZED_PRECISE=1 (default)  -> compute_max_error uses the vectorized
#     check_corners_tensor / check_nonlinear_boundary_tensor path.
# PASADO_VECTORIZED_PRECISE=0            -> falls back to the original per-neuron
#     check_corners / check_nonlinear_boundary Python-loop path.
#
# The selector affects ONLY the corner and nonlinear-boundary error checks. The
# linear regression, the ABCs, the np.random perturbation of A, and every bound
# computation are byte-for-byte identical on both paths, so a seeded run differs
# only by floating-point reordering inside the two checks.
#
# The module-level flags are read at *call* time, so an in-process harness can
# flip e.g. `precise_transformer.USE_VECTORIZED_PRECISE = False` between runs.
# ---------------------------------------------------------------------------
USE_VECTORIZED_PRECISE = os.environ.get("PASADO_VECTORIZED_PRECISE", "1") != "0"

# PASADO_VEC_BOUNDARY lets the nonlinear-boundary check be toggled independently
# of the corner check, so the benchmark matrix can express the "corners only"
# intermediate variant. Default follows PASADO_VECTORIZED_PRECISE. It only has an
# effect when ABCs_tensor is supplied (i.e. USE_VECTORIZED_PRECISE is on).
USE_VECTORIZED_BOUNDARY = os.environ.get("PASADO_VEC_BOUNDARY", "1") != "0"

# PASADO_BATCHED_LSTSQ=1 -> sigmoid_prime_product_tensor solves all per-neuron
#     regressions with a single batched lin_reg_tensor_batched call per layer.
# PASADO_BATCHED_LSTSQ=0 (default) -> original per-neuron lin_reg_tensor loop.
# The batched solve is bit-identical to the loop (same design matrices, same
# gelsd driver, batch iterated at C++ level) and consumes the np.random stream
# in the same per-neuron order, so seeded runs match exactly. Default stays off
# so production behavior is unchanged unless the variant is selected explicitly.
USE_BATCHED_LSTSQ = os.environ.get("PASADO_BATCHED_LSTSQ", "0") != "0"

# PASADO_BATCHED_GRID=1 -> the per-neuron sampling grids are built by
#     get_linspace_batched in one shot instead of the per-neuron torch.linspace /
#     cartesian_prod loop. Verified bit-identical (see get_linspace_batched).
# PASADO_BATCHED_GRID=0 (default) -> original loop. Default off so production
#     behavior is unchanged unless the variant is selected explicitly.
USE_BATCHED_GRID = os.environ.get("PASADO_BATCHED_GRID", "0") != "0"

# PASADO_REAL_CUBIC=1 -> inverse_sigmoid_2nd_deriv solves the cubic with Viete's
#     real trigonometric form instead of Cardano's formula in complex64.
# PASADO_REAL_CUBIC=0 (default) -> original complex64 path.
#
# Unlike the other selectors this one is NOT bit-identical: it deliberately
# changes numerics, because the complex64 path computes the roots in single
# precision inside an otherwise float64 analysis (measured residuals 3.8e-06 vs
# 1.2e-16 -- see inverse_poly_real_tensor). It also removes the only complex
# arithmetic in the precise path, which is one of the three reasons TorchInductor
# could not compile it. Default stays off until the end-to-end effect on the
# reported bounds has been measured.
USE_REAL_CUBIC = os.environ.get("PASADO_REAL_CUBIC", "0") != "0"


# lx, ly, ux, uy are TENSORS/VECTORS who have as many columns as neurons/variables in a layer
def get_linspace(lx, ux, ly, uy, STEPS=5):
    xs = [torch.linspace(lx[i].item(), ux[i].item(), steps=STEPS) for i in range(lx.shape[0])]
    ys = [torch.linspace(ly[i].item(), uy[i].item(), steps=STEPS) for i in range(ly.shape[0])]

    zs = [torch.cartesian_prod(xs[i], ys[i]) for i in range(len(xs))]
    return zs


def _linspace_rows(lo, hi, STEPS):
    """Row-wise torch.linspace: [n] x [n] -> [n, STEPS], bit-identical.

    torch.linspace does NOT compute start + i*step throughout; it fills the
    first half forward from `start` and the second half backward from `end`
    (the same trick numpy uses, so the endpoints are exact and the result is
    symmetric). Reproducing only the forward half of that formula differs by
    1 ULP on roughly a quarter of random inputs -- which is what an earlier
    note in NEXT_RESEARCH_DIRECTIONS.md recorded as "batched linspace is not
    bit-identical". Mirroring both halves makes it exact, so batching the grid
    needs no precision concession at all.
    """
    t = torch.arange(STEPS, dtype=lo.dtype, device=lo.device)
    step = (hi - lo) / (STEPS - 1)
    out = lo.unsqueeze(1) + t * step.unsqueeze(1)
    tail = torch.arange(STEPS, device=lo.device) >= STEPS // 2
    out[:, tail] = hi.unsqueeze(1) - (STEPS - 1 - t[tail]) * step.unsqueeze(1)
    return out


def get_linspace_batched(lx, ux, ly, uy, STEPS=5):
    """[n, STEPS*STEPS, 2], equal to torch.stack(get_linspace(...)) bit for bit.

    Replaces 2n torch.linspace calls, n torch.cartesian_prod calls, 4n .item()
    calls and the Python loop with a handful of whole-tensor operations. The
    .item() calls matter beyond their cost: each is a device-host sync, so the
    loop version cannot run on a GPU without stalling on every neuron
    (7.87M syncs on the `big` benchmark).

    The cartesian product is assembled by repeat_interleave / repeat, which is
    pure data movement and therefore exact by construction. Verified bit-exact
    against the loop for float32 and float64, STEPS in {3,5,7,9}, point boxes
    (lx==ux, ly==uy), widths down to 1e-12, magnitudes up to 1e6, and negative
    ranges.
    """
    xs = _linspace_rows(lx, ux, STEPS)
    ys = _linspace_rows(ly, uy, STEPS)
    return torch.stack((xs.repeat_interleave(STEPS, dim=1),
                        ys.repeat(1, STEPS)), dim=2)

# sigmoid(x)*(1-sigmoid(x))*y
def sigmoid_prime_times_y(xy_tensor):
    xs = xy_tensor[:, 0]
    ys = xy_tensor[:, 1]
    return (1.0 - torch.sigmoid(xs)) * (torch.sigmoid(xs) * ys)


def softplus_prime_times_y(xy_tensor):
    xs = xy_tensor[:, 0]
    ys = xy_tensor[:, 1]
    return (torch.sigmoid(xs) * ys)


# converts a list of tensors like [tensor(3.4), tensor(11.2), tensor(7)]
# to a tensor of the form tensor([3.4, 11.2, 7])
def lst_as_tensor(lst_of_tensors):
    # compute_max_error now returns an already-stacked [n] tensor on the fully
    # vectorized path; pass it through so callers need no branch of their own.
    if torch.is_tensor(lst_of_tensors):
        return lst_of_tensors
    res = torch.stack(lst_of_tensors)
    return res

def PlanarApproximation(A, B, C, xy_tensor):
    xs = xy_tensor[:, 0]
    ys = xy_tensor[:, 1]
    return (A * xs + B * ys + C)  # this does elementwise multiplication (thats what we want)


# works
def inv_sigmoid(y):
    return torch.log(y / (1. - y))


# useful for checking/debugging the inverse function
def sigmoid_double_prime(xs):
    return (1.0 - torch.sigmoid(xs)) * (torch.sigmoid(xs)) * (1 - 2 * torch.sigmoid(xs))


"""
#sklearn version so probably inefficient compared to pytorch  ltsq function
#https://pytorch.org/docs/stable/generated/torch.lstsq.html
def lin_reg(input_points,zs):
	#zs = f(input_points)
	ols = linear_model.LinearRegression()
	model = ols.fit(input_points, zs)
	A,B = model.coef_
	C = model.intercept_
	return (A,B,C)
"""


# torch vectorized version of linear regression
# https://medium.com/@rcorbish/linear-regression-using-pytorch-dcf0165e3a6e
def lin_reg_tensor(x, zs):
    # zs = f(x)
    xplusone = torch.cat((torch.ones(x.size(0), 1), x), 1)
    R = torch.linalg.lstsq(xplusone, zs, driver='gelsd').solution  # torch's least squares solver for linear regression
    R = R[0:xplusone.size(1)]

    #	R = torch.linalg.lstsq(xplusone,zs).solution
    #	shuffled = torch.ones(R.shape)
    #	shuffled[0:-1]=R[1:]
    #	shuffled[-1]=R[0] #THIS IS BECAUSE TORCH ORIGINALLY PUTS THE INTERCEPT FIRST
    #	return shuffled
    #	yh = xplusone.mm( R )
    #	print("R ",R)
    return R


# EXPERIMENTAL batched version of lin_reg_tensor (PASADO_BATCHED_LSTSQ=1).
# Solves the n independent per-neuron planar regressions in ONE batched
# least-squares call instead of a Python loop.
#
#   x_batch:  [n, m, 2] stacked per-neuron (x, y) grid points
#   zs_batch: [n, m] or [n, m, 1] stacked per-neuron target values
#   returns:  [n, 3] coefficients, intercept-first (column 0 = C, 1 = A, 2 = B),
#             exactly the layout lin_reg_tensor returns per neuron.
#
# Matches lin_reg_tensor bit-for-bit: the design matrix is the same [ones | x]
# concatenation and the same LAPACK gelsd routine solves each matrix in the
# batch (ATen iterates the batch at C++ level). Preserves dtype and device of
# x_batch. Validated in experiments/test_batched_lin_reg.py.
def lin_reg_tensor_batched(x_batch, zs_batch):
    if zs_batch.dim() == 2:
        zs_batch = zs_batch.unsqueeze(-1)  # [n, m, 1]
    ones = torch.ones(x_batch.shape[0], x_batch.shape[1], 1,
                      dtype=x_batch.dtype, device=x_batch.device)
    xplusone = torch.cat((ones, x_batch), dim=2)  # [n, m, 3], intercept-first
    R = torch.linalg.lstsq(xplusone, zs_batch, driver='gelsd').solution  # [n, 3, 1]
    return R.squeeze(-1)  # [n, 3]


# tensorized/vectorized version (torch is overloaded for the ** operator)
def cuberoot(x):
    return (x ** (0.33333))


# tensorized/vectorized version
def inverse_poly_tensor(y):
    Q = torch.tensor(-1 / 12.)
    R = 0.25 * y

    R = R.type(torch.complex64)

    J = torch.tensor(1.j)

    sqrt_arg = ((Q ** 3) + (R ** 2))

    sqrt_arg = sqrt_arg.type(torch.complex64)

    S = cuberoot(R + torch.sqrt(sqrt_arg))
    T = cuberoot(R - torch.sqrt(sqrt_arg))

    x1 = S + T + 0.5

    sqrt_3_by_2 = torch.sqrt(torch.tensor(3.)) / 2.
    x2 = -0.5 * (S + T) + 0.5 + (sqrt_3_by_2 * J) * (S - T)
    x3 = -0.5 * (S + T) + 0.5 - (sqrt_3_by_2 * J) * (S - T)

    return (x1, x2, x3)


# Smallest tolerance that never discards a root the complex64 path accepts (see
# the module docstring note on PASADO_REAL_CUBIC). It equals float32 epsilon
# because the production path decides the same discriminant question in
# complex64, i.e. at float32 precision.
_CUBIC_BOUNDARY_TOL = 1e-7


def inverse_poly_real_tensor(y, tol=_CUBIC_BOUNDARY_TOL):
    """The same three roots as inverse_poly_tensor, in real arithmetic.

    inverse_poly_tensor solves the depressed cubic  u^3 - u/4 - y/2 = 0  (which
    is sigma''(x) = y after substituting s = sigma(x) = u + 1/2) by Cardano's
    formula evaluated in complex64. For this cubic p = -1/4 and q = -y/2, so
    4p^3 + 27q^2 < 0 -- all three roots real -- exactly when
    |y| < 1/(6*sqrt(3)), and that bound IS the maximum of |sigma''|. Over the
    whole reachable range the complex path therefore computes three real numbers
    through complex arithmetic. Viete's trigonometric form gives them directly:

        u_k = (1/sqrt(3)) * cos( acos(6*sqrt(3)*y)/3 - 2*pi*k/3 ),  k = 0,1,2

    and the acos argument lies in [-1, 1] precisely when the roots are real, so
    the range test comes for free.

    Two measured consequences (experiments in lipschitz-comparison/probe_*.py):

    * Accuracy. Residuals |2s^3 - 3s^2 + s - y| for the returned roots:
      complex64 path median 3.8e-06, this path median 1.2e-16. The old roots are
      single-precision because complex64 is, inside an analysis that is float64
      everywhere else -- and inv_sigmoid amplifies a 1e-6 error in s to ~1 in the
      returned x when s approaches 0 or 1.

    * Root count. Compared as sets (the two formulas order roots differently),
      the two paths agree on how many roots are real for every in-range,
      out-of-range and realistic input tested. They disagree only within ~1e-6
      of the boundary, where the complex path's float32 discriminant is itself
      unreliable. `tol` handles that: inside the band the argument is clamped,
      which returns the near-double root rather than discarding it. Keeping a
      candidate can only enlarge the error term (more conservative); dropping one
      is what would risk unsoundness. At tol=1e-7 no candidate is ever dropped.
    """
    arg = 6.0 * math.sqrt(3.0) * y
    phi = torch.acos(arg.clamp(-1.0, 1.0)) / 3.0
    phi = torch.where(arg.abs() <= 1.0 + tol, phi,
                      torch.full_like(phi, float("nan")))
    scale = 1.0 / math.sqrt(3.0)
    two_pi_3 = 2.0 * math.pi / 3.0
    return tuple(scale * torch.cos(phi - k * two_pi_3) + 0.5 for k in (0, 1, 2))



# takes in a bool tensor and returns a float tensor where false gets mapped to NaN and true becomes 1.0
def bool_to_nan(x):
    b1 = torch.isreal(x)
    b1 = b1.float()
    b1[b1 < 0.5] = float('nan')
    return b1


def inf_to_nan(x):
    b = x
    b[torch.isinf(b)] = float('nan')
    return b


# takes a tensor x and returns a new tensor where any x_i not in the range [l,u] is replaced with NAN
def filter_range(l, u, x):
    b = x
    guard = torch.logical_not(torch.logical_and((b <= u), (b >= l)))
    b[guard] = float('nan')
    return b


def inverse_sigmoid_2nd_deriv(y):
    if USE_REAL_CUBIC:
        # Real Viete form: already NaN where no real root exists, so the
        # bool_to_nan/.real dance the complex path needs does not apply.
        (x1, x2, x3) = inverse_poly_real_tensor(y)
        return (inv_sigmoid(x1), inv_sigmoid(x2), inv_sigmoid(x3))

    (x1, x2, x3) = inverse_poly_tensor(y)

    b1 = bool_to_nan(x1)  # sets any complex root to just NaN
    x1 = x1.real * b1

    b2 = bool_to_nan(x2)  # sets any complex root to just NaN
    x2 = x2.real * b2

    b3 = bool_to_nan(x3)  # sets any complex root to just NaN
    x3 = x3.real * b3

    inverted_x1 = (inv_sigmoid(x1))  # inv_sigmoid applied to a NaN is still just NaN
    inverted_x2 = (inv_sigmoid(x2))
    inverted_x3 = (inv_sigmoid(x3))

    return (inverted_x1, inverted_x2, inverted_x3)


def inverse_quadratic_tensor(y):
    inside_sqrt = 1.0 - (4.0 * y)
    inside_sqrt = inside_sqrt.type(torch.complex64)

    radical = torch.sqrt(inside_sqrt)

    x1 = radical * (-0.5) + 0.5
    x2 = (-radical * (-0.5)) + 0.5
    return (x1, x2)


def inverse_softplus_2nd_deriv(y):
    (x1, x2) = inverse_quadratic_tensor(y)

    b1 = bool_to_nan(x1)  # sets any complex root to just NaN
    x1 = x1.real * b1

    b2 = bool_to_nan(x2)  # sets any complex root to just NaN
    x2 = x2.real * b2

    inverted_x1 = (inv_sigmoid(x1))  # inv_sigmoid applied to a NaN is still just NaN
    inverted_x2 = (inv_sigmoid(x2))

    return (inverted_x1, inverted_x2)


# TESTED - seems correct
# evaluate this at the corners
def objective_fn(A, B, C, xy_tensor):
    return sigmoid_prime_times_y(xy_tensor) - PlanarApproximation(A, B, C, xy_tensor)


def objective_fn_softplus(A, B, C, xy_tensor):
    return softplus_prime_times_y(xy_tensor) - PlanarApproximation(A, B, C, xy_tensor)


# CHECKS BOTH F AND -F at the corners
# where F = (f'(x)y-(Ax+By+C))
# and  -F = -(f'(x)y-(Ax+By+C)) = (Ax+By+C)-(f'(x)y)
def check_corners(ABCs, lx, ux, ly, uy):
    maxs = []
    for i in range(lx.size()[0]):
        corner1 = torch.tensor([lx[i], ly[i]])

        corner2 = torch.tensor([lx[i], uy[i]])

        corner3 = torch.tensor([ux[i], ly[i]])

        corner4 = torch.tensor([ux[i], uy[i]])

        all_corners = torch.stack((corner1, corner2, corner3, corner4))
        # print("all corners ",all_corners)
        evaluation = objective_fn(ABCs[i][0], ABCs[i][1], ABCs[i][2], all_corners)
        # print("Evaluation of the corners ",evaluation)

        maxs.append(torch.max(torch.max(evaluation), -torch.min(evaluation)))

    return maxs


# EXPERIMENTAL / vectorized version of check_corners.
# Same semantics as check_corners, but ABCs is passed as a single [n,3] tensor
# (columns A,B,C) instead of a Python list of (A,B,C) tuples, and all n neurons
# are evaluated in one batch instead of a Python for-loop.
#
# Used by compute_max_error when ABCs_tensor is provided; check_corners is kept
# as the default (ABCs_tensor=None) path for rollback.
def check_corners_tensor(ABCs_tensor, lx, ux, ly, uy):
    A = ABCs_tensor[:, 0]
    B = ABCs_tensor[:, 1]
    C = ABCs_tensor[:, 2]

    # all_corners per neuron, in the same order as check_corners:
    # (lx,ly), (lx,uy), (ux,ly), (ux,uy)  -> shape [n, 4]
    xs = torch.stack((lx, lx, ux, ux), dim=1)
    ys = torch.stack((ly, uy, ly, uy), dim=1)

    sig = torch.sigmoid(xs)
    # same association as sigmoid_prime_times_y: (1-s) * (s*y), for bit-identical results
    sigmoid_prime_y = (1.0 - sig) * (sig * ys)
    planar = A.unsqueeze(1) * xs + B.unsqueeze(1) * ys + C.unsqueeze(1)

    evaluation = sigmoid_prime_y - planar  # [n, 4], same as objective_fn per corner

    row_max = torch.max(evaluation, dim=1).values
    row_min = torch.min(evaluation, dim=1).values

    return torch.max(row_max, -row_min)  # [n]


def check_corners_softplus(ABCs, lx, ux, ly, uy):
    maxs = []
    for i in range(lx.size()[0]):
        corner1 = torch.tensor([lx[i], ly[i]])

        corner2 = torch.tensor([lx[i], uy[i]])

        corner3 = torch.tensor([ux[i], ly[i]])

        corner4 = torch.tensor([ux[i], uy[i]])

        all_corners = torch.stack((corner1, corner2, corner3, corner4))
        # print("all corners ",all_corners)
        evaluation = objective_fn_softplus(ABCs[i][0], ABCs[i][1], ABCs[i][2], all_corners)
        # print("Evaluation of the corners ",evaluation)

        maxs.append(torch.max(torch.max(evaluation), -torch.min(evaluation)))

    return maxs


# solves the problem
# max xi in [l_xi,u_xi] sigmoid'(xi)yi - (Ai*xi + Bi*yi + Ci) where yi is fixed as either l_yi or u_yi
def check_nonlinear_boundary(lx, ux, ly, uy, ABCs):
    A_by_ly = torch.tensor([ABCs[i][0] / ly[i] for i in range(
        len(ABCs))])
    A_by_uy = torch.tensor([ABCs[i][0] / uy[i] for i in range(len(ABCs))])
    A_by_ly = (inf_to_nan(A_by_ly))  # Checks the case ly=0 (meaning A/ly is undefined)
    A_by_uy = (inf_to_nan(A_by_uy))

    l = -1.0 / (6.0 * np.sqrt(3.0))
    u = -l

    A_by_ly = filter_range(l, u, A_by_ly)  # checks the case A/ly is NOT in [-.1,.1] (not in image of sigmoid''(x))
    A_by_uy = filter_range(l, u, A_by_uy)

    # NEED TO DO ALL THIS AGAIN FOR A_by_uy
    root1, root2, root3 = inverse_sigmoid_2nd_deriv(
        A_by_ly)  # because of the nature of sigmoid''(x), only at most 2 of these will be real

    # checks if the roots (where the extrema is reached) is within [lx,ux], if not the root is replaced with NaN
    root1 = filter_range(lx, ux, root1)
    root2 = filter_range(lx, ux, root2)
    root3 = filter_range(lx, ux, root3)

    # need to evaluate the original objective at these roots
    root1_paired = torch.stack((root1, ly), dim=1)
    root2_paired = torch.stack((root2, ly), dim=1)
    root3_paired = torch.stack((root3, ly), dim=1)

    roots_paired_stacked = torch.stack((root1_paired, root2_paired, root3_paired), dim=1)

    maxs = []
    for i in range(lx.size()[0]):

        evaluation = objective_fn(ABCs[i][0], ABCs[i][1], ABCs[i][2], roots_paired_stacked[i])
        evaluation = torch.nan_to_num(evaluation, -float("Inf"))
        maxs.append((torch.max(evaluation)))

    # NEED TO DO ALL THIS AGAIN FOR A_by_uy
    root1_uy, root2_uy, root3_uy = inverse_sigmoid_2nd_deriv(
        A_by_uy)  # because of the nature of sigmoid''(x), only at most 2 of these will be real

    # checks if the roots (where the extrema is reached) is within [lx,ux], if not the root is replaced with NaN
    root1_uy = filter_range(lx, ux, root1_uy)
    root2_uy = filter_range(lx, ux, root2_uy)
    root3_uy = filter_range(lx, ux, root3_uy)

    # need to evaluate the original objective at these roots
    root1_uy_paired = torch.stack((root1_uy, uy), dim=1)
    root2_uy_paired = torch.stack((root2_uy, uy), dim=1)
    root3_uy_paired = torch.stack((root3_uy, uy), dim=1)

    roots_uy_paired_stacked = torch.stack((root1_uy_paired, root2_uy_paired, root3_uy_paired), dim=1)

    maxs_uy = []
    for i in range(lx.size()[0]):

        evaluation_uy = objective_fn(ABCs[i][0], ABCs[i][1], ABCs[i][2], roots_uy_paired_stacked[i])
        evaluation_uy = torch.nan_to_num(evaluation_uy, -float("Inf"))
        maxs_uy.append((torch.max(evaluation_uy)))

    return maxs, maxs_uy


def _max_objective_over_x_candidates(A, B, C, xs, ys):
    # xs: [n, k] candidate x-roots per neuron, ys: [n, 1] the fixed y (ly or uy)
    # for that neuron, broadcast against xs. Same math as objective_fn, batched,
    # with NaN candidates (filtered-out roots) mapped to -inf before the max so
    # they can never win - mirrors the per-neuron loop in check_nonlinear_boundary.
    sig = torch.sigmoid(xs)
    # same association as sigmoid_prime_times_y: (1-s) * (s*y), for bit-identical results
    sigmoid_prime_y = (1.0 - sig) * (sig * ys)
    planar = A.unsqueeze(1) * xs + B.unsqueeze(1) * ys + C.unsqueeze(1)
    evaluation = sigmoid_prime_y - planar
    evaluation = torch.nan_to_num(evaluation, -float("Inf"))
    return torch.max(evaluation, dim=1).values


# EXPERIMENTAL / vectorized version of check_nonlinear_boundary.
# Same semantics, but ABCs is passed as a single [n,3] tensor (columns A,B,C)
# instead of a Python list of (A,B,C) tuples, and the two per-neuron
# objective-max loops are replaced with batched tensor ops.
#
# Used by compute_max_error when ABCs_tensor is provided; check_nonlinear_boundary
# is kept as the default (ABCs_tensor=None) path for rollback.
def check_nonlinear_boundary_tensor(lx, ux, ly, uy, ABCs_tensor):
    A = ABCs_tensor[:, 0]
    B = ABCs_tensor[:, 1]
    C = ABCs_tensor[:, 2]

    A_by_ly = A / ly
    A_by_uy = A / uy
    A_by_ly = inf_to_nan(A_by_ly)  # Checks the case ly=0 (meaning A/ly is undefined)
    A_by_uy = inf_to_nan(A_by_uy)

    l = -1.0 / (6.0 * np.sqrt(3.0))
    u = -l

    A_by_ly = filter_range(l, u, A_by_ly)  # checks the case A/ly is NOT in [-.1,.1] (not in image of sigmoid''(x))
    A_by_uy = filter_range(l, u, A_by_uy)

    root1, root2, root3 = inverse_sigmoid_2nd_deriv(
        A_by_ly)  # because of the nature of sigmoid''(x), only at most 2 of these will be real

    # checks if the roots (where the extrema is reached) is within [lx,ux], if not the root is replaced with NaN
    root1 = filter_range(lx, ux, root1)
    root2 = filter_range(lx, ux, root2)
    root3 = filter_range(lx, ux, root3)

    # inverse_sigmoid_2nd_deriv computes through complex64, so the roots come
    # back float32. The original check_nonlinear_boundary stacks each root with
    # ly/uy, which promotes to the common dtype before objective_fn evaluates
    # sigmoid; promote explicitly here so the objective is evaluated at the same
    # precision as the original path (bit-identical results).
    roots_x = torch.stack((root1, root2, root3), dim=1).to(ly.dtype)  # [n, 3]
    roots_y = ly.unsqueeze(1)  # [n, 1], broadcasts against roots_x

    maxs = _max_objective_over_x_candidates(A, B, C, roots_x, roots_y)  # [n]

    # NEED TO DO ALL THIS AGAIN FOR A_by_uy
    root1_uy, root2_uy, root3_uy = inverse_sigmoid_2nd_deriv(A_by_uy)

    root1_uy = filter_range(lx, ux, root1_uy)
    root2_uy = filter_range(lx, ux, root2_uy)
    root3_uy = filter_range(lx, ux, root3_uy)

    # same dtype promotion as the roots_x branch above
    roots_x_uy = torch.stack((root1_uy, root2_uy, root3_uy), dim=1).to(uy.dtype)  # [n, 3]
    roots_y_uy = uy.unsqueeze(1)  # [n, 1]

    maxs_uy = _max_objective_over_x_candidates(A, B, C, roots_x_uy, roots_y_uy)  # [n]

    return maxs, maxs_uy


def check_nonlinear_boundary_softplus(lx, ux, ly, uy, ABCs):
    A_by_ly = torch.tensor([ABCs[i][0] / ly[i] for i in range(
        len(ABCs))])
    A_by_uy = torch.tensor([ABCs[i][0] / uy[i] for i in range(len(ABCs))])
    A_by_ly = (inf_to_nan(A_by_ly))  # Checks the case ly=0 (meaning A/ly is undefined)
    A_by_uy = (inf_to_nan(A_by_uy))

    l = 0
    u = 0.25

    A_by_ly = filter_range(l, u, A_by_ly)  # checks the case A/ly is NOT in [-.1,.1] (not in image of sigmoid''(x))
    A_by_uy = filter_range(l, u, A_by_uy)

    # NEED TO DO ALL THIS AGAIN FOR A_by_uy
    root1, root2 = inverse_softplus_2nd_deriv(
        A_by_ly)  # because of the nature of sigmoid''(x), only at most 2 of these will be real

    # checks if the roots (where the extrema is reached) is within [lx,ux], if not the root is replaced with NaN
    root1 = filter_range(lx, ux, root1)
    root2 = filter_range(lx, ux, root2)

    # need to evaluate the original objective at these roots
    root1_paired = torch.stack((root1, ly), dim=1)
    root2_paired = torch.stack((root2, ly), dim=1)

    roots_paired_stacked = torch.stack((root1_paired, root2_paired), dim=1)

    maxs = []
    for i in range(lx.size()[0]):

        evaluation = objective_fn_softplus(ABCs[i][0], ABCs[i][1], ABCs[i][2], roots_paired_stacked[i])
        evaluation = torch.nan_to_num(evaluation, -float("Inf"))
        maxs.append((torch.max(evaluation)))

    # NEED TO DO ALL THIS AGAIN FOR A_by_uy
    root1_uy, root2_uy = inverse_softplus_2nd_deriv(
        A_by_uy)  # because of the nature of sigmoid''(x), only at most 2 of these will be real

    # checks if the roots (where the extrema is reached) is within [lx,ux], if not the root is replaced with NaN
    root1_uy = filter_range(lx, ux, root1_uy)
    root2_uy = filter_range(lx, ux, root2_uy)

    # need to evaluate the original objective at these roots
    root1_uy_paired = torch.stack((root1_uy, uy), dim=1)
    root2_uy_paired = torch.stack((root2_uy, uy), dim=1)

    roots_uy_paired_stacked = torch.stack((root1_uy_paired, root2_uy_paired), dim=1)

    maxs_uy = []
    for i in range(lx.size()[0]):

        evaluation_uy = objective_fn_softplus(ABCs[i][0], ABCs[i][1], ABCs[i][2], roots_uy_paired_stacked[i])
        evaluation_uy = torch.nan_to_num(evaluation_uy, -float("Inf"))
        maxs_uy.append((torch.max(evaluation_uy)))

    return maxs, maxs_uy


"""
def check_corners_negated(ABCs,lx,ux,ly,uy):

	maxs = []
	for i in range(lx.size()[0]):
		corner1 = torch.tensor([lx[i],ly[i]])

		corner2 = torch.tensor([lx[i],uy[i]])

		corner3 = torch.tensor([ux[i],ly[i]])

		corner4 = torch.tensor([ux[i],uy[i]])

		all_corners = torch.stack((corner1,corner2,corner3,corner4))

		evaluation = (-objective_fn(ABCs[i][0],ABCs[i][1],ABCs[i][2],all_corners))

		maxs.append(torch.max(evaluation))


	return maxs
"""


def compute_max_error(lx, ux, ly, uy, ABCs, ABCs_tensor=None):
    """Returns a [n] tensor when both checks are vectorized, else a list of n
    0-dim tensors (the original shape). The caller normalizes with
    lst_as_tensor, which now passes an already-stacked tensor through."""
    fully_vectorized = ABCs_tensor is not None and USE_VECTORIZED_BOUNDARY
    if fully_vectorized:
        # Both checks already produce [n] tensors. The previous version unbound
        # them into Python lists, recombined element-by-element in a loop, and
        # let lst_as_tensor stack the result straight back into a [n] tensor --
        # ~20s of pure round-trip on the `big` profile. Elementwise maximum is
        # exact, so keeping tensors throughout is bit-identical, not merely close.
        errors = check_corners_tensor(ABCs_tensor, lx, ux, ly, uy)
        errors2_ly, errors2_uy = check_nonlinear_boundary_tensor(lx, ux, ly, uy, ABCs_tensor)
        return torch.maximum(torch.maximum(errors, errors2_ly), errors2_uy)

    # checks along the linear function boundaries (by just checking the corners)
    if ABCs_tensor is not None:
        # Vectorized corners but original boundary check ("corners only"
        # variant): the boundary path returns lists, so match its shape.
        errors = list(check_corners_tensor(ABCs_tensor, lx, ux, ly, uy).unbind(0))
    else:
        errors = check_corners(ABCs, lx, ux, ly, uy)  # This seems to be correct

    # still need to check along the nonlinear boudnary, e.g.
    # f'(x)*l_y...
    # f'(x)*u_y...
    errors2_ly, errors2_uy = check_nonlinear_boundary(lx, ux, ly, uy, ABCs)

    max_errors = []
    assert (len(errors) == len(errors2_ly) and len(errors) == len(errors2_uy))
    for i in range(len(errors)):
        max_errors.append(torch.max(torch.max(errors[i], errors2_ly[i]), errors2_uy[i]))

    return max_errors


def compute_max_error_softplus(lx, ux, ly, uy, ABCs):
    # checks along the linear function boundaries (by just checking the corners)
    errors = check_corners_softplus(ABCs, lx, ux, ly, uy)  # This seems to be correct

    # still need to check along the nonlinear boudnary, e.g.
    # f'(x)*l_y...
    # f'(x)*u_y...
    errors2_ly, errors2_uy = check_nonlinear_boundary_softplus(lx, ux, ly, uy, ABCs)
    max_errors = []
    assert (len(errors) == len(errors2_ly) and len(errors) == len(errors2_uy))
    for i in range(len(errors)):
        max_errors.append(torch.max(torch.max(errors[i], errors2_ly[i]), errors2_uy[i]))

    return max_errors


def sigmoid_prime_product_tensor(x, y):
    lx = x.get_lb()
    ux = x.get_ub()
    ly = y.get_lb()
    uy = y.get_ub()

    # EXPERIMENTAL batched grid (PASADO_BATCHED_GRID=1): build [n,25,2] directly
    # instead of a Python list of n [25,2] tensors. Bit-identical either way, so
    # the two selectors compose freely; whichever consumer needs the other shape
    # converts once here rather than inside the hot loop.
    if USE_BATCHED_GRID:
        xys_batch = get_linspace_batched(lx, ux, ly, uy)
        xys = None if USE_BATCHED_LSTSQ else list(xys_batch.unbind(0))
    else:
        xys = get_linspace(lx, ux, ly, uy)  # this is a list of tensors
        xys_batch = torch.stack(xys) if USE_BATCHED_LSTSQ else None

    if USE_BATCHED_LSTSQ:
        # EXPERIMENTAL batched regression path (PASADO_BATCHED_LSTSQ=1): solve
        # all planar regressions in one batched gelsd call - proven bit-identical
        # to the per-neuron loop (experiments/test_batched_lin_reg.py). The
        # np.random stream is consumed one scalar draw per neuron in the same
        # order as the loop path, so seeded runs match it exactly.
        x_batch = xys_batch             # [n, 25, 2], exact same grid values
        xs_grid = x_batch[..., 0]
        ys_grid = x_batch[..., 1]
        # same association as sigmoid_prime_times_y: (1-s) * (s*y)
        zs_batch = (1.0 - torch.sigmoid(xs_grid)) * (torch.sigmoid(xs_grid) * ys_grid)
        lineqs_b = lin_reg_tensor_batched(x_batch, zs_batch)  # [n, 3] intercept-first
        perturb = torch.tensor(
            [np.random.normal(0, .0001) for _ in range(lineqs_b.shape[0])],
            dtype=lineqs_b.dtype, device=lineqs_b.device)
        As = lineqs_b[:, 1] + perturb
        Bs = lineqs_b[:, 2]
        Cs = lineqs_b[:, 0]
        assert bool(((As > 0) | (As < 0)).all())
        # The tuple list is consumed by whichever check still runs the original
        # per-neuron path. Testing "not USE_VECTORIZED_PRECISE" was not enough:
        # with vectorized corners but the original boundary check
        # (PASADO_VEC_BOUNDARY=0) the boundary fallback still needs the list, and
        # passing None crashed it. That combination was never in the benchmark
        # matrix, so the latent bug went unnoticed until test_batched_grid.py
        # enumerated all selector combinations.
        ABCs = None if (USE_VECTORIZED_PRECISE and USE_VECTORIZED_BOUNDARY) else \
            [(As[i], Bs[i], Cs[i]) for i in range(As.shape[0])]
    else:
        zs = [sigmoid_prime_times_y(xy) for xy in xys]

        lineqs = [lin_reg_tensor(xys[i], zs[i]) for i in
                  range(len(zs))]  # list of tensors([A,B,C]) where Ax+By+C is the planar regression

        # this has to be 1,2,0 since the linreg function gives the constant offset (c) as the 0th element of the tuple
        ABCs = [(lineqs[i][1], lineqs[i][2], lineqs[i][0]) for i in range(len(lineqs))]
        ABCs = [(x[0] + np.random.normal(0, .0001), x[1], x[2]) for x in ABCs]
        assert (all([(x[0] > 0 or x[0] < 0) for x in ABCs]))
        As = torch.tensor([x[0] for x in ABCs])
        Bs = torch.tensor([x[1] for x in ABCs])
        Cs = torch.tensor([x[2] for x in ABCs])

    # EXPERIMENTAL: tensor form of ABCs for the vectorized corner check. When the
    # selector is off we pass ABCs_tensor=None so compute_max_error falls back to
    # the original per-neuron check_corners / check_nonlinear_boundary path.
    ABCs_tensor = torch.stack((As, Bs, Cs), dim=1) if USE_VECTORIZED_PRECISE else None

    # Everything up to this point works (the linear regression and the lower and upper bounds of the input Zonotopes)
    errors = compute_max_error(lx, ux, ly, uy, ABCs, ABCs_tensor)
    errors = lst_as_tensor(errors)
    errors = traceify(errors)

    x_centers = x.centers
    x_generators = x.generators

    y_centers = y.centers
    y_generators = y.generators

    new_centers = As * x_centers + Bs * y_centers + Cs
    new_generators = As * x_generators + Bs * y_generators

    # add in the fresh error symbols obtained by solving the optimization problem
    new_generators = torch.cat((new_generators, errors), 0)

    return Zonotope(new_centers, new_generators)


def softplus_prime_product_tensor(x, y):
    lx = x.get_lb()
    ux = x.get_ub()
    ly = y.get_lb()
    uy = y.get_ub()

    xys = get_linspace(lx, ux, ly, uy)  # this is a list of tensors

    zs = [softplus_prime_times_y(xy) for xy in xys]

    lineqs = [lin_reg_tensor(xys[i], zs[i]) for i in
              range(len(zs))]  # list of tensors([A,B,C]) where Ax+By+C is the planar regression

    # this has to be 1,2,0 since the linreg function gives the constant offset (c) as the 0th element of the tuple
    ABCs = [(lineqs[i][1], lineqs[i][2], lineqs[i][0]) for i in range(len(lineqs))]
    assert (all([(x[0] > 0 or x[0] < 0) for x in ABCs]))
    As = torch.tensor([x[0] for x in ABCs])
    Bs = torch.tensor([x[1] for x in ABCs])
    Cs = torch.tensor([x[2] for x in ABCs])

    # Everything up to this point works (the linear regression and the lower and upper bounds of the input Zonotopes)
    errors = compute_max_error_softplus(lx, ux, ly, uy, ABCs)
    errors = lst_as_tensor(errors)
    errors = traceify(errors)

    x_centers = x.centers
    x_generators = x.generators

    y_centers = y.centers
    y_generators = y.generators

    new_centers = As * x_centers + Bs * y_centers + Cs
    new_generators = As * x_generators + Bs * y_generators

    # add in the fresh error symbols obtained by solving the optimization problem
    new_generators = torch.cat((new_generators, errors), 0)

    return Zonotope(new_centers, new_generators)


# this is where the above transformer is actually integrated into the entire expression for the Dual Zonotope
def PreciseSigmoidDualZonotope(DualZono):
    old_num_syms = DualZono.real.get_num_noise_symbs()
    old_num_syms_dual = DualZono.dual.get_num_noise_symbs()
    assert (old_num_syms == old_num_syms_dual)

    sigmoid_real = SigmoidZonotope(DualZono.real)

    real_num_noise_terms = sigmoid_real.get_num_noise_symbs()

    # DualZono.real.expand(real_num_noise_terms-old_num_syms) #dont need this the SigmoidZonotope already executes this exact command inside that function
    if 0 < real_num_noise_terms - old_num_syms_dual:
        DualZono.dual.expand(real_num_noise_terms - old_num_syms_dual)  # DO need this

    dual = sigmoid_prime_product_tensor(DualZono.real, DualZono.dual)  # This has NO side effects on the other zonotopes

    dual_num_noise_terms = dual.get_num_noise_symbs()
    sigmoid_real.expand(dual_num_noise_terms - real_num_noise_terms)


    return DualZonotope(sigmoid_real.centers, sigmoid_real.generators, dual.centers, dual.generators)


# this is where the above transformer is actually integrated into the entire expression for the Dual Zonotope
def PreciseSoftplusDualZonotope(DualZono):
    old_num_syms = DualZono.real.get_num_noise_symbs()
    old_num_syms_dual = DualZono.dual.get_num_noise_symbs()
    assert (old_num_syms == old_num_syms_dual)

    # sigmoid_real = SigmoidZonotope(DualZono.real)
    smoothrelu_real = SoftPlusZonoChebyshev(DualZono.real)

    real_num_noise_terms = smoothrelu_real.get_num_noise_symbs()

    # DualZono.real.expand(real_num_noise_terms-old_num_syms) #dont need this the SigmoidZonotope already executes this exact command inside that function
    DualZono.dual.expand(real_num_noise_terms - old_num_syms_dual)  # DO need this

    dual = softplus_prime_product_tensor(DualZono.real,
                                         DualZono.dual)  # This has NO side effects on the other zonotopes

    dual_num_noise_terms = dual.get_num_noise_symbs()
    smoothrelu_real.expand(dual_num_noise_terms - real_num_noise_terms)


    return DualZonotope(smoothrelu_real.centers, smoothrelu_real.generators, dual.centers, dual.generators)
