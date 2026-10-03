import math
import os
import torch
from sklearn import linear_model
from SimpleZono import *


# Switches for the optimized code paths. They are read when the code runs, so a
# script can flip them between runs. The original code stays as the fallback.
# Set the environment variable to 0 or 1 to change a default.

# Vectorized corner and nonlinear boundary checks in compute_max_error.
USE_VECTORIZED_PRECISE = os.environ.get("PASADO_VECTORIZED_PRECISE", "1") != "0"

# Only the boundary check, so "vectorized corners only" can be run on its own.
# Needs USE_VECTORIZED_PRECISE.
USE_VECTORIZED_BOUNDARY = os.environ.get("PASADO_VEC_BOUNDARY", "1") != "0"

# One batched lstsq per layer instead of one per neuron. Same results as the loop.
USE_BATCHED_LSTSQ = os.environ.get("PASADO_BATCHED_LSTSQ", "0") != "0"

# Build all sampling grids at once. Same results as the loop.
USE_BATCHED_GRID = os.environ.get("PASADO_BATCHED_GRID", "0") != "0"

# Solve the cubic with Viete's real formula instead of complex64 Cardano. This
# changes the numbers slightly (see inverse_poly_real_tensor), so it stays off.
USE_REAL_CUBIC = os.environ.get("PASADO_REAL_CUBIC", "0") != "0"


# lx, ly, ux, uy are TENSORS/VECTORS who have as many columns as neurons/variables in a layer
def get_linspace(lx, ux, ly, uy, STEPS=5):
    xs = [torch.linspace(lx[i].item(), ux[i].item(), steps=STEPS) for i in range(lx.shape[0])]
    ys = [torch.linspace(ly[i].item(), uy[i].item(), steps=STEPS) for i in range(ly.shape[0])]

    zs = [torch.cartesian_prod(xs[i], ys[i]) for i in range(len(xs))]
    return zs


def _linspace_rows(lo, hi, STEPS):
    """torch.linspace for each row, bit for bit.

    torch.linspace fills the first half forward from the start and the second half
    backward from the end. Copying only the forward half is off by 1 ULP in about
    a quarter of cases, so both halves are reproduced.
    """
    t = torch.arange(STEPS, dtype=lo.dtype, device=lo.device)
    step = (hi - lo) / (STEPS - 1)
    out = lo.unsqueeze(1) + t * step.unsqueeze(1)
    # A slice, not a boolean mask: a mask goes through nonzero, whose shape depends
    # on the data and breaks the Dynamo graph.
    half = STEPS // 2
    out[:, half:] = hi.unsqueeze(1) - (STEPS - 1 - t[half:]) * step.unsqueeze(1)
    return out


def get_linspace_batched(lx, ux, ly, uy, STEPS=5):
    """[n, STEPS*STEPS, 2], identical to torch.stack(get_linspace(...)).

    One batch of tensor operations instead of 2n linspace calls, n cartesian_prod
    calls and 4n .item() calls. Each .item() also forces a device sync, which
    would stall a GPU run.
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


def lin_reg_tensor_batched(x_batch, zs_batch):
    """All per-neuron planar regressions of a layer in one lstsq call.

    x_batch is [n, m, 2] grid points, zs_batch is [n, m] (or [n, m, 1]) targets.
    Returns [n, 3] with the intercept first (C, A, B), like lin_reg_tensor. The
    result is bit-identical to calling lin_reg_tensor once per neuron.
    """
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


# Tolerance for the real cubic solver. It equals float32 epsilon because the
# complex64 path decides the same question in float32.
_CUBIC_BOUNDARY_TOL = 1e-7


def inverse_poly_real_tensor(y, tol=_CUBIC_BOUNDARY_TOL):
    """The roots of inverse_poly_tensor (s = u + 1/2), in real arithmetic (Viete).

    The cubic u^3 - u/4 - y/2 = 0 has three real roots exactly when
    |y| < 1/(6*sqrt(3)), the maximum of |sigma''|. Viete's form gives them directly:

        u_k = cos(acos(6*sqrt(3)*y)/3 - 2*pi*k/3) / sqrt(3),  k = 0, 1, 2

    The acos argument is in [-1, 1] exactly when the roots are real.

    The complex64 path is only accurate to single precision (median residual 3.8e-06
    against 1.2e-16 here), and inv_sigmoid can amplify that near 0 and 1. Both paths
    agree on how many roots are real, except within about 1e-6 of the boundary.
    There `tol` clamps the argument, so a near-double root is kept instead of
    dropped: an extra candidate only makes the error bound larger, while a dropped
    one could make it unsound.
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


# Vectorized check_corners: ABCs is an [n, 3] tensor (columns A, B, C) instead of a
# list of tuples, and all neurons are evaluated at once.
def check_corners_tensor(ABCs_tensor, lx, ux, ly, uy):
    A = ABCs_tensor[:, 0]
    B = ABCs_tensor[:, 1]
    C = ABCs_tensor[:, 2]

    # corners in the same order as check_corners: (lx,ly), (lx,uy), (ux,ly), (ux,uy)
    xs = torch.stack((lx, lx, ux, ux), dim=1)
    ys = torch.stack((ly, uy, ly, uy), dim=1)

    sig = torch.sigmoid(xs)
    # same operation order as sigmoid_prime_times_y, to stay bit-identical
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
    # xs is [n, k] candidate roots, ys is [n, 1]. NaN candidates (filtered-out roots)
    # become -inf so they never win the max, as in the per-neuron loop.
    sig = torch.sigmoid(xs)
    # same operation order as sigmoid_prime_times_y, to stay bit-identical
    sigmoid_prime_y = (1.0 - sig) * (sig * ys)
    planar = A.unsqueeze(1) * xs + B.unsqueeze(1) * ys + C.unsqueeze(1)
    evaluation = sigmoid_prime_y - planar
    evaluation = torch.nan_to_num(evaluation, -float("Inf"))
    return torch.max(evaluation, dim=1).values


# Vectorized check_nonlinear_boundary: ABCs is an [n, 3] tensor, and the per-neuron
# loops are replaced by batched operations.
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

    # The roots are float32 (complex64 solve). The original stacks them with ly/uy,
    # which promotes them to the common dtype before sigmoid, so promote explicitly
    # to get the same result.
    roots_x = torch.stack((root1, root2, root3), dim=1).to(ly.dtype)  # [n, 3]
    roots_y = ly.unsqueeze(1)  # [n, 1], broadcasts against roots_x

    maxs = _max_objective_over_x_candidates(A, B, C, roots_x, roots_y)  # [n]

    # same for the uy branch
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
    """[n] tensor when both checks are vectorized, otherwise a list of 0-dim tensors."""
    fully_vectorized = ABCs_tensor is not None and USE_VECTORIZED_BOUNDARY
    if fully_vectorized:
        # Keep everything as tensors. An elementwise maximum is exact, so this is
        # bit-identical to unbinding into lists and stacking again.
        errors = check_corners_tensor(ABCs_tensor, lx, ux, ly, uy)
        errors2_ly, errors2_uy = check_nonlinear_boundary_tensor(lx, ux, ly, uy, ABCs_tensor)
        return torch.maximum(torch.maximum(errors, errors2_ly), errors2_uy)

    # checks along the linear function boundaries (by just checking the corners)
    if ABCs_tensor is not None:
        # "Corners only" variant: the original boundary check returns lists.
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

    # Batched grid: build [n, 25, 2] directly. Whichever path needs the other shape
    # converts once here.
    if USE_BATCHED_GRID:
        xys_batch = get_linspace_batched(lx, ux, ly, uy)
        xys = None if USE_BATCHED_LSTSQ else list(xys_batch.unbind(0))
    else:
        xys = get_linspace(lx, ux, ly, uy)  # this is a list of tensors
        xys_batch = torch.stack(xys) if USE_BATCHED_LSTSQ else None

    if USE_BATCHED_LSTSQ:
        # Batched regression: one lstsq call for all neurons. np.random is drawn
        # once per neuron in the same order as the loop path, so seeded runs match.
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
        # The tuple list is needed by whichever check still runs the original
        # per-neuron path (for example PASADO_VEC_BOUNDARY=0 with vectorized corners).
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

    # [n, 3] tensor form of ABCs for the vectorized checks; None selects the
    # original per-neuron path.
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
