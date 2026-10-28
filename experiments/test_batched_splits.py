"""Tests for forward_mode_tensorized_src/batched_splits.py against the sequential code.

The reference is the original loop over sub-intervals (DualIntervalTensor ->
HyperDualIntervalToDualZonotope -> PreciseSigmoidDualZonotope), run once per split
with np.random reseeded each time. The batched pass must reproduce every per-split
bound, not just their maximum.

With --closed-form the batched pass fits the planes with planar_fit.py
(batched_splits.USE_CLOSED_FORM_FIT) while the sequential reference keeps gelsd, so
the same checks compare the closed form against the original code.

Run (CPU, float64):
  python experiments/test_batched_splits.py
  python experiments/test_batched_splits.py --device cuda
  python experiments/test_batched_splits.py --closed-form
  python experiments/test_batched_splits.py --compiled
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))
sys.path.insert(0, HERE)

import precise_transformer as pt
import batched_splits as bs
import planar_fit
from Duals import DualIntervalTensor
from SimpleZono import HyperDualIntervalToDualZonotope
from measure_split_sweep import build_forward_precise, load_network_and_images
from check_sweep_soundness import sampled_max_derivative

# float64 reordering noise is ~1e-16 per operation; 1e-10 leaves room for the 4 to 5
# layers of accumulation but is far below any difference between two real bounds.
RTOL = 1e-10
EPS_SCHEDULE = [10 ** (-k / 4) * 2 for k in range(2, 18)]

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def sequential_split_bounds(forward_fn, img, epsilon, n_splits, seed, device, dtype):
    """Per-split bounds from the original loop."""
    delta = epsilon / n_splits
    one = torch.tensor([1.0], dtype=dtype, device=device)
    out = []
    with torch.no_grad():
        for i in range(n_splits):
            l_val = i * delta
            interval = DualIntervalTensor(
                real_l=torch.tensor([l_val], dtype=dtype, device=device),
                real_u=torch.tensor([l_val + delta], dtype=dtype, device=device))
            interval.e1_l[0] = 1
            interval.e1_u[0] = 1
            zono_eps = HyperDualIntervalToDualZonotope(interval)
            hazed = (one - zono_eps) * img + zono_eps
            np.random.seed(seed)
            output = forward_fn(hazed)
            out.append(torch.max(torch.maximum(output.dual.get_lb().abs(),
                                               output.dual.get_ub().abs())).item())
    return torch.tensor(out, dtype=torch.float64)


def rel_diff(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).abs() / b.abs().clamp_min(1e-300)).max().item()


def test_matches_sequential(device, dtype, networks, split_values, eps_indices, num_images):
    print("batched == sequential, per split")
    for network in networks:
        net, images = load_network_and_images(network, device, dtype, num_images)
        forward_fn = build_forward_precise(pt, net, network)
        layers = bs.extract_affine_layers(net, network)
        worst = 0.0
        for image_id, img in images:
            for eps_idx in eps_indices:
                for n_splits in split_values:
                    eps = EPS_SCHEDULE[eps_idx]
                    ref = sequential_split_bounds(forward_fn, img, eps, n_splits, 7, device, dtype)
                    got = bs.split_bounds(layers, img, eps, n_splits, seed=7)
                    worst = max(worst, rel_diff(got, ref))
        check(f"{network}: all per-split bounds", worst <= RTOL,
              f"(max rel diff {worst:.2e}, splits {split_values}, eps {eps_indices}, "
              f"{len(images)} images)")


def test_chunking(device, dtype, network="3layer"):
    print("chunking does not change the result")
    net, images = load_network_and_images(network, device, dtype, 1)
    layers = bs.extract_affine_layers(net, network)
    img = images[0][1]
    eps = EPS_SCHEDULE[4]
    full = bs.split_bounds(layers, img, eps, 8, seed=3)
    for chunk in (1, 3, 8, 100):
        part = bs.split_bounds(layers, img, eps, 8, seed=3, chunk=chunk)
        check(f"chunk={chunk}", rel_diff(part, full) <= RTOL, f"({rel_diff(part, full):.2e})")


def test_split_one_is_plain_bound(device, dtype, network="3layer"):
    print("n_splits=1 equals one sequential forward pass")
    net, images = load_network_and_images(network, device, dtype, 1)
    forward_fn = build_forward_precise(pt, net, network)
    layers = bs.extract_affine_layers(net, network)
    img = images[0][1]
    eps = EPS_SCHEDULE[0]
    ref = sequential_split_bounds(forward_fn, img, eps, 1, 5, device, dtype)
    got = bs.split_bounds(layers, img, eps, 1, seed=5)
    check("single split", rel_diff(got, ref) <= RTOL)


def test_unseeded_is_sound(device, dtype, network="3layer"):
    """Without a seed there is no reference to match, so check soundness instead:
    the bound must not fall below a sampled maximum of the true derivative."""
    print("unseeded noise: bound stays above the sampled autograd maximum")
    net, images = load_network_and_images(network, device, dtype, 2)
    layers = bs.extract_affine_layers(net, network)
    # a copy: Module.to moves parameters in place, which would pull the bias
    # tensors in `layers` off the device while their weights stay on it
    net64 = copy.deepcopy(net).to("cpu", torch.float64)
    for image_id, img in images:
        for eps_idx in (0, 8):
            eps = EPS_SCHEDULE[eps_idx]
            sampled = sampled_max_derivative(net64, img.cpu().double(), eps, 1001)
            for n_splits in (1, 16):
                lc = bs.lipschitz_bound(layers, img, eps, n_splits, seed=None)
                check(f"image {image_id} eps_idx {eps_idx} n={n_splits}", lc >= sampled,
                      f"(lc={lc:.4f}, sampled={sampled:.4f})")


def test_splits_are_independent(device, dtype, network="3layer"):
    """Bound for a split must not depend on which other splits share its batch."""
    print("per-split bound is independent of the other splits in the batch")
    net, images = load_network_and_images(network, device, dtype, 1)
    layers = bs.extract_affine_layers(net, network)
    img = images[0][1]
    eps = EPS_SCHEDULE[2]
    all16 = bs.split_bounds(layers, img, eps, 16, seed=1)
    # the same sub-intervals, evaluated as 4 batches of 4
    parts = bs.split_bounds(layers, img, eps, 16, seed=1, chunk=4)
    check("16 splits vs 4 chunks of 4", rel_diff(parts, all16) <= RTOL)


def _numpy_state():
    """The whole state of np.random: name, key array, position, and the cached Gaussian."""
    state = np.random.get_state()
    return state[0], state[1].copy(), state[2], state[3], state[4]


def _same_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def test_noise_protocol(device, dtype, network="3layer"):
    """The noise on A is drawn outside precise_sigmoid, one draw per sigmoid layer, in
    the order of the sequential code: the NumPy generator ends in exactly the state of
    those draws, and the same seed gives the same bits."""
    print("noise on A: NumPy draws and seeds")
    net, images = load_network_and_images(network, device, dtype, 1)
    layers = bs.extract_affine_layers(net, network)
    widths = [w.shape[1] for w, _ in layers[:-1]]
    img = images[0][1]
    eps = EPS_SCHEDULE[3]
    S = 4

    np.random.seed(3)
    for n in widths:
        np.random.normal(0, .0001, size=n)
    want = _numpy_state()
    for chunk in (None, 2):
        np.random.seed(12345)  # the pass reseeds, so this must not matter
        first = bs.split_bounds(layers, img, eps, S, seed=3, chunk=chunk)
        check(f"seeded, chunk={chunk}: generator state after the pass", _same_state(_numpy_state(), want))
        second = bs.split_bounds(layers, img, eps, S, seed=3, chunk=chunk)
        check(f"seeded, chunk={chunk}: same seed, same bits", torch.equal(first, second))

    np.random.seed(11)
    for n in widths:
        np.random.normal(0, .0001, size=(S, n))
    want = _numpy_state()
    np.random.seed(11)
    bs.split_bounds(layers, img, eps, S, seed=None)
    check("unseeded: generator state after the pass", _same_state(_numpy_state(), want))


def test_nonzero_a_is_checked(device, dtype, network="3layer"):
    """A perturbed A equal to 0 must stop the pass, as the assert in the sequential code."""
    print("a perturbed A of 0 stops the pass")
    net, images = load_network_and_images(network, device, dtype, 1)
    layers = bs.extract_affine_layers(net, network)
    saved = bs.plane_fit_for, bs._perturbation
    bs.plane_fit_for = lambda grid: (lambda g, z: torch.zeros(g.shape[0], 3, dtype=g.dtype,
                                                              device=g.device))
    bs._perturbation = lambda S, N, seed, dtype_, device_: torch.zeros(S, N, dtype=dtype_,
                                                                       device=device_)
    try:
        bs.split_bounds(layers, images[0][1], EPS_SCHEDULE[3], 2, seed=1)
        check("AssertionError", False, "(the pass went through)")
    except AssertionError:
        check("AssertionError", True)
    finally:
        bs.plane_fit_for, bs._perturbation = saved


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--quick", action="store_true", help="3layer only, fewer cases")
    ap.add_argument("--closed-form", action="store_true",
                    help="batched pass with planar_fit instead of gelsd")
    ap.add_argument("--compiled", action="store_true",
                    help="as --closed-form, with planar_fit compiled by TorchInductor")
    args = ap.parse_args()
    args.closed_form = args.closed_form or args.compiled

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    pt.USE_VECTORIZED_PRECISE = True
    pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = True
    pt.USE_BATCHED_GRID = True
    pt.USE_REAL_CUBIC = True
    bs.USE_CLOSED_FORM_FIT = args.closed_form
    bs.USE_COMPILED_FIT = args.compiled
    print("batched plane fit:", "planar_fit compiled" if args.compiled
          else "planar_fit (closed form)" if args.closed_form else "gelsd")
    # count the calls of the selected closed form, so a switch that does not switch
    # is caught; the compiled one is counted outside the compiled function
    closed_form_calls = [0]
    if args.compiled:
        make_compiled = planar_fit.compiled_fit_planes

        def counted_compiled(mode=None):
            fn = make_compiled(mode)

            def run(grid, z, rcond=None):
                closed_form_calls[0] += 1
                return fn(grid, z, rcond)
            return run

        planar_fit.compiled_fit_planes = counted_compiled
    else:
        fit_planes = planar_fit.fit_planes

        def counted_fit_planes(grid, z, rcond=None):
            closed_form_calls[0] += 1
            return fit_planes(grid, z, rcond)

        planar_fit.fit_planes = counted_fit_planes

    if args.quick:
        test_matches_sequential(device, dtype, ["3layer"], [1, 4], [0, 8], 1)
    else:
        test_matches_sequential(device, dtype, ["3layer", "4layer", "5layer"],
                                [1, 2, 3, 8], [0, 8, 15], 3)
        test_matches_sequential(device, dtype, ["big"], [1, 4], [0, 8], 1)
    test_chunking(device, dtype)
    test_split_one_is_plain_bound(device, dtype)
    test_splits_are_independent(device, dtype)
    test_unseeded_is_sound(device, dtype)
    test_noise_protocol(device, dtype)
    test_nonzero_a_is_checked(device, dtype)
    check("the selected plane fit was used", (closed_form_calls[0] > 0) == args.closed_form,
          f"({closed_form_calls[0]} closed-form calls)")
    if args.compiled:
        graphs = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)
        check("compiled, not the eager fallback", not planar_fit.compile_fallback and graphs > 0,
              f"({planar_fit.compile_fallback or f'{graphs} graph(s) compiled'})")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
