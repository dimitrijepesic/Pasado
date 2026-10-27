"""Tests for forward_mode_tensorized_src/planar_fit.py.

1. Same bits as candidate A2 of lstsq_candidates.py, the solver the pre-registered
   comparison chose, on every box family of that comparison (other seeds), on the
   self-test family offset_narrow, on non-finite boxes (NaN in the same places) and
   on boxes recorded from the batched forward pass of 3layer.
2. Accurate box by box: on every box whose rank the fit keeps exact, the distance to
   the exact minimum-norm solution is at most 32 * eps * (1 + max|x| + max|y|), a few
   ulps of the size of the data (see experiments/reproduce_target_centring.py).
2b. The contract checks C1-C7 of lstsq_contract.py.
3. Close to gelsd (lin_reg_tensor_batched) on well-conditioned boxes.
4. With the inputs on another device: result on that device, in the input dtype, and
   close to the CPU result.

With --compiled the same checks run on planar_fit.compiled_fit_planes(), which is not
bit-identical to fit_planes, so check 1 becomes, box by box: non-finite results in the
same places, and within 64 * eps * (1 + max|x| + max|y|) of fit_planes on every box
outside the band 0.9 to 1.1 times the rank cutoff (where the two may decide the rank
differently; counted and printed). It also checks that compilation did not fall back to
the eager function.

Run:
  python experiments/test_planar_fit.py
  python experiments/test_planar_fit.py --device cuda
  python experiments/test_planar_fit.py --device cuda --compiled
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

import planar_fit  # noqa: E402
import precise_transformer as pt  # noqa: E402
import compare_lstsq_solvers as h  # noqa: E402
import lstsq_candidates as lc  # noqa: E402
import lstsq_contract as ct  # noqa: E402
import lstsq_oracle as orc  # noqa: E402

SEED = 4242  # not a seed of the comparison
BOXES = 500
# gelsd is accurate to about cond(X) * eps here (widths >= 0.5, centres within 4)
GELSD_RTOL = {"float64": 1e-10, "float32": 1e-3}
DEVICE_RTOL = {"float64": 1e-12, "float32": 1e-5}
ACCURACY_ULPS = 32
COMPILED_ULPS = 64

failures = []
solve = planar_fit.fit_planes  # the function under test; main may switch to the compiled one


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def same_bits(a, b):
    """Equal, with NaN in the same places counting as equal."""
    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


def rel_diff(a, b):
    scale = b.abs().max(dim=1).values.clamp_min(torch.finfo(b.dtype).tiny)
    return ((a - b).abs().max(dim=1).values / scale).max().item()


def family_bounds(family, dtype_name, k):
    if family == "offset_narrow":
        return h.offset_narrow_bounds(dtype_name)
    if family == "nonfinite":
        return h.nonfinite_bounds()
    return h.synthetic_bounds(family, BOXES, SEED + k, dtype_name)


def close_to_eager(grid, z):
    """(ok, detail): the compiled function against fit_planes, box by box."""
    got, ref = solve(grid, z).cpu(), planar_fit.fit_planes(grid, z).cpu()
    fin_got, fin_ref = torch.isfinite(got).all(1), torch.isfinite(ref).all(1)
    if not torch.equal(fin_got, fin_ref):
        return False, f"({int((fin_got != fin_ref).sum())} boxes finite in only one of them)"
    rcond = lc.default_rcond(grid.dtype, grid.shape[1])
    r2, r3 = (r.cpu() for r in lc.singular_value_ratios(grid))
    band = ((r2 > 0.9 * rcond) & (r2 < 1.1 * rcond)) | ((r3 > 0.9 * rcond) & (r3 < 1.1 * rcond))
    scale = (1 + grid[..., 0].abs().amax(1) + grid[..., 1].abs().amax(1)).cpu()
    eps = torch.finfo(grid.dtype).eps
    rows = [i for i in range(grid.shape[0]) if fin_ref[i] and not band[i]]
    if not rows:
        return True, f"(no finite box outside the cutoff band, {int(band.sum())} in it)"
    d = (got - ref).abs().max(1).values / ref.abs().max(1).values.clamp_min(torch.finfo(grid.dtype).tiny)
    ratio = d / (eps * scale)
    worst = float(ratio[rows].max())
    over = int((ratio[rows] > COMPILED_ULPS).sum())
    return over == 0, f"(worst {worst:.2f}, {over} of {len(rows)} over, {int(band.sum())} in the cutoff band)"


def test_same_as_candidate(device, dtype_name, compiled):
    print(f"{'close to fit_planes box by box' if compiled else 'same bits as candidate A2'} "
          f"({device}, {dtype_name})")
    dtype = getattr(torch, dtype_name)
    families = h.FAMILIES[:10] + ["offset_narrow", "nonfinite"]
    for k, family in enumerate(families):
        b = [t.to(device=device, dtype=dtype) for t in family_bounds(family, dtype_name, k)]
        grid, z = h.grid_and_targets(*b)
        if compiled:
            check(family, *close_to_eager(grid, z))
        else:
            check(family, same_bits(planar_fit.fit_planes(grid, z), lc.solve_A2(grid, z)),
                  f"({grid.shape[0]} boxes)")


def test_same_on_real_boxes(device, dtype_name, compiled):
    print(f"boxes from the batched forward pass of 3layer ({device}, {dtype_name})")
    cfg = dict(networks=["3layer"], num_images=1, eps_indices=[0, 8, 15], n_splits=4, seed=7,
               per_network=2000, seed_family=SEED)
    bounds, _, _ = h.capture_real_boxes(cfg, torch.device("cpu"))
    dtype = getattr(torch, dtype_name)
    grid, z = h.grid_and_targets(*(t.to(device=device, dtype=dtype) for t in bounds))
    if compiled:
        check("real 3layer, close to fit_planes", *close_to_eager(grid, z))
    else:
        check("real 3layer, same bits as candidate A2",
              same_bits(planar_fit.fit_planes(grid, z), lc.solve_A2(grid, z)),
              f"({grid.shape[0]} boxes)")


def per_box_accuracy(grid, z, coef):
    """(worst error / (eps (1 + max|x| + max|y|)), boxes over ACCURACY_ULPS, boxes checked)
    over the boxes whose rank the fit keeps exact."""
    keep = (lc.chosen_rank("A2", grid, z) == lc.exact_rank(grid).cpu()).tolist()
    keep = [k and bool(torch.isfinite(grid[i]).all()) for i, k in enumerate(keep)]
    if not any(keep):
        return 0.0, 0, 0
    s = math.isqrt(grid.shape[1])
    xs, ys, zl = grid[:, ::s, 0].tolist(), grid[:, :s, 1].tolist(), z.tolist()
    scale = (1 + grid[..., 0].abs().amax(1) + grid[..., 1].abs().amax(1)).tolist()
    eps = torch.finfo(grid.dtype).eps
    coef = coef.cpu()
    ratios = [orc.relative_error(coef[i].tolist(), orc.exact_min_norm(xs[i], ys[i], zl[i]))
              / (eps * scale[i]) for i in range(grid.shape[0]) if keep[i]]
    return max(ratios), sum(r > ACCURACY_ULPS for r in ratios), len(ratios)


def test_accurate_per_box(device, dtype_name):
    print(f"accurate box by box, at most {ACCURACY_ULPS} eps (1 + max|x| + max|y|) "
          f"({device}, {dtype_name})")
    dtype = getattr(torch, dtype_name)
    for k, family in enumerate(h.FAMILIES[:10] + ["offset_narrow"]):
        b = [t.to(device=device, dtype=dtype) for t in family_bounds(family, dtype_name, k)]
        grid, z = h.grid_and_targets(*b)
        worst, over, n = per_box_accuracy(grid, z, solve(grid, z))
        check(family, over == 0, f"(worst {worst:.2f}, {over} of {n} boxes over)")


def test_contract(device, dtype_name):
    print(f"contract checks C1-C7 ({device}, {dtype_name})")
    rows = ct.contract_checks("planar_fit", solve,
                              lambda grid, z: lc.chosen_rank("A2", grid, z), True, dtype_name,
                              device=device)
    for r in rows:
        if r["n_failed"]:
            check(f"{r['check']} {r['family']}", False, f"{r['n_failed']} of {r['n_checked']}")
    check("all checks", not any(r["n_failed"] for r in rows), f"({len(rows)} checks)")


def test_close_to_gelsd(device, dtype_name):
    print(f"close to gelsd on well-conditioned boxes ({device}, {dtype_name})")
    dtype = getattr(torch, dtype_name)
    for k, family in enumerate(ct.G0_FAMILIES):
        b = [t.to(device=device, dtype=dtype) for t in ct.g0_bounds(family, BOXES, SEED + 50 + k)]
        grid, z = h.grid_and_targets(*b)
        d = rel_diff(solve(grid, z), pt.lin_reg_tensor_batched(grid, z))
        check(family, d <= GELSD_RTOL[dtype_name], f"(max rel diff {d:.2e})")


def test_device(device, dtype_name):
    print(f"inputs on {device} ({dtype_name})")
    dtype = getattr(torch, dtype_name)
    b = [t.to(dtype) for t in ct.g0_bounds("well_conditioned", BOXES, SEED + 99)]
    grid, z = h.grid_and_targets(*b)
    ref = planar_fit.fit_planes(grid, z)
    got = solve(grid.to(device), z.to(device))
    check("device and dtype kept", got.device.type == device.type and got.dtype == dtype)
    d = rel_diff(got.cpu(), ref)
    check("close to the CPU result", d <= DEVICE_RTOL[dtype_name], f"(max rel diff {d:.2e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compiled", action="store_true",
                    help="test planar_fit.compiled_fit_planes() instead of fit_planes")
    args = ap.parse_args()
    device = torch.device(args.device)
    h.set_production_flags()
    global solve
    if args.compiled:
        solve = planar_fit.compiled_fit_planes()
    print("function under test:", "planar_fit.compiled_fit_planes()" if args.compiled
          else "planar_fit.fit_planes")

    for dtype_name in ("float64", "float32"):
        if device.type == "mps" and dtype_name == "float64":
            print("SKIP float64 on mps: the device has no float64")
            continue
        test_same_as_candidate(device, dtype_name, args.compiled)
        test_same_on_real_boxes(device, dtype_name, args.compiled)
        test_accurate_per_box(device, dtype_name)
        test_contract(device, dtype_name)
        test_close_to_gelsd(device, dtype_name)
        if device.type != "cpu":
            test_device(device, dtype_name)
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
