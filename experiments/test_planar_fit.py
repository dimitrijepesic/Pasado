"""Tests for forward_mode_tensorized_src/planar_fit.py.

1. Same bits as candidate A2 of lstsq_candidates.py, the solver the pre-registered
   comparison chose, on every box family of that comparison (other seeds), on the
   self-test family offset_narrow, on non-finite boxes (NaN in the same places) and
   on boxes recorded from the batched forward pass of 3layer.
2. The contract checks C1-C7 of lstsq_contract.py.
3. Close to gelsd (lin_reg_tensor_batched) on well-conditioned boxes.
4. With the inputs on another device: result on that device, in the input dtype, and
   close to the CPU result.

Run:
  python experiments/test_planar_fit.py
  python experiments/test_planar_fit.py --device cuda
"""
import argparse
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

SEED = 4242  # not a seed of the comparison
BOXES = 500
# gelsd is accurate to about cond(X) * eps here (widths >= 0.5, centres within 4)
GELSD_RTOL = {"float64": 1e-10, "float32": 1e-3}
DEVICE_RTOL = {"float64": 1e-12, "float32": 1e-5}

failures = []


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


def test_same_as_candidate(device, dtype_name):
    print(f"same bits as candidate A2 ({device}, {dtype_name})")
    dtype = getattr(torch, dtype_name)
    families = h.FAMILIES[:10] + ["offset_narrow", "nonfinite"]
    for k, family in enumerate(families):
        b = [t.to(device=device, dtype=dtype) for t in family_bounds(family, dtype_name, k)]
        grid, z = h.grid_and_targets(*b)
        check(family, same_bits(planar_fit.fit_planes(grid, z), lc.solve_A2(grid, z)),
              f"({grid.shape[0]} boxes)")


def test_same_on_real_boxes(device, dtype_name):
    print(f"same bits on boxes from the batched forward pass of 3layer ({device}, {dtype_name})")
    cfg = dict(networks=["3layer"], num_images=1, eps_indices=[0, 8, 15], n_splits=4, seed=7,
               per_network=2000, seed_family=SEED)
    bounds, _, _ = h.capture_real_boxes(cfg, torch.device("cpu"))
    dtype = getattr(torch, dtype_name)
    grid, z = h.grid_and_targets(*(t.to(device=device, dtype=dtype) for t in bounds))
    check("real 3layer", same_bits(planar_fit.fit_planes(grid, z), lc.solve_A2(grid, z)),
          f"({grid.shape[0]} boxes)")


def test_contract(device, dtype_name):
    print(f"contract checks C1-C7 ({device}, {dtype_name})")
    rows = ct.contract_checks("planar_fit", planar_fit.fit_planes,
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
        d = rel_diff(planar_fit.fit_planes(grid, z), pt.lin_reg_tensor_batched(grid, z))
        check(family, d <= GELSD_RTOL[dtype_name], f"(max rel diff {d:.2e})")


def test_device(device, dtype_name):
    print(f"inputs on {device} ({dtype_name})")
    dtype = getattr(torch, dtype_name)
    b = [t.to(dtype) for t in ct.g0_bounds("well_conditioned", BOXES, SEED + 99)]
    grid, z = h.grid_and_targets(*b)
    ref = planar_fit.fit_planes(grid, z)
    got = planar_fit.fit_planes(grid.to(device), z.to(device))
    check("device and dtype kept", got.device.type == device.type and got.dtype == dtype)
    d = rel_diff(got.cpu(), ref)
    check("close to the CPU result", d <= DEVICE_RTOL[dtype_name], f"(max rel diff {d:.2e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    h.set_production_flags()

    for dtype_name in ("float64", "float32"):
        if device.type == "mps" and dtype_name == "float64":
            print("SKIP float64 on mps: the device has no float64")
            continue
        test_same_as_candidate(device, dtype_name)
        test_same_on_real_boxes(device, dtype_name)
        test_contract(device, dtype_name)
        test_close_to_gelsd(device, dtype_name)
        if device.type != "cpu":
            test_device(device, dtype_name)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
