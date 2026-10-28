"""Tests for the automatic choice of the plane fit (batched_splits.plane_fit_for with
USE_AUTO_PLANE_FIT, env PASADO_AUTO_PLANE_FIT).

The choice holds on CUDA only. On a machine without CUDA the tests of the choice itself
widen batched_splits.AUTO_PLANE_FIT_DEVICES to the CPU, and a separate check confirms that
with the default devices the CPU keeps gelsd. With --device cuda nothing is widened.

0. The thresholds in batched_splits are the measured ones (512 for float64, 640 for
   float32, results/plane_fit_crossover_2026-10-28/thresholds.json), checked against
   constants here and against that file when it is present.
1. By size: threshold - 1 boxes -> gelsd; threshold and threshold + 1 -> the eager closed
   form, for float64 and float32, with spies on both functions.
2. The override (PLANE_FIT_THRESHOLD_OVERRIDE, env PASADO_PLANE_FIT_THRESHOLD).
3. The automatic choice combined with PASADO_CLOSED_FORM_FIT or PASADO_COMPILED_FIT raises.
4. Outside the automatic devices: gelsd at every size.
5. Batched passes: the bounds equal, bit for bit, those of the explicit path for that
   size; every per-split bound against the sequential loop; chunking changes the boxes
   per call and with them the choice, as it should.
6. No TorchInductor graph is compiled in automatic mode.

Run:
  python experiments/test_auto_plane_fit.py
  python experiments/test_auto_plane_fit.py --device cuda
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))
sys.path.insert(0, HERE)

import batched_splits as bs  # noqa: E402
import planar_fit  # noqa: E402
import precise_transformer as pt  # noqa: E402
import compare_lstsq_solvers as h  # noqa: E402
import lstsq_contract as ct  # noqa: E402
from measure_split_sweep import build_forward_precise, load_network_and_images  # noqa: E402
from test_batched_splits import sequential_split_bounds  # noqa: E402

RTOL = 1e-10
# the A100 crossover of the v3 code, independent of the module under test
MEASURED = {torch.float64: 512, torch.float32: 640}
MEASURED_FILE = os.path.join(ROOT, "results", "plane_fit_crossover_2026-10-28", "thresholds.json")
failures = []
calls = {"gelsd": 0, "closed_form": 0}
gelsd, fit_planes = pt.lin_reg_tensor_batched, planar_fit.fit_planes


def counted_gelsd(grid, z):
    calls["gelsd"] += 1
    return gelsd(grid, z)


def counted_fit_planes(grid, z, rcond=None):
    calls["closed_form"] += 1
    return fit_planes(grid, z, rcond)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def switches(auto=False, closed_form=False, compiled=False, override=None):
    bs.USE_AUTO_PLANE_FIT, bs.USE_CLOSED_FORM_FIT, bs.USE_COMPILED_FIT = auto, closed_form, compiled
    bs.PLANE_FIT_THRESHOLD_OVERRIDE = override


def boxes(n, dtype, device):
    b = ct.g0_bounds("well_conditioned", n, 900 + n)
    return h.grid_and_targets(*(t.to(device=device, dtype=dtype) for t in b))


def chosen(n, dtype, device):
    """Which fit plane_fit_for picks for n boxes, by calling it once."""
    grid, z = boxes(n, dtype, device)
    calls["gelsd"] = calls["closed_form"] = 0
    bs.plane_fit_for(grid)(grid, z)
    if calls["gelsd"] and not calls["closed_form"]:
        return "gelsd"
    if calls["closed_form"] and not calls["gelsd"]:
        return "closed_form"
    return f"unclear {calls}"


def test_thresholds():
    print("thresholds are the measured ones")
    check("batched_splits.AUTO_PLANE_FIT_THRESHOLD", bs.AUTO_PLANE_FIT_THRESHOLD == MEASURED,
          f"({bs.AUTO_PLANE_FIT_THRESHOLD})")
    if os.path.exists(MEASURED_FILE):
        with open(MEASURED_FILE) as f:
            t = json.load(f)["thresholds"]
        from_file = {torch.float64: t["float64"]["threshold"], torch.float32: t["float32"]["threshold"]}
        check("constants match the archived measurement", from_file == MEASURED, f"({from_file})")


def test_by_size(device):
    print(f"choice by size ({device})")
    switches(auto=True)
    for dtype, thr in MEASURED.items():
        for n, want in ((thr - 1, "gelsd"), (thr, "closed_form"), (thr + 1, "closed_form")):
            got = chosen(n, dtype, device)
            check(f"{str(dtype)[6:]} n={n}", got == want, f"(threshold {thr}: {got})")


def test_override(device):
    print(f"threshold override ({device})")
    switches(auto=True, override=100)
    for dtype in MEASURED:
        check(f"{str(dtype)[6:]} n=99 below the override", chosen(99, dtype, device) == "gelsd")
        check(f"{str(dtype)[6:]} n=100 at the override", chosen(100, dtype, device) == "closed_form")


def test_invalid_combinations(device):
    print("invalid combinations")
    grid, _ = boxes(16, torch.float64, device)
    for kwargs in (dict(closed_form=True), dict(compiled=True), dict(closed_form=True, compiled=True)):
        switches(auto=True, **kwargs)
        try:
            bs.plane_fit_for(grid)
            check(f"auto with {kwargs}", False, "(no error)")
        except ValueError:
            check(f"auto with {kwargs}", True, "(ValueError)")


def test_other_devices(saved_devices):
    print(f"default devices {saved_devices}: the CPU keeps gelsd")
    bs.AUTO_PLANE_FIT_DEVICES = saved_devices
    switches(auto=True)
    for dtype, thr in MEASURED.items():
        check(f"cpu {str(dtype)[6:]} n={4 * thr}", chosen(4 * thr, dtype, torch.device("cpu")) == "gelsd")


def bounds(layers, img, eps, n_splits, chunk, **kwargs):
    switches(**kwargs)
    calls["gelsd"] = calls["closed_form"] = 0
    out = bs.split_bounds(layers, img, eps, n_splits, seed=7, chunk=chunk).cpu().double()
    return out, dict(calls)


def test_batched(device):
    print(f"batched passes against the explicit paths and the sequential loop ({device})")
    dtype = torch.float64
    eps = h.EPS_SCHEDULE[0]
    for network, cases in (("3layer", ((4, None, "gelsd"), (8, None, "closed_form"),
                                       (8, 4, "gelsd"), (8, 8, "closed_form"))),
                           ("big", ((1, None, "closed_form"),))):
        net, images = load_network_and_images(network, device, dtype, 1)
        layers = bs.extract_affine_layers(net, network)
        forward_fn = build_forward_precise(pt, net, network)
        img = images[0][1]
        width = 1024 if network == "big" else 100
        for n_splits, chunk, want in cases:
            per_call = (chunk or n_splits) * width
            name = f"{network} n_splits={n_splits} chunk={chunk} ({per_call} boxes per call)"
            auto, used = bounds(layers, img, eps, n_splits, chunk, auto=True)
            explicit, _ = bounds(layers, img, eps, n_splits, chunk,
                                 closed_form=(want == "closed_form"))
            only = used[want] > 0 and used["gelsd" if want == "closed_form" else "closed_form"] == 0
            check(f"{name}: {want} only", only, f"({used})")
            check(f"{name}: same bits as the explicit {want} path", torch.equal(auto, explicit))
            switches()
            ref = sequential_split_bounds(forward_fn, img, eps, n_splits, 7, device, dtype)
            d = ((auto - ref).abs() / ref.abs()).max().item()
            check(f"{name}: per-split bounds against the sequential loop", d <= RTOL, f"({d:.2e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    h.set_production_flags()
    torch.set_default_dtype(torch.float64)
    pt.lin_reg_tensor_batched, planar_fit.fit_planes = counted_gelsd, counted_fit_planes
    saved = bs.AUTO_PLANE_FIT_DEVICES
    if device.type not in saved:
        print(f"widening AUTO_PLANE_FIT_DEVICES to {device.type} for the tests of the choice")
        bs.AUTO_PLANE_FIT_DEVICES = tuple(saved) + (device.type,)
    try:
        with torch.no_grad():
            test_thresholds()
            test_by_size(device)
            test_override(device)
            test_invalid_combinations(device)
            test_batched(device)
            test_other_devices(saved)
    finally:
        bs.AUTO_PLANE_FIT_DEVICES = saved
        switches()
    graphs = torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)
    check("no TorchInductor graph compiled in automatic mode", graphs == 0, f"({graphs})")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
