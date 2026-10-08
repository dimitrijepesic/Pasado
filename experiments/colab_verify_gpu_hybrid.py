"""Runs the full CPU/CUDA verification matrix for the gpu-hybrid branch.

The default covers all four FFNNs, 30 images and all 16 epsilons. Use --quick
for a five-image smoke test over epsilon indices 0, 8 and 15.
"""
import argparse
import ast
import csv
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTION = os.path.join(ROOT, "Section_5_4")
ALL_NETWORKS = ("3layer", "4layer", "5layer", "big")
EPSILONS = [10 ** (-k / 4) * 2 for k in range(2, 18)]
SELECTORS = {
    "PASADO_VECTORIZED_PRECISE": "1",
    "PASADO_VEC_BOUNDARY": "1",
    "PASADO_BATCHED_LSTSQ": "1",
    "PASADO_BATCHED_GRID": "1",
}


def output_values(text, name):
    prefix = name + " "
    for line in text.splitlines():
        if line.startswith(prefix):
            return ast.literal_eval(line[len(prefix):])
    raise RuntimeError(f"missing {name!r} in get_lipschitz.py output")


def run_analysis(network, device, dtype, real_cubic, precise_only,
                 num_images, eps_indices):
    env = dict(os.environ)
    env.update(SELECTORS)
    env["PASADO_REAL_CUBIC"] = "1" if real_cubic else "0"
    cmd = [
        sys.executable,
        "get_lipschitz.py",
        "--network", network,
        "--device", device,
        "--dtype", dtype,
        "--num-images", str(num_images),
        "--seed", "7",
        "--no-save",
    ]
    if precise_only:
        cmd.append("--precise-only")
    if eps_indices is not None:
        cmd.extend(["--eps-indices", *[str(index) for index in eps_indices]])

    label = (
        f"{network} device={device} dtype={dtype} "
        f"real={int(real_cubic)} precise_only={int(precise_only)}"
    )
    print("RUN", label, flush=True)
    completed = subprocess.run(
        cmd,
        cwd=SECTION,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout[-4000:] + completed.stderr[-4000:])
        raise RuntimeError(f"analysis failed: {label}")

    result = {"lc_precise": output_values(completed.stdout, "lc_precise")}
    if not precise_only:
        result["lc_zonos"] = output_values(completed.stdout, "lc_zonos")
        result["lc_intervals"] = output_values(completed.stdout, "lc_intervals")
    return result


def relative_difference(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError(f"length mismatch: {len(reference)} != {len(candidate)}")
    differences = [
        abs(a - b) / max(abs(a), 1e-300)
        for a, b in zip(reference, candidate)
    ]
    index = max(range(len(differences)), key=differences.__getitem__)
    return differences[index], index


def record(rows, network, check, maximum, eps_index, limit, passed):
    rows.append({
        "network": network,
        "check": check,
        "max_relative_difference": f"{maximum:.17g}",
        "epsilon_index": eps_index,
        "epsilon": f"{EPSILONS[eps_index]:.17g}",
        "limit": f"{limit:.17g}",
        "result": "PASS" if passed else "FAIL",
    })
    print(
        f"  {check:42s} max_rel={maximum:.3e} "
        f"eps_index={eps_index:2d} limit={limit:.1e} "
        f"{'PASS' if passed else 'FAIL'}"
    )


def verify_network(network, num_images, eps_indices, rows,
                   float64_limit, float32_limit):
    ok = True
    reported_index = (
        (lambda position: eps_indices[position])
        if eps_indices is not None else
        (lambda position: position)
    )

    cpu_full = run_analysis(
        network, "cpu", "float64", True, False, num_images, eps_indices
    )
    cuda_full = run_analysis(
        network, "cuda", "float64", True, False, num_images, eps_indices
    )
    for metric in ("lc_precise", "lc_zonos", "lc_intervals"):
        maximum, index = relative_difference(cpu_full[metric], cuda_full[metric])
        passed = maximum < float64_limit
        record(rows, network, f"real float64 full {metric}",
               maximum, reported_index(index), float64_limit, passed)
        ok &= passed

    cuda_precise_a = run_analysis(
        network, "cuda", "float64", True, True, num_images, eps_indices
    )
    cuda_precise_b = run_analysis(
        network, "cuda", "float64", True, True, num_images, eps_indices
    )
    maximum, index = relative_difference(
        cuda_precise_a["lc_precise"], cuda_precise_b["lc_precise"]
    )
    deterministic = cuda_precise_a["lc_precise"] == cuda_precise_b["lc_precise"]
    record(rows, network, "real float64 CUDA repeat",
           maximum, reported_index(index), 0.0, deterministic)
    ok &= deterministic

    maximum, index = relative_difference(
        cuda_full["lc_precise"], cuda_precise_a["lc_precise"]
    )
    same_mode = cuda_full["lc_precise"] == cuda_precise_a["lc_precise"]
    record(rows, network, "full vs precise-only CUDA",
           maximum, reported_index(index), 0.0, same_mode)
    ok &= same_mode

    cpu_float32 = run_analysis(
        network, "cpu", "float32", True, True, num_images, eps_indices
    )
    cuda_float32 = run_analysis(
        network, "cuda", "float32", True, True, num_images, eps_indices
    )
    maximum, index = relative_difference(
        cpu_float32["lc_precise"], cuda_float32["lc_precise"]
    )
    passed = maximum < float32_limit
    record(rows, network, "real float32 precise lc_precise",
           maximum, reported_index(index), float32_limit, passed)
    ok &= passed

    cpu_complex = run_analysis(
        network, "cpu", "float64", False, True, num_images, eps_indices
    )
    cuda_complex = run_analysis(
        network, "cuda", "float64", False, True, num_images, eps_indices
    )
    maximum, index = relative_difference(
        cpu_complex["lc_precise"], cuda_complex["lc_precise"]
    )
    diverged = maximum >= float64_limit
    record(rows, network, "complex float64 CPU/CUDA divergence",
           maximum, reported_index(index), float64_limit, diverged)
    ok &= diverged
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", choices=ALL_NETWORKS,
                    default=list(ALL_NETWORKS))
    ap.add_argument("--quick", action="store_true",
                    help="use 5 images and epsilon indices 0, 8 and 15")
    ap.add_argument("--float64-limit", type=float, default=1e-9)
    ap.add_argument("--float32-limit", type=float, default=1e-5,
                    help="relative CPU/CUDA tolerance; default matches float32 assert_close")
    ap.add_argument("--out", default=None,
                    help="optional CSV output path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print("CUDA", torch.cuda.get_device_name(0))

    num_images = 5 if args.quick else 30
    eps_indices = [0, 8, 15] if args.quick else None
    rows = []
    results = [
        verify_network(
            network,
            num_images,
            eps_indices,
            rows,
            args.float64_limit,
            args.float32_limit,
        )
        for network in args.networks
    ]

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("wrote", out_path)

    ok = all(results)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
