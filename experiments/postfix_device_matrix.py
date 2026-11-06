"""One device's half of the CPU/CUDA matrix, written per image (section D of
results/postfix_rerun_2026-11/PROTOCOL.md).

Every configuration runs Section_5_4/get_lipschitz.py in a subprocess with --no-save and
--per-image-csv, so the reference .pth files are never written and every (image, epsilon)
is recorded. A configuration whose CSV already exists is skipped, so a rerun resumes.

  python experiments/postfix_device_matrix.py --device cpu  --out-dir DIR
  python experiments/postfix_device_matrix.py --device cuda --out-dir DIR --repeat-precise

Compare two halves with experiments/compare_device_matrix.py.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTION = os.path.join(ROOT, "Section_5_4")
NETWORKS = ("3layer", "4layer", "5layer", "big")
SELECTORS = {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_VEC_BOUNDARY": "1",
             "PASADO_BATCHED_LSTSQ": "1", "PASADO_BATCHED_GRID": "1"}
# name: (dtype, real cubic, precise only)
CONFIGS = {
    "full_f64_real": ("float64", True, False),
    "precise_f64_real": ("float64", True, True),
    "precise_f32_real": ("float32", True, True),
    "precise_f64_complex": ("float64", False, True),
}


def run_config(network, config, device, out_csv, num_images, eps_indices, seed):
    dtype, real, precise_only = CONFIGS[config]
    env = dict(os.environ, **SELECTORS, PASADO_REAL_CUBIC="1" if real else "0")
    cmd = [sys.executable, "get_lipschitz.py", "--network", network, "--device", device,
           "--dtype", dtype, "--num-images", str(num_images), "--seed", str(seed),
           "--no-save", "--per-image-csv", out_csv]
    if precise_only:
        cmd.append("--precise-only")
    if eps_indices is not None:
        cmd += ["--eps-indices", *map(str, eps_indices)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=SECTION, env=env, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        if os.path.exists(out_csv):
            os.remove(out_csv)
        raise RuntimeError(f"{network} {config} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return seconds, proc.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--networks", nargs="+", choices=NETWORKS, default=list(NETWORKS))
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=None,
                    help="default: full_f64_real precise_f32_real precise_f64_complex, "
                         "plus precise_f64_real with --repeat-precise")
    ap.add_argument("--repeat-precise", action="store_true",
                    help="also run precise_f64_real twice (repeat_1, repeat_2)")
    ap.add_argument("--num-images", type=int, default=30)
    ap.add_argument("--eps-indices", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    configs = args.configs or ["full_f64_real", "precise_f32_real", "precise_f64_complex"]
    jobs = [(c, "") for c in configs]
    if args.repeat_precise:
        jobs += [("precise_f64_real", "_repeat_1"), ("precise_f64_real", "_repeat_2")]
    args.out_dir = os.path.abspath(args.out_dir)  # the subprocess runs in Section_5_4
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "runs.jsonl")
    for network in args.networks:
        for config, tag in jobs:
            name = f"{network}_{config}{tag}"
            out_csv = os.path.join(args.out_dir, name + ".csv")
            if os.path.exists(out_csv):
                print(f"skip {name} (exists)", flush=True)
                continue
            seconds, stdout = run_config(network, config, args.device, out_csv,
                                         args.num_images, args.eps_indices, args.seed)
            cfg_line = next((l for l in stdout.splitlines() if l.startswith("config ")), "")
            with open(log_path, "a") as handle:
                handle.write(json.dumps(dict(name=name, network=network, config=config,
                                             device=args.device, seconds=seconds,
                                             config_line=cfg_line, host=platform.node(),
                                             machine=platform.machine())) + "\n")
            print(f"done {name} in {seconds:.1f} s", flush=True)


if __name__ == "__main__":
    main()
