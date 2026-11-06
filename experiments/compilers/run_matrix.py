"""Runs bench.py for every (tool, mode, kernel, size): a cold process with an empty cache,
then a warm process sharing it (results/compilers_2026-11/PROTOCOL.md). Resumable: a
combination whose record is already in --out is skipped.

  python experiments/compilers/run_matrix.py --device cuda --out DIR
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kernels_torch import DEFAULT_SIZES  # noqa: E402

TORCH_MODES = ("eager", "default", "max-autotune", "reduce-overhead")


def done(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            keys.add((r["tool"], r["mode"], r["kernel"], r["size"], r["phase"]))
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="directory; records go to records.jsonl")
    ap.add_argument("--kernels", nargs="+", default=list(DEFAULT_SIZES))
    ap.add_argument("--tools", nargs="+", default=["torch", "jax"])
    ap.add_argument("--torch-modes", nargs="+", default=list(TORCH_MODES))
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--sizes", nargs="*", default=None, help="override, e.g. fit=1024")
    ap.add_argument("--timeout-s", type=int, default=1800)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "records.jsonl")
    sizes = dict(DEFAULT_SIZES)
    for item in args.sizes or []:
        k, v = item.split("=")
        sizes[k] = v.split(",")
    have = done(out)
    for kernel in args.kernels:
        for size in sizes[kernel]:
            combos = [("torch", m) for m in args.torch_modes if "torch" in args.tools]
            combos += [("jax", "jit")] if "jax" in args.tools else []
            for tool, mode in combos:
                phases = ["none"] if mode == "eager" else ["cold", "warm"]
                if all((tool, mode, kernel, size, p) in have for p in phases):
                    continue
                cache = tempfile.mkdtemp(prefix=f"cache_{tool}_{mode}_{kernel}_")
                try:
                    for phase in phases:
                        if (tool, mode, kernel, size, phase) in have:
                            continue
                        cmd = [sys.executable, os.path.join(HERE, "bench.py"), "--tool", tool, "--kernel", kernel,
                               "--size", size, "--device", args.device, "--phase", phase,
                               "--repeats", str(args.repeats), "--out", out]
                        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=os.path.join(cache, "inductor"),
                                   XLA_PYTHON_CLIENT_PREALLOCATE="false")
                        if tool == "torch":
                            cmd += ["--mode", mode]
                        else:
                            cmd += ["--jax-cache-dir", os.path.join(cache, "jax")]
                        try:
                            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout_s)
                            line = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
                            if proc.returncode != 0:
                                line += " | exit %d: %s" % (proc.returncode, proc.stderr.strip()[-300:])
                        except subprocess.TimeoutExpired:
                            line = "TIMEOUT"
                            with open(out, "a") as handle:
                                handle.write(json.dumps(dict(tool=tool, mode=mode, kernel=kernel, size=size,
                                                             phase=phase, device=args.device,
                                                             error=f"timeout after {args.timeout_s} s")) + "\n")
                        print(line, flush=True)
                finally:
                    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
