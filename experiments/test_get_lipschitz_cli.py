import ast
import os
import pstats
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SEC = os.path.join(os.path.dirname(HERE), "Section_5_4")

# All five selectors set, so the result does not depend on the shell.
FLAGS = {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_VEC_BOUNDARY": "1",
         "PASADO_BATCHED_LSTSQ": "1", "PASADO_BATCHED_GRID": "1", "PASADO_REAL_CUBIC": "0"}

# Small run: 3 images, 2 epsilons, never saves anything.
BASE = ["--network", "3layer", "--num-images", "3", "--eps-indices", "0", "15", "--no-save"]


def run(*extra):
    env = {**os.environ, **FLAGS}
    env.pop("PASADO_DEVICE", None)
    p = subprocess.run([sys.executable, "get_lipschitz.py", *BASE, *extra],
                       cwd=SEC, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def called_functions(*extra):
    """Names of all functions get_lipschitz.py calls, from a cProfile run."""
    env = {**os.environ, **FLAGS}
    with tempfile.TemporaryDirectory() as tmp:
        prof = os.path.join(tmp, "run.prof")
        subprocess.run([sys.executable, "-m", "cProfile", "-o", prof, "get_lipschitz.py",
                        *BASE, *extra], cwd=SEC, env=env, capture_output=True, check=True)
        return {key[2] for key in pstats.Stats(prof).stats}


def values(out, name):
    for line in out.splitlines():
        if line.startswith(name + " "):
            return ast.literal_eval(line[len(name) + 1:])
    raise AssertionError(f"no '{name}' line in the output")


def main():
    failures = []

    def check(ok, message):
        print(f"{'ok  ' if ok else 'FAIL'} {message}")
        if not ok:
            failures.append(message)

    code, out_a = run("--seed", "5")
    _, out_b = run("--seed", "5")
    check(code == 0, "seeded run exits with 0")
    check(values(out_a, "lc_precise") == values(out_b, "lc_precise"),
          "same seed twice gives identical lc_precise")

    _, out_c = run("--seed", "6")
    check(values(out_a, "lc_precise") != values(out_c, "lc_precise"),
          "a different seed gives a different lc_precise")

    _, out_cpu = run("--seed", "5", "--device", "cpu")
    check(values(out_a, "lc_precise") == values(out_cpu, "lc_precise")
          and values(out_a, "lc_zonos") == values(out_cpu, "lc_zonos"),
          "--device cpu gives the same result as the default")

    _, out_po = run("--seed", "5", "--precise-only")
    check(values(out_a, "lc_precise") == values(out_po, "lc_precise"),
          "--precise-only gives the same lc_precise as the full run")
    check("lc_zonos " not in out_po, "--precise-only does not print lc_zonos")

    # Printing is a separate switch from computing, so look at what actually ran.
    full = called_functions()
    only = called_functions("--precise-only")
    check({"forward_zono", "forward_interval", "forward_zono_precise"} <= full,
          "the full run calls the zonotope, interval and precise analyses")
    check("forward_zono_precise" in only
          and "forward_zono" not in only and "forward_interval" not in only,
          "--precise-only runs only the precise analysis")

    check("WARNING: --num-images 3" in out_a, "--num-images other than 30 prints a warning")
    check(len(values(out_a, "image_ids")) == 3, "image_ids lists the images that were used")
    check("config {" in out_a, "the run prints its configuration")

    if not torch.cuda.is_available():
        code, out = run("--device", "cuda")
        check(code != 0 and "CUDA is not available" in out,
              "--device cuda without CUDA stops with a clear message")

    print("ALL PASS" if not failures else f"{len(failures)} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
