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


def saved_files(*extra):
    env = {**os.environ, **FLAGS}
    env.pop("PASADO_DEVICE", None)
    with tempfile.TemporaryDirectory() as tmp:
        base = [arg for arg in BASE if arg != "--no-save"]
        p = subprocess.run(
            [sys.executable, "get_lipschitz.py", *base,
             "--results-dir", tmp, *extra],
            cwd=SEC, env=env, capture_output=True, text=True,
        )
        return p.returncode, p.stdout + p.stderr, sorted(os.listdir(tmp))


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

    code, _, full_files = saved_files("--seed", "5")
    check(code == 0 and len(full_files) == 6,
          "a non-default full run saves exactly six files")
    check(all("_n3_eps0-15_seed5_sel11110" in name for name in full_files),
          "non-default image/epsilon/seed/selectors are encoded in every filename")

    code, _, precise_files = saved_files("--seed", "5", "--precise-only")
    check(code == 0 and len(precise_files) == 2,
          "a precise-only run saves exactly two files")
    check(all("_preciseonly_" in name for name in precise_files),
          "precise-only files cannot overwrite default results")

    invalid_cases = [
        (("--num-images", "0"), "must be at least 1"),
        (("--n-splits", "0"), "must be at least 1"),
        (("--eps-indices", "0", "0"), "must not contain duplicates"),
        (("--eps-indices", "16"), "between 0 and 15"),
        (("--seed", "-1"), "between 0 and 2**32 - 1"),
        (("--device", "not-a-device"), "invalid --device"),
    ]
    for extra, message in invalid_cases:
        code, out = run(*extra)
        check(code != 0 and message in out,
              f"invalid {' '.join(extra)} is rejected during argument validation")

    if not torch.cuda.is_available():
        code, out = run("--device", "cuda")
        check(code != 0 and "CUDA is not available" in out,
              "--device cuda without CUDA stops with a clear message")

    if torch.backends.mps.is_available():
        code, out_mps = run("--device", "mps", "--num-images", "1",
                             "--eps-indices", "0", "--precise-only", "--seed", "5")
        _, out_ref = run("--device", "cpu", "--num-images", "1",
                         "--eps-indices", "0", "--precise-only", "--seed", "5")
        mps_lc = values(out_mps, "lc_precise")[0]
        cpu_lc = values(out_ref, "lc_precise")[0]
        check(code == 0 and abs(mps_lc - cpu_lc) / abs(cpu_lc) < 1e-3,
              "the real get_lipschitz MPS CLI path runs within float32 tolerance")

    print("ALL PASS" if not failures else f"{len(failures)} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
