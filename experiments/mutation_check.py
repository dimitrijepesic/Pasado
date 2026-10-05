"""Checks that the tests can fail. One deliberate bug at a time is planted in a
temporary copy of precise_transformer.py and every test is run against it.
A test that still passes does not guard against that bug. The repository is
never modified.

Run: python experiments/mutation_check.py [--only M3 M7]"""
import argparse
import concurrent.futures as cf
import difflib
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "forward_mode_tensorized_src"
TARGET = f"{SRC}/precise_transformer.py"

# (short name, path relative to the repo root)
TESTS = [
    ("grid",      f"{SRC}/test_batched_grid.py"),
    ("selector",  f"{SRC}/test_selector_paths.py"),
    ("corners",   f"{SRC}/test_check_corners_tensor.py"),
    ("boundary",  f"{SRC}/test_check_nonlinear_boundary_tensor.py"),
    ("max_error", f"{SRC}/test_compute_max_error_tensor.py"),
    ("lin_reg",   "experiments/test_batched_lin_reg.py"),
    ("variants",  "experiments/test_benchmark_variants.py"),
    ("cli",       "experiments/test_get_lipschitz_cli.py"),
    # End-to-end comparison on real images. Slower than the unit tests, and it is
    # the check that originally caught the float32 root precision bug (M6).
    ("e2e",       "experiments/correctness_e2e.py"),
]

# Each mutant: id, the function it breaks, why it is a realistic mistake, and an
# exact (old, new) text replacement inside precise_transformer.py.
MUTANTS = [
    dict(id="M1", func="_linspace_rows",
         why="wrong step size in the batched sampling grid",
         old="    step = (hi - lo) / (STEPS - 1)\n",
         new="    step = (hi - lo) / STEPS\n"),
    dict(id="M2", func="check_corners_tensor",
         why="forget the negated side of the error (only F, not -F)",
         old="    return torch.max(row_max, -row_min)  # [n]\n",
         new="    return row_max  # [n]\n"),
    dict(id="M3", func="check_corners_tensor",
         why="use a wrong corner (lower-x corner twice, upper-x corner dropped)",
         old="    xs = torch.stack((lx, lx, ux, ux), dim=1)\n",
         new="    xs = torch.stack((lx, lx, lx, ux), dim=1)\n"),
    dict(id="M4", func="check_nonlinear_boundary_tensor",
         why="the uy branch divides by ly instead of uy",
         old="    A_by_uy = A / uy\n",
         new="    A_by_uy = A / ly\n"),
    dict(id="M5", func="_max_objective_over_x_candidates",
         why="filtered-out roots count as 0 instead of -inf, so they can win the max",
         old=('    evaluation = torch.nan_to_num(evaluation, -float("Inf"))\n'
              '    return torch.max(evaluation, dim=1).values\n'),
         new=('    evaluation = torch.nan_to_num(evaluation, 0.0)\n'
              '    return torch.max(evaluation, dim=1).values\n')),
    dict(id="M6", func="check_nonlinear_boundary_tensor",
         why="roots are left in float32 instead of promoted to the input dtype "
             "(a mistake that was actually made once during development)",
         old="    roots_x = torch.stack((root1, root2, root3), dim=1).to(ly.dtype)  # [n, 3]\n",
         new="    roots_x = torch.stack((root1, root2, root3), dim=1)  # [n, 3]\n"),
    dict(id="M7", func="compute_max_error",
         why="drop the uy boundary term from the combined maximum",
         old="        return torch.maximum(torch.maximum(errors, errors2_ly), errors2_uy)\n",
         new="        return torch.maximum(errors, errors2_ly)\n"),
    dict(id="M8", func="sigmoid_prime_product_tensor",
         why="selector inverted: vectorized path used exactly when it is switched off",
         old="    ABCs_tensor = torch.stack((As, Bs, Cs), dim=1) if USE_VECTORIZED_PRECISE else None\n",
         new="    ABCs_tensor = torch.stack((As, Bs, Cs), dim=1) if not USE_VECTORIZED_PRECISE else None\n"),
    dict(id="M9", func="lin_reg_tensor_batched",
         why="design matrix columns in the wrong order (intercept last)",
         old="    xplusone = torch.cat((ones, x_batch), dim=2)  # [n, m, 3], intercept-first\n",
         new="    xplusone = torch.cat((x_batch, ones), dim=2)  # [n, m, 3], intercept-last\n"),
    dict(id="M10", func="lin_reg_tensor_batched",
         why="different LAPACK driver (gelsy instead of gelsd): same answer up to "
             "rounding, not bit-identical",
         old="    R = torch.linalg.lstsq(xplusone, zs_batch, driver='gelsd').solution  # [n, 3, 1]\n",
         new="    R = torch.linalg.lstsq(xplusone, zs_batch, driver='gelsy').solution  # [n, 3, 1]\n"),
]


def clean_env():
    """Environment without PASADO_* overrides, so the tests use the module defaults."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PASADO_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def make_workdir():
    """Temporary repo root containing every file that the mutation tests can write."""
    root = tempfile.mkdtemp(prefix="pasado_mutation_")
    try:
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        for d in (SRC, "experiments"):
            shutil.copytree(os.path.join(REPO, d), os.path.join(root, d), ignore=ignore)
        section = os.path.join(root, "Section_5_4")
        os.makedirs(section)
        for filename in ("model.py", "get_lipschitz.py"):
            shutil.copy2(os.path.join(REPO, "Section_5_4", filename), section)
        shutil.copytree(os.path.join(REPO, "Section_5_4", "trained"),
                        os.path.join(section, "trained"), ignore=ignore)
        shutil.copytree(os.path.join(REPO, "Section_5_4", "MNIST_Data"),
                        os.path.join(section, "MNIST_Data"), ignore=ignore)
        os.makedirs(os.path.join(root, "logs"), exist_ok=True)   # e2e writes a CSV here
        return root
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def run_one_test(root, rel_path):
    """Run one test script in the temporary root; returns (exit code, last output line)."""
    path = os.path.join(root, rel_path)
    try:
        p = subprocess.run([sys.executable, path], cwd=os.path.dirname(path),
                           env=clean_env(), capture_output=True, text=True,
                           timeout=900)
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    out = (p.stdout + p.stderr).strip().splitlines()
    tail = out[-1] if out else ""
    # test_batched_lin_reg reports "bitwise-identical everywhere" without
    # enforcing it; surface that so a silent change in numerics is visible.
    bitwise_lost = "bitwise-identical everywhere: NO" in p.stdout
    return p.returncode, ("BITWISE LOST; " if bitwise_lost else "") + tail[:90]


def run_all_tests(root):
    """Run all tests in parallel; returns {name: (exit code, last line)}."""
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futs = {name: ex.submit(run_one_test, root, rel) for name, rel in TESTS}
        return {name: f.result() for name, f in futs.items()}


def show_diff(m, original):
    """Print the mutant as a diff."""
    mutated = original.replace(m["old"], m["new"])
    diff = difflib.unified_diff(original.splitlines(True), mutated.splitlines(True),
                                f"a/{TARGET}", f"b/{TARGET}", n=2)
    print("".join(diff))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="run only these mutant ids, e.g. M3 M7")
    args = ap.parse_args()

    known_ids = {m["id"] for m in MUTANTS}
    unknown = sorted(set(args.only or ()) - known_ids)
    if unknown:
        ap.error(f"unknown mutant id(s): {', '.join(unknown)}")

    mutants = [m for m in MUTANTS if not args.only or m["id"] in args.only]
    if not mutants:
        ap.error("no mutants selected")

    root = None
    try:
        root = make_workdir()
        target_path = os.path.join(root, TARGET)
        with open(target_path, encoding="utf-8") as f:
            original = f.read()

        print("=== baseline: all tests on the unmodified copy ===")
        base = run_all_tests(root)
        for name, (code, tail) in base.items():
            print(f"  {name:10s} exit={code}  {tail}")
        if any(code != 0 for code, _ in base.values()):
            print("Baseline is not green, so mutation results would mean nothing. Stopping.")
            return 2

        matrix = {}
        for m in mutants:
            if original.count(m["old"]) != 1:
                print(f"\n{m['id']}: original text found {original.count(m['old'])} times, expected 1")
                return 2
            print(f"\n=== {m['id']}  {m['func']}: {m['why']} ===")
            show_diff(m, original)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(original.replace(m["old"], m["new"]))
            try:
                res = run_all_tests(root)
            finally:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(original)
            matrix[m["id"]] = res
            for name, (code, tail) in res.items():
                verdict = "KILLED  " if code != 0 else "survived"
                print(f"  {name:10s} {verdict} exit={code}  {tail}")

        names = [n for n, _ in TESTS]
        print("\n=== summary (K = killed, . = survived, K* = survived with bitwise loss) ===")
        print(f"{'':5s}" + "".join(f"{n:>11s}" for n in names) + "   caught by any test?")
        all_caught = True
        for m in mutants:
            cells, caught = [], False
            for n in names:
                code, tail = matrix[m["id"]][n]
                if code != 0:
                    cells.append("K"); caught = True
                elif tail.startswith("BITWISE LOST"):
                    cells.append("K*")
                else:
                    cells.append(".")
            note = "yes" if caught else ("only as a printed warning" if "K*" in cells
                                        else "NO, none of the tests notices this")
            print(f"{m['id']:5s}" + "".join(f"{c:>11s}" for c in cells) + f"   {note}")
            all_caught &= caught
        return 0 if all_caught else 1
    finally:
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
