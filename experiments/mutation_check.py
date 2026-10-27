"""Checks that the tests can fail. One deliberate bug at a time is planted in a
temporary copy of the source tree and every test is run against it.
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
SIMPLE_ZONO = f"{SRC}/SimpleZono.py"

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
    # Brute-force and autograd lower bounds; the other tests compare code paths
    # that share SimpleZono, so they cannot see a bug common to both.
    ("soundness", "experiments/test_soundness.py"),
    # The closed-form plane fit of the batched path: bit-identical to the candidate
    # chosen by experiments/compare_lstsq_solvers.py, and the batched pass with and
    # without it against the sequential code.
    ("planar_fit", "experiments/test_planar_fit.py"),
    ("batched",   "experiments/test_batched_splits.py", ["--quick"]),
    ("batched_cf", "experiments/test_batched_splits.py", ["--quick", "--closed-form"]),
]
PLANAR_FIT = f"{SRC}/planar_fit.py"
BATCHED = f"{SRC}/batched_splits.py"

# Each mutant: id, the function it breaks, why it is a realistic mistake, and an
# exact (old, new) text replacement inside `file` (default: precise_transformer.py).
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
    dict(id="M11", func="Zonotope.__neg__", file=SIMPLE_ZONO,
         why="negate the center only, keeping the generators (the upstream bug)",
         old="        return Zonotope(-self.centers, -self.generators)\n",
         new="        return Zonotope(-self.centers, self.generators)\n"),
    dict(id="M12", func="DualZonotope.__neg__", file=SIMPLE_ZONO,
         why="negate the centers only, keeping the generators (the upstream bug)",
         old="        return DualZonotope(-self.real.centers, -self.real.generators, "
             "-self.dual.centers, -self.dual.generators)\n",
         new="        return DualZonotope(-self.real.centers, self.real.generators, "
             "-self.dual.centers, self.dual.generators)\n"),
    dict(id="M13", func="_max_objective_over_x_candidates",
         why="maximize E instead of |E| at the boundary roots (the upstream bug)",
         old="    evaluation = (sigmoid_prime_y - planar).abs()\n",
         new="    evaluation = sigmoid_prime_y - planar\n"),
    dict(id="M14", func="check_nonlinear_boundary",
         why="maximize E instead of |E| on the ly edge of the original path (the upstream bug)",
         old="        evaluation = objective_fn(ABCs[i][0], ABCs[i][1], ABCs[i][2], "
             "roots_paired_stacked[i]).abs()\n",
         new="        evaluation = objective_fn(ABCs[i][0], ABCs[i][1], ABCs[i][2], "
             "roots_paired_stacked[i])\n"),
    dict(id="M15", func="planar_fit._fit_targets", file=PLANAR_FIT,
         why="slope A with the wrong sign",
         old="    a0 = torch.where(xdeg, zero, (st.dxn * (px / sx[:, None])).sum(1)\n",
         new="    a0 = torch.where(xdeg, zero, -(st.dxn * (px / sx[:, None])).sum(1)\n"),
    dict(id="M16", func="planar_fit._min_norm", file=PLANAR_FIT,
         why="the intercept is not split on a degenerate axis (C = k)",
         old="    C = km * inv / denom\n",
         new="    C = k\n"),
    dict(id="M17", func="planar_fit._centred", file=PLANAR_FIT,
         why="centre on the computed mean instead of a data point",
         old="    r = v - v[:, :1]\n    rm = r.mean(dim=1, keepdim=True)\n    return r - rm, v[:, 0] + rm[:, 0]\n",
         new="    m = v.mean(dim=1, keepdim=True)\n    return v - m, m[:, 0]\n"),
    dict(id="M18", func="planar_fit.gelsd_cutoff_axes", file=PLANAR_FIT,
         why="cutoff compared with eigenvalues without squaring rcond",
         old="    r2 = rcond * rcond\n",
         new="    r2 = rcond\n"),
    dict(id="M19", func="batched_splits.precise_sigmoid", file=BATCHED,
         why="switch inverted: closed form used exactly when it is off",
         old="    fit = planar_fit.fit_planes if USE_CLOSED_FORM_FIT else pt.lin_reg_tensor_batched\n",
         new="    fit = planar_fit.fit_planes if not USE_CLOSED_FORM_FIT else pt.lin_reg_tensor_batched\n"),
    dict(id="M21", func="planar_fit._fit_targets", file=PLANAR_FIT,
         why="x slope from the targets centred on their computed mean (the bug fixed by "
             "pivot-centring z): depends on the order of summation on narrow boxes",
         old="    px = (z3 - z3[:, :1, :]).mean(2)  # [n, s], one value per x\n"
             "    px = px - px.mean(1, keepdim=True)\n",
         new="    px = (z3 - zbar[:, None, None]).mean(2)  # [n, s], one value per x\n"),
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


def run_one_test(root, rel_path, args=()):
    """Run one test script in the temporary root; returns (exit code, last output line)."""
    path = os.path.join(root, rel_path)
    try:
        p = subprocess.run([sys.executable, path, *args], cwd=os.path.dirname(path),
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
        futs = {name: ex.submit(run_one_test, root, rel, *extra) for name, rel, *extra in TESTS}
        return {name: f.result() for name, f in futs.items()}


def show_diff(m, original):
    """Print the mutant as a diff."""
    target = m.get("file", TARGET)
    mutated = original.replace(m["old"], m["new"])
    diff = difflib.unified_diff(original.splitlines(True), mutated.splitlines(True),
                                f"a/{target}", f"b/{target}", n=2)
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
        originals = {}
        for target in {m.get("file", TARGET) for m in mutants}:
            with open(os.path.join(root, target), encoding="utf-8") as f:
                originals[target] = f.read()

        print("=== baseline: all tests on the unmodified copy ===")
        base = run_all_tests(root)
        for name, (code, tail) in base.items():
            print(f"  {name:10s} exit={code}  {tail}")
        if any(code != 0 for code, _ in base.values()):
            print("Baseline is not green, so mutation results would mean nothing. Stopping.")
            return 2

        matrix = {}
        for m in mutants:
            target = m.get("file", TARGET)
            target_path = os.path.join(root, target)
            original = originals[target]
            if original.count(m["old"]) != 1:
                print(f"\n{m['id']}: text in {target} found {original.count(m['old'])} times, expected 1")
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

        names = [n for n, *_ in TESTS]
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
