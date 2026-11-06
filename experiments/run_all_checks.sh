#!/bin/sh
# Runs every check that has to pass after a code change, in the order of rule 3 of the
# checkpoint brief: all existing tests, the soundness checker on a freshly computed
# split sweep (264 rows), and the mutation check. Writes one log per step to OUT.
#
#   sh experiments/run_all_checks.sh [OUT]      (default OUT: logs/checks_<UTC time>)
#
# The split sweep is recomputed with the code as it is (sequential, CPU, float64, real
# cubic, seed 7, the image/epsilon/split grid of results/soundness_fix_2026-10-22/) with
# one timed repeat and no warm-up: only the bounds are used. Its bounds are also compared
# with the archived 2026-10-22 sweep, which must give the same values.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="${1:-$ROOT/logs/checks_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
cd "$ROOT" || exit 1
FAIL=0
run() {  # name, then the command
    name="$1"; shift
    start=$(date +%s)
    if "$@" > "$OUT/$name.log" 2>&1; then status=PASS; else status=FAIL; FAIL=1; fi
    echo "$status  $name  ($(( $(date +%s) - start )) s)" | tee -a "$OUT/summary.txt"
}

echo "commit $(git rev-parse HEAD)$(git diff --quiet || echo ' + uncommitted changes')" > "$OUT/summary.txt"
"$PY" -c "import platform, torch; print('python', platform.python_version(), 'torch', torch.__version__, platform.machine())" >> "$OUT/summary.txt"

# 1. Existing tests: the upstream source-level tests ...
for t in forward_mode_non_tensorized_src/tests/*.py reverse_mode_non_tensorized_src/tests/*.py; do
    d=$(dirname "$t"); b=$(basename "$t" .py)
    run "upstream_$(basename "$(dirname "$d")")_$b" sh -c "cd '$d' && PYTHONPATH='$ROOT/$(dirname "$d")' '$PY' '$(basename "$t")'"
done
# ... and every test of the mutation check's list (same scripts, same arguments).
"$PY" - "$ROOT" > "$OUT/test_list.txt" <<'EOF'
import sys, importlib.util
spec = importlib.util.spec_from_file_location("mc", sys.argv[1] + "/experiments/mutation_check.py")
mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
for t in mc.TESTS:
    print(t[0], t[1], " ".join(t[2]) if len(t) > 2 else "")
EOF
while read -r name script args; do
    # shellcheck disable=SC2086
    run "test_$name" "$PY" "$script" $args
done < "$OUT/test_list.txt"

# New in the November checkpoint: the preliminary outward rounding of D (off by default;
# the test also checks that the switch-off path is bit-identical to batched_splits.py of 1d754c9).
run test_outward_d "$PY" experiments/test_outward_d.py

# 2. Soundness checker on a fresh sweep (180 + 84 = 264 rows).
run sweep_fp64 "$PY" experiments/measure_split_sweep.py --device cpu --dtype float64 \
    --networks 3layer 4layer 5layer big --splits 1 2 4 8 16 --num-images 3 \
    --eps-indices 0 8 15 --seed 7 --warmup 0 --repeats 1 --out "$OUT/sweep_fp64.csv"
run sweep_fp64_to64 "$PY" experiments/measure_split_sweep.py --device cpu --dtype float64 \
    --networks 3layer 4layer 5layer big --splits 1 2 4 8 16 32 64 --num-images 3 \
    --eps-indices 0 --seed 7 --warmup 0 --repeats 1 --out "$OUT/sweep_fp64_to64.csv"
run soundness_checker "$PY" experiments/check_sweep_soundness.py \
    "$OUT/sweep_fp64.csv" "$OUT/sweep_fp64_to64.csv" --out "$OUT/soundness_check.csv"
run sweep_vs_2026-10-22 "$PY" - "$OUT" <<'EOF'
import csv, sys
out = sys.argv[1]
pairs = [("sweep_fp64.csv", "split_sweep_mac_cpu_fp64.csv"),
         ("sweep_fp64_to64.csv", "split_sweep_mac_cpu_fp64_to64.csv")]
worst, n = 0.0, 0
for new, old in pairs:
    key = lambda r: (r["network"], r["image_id"], r["epsilon_index"], r["n_splits"])
    a = {key(r): float(r["lc_precise"]) for r in csv.DictReader(open(f"{out}/{new}"))}
    b = {key(r): float(r["lc_precise"]) for r in csv.DictReader(open(f"results/soundness_fix_2026-10-22/{old}"))}
    assert a.keys() == b.keys(), (new, len(a), len(b))
    for k in a:
        worst = max(worst, abs(a[k] - b[k]) / abs(b[k])); n += 1
print(f"{n} rows, largest relative difference to the 2026-10-22 sweep: {worst:.3e}")
sys.exit(0 if worst <= 1e-12 else 1)
EOF

# 3. Mutation check (every mutant must be caught by at least one test).
run mutation_check "$PY" experiments/mutation_check.py

# Rule 2 of the checkpoint brief: the reference .pth files must be untouched.
run reference_pth_unchanged git diff --quiet HEAD -- Section_5_4/results/

echo "overall: $([ $FAIL = 0 ] && echo ALL PASS || echo FAILURES)" | tee -a "$OUT/summary.txt"
exit $FAIL
