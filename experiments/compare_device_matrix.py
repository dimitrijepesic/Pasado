"""Compares two halves of the device matrix written by postfix_device_matrix.py
(section D of results/postfix_rerun_2026-11/PROTOCOL.md).

  python experiments/compare_device_matrix.py --cpu DIR_CPU --device DIR_CUDA --out summary.csv

For every network and configuration present on both sides it reports the largest relative
difference over the images (per image) and of the 30-image averages (the quantity of the
2026-10-08 matrix), per analysis (precise, zonotope, interval), and checks them against
1e-9 (float64) and 1e-5 (float32). On the device side it also checks that the two repeats
of precise_f64_real are identical and that precise-only equals the precise column of the
full analysis. The complex64 configuration is a negative control: reported, not judged.
Exit status 1 if a judged check fails.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

LIMITS = {"float64": 1e-9, "float32": 1e-5}
COLUMNS = ("lc_precise", "lc_zono", "lc_interval")


def load(path):
    rows = {}
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            rows[(int(r["eps_index"]), int(r["image_id"]))] = r
    return rows


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == b else float("inf"))


ALLOW_SUBSET = False


def compare(a_rows, b_rows, column):
    """(largest per-image difference, largest difference of per-eps averages, n images).

    a_rows is the device side. With --allow-subset it may cover a subset of b_rows."""
    if a_rows.keys() != b_rows.keys():
        if not (ALLOW_SUBSET and set(a_rows) <= set(b_rows)):
            raise ValueError(f"different (eps, image) sets: {len(a_rows)} vs {len(b_rows)}")
        b_rows = {k: b_rows[k] for k in a_rows}
    if a_rows[next(iter(a_rows))][column] in ("", None):
        return None
    per_image = max(rel(float(a_rows[k][column]), float(b_rows[k][column])) for k in a_rows)
    by_eps = defaultdict(lambda: [0.0, 0.0, 0])
    for k in a_rows:
        acc = by_eps[k[0]]
        acc[0] += float(a_rows[k][column]); acc[1] += float(b_rows[k][column]); acc[2] += 1
    per_avg = max(rel(a / n, b / n) for a, b, n in by_eps.values())
    images = {k[1] for k in a_rows}
    return per_image, per_avg, len(images)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-subset", action="store_true",
                    help="the device side may cover a subset of the CPU (eps, image) pairs")
    args = ap.parse_args()
    global ALLOW_SUBSET
    ALLOW_SUBSET = args.allow_subset

    results, failed = [], False
    for path in sorted(glob.glob(os.path.join(args.device, "*.csv"))):
        name = os.path.basename(path)[:-4]
        if name.endswith("_repeat_2"):
            continue
        base = name.replace("_repeat_1", "")
        network, config = base.split("_", 1)
        dtype = "float32" if "_f32_" in config else "float64"
        cpu_path = os.path.join(args.cpu, base + ".csv")
        if not os.path.exists(cpu_path):
            # precise_f64_real is only run on the device; its CPU side is the full run
            cpu_path = os.path.join(args.cpu, f"{network}_full_f64_real.csv")
            if config != "precise_f64_real" or not os.path.exists(cpu_path):
                continue
        dev_rows, cpu_rows = load(path), load(cpu_path)
        for column in COLUMNS:
            got = compare(dev_rows, cpu_rows, column)
            if got is None:
                continue
            per_image, per_avg, n = got
            judged = config != "precise_f64_complex"
            ok = (max(per_image, per_avg) <= LIMITS[dtype]) if judged else None
            failed |= ok is False
            results.append(dict(check="cpu_vs_device", network=network, config=name,
                                column=column, dtype=dtype, images=n,
                                max_rel_per_image=per_image, max_rel_of_average=per_avg,
                                limit=LIMITS[dtype] if judged else "", passed=ok))
    # device-only checks
    for p1 in sorted(glob.glob(os.path.join(args.device, "*_precise_f64_real_repeat_1.csv"))):
        network = os.path.basename(p1).split("_", 1)[0]
        p2 = p1.replace("_repeat_1", "_repeat_2")
        full = os.path.join(args.device, f"{network}_full_f64_real.csv")
        r1 = load(p1)
        for label, other in (("repeat_identical", p2), ("full_vs_precise_only_identical", full)):
            if not os.path.exists(other):
                continue
            r2 = load(other)
            same = r1.keys() == r2.keys() and all(
                float(r1[k]["lc_precise"]) == float(r2[k]["lc_precise"]) for k in r1)
            worst = max(rel(float(r1[k]["lc_precise"]), float(r2[k]["lc_precise"])) for k in r1)
            failed |= not same
            results.append(dict(check=label, network=network, config="precise_f64_real",
                                column="lc_precise", dtype="float64", images=len({k[1] for k in r1}),
                                max_rel_per_image=worst, max_rel_of_average="", limit=0.0,
                                passed=same))
    for r in results:
        flag = {True: "PASS", False: "FAIL", None: "info"}[r["passed"]]
        avg = r["max_rel_of_average"]
        avg = f"{avg:.3e}" if isinstance(avg, float) else "-"
        print(f"{flag:4}  {r['check']:31} {r['network']:6} {r['config']:30} {r['column']:11} "
              f"per-image {r['max_rel_per_image']:.3e}  of-average {avg}")
    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
    print("ALL JUDGED CHECKS PASS" if not failed else "SOME CHECKS FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
