"""Tables of results/lirpa_postfix_2026-11/PROTOCOL.md from the JSON lines written by
experiments/lirpa_same_gpu.py.

  python experiments/analyze_lirpa_postfix.py DIR [--out-md tables.md]

Only (network, eps index, image) records present for both tools are used, so the two sides
always cover the same images. Times are per image and epsilon, measured on the device in
the records (one device per directory is enforced).
"""
import argparse
import json
import os
import statistics
from collections import defaultdict

EPS = [10 ** (-k / 4) * 2 for k in range(2, 18)]


def load(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out[(r["network"], r["eps_index"], r["image_id"])] = r
    return out


def crossover(ks, ratios):
    """Linearly interpolated K at which the ratio falls to 1 (None if it never does)."""
    for (k0, r0), (k1, r1) in zip(zip(ks, ratios), zip(ks[1:], ratios[1:])):
        if r0 >= 1 > r1:
            return k0 + (r0 - 1) * (k1 - k0) / (r0 - r1)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    lirpa, pasado = load(os.path.join(args.dir, "lirpa.jsonl")), load(os.path.join(args.dir, "pasado.jsonl"))
    keys = sorted(set(lirpa) & set(pasado))
    if not keys:
        raise SystemExit("no (network, eps, image) present for both tools")
    gpus = {lirpa[k]["env"].get("gpu", lirpa[k]["env"]["device"]) for k in keys} | \
           {pasado[k]["env"].get("gpu", pasado[k]["env"]["device"]) for k in keys}
    if len(gpus) != 1:
        raise SystemExit(f"records from more than one device: {gpus}")
    device = gpus.pop()
    commits = {pasado[k]["env"].get("pasado_commit", "") for k in keys}
    groups = defaultdict(list)
    for k in keys:
        groups[k[:2]].append(k)
    unsound = [k for k in keys if not (lirpa[k]["sound_vs_sampled"] and pasado[k]["sound_vs_sampled"])]

    lines = [f"Device: {device}; Pasado commit(s): {', '.join(sorted(commits)) or 'not recorded'}; "
             f"post-fix code; float64; values are means over the images listed; "
             f"Pasado time = median of 3 passes, auto_LiRPA time = measured cumulative solve time.",
             f"Records below the sampled autograd maximum: {len(unsound)} of {2 * len(keys)}.", ""]

    # 1 + 2: single pass and equal budget
    lines += ["| network | eps | images | K=1 | K=3 | K=5 | K=7 | K=9 | crossover K |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (net, ei), ks in sorted(groups.items()):
        budgets = sorted({int(b) for k in ks for b in lirpa[k]["curve"]})
        budgets = [b for b in budgets if all(str(b) in pasado[k]["splits"] and str(b) in lirpa[k]["curve"] for k in ks)]
        ratios = [statistics.mean(lirpa[k]["curve"][str(b)]["bound"] for k in ks) /
                  statistics.mean(pasado[k]["splits"][str(b)]["bound"] for k in ks) for b in budgets]
        cells = {b: f"{r:.3f}" for b, r in zip(budgets, ratios)}
        cx = crossover(budgets, ratios) if len(budgets) > 1 else None
        cx_txt = f"~{cx:.1f}" if cx else (f">{budgets[-1]}" if len(budgets) > 1 else "-")
        lines.append(f"| {net} | {EPS[ei]:.3g} | {len(ks)} | " +
                     " | ".join(cells.get(b, "-") for b in (1, 3, 5, 7, 9)) + f" | {cx_txt} |")
    lines += ["", "Ratio = mean auto_LiRPA bound / mean Pasado bound at the same number of "
              "subproblems (> 1: Pasado tighter).", ""]

    # 3: equal time
    lines += ["| network | eps | auto_LiRPA K=max | its time (s) | smallest Pasado k at least as tight | "
              "its time (s) | time ratio | tightest Pasado k within that time | its bound | tightness ratio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (net, ei), ks in sorted(groups.items()):
        kmax = max(int(b) for b in lirpa[ks[0]]["curve"])
        l_bound = statistics.mean(lirpa[k]["curve"][str(kmax)]["bound"] for k in ks)
        l_time = statistics.mean(lirpa[k]["curve"][str(kmax)]["seconds"] for k in ks)
        p = {int(s): (statistics.mean(pasado[k]["splits"][s]["bound"] for k in ks),
                      statistics.mean(pasado[k]["splits"][s]["seconds"] for k in ks))
             for s in pasado[ks[0]]["splits"]}
        tight = [s for s in sorted(p) if p[s][0] <= l_bound]
        within = [s for s in sorted(p) if p[s][1] <= l_time]
        a = f"{tight[0]} | {p[tight[0]][1]:.3f} | {l_time / p[tight[0]][1]:.0f}x" if tight else "none measured | - | -"
        b = (f"{within[-1]} | {p[within[-1]][0]:.2f} | {l_bound / p[within[-1]][0]:.3f}"
             if within else "none | - | -")
        lines.append(f"| {net} | {EPS[ei]:.3g} | {l_bound:.2f} (K={kmax}) | {l_time:.2f} | {a} | {b} |")
    text = "\n".join(lines)
    print(text)
    if args.out_md:
        open(args.out_md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
