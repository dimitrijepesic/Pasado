"""Tables and figures for CHECKPOINT.md from the results of the checkpoint notebooks.

  python experiments/make_checkpoint_report.py --postfix DIR [--lirpa DIR] [--compilers DIR]
      [--label NAME] [--update-checkpoint]

DIR is a results folder of colab/01_postfix_rerun.ipynb (the one holding environment.json,
A_sweep/, B_cpu_share/, ...), copied from Drive; the same layout works for a local run.
Writes figures/<label>_*.png and <DIR>/SUMMARY.md, and with --update-checkpoint replaces the
text between the markers <!-- BEGIN:<label> --> and <!-- END:<label> --> in CHECKPOINT.md.
A section whose files are missing is reported as "not measured"; nothing is estimated.
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETS = ("3layer", "4layer", "5layer", "big")
MODES = ("sequential", "batched_G", "batched_auto")
A_TIMING = "median of 5 timed passes after 2 warm-up passes"  # --a-timing overrides


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def label(env, extra=""):
    gpu = env.get("gpu") or env.get("nvidia_smi") or "no GPU"
    return (f"*post-fix; commit `{env.get('commit', '?')[:10]}`{' (dirty)' if env.get('dirty') else ''}; "
            f"GPU {gpu}; host CPU {env.get('cpu', '?')}; torch {env.get('torch', '?')}, "
            f"CUDA {env.get('cuda', '?')}; float64{extra}*")


def sweep_rows(d):
    out = []
    for path in sorted(glob.glob(os.path.join(d, "A_sweep", "*.csv"))):
        mode = os.path.basename(path)[:-4].split("_", 1)[1]
        for r in rows(path):
            r["mode"] = mode
            out.append(r)
    return out


def section_a(d, env, fig_prefix):
    data = sweep_rows(d)
    if not data:
        return "**A. Split sweep: not measured.**\n", []
    md = ["**A. Split sweep** " + label(env, f"; 3 images; {A_TIMING}; baseline: n = 1 of the same mode")
          + "\n"]
    # agreement batched vs sequential
    seq = {(r["network"], r["image_id"], r["epsilon_index"], r["n_splits"]): float(r["lc_precise"])
           for r in data if r["mode"] == "sequential"}
    worst = 0.0
    for r in data:
        k = (r["network"], r["image_id"], r["epsilon_index"], r["n_splits"])
        if r["mode"] != "sequential" and k in seq:
            worst = max(worst, abs(float(r["lc_precise"]) - seq[k]) / abs(seq[k]))
    md.append(f"Largest relative difference of a batched bound from the sequential bound: "
              f"{worst:.2e} (criterion 1e-10: {'pass' if worst <= 1e-10 else 'FAIL'}).")
    snd = glob.glob(os.path.join(d, "A_soundness.csv"))
    if snd:
        sr = rows(snd[0])
        below = [r for r in sr if r["sound"] != "True"]
        md.append(f"Soundness checker on the sequential rows (4001 sampled points): {len(sr)} rows, "
                  f"smallest bound / sampled maximum {min(float(r['ratio']) for r in sr):.6f}, "
                  f"rows below: {len(below)}.")
    else:
        md.append("Soundness checker on the sequential rows: not run.")
    md.append("")
    for eps_idx in sorted({r["epsilon_index"] for r in data}, key=int):
        md.append(f"eps index {eps_idx} (mean over images; time = median ms per pass; VRAM = peak MiB):\n")
        md.append("| network | n | bound | bound / bound(n=1) | seq ms | batched_G ms | batched_auto ms "
                  "| seq VRAM | batched_G VRAM | batched_auto VRAM |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for net in NETS:
            sub = [r for r in data if r["network"] == net and r["epsilon_index"] == eps_idx]
            if not sub:
                continue
            for n in sorted({int(r["n_splits"]) for r in sub}):
                cell = {}
                for mode in MODES:
                    m = [r for r in sub if r["mode"] == mode and int(r["n_splits"]) == n]
                    if m:
                        cell[mode] = (statistics.mean(float(r["lc_precise"]) for r in m),
                                      statistics.mean(float(r["lc_split1"]) for r in m),
                                      statistics.mean(float(r["time_median_ms"]) for r in m),
                                      max(float(r["peak_vram_mb"]) for r in m))
                ref = cell.get("sequential") or next(iter(cell.values()))
                t = lambda m: f"{cell[m][2]:.1f}" if m in cell else "-"  # noqa: E731
                v = lambda m: f"{cell[m][3]:.0f}" if m in cell else "-"  # noqa: E731
                md.append(f"| {net} | {n} | {ref[0]:.2f} | {ref[0] / ref[1]:.3f} | {t('sequential')} | "
                          f"{t('batched_G')} | {t('batched_auto')} | {v('sequential')} | {v('batched_G')} | "
                          f"{v('batched_auto')} |")
        md.append("")
    figs = figure_a(data, fig_prefix, env)
    return "\n".join(md) + "\n", figs


def figure_a(data, fig_prefix, env):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    eps0 = min({r["epsilon_index"] for r in data}, key=int)
    nets = [n for n in NETS if any(r["network"] == n for r in data)]
    fig, axes = plt.subplots(1, len(nets), figsize=(4.2 * len(nets), 3.6), squeeze=False)
    styles = {"sequential": ("tab:red", "o"), "batched_G": ("tab:orange", "s"),
              "batched_auto": ("tab:green", "^")}
    for ax, net in zip(axes[0], nets):
        sub = [r for r in data if r["network"] == net and r["epsilon_index"] == eps0]
        ns = sorted({int(r["n_splits"]) for r in sub})
        seq = [r for r in sub if r["mode"] == "sequential"] or sub
        norm = [statistics.mean(float(r["lc_precise"]) / float(r["lc_split1"]) for r in seq
                                if int(r["n_splits"]) == n) for n in ns]
        ax.plot(ns, norm, "k-", marker="o", label="bound / bound(n=1)")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("n_splits")
        ax.set_ylabel("normalised bound")
        ax.set_title(net)
        ax2 = ax.twinx()
        for mode, (color, marker) in styles.items():
            m = [r for r in sub if r["mode"] == mode]
            if not m:
                continue
            ts = [statistics.mean(float(r["time_median_ms"]) for r in m if int(r["n_splits"]) == n) for n in ns]
            ax2.plot(ns, ts, color=color, marker=marker, linestyle="--", label=f"time, {mode}")
        ax2.set_yscale("log")
        ax2.set_ylabel("time per pass (ms)")
        if net == nets[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            fig.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center", ncol=4)
    gpu = env.get("gpu") or "CPU " + str(env.get("cpu", ""))
    fig.suptitle(f"Bound (normalised to n = 1) and time per pass, eps index {eps0}; {gpu}; "
                 f"commit {env.get('commit', '?')[:8]}; post-fix; float64", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = fig_prefix + "split_sweep_dual_axis.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def section_b(d, env):
    files = sorted(glob.glob(os.path.join(d, "B_cpu_share", "*.csv")))
    if not files:
        return "**B. Share of the CPU part: not measured.**\n"
    data = [r for f in files for r in rows(f)]
    md = ["**B. Share of the CPU part** " + label(env, "; 3 images, eps index 0; median over images "
                                                "and 5 repeats; instrumented and plain passes alternate") + "\n",
          "| network | mode | S | pass ms (plain) | pass ms (instrumented) | CPU part ms | share of "
          "instrumented pass | share of plain pass | gelsd ms | copies ms | closed form ms |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for net in NETS:
        for mode in MODES:
            for S in sorted({int(r["n_splits"]) for r in data}):
                m = [r for r in data if r["network"] == net and r["mode"] == mode and int(r["n_splits"]) == S]
                if not m:
                    continue
                med = lambda k: statistics.median(float(r[k]) for r in m)  # noqa: E731
                cpu = med("cpu_part_ms")
                md.append(f"| {net} | {mode} | {S} | {med('pass_ms_uninstrumented'):.2f} | "
                          f"{med('pass_ms_instrumented'):.2f} | {cpu:.2f} | "
                          f"{100 * cpu / med('pass_ms_instrumented'):.1f}% | "
                          f"{100 * cpu / med('pass_ms_uninstrumented'):.1f}% | {med('gelsd_ms'):.2f} | "
                          f"{med('d2h_ms') + med('h2d_ms'):.2f} | {med('closed_form_ms'):.2f} |")
    return "\n".join(md) + "\n"


def section_c(d, env):
    cpu_files = glob.glob(os.path.join(d, "C_cpu_n1.csv"))
    data = sweep_rows(d)
    if not cpu_files or not data:
        return "**C. Equal time on the same instance: not measured.**\n"
    cpu = rows(cpu_files[0])
    md = ["**C. Equal time on the same instance** " + label(env, "; CPU side: the Colab host CPU, "
                                                        "sequential n = 1, median of 3; CUDA side: section A") + "\n",
          "| network | eps index | CPU n=1 ms | CPU n=1 bound | mode | k* (largest n with CUDA time <= CPU "
          "n=1 time) | CUDA ms at k* | bound at k* | CPU bound / bound at k* |",
          "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for net in NETS:
        for ei in sorted({r["epsilon_index"] for r in cpu}, key=int):
            c = [r for r in cpu if r["network"] == net and r["epsilon_index"] == ei]
            if not c:
                continue
            c_ms = statistics.mean(float(r["time_median_ms"]) for r in c)
            c_b = statistics.mean(float(r["lc_precise"]) for r in c)
            for mode in MODES:
                g = [r for r in data if r["network"] == net and r["epsilon_index"] == ei and r["mode"] == mode]
                by_n = defaultdict(list)
                for r in g:
                    by_n[int(r["n_splits"])].append(r)
                fit = [n for n in sorted(by_n) if statistics.mean(float(r["time_median_ms"]) for r in by_n[n]) <= c_ms]
                if not fit:
                    md.append(f"| {net} | {ei} | {c_ms:.1f} | {c_b:.2f} | {mode} | none measured | - | - | - |")
                    continue
                k = fit[-1]
                t = statistics.mean(float(r["time_median_ms"]) for r in by_n[k])
                b = statistics.mean(float(r["lc_precise"]) for r in by_n[k])
                cap = " (largest n measured)" if k == max(by_n) else ""
                md.append(f"| {net} | {ei} | {c_ms:.1f} | {c_b:.2f} | {mode} | {k}{cap} | {t:.1f} | {b:.2f} | "
                          f"{c_b / b:.2f}x |")
    return "\n".join(md) + "\n"


def section_d(d, env):
    f = glob.glob(os.path.join(d, "D_summary.csv"))
    if not f:
        return "**D. Device matrix: not measured.**\n"
    md = ["**D. Device matrix (CPU against CUDA)** " + label(env, "; CPU side: Apple M1 Pro (local "
                                                         "reference) unless noted; 30 images x 16 eps; seed 7") + "\n",
          "| check | network | config | column | per image | of the 30-image average | limit | result |",
          "|---|---|---|---|---:|---:|---:|---|"]
    for r in rows(f[0]):
        res = {"True": "pass", "False": "FAIL", "None": "info", "": "info"}[r["passed"]]
        avg = r["max_rel_of_average"]
        avg = f"{float(avg):.2e}" if avg not in ("", None) else "-"
        md.append(f"| {r['check']} | {r['network']} | {r['config']} | {r['column']} | "
                  f"{float(r['max_rel_per_image']):.2e} | {avg} | {r['limit'] or '-'} | {res} |")
    for extra, title in (("D_spotcheck_summary.csv", "Colab CPU against the local reference (subset)"),
                         ("D_spotcheck_cuda_summary.csv", "CUDA against the Colab CPU (subset)")):
        p = os.path.join(d, extra)
        if os.path.exists(p):
            worst = max(float(r["max_rel_per_image"]) for r in rows(p))
            md.append(f"\n{title}: largest relative difference per image {worst:.2e}.")
    return "\n".join(md) + "\n"


def section_e(d, env):
    f = glob.glob(os.path.join(d, "E_layerwise.csv"))
    if not f:
        return "**E. Layerwise CPU/CUDA difference: not measured.**\n"
    data = rows(f[0])
    md = ["**E. Layerwise CPU/CUDA difference** " + label(env, "; 5 images x 5 eps; largest relative "
                                                      "difference of the layer bound over images and eps") + "\n",
          "| network | solver | layer | bound rel. diff (max) | final bound rel. diff (max) |",
          "|---|---|---|---:|---:|"]
    seen = defaultdict(lambda: [0.0, 0.0])
    order = []
    for r in data:
        k = (r["network"], r["solver"], r["layer"])
        if k not in seen:
            order.append(k)
        seen[k][0] = max(seen[k][0], float(r["bound_rel_diff"] or 0))
        seen[k][1] = max(seen[k][1], float(r["final_lc_rel_diff"] or 0))
    for k in order:
        md.append(f"| {k[0]} | {k[1]} | {k[2]} | {seen[k][0]:.2e} | {seen[k][1]:.2e} |")
    return "\n".join(md) + "\n"


def section_f(d, env):
    files = sorted(glob.glob(os.path.join(d, "F_ramp", "*.csv")))
    md = []
    if not files:
        md.append("**F. Splits below one minute per pass: not measured.**")
    else:
        data = [r for f in files for r in rows(f)]
        md += ["**F. Splits below one minute per pass** " + label(env, "; 1 image, eps index 0; median of "
                                                              "2 passes (1 above 10 s)") + "\n",
               "| network | mode | largest n measured below 60 s | its time (s) | next n measured | its time (s) "
               "| peak VRAM at largest n (MiB) |", "|---|---|---:|---:|---:|---:|---:|"]
        for net in NETS:
            for mode in MODES:
                m = sorted((r for r in data if r["network"] == net and r["mode"] == mode),
                           key=lambda r: int(r["n_splits"]))
                if not m:
                    continue
                below = [r for r in m if r["below_limit"] == "True"]
                above = [r for r in m if r["below_limit"] != "True"]
                b = below[-1] if below else None
                a = above[0] if above else None
                md.append(f"| {net} | {mode} | {b['n_splits'] if b else 'none'} | "
                          f"{float(b['median_s']):.2f} | " if b else f"| {net} | {mode} | none | - | ")
                md[-1] += (f"{a['n_splits']} | {float(a['median_s']):.2f} | " if a else "not reached (max n) | - | ")
                md[-1] += f"{float(b['peak_vram_mb']):.0f} |" if b else "- |"
    full = glob.glob(os.path.join(d, "F_full_run.csv"))
    if full:
        md += ["", "Full standard runs (30 images x 16 eps, timed end to end):", "",
               "| network | n_splits | mode | passes | total s |", "|---|---:|---|---:|---:|"]
        for r in rows(full[0]):
            md.append(f"| {r['network']} | {r['n_splits']} | {r['mode']} | {r['passes']} | {float(r['total_s']):.1f} |")
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--postfix", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--update-checkpoint", action="store_true")
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--a-timing", default=None, help="repetitions of section A, if not the protocol's")
    args = ap.parse_args()
    global A_TIMING
    A_TIMING = args.a_timing or A_TIMING
    d = args.postfix
    env = json.load(open(os.path.join(d, "environment.json"))) if os.path.exists(
        os.path.join(d, "environment.json")) else {}
    name = args.label or os.path.basename(os.path.normpath(d))
    fig_prefix = os.path.join(args.fig_dir, name + "_")
    a_md, figs = section_a(d, env, fig_prefix)
    text = "\n".join([a_md, section_b(d, env), section_c(d, env), section_d(d, env), section_e(d, env),
                      section_f(d, env)])
    if figs:
        text += "\nFigures: " + ", ".join(f"`figures/{os.path.basename(f)}`" for f in figs) + "\n"
    with open(os.path.join(d, "SUMMARY.md"), "w") as handle:
        handle.write(text)
    print(text)
    if args.update_checkpoint:
        path = os.path.join(ROOT, "CHECKPOINT.md")
        doc = open(path).read()
        begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
        if begin not in doc or end not in doc:
            raise SystemExit(f"markers {begin} / {end} not found in CHECKPOINT.md")
        head, rest = doc.split(begin, 1)
        _, tail = rest.split(end, 1)
        open(path, "w").write(head + begin + "\n" + text + "\n" + end + tail)
        print(f"updated {path}")


if __name__ == "__main__":
    main()
