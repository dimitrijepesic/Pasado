"""Generate dual-axis plots of bound reduction vs. runtime across --n-splits.

For each network configuration (3layer, 4layer, 5layer, big), plots:
  - Left Y-axis:  normalized bound lc_precise(n) / lc_precise(1), fixed at 1.0 for n=1
  - Right Y-axis: execution time (ms) as a function of n_splits
"""
import argparse
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary_by_network(csv_path, eps_index=0):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["epsilon_index"]) == eps_index:
                rows.append(row)

    networks = ["3layer", "4layer", "5layer", "big"]
    data = {}
    for net in networks:
        net_rows = [r for r in rows if r["network"] == net]
        if not net_rows:
            continue
        splits = sorted({int(r["n_splits"]) for r in net_rows})
        net_summary = []
        for s in splits:
            sub = [r for r in net_rows if int(r["n_splits"]) == s]
            mean_lc = statistics.mean(float(r["lc_precise"]) for r in sub)
            mean_lc1 = statistics.mean(float(r["lc_split1"]) for r in sub)
            mean_t = statistics.mean(float(r["time_median_ms"]) for r in sub)
            mean_t1 = statistics.mean(float(r["time_split1_median_ms"]) for r in sub)
            net_summary.append(
                {
                    "n_splits": s,
                    "mean_lc": mean_lc,
                    "bound_ratio": mean_lc / mean_lc1,
                    "time_ms": mean_t,
                    "time_mult": mean_t / mean_t1,
                    "per_split_ms": mean_t / s,
                }
            )
        data[net] = net_summary
    return data


def make_plots(cuda_csv, cpu64_csv, out_path_64, out_path_16):
    cuda_data = load_summary_by_network(cuda_csv, eps_index=0)
    cpu64_data = load_summary_by_network(cpu64_csv, eps_index=0)

    color_bound = "#1f77b4"
    color_time = "#d62728"

    # ------------------------------------------------------------------
    # Plot 1: Cleanest possible 2x2 grid (n_splits = 1..64, measured runtime + normalized bound)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160)
    axes = axes.flatten()

    for idx, net in enumerate(["3layer", "4layer", "5layer", "big"]):
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        c64 = cpu64_data[net]
        splits = [d["n_splits"] for d in c64]
        ratios = [d["bound_ratio"] for d in c64]
        times = [d["time_ms"] for d in c64]

        line1 = ax1.plot(
            splits,
            ratios,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color_bound,
            label="Normalized bound",
        )
        line2 = ax2.plot(
            splits,
            times,
            marker="s",
            linewidth=2.0,
            markersize=5,
            color=color_time,
            label="Runtime (ms)",
        )

        ax1.set_title(net, fontsize=12, fontweight="bold")
        ax1.set_xlabel("n_splits", fontsize=10)
        ax1.set_ylabel("Normalized bound", color=color_bound, fontsize=10)
        ax2.set_ylabel("Runtime (ms)", color=color_time, fontsize=10)

        ax1.set_ylim(0.0, 1.05)
        ax1.set_xlim(0, 66)
        ax1.set_xticks([1, 8, 16, 32, 64])
        ax1.tick_params(axis="y", labelcolor=color_bound)
        ax2.tick_params(axis="y", labelcolor=color_time)
        ax2.set_ylim(0.0, max(times) * 1.1)
        ax1.grid(True, linestyle=":", alpha=0.5)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path_16, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Same minimal style with CUDA time (1..16 measured, 32..64 scaled)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160)
    axes = axes.flatten()

    for idx, net in enumerate(["3layer", "4layer", "5layer", "big"]):
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        c64 = cpu64_data[net]
        cu = cuda_data[net]
        cu_by_s = {d["n_splits"]: d for d in cu}
        cuda_ms_per_split = cu_by_s[16]["per_split_ms"]

        splits = [d["n_splits"] for d in c64]
        ratios = [d["bound_ratio"] for d in c64]
        times = [
            cu_by_s[s]["time_ms"] if s in cu_by_s else s * cuda_ms_per_split
            for s in splits
        ]

        line1 = ax1.plot(
            splits,
            ratios,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color_bound,
            label="Normalized bound",
        )
        line2 = ax2.plot(
            splits,
            times,
            marker="s",
            linewidth=2.0,
            markersize=5,
            color=color_time,
            label="Runtime (ms)",
        )

        ax1.set_title(net, fontsize=12, fontweight="bold")
        ax1.set_xlabel("n_splits", fontsize=10)
        ax1.set_ylabel("Normalized bound", color=color_bound, fontsize=10)
        ax2.set_ylabel("Runtime (ms)", color=color_time, fontsize=10)

        ax1.set_ylim(0.0, 1.05)
        ax1.set_xlim(0, 66)
        ax1.set_xticks([1, 8, 16, 32, 64])
        ax1.tick_params(axis="y", labelcolor=color_bound)
        ax2.tick_params(axis="y", labelcolor=color_time)
        ax2.set_ylim(0.0, max(times) * 1.1)
        ax1.grid(True, linestyle=":", alpha=0.5)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path_64, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cuda-csv",
        default="results/gpu_hybrid_split_sweep_2026-10-17/split_sweep_cuda_fp64_to64.csv",
    )
    ap.add_argument(
        "--cpu64-csv",
        default="results/gpu_hybrid_split_sweep_2026-10-17/split_sweep_mac_cpu_fp64_to64.csv",
    )
    ap.add_argument(
        "--out-cuda",
        default="results/gpu_hybrid_split_sweep_2026-10-17/split_sweep_dual_axis_cuda_to64.png",
    )
    ap.add_argument(
        "--out-measured64",
        default="results/gpu_hybrid_split_sweep_2026-10-17/split_sweep_dual_axis_measured_to64.png",
    )
    args = ap.parse_args()
    make_plots(args.cuda_csv, args.cpu64_csv, args.out_cuda, args.out_measured64)
    print(f"Saved {args.out_cuda} and {args.out_measured64}")


if __name__ == "__main__":
    main()
