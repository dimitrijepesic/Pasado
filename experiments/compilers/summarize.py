"""Table tool x kernel from the records of run_matrix.py (results/compilers_2026-11/PROTOCOL.md).

  python experiments/compilers/summarize.py DIR [--out-md DIR/table.md]
"""
import argparse
import json
import os


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.dir, "records.jsonl"))]
    by = {}
    for r in recs:
        by[(r["kernel"], r["size"], r["tool"], r["mode"], r["phase"])] = r
    devices = sorted({r.get("gpu") or r.get("versions", {}).get("device") or r["device"] for r in recs})
    commits = sorted({r.get("commit", "")[:10] for r in recs})
    lines = [f"Devices: {', '.join(map(str, devices))}; commit(s): {', '.join(commits) or 'not recorded'}; "
             f"float64 inputs (complex_cubic computes in complex64 by design); steady = median of "
             f"{recs[0].get('repeats')} calls after 10 warm-up calls; post-fix code.", "",
             "| kernel | size | tool / mode | vs eager (normwise) | NaN pattern | cold first call (s) | "
             "warm first call (s) | steady wall ms | steady CUDA-event ms | kernels | graph breaks | "
             "recompiles for 2 new sizes | notes |",
             "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|"]
    keys = sorted({(r["kernel"], r["size"], r["tool"], r["mode"]) for r in recs},
                  key=lambda k: (k[0], k[1], k[2] != "torch", ["eager", "default", "max-autotune",
                                                                 "reduce-overhead", "jit"].index(k[3])))
    for kernel, size, tool, mode in keys:
        cold = by.get((kernel, size, tool, mode, "cold")) or by.get((kernel, size, tool, mode, "none"))
        warm = by.get((kernel, size, tool, mode, "warm"))
        r = warm if warm and not warm.get("error") else cold
        if r is None:
            continue
        if tool == "torch":
            kern = r.get("cuda_kernels_per_call")
            kern = f"{kern} launched" if kern is not None else (
                f"{r.get('generated_kernels')} generated + {r.get('extern_kernel_calls')} extern"
                if r.get("generated_kernels") is not None else "-")
            recomp = r.get("dynamo_unique_graphs_added")
            breaks = r.get("graph_break_count")
        else:
            kern = f"{r.get('hlo_launch_ops')} HLO ops ({r.get('hlo_fusions')} fusions)" if r.get("hlo_launch_ops") else "-"
            recomp = (r.get("jax_compiled_variants") or 1) - 1 if "jax_compiled_variants" in r else None
            breaks = "n/a (traces whole function)" if not r.get("error") else "-"
        warns = (cold or {}).get("warnings", []) + (warm or {}).get("warnings", [])
        notes = r.get("error") or "; ".join(sorted(set(r.get("break_reasons", []) + (cold or {}).get(
            "break_reasons", [])))[:2]) or "; ".join(sorted({w for w in warns if "complex" in w.lower()
                                                               or "fallback" in w.lower()}))[:160]
        lines.append(
            f"| {kernel} | {size} | {tool} {mode} | {fmt(r.get('max_rel_vs_eager'), '.1e')} | "
            f"{'same' if r.get('nan_pattern_equal') else ('differs' if r.get('nan_pattern_equal') is False else '-')} | "
            f"{fmt(cold.get('first_call_s') if cold and mode not in ('eager',) else None, '.2f')} | "
            f"{fmt(warm.get('first_call_s') if warm else None, '.2f')} | {fmt(r.get('steady_wall_ms_median'), '.3f')} | "
            f"{fmt(r.get('steady_cuda_event_ms_median'), '.3f')} | {kern} | {fmt(breaks, '')} | "
            f"{fmt(recomp, '')} | {notes} |")
    text = "\n".join(lines)
    print(text)
    if args.out_md:
        open(args.out_md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
