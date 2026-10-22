# Split sweep after the soundness fixes - Mac CPU, 2026-10-22

Post-fix rerun of the CPU split sweeps from `../gpu_hybrid_split_sweep_2026-10-17/`,
on the commit that adds this directory. It contains the two fixes described in
section 5 of that README: `Zonotope.__neg__` / `DualZonotope.__neg__` negate the
generators, and the boundary checks maximize `|E|` instead of `E`.

## Protocol

- `experiments/measure_split_sweep.py`, `--device cpu --dtype float64`, real Viète
  cubic solver, all optimization selectors on, `seed = 7`.
- 3 correctly classified MNIST test images (`image_ids = [0, 1, 2]`), 1 warm-up pass
  and 3 timed repeats per `(image, epsilon, n_splits)`.
- Apple Silicon Mac, `torch 2.12.1`, 5 CPU threads.
- `split_sweep_mac_cpu_fp64.{csv,log}`: `n_splits = 1, 2, 4, 8, 16`, epsilon indices `0, 8, 15`.
- `split_sweep_mac_cpu_fp64_to64.{csv,log}`: `n_splits = 1, ..., 64`, epsilon index `0`.
- `soundness_check.{csv,log}`: `experiments/check_sweep_soundness.py` on both CSVs.
  For each row it samples `max |d logit / dt|` along the haze path with autograd
  (4001 points) and compares it with `lc_precise`.

## Soundness

All 264 rows satisfy `lc_precise >= sampled maximum`; the smallest ratio is
`1.000041`. The same check on the pre-fix CSVs finds 13 rows below the sampled
maximum in `split_sweep_cuda_fp64.csv` and 6 in `split_sweep_cpu_fp64.csv`.

## Pre-fix vs. post-fix (`eps_idx = 0`, $\varepsilon = 0.632456$, mean over 3 images)

Pre-fix values are from `../gpu_hybrid_split_sweep_2026-10-17/split_sweep_mac_cpu_fp64_to64.csv`.

| Network | `n=1` pre | `n=1` post | `n=64` pre | `n=64` post | Sampled max | Ratio `n=1 / n=64` pre | Ratio `n=1 / n=64` post |
|---|---:|---:|---:|---:|---:|---:|---:|
| `3layer` | `591.04` | `492.18` | `194.85` | `195.17` | `188.17` | `3.03x` | `2.52x` |
| `4layer` | `1750.56` | `1213.33` | `247.45` | `248.24` | `229.59` | `7.07x` | `4.89x` |
| `5layer` | `4113.32` | `2602.14` | `271.00` | `270.21` | `230.96` | `15.18x` | `9.63x` |
| `big` | `14191.72` | `9927.37` | `254.12` | `250.44` | `234.31` | `55.85x` | `39.64x` |

The negation fix makes the hazed input a single segment instead of a wider wrong
one, which lowers the bound at `n_splits = 1` (by 17-37%). The `|E|` fix adds the
missed negative errors, which raises it slightly at fine splits. The net effect is
smaller tightening ratios than reported before the fixes; the bound at 64 splits
stays 4-17% above the sampled maximum.

## Not re-run

CUDA results (split sweep, layerwise divergence, fallback timing and the
2026-10-08 verification matrix) still come from the pre-fix code and need a
Colab rerun on the fixed commit.
