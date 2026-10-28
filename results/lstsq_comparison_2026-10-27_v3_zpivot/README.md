# Comparison of GPU replacements for gelsd, run v3 (2026-10-27)

Same pre-registered rule as v1 and v2 (`385397e`, self-test of `67f4654`), same seeds and
box families, on the CPU (Apple M5 Pro, torch 2.12.1). Code at `500ae16`: the candidates
A1, A2 and H (and `planar_fit.py`) now pivot-centre the targets z along each axis before
forming the slope sums. A3, G and P are unchanged.

## Why v3

Compiling `planar_fit.fit_planes` with TorchInductor showed that the result depended on
the order of summation on narrow boxes: the slopes came from z centred on its computed
mean, so a variation of about 1e-13 sat next to an offset of about 0.1 that cancels in
the sum (`experiments/reproduce_target_centring.py`). Eager PyTorch on the CPU added in a
lucky order on many boxes. With z pivot-centred, every box whose rank the fit keeps exact
is within 32 eps (1 + max|x| + max|y|) of the exact solution, eager and compiled; before
the fix 169 (eager) and 247 (compiled) of 1195 checked boxes were outside, by up to
6e11 times. The fix was motivated by this defect, not by the ranking.

## Outcome (`decide.log`)

| | v2 | v3 |
|---|---|---|
| Self-test | 10 of 10 caught | 10 of 10 caught |
| Gates | P fails G0 C2 (CPU gesdd), all others pass | the same |
| R1 (180 network cases) | tie, 1.97e-13 | tie, 1.97e-13 |
| R2a | G, A2, H 1.00; A1, A3 1.25 | the same |
| R3 | G 7; A1, A2, H -7; A3 -4 | the same |
| R4 (A100, float64, n = 65536) | measured on 2026-10-27 for the v2 code | A2 3.83 ms, H 3.89 ms: same 20% step |
| Result | A2 (with the A100 timing; R5) | **A2 (R5, no host sync)** |

The A2 code changed in v3, so R4 was measured again on 2026-10-28 on an A100-SXM4-40GB
(`timing_cuda.csv`, `timing_cuda_meta.json`, `timing_level.log`, copied from
`../v3_compiled_cuda_2026-10-28/`; code `832323a`, whose `fit_planes` and candidates are
those of `500ae16`). `decide.log` applies the rule with it: **Winner: A2 (decided by R5
simplicity)**. `decide_before_cuda.log` is the earlier output without the timing
(undecided at R4).

## What the fix changes in the numbers

R3 uses medians and does not move. The worst boxes do (A2, distance to the exact solution
on the boxes where gelsd keeps the exact rank, `box_summary.csv`):

| family | v2 median / max | v3 median / max |
|---|---|---|
| cutoff_band_x, float64 | 2.1e-16 / 4.7e-4 | 1.2e-16 / 9.7e-16 |
| cutoff_band_y, float64 | 5.5e-16 / 2.5e-4 | 1.7e-16 / 2.9e-15 |
| mixed_scale, float64 | 4.7e-11 / 2.9e-3 | 2.2e-11 / 1.6e-10 |
| cutoff_band_x, float32 | 1.0e-7 / 4.3e-4 | 5.6e-8 / 3.6e-7 |
| cutoff_band_y, float32 | 4.1e-7 / 2.1e-4 | 1.1e-7 / 2.5e-6 |
| real, float64 | 1.5e-16 / 1.5e-14 | 1.4e-16 / 1.6e-14 |

The float32 real-box maximum of 1.0 in both runs is the one box at s3 / s1 = 0.9975 rcond
where A2's analytic cutoff truncates and gelsd does not (see the v2 README).

## Files

Same layout as v1 and v2. To apply the rule again:
`gzip -dk box_per_box.csv.gz`, then `--level decide --out` this folder.
