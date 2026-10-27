# Comparison of GPU replacements for gelsd on CUDA (2026-10-27), and the choice

Colab, NVIDIA A100-SXM4-80GB, torch 2.11.0+cu130, CUDA 13.0, code at `7740ba4`
(rule of `385397e`, candidates of `077ab86`, self-test of `67f4654`), notebook
`COLAB_LSTSQ_COMPARISON.ipynb` of the workspace. All levels of the rule ran on the GPU:
timing (R4), box level, network level, decision.

## Choice: A2

The planned protocol ran the box and network levels on the CPU and used the A100 only for
R4; the CUDA box level was an additional check, mainly for P. Both readings are reported.

| Reading | Data | Result |
|---|---|---|
| primary (the planned protocol) | CPU run v2 + `timing_cuda.csv` of this folder | **A2**: A2 and H tie through R4 (both 6 steps of 20% above the fastest), R5 prefers A2 (no host sync); `../lstsq_comparison_2026-10-25_v2_scaled/decide_with_a100_timing.log` |
| sensitivity (everything on CUDA) | this folder | **A3**: `decide.log` |

The rule did not say which box level counts when both a CPU and a CUDA one exist. A2 is
the choice; A3 is reported as a sensitivity result. The rule is not changed and no v3 is
run.

Why the CUDA reading ends at A3:

- R2a: on a few `near_degenerate` float64 boxes P reaches a radius W about 11 times
  smaller than every other candidate, so every candidate except P gets the same R2a of
  11.48 and R2a no longer separates them.
- R2b then splits at the edge of a decade: worst shortfall on `near_degenerate` float32,
  where the boxes collapse to points and W is at rounding level: G 0.044, A3 0.0945,
  A1, A2 and H 0.142, so G and A3 land in bucket -2 and the others in -1. On the CPU the
  same family put G in bucket -1. These numbers come from the known float32 limitation of
  Pasado's max-error path (`experiments/reproduce_f32_root_shortfall.py`), and the
  absolute shortfalls of the candidates are close.
- R3 then prefers A3 over G.

What holds on both devices:

- Every closed form passes G0-G3; R1 ties (largest difference to gelsd over 180 network
  cases: 1.94e-13 on CUDA, 1.97e-13 on the CPU).
- R3: the worst family median distance to the exact solution is about 4e-7 for A1, A2
  and H, about 5e-4 for A3 and about 4e7 for gelsd.
- A2 is the only closed form that keeps gelsd's rank rule (analytically), works on the
  25 production targets and does not assume the separable form sigmoid'(x) * y that A3
  relies on, so it can also replace the regression of other transformers.
- A2 and H give practically the same numbers; A2 has no host sync and no CPU fallback.

## Solver time on the A100 (median ms, `timing_cuda.csv`)

| candidate | n = 1024 | 8192 | 32768 | 65536 (float64) | 65536 (float32) |
|---|---:|---:|---:|---:|---:|
| G (gelsd on CPU, with transfers) | 10.50 | 71.01 | 280.02 | 570.56 | 509.38 |
| P (pinv, gesvdj) | 1.17 | 5.58 | 18.06 | 35.41 | 21.01 |
| A2 | 3.96 | 3.73 | 3.73 | 3.72 | 3.72 |
| H | 3.77 | 3.76 | 3.77 | 3.83 | 3.77 |
| A1 | 1.48 | 1.48 | 1.49 | 1.50 | 1.53 |
| A3 | 1.27 | 1.25 | 1.26 | 1.25 | 1.26 |

At n = 65536 in float64 A2 is 153 times faster than gelsd.

## Whole batched pass on `big` (informational, not part of the rule)

`e2e_big_batched.csv`: one image, eps index 0, seed 7, chunk 64, median of 3.

| n_splits | gelsd (ms) | A2 (ms) | H (ms) | A2 speedup over gelsd |
|---:|---:|---:|---:|---:|
| 1 | 53.7 | 31.4 | 31.5 | 1.71x |
| 8 | 325.2 | 58.5 | 53.2 | 5.56x |
| 32 | 1249.6 | 152.3 | 153.2 | 8.20x |
| 64 | 2410.4 | 285.9 | 287.0 | 8.43x |

Bounds agree with gelsd to 8.2e-14 or better. Against the sequential loop measured on an
A100 on 2026-10-24 (4066.7 ms at 64 splits, `../gpu_hybrid_batched_splits_2026-10-24/`)
A2's 285.9 ms is about 14 times faster; the two numbers come from different sessions.
That run estimated the pass without the regression at 287 ms.

A3 was not run end to end. From the solver times, A3 would save about
4 sigmoid layers x (3.72 - 1.25) ms = 9.9 ms of the 286 ms at 64 splits; this is an
estimate, not a measurement.

## Other observations on CUDA

- P passes G0 C2 on CUDA (cuSOLVER gesvdj gives bit-identical results for a shuffled
  batch), unlike LAPACK gesdd on the CPU. It ranks below the closed forms on R2a and R3.
- Self-test: all 10 planted bugs caught, S4 through accuracy (`g0_selftest.csv`).
- Real boxes: the same 119 904 float64 boxes as on the CPU; gelsd truncates none beyond
  the exact rank.
- On the informational `nonfinite` family gelsd stops with a PyTorch internal assert on
  this torch version (a `LinAlgError` on the CPU).

## Next

A2 goes into the batched path behind a switch that is off by default; gelsd stays the
reference. A2 becomes the default only after the full regression tests pass on the CPU
and on CUDA.

## Files

Same layout as v1, plus `timing_cuda.csv` and `timing_cuda_meta.json` (R4),
`e2e_big_batched.csv`, `colab_environment.json` and `smoke_cuda.log`. To apply the rule
again: `gzip -dk box_per_box.csv.gz`, then `--level decide --out` this folder.
