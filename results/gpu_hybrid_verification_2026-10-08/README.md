# GPU hybrid verification - A100, 2026-10-08

This directory contains the raw output of the final CPU/CUDA correctness
matrix for commit `c2a2c657d9249b084d8fc0a91c67d683adfda86e` on an
NVIDIA A100-SXM4-80GB.

## Files

- `gpu_hybrid_quick.csv` and `.log`: 5 images, epsilon indices 0, 8 and 15.
- `gpu_hybrid_full.csv` and `.log`: 30 images and all 16 epsilon values.

Both CSV files contain 28/28 PASS rows. Both logs terminate with `ALL PASS`.

## Protocol

For each of 3layer, 4layer, 5layer and big, the matrix compares:

1. real-cubic float64 CPU and CUDA for the full precise, zonotope and interval
   analyses;
2. two repeated real-cubic float64 CUDA precise-only runs;
3. full versus precise-only real-cubic CUDA output;
4. real-cubic float32 CPU and CUDA precise output;
5. the obsolete complex solver as an expected-divergence negative control.

The relative limits are `1e-9` for float64 and `1e-5` for float32. The
float32 limit matches the tolerance already used by the float32 regression
tests; applying the float64 `1e-9` limit to float32 was rejected as an invalid
protocol.

## Full-run summary

| Result | Maximum relative difference |
|---|---:|
| real float64 precise | `1.786e-15` |
| real float64 zonotope | `4.792e-16` |
| real float64 interval | `5.517e-16` |
| real float32 precise | `8.318e-7` |
| repeated CUDA run | `0` |
| full versus precise-only CUDA | `0` |

The complex negative control diverges by 14.39% on 3layer, 33.42% on
4layer, 48.72% on 5layer and 63.78% on big. Its PASS rows mean that the
expected divergence was observed; they do not endorse the complex solver.

## Interpretation and limits

The real-cubic float64 implementation is reproducible across CPU and A100 to
approximately machine precision. CUDA is deterministic, and skipping the
plain zonotope and interval analyses does not change the precise result.

These files establish correctness, not acceleration. They contain no
controlled timing data. The 30-image outputs are averages, so a separate
per-image sweep is still needed to report the maximum individual discrepancy.
