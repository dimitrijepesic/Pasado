# Crossover of gelsd and the eager closed-form plane fit on the A100 (2026-10-28)

Colab, NVIDIA A100-SXM4-80GB, torch 2.11.0+cu130, CUDA 13.0, code at `8837f68`. The
measurement and the rule for the threshold were committed before the run, in the header of
`experiments/measure_plane_fit_crossover.py`. The other files of this folder are the raw
output, unchanged.

G is `lin_reg_tensor_batched` (gelsd on the CPU, transfers included), A2 is
`planar_fit.fit_planes` (eager, v3 code). Same well-conditioned boxes for both; 3 rounds,
each with 5 warm-up and 20 timed calls per size, solver and dtype.

## Agreement checks (`crossover.log`)

All passed: coefficients within 9.1e-14 (float64) and 7.6e-6 (float32) of gelsd, max
error D within 8.9e-14 and 3.5e-5.

## Speedup of eager A2 over G, median per round (`crossover.csv`)

| n | 64 | 100 | 128 | 200 | 256 | 400 | 512 | 640 | 800 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| float64 | 0.25 | 0.33-0.34 | 0.40-0.41 | 0.54-0.57 | 0.68-0.69 | 0.99-1.00 | 1.24-1.29 | 1.52-1.56 | 1.89 | 2.33-2.36 | 4.81-4.88 |
| float32 | 0.21 | 0.28-0.29 | 0.34 | 0.47-0.50 | 0.60-0.61 | 0.89-0.91 | 1.12-1.13 | 1.41-1.44 | 1.71-1.76 | 2.10-2.15 | 4.19-4.29 |

Eager A2 costs about 3.85 ms per call whatever n; gelsd grows by about 9 microseconds per
box (float64).

## Thresholds (`thresholds.json`)

| dtype | per round | stable | threshold |
|---|---|---|---:|
| float64 | 512, 512, 512 | yes | **512** |
| float32 | 640, 640, 640 | yes | **640** |

At and above the threshold A2 stays faster even in the worst case over all rounds (fastest
gelsd call against slowest A2 call): 1.21x (float64), 1.18x (float32).

The expectation before the measurement was about 800 (float64) and 1024 (float32), from a
session in which eager A2 took 5.3 ms per call; with 3.85 ms the crossover is lower.

The number of boxes per call is the number of splits in a pass times the layer width, so
the small networks (width 100) stay below the float64 threshold up to 4 splits per pass and
`big` (width 1024) is above it from 1 split.
