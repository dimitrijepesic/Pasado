# Speedup ladder

Commit `f89a5d5d01846a3e4824fa048772bb28d0662807`; CPU Apple M1 Pro (8 torch threads); device cpu; torch 2.12.1; float64; seed 7; 3 images, eps indices [0, 8]; median of up to 5 timed passes per (image, epsilon), mean over the pairs; steps interleaved. Step 0 is the current code with all selectors off (upstream code + the soundness fixes). `*` = fewer repeats or pairs than planned (the pass was too long for the budget); speedups use the pairs both steps measured. The first pass of a step carries its compilation.

## 5layer, 1 split

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 53.55 | 1.00 | - | 1412.1143 | 0.0e+00 | 1.0208 | - | - | - | 0 | 0.06 |
| 1 vectorized | cpu | 25.83 | 2.07 | 2.07 | 1412.1143 | 2.2e-15 | 1.0208 | - | - | - | 0 | 0.03 |
| 2 batched_lstsq | cpu | 7.83 | 6.84 | 3.30 | 1412.1143 | 2.2e-15 | 1.0208 | - | - | - | 0 | 0.01 |
| 3 real_cubic | cpu | 7.12 | 7.52 | 1.10 | 1412.1285 | 1.1e-04 | 1.0208 | - | - | - | 0 | 0.01 |
| 5 batched_G | cpu | 5.72 | 9.35 | 1.24 | 1412.1285 | 1.1e-04 | 1.0208 | - | - | - | 0 | 0.01 |
| 6 a2_eager | cpu | 6.30 | 8.50 | 0.91 | 1412.1285 | 1.1e-04 | 1.0208 | - | - | - | 0 | 0.01 |
| 7 a2_compiled_fit | cpu | 4.87 | 11.00 | 1.30 | 1412.1285 | 1.1e-04 | 1.0208 | - | - | - | 0 | 1.24 |
| 8 compiled_sigmoid | cpu | 2.03 | 26.33 | 2.39 | 1412.1285 | 1.1e-04 | 1.0208 | - | - | - | 0 | 0.86 |

## 5layer, 64 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 3242.32 | 1.00 | - | 238.6613 | 0.0e+00 | 1.0002 | - | - | - | 0 | 3.21 |
| 1 vectorized | cpu | 1564.69 | 2.07 | 2.07 | 238.6613 | 4.5e-15 | 1.0002 | - | - | - | 0 | 1.51 |
| 2 batched_lstsq | cpu | 462.91 | 7.00 | 3.38 | 238.6613 | 4.5e-15 | 1.0002 | - | - | - | 0 | 0.46 |
| 3 real_cubic | cpu | 428.16 | 7.57 | 1.08 | 238.6613 | 3.8e-06 | 1.0002 | - | - | - | 0 | 0.42 |
| 5 batched_G | cpu | 99.55 | 32.57 | 4.30 | 238.6613 | 3.8e-06 | 1.0002 | - | - | - | 0 | 0.16 |
| 6 a2_eager | cpu | 47.46 | 68.32 | 2.10 | 238.6613 | 3.8e-06 | 1.0002 | - | - | - | 0 | 0.05 |
| 7 a2_compiled_fit | cpu | 42.79 | 75.78 | 1.11 | 238.6613 | 3.8e-06 | 1.0002 | - | - | - | 0 | 0.28 |
| 8 compiled_sigmoid | cpu | 31.38 | 103.31 | 1.36 | 238.6613 | 3.8e-06 | 1.0002 | - | - | - | 0 | 1.67 |

## big, 1 split

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 644.29 | 1.00 | - | 5075.9168 | 0.0e+00 | 1.0095 | - | - | - | 0 | 0.97 |
| 1 vectorized | cpu | 367.16 | 1.75 | 1.75 | 5075.9168 | 2.2e-16 | 1.0095 | - | - | - | 0 | 0.36 |
| 2 batched_lstsq | cpu | 187.71 | 3.43 | 1.96 | 5075.9168 | 2.2e-16 | 1.0095 | - | - | - | 0 | 0.18 |
| 3 real_cubic | cpu | 186.32 | 3.46 | 1.01 | 5075.9343 | 1.9e-05 | 1.0095 | - | - | - | 0 | 0.18 |
| 5 batched_G | cpu | 145.50 | 4.43 | 1.28 | 5075.9343 | 1.9e-05 | 1.0095 | - | - | - | 0 | 0.14 |
| 6 a2_eager | cpu | 136.22 | 4.73 | 1.07 | 5075.9343 | 1.9e-05 | 1.0095 | - | - | - | 0 | 0.13 |
| 7 a2_compiled_fit | cpu | 134.53 | 4.79 | 1.01 | 5075.9343 | 1.9e-05 | 1.0095 | - | - | - | 0 | 0.15 |
| 8 compiled_sigmoid | cpu | 135.20 | 4.77 | 1.00 | 5075.9343 | 1.9e-05 | 1.0095 | - | - | - | 0 | 1.82 |

## big, 64 splits

Times and bounds over the 2 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 42975.88* | 1.00 | - | 254.8496 | 0.0e+00 | 1.0003 | - | - | - | 0 | 42.86 |
| 1 vectorized | cpu | 23782.63* | 1.81 | 1.81 | 254.8496 | 8.0e-16 | 1.0002 | - | - | - | 0 | 23.03 |
| 2 batched_lstsq | cpu | 12251.66* | 3.51 | 1.98 | 254.8496 | 8.0e-16 | 1.0002 | - | - | - | 0 | 11.23 |
| 3 real_cubic | cpu | 12208.60* | 3.52 | 1.01 | 254.8496 | 2.9e-09 | 1.0002 | - | - | - | 0 | 11.37 |
| 5 batched_G | cpu | 14895.64* | 2.89 | 0.81 | 254.8496 | 2.9e-09 | 1.0002 | - | - | - | 0 | 20.32 |
| 6 a2_eager | cpu | 8322.75* | 5.16 | 1.21 | 254.8496 | 2.9e-09 | 1.0002 | - | - | - | 0 | 18.82 |
| 7 a2_compiled_fit | cpu | 11288.10* | 3.81 | 0.96 | 254.8496 | 2.9e-09 | 1.0002 | - | - | - | 0 | 17.61 |
| 8 compiled_sigmoid | cpu | 7320.11* | 5.87 | 1.67 | 254.8496 | 2.9e-09 | 1.0002 | - | - | - | 0 | 16.33 |

Percentages come from one instrumented pass per step (device synchronized around every stage), not from the timed passes; inside the compiled sigmoid the stages cannot be separated. Full profiles: `profiles/`.
