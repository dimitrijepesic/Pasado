# Speedup ladder

Commit `418752f549187217b8fe340f0730e6937f1ceed1`; CPU Intel(R) Xeon(R) CPU @ 2.20GHz (6 torch threads); device NVIDIA A100-SXM4-40GB; torch 2.14.1+cu130; float64; seed 7; 3 images, eps indices [0, 8]; median of up to 5 timed passes per (image, epsilon), mean over the pairs; steps interleaved. Step 0 is the current code with all selectors off (upstream code + the soundness fixes). `*` = fewer repeats or pairs than planned (the pass was too long for the budget); speedups use the pairs both steps measured. The first pass of a step carries its compilation.

## 5layer, 1 split

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 173.37 | 1.00 | - | 1412.1143 | 0.0e+00 | 1.0208 | 21.8 | 0.5 | 54.0 | 0 | 0.18 |
| 1 vectorized | cpu | 84.28 | 2.06 | 2.06 | 1412.1143 | 4.1e-15 | 1.0208 | 43.7 | 1.4 | 8.1 | 0 | 0.08 |
| 2 batched_lstsq | cpu | 19.57 | 8.86 | 4.31 | 1412.1143 | 4.1e-15 | 1.0208 | 21.4 | 4.9 | 31.6 | 0 | 0.02 |
| 3 real_cubic | cpu | 16.79 | 10.32 | 1.17 | 1412.1285 | 1.1e-04 | 1.0208 | 24.1 | 4.9 | 23.7 | 0 | 0.02 |
| 4 gpu_hybrid | cuda | 24.61 | 7.04 | 0.68 | 1412.1285 | 1.1e-04 | 1.0208 | 17.3 | 3.6 | 34.7 | 13 | 0.40 |
| 5 batched_G | cuda | 21.30 | 8.14 | 1.16 | 1412.1285 | 1.1e-04 | 1.0208 | 19.8 | 4.2 | 39.9 | 14 | 0.02 |
| 6 a2_eager | cuda | 32.17 | 5.39 | 0.66 | 1412.1285 | 1.1e-04 | 1.0208 | 46.2 | 3.1 | 26.4 | 14 | 0.10 |
| 7 a2_compiled_fit | cuda | 17.62 | 9.84 | 1.83 | 1412.1285 | 1.1e-04 | 1.0208 | 8.0 | 4.8 | 46.2 | 14 | 15.03 |
| 8 compiled_sigmoid | cuda | 3.85 | 45.08 | 4.58 | 1412.1285 | 1.1e-04 | 1.0208 | - | 20.6 | - | 12 | 14.02 |

## 5layer, 64 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 11105.67* | 1.00 | - | 238.6613 | 0.0e+00 | 1.0002 | 22.1 | 0.6 | 53.8 | 0 | 11.08 |
| 1 vectorized | cpu | 5395.45 | 2.06 | 2.06 | 238.6613 | 2.7e-15 | 1.0002 | 43.2 | 1.0 | 8.2 | 0 | 5.42 |
| 2 batched_lstsq | cpu | 1248.10 | 8.90 | 4.32 | 238.6613 | 2.7e-15 | 1.0002 | 20.0 | 4.2 | 35.9 | 0 | 1.28 |
| 3 real_cubic | cpu | 1078.62 | 10.30 | 1.16 | 238.6613 | 3.8e-06 | 1.0002 | 23.2 | 4.8 | 26.0 | 0 | 1.18 |
| 4 gpu_hybrid | cuda | 1518.97 | 7.31 | 0.71 | 238.6613 | 3.8e-06 | 1.0002 | 18.3 | 3.0 | 36.3 | 13 | 1.52 |
| 5 batched_G | cuda | 206.72 | 53.72 | 7.35 | 238.6613 | 3.8e-06 | 1.0002 | 89.8 | 0.8 | 4.6 | 303 | 0.22 |
| 6 a2_eager | cuda | 32.56 | 341.04 | 6.35 | 238.6613 | 3.8e-06 | 1.0002 | 44.9 | 4.5 | 26.0 | 303 | 0.03 |
| 7 a2_compiled_fit | cuda | 18.17 | 611.35 | 1.79 | 238.6613 | 3.8e-06 | 1.0002 | 8.1 | 6.8 | 44.1 | 303 | 13.42 |
| 8 compiled_sigmoid | cuda | 4.70 | 2364.95 | 3.87 | 238.6613 | 3.8e-06 | 1.0002 | - | 22.8 | - | 210 | 17.86 |

## big, 1 split

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 1927.93 | 1.00 | - | 5075.9168 | 0.0e+00 | 1.0095 | 19.8 | 7.9 | 47.1 | 0 | 1.94 |
| 1 vectorized | cpu | 989.16 | 1.95 | 1.95 | 5075.9168 | 5.6e-16 | 1.0095 | 36.9 | 17.8 | 1.2 | 0 | 2.29 |
| 2 batched_lstsq | cpu | 311.19 | 6.20 | 3.18 | 5075.9168 | 5.6e-16 | 1.0095 | 6.6 | 56.5 | 2.7 | 0 | 0.31 |
| 3 real_cubic | cpu | 309.65 | 6.23 | 1.00 | 5075.9343 | 1.9e-05 | 1.0095 | 10.6 | 47.8 | 2.4 | 0 | 0.30 |
| 4 gpu_hybrid | cuda | 63.56 | 30.33 | 4.87 | 5075.9343 | 1.9e-05 | 1.0095 | 49.6 | 7.1 | 14.0 | 479 | 0.07 |
| 5 batched_G | cuda | 52.37 | 36.82 | 1.21 | 5075.9343 | 1.9e-05 | 1.0095 | 57.8 | 8.3 | 16.9 | 503 | 0.05 |
| 6 a2_eager | cuda | 32.58 | 59.17 | 1.61 | 5075.9343 | 1.9e-05 | 1.0095 | 40.7 | 12.2 | 23.4 | 503 | 0.03 |
| 7 a2_compiled_fit | cuda | 18.20 | 105.95 | 1.79 | 5075.9343 | 1.9e-05 | 1.0095 | 6.7 | 18.4 | 37.4 | 503 | 0.02 |
| 8 compiled_sigmoid | cuda | 10.89 | 176.97 | 1.67 | 5075.9343 | 1.9e-05 | 1.0095 | - | 35.6 | - | 359 | 18.06 |

## big, 64 splits

Times and bounds over the 2 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 126599.65* | 1.00 | - | 254.8496 | 0.0e+00 | 1.0003 | - | - | - | 0 | 129.64 |
| 1 vectorized | cpu | 67514.97* | 1.88 | 1.88 | 254.8496 | 4.6e-16 | 1.0003 | - | - | - | 0 | 67.39 |
| 2 batched_lstsq | cpu | 21603.54* | 5.86 | 3.13 | 254.8496 | 4.6e-16 | 1.0002 | 9.6 | 49.6 | 3.9 | 0 | 21.07 |
| 3 real_cubic | cpu | 21506.33* | 5.89 | 1.04 | 254.8496 | 2.9e-09 | 1.0002 | 10.2 | 49.7 | 2.3 | 0 | 21.82 |
| 4 gpu_hybrid | cuda | 3914.99 | 32.34 | 5.35 | 254.8496 | 2.9e-09 | 1.0002 | 51.4 | 6.8 | 14.1 | 481 | 3.99 |
| 5 batched_G | cuda | 2272.60 | 55.71 | 1.73 | 254.8496 | 2.9e-09 | 1.0002 | 84.0 | 10.0 | 0.9 | 14916 | 2.35 |
| 6 a2_eager | cuda | 341.36 | 370.87 | 6.59 | 254.8496 | 2.9e-09 | 1.0002 | 8.4 | 56.0 | 4.9 | 14916 | 0.34 |
| 7 a2_compiled_fit | cuda | 301.06 | 420.52 | 1.14 | 254.8496 | 2.9e-09 | 1.0002 | 1.0 | 57.0 | 5.9 | 14916 | 0.30 |
| 8 compiled_sigmoid | cuda | 231.44 | 547.00 | 1.30 | 254.8496 | 2.9e-09 | 1.0002 | - | 78.6 | - | 10288 | 19.39 |

Percentages come from one instrumented pass per step (device synchronized around every stage), not from the timed passes; inside the compiled sigmoid the stages cannot be separated. Full profiles: `profiles/`.
