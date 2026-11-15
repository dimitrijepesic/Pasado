# Speedup ladder

Commit `17ad4b00526f0b879c8c0da5e37e1ea6bd8649cc`; CPU Intel(R) Xeon(R) CPU @ 2.20GHz (6 torch threads); device NVIDIA A100-SXM4-40GB; torch 2.14.1+cu130; float64; seed 7; 3 images, eps indices [0, 8]; median of up to 5 timed passes per (image, epsilon), mean over the pairs; steps interleaved. Step 0 is the current code with all selectors off (upstream code + the soundness fixes). `*` = fewer repeats or pairs than planned (the pass was too long for the budget); speedups use the pairs both steps measured. The first pass of a step carries its compilation.

## 3layer, 1 split

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 88.72 | 1.00 | - | 328.9501 | 0.0e+00 | 1.0056 | 20.6 | 0.7 | 54.3 | 0 | 0.10 |
| 1 vectorized | cpu | 43.51 | 2.04 | 2.04 | 328.9501 | 9.5e-16 | 1.0056 | 41.0 | 1.1 | 8.8 | 0 | 0.05 |
| 2 batched_lstsq | cpu | 10.40 | 8.53 | 4.18 | 328.9501 | 9.5e-16 | 1.0056 | 18.0 | 4.7 | 34.8 | 0 | 0.01 |
| 3 real_cubic | cpu | 8.78 | 10.10 | 1.18 | 328.9523 | 2.1e-05 | 1.0056 | 20.3 | 5.4 | 25.2 | 0 | 0.01 |
| 4 gpu_hybrid | cuda | 13.67 | 6.49 | 0.64 | 328.9523 | 2.1e-05 | 1.0056 | 15.9 | 4.6 | 32.7 | 11 | 0.39 |
| 5 batched_G | cuda | 11.21 | 7.91 | 1.22 | 328.9523 | 2.1e-05 | 1.0056 | 18.3 | 5.2 | 37.1 | 11 | 0.01 |
| 6 a2_eager | cuda | 16.74 | 5.30 | 0.67 | 328.9523 | 2.1e-05 | 1.0056 | 42.2 | 4.0 | 25.3 | 11 | 0.09 |
| 7 a2_compiled_fit | cuda | 9.57 | 9.27 | 1.75 | 328.9523 | 2.1e-05 | 1.0056 | 8.2 | 6.2 | 41.8 | 11 | 3.34 |
| 8 compiled_sigmoid | cuda | 2.60 | 34.15 | 3.68 | 328.9523 | 2.1e-05 | 1.0056 | - | 18.4 | - | 10 | 3.48 |

## 3layer, 8 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 706.86 | 1.00 | - | 246.2278 | 0.0e+00 | 1.0003 | 20.7 | 0.7 | 54.0 | 0 | 0.72 |
| 1 vectorized | cpu | 340.95 | 2.07 | 2.07 | 246.2278 | 3.2e-16 | 1.0003 | 41.5 | 1.0 | 8.8 | 0 | 0.37 |
| 2 batched_lstsq | cpu | 77.41 | 9.13 | 4.40 | 246.2278 | 3.2e-16 | 1.0003 | 19.1 | 3.7 | 36.4 | 0 | 0.08 |
| 3 real_cubic | cpu | 66.02 | 10.71 | 1.17 | 246.2279 | 1.3e-06 | 1.0003 | 22.1 | 4.4 | 26.4 | 0 | 0.07 |
| 4 gpu_hybrid | cuda | 103.03 | 6.86 | 0.64 | 246.2279 | 1.3e-06 | 1.0003 | 16.9 | 3.7 | 34.0 | 11 | 0.10 |
| 5 batched_G | cuda | 22.27 | 31.73 | 4.63 | 246.2279 | 1.3e-06 | 1.0003 | 55.3 | 3.5 | 21.3 | 24 | 0.03 |
| 6 a2_eager | cuda | 17.07 | 41.40 | 1.30 | 246.2279 | 1.3e-06 | 1.0003 | 43.8 | 4.0 | 25.6 | 24 | 0.02 |
| 7 a2_compiled_fit | cuda | 9.84 | 71.84 | 1.74 | 246.2279 | 1.3e-06 | 1.0003 | 9.1 | 6.1 | 41.7 | 24 | 1.11 |
| 8 compiled_sigmoid | cuda | 2.78 | 254.39 | 3.54 | 246.2279 | 1.3e-06 | 1.0003 | - | 18.3 | - | 19 | 6.55 |

## 3layer, 64 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 5658.46 | 1.00 | - | 179.4439 | 0.0e+00 | 1.0000 | 20.6 | 0.7 | 54.0 | 0 | 5.65 |
| 1 vectorized | cpu | 2715.84 | 2.08 | 2.08 | 179.4439 | 0.0e+00 | 1.0000 | 41.5 | 0.9 | 8.8 | 0 | 2.90 |
| 2 batched_lstsq | cpu | 622.03 | 9.10 | 4.37 | 179.4439 | 0.0e+00 | 1.0000 | 19.5 | 4.1 | 36.0 | 0 | 0.61 |
| 3 real_cubic | cpu | 536.22 | 10.55 | 1.16 | 179.4439 | 2.6e-07 | 1.0000 | 22.4 | 4.3 | 26.6 | 0 | 0.52 |
| 4 gpu_hybrid | cuda | 819.50 | 6.90 | 0.65 | 179.4439 | 2.6e-07 | 1.0000 | 17.0 | 3.7 | 34.2 | 11 | 0.82 |
| 5 batched_G | cuda | 104.68 | 54.05 | 7.83 | 179.4439 | 2.6e-07 | 1.0000 | 88.6 | 0.9 | 5.1 | 123 | 0.11 |
| 6 a2_eager | cuda | 17.35 | 326.08 | 6.03 | 179.4439 | 2.6e-07 | 1.0000 | 41.1 | 4.7 | 24.1 | 123 | 0.02 |
| 7 a2_compiled_fit | cuda | 9.93 | 569.69 | 1.75 | 179.4439 | 2.6e-07 | 1.0000 | 9.8 | 6.2 | 40.0 | 123 | 0.01 |
| 8 compiled_sigmoid | cuda | 3.17 | 1786.04 | 3.14 | 179.4439 | 2.6e-07 | 1.0000 | - | 17.7 | - | 90 | 0.00 |

## 4layer, 1 split

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 136.02 | 1.00 | - | 719.6251 | 0.0e+00 | 1.0131 | 21.5 | 0.6 | 52.9 | 0 | 0.14 |
| 1 vectorized | cpu | 66.35 | 2.05 | 2.05 | 719.6251 | 1.4e-15 | 1.0131 | 41.5 | 1.3 | 8.7 | 0 | 0.07 |
| 2 batched_lstsq | cpu | 15.52 | 8.76 | 4.27 | 719.6251 | 1.4e-15 | 1.0131 | 18.5 | 4.5 | 34.5 | 0 | 0.02 |
| 3 real_cubic | cpu | 13.10 | 10.38 | 1.19 | 719.6345 | 7.3e-05 | 1.0131 | 21.3 | 5.7 | 25.3 | 0 | 0.01 |
| 4 gpu_hybrid | cuda | 19.60 | 6.94 | 0.67 | 719.6345 | 7.3e-05 | 1.0131 | 16.3 | 3.8 | 33.9 | 12 | 0.02 |
| 5 batched_G | cuda | 16.59 | 8.20 | 1.18 | 719.6345 | 7.3e-05 | 1.0131 | 19.8 | 4.8 | 39.5 | 12 | 0.02 |
| 6 a2_eager | cuda | 24.96 | 5.45 | 0.66 | 719.6345 | 7.3e-05 | 1.0131 | 44.0 | 3.4 | 27.5 | 12 | 0.03 |
| 7 a2_compiled_fit | cuda | 14.09 | 9.66 | 1.77 | 719.6345 | 7.3e-05 | 1.0131 | 9.1 | 6.2 | 42.9 | 12 | 0.01 |
| 8 compiled_sigmoid | cuda | 3.38 | 40.21 | 4.16 | 719.6345 | 7.3e-05 | 1.0131 | - | 20.6 | - | 11 | 0.00 |

## 4layer, 8 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 1060.01 | 1.00 | - | 429.1495 | 0.0e+00 | 1.0004 | 20.9 | 0.6 | 54.0 | 0 | 1.07 |
| 1 vectorized | cpu | 512.77 | 2.07 | 2.07 | 429.1495 | 1.8e-15 | 1.0004 | 42.2 | 1.1 | 8.7 | 0 | 0.52 |
| 2 batched_lstsq | cpu | 117.55 | 9.02 | 4.36 | 429.1495 | 1.8e-15 | 1.0004 | 19.1 | 4.1 | 36.5 | 0 | 0.12 |
| 3 real_cubic | cpu | 100.90 | 10.51 | 1.16 | 429.1500 | 1.4e-05 | 1.0004 | 22.3 | 5.2 | 25.7 | 0 | 0.10 |
| 4 gpu_hybrid | cuda | 149.72 | 7.08 | 0.67 | 429.1500 | 1.4e-05 | 1.0004 | 17.7 | 3.6 | 34.8 | 12 | 0.15 |
| 5 batched_G | cuda | 33.40 | 31.74 | 4.48 | 429.1500 | 1.4e-05 | 1.0004 | 53.3 | 3.8 | 20.4 | 34 | 0.03 |
| 6 a2_eager | cuda | 25.13 | 42.18 | 1.33 | 429.1500 | 1.4e-05 | 1.0004 | 44.4 | 4.2 | 25.8 | 34 | 0.02 |
| 7 a2_compiled_fit | cuda | 14.30 | 74.14 | 1.76 | 429.1500 | 1.4e-05 | 1.0004 | 9.3 | 6.3 | 41.9 | 34 | 0.01 |
| 8 compiled_sigmoid | cuda | 3.56 | 297.83 | 4.02 | 429.1500 | 1.4e-05 | 1.0004 | - | 18.7 | - | 26 | 0.00 |

## 4layer, 64 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 8595.08* | 1.00 | - | 233.8387 | 0.0e+00 | 1.0001 | 20.8 | 0.7 | 53.4 | 0 | 8.69 |
| 1 vectorized | cpu | 4167.60 | 2.06 | 2.06 | 233.8387 | 2.4e-15 | 1.0001 | 42.1 | 1.1 | 8.8 | 0 | 4.24 |
| 2 batched_lstsq | cpu | 957.63 | 8.98 | 4.35 | 233.8387 | 2.4e-15 | 1.0001 | 19.7 | 4.1 | 35.9 | 0 | 0.96 |
| 3 real_cubic | cpu | 824.49 | 10.42 | 1.16 | 233.8387 | 1.3e-06 | 1.0001 | 22.8 | 4.7 | 26.0 | 0 | 0.82 |
| 4 gpu_hybrid | cuda | 1188.85 | 7.23 | 0.69 | 233.8387 | 1.3e-06 | 1.0001 | 18.0 | 3.4 | 35.3 | 12 | 1.21 |
| 5 batched_G | cuda | 160.47 | 53.56 | 7.41 | 233.8387 | 1.3e-06 | 1.0001 | 88.4 | 0.9 | 5.1 | 214 | 0.17 |
| 6 a2_eager | cuda | 25.44 | 337.85 | 6.31 | 233.8387 | 1.3e-06 | 1.0001 | 44.5 | 4.6 | 25.6 | 214 | 0.03 |
| 7 a2_compiled_fit | cuda | 14.41 | 596.34 | 1.77 | 233.8387 | 1.3e-06 | 1.0001 | 9.9 | 6.7 | 43.1 | 214 | 0.01 |
| 8 compiled_sigmoid | cuda | 4.43 | 1940.64 | 3.25 | 233.8387 | 1.3e-06 | 1.0001 | - | 18.8 | - | 150 | 0.00 |

## 5layer, 1 split

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 181.96 | 1.00 | - | 1412.1143 | 0.0e+00 | 1.0208 | 21.2 | 0.7 | 53.0 | 0 | 0.18 |
| 1 vectorized | cpu | 89.24 | 2.04 | 2.04 | 1412.1143 | 4.1e-15 | 1.0208 | 42.3 | 1.3 | 8.7 | 0 | 0.09 |
| 2 batched_lstsq | cpu | 21.96 | 8.29 | 4.06 | 1412.1143 | 4.1e-15 | 1.0208 | 21.7 | 5.0 | 31.6 | 0 | 0.02 |
| 3 real_cubic | cpu | 18.89 | 9.63 | 1.16 | 1412.1285 | 1.1e-04 | 1.0208 | 24.1 | 5.5 | 24.1 | 0 | 0.02 |
| 4 gpu_hybrid | cuda | 25.73 | 7.07 | 0.73 | 1412.1285 | 1.1e-04 | 1.0208 | 18.0 | 3.5 | 34.8 | 13 | 0.03 |
| 5 batched_G | cuda | 22.02 | 8.26 | 1.17 | 1412.1285 | 1.1e-04 | 1.0208 | 20.2 | 4.3 | 39.6 | 14 | 0.02 |
| 6 a2_eager | cuda | 33.10 | 5.50 | 0.67 | 1412.1285 | 1.1e-04 | 1.0208 | 46.0 | 3.0 | 26.9 | 14 | 0.03 |
| 7 a2_compiled_fit | cuda | 18.47 | 9.85 | 1.79 | 1412.1285 | 1.1e-04 | 1.0208 | 8.0 | 5.0 | 45.5 | 14 | 0.02 |
| 8 compiled_sigmoid | cuda | 4.08 | 44.56 | 4.52 | 1412.1285 | 1.1e-04 | 1.0208 | - | 18.2 | - | 12 | 0.00 |

## 5layer, 8 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 1425.72 | 1.00 | - | 713.3234 | 0.0e+00 | 1.0009 | 21.1 | 0.7 | 53.8 | 0 | 1.42 |
| 1 vectorized | cpu | 695.82 | 2.05 | 2.05 | 713.3234 | 6.3e-15 | 1.0009 | 42.3 | 1.2 | 8.8 | 0 | 0.70 |
| 2 batched_lstsq | cpu | 163.35 | 8.73 | 4.26 | 713.3234 | 6.3e-15 | 1.0009 | 19.2 | 4.5 | 34.7 | 0 | 0.16 |
| 3 real_cubic | cpu | 140.48 | 10.15 | 1.16 | 713.3235 | 2.5e-06 | 1.0009 | 22.4 | 4.9 | 25.5 | 0 | 0.14 |
| 4 gpu_hybrid | cuda | 196.48 | 7.26 | 0.71 | 713.3235 | 2.5e-06 | 1.0009 | 18.3 | 3.4 | 35.5 | 13 | 0.19 |
| 5 batched_G | cuda | 44.62 | 31.95 | 4.40 | 713.3235 | 2.5e-06 | 1.0009 | 57.4 | 2.9 | 20.8 | 46 | 0.05 |
| 6 a2_eager | cuda | 33.26 | 42.87 | 1.34 | 713.3235 | 2.5e-06 | 1.0009 | 45.7 | 3.5 | 26.4 | 46 | 0.03 |
| 7 a2_compiled_fit | cuda | 18.78 | 75.91 | 1.77 | 713.3235 | 2.5e-06 | 1.0009 | 9.1 | 5.8 | 44.8 | 46 | 0.02 |
| 8 compiled_sigmoid | cuda | 4.47 | 319.13 | 4.20 | 713.3235 | 2.5e-06 | 1.0009 | - | 19.3 | - | 34 | 0.00 |

## 5layer, 64 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 11509.50* | 1.00 | - | 238.6613 | 0.0e+00 | 1.0002 | 21.1 | 0.7 | 53.3 | 0 | 11.40 |
| 1 vectorized | cpu | 5619.61 | 2.05 | 2.05 | 238.6613 | 2.7e-15 | 1.0002 | 41.9 | 1.1 | 8.8 | 0 | 5.79 |
| 2 batched_lstsq | cpu | 1322.47 | 8.70 | 4.25 | 238.6613 | 2.7e-15 | 1.0002 | 19.6 | 4.5 | 34.9 | 0 | 1.33 |
| 3 real_cubic | cpu | 1138.88 | 10.11 | 1.16 | 238.6613 | 3.8e-06 | 1.0002 | 22.6 | 4.9 | 25.4 | 0 | 1.28 |
| 4 gpu_hybrid | cuda | 1563.10 | 7.36 | 0.73 | 238.6613 | 3.8e-06 | 1.0002 | 18.5 | 3.2 | 35.8 | 13 | 1.62 |
| 5 batched_G | cuda | 218.28 | 52.73 | 7.16 | 238.6613 | 3.8e-06 | 1.0002 | 89.5 | 0.9 | 4.9 | 304 | 0.23 |
| 6 a2_eager | cuda | 33.91 | 339.46 | 6.44 | 238.6613 | 3.8e-06 | 1.0002 | 44.1 | 4.7 | 26.2 | 304 | 0.03 |
| 7 a2_compiled_fit | cuda | 19.02 | 604.98 | 1.78 | 238.6613 | 3.8e-06 | 1.0002 | 10.1 | 6.7 | 42.4 | 304 | 0.02 |
| 8 compiled_sigmoid | cuda | 5.82 | 1975.91 | 3.27 | 238.6613 | 3.8e-06 | 1.0002 | - | 18.7 | - | 210 | 0.01 |

## big, 1 split

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 2012.64 | 1.00 | - | 5075.9168 | 0.0e+00 | 1.0095 | 19.4 | 9.5 | 45.5 | 0 | 2.01 |
| 1 vectorized | cpu | 1054.08 | 1.91 | 1.91 | 5075.9168 | 5.6e-16 | 1.0095 | 31.9 | 22.5 | 1.1 | 0 | 1.05 |
| 2 batched_lstsq | cpu | 340.94 | 5.90 | 3.09 | 5075.9168 | 5.6e-16 | 1.0095 | 9.6 | 46.2 | 3.7 | 0 | 0.53 |
| 3 real_cubic | cpu | 346.70 | 5.81 | 0.98 | 5075.9343 | 1.9e-05 | 1.0095 | 9.3 | 48.5 | 2.0 | 0 | 0.52 |
| 4 gpu_hybrid | cuda | 67.68 | 29.74 | 5.12 | 5075.9343 | 1.9e-05 | 1.0095 | 49.1 | 7.1 | 14.6 | 479 | 0.08 |
| 5 batched_G | cuda | 56.72 | 35.48 | 1.19 | 5075.9343 | 1.9e-05 | 1.0095 | 57.7 | 8.1 | 16.9 | 503 | 0.06 |
| 6 a2_eager | cuda | 34.66 | 58.07 | 1.64 | 5075.9343 | 1.9e-05 | 1.0095 | 40.8 | 11.5 | 23.5 | 503 | 0.03 |
| 7 a2_compiled_fit | cuda | 19.47 | 103.36 | 1.78 | 5075.9343 | 1.9e-05 | 1.0095 | 8.0 | 18.0 | 36.5 | 503 | 0.02 |
| 8 compiled_sigmoid | cuda | 9.74 | 206.55 | 2.00 | 5075.9343 | 1.9e-05 | 1.0095 | - | 39.3 | - | 359 | 18.88 |

## big, 8 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 16413.80* | 1.00 | - | 929.7062 | 0.0e+00 | 1.0006 | 19.3 | 8.2 | 46.3 | 0 | 16.22 |
| 1 vectorized | cpu | 8905.40* | 1.84 | 1.84 | 929.7062 | 8.0e-16 | 1.0006 | 35.7 | 16.1 | 1.3 | 0 | 9.33 |
| 2 batched_lstsq | cpu | 2755.65 | 5.96 | 3.23 | 929.7062 | 8.0e-16 | 1.0006 | 9.3 | 48.7 | 3.6 | 0 | 2.74 |
| 3 real_cubic | cpu | 2866.01 | 5.73 | 0.96 | 929.7062 | 3.9e-07 | 1.0006 | 9.7 | 45.6 | 2.3 | 0 | 2.69 |
| 4 gpu_hybrid | cuda | 527.94 | 31.09 | 5.43 | 929.7062 | 3.9e-07 | 1.0006 | 49.6 | 6.9 | 14.7 | 481 | 0.53 |
| 5 batched_G | cuda | 308.80 | 53.15 | 1.71 | 929.7062 | 3.9e-07 | 1.0006 | 79.8 | 9.7 | 3.5 | 3759 | 0.32 |
| 6 a2_eager | cuda | 59.29 | 276.83 | 5.21 | 929.7062 | 3.9e-07 | 1.0006 | 21.3 | 39.3 | 12.5 | 3759 | 0.06 |
| 7 a2_compiled_fit | cuda | 52.31 | 313.78 | 1.13 | 929.7062 | 3.9e-07 | 1.0006 | 4.2 | 41.8 | 17.6 | 3759 | 0.05 |
| 8 compiled_sigmoid | cuda | 40.91 | 401.23 | 1.28 | 929.7062 | 3.9e-07 | 1.0006 | - | 65.9 | - | 2601 | 18.31 |

## big, 64 splits

Times and bounds over the 2 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 original | cpu | 128110.46* | 1.00 | - | 254.8496 | 0.0e+00 | 1.0003 | - | - | - | 0 | 127.01 |
| 1 vectorized | cpu | 67491.35* | 1.90 | 1.90 | 254.8496 | 4.6e-16 | 1.0003 | - | - | - | 0 | 67.27 |
| 2 batched_lstsq | cpu | 22985.47* | 5.57 | 2.94 | 254.8496 | 4.6e-16 | 1.0002 | 9.3 | 47.6 | 3.7 | 0 | 23.12 |
| 3 real_cubic | cpu | 23080.85* | 5.55 | 1.02 | 254.8496 | 2.9e-09 | 1.0002 | 9.2 | 48.0 | 2.1 | 0 | 22.54 |
| 4 gpu_hybrid | cuda | 4135.51 | 30.98 | 5.56 | 254.8496 | 2.9e-09 | 1.0002 | 50.5 | 6.8 | 14.5 | 481 | 4.40 |
| 5 batched_G | cuda | 2317.73 | 55.27 | 1.79 | 254.8496 | 2.9e-09 | 1.0002 | 84.2 | 9.8 | 0.9 | 14916 | 2.40 |
| 6 a2_eager | cuda | 349.47 | 366.58 | 6.72 | 254.8496 | 2.9e-09 | 1.0002 | 8.7 | 55.8 | 5.1 | 14916 | 0.35 |
| 7 a2_compiled_fit | cuda | 306.20 | 418.39 | 1.13 | 254.8496 | 2.9e-09 | 1.0002 | 2.4 | 55.4 | 6.2 | 14916 | 0.31 |
| 8 compiled_sigmoid | cuda | 238.69 | 536.73 | 1.28 | 254.8496 | 2.9e-09 | 1.0002 | - | 75.9 | - | 10288 | 0.24 |

Percentages come from one instrumented pass per step (device synchronized around every stage), not from the timed passes; inside the compiled sigmoid the stages cannot be separated. Full profiles: `profiles/`.
