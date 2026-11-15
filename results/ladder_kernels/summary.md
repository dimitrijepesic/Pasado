# Speedup ladder

Commit `1c74a3579c9d1d90778349681eea8453d68cf265`; CPU Intel(R) Xeon(R) CPU @ 2.20GHz (6 torch threads); device NVIDIA A100-SXM4-40GB; torch 2.14.1+cu130; float64; seed 7; 3 images, eps indices [0, 8]; median of up to 5 timed passes per (image, epsilon), mean over the pairs; steps interleaved. Step 0 is the current code with all selectors off (upstream code + the soundness fixes). `*` = fewer repeats or pairs than planned (the pass was too long for the budget); speedups use the pairs both steps measured. The first pass of a step carries its compilation.

## 5layer, 64 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 batched_G | cuda | 216.97 | - | - | 238.6613 | - | 1.0002 | 90.8 | 0.6 | 4.6 | 304 | 0.61 |
| 6 a2_eager | cuda | 33.18 | - | 6.54 | 238.6613 | - | 1.0002 | 47.5 | 3.5 | 26.4 | 304 | 0.10 |
| 7 a2_compiled_fit | cuda | 18.28 | - | 1.82 | 238.6613 | - | 1.0002 | 6.0 | 5.8 | 48.0 | 304 | 15.10 |
| 8 compiled_sigmoid | cuda | 4.61 | - | 3.97 | 238.6613 | - | 1.0002 | - | 23.0 | - | 210 | 14.38 |

## big, 64 splits

Times and bounds over the 6 (image, epsilon) pairs every step measured (of 6).

| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original | bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 batched_G | cuda | 2295.92 | - | - | 234.3814 | - | 1.0002 | 84.1 | 9.6 | 1.2 | 14916 | 2.42 |
| 6 a2_eager | cuda | 341.21 | - | 6.73 | 234.3814 | - | 1.0002 | 11.8 | 51.7 | 6.6 | 14916 | 0.35 |
| 7 a2_compiled_fit | cuda | 307.15 | - | 1.11 | 234.3814 | - | 1.0002 | 2.0 | 54.5 | 8.0 | 14916 | 1.31 |
| 8 compiled_sigmoid | cuda | 237.48 | - | 1.29 | 234.3814 | - | 1.0002 | - | 75.8 | - | 10288 | 7.31 |

Percentages come from one instrumented pass per step (device synchronized around every stage), not from the timed passes; inside the compiled sigmoid the stages cannot be separated. Full profiles: `profiles/`.
