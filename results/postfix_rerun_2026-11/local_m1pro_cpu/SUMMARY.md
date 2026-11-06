**A. Split sweep** *post-fix; commit `1d754c9128` (dirty); GPU no GPU; host CPU Apple M1 Pro; torch 2.12.1, CUDA None; float64; 3 images; median of 3 timed passes after 1 warm-up pass; big batched chunk 16; baseline: n = 1 of the same mode*

Largest relative difference of a batched bound from the sequential bound: 9.39e-15 (criterion 1e-10: pass).
Soundness checker on the sequential rows (4001 sampled points): 84 rows, smallest bound / sampled maximum 1.012815, rows below: 0.

eps index 0 (mean over images; time = median ms per pass; VRAM = peak MiB):

| network | n | bound | bound / bound(n=1) | seq ms | batched_G ms | batched_auto ms | seq VRAM | batched_G VRAM | batched_auto VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3layer | 1 | 492.18 | 1.000 | 2.6 | 2.4 | - | 0 | 0 | - |
| 3layer | 2 | 448.03 | 0.910 | 5.0 | 3.5 | - | 0 | 0 | - |
| 3layer | 4 | 395.29 | 0.803 | 10.1 | 5.0 | - | 0 | 0 | - |
| 3layer | 8 | 328.71 | 0.668 | 20.0 | 8.0 | - | 0 | 0 | - |
| 3layer | 16 | 239.27 | 0.486 | 40.8 | 13.9 | - | 0 | 0 | - |
| 3layer | 32 | 213.66 | 0.434 | 80.4 | 25.3 | - | 0 | 0 | - |
| 3layer | 64 | 195.17 | 0.397 | 160.6 | 43.6 | - | 0 | 0 | - |
| 4layer | 1 | 1213.33 | 1.000 | 4.8 | 3.8 | - | 0 | 0 | - |
| 4layer | 2 | 1055.13 | 0.870 | 9.4 | 5.5 | - | 0 | 0 | - |
| 4layer | 4 | 866.81 | 0.714 | 18.7 | 8.0 | - | 0 | 0 | - |
| 4layer | 8 | 638.75 | 0.526 | 36.7 | 12.7 | - | 0 | 0 | - |
| 4layer | 16 | 403.68 | 0.333 | 74.3 | 22.5 | - | 0 | 0 | - |
| 4layer | 32 | 282.89 | 0.233 | 147.2 | 40.6 | - | 0 | 0 | - |
| 4layer | 64 | 248.24 | 0.205 | 294.3 | 70.5 | - | 0 | 0 | - |
| 5layer | 1 | 2602.14 | 1.000 | 7.0 | 5.3 | - | 0 | 0 | - |
| 5layer | 2 | 2247.83 | 0.864 | 14.0 | 7.6 | - | 0 | 0 | - |
| 5layer | 4 | 1770.43 | 0.680 | 27.5 | 11.5 | - | 0 | 0 | - |
| 5layer | 8 | 1219.27 | 0.469 | 54.6 | 17.8 | - | 0 | 0 | - |
| 5layer | 16 | 723.31 | 0.278 | 109.8 | 32.1 | - | 0 | 0 | - |
| 5layer | 32 | 349.65 | 0.134 | 219.4 | 57.6 | - | 0 | 0 | - |
| 5layer | 64 | 270.21 | 0.104 | 443.9 | 101.0 | - | 0 | 0 | - |
| big | 1 | 9927.37 | 1.000 | 188.8 | 160.6 | - | 0 | 0 | - |
| big | 2 | 6921.13 | 0.697 | 370.2 | 317.9 | - | 0 | 0 | - |
| big | 4 | 3941.03 | 0.397 | 746.6 | 606.7 | - | 0 | 0 | - |
| big | 8 | 1641.00 | 0.165 | 1505.3 | 2203.1 | - | 0 | 0 | - |
| big | 16 | 610.10 | 0.061 | 3001.3 | 4942.4 | - | 0 | 0 | - |
| big | 32 | 309.29 | 0.031 | 5997.5 | 9618.3 | - | 0 | 0 | - |
| big | 64 | 250.44 | 0.025 | 11929.5 | 19665.0 | - | 0 | 0 | - |


**B. Share of the CPU part: not measured.**

**C. Equal time on the same instance: not measured.**

**D. Device matrix: not measured.**

**E. Layerwise CPU/CUDA difference: not measured.**

**F. Splits below one minute per pass: not measured.**

Figures: `figures/postfix_local_m1pro_cpu_split_sweep_dual_axis.png`
