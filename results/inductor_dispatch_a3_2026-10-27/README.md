# TorchInductor on planar_fit, the crossover with gelsd, and A3 end to end (2026-10-27)

Colab, NVIDIA A100-SXM4-80GB, torch 2.11.0+cu130, CUDA 13.0, code at `7e4cdcd`, notebook
`COLAB_INDUCTOR_DISPATCH_A3.ipynb` of the workspace. Measurements only; no code changed.

## 1. TorchInductor on `fit_planes` (`inductor_fit_planes.csv`)

| variant | CUDA kernels per call | steady time per call | first call | difference to eager |
|---|---:|---:|---:|---:|
| eager | 303 | 5.2 to 5.5 ms | | |
| `torch.compile(fullgraph=True)` | 5 | 0.32 to 0.45 ms | 8.5 s and 11.0 s for the first two shapes, then about 0 | float64 <= 5.5e-14, float32 <= 3.3e-5 |
| same, `mode="reduce-overhead"` (CUDA graph) | 6 | 0.22 to 0.25 ms | 4.3 s and 5.2 s, then about 0.005 s | same |
| `dynamic=True` | | | fails: `Unsupported: torch.* op returned non-Tensor` | |

- Dynamo captures `fit_planes` as one graph without breaks, and Inductor fuses the 303
  kernels of the eager version into 5. The constant cost of about 4 to 5 ms per call was
  kernel launches.
- After the second shape Dynamo marks the batch size as dynamic by itself, so later
  shapes do not recompile; `dynamic=True` is not needed.
- The compiled function is not bit-identical to the eager one (fusion changes the order
  of roundings). The float32 difference of up to 3.3e-5 is in line with the condition of
  boxes down to width 1e-3 times float32 epsilon.
- Eager A2 took 5.2 to 5.5 ms per call here against 3.7 to 4.0 ms in the earlier sessions
  with the same code; presumably a different launch latency of the Colab machine, not
  measured.

## 2. Crossover against gelsd, solver only (`crossover.csv`, `thresholds.json`)

Speedup over G (gelsd on the CPU with transfers), float64:

| n | A2 eager | A2 compiled (reduce-overhead) | A3 eager |
|---:|---:|---:|---:|
| 64 | 0.13x | 4.6x | 0.58x |
| 100 | 0.25x | 5.6x | 0.77x |
| 800 | 1.28x | 30x | 4.0x |
| 1024 | 1.63x | 38x | 5.0x |
| 65536 | 102x | 1211x | 313x |

Thresholds (smallest n with at least 1.2x and faster at every larger n): A2 eager 800
(float64) and 1024 (float32); A3 200 and 256; A2 compiled 64 (the smallest n tested) in
both dtypes. These are steady-state numbers and ignore compilation.

## 3. End to end (`e2e.csv`, `e2e_summary.csv`)

Sequential gelsd, batched gelsd, batched A2 eager, batched A2 compiled (reduce-overhead,
outputs cloned), batched A3; 4 networks, eps index 0 and 8, 1/8/32/64 splits, 3 images,
median of 3 warm passes. Every bound agrees with the sequential loop to 2.13e-13.

| configuration | batched gelsd (ms) | A2 eager | A2 compiled | A3 | compiled vs batched gelsd | compiled vs sequential |
|---|---:|---:|---:|---:|---:|---:|
| 3layer, 1 split (100 boxes per call) | 13.9 | 22.2 | 12.1 | 14.8 | 1.15x | 1.35x |
| 3layer, 8 splits | 24.6 | 22.2 | 12.2 | 15.1 | 2.02x | 10.5x |
| 5layer, 64 splits | 231.5 | 43.8 | 23.3 | 28.8 | 9.95x | 84x |
| big, 1 split | 58.4 | 44.7 | 23.9 | 29.0 | 2.44x | 2.9x |
| big, 32 splits | 1279.0 | 159.1 | 152.5 | 153.4 | 8.39x | 14.7x |
| big, 64 splits | 2498.8 | 292.5 | 285.0 | 285.9 | 8.77x | 15.6x |

(eps index 0; eps index 8 gives the same picture.) In warm steady state the compiled A2
is faster than batched gelsd in all 32 configurations, at least 1.14x. A3 is slower than
the compiled A2 everywhere and still loses to batched gelsd at 1 split on the small
networks (0.85x to 0.94x); it is not needed. On `big` with 32 or more splits eager and
compiled are within 5% of each other: the solver no longer dominates.

## Compilation cost decides the default

The warm numbers leave out the cold start. In `e2e.csv` the first compiled pass took
9.5 s and the second shape another 0.9 s; all other shapes add almost nothing (11.4 s in
total over the 96 configurations). With about 10.4 s of compilation, the passes needed
to win it back against batched gelsd:

| network | 1 split | 8 splits | 32 splits | 64 splits |
|---|---:|---:|---:|---:|
| 3layer | 5795 | 838 | 206 | 105 |
| 4layer | 3864 | 532 | 135 | 69 |
| 5layer | 2815 | 394 | 100 | 50 |
| big | 301 | 40 | 9 | 5 |

A standard run of 30 images and 16 epsilons (480 passes) does not win the compilation
back on the small networks at 1 split, nor on 3layer and 4layer at 8 splits. So:

- in a warm, long-lived process no threshold is needed;
- in a fresh process or a short run the compiled path can be slower overall.

The compiled A2 is therefore not made the default. A safe default for now is gelsd below
the measured threshold, eager A2 above it, and the compiled A2 only on explicit request
or when enough passes are expected. Still to measure: start-up with a warm Inductor disk
cache in a second process, the break-even per n, plain compile against reduce-overhead
with a test of output aliasing, and the checks C1-C7 and `test_batched_splits.py` on the
compiled function.

## Next lever

On the small networks the batched pass now takes about 12 ms whatever the number of
splits, while the solver accounts for 2 x 0.24 ms: the rest of the precise transformer
(corner and boundary checks, the cubic) is launch-bound in the same way. Compiling the
whole `precise_sigmoid` rather than only the fit is the next step, and the comparison
with TVM and XLA can start from there.
