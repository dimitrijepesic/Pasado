# Speedup ladder on one machine (A100-SXM4-40GB, Colab, 2026-11-15)

Every optimization step from the original Pasado to the compiled sigmoid, timed and profiled on one machine. Steps 0–3 run on that machine's CPU and steps 4–8 on its GPU. Produced by [`experiments/measure_speedup_ladder.py`](../../experiments/measure_speedup_ladder.py).

- **Measured at commit:** `17ad4b0`. The report was regenerated with `4dcb3a1`, which fixes the set of pairs used for the step-8 row (only big with 64 splits was affected).
- **Machine:** Intel Xeon @ 2.20 GHz (6 torch threads), NVIDIA A100-SXM4-40GB, torch 2.14.1+cu130, CUDA 13.0.
- **Protocol:** float64, seed 7. 3 images × ε indices {0, 8}. Median of up to 5 timed passes per (image, ε), after 2 warmup passes. Steps run interleaved, with the device synchronized at the timer boundaries.
- **Files:** [`environment.json`](environment.json), [`summary.md`](summary.md), [`summary.csv`](summary.csv), [`passes.csv`](passes.csv), [`cells/`](cells/), [`profiles/`](profiles/), and [`run.log`](run.log).

## Total speedup over the original (×)

| network | 1 split | 8 splits | 64 splits |
|---|---:|---:|---:|
| 3layer | 34.2 | 254.4 | 1,786 |
| 4layer | 40.2 | 297.8 | 1,941 |
| 5layer | 44.6 | 319.1 | 1,976 |
| FFNNBig | 206.6 | 401.2 | 536.7 |

## Factor of each step over the previous one, 64 splits

| step | small networks | FFNNBig |
|---|---:|---:|
| 1 vectorized checks | 2.05–2.08 | 1.90 |
| 2 batched `gelsd` + batched grid | 4.25–4.37 | 2.94 |
| 3 real Viète cubic | 1.16 | 1.02 |
| 4 GPU hybrid (`gelsd` stays on CPU) | **0.65–0.73** | 5.56 |
| 5 splits batched as [S, G, N] | 7.2–7.8 | 1.79 |
| 6 closed-form fit (A2), eager | 6.0–6.4 | 6.72 |
| 7 compiled fit | 1.75–1.78 | 1.13 |
| 8 whole sigmoid compiled | 3.1–3.3 | 1.28 |

How the total splits between the two devices:

- **CPU, steps 1–3:** 10.1–10.6× on the small networks and 5.5× on FFNNBig.
- **GPU, steps 4–8 on top of step 3:** 169–195× on the small networks and 97× on FFNNBig.

## Soundness

- **Bound vs. sampled maximum.** Every step, cell and pair gives bound / sampled max ≥ 1. The sampled maximum is an autograd |d logit / dt| over 4,001 points of the haze path. The smallest ratio is 1.0000 (3layer, 64 splits) and the largest 1.0208 (5layer, 1 split).
- **Steps 1–2 vs. the original.** The bounds agree with the original to rounding error, at most 6.3e-15 relative.
- **From step 3 on.** The real cubic shifts the bound by at most 1.1e-4 relative, always upward. Steps 4–8 agree with step 3.

## Observations

- **The GPU alone loses on the small networks.**
  - Step 4 is slower than CPU step 3 (0.64–0.73×), because the per-neuron work is too small to fill the device and `gelsd` still runs on the host.
  - The gain comes from batching the splits (step 5) and moving the fit to the GPU (step 6).
- **A2 has a crossover.**
  - At 1 split, step 6 is slower than step 5 (0.66–0.67×) on the small networks, which is why `PASADO_AUTO_PLANE_FIT` picks by size.
  - With 8 or more splits, A2 wins everywhere.
- **On FFNNBig the bottleneck moves to the linear layers.**
  - After compilation the affine layers take 66% (8 splits) and 76% (64 splits) of an instrumented pass. The compiled sigmoid adds only 1.28× there.
  - The next target is the affine layers, not the sigmoid.
- **The dispatch count falls by about 2,600×.** These are aten operators per pass from the torch profiler, for FFNNBig with 64 splits:

  | step | aten operators per pass |
  |---|---:|
  | 5 batched G | 1,054,706 |
  | 6 A2 eager | 9,764 |
  | 7 compiled fit | 5,204 |
  | 8 compiled sigmoid | 404 |

  CUDA kernels per pass, from a profiling rerun on `1c74a35` (this run's profiler recorded no device events, see below):

  | step | 5layer, 64 splits | FFNNBig, 64 splits |
  |---|---:|---:|
  | 5 batched G | 946 | 1,908 |
  | 6 A2 eager | 2,198 | 4,412 |
  | 7 compiled fit | 966 | 1,948 |
  | 8 compiled sigmoid | 110 | 224 |

  - **A2 adds kernels.** Step 6 launches more kernels than step 5 because the fit moves from host `gelsd` to the GPU.
  - **Agreement with v3.** On 40 GB, FFNNBig's 64 splits run as two chunks of 32. Per chunk that is about 2,206 → 112, close to the v3 run's 2,229 → 103 with one chunk on 80 GB.
  - **Why `cuda_kernels` is 0 in this run.** This run profiled CPU-only steps first. After a CPU-only profile, later profiles in the same process recorded no device events, so `cuda_kernels` is 0 and is not a measurement. [`experiments/diag_profiler_cuda.py`](../../experiments/diag_profiler_cuda.py) isolates this; `1c74a35` fixes it.
- **Compilation costs time before it pays off.**
  - The first compiled pass costs 3.3–6.5 s on 3layer and about 19 s on FFNNBig, with the Inductor cache already warm from the probe run.
  - Against A2 eager that pays back after 150–1,000 passes, depending on the cell.
  - The compiled graph is reused across the later cells, so their first-pass times (≈0 s) and break-even values (≈0.2 passes) do not count compilation.
- **VRAM peaks at 14.9 GiB.** That is batched G and A2 on FFNNBig with 64 splits. The compiled sigmoid needs 10.3 GiB.

## Second session (same machine type, fresh Colab runtime)

`418752f`, 5layer and FFNNBig at 1 and 64 splits, all steps, empty Inductor cache: [`../ladder_a100_40gb_rep2/`](../ladder_a100_40gb_rep2/summary.md). The kernel rerun (`1c74a35`, [`../ladder_kernels/`](../ladder_kernels/summary.md)) adds a third measurement of steps 5–8 at 64 splits.

- **Bounds** are identical to this run.
- **Steps 0–7** were 0–11% faster in the second session, on CPU and GPU alike (median 5%). The step ratios hardly move.
- **The compiled sigmoid (step 8) depends on what was compiled before it in the process.**

  | step 8 | first run | second session | kernel rerun |
  |---|---:|---:|---:|
  | 5layer, 64 splits | 5.82 ms | 4.70 ms | 4.61 ms |
  | FFNNBig, 1 split | 9.74 ms | 10.89 ms | – |
  | FFNNBig, 64 splits | 238.69 ms | 231.44 ms | 237.48 ms |

  - In this run 5layer reused a graph compiled earlier for 3layer.
  - When 5layer's graph is compiled for its own shapes, the time reproduces within 2%.
  - Total speedups at 64 splits: 5layer 1,976× here and 2,365× in the second session; FFNNBig 537× and 547×.
- **Compilation with an empty cache** takes 13–19 s, against 3.3–3.5 s warm. Against A2 eager it pays back after (second session; the kernel rerun had a warm Triton cache from it):

  | cell | step 7 (passes) | step 8 (passes) |
  |---|---:|---:|
  | 5layer, 1 split | about 1,030 | about 500 |
  | 5layer, 64 splits | about 930 | about 640 |
  | FFNNBig, 1 split | – | about 830 |
  | FFNNBig, 64 splits | – | about 170 |

## Limits

- **FFNNBig, 64 splits.** Steps 0–1 measured 2 of the 6 pairs, with 1 pass each (about 128 s and 67 s per pass), and steps 2–3 measured 1 pass per pair. Speedups in that cell use the pairs both steps measured.
- **Other reduced cells.** 4layer and 5layer step 0 at 64 splits, and FFNNBig steps 0–1 at 8 splits, ran fewer repeats. These are marked `*` in [`summary.md`](summary.md).
- **One machine only.** The numbers hold for this CPU, this GPU and this torch build. A faster host CPU would shrink the GPU-over-CPU factors.
- **What the percentages are.** Stage percentages come from one instrumented pass with a synchronize around every stage, not from the timed passes.
