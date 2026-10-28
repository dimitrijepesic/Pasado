# v3 plane fit and its compiled version on CUDA (2026-10-28)

Colab, notebook `COLAB_V3_COMPILED.ipynb` of the workspace. The files of this folder are
the raw output of that run, unchanged.

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 39.49 GiB |
| torch, CUDA | 2.11.0+cu130, 13.0 |
| code | `832323a` (opt-in TorchInductor path on top of the v3 fix `500ae16`; `fit_planes` and the candidates are the same as in `500ae16`) |
| `big` chunk | 32 splits per pass |

The earlier CUDA runs used an A100-SXM4-80GB with `big` at 64 splits per pass
(`../planar_fit_cuda_2026-10-27/`, `../inductor_dispatch_a3_2026-10-27/`), so their
absolute times are not directly comparable with these.

## Correctness

- `reproduce_target_centring_cuda.log`: boxes outside 32 eps (1 + max|x| + max|y|) of the
  exact solution: 155 (eager) and 247 (compiled) with the targets centred on their mean,
  0 and 0 with the pivot-centred targets. `REPRODUCED`.
- `test_planar_fit_cuda.log` and `test_planar_fit_compiled_cuda.log`: all checks pass,
  float64 and float32 (largest per-box ratio 1.82 against bounds of 32 and 64). The
  compiled run did not fall back to eager: 3 graphs compiled.
- `test_batched_splits_{gelsd,closed_form,compiled}_cuda.log`: all three batched modes
  pass (0, 396 and 396 closed-form calls; the compiled one with 2 graphs compiled).
- `e2e.csv`: every bound of every mode agrees with the sequential loop to 2.13e-13.

## `reduce-overhead` overwrites its outputs (`aliasing.json`)

With `mode="reduce-overhead"` (CUDA graph) a second call of the same shape overwrote the
output of the first call that was kept without a copy. The batched pass without a copy
still gives exactly the bound of the eager closed form (difference 0), because
`precise_sigmoid` uses the coefficients before the next call. End to end the two
compiled variants are equally fast (`e2e_summary.csv`), so the ordinary compile
(`torch.compile(fullgraph=True)`, what `PASADO_COMPILED_FIT=1` uses) is the recommended
compiled variant.

## R4 for v3 (`timing_cuda.csv`, float64, n = 65536, median ms)

| G | P | A2 | H | A1 | A3 |
|---:|---:|---:|---:|---:|---:|
| 565.13 | 35.38 | 3.83 | 3.89 | 1.62 | 1.25 |

A2 and H fall in the same 20% step of R4, so R5 decides: A2 (no host sync). The rule
applied to run v3 with this file is in `../lstsq_comparison_2026-10-27_v3_zpivot/`.

## Compilation, cold and warm (`cold_warm_start.csv`)

Two separate processes per mode sharing one Inductor disk cache that starts empty; time
of the first call for n = 100 plus n = 1024 (the third shape, n = 65536, does not
recompile because the batch size has become dynamic).

| mode | cold (empty cache) | warm (second process) |
|---|---:|---:|
| ordinary compile | 31.33 s | 4.30 s |
| reduce-overhead | 28.59 s | 4.29 s |

The cold start is about three times the 10.4 s of the 2026-10-27 session on the 80 GB
machine; the cause was not measured.

## End to end (`e2e_summary.csv`, warm steady state, mean over 3 images)

- Small networks, 1 split: compiled A2 1.11x to 1.22x against batched gelsd (eager A2
  0.67x to 0.68x).
- `big`, 64 splits, eps index 0: compiled A2 312.35 ms, batched gelsd 2555.40 ms,
  sequential loop 4049.67 ms: 8.18x against batched gelsd, 12.97x against sequential.

The compiled times are warm steady state: the first pass of every configuration is
excluded. They are not the total time of a short process that starts cold.

## Break-even (`break_even.csv`)

Passes needed before the compilation pays off against batched gelsd, eps index 0:

| | 1 split | 8 splits | 64 splits |
|---|---:|---:|---:|
| ordinary compile, cold, small networks | 9068 to 18514 | 1201 to 2442 | 153 to 318 |
| ordinary compile, warm, small networks | 1244 to 2539 | 165 to 335 | 21 to 44 |
| ordinary compile, cold, `big` | 942 | 121 | 14 |
| ordinary compile, warm, `big` | 129 | 17 | 2 |

Caveat: `compile_s` in that file is the first-call time for n = 100 plus n = 1024, the
cost of a run that meets both shapes. It is a conservative estimate, not the exact cost
of one fixed configuration, which may compile once or twice depending on the shapes it
meets.

## Decisions taken on these results

- A2 is the final solver; eager A2 stays the production candidate.
- The ordinary compile stays opt-in (`PASADO_COMPILED_FIT=1`); it is not switched on
  automatically because of the cold start of 28 to 31 s (about 4.3 s with a warm disk
  cache).
- `reduce-overhead` does not go into the production path.
