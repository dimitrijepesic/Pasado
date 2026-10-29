# Whole precise_sigmoid compiled by TorchInductor on CUDA, version 3: complete run (2026-10-29)

Third and complete run of the pre-registered experiment on compiling the whole batched
`precise_sigmoid` with TorchInductor. Colab, notebook `COLAB_COMPILED_PRECISE_SIGMOID.ipynb`
of the workspace. Every file of this folder except this README and `SHA256SUMS` is the raw
output of that run, unchanged.

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, 80 GiB of VRAM, driver 580.82.07 |
| Python | 3.13.16 |
| torch, CUDA, Triton | 2.11.0+cu130, 13.0, 3.6.0 |
| Pasado code | `a57b560d8cd18761b9bcd745d51478358837f119` (library unchanged) |
| harness | `compiled_sigmoid_harness.py`, sha256 `898980d93b74822f3ea31b2674653762b511da052c91a5ed31c9bb7225907bec` |
| `big` chunk | 64 splits per pass |
| run time | 53.2 minutes |
| downloaded zip | `compiled_sigmoid_v3_cuda.zip`, sha256 `00471c021829987d6e7747401ffde0a12dec64ce8e9d78ed1638d16d5829ea62` |

The zip held 259 files, all under `compiled_sigmoid_v3_cuda/`. Each was extracted here and
compared with its zip entry by sha256: all 259 are identical. `SHA256SUMS` lists them
(`shasum -a 256 -c SHA256SUMS` checks the folder).

## 2. Scope

Five formal modes, all float64, seed 7, the same grid: networks 3layer, 4layer, 5layer,
big; S = n_splits in 1, 8, 32, 64; 3 images; eps indices 0 and 8.

| mode | what runs |
|---|---|
| `sequential_G` | the original loop over the splits, plane fit by gelsd on the CPU |
| `batched_G` | all splits in one batched pass, gelsd on the CPU (the production default) |
| `batched_A2_eager` | batched pass, closed-form plane fit `planar_fit.fit_planes` eager (`PASADO_CLOSED_FORM_FIT=1`) |
| `batched_A2_compiled_fit` | as above with `fit_planes` compiled by TorchInductor (`PASADO_COMPILED_FIT=1`) |
| `batched_compiled_sigmoid` | the formal candidate: the whole `precise_sigmoid` compiled with `fullgraph=True` |

The formal candidate is `precise_sigmoid_instrumented` from the harness: a line-for-line
copy of `batched_splits.precise_sigmoid` at `a57b560` that also returns the diagnostic
tensors A, B, C, the box (lx, ux, ly, uy) and D as outputs of the compiled function. The
generator dimension is marked dynamic before every call (`torch._dynamo.mark_dynamic`), and
the eager `fit_planes` runs inside the graph (no nested compile). The batched pass uses the
main outputs and drops the diagnostics; computing and writing them is part of the measured
cost of the candidate. The same compiled graph is checked for correctness, checked against
the D contract, and timed.

A sixth mode, `batched_compiled_sigmoid_uninstrumented` (the library's own
`precise_sigmoid` compiled the same way, without diagnostic outputs), is an ablation only:
it is timed and compared, and it takes no part in the outcome.

This run changed no production code. The compiled whole sigmoid is not a production
feature: no switch in the library selects it.

## 3. Pre-registration and outcome

The protocol, tolerances, criteria and outcome rule are in the header of
`compiled_sigmoid_harness.py`, committed in the workspace before the run
(`85de132`, harness sha256 `898980d9...`). The notebook refused to run with any other
harness.

Mechanical outcome (`criteria.json`): **opt-in**. No configuration is missing;
`shared_rc = 0`, `steady_rc = 0`, `start_failures = []`.

| criterion | result |
|---|---|
| C1 correctness (gate, steady checks, start checks) | passes |
| C2 compiled whole sigmoid not slower than compiled fit by more than 5%, every (network, S) | passes |
| C3 at least 1.2x faster than compiled fit on the small networks at S = 1 | passes |
| C4 warm break-even against the best path without compilation <= 480 passes on the small networks at S = 1 | fails on 3layer only: 582.66 passes |

The outcome rule makes a candidate for the automatic mode only if C1, C2 and C3 hold and
the warm break-even is at most 480 passes (one standard sweep of a network: 30 images x 16
eps) on all three small networks at S = 1. 3layer needs 582.66, so the outcome is opt-in.
The compiled whole sigmoid is neither a default nor an automatic-mode candidate.

## 4. Correctness

- Gate: 1088 of 1088 checks pass. Steady process: 280 of 280. Start processes: 192 of 192.
- Every per-split bound of the compiled pass within 7.752478098523723e-15 (relative) of
  eager A2; every final bound within 2.1270558326495904e-13 of sequential G (contract 1e-10).
- Worst judged tensor difference against eager (rc, rg, dc and the generator and zero
  blocks of dg, normwise): 5.795839809306759e-13 (limit 1e-12).
- Same seed, same bits: 96 of 96. The NumPy noise stream as in the eager pass: 96 of 96.
- One dynamic graph serves every sigmoid layer of each configuration (16 of 16); every
  sigmoid call goes through the compiled candidate; no fallback, no lstsq in the compiled
  pass, no nested compile.
- The candidate run eagerly is bit-identical to `batched_splits.precise_sigmoid` (96 of 96),
  and the D it returns equals, bit for bit, the D it propagates (96 of 96).
- Every bound stays above the sampled autograd maximum of the derivative (96 seeded, 8
  unseeded).
- No configuration is missing.

## 5. D contract and the floating-point caveat

At S = 64, eps index 8 (the narrowest sub-intervals of the grid), the error term D was
checked as in G1 of `experiments/compare_lstsq_solvers.py`: per box, the shortfall of D
below the true maximum of |sigmoid'(x) y - (A x + B y + C)| over the box (mpmath, 50
digits), relative to the neuron's radius W = |A| r_x + |B| r_y + D, with A, B, C and the box
taken from the outputs of the graph under test. Pass, per network: worst compiled shortfall
at most 10 x max(1e-12, worst eager shortfall).

4992 boxes were sampled by the pre-registered rule (`gate_d_contract.csv`); every mpmath
evaluation converged. Worst shortfall over W, from `gate_d_contract.csv` (the `limit`
column of `gate_checks.csv` is 10 times the eager value):

| network | boxes | compiled | eager | result |
|---|---:|---:|---:|---|
| 3layer | 768 | 6.633995323591303e-10 | 7.074451663199927e-10 | pass |
| 4layer | 1152 | 2.1916498568066814e-09 | 1.4116413894917793e-09 | pass |
| 5layer | 1536 | 2.752934349244645e-10 | 3.902655201894371e-10 | pass |
| big | 1536 | 3.8635751903698146e-09 | 3.437041416050687e-09 | pass |

Over all boxes: worst compiled 3.8635751903698146e-09, worst eager 3.437041416050687e-09;
largest absolute shortfall about 2.49e-14 (compiled 2.494e-14, eager 2.486e-14). The eager
D is below the true maximum on 2562 of 4992 boxes (51.3%), the compiled D on 2383 (47.7%).
Measured against the size of the terms of the error, 0.25 |y| + |A x| + |B y| + |C| over the
box, every shortfall is at most 2.4 ulps (median 0.2 ulps).

This is not a new compiler bug. It is the existing floating-point behaviour of Pasado's
error term: D is computed in floating point with cancellation and no outward rounding, so on
narrow boxes it falls short of the exact maximum by a few ulps. The compiled D is not
systematically worse than the eager D (worst ratio 1.55 on 4layer, below 1 on 3layer and
5layer). Pasado is not formally sound in floating point; the sampled autograd checks and
the pre-registered D/W contract pass.

## 6. Steady-state timing

Median of 30 samples per (network, S, mode) in one process with a warm cache, the six
modes interleaved with a rotating order (`steady_summary.csv`, ms):

| network, S | sequential G | batched G | eager A2 | compiled fit | compiled whole sigmoid | uninstrumented (ablation) |
|---|---:|---:|---:|---:|---:|---:|
| 3layer, 1 | 12.412 | 10.658 | 15.854 | 8.894 | 2.445 | 2.358 |
| 3layer, 64 | 988.714 | 106.938 | 22.218 | 12.144 | 3.429 | 3.146 |
| 4layer, 1 | 22.770 | 19.879 | 32.392 | 17.051 | 3.836 | 3.723 |
| 5layer, 1 | 29.754 | 26.065 | 42.422 | 22.294 | 4.822 | 4.595 |
| big, 1 | 67.721 | 57.556 | 43.552 | 23.212 | 10.643 | 12.567 |
| big, 64 | 4589.672 | 2298.359 | 283.646 | 274.980 | 220.565 | 217.710 |

Speedup of the compiled whole sigmoid:

| network, S | vs batched G | vs sequential G | vs compiled fit |
|---|---:|---:|---:|
| 3layer, 1 | 4.36x | 5.08x | 3.64x |
| 3layer, 64 | 31.19x | 288.36x | 3.54x |
| 4layer, 1 | 5.18x | 5.94x | 4.44x |
| 5layer, 1 | 5.41x | 6.17x | 4.62x |
| big, 1 | 5.41x | 6.36x | 2.18x |
| big, 64 | 10.42x | 20.81x | 1.25x |

## 7. C2 and C3

- C2 passes in all 16 (network, S) cells. The ratio median(compiled whole sigmoid) /
  median(compiled fit) is 0.22 to 0.28 on the small networks and 0.46 to 0.80 on big
  (0.459, 0.733, 0.766, 0.802 for S = 1, 8, 32, 64).
- C3, compiled fit over compiled whole sigmoid at S = 1: 3layer 3.6369x, 4layer 4.4447x,
  5layer 4.6235x; the target was at least 1.2x.

## 8. Kernel fusion

CUDA kernels counted by the profiler in this environment (`steady_kernels.csv`):

| | batched G | eager A2 | compiled fit | compiled whole sigmoid | uninstrumented |
|---|---:|---:|---:|---:|---:|
| one 3layer pass, S = 1 | 481 | 1107 | 487 | 65 | 63 |
| one big pass, S = 64 | 977 | 2229 | 989 | 103 | 107 |
| one sigmoid layer (first chunk) | 220 to 241 | 533 to 554 | 223 to 244 | 12 to 13 | 11 to 14 |

The compiled whole sigmoid runs about 12 to 13 kernels per sigmoid layer, where the
compiled fit alone still needs about 223. These counts belong to this GPU, torch, CUDA and
Triton; other versions may fuse differently.

## 9. Memory

Peak VRAM of one big pass at S = 64 (`steady_memory.csv`, MiB): sequential G 480.52,
batched G 29803.89, eager A2 29803.89, compiled fit 29803.89, compiled whole sigmoid
20543.89, uninstrumented 20544.89. The compiled whole sigmoid needs about 31% less than
batched G (presumably because fused intermediates are not materialized; not measured). Batched memory still grows linearly
with the number of splits in a pass; the chunk size bounds it.

## 10. Cold and warm compilation

Per (network, S, compiled mode), two processes shared one cache directory that started
empty: the first cold, the second warm (`start/`, `start_summary.csv`). One graph was
compiled in every start process.

- Compiled whole sigmoid: overhead 16.8 to 16.9 s cold and 4.7 to 4.8 s warm on the small
  networks at S = 1 (16.8 to 20.7 s cold, 4.7 to 5.0 s warm over the whole grid).
- Compiled fit, for comparison: about 14.5 to 14.9 s cold and 2.2 s warm.

Break-even of the compiled whole sigmoid against the best path without compilation, small
networks at S = 1 (`break_even.csv`, passes):

| | 3layer | 4layer | 5layer |
|---|---:|---:|---:|
| warm cache | 582.66 | 295.53 | 221.65 |
| cold cache | 2048.06 | 1049.82 | 795.78 |

A standard sweep of one network is 480 passes. With a warm cache the compilation pays off
within a sweep on 4layer and 5layer but not on 3layer; with a cold cache on none of the
three. This is why the outcome is opt-in. At S >= 8 the warm break-even is 77 to 259
passes and the cold one 320 to 932.

## 11. Instrumented against uninstrumented (ablation, not part of the outcome)

- On the small networks the candidate takes 3 to 13% longer than the uninstrumented graph
  (ratio of medians 1.031 to 1.128): the cost of the diagnostic outputs.
- On big at S = 1 and 8 the instrumented candidate was faster in this run (ratios 0.847 and
  0.889); 0.980 and 1.013 at S = 32 and 64. The cause was not measured.
- The two graphs never gave bit-identical outputs; the largest normwise difference is
  about 1.1e-12 (`gate_tensors.csv`, `ablation_*` columns).
- The uninstrumented graph is not the formal candidate and is not verified for production.
  A production version without diagnostic outputs needs its own verification.

## 12. Recompilation and cache

- Per (network, S) from a fresh callable: one graph, with G0 dynamic, no recompile limit
  hit (`recompiles.csv`, `gate_compile_events.csv`).
- One shared callable over the whole grid, two rounds, Dynamo's default recompile limit of
  8: 4 graphs for the compiled whole sigmoid (3layer S = 1 and 8, big S = 1 and 8) and 2 for
  the compiled fit; no new graph in the second round (`shared_callable.csv`).
- Marking G0 dynamic removed the history-dependent first pass found in version 1.
- Cold and warm caches were measured in separate processes.

## 13. Files

| file | content |
|---|---|
| `criteria.json` | the mechanical evaluation: C1 (gate, steady and start checks), C2, C3, C4 warm and cold, missing configurations, outcome, the ablation ratios |
| `gate_summary.json` | number of gate checks and failures, worst differences, D contract summary |
| `gate_checks.csv` | every gate check: name, network, S, image, eps index, value, limit, pass |
| `gate_d_contract.csv` | the 4992 sampled boxes: reason for the choice, compiled and eager A, B, C, box, D, W, D_true, shortfalls |
| `gate_tensors.csv` | every compared sigmoid call: normwise differences per tensor and dg block, copy and D-output flags, ablation differences |
| `gate_bounds.csv` | per (network, S, image, eps): final bounds of sequential G, eager A2 and the candidate, per-split differences, sampled autograd maximum |
| `gate_compile_events.csv` | every call that compiled a graph during the gate: shape and host time |
| `steady_summary.csv` | median, min, max per (network, S, mode), with speedups |
| `steady_times.csv` | every timed pass: mode, image, eps, repeat, position in the rotation, time, bound |
| `steady_first_pass.csv` | first pass of every mode per (network, S), including warm compilation |
| `steady_kernels.csv` | profiler counts per pass and per sigmoid layer: kernels, memcpy, memset, compiled regions, lstsq |
| `steady_memory.csv` | peak VRAM of one pass per mode |
| `steady_checks.csv` | path and bound checks of the steady process (`formal=False` for the ablation) |
| `break_even.csv` | overhead and break-even per (network, S, compiled mode, cache) against batched G and the best path without compilation |
| `start_summary.csv`, `start/` | one JSON per start process (cold, warm) and their summary |
| `start_checks.csv` | checks of the start processes: no fallback, graphs compiled, bounds against sequential G |
| `recompiles.csv` | per (network, S): graphs, triggering shapes, dynamic dimensions, recompile reasons, limit hit |
| `shared_callable.csv`, `shared_compile_events.csv` | the shared-callable experiment |
| `logs/` | stdout and stderr of every process, including `TORCH_LOGS` output; `compiler_*` holds graph breaks, recompiles, fusion decisions and output code for 3layer S = 1 and big S = 32 |
| `kernel_code/` | the generated output code (Triton kernels) of those three compiler-log configurations |
| `compiled_sigmoid_harness.py` | the pre-registered harness, as run |
| `colab_environment.json` | GPU, versions, commit, harness sha256, chunk, run time, return codes, outcome |
| `SHA256SUMS` | sha256 of every raw file, added when archiving |

## 14. Interpretation

The technical result is strongly positive: compiling the whole sigmoid transformer gives a
large steady-state gain (4.4x to 5.4x over batched G at S = 1 on the small networks, 10.4x
over batched G and 20.8x over sequential G on big at S = 64), 7.4x to 9.5x fewer kernels per
pass than batched G and about 31% less peak memory on big, with every correctness check passing. The formal
outcome stays opt-in because the compilation does not amortize within one standard sweep in
every case. Nothing here allows a change of the production default, and this archive adds
no production implementation of the compiled whole sigmoid. Other compilers (TVM, XLA) and
a separate verification of an uninstrumented version are open items outside this run.

## 15. Related archived runs

- [`../compiled_sigmoid_cuda_2026-10-28_gate_stop/`](../compiled_sigmoid_cuda_2026-10-28_gate_stop/): version 1, stopped by its gate (G4, G5).
- [`../compiled_sigmoid_v2_cuda_2026-10-29_gate_stop/`](../compiled_sigmoid_v2_cuda_2026-10-29_gate_stop/): version 2, stopped by its gate (D contract not evaluable).
- [`../v3_compiled_cuda_2026-10-28/`](../v3_compiled_cuda_2026-10-28/): the compiled planar fit (`PASADO_COMPILED_FIT`) on CUDA.
- [`../lstsq_comparison_2026-10-27_v3_zpivot/`](../lstsq_comparison_2026-10-27_v3_zpivot/): the pre-registered solver comparison that chose A2.
- [`../plane_fit_crossover_2026-10-28/`](../plane_fit_crossover_2026-10-28/): the gelsd against A2 crossover behind `PASADO_AUTO_PLANE_FIT`.
- [`../auto_plane_fit_cuda_2026-10-28/`](../auto_plane_fit_cuda_2026-10-28/): CUDA check of the automatic plane-fit choice.
