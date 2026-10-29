# Whole precise_sigmoid compiled by TorchInductor on CUDA, version 2: run stopped by its gate (2026-10-29)

Second run of the pre-registered experiment, after version 1 stopped at its gate
(`../compiled_sigmoid_cuda_2026-10-28_gate_stop/`). Colab, notebook
`COLAB_COMPILED_PRECISE_SIGMOID.ipynb` of the workspace, pre-registered in workspace
commit `7c9a8ef` before the run. The files of this folder are the raw output of that
run, unchanged (the downloaded `compiled_sigmoid_v2_cuda.zip` had sha256
`ec570527615ccdbddfaa417f2c83c4930badf81b7ce3d196ac30555284cab0ba`).

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, 80.0 GiB, driver 580.82.07 |
| torch, CUDA, Triton | 2.11.0+cu130, 13.0, 3.6.0 |
| code | `a57b560`, library unchanged |
| harness | `compiled_sigmoid_harness.py`, sha256 `9e8535ffbc296bb3d8b6ebdb3e50ef06d28e4af75b1bdce4843bc8967a3d51b5`; protocol, tolerances, criteria and the changes from version 1 in its header |
| `big` chunk | 64 splits per pass |

Workspace commits: the workspace history was rewritten once before its first push, to
leave out unrelated files; the pre-registration contents did not change (same blobs, same
harness sha256). Version 1 was pre-registered in `91e02e4`, now `788589d`; version 2 in
`e18278c`, now `7c9a8ef`. The harness header and the README of version 1 cite the old
hashes.

## What ran

Only the correctness gate. It failed, so the notebook stopped as designed: no
shared-callable experiment, no cold or warm start, no steady-state timing, no compiler
logs, no criteria. This folder holds no timing result.

## Gate: 892 of 908 checks pass

`gate_checks.csv`, summed by check:

| check | checks | pass |
|---|---:|---:|
| G1 fullgraph compile and run | 18 | 18 |
| G2 at least one graph compiled | 16 | 16 |
| G2 every sigmoid call through the compiled callable | 96 | 96 |
| G2 one graph serves every sigmoid layer | 16 | 16 |
| G2 planar_fit.compile_fallback empty | 16 | 16 |
| G2 plane fit inside is the eager fit_planes (no nested compile) | 16 | 16 |
| G2 profiled pass: no lstsq | 16 | 16 |
| G2 profiled pass: one compiled region per sigmoid call | 16 | 16 |
| G3 NumPy state after the pass as eager A2 | 96 | 96 |
| G3 unseeded: NumPy state after the pass as eager A2 | 2 | 2 |
| G4 same seed, same bits | 96 | 96 |
| G5 D: diagnostic graph bit-identical to the production compiled graph | 12 | **0** |
| G5 D: worst compiled shortfall / W <= 10 * max(1e-12, worst eager shortfall / W) | 4 | **0** |
| G5 rc, rg, dc and the dg generator and zero blocks vs eager (normwise) | 96 | 96 |
| G5 shapes and nonzero flag as eager | 96 | 96 |
| G6 per-split bound vs eager A2 | 96 | 96 |
| G7 final bound vs sequential G | 96 | 96 |
| G8 final bound >= sampled autograd maximum | 96 | 96 |
| G8 unseeded bound >= sampled autograd maximum | 8 | 8 |

## The two changes of version 2 worked

- `mark_dynamic` on the generator dimension: G4 (same seed, same bits) passes 96 of 96,
  where version 1 failed 4 times. Exactly one graph serves every sigmoid layer in all 16
  configurations, and nothing recompiles (`gate_compile_events.csv`: 18 graphs, one per
  (network, S) of the grid plus one for each of the two unseeded configurations).
- G5 by blocks of `dg` (`gate_tensors.csv`, worst over the grid): `rc` 1.1e-15, `rg`
  5.8e-13, `dc` 1.2e-13, `dg` propagated generators 3.9e-13, `dg` zero blocks exactly 0.
  The version 1 metric over the whole `dg`, still recorded, reaches 1.79e-12 at S = 64,
  eps index 8 and would have failed again.

Every gate on the bounds and their soundness passes: G6, every per-split bound of the
compiled pass within 9.97e-15 of eager A2; G7, every final bound within 2.12e-13 of
sequential G (eager A2 itself: 2.12e-13); G8, every bound above the sampled autograd
maximum, seeded and unseeded; G3, the NumPy stream as in the eager pass.

## The D contract did not run

At S = 64, eps index 8 the contract needs (A, B, C) and the box of every sampled
(split, neuron). Version 2 took them from a diagnostic graph: the same precise_sigmoid
compiled under a second name, with the corner check wrapped so that it returned those
tensors as extra outputs. The pre-registered rule used them only if the diagnostic graph
gave its four main outputs and the flag bit-identical to the production compiled graph.
On the A100 it did so in none of the 39 calls (13 sigmoid layers over the four networks,
3 images). So no box reached the mpmath evaluation (`gate_d_contract.csv` is empty), and
the four contract checks failed with 0 boxes.

The assumption that failed is the bit-identity of two differently fused compiled graphs:
the extra outputs change the generated kernels, and with them the rounding. On the CPU
the two graphs had been identical. The harness recorded only whether the outputs were
identical, not how far apart they were.

This run therefore says nothing about the soundness of the compiled D, in either
direction. Recorded but not judged: the normwise difference of D itself against eager
(max |dD| / max |D_eager|) grows as the sub-intervals narrow, up to 4.9e-11 at S = 64,
eps index 8 (1.7e-15 at S = 1, eps index 0).

## Plan for version 3 (pre-registered separately, before its run)

The graph whose A, B, C and D are checked must be the graph whose correctness, speed,
memory and break-even are measured. The formal candidate becomes precise_sigmoid with
diagnostic tensor outputs (A, B, C, lx, ux, ly, uy, D) returned by the compiled function
itself; the batched pass uses its main outputs and drops the diagnostics, whose cost is
part of the candidate. The uninstrumented compiled graph stays as an ablation only
(difference of outputs, speed, kernels), outside the formal outcome. Everything else as
in version 2: `mark_dynamic`, strict G4, G5 at 1e-12 by blocks, the G1/W contract for D
on the same fixed sample, G6 to G8, C2 to C4.
