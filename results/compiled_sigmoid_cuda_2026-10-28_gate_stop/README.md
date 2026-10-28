# Whole precise_sigmoid compiled by TorchInductor on CUDA: run stopped by its gate (2026-10-28)

Colab, notebook `COLAB_COMPILED_PRECISE_SIGMOID.ipynb` of the workspace, whose harness,
builder and notebook were committed in the workspace before the run (`91e02e4`, the
pre-registration). The files of this folder are the raw output of that run, unchanged
(the downloaded `compiled_sigmoid_cuda.zip` had sha256
`9e1ffc3438060a540788887adb0a2bc9ae5db98678f537fd82b696aa868f74ee`).

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, 80.0 GiB, driver 580.82.07 |
| torch, CUDA, Triton | 2.11.0+cu130, 13.0, 3.6.0 |
| code | `a57b560`, library unchanged: the modes are selected from outside (module flags, and `batched_splits.precise_sigmoid` swapped for the compiled function) |
| harness | `compiled_sigmoid_harness.py`, sha256 `df0e88f5f43486d061c990aa6b6a0e70ad4b14e0edf821560e269f1742fdd21c`; protocol, tolerances, criteria and outcome rule in its header |
| `big` chunk | 64 splits per pass |

## What ran

Only the correctness gate (section 1 of the notebook). It failed, so the notebook
stopped as designed, before anything else. Not run: the shared-callable recompilation
experiment, the cold and warm start processes, the steady-state timing, the compiler logs
and the criteria. This folder therefore holds no timing result.

## Gate: 868 of 876 checks pass

`gate_checks.csv`, summed by check. The grid: 3layer, 4layer, 5layer, big; S = 1, 8,
32, 64; 3 images; eps indices 0 and 8; a new compiled callable per (network, S); plus
unseeded noise on 3layer (S = 1 and 16).

| check | checks | pass |
|---|---:|---:|
| G1 fullgraph compile and run | 18 | 18 |
| G2 at least one graph compiled | 16 | 16 |
| G2 every sigmoid call through the compiled callable | 96 | 96 |
| G2 planar_fit.compile_fallback empty | 16 | 16 |
| G2 plane fit inside is the eager fit_planes (no nested compile) | 16 | 16 |
| G2 profiled pass: no lstsq | 16 | 16 |
| G2 profiled pass: one compiled region per sigmoid call | 16 | 16 |
| G3 NumPy state after the pass as eager A2 | 96 | 96 |
| G3 unseeded: NumPy state after the pass as eager A2 | 2 | 2 |
| G4 same seed, same bits | 96 | **92** |
| G5 four output tensors vs eager (normwise) | 96 | **92** |
| G5 shapes and nonzero flag as eager | 96 | 96 |
| G6 per-split bound vs eager A2 | 96 | 96 |
| G7 final bound vs sequential G | 96 | 96 |
| G8 final bound >= sampled autograd maximum | 96 | 96 |
| G8 unseeded bound >= sampled autograd maximum | 8 | 8 |

G1, G2, G3, G6, G7 and G8 pass everywhere. The bounds stay far inside the contract of
1e-10 (`gate_bounds.csv`): every per-split bound of the compiled pass within 9.83e-15 of
eager A2, every final bound within 2.13e-13 of sequential G (eager A2 itself: 2.12e-13).

The failures (`gate_summary.json`):

- G4, 4 times: 3layer S = 1, 4layer S = 8, 5layer S = 8, big S = 8, each at image 0 and
  eps index 0. The per-split bounds of those passes are within 2.5e-15 of eager A2.
- G5, 4 times, only the dual generators `dg`, only at S = 64 and eps index 8: 5layer
  image 1 (1.72e-12), big images 0, 1, 2 (1.42e-12, 1.33e-12, 1.06e-12), against 1e-12.

## Causes

### G4: which graph serves the first layer depends on the shapes seen before

All four failures are in the first pass of a configuration, the pass that compiles. In
that pass the first sigmoid layer (G0 = 2 generators) compiles a graph with static shapes;
the second layer (G0 = 202, or 2050 on big) fails its guard and compiles a second graph
with G0 dynamic. From the second pass on, Dynamo serves the first layer with the second,
dynamic graph as well. A local reproduction (CPU, `a57b560`, 4layer, S = 8) shows the
compile ids of the "Torch-Compiled Region" events per sigmoid layer: first pass 0/0, 0/1,
0/1; every later pass 0/1, 0/1, 0/1. On the CPU the two graphs give the same bits; on the
A100 they differ in the last bits in 4 of the 16 configurations. The first layer of a
pass therefore depends on the history of the callable, not only on its input. The noise
stream is not involved (G3 passes 96 of 96).

### G5: one normwise metric over the whole `dg` is dominated by the error term D on narrow boxes

`dg` holds three blocks per split: the propagated generators A rg + B dg (G0 rows), zeros
(N rows) and diag(D) (N rows). The largest normwise differences against eager
(`gate_tensors.csv`, maximum over networks and images):

| eps index | S = 1 | S = 8 | S = 32 | S = 64 |
|---:|---:|---:|---:|---:|
| 0 | 1.57e-15 | 1.72e-15 | 5.73e-15 | 1.80e-14 |
| 8 | 1.90e-14 | 2.14e-13 | 8.72e-13 | 1.72e-12 |

The difference grows as the sub-intervals narrow (eps index 8, larger S), where every
entry of `dg` is small. The other tensors stay below 1e-12 everywhere: `rc` 3.3e-16, `dc`
1.3e-13, `rg` 5.8e-13 (also at S = 64, eps index 8). The diagnostic of D itself: the
compiled D falls below the eager D by at most 1.71e-13 absolute, where D is about 290
(5.9e-16 relative, one or two ulps), and by up to 2.5e-8 relative where D is tiny
(about 4e-14 absolute). The harness recorded the normwise value of the whole `dg` only,
so how the difference splits between the propagated generators and D was not measured.

Relative to D the difference is rounding noise of a quantity computed with cancellation.
For that reason `experiments/compare_lstsq_solvers.py` (G1) judges D by its shortfall
below the true maximum error, relative to the neuron's radius W = |A| r_x + |B| r_y + D.

## Plan for v2 (pre-registered separately, before its run)

- G4 unchanged (same seed, same bits). The cause goes: before every compiled call the
  generator dimension is marked dynamic (`torch._dynamo.mark_dynamic` on `z.rg` and
  `z.dg`, dimension 1), so one graph serves every layer of a (network, S) from the first
  pass.
- G5: `rc`, `rg`, `dc` and the first two blocks of `dg` at 1e-12 normwise as before; D by
  the G1 contract: the shortfall of the compiled D below the true maximum (mpmath, 50
  digits) for the A, B, C and the box the compiled graph used, relative to W, at most
  10 * max(1e-12, the eager shortfall), on a sample of boxes fixed before the run.
- The tolerance 1e-12 is not raised. Everything else as in this run.

## Other observations

- Compilation (`gate_compile_events.csv`, host time of the call that compiled): 13 to
  18 s per graph for 3layer and big, 2.0 to 2.8 s for 4layer and 5layer. The short times
  are consistent with hits in the disk cache that the 3layer graphs of the same shapes
  had written (N = 100 on all three small networks); the cache itself was not logged.
