# GPU_READINESS - audit of the tensorized precise-transformer path (Phase 10)

Static audit of `forward_mode_tensorized_src/` (precise_transformer.py,
SimpleZono.py, Duals.py, HyperDuals.py) and `Section_5_4/get_lipschitz.py`.
No GPU is available on the primary development machine (torch 2.12.1+cpu,
CUDA=False), so the findings table below is a code-level classification, not a
measured GPU result. The companion script `experiments/colab_gpu_profile.py`
was written to run the actual CPU-vs-GPU check on Colab - see "Phase 11:
measured Colab T4 result" below for what it found once actually run.

**Current status (2026-10-05):** the findings table is the historical audit
that motivated the work. The FFNN fixes described in "Phase 14" at the end of
this document are now applied and covered by CPU/MPS tests. CUDA end-to-end
verification still has to run on Colab.

## Classification legend

- **READY** - pure tensor ops on the input's device; runs on GPU as-is.
- **DEVFIX** - needs a small, semantics-preserving device/dtype propagation
  (`x.new_ones(...)`, `device=x.device`).
- **CPU-ONLY** - depends on a CPU-only backend; needs a real decision.
- **SYNC** - forces host-device synchronization / GPU->CPU transfer per call.
- **GRAPHBREAK** - Python-side control/RNG that would break torch.compile and
  serialize GPU streams.

## Findings

| # | Location | Issue | Class |
|---|---|---|---|
| 1 | `get_linspace` (precise_transformer:41-42) | `lx[i].item()` x4 per neuron (7.87M `.item()` calls on big) + `torch.linspace` builds on CPU/default device | **SYNC** + DEVFIX (or batch it away) |
| 2 | `inverse_poly_tensor` (:120,125,136) | `torch.tensor(-1/12.)`, `torch.tensor(1.j)`, `torch.sqrt(torch.tensor(3.))` created per call on CPU; on GPU inputs this raises a cross-device error | **DEVFIX** (`y.new_tensor(...)` / `device=y.device`) |
| 3 | `inverse_poly_tensor` `.type(torch.complex64)` | works on GPU; note it fixes root precision at float32 *by design* (pre-existing; see the dtype-bug writeup in RAW_NOTES) | READY (with the known precision caveat) |
| 4 | `lin_reg_tensor` (:99) | `torch.ones(x.size(0),1)` in default dtype/device; errors on GPU input | **DEVFIX** - `lin_reg_tensor_batched` already propagates dtype+device |
| 5 | `torch.linalg.lstsq(driver='gelsd')` | **CUDA supports only `driver='gels'`**, no minimum-norm guarantee for rank-deficient systems. gelsd is CPU-only. **Measured (Phase 12): confirmed unsafe** - on degenerate boxes `gels` output diverges from `gelsd` by 15-34 orders of magnitude, not just "somewhat different." | **CPU-ONLY** - must solve on CPU (transfer+sync per layer) unless a rank-check + CPU/SVD fallback is added specifically for degenerate boxes |
| 6 | `sigmoid_prime_product_tensor` (:575-579) | `np.random.normal` per neuron (host RNG) + `torch.tensor([list of 0-dim tensors])` - on GPU the latter syncs and lands on CPU | **SYNC** + **GRAPHBREAK**; the batched path reduces it to one host-built perturbation vector per layer |
| 7 | old `check_corners` / `check_nonlinear_boundary` (:227-233, :299-301) | per-neuron `torch.tensor([lx[i], ly[i]])` from 0-dim tensors - CPU construction + implicit sync xN | SYNC (old fallback path only) |
| 8 | `check_corners_tensor`, `_max_objective_over_x_candidates`, `check_nonlinear_boundary_tensor` | pure stack/sigmoid/mul/max on the input device (constants in #2 aside) | **READY** |
| 9 | `traceify` (SimpleZono:16-19) | `torch.eye(leng)` on default device | DEVFIX |
| 10 | `Zonotope.expand` (SimpleZono:75), `SigmoidDualZonotope` (:328), `Duals.py:15-17`, `HyperDuals.py:165-175` | `torch.zeros(shape)` on default device | DEVFIX (mechanical, ~10 sites) |
| 11 | `HyperDuals.py:87,93-95`, `Duals.py:96` | `torch.ones(a.shape)` default device; `torch.tensor(np.inf)` 0-dim CPU scalars (0-dim scalars are cross-device legal, but tidy up while touching) | DEVFIX / minor |
| 12 | `get_lipschitz.py` | model + data loaded with `map_location='cpu'`; no `.to(device)` anywhere; everything rides on the default device | DEVFIX (entry-point plumbing: `net.to(dev)`, `img.to(dev)`, or `torch.set_default_device`) |
| 13 | `torch.set_default_dtype(torch.float64)` | **measured** (Phase 11, Colab T4): fp64 matmul runs at ~1/17 of fp32 throughput (250.7 vs 4262.8 GFLOP/s, 2048^2). Confirms the dense `AffineZonotope` matmul (the #1 cost on `big`) would not get the hoped-for GPU win on this card - see Phase 11 below. | confirmed, not just anticipated |
| 14 | `conv.py:38` `np.product` | removed in NumPy>=2.0 - crashes the CNN path on a fresh Colab install (`np.prod` is the fix). FC benchmark unaffected. | known compat fix |
| 15 | `np.prod(list(shape))` (SimpleZono:11,49), `np.sqrt(3.0)` scalars | host-side Python/NumPy scalars only - no tensor transfer | READY |

## Phase 11: measured Colab T4 result

`experiments/colab_gpu_profile.py` was actually run (Colab, torch 2.11.0+cu128,
Tesla T4). This is a real measurement, not the static prediction from the
table above.

```
CUDA: available - Tesla T4
  matmul 2048^2 fp32:     4.03 ms  (  4262.8 GFLOP/s)
  matmul 2048^2 fp64:    68.54 ms  (   250.7 GFLOP/s)
CPU reduced precise layer (n=100): median 23.9 ms
check_corners_tensor: max|cpu-gpu|=0.000e+00  cpu 142 us vs gpu 253 us
batched lstsq: CUDA driver forced to 'gels' (gelsd is CPU-only);
  max|gelsd_cpu-gels_gpu|=2.665e-15
```

**fp64 penalty confirmed: ~17x.** T4 fp64 throughput (250.7 GFLOP/s) is close
to its NVIDIA-spec peak (~254 GFLOP/s) - this is not a fluke, it is what the
card can do.

**The decisive comparison: T4 fp64 vs. this project's own CPU.** The `big`
wall-clock benchmark's machine-health probe (same shape of measurement - a
fixed float64 matmul) recorded 224-238 GFLOP/s on the local i5-12450H. T4 fp64
(250.7 GFLOP/s) is **essentially the same order of magnitude as a laptop CPU**,
not a GPU-class number. Porting the dominant `AffineZonotope` matmul to a T4
would not deliver the hoped-for win - the op itself would run at roughly the
same rate, while adding PCIe transfer and kernel-launch overhead on top.

**Small vectorized kernels are slower on GPU, not faster.**
`check_corners_tensor` is bit-identical (0.000e+00) between CPU and GPU, but
CPU wins on latency: 142us vs 253us. This is the same story as the
`torch.compile` finding in `TORCH_COMPILE_ANALYSIS.md` - after Phase 3-6
vectorization these kernels are cheap enough that dispatch/launch overhead
dominates, on GPU as much as under Inductor.

**`gels` vs `gelsd` numeric closeness is not yet evidence of safety.** The
2.665e-15 difference is on this script's random test input, which is
full-rank. It says nothing about the degenerate (`lx==ux`) rank-deficient
boxes that actually occur in the real analysis (finding #5's open question).
A dedicated rank-deficient test would be needed before trusting `gels` on GPU
 -  see Phase 12 below, which ran exactly that test and closes the question.

**Verdict for this workload, on this card:** GPU porting is **not a promising
direction on a T4** - not for the dominant fp64 matmul (no throughput win over
the CPU already in hand) and not for the cheap vectorized kernels (launch
overhead loses). This would plausibly look different on an A100 (~9.7 TFLOP/s
fp64) or H100 (~33.5 TFLOP/s fp64) - 40-130x T4's fp64 rate - but that is
outside current compute access (Colab Pro T4/L4 tier) and remains untested.

## Phase 12: rank-deficient driver test - `gels` confirmed unsafe

`experiments/colab_rank_deficient_lstsq_test.py` builds the same `[n, 25, 3] \
[n, 25]` batch shape as the real per-layer regression, with a controlled
fraction of rows forced rank-deficient the way a degenerate box (`lx==ux`
and/or `ly==uy`) actually produces it: point box (both degenerate, rank 1),
degenerate-x only (rank 2), and full rank. Run on Colab, torch 2.11.0+cu128,
**NVIDIA A100-SXM4-80GB**, n=2000 rows:

```
point box (rank 1)     n=  200  max|gelsd_cpu-gels_gpu| per row: median=2.616e+31  max=3.354e+34  (>1e-6: 100.0%)
degenerate x (rank 2)  n=  600  max|gelsd_cpu-gels_gpu| per row: median=1.273e+15  max=1.206e+18  (>1e-6: 100.0%)
full rank              n= 1200  max|gelsd_cpu-gels_gpu| per row: median=4.441e-16  max=6.946e-12  (>1e-6:   0.0%)
```

**Full rank confirms the earlier result**: median 4.4e-16 is machine-precision
agreement, consistent with the 2.665e-15 seen in Phase 11 on a different card.

**Rank-deficient is not "somewhat off" - it is unusable.** Differences are
15 to 34 *orders of magnitude*, not a precision issue. `gels` (QR-based) has
no defined behavior for rank-deficient input; without `gelsd`'s SVD-based
minimum-norm handling, near-zero pivots in the QR path blow up the solved
coefficients. This gets worse, not better, with more degeneracy (rank 1 is
worse than rank 2).

**Conclusion: finding #5's open question is closed.** `gels` on GPU must
never run on a degenerate box unchecked. Any GPU port of the regression step
needs either (a) keep the solve on CPU entirely, or (b) a rank check per
layer that routes degenerate rows to a CPU/SVD fallback and only sends
full-rank rows to `gels` on GPU - silently swapping the driver wholesale is
now a demonstrated correctness bug, not a hypothetical one.

**Note on hardware**: this test ran on an A100, not the T4 used in Phase 11.
The driver-safety conclusion is hardware-independent (it's an algorithmic
property of `gels` vs `gelsd`, not a throughput question), but this also
means the A100's fp64 throughput has not yet been measured here - see
"Suggested next step" below.

## Phase 13: A100 fp64 throughput - measured, reverses the Phase 11 verdict

`colab_gpu_profile.py` re-run on the A100 that Phase 12 happened to land on
(Colab, torch 2.11.0+cu128):

```
CUDA: available - NVIDIA A100-SXM4-80GB
  matmul 2048^2 fp32:     1.13 ms  ( 15192.6 GFLOP/s)
  matmul 2048^2 fp64:     1.12 ms  ( 15295.4 GFLOP/s)
CPU reduced precise layer (n=100): median 23.2 ms
check_corners_tensor: max|cpu-gpu|=0.000e+00  cpu 146 us vs gpu 285 us
batched lstsq: max|gelsd_cpu-gels_gpu|=2.665e-15  (full-rank test, same as Phase 11)
```

**fp64 is not penalized on A100 - it is essentially free.** fp64 throughput
(15295.4 GFLOP/s) is *not* a fraction of fp32 (15192.6 GFLOP/s), it's on par
with it. This is expected and hardware-specific: unlike T4/consumer cards,
A100 has dedicated FP64 Tensor Cores (spec peak ~19.5 TFLOP/s), so a fp64
matmul here isn't running the same crippled path Phase 11 measured on T4.

**This reverses the Phase 11 conclusion.** A100 fp64 (15295.4 GFLOP/s) is
**~65x** the local CPU's rate (224-238 GFLOP/s, from the `big` benchmark's
health probe) and **~61x** T4's rate (250.7 GFLOP/s). The dense
`AffineZonotope` matmul that dominates `big` (416s, 33% of the profile) would
very plausibly see a large real win on an A100 - the "GPU not promising"
verdict from Phase 11 was T4-specific, not general.

**What did not change**: `check_corners_tensor` is still slower on GPU (146us
CPU vs 285us A100) - the already-vectorized check kernels are cheap enough
that launch/transfer overhead dominates regardless of which GPU tier is
under them. The A100 opportunity is specifically in the large dense matmul,
not these small kernels. And Phase 12's `gels`-unsafe-on-rank-deficient-input
finding is a driver-level (not hardware-level) issue - it applies here too.

**Caveat**: Colab GPU allocation is not guaranteed or stable across sessions
(A100 access here was incidental, not requested) - this is evidence of the
*ceiling*, not a resource that can be relied on for a repeatable benchmark
without a paid dedicated-tier or cluster allocation.

**Next step, if pursued**: this is now a real research direction, not a
closed-off one. Porting the actual `AffineZonotope` dense matmul (not just
the small check kernels) to a confirmed-available A100/H100-class device,
with the `gelsd`->CPU-fallback / rank-check plumbing from Phase 12 for the
regression step, would be the next concrete experiment - out of scope for
this sprint's "no device rewrites" rule, but no longer blocked by an
open fp64 question.

## Summary

* The **vectorized check path (our Phase-3 work) is GPU-ready** modulo the
  shared constants in `inverse_poly_tensor` (#2) - one small DEVFIX.
* The **per-neuron regression loop is the worst GPU citizen** (#1, #4, #6):
  `.item()` syncs, CPU tensor construction, host RNG - per neuron. The batched
  Phase-6 path removes almost all of it; what remains is one grid build and one
  perturbation vector per layer.
* The **one hard blocker is `driver='gelsd'`** (#5): CUDA only offers `gels`,
  and Phase 12 measured that this is **not a safe substitute** on degenerate
  boxes - differences of 15-34 orders of magnitude, not a precision issue.
  Viable options: (a) keep the solve on CPU (one transfer per layer after
  batching - small), (b) a rank check per layer routing degenerate rows to a
  CPU/SVD fallback and only full-rank rows to `gels` on GPU, (c) QR-based
  closed form. Do not silently swap drivers to `gels` wholesale.
* **fp64 verdict is hardware-dependent, now measured on both ends.** T4
  (Phase 11): fp64 ~1/17 of fp32, on par with a laptop CPU - GPU not
  promising. A100 (Phase 13): fp64 ~on par with fp32 (dedicated FP64 Tensor
  Cores), ~65x the CPU rate - GPU *is* promising, for the dense matmul
  specifically, if A100/H100-class access can be relied on.
* **fp64 on consumer GPUs** (#13) cancels the matmul win on a T4, measured:
  see Phase 11. T4 fp64 throughput lands in the same range as this project's
  own laptop CPU, so a naive port would not pay for its transfer/launch
  overhead. Untested on A100/H100-class fp64 rates.
* Minimal-fix policy honored: no device rewrites were made in this sprint;
  the only GPU-relevant improvement shipped is that `lin_reg_tensor_batched`
  propagates dtype/device correctly by construction.

## Phase 14: device remediation applied

The FFNN device work is no longer deferred:

1. Non-CPU `get_linspace` calls use the device-safe batched grid.
2. Both regression implementations keep `gelsd` on CPU and return the result
   to the input device.
3. Legacy corner and boundary checks use `torch.stack` instead of constructing
   CPU tensors from device scalars.
4. `traceify`, zonotope expansion, dual intervals, hyper-dual intervals and
   hyper-dual zonotope conversion inherit device and dtype.
5. `get_lipschitz.py` supports CPU, CUDA and MPS, selects float32 for MPS,
   accepts an explicit dtype and synchronizes the selected CUDA device.
6. `check_device_plumbing.py --selector-matrix` covers all 16 selector
   combinations. Both complex and real-cubic matrices pass on MPS.
7. The actual full `get_lipschitz.py` MPS path passes for 3layer, 4layer,
   5layer and big, including zonotope and interval analyses.

The 0-dimensional CPU constants in `inverse_poly_tensor` remain intentionally
unchanged: PyTorch permits scalar CPU tensors in CUDA/MPS arithmetic, and both
MPS selector matrices exercise that path.

The separate CNN compatibility issue (`np.product` in `conv.py`) remains
outside this FFNN GPU remediation.

## Phase 15: full A100 correctness matrix

The final matrix ran on an NVIDIA A100-SXM4-80GB on 2026-10-08. It covered
3layer, 4layer, 5layer and big; all 30 selected images; and all 16 epsilon
values. The reported values are per-network/per-epsilon averages over the 30
images.

All 28 summary checks passed:

| check | worst full-run result |
|---|---:|
| real-cubic float64 precise CPU/CUDA relative difference | `1.786e-15` |
| real-cubic float64 zonotope CPU/CUDA relative difference | `4.792e-16` |
| real-cubic float64 interval CPU/CUDA relative difference | `5.517e-16` |
| real-cubic float32 precise CPU/CUDA relative difference | `8.318e-7` |
| repeated CUDA run | exactly identical |
| full versus precise-only CUDA | exactly identical |

The dtype-specific limits were `1e-9` for float64 and `1e-5` for float32.
The first draft used `1e-9` for both, which is below float32's practical
precision; the protocol was corrected explicitly rather than hiding the
initial failed check.

The obsolete complex solver remained a strong negative control. Its CPU/CUDA
relative divergence increased with network depth: 14.39% (3layer), 33.42%
(4layer), 48.72% (5layer), and 63.78% (big). The real-cubic solver removes
that device-dependent root filtering and restores float64 agreement to about
machine precision.

Raw CSV files and complete logs are stored in
`results/gpu_hybrid_verification_2026-10-08/`.

This phase establishes cross-device correctness and reproducibility, not
speed. The matrix does not report end-to-end timing, and its 30-image values
are averages rather than per-image maxima. Controlled CPU/A100 timing and a
per-image discrepancy sweep remain separate follow-up experiments.
