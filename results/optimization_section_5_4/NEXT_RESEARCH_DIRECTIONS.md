# NEXT_RESEARCH_DIRECTIONS - Section 5.4 precise transformer

> **Measured before the soundness fixes of 2026-10-21/22 (`b526762`); marked 2026-11-06.**
> Every number here comes from the code before `Zonotope.__neg__` / `DualZonotope.__neg__`
> negated the generators and before the boundary checks bounded `|E|` instead of `E`.
> Speedups, call counts and profiles compare two implementations of that same pre-fix
> code; the fixes add one elementwise `abs` per boundary check and a sign flip, so the
> ratios are not expected to move, but they have **not been re-measured**. Lipschitz bound
> values quoted here are **not valid** after the fix. Hardware measurements (GFLOP/s,
> `gels` against `gelsd`) do not depend on Pasado's code. Current state: `CHECKPOINT.md`.

## Status of every direction on 2026-11-06

This list was written at the end of the July 2026 CPU sprint. What has happened to each item
since then (details in `FORMAL_SUMMARY.md` and the dated folders under `results/`):

| Item | Status | Where |
|---|---|---|
| 1.1 batch `get_linspace` | **done** (2026-08-12), bit-identical, `PASADO_BATCHED_GRID`; the ~1 ULP caveat below was wrong (artifact of the naive formula) | `FORMAL_SUMMARY.md` |
| 1.2 `unbind` round-trip in `compute_max_error` | not done on the sequential path; the batched path (`batched_splits.precise_sigmoid`) combines the three checks with `torch.maximum` and has no round-trip | `forward_mode_tensorized_src/batched_splits.py` |
| 1.3 re-measure `big` | **done** (2026-08-06): 1.650x, interleaved, pinned, n = 3 | `FORMAL_SUMMARY.md` |
| 2.1 fewer noise symbols | open (changes the analysis; needs a decision) | - |
| 2.2 GPU execution | **done in large part**: device plumbing, CPU `gelsd` bridge, real cubic (the complex64 path is wrong on CUDA), batched splits, closed-form A2 plane fit chosen by a pre-registered rule, automatic gelsd/A2 choice (opt-in). Results before 2026-10-22 are pre-fix | `GPU_READINESS.md`, `results/gpu_hybrid_*`, `results/lstsq_comparison_*`, `results/planar_fit_cuda_2026-10-27/` |
| 2.3 `torch.compile` | **done on CUDA**: compiled `fit_planes` (opt-in), compiled whole `precise_sigmoid` (experiment, outcome opt-in); CPU/MSVC path still not tried | `results/v3_compiled_cuda_2026-10-28/`, `results/compiled_sigmoid_v3_cuda_2026-10-29/` |
| 2.4 float32 / mixed precision | partly: float32 CPU/CUDA agreement 8.3e-7 (pre-fix), known float32 max-error shortfall near sigmoid roots (`experiments/reproduce_f32_root_shortfall.py`); no study of per-stage precision | `results/gpu_hybrid_verification_2026-10-08/` |
| 3.1 Triton | Inductor now generates Triton kernels for the compiled paths (archived in `results/compiled_sigmoid_v3_cuda_2026-10-29/kernel_code/`); no hand-written kernel | - |
| 3.2 Inductor graph analysis | in progress: part of the compiler comparison of the November checkpoint | `results/compilers_2026-11/PROTOCOL.md` |
| 3.3 JAX frontend | in progress: isolated kernels ported for the compiler comparison | `results/compilers_2026-11/PROTOCOL.md` |
| 3.4 Neptune | static analysis of whether its patterns exist in Pasado | `docs/NEPTUNE_ANALYSIS.md` |
| 3.5 cross-hardware study | open (T4, A100 and Apple CPU/MPS measured separately; no common protocol) | - |
| rejected: closed form | still rejected on the CPU; **chosen on CUDA** (A2), where `gelsd` does not exist | `results/lstsq_comparison_2026-10-27_v3_zpivot/` |
| rejected: `torch.compile` | still rejected on the Windows CPU; **works on CUDA** (opt-in) | see 2.3 |

---

Ranked by how directly the sprint's measurements support them. A deliberate
distinction is kept throughout between **implementation optimization** (same
analysis, less time) and a **publishable research contribution** (new analysis
capability, better bounds, or a transferable technique).

---

## Tier 1 - directly justified by the profile

### 1.1 Batch `get_linspace` (the current top precise-path bottleneck)

Evidence: after batching the regressions, `get_linspace` is ~44 % of
`sigmoid_prime_product_tensor`'s cumulative time on 3layer; torch.profiler
shows `select` + `linspace` + `meshgrid` + `item` as the dominant remaining
operators, with call counts **identical across all three variants** (we never
touched them): 192 000 `linspace`, 96 000 `cartesian_prod`, 385 456 `.item()`
per 3layer benchmark; ~2x/4x those on `big`.

Design: build all n grids in one shot from the bound vectors - no `.item()`,
no Python loop - via `lx.unsqueeze(1) + (ux-lx).unsqueeze(1) * t` and a
`repeat_interleave`/`repeat` cartesian assembly.

**Caveat that makes this research-flavored, not mechanical:** the cartesian
assembly is bit-identical (verified, 0/2000 mismatches), but batched linspace
is **not** - it differs from `torch.linspace` by ~1 ULP in ~14 % of cases
(280/2000). So this optimization requires an explicit, documented precision
decision. Recommended framing: implement behind a selector, quantify the
end-to-end bound difference on the reduced harness, and report it as
"1-ULP grid perturbation changes final Lipschitz bounds by < X" rather than
silently swapping it in. Expected payoff: the largest remaining precise-path
block.

### 1.2 Remove the residual `unbind`/list round-trip in `compute_max_error`

Evidence: 14.9 s self + 5.6 s `unbind` on the big cProfile. The vectorized
checks produce `[n]` tensors, which are unbound into Python lists only to be
recombined in a Python loop. Keeping tensors throughout and combining with two
`torch.maximum` calls is bit-exact and mechanical. Small but free.

### 1.3 Re-measure `big` under controlled conditions

Evidence: two runs in this sprint were corrupted by background system activity
(a `big vec_both` run at 2514 s that cannot be explained by code - the entire
footprint of the changed functions is < 60 s in the profile - and a 3layer
cProfile run at 2998 s vs the expected ~22 s). The `big` speedup claim is
currently the weakest link in the results. Needs: idle machine, several runs
if affordable, or at minimum a machine-health probe (a fixed matmul benchmark)
recorded alongside each run.

---

## Tier 2 - plausible, needs an experiment first

### 2.1 Reduce the number of noise symbols (algorithmic, changes semantics)

Evidence: `AffineZonotope` is **416 s self on big - 33 % of everything**, and
it is genuine BLAS, not Python overhead. It scales as width^2 x #generators,
and the precise transformer *adds fresh error symbols at every sigmoid*, so
generator count grows layer by layer. No amount of vectorization touches this.
The only real levers are algorithmic: symbol merging / order reduction, with a
precision-vs-cost tradeoff. This is the one direction that could plausibly
change the asymptotics of the whole analysis - and it is a genuine research
question (how much tightness is lost per symbol removed?), not an
implementation detail. Out of scope for this sprint by rule ("do not change
the mathematical algorithm"), but it is where the time actually is.

### 2.2 GPU execution - both open questions now measured (Phases 11-13)

Evidence in `GPU_READINESS.md`. Both hard facts from the original write-up
have since been measured, not just anticipated:

(a) **Driver**: `driver='gelsd'` does not exist on CUDA - the only CUDA
driver is `gels`. Phase 12 measured this directly on rank-deficient boxes
(the actual `lx==ux` case, not a hypothetical): `gels` diverges from `gelsd`
by 15-34 *orders of magnitude* - confirmed unsafe, not just unproven. Any GPU
regression path needs a rank check routing degenerate rows to a CPU/SVD
fallback.

(b) **fp64 throughput is hardware-dependent, now measured on both ends**:
T4 (Phase 11) runs fp64 at ~1/17 of fp32 - on par with a laptop CPU, so the
dominant `AffineZonotope` matmul would not meaningfully speed up there. A100
(Phase 13) runs fp64 at ~parity with fp32 via dedicated FP64 Tensor Cores  -
~65x the CPU rate. The matmul *would* plausibly win big on A100/H100-class
hardware; it would not on T4/L4.

**Revised next step**: no longer blocked on open questions. The remaining
gap is reliable access to A100/H100-class compute (Colab's allocation isn't
guaranteed) and the engineering to route the regression step's rank-deficient
rows around `gels`. Both are now well-scoped, not speculative.

### 2.3 torch.compile, after installing MSVC

Evidence: `TORCH_COMPILE_ANALYSIS.md`. Dynamo captures the vectorized check
kernels with **0 graph breaks**, but the Inductor CPU backend could not run at
all (`cl.exe` missing), `linalg_lstsq` graph-breaks anyway (dynamic output
shape), and Inductor cannot codegen the `complex64` root solve. Also, the
cleanly-capturable kernels are now too cheap (< 1 ms at n=1024) to move the
end-to-end number. Revisit only after 1.1 creates a big pure-elementwise
kernel worth fusing.

### 2.4 float32 / mixed precision as an explicit study

Not attempted (would change precision). Worth noting that the root solve
*already* passes through `complex64`, i.e. the cube roots are effectively
computed in single precision today - a latent precision property of the
original algorithm that the dtype bug found in this sprint brought to light.
A study of "what precision does each stage actually need?" would be defensible
and could unlock both GPU throughput and fp32 BLAS on CPU.

---

## Tier 3 - longer-term research directions

### 3.1 Triton kernels - only after a stable pure-tensor kernel exists

Premature today: the natural Triton target would be the grid-generation +
objective-evaluation kernel, which does not exist as a single fused kernel
until 1.1 lands. Sequence: batch the grid -> confirm it is a stable
elementwise-plus-reduction shape -> only then consider hand-written Triton,
and only on GPU where the fp64 question (2.2) has been settled.

### 3.2 TorchInductor graph analysis as a source of insight, not just speed

Even without a working backend, inspecting the captured FX graphs for the
precise transformer would document exactly which parts of an abstract
interpreter are fusible and which are inherently data-dependent
(`linalg_lstsq`, NaN-filtered root selection). That characterization is
transferable to other abstract-interpretation frameworks and is closer to a
paper contribution than a speed number.

### 3.3 JAX / JAXPR frontend

The precise transformer is a natural `vmap` target: the per-neuron regression
and error checks are exactly "the same computation over n independent
instances", which is what this sprint hand-vectorized. A JAX frontend would
get that structure for free, plus XLA fusion and straightforward GPU/TPU
execution. The interesting research question is whether the whole
zonotope/dual-number machinery can be expressed so that `vmap` + `jit` recover
the hand-written batching automatically - i.e. whether the manual work in this
sprint was avoidable in principle.

### 3.4 Comparison with fusion/locality ideas from Neptune

The bottleneck structure found here (many small per-neuron kernels whose cost
is dispatch, sitting between two large dense BLAS operations) is precisely the
setting where Neptune-style fusion and locality arguments apply. A principled
comparison - hand vectorization vs compiler fusion vs a `vmap`-style frontend,
on the same abstract-interpretation workload - would be a stronger
contribution than any single speedup number.

### 3.5 Cross-hardware study

Once 2.2 is settled: same benchmark on CPU / consumer GPU / datacenter GPU,
reporting how the balance between dense-BLAS cost and per-neuron orchestration
shifts. The sprint already showed that this balance **changes with network
size** (3layer is dominated by orchestration, `big` by dense matmul); showing
it also changes with hardware would make the point sharper.

---

## Explicitly rejected (documented negative results)

* **Caching** grids / design matrices - measured 0 % exact-repeat rate
  (400/400 distinct); a value-keyed cache can never hit. `CACHING_ANALYSIS.md`.
* **Closed-form / normal-equations / pinv** replacement for `lstsq`  -
  unnecessary (batched `gelsd` is already native and bit-exact) and worse
  conditioned. `LSTSQ_ANALYSIS.md`.
* **torch.compile in the production path** - backend unavailable, graph breaks
  where it matters, target kernels too cheap. `TORCH_COMPILE_ANALYSIS.md`.
