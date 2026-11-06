# Neptune and Pasado: does its reduction fusion apply?

2026-11-06. Sources: Neptune at commit `f797501` (2026-06-18), cloned with `git clone --depth 1`
and only read (nothing was built or run); the PLDI 2026 paper (arXiv 2510.08726v2, read through
a summarizing web fetch, so section references are not verbatim checks); Pasado at `1d754c9`
(branch `checkpoint`); the archived TorchInductor output in
`results/compiled_sigmoid_v3_cuda_2026-10-29/kernel_code/`. **[V]** marks what was verified in
code or archived output, **[I]** marks inference. Neptune paths are relative to its repository.

## 1. What Neptune fuses

- **Mechanism [V].** Neptune is a fork of TVM 0.18.0 (`version.py:47`) that adds two TIR
  schedule primitives, `rolling_update` and `split_k_update`
  (`python/tvm/tir/schedule/schedule.py:2610-2631`; C++ in
  `src/tir/schedule/primitive/reduction.cc:1389-1510`). Both fuse a reduction `b0` into the
  *tiled* loop `l` of a producer reduction whose complete result `b0` needs, which classic fusion
  forbids. `rolling_update` turns the producer into a scan and repairs `b0` with a correction
  term per tile (FlashAttention). `split_k_update` keeps per-tile partial results and combines
  them afterwards, so `l` stays parallel (FlashDecoding) (`src/tir/schedule/primitive.h:563-679`).
  The canonical chain is max, then exp(x - max), then sum, then P·V, all over the same key axis.
  The repo's workloads are attention variants and softmax
  (`scripts/neptune_bench/operators.py:90-142`, `scripts/neptune_bench/ours/softmax.py`). The
  paper adds L2 norm, RMSNorm with quantization and Performer, with reduction lengths from 1024
  to 131072 (§7).
- **When the rewrite exists [V].** The correction is derived with sympy. The consumer's map must
  expand to a pure sum or product whose terms each depend either only on the incomplete values or
  only on the rest (`python/tvm/arith/sympy_tools.py:241-299`; mixed terms are rejected at line
  288). The combiner must be invertible (302-321), and the correction must commute with the
  reducer, which is proved symbolically (`reduction.cc:1818-1878`). The converter knows `+ - * /`,
  casts and exp/log/sqrt only (`sympy_tools.py:56-110`): there is no abs, max/min or select. The
  paper names mean and variance (Welford) as a case its solver cannot repair (§4.2). In the code,
  a failure is an ICHECK error (`reduction.cc:1853, 1874`). The header also documents that the
  rewrite gives wrong results when another block reads the fused intermediates
  (`primitive.h:602-607`).
- **When it is not needed [V].** If one block processes a whole row, plain `compute_at` fusion
  is enough. Neptune's own `schedule_softmax_0` does exactly that without any Neptune primitive
  (`ours/softmax.py:37-54`).
- **Frontend and preconditions [V].** The input is TVM TE (`te.placeholder`/`te.compute`, then
  `te.create_prim_func`) plus a hand-written TIR schedule, or a MetaSchedule search over one
  (`python/tvm/neptune/build.py:103-153`). There is no PyTorch frontend. Generated Triton kernels
  can be wrapped for PyTorch with `torch.library.wrap_triton` (`build.py:44-63`, which carries
  "TODO: multiple kernels need to compose sometimes"). Shapes are concrete integers in all the
  examples (`ours/attn_plain.py:118-122`), and the tiling helpers read integer loop extents and
  assert divisibility (`python/tvm/neptune/sch_utils.py:22-31, 85-86`). The Triton code generator
  maps float64 to `tl.float64` (`src/target/source/codegen_triton.cc:56-67`), but its matmul
  accepts only fp32 x fp32 -> fp32, fp16 and fp8 (`codegen_triton.cc:429-470`). Its only math
  calls are exp, rsqrt and tanh, and any other call is fatal (`codegen_triton.cc:205-213`). The
  tests and benchmarks use fp16 inputs with fp32 accumulation
  (`tests/python/tir-schedule/test_tir_schedule_rolling_update.py:9-18`). The paper uses fp64 only
  as an accuracy baseline (§8.4).
- **Installation [V].** You either build the whole TVM tree from source (CMake >= 3.18, LLVM 15
  to 19, CUDA >= 12; `README.md:19-26, 61-88`) or use the Docker image
  `evanzhao16/neptune-env:nvidia-latest` (`README.md:37`). The dev extras pin `torch == 2.6.0` and
  `triton == 3.2.0` (`pyproject.toml:58-59`), while Pasado's A100 runs use torch 2.11 and Triton
  3.6 (v3 README §1). It runs on any GPU that Triton supports, and the benchmark runner also
  handles ROCm (`ours/runner.py:53`). The README mentions no wheel or release.

## 2. Reductions in the Pasado paths

**Shapes [V].** A sigmoid layer fits S·N boxes at once (`batched_splits.py:190-191`), up to
64 x 1024 = 65536. Each sigmoid layer adds **2N** noise symbols, not N
(`batched_splits.py:168, 211`). The generator axis G entering the sigmoid layers is therefore 2,
202, 402, 602 at width 100 and 2, 2050, 4098, 6146 for big (`steady_kernels.csv` of the v3 run).

**Two corrections to the framing [V].** First, the 25-point fit splits into 5-element
reductions along each grid axis, and only the mean of z spans all 25 points. Second, the
worst-case error is not a maximum over the 25 grid points. It is taken over the 4 box corners and
up to 3 + 3 analytic candidates on the edges y = ly and y = uy.

| # | Location | Reduced | Axis (length) | Consumer | Dependent reduction, same axis? | Neptune |
|---|---|---|---|---|---|---|
| 1 | `planar_fit.py:56` `_centred` | grid x (or y), mean | grid axis (5) | `dx = r - rm`, `xbar` | starts chain (rows 2, 3) | n/a |
| 2 | `planar_fit.py:68` `_pow2_scale` | max \|dx\| | same (5) | frexp/ldexp scale `ax`, `dxn = dx/ax` | yes, on row 1 | no: abs, frexp/ldexp unsupported; row fits in registers |
| 3 | `planar_fit.py:84-85, 89-90` | sum dxn², sum dx² | same (5) | spreads, gelsd cutoff, slope denominators, degeneracy mask | yes, on rows 1-2 | no: centred sum of squares is the Welford case |
| 4 | `planar_fit.py:137` | z, mean | all 25 points | intercept k (elementwise) | no | n/a |
| 5 | `planar_fit.py:138, 140` | z minus pivot row or column, mean | one grid axis (5) | px, py [n, 5] | feeds rows 6-7 over the other grid axis | n/a |
| 6 | `planar_fit.py:139, 141, 142` | px (py) mean, then max \|.\| | other grid axis (5) | scales sx, sy | yes (mean, then max) | as row 2 |
| 7 | `planar_fit.py:146-149` | sum dxn·px/sx | grid axis (5) | slopes a0, b0, then k, then `_min_norm` gives (C, A, B) | yes, on rows 1, 2, 6 | as row 3 |
| 8 | `precise_transformer.py:355-356` `check_corners_tensor` | residual, max and min | 4 corners | `max(row_max, -row_min)` | no | no |
| 9 | `precise_transformer.py:454` (called 489, 502) | \|residual\| at Viete roots | 3 candidates | `torch.maximum` with row 8 (`batched_splits.py:204`) | no | no |
| 10 | `batched_splits.py:143-144` `affine`; `SimpleZono.py:173, 178` | GEMM | input neurons (100 or 1024; the image for layer 1) | generators of the next zonotope (stored), row 11 | no (row 11 is over G) | no: fp64 `tl.dot` rejected |
| 11 | `batched_splits.py:148` `_bounds` (called 178, 179, 237); `SimpleZono.py:55` `get_coeff_abs` (called twice, 59 and 65) | sum \|generator\| | G (2 to 602, or 2 to 6146) | elementwise: sigmoid, lambda, mu1, mu2, box grid; lambda, A, B rescale the same G rows (212, 216) | no: the next G reduction comes after the next GEMM over N | no: a GEMM epilogue, no dependency to break, and the GEMM output must be stored anyway |
| 12 | `batched_splits.py:238` | max(\|lb\|, \|ub\|) | output neurons | bound per split | after row 11, other axis | no |
| 13 | `batched_splits.py:198, 234`; `SimpleZono.py:304` | boolean all | S·N; layers; N | host assert | no | no |

**No reductions [V]:** `gram_eigenvalues` (8 elementwise Newton steps, `planar_fit.py:154-192`),
`_min_norm` (105-122), `inverse_poly_real_tensor` (acos/cos, `precise_transformer.py:203-228`) and
the generator construction (`batched_splits.py:211-217`).

**What TorchInductor already does [V].** In the archived compiled `precise_sigmoid` (big,
S = 32, float64, `kernel_code/big_S32_batched_compiled_sigmoid/`), the graph launches 13 Triton
kernels and contains no GEMM, because the affine GEMMs run outside it (`batched_splits.py:230`).
Rows 1-4 and 6-7 plus the 8 Newton steps (unrolled) run in one persistent-reduction kernel
(`uo/cuohpa5....py`: `persistent_reduction`, `xnumel = 32768` boxes, `r0_numel = 25`). Inside it,
the targets are summed over the 25 points (line 67), and the 5-element reductions are unrolled
over per-box scalar loads (lines 34-55). Row 5 is one
pointwise kernel (`cz/...`, 163840 = 32768 x 5 outputs). Row 11 and the sigmoid, lambda, mu1 and
mu2 epilogue form one looped reduction over G (`yt/...`,
`triton_red_fused_abs_add_copy_diagonal_minimum_mul_rsub_sigmoid_slice_sub_sum_1`).

## 3. Verdict

**Does the pattern exist in Pasado? Partly [V].** `fit_planes` has reductions that need the
complete result of an earlier reduction over the same axis: a mean, then max |.|, then a sum of
squares or products (rows 1-3 and 6-7). In shape this is Neptune's pattern, but the axis has only
5 elements. On the long generator axis there is one reduction per tensor and layer, followed
only by elementwise work. The next reduction over G comes after the next layer's GEMM over N.

**Would it help? No.**
1. *Not needed [V].* Neptune breaks the dependency only to fuse across *tiles* of a long axis. A
   row of 5 or 25 elements sits in one thread's registers, ordinary fusion covers it, and
   Inductor already emits a single kernel for the whole fit. There is no locality left to gain:
   each box's 50 grid values and 25 targets are read once.
2. *Not applicable [V].* Max |.|, frexp/ldexp and the centred sum of squares (the Welford form)
   are outside the solver. The code generator lacks abs, sqrt, log, acos, cos, frexp/ldexp and fp64
   matmul.
3. *Not where the time is [I].* For big at S = 64, the affine GEMMs on the generators cost about
   2 (real and dual) x 2 x 64 x (2050 + 4098 + 6146) x 1024² = 3.3e12 fp64 flops per pass. At the
   A100's FP64 tensor-core peak of 19.5 TFLOP/s that is at least about 170 ms, against 220.6 ms
   measured per pass for the compiled whole sigmoid (v3 README §6). If this holds, three quarters
   or more of the pass is DGEMM, which Neptune cannot emit in fp64. No per-kernel time breakdown
   has been archived, so this is unmeasured. The width-100 networks take 2.4 to 4.8 ms per pass at
   S = 1 (3layer: 63 to 65 kernels), and their fusion is already done.

*Aside [I], unrelated to Neptune:* of the 2N rows that each sigmoid layer appends, N are diagonal
(mu2 or the errors) and N are zero (`batched_splits.py:211-217`). They go through the next dense
GEMM unchanged (`batched_splits.py:143-144`). For big, layers 2 to 4, that is 6144 of 12294 GEMM
rows, about half the flops. Replacing those products with row scalings of W and skipping the zero
blocks would do more than any reduction fusion. A profile should confirm the GEMM share first.

**Minimal experiment.** None is warranted for Neptune. The measurement that matters for the
decision is a per-kernel time profile of one big S = 64 pass (`torch.profiler`), which tests point 3.
If Neptune is tried anyway, the only candidate kernel is `fit_planes` at n = 65536 in float64. It
would be compared with eager (303 kernels), Inductor (5 kernels, 0.32 to 0.45 ms per call, or 0.22
to 0.25 ms with CUDA graphs; `results/inductor_dispatch_a3_2026-10-27/README.md` §1) and `jax.jit`
with x64 enabled. Porting it to TE would mean dropping the power-of-two scaling and the abs, both
unsupported. That changes the numerics the D contract checks, and the best expected result is
parity with Inductor [I]. As a second compiler for the whole sigmoid (the "TVM, XLA" open item of
the v3 README §14), JAX/XLA is much cheaper than Neptune: it takes Python, supports float64 and
needs no source build [I].

**Blockers.**
- Frontend: each kernel needs a manual rewrite in TE with a hand-written schedule. There is no
  PyTorch capture.
- float64: matmul is rejected, and no test or benchmark in the repo exercises elementwise fp64.
- Operators: sqrt, log, acos, cos, frexp/ldexp and abs are missing from the Triton code generator.
  Whether TVM legalizes some of them earlier was not determined.
- Shapes: shapes are static per compile, but G changes with every layer and S·N with every chunk
  size. Inductor handles G with `mark_dynamic`.
- Installation: it needs a source build or Docker, and its torch 2.6 / Triton 3.2 pins conflict
  with Pasado's environment.
- Soundness: in a verifier, the documented wrong-result case of `rolling_update`
  (`primitive.h:602-607`) would need its own validation.

**Not determined.** Three questions remain open: whether Neptune's primitives lower through TVM's
ordinary CUDA backend (bypassing the Triton path and its fp64 limits), whether TVM symbolic shapes
work with them, and the actual time split of a big pass. Nothing from Neptune was built or run.
