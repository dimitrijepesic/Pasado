# What default compilation by different tools gives on Pasado kernels (protocol)

Written 2026-11-06, before the run. Code: `experiments/compilers/`. Notebook (A100):
`colab/02_compilers.ipynb`. Local part: Apple M1 Pro CPU (same scripts, `--device cpu`).

## Question (from the mentor)

What does plain, default compilation with different tools give on Pasado, where do the
compilers fuse operators and where do they stop (graph breaks, unsupported operations,
float64, complex, lstsq, dynamic shapes)?

## Kernels (isolated, float64, inputs fixed by seed 0)

| id | what | taken from | sizes |
|---|---|---|---|
| `fit` | closed-form plane fit A2 | `planar_fit.fit_planes` on grids of `get_linspace_batched` and the sigmoid targets of `precise_sigmoid` | n = 1024 and 65536 boxes, 25 points |
| `viete` | real cubic roots and their inverse sigmoid | `precise_transformer.inverse_sigmoid_2nd_deriv` with `USE_REAL_CUBIC` | n = 65536 |
| `bounds` | affine map of the generators and the zonotope bounds | `batched_splits.affine` + `_bounds` (`rg @ W`, then `abs().sum` over the generators) | small: S=64, G=202, N=100; big: S=8, G=2050, N=1024 |
| `complex_cubic` | legacy Cardano roots in complex64 | `precise_transformer.inverse_poly_tensor` | n = 65536 |
| `lstsq` | batched regression with gelsd on the CPU | `precise_transformer.lin_reg_tensor_batched` | n = 1024 |
| `sigmoid` | the whole batched precise sigmoid transformer | `batched_splits.precise_sigmoid` on the first sigmoid layer input of a real image | 3layer S=1; big S=8 |

The JAX ports (`experiments/compilers/kernels_jax.py`) cover `fit`, `viete`, `bounds` and
`complex_cubic`; they are line-by-line translations and are checked against the eager PyTorch
reference before any timing. `lstsq` and `sigmoid` are PyTorch only (JAX column: "not ported").

## Tools and modes

- PyTorch eager (the reference).
- `torch.compile` with Inductor, `fullgraph=False`, modes `default`, `max-autotune`,
  `reduce-overhead` (CUDA graphs; on the CPU this mode has no CUDA graphs and is reported as
  such). Graph breaks from `torch._dynamo.explain`.
- JAX/XLA: `jax.jit`, `jax_enable_x64=True`, default XLA flags.
- TVM: at most 60 minutes for an installation with CUDA support on Colab; if it imports, the
  `fit` kernel through `torch.export` and the Relax PyTorch frontend. Every step and its error
  is recorded; a failure is a result.
- Neptune: not run - `docs/NEPTUNE_ANALYSIS.md` found that its reduction pattern exists only
  over 5- to 25-element axes and that its solver does not handle `abs`/`max` or float64 matmul.

## Measured per (tool, mode, kernel, size, device), each in a fresh process

1. Correctness against eager float64: largest normwise relative difference
   max|a - b| / max|b| over finite entries, and whether the NaN patterns are identical
   (the `viete` outputs are NaN where no real root exists).
2. Compilation: first-call time in a process with an empty cache (cold) and in a second
   process sharing that cache (warm): `TORCHINDUCTOR_CACHE_DIR` for Inductor, the persistent
   compilation cache for JAX.
3. Steady state: 10 warm-up calls, then the median (and min) of 50 timed calls. CUDA: CUDA
   events for PyTorch; for JAX wall time around `block_until_ready` (JAX exposes no CUDA
   events) - both are also given as wall time, so the two tools can be compared on one
   clock. CPU: wall time.
4. Kernels: CUDA - launched kernels per call from `torch.profiler` (PyTorch eager and
   compiled); JAX - instructions in the entry computation of the optimised HLO that are not
   parameters, constants, tuples or bitcasts (each is one fusion or library call, i.e. about
   one launch on a GPU; labelled as an HLO count). CPU - generated kernels in Inductor's
   output code (`run_and_get_code`) and the same HLO count for JAX.
5. Dynamic shapes: the compiled function is called with two more sizes (2x and 3x n);
   recorded: recompilations (Dynamo frame count; JAX cache size) and the first-call times.

No decision rule: the result is the table tool x kernel and a short text on where fusion
works, where it stops and why. Numbers from different GPU models are not pooled.
