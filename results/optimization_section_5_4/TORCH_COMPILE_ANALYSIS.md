# TORCH_COMPILE_ANALYSIS (Phase 8) — negative result on this platform

Experiment: `experiments/compile_experiment.py`, torch 2.12.1+cpu, Windows 11,
float64, candidates at n=100 and n=1024.

## Headline

**torch.compile could not be evaluated end-to-end on this machine, and two of
the four candidates would not benefit even if it could.** No compile path is
wired into the benchmark.

## 1. Which functions compile / graph-break (Dynamo capture)

Dynamo capture was measured with `torch._dynamo.explain` and *does* work
(capture is independent of the backend compiler):

| candidate | graph breaks | reason |
|---|---|---|
| `check_corners_tensor` | **0** | fully capturable |
| `check_nonlinear_boundary_tensor` | **0** | capturable, but see the complex-op caveat below |
| `_max_objective_over_x_candidates` | **0** | fully capturable |
| `lin_reg_tensor_batched` | **1** | `aten.linalg_lstsq.default` is a dynamic-shape operator (data-dependent output shape) — cannot be traced into the graph |

## 2. Backend failure (the blocker)

Every `torch.compile(...)` invocation failed at code generation:

```
InductorError: RuntimeError: Compiler: cl is not found.
```

TorchInductor's CPU backend generates C++ and needs MSVC `cl.exe`. Visual
Studio 2019/2022 directories exist on the machine but `cl.exe` is not on PATH
(no Build Tools workload / no developer shell). So **first-call latency and
steady-state compiled runtime could not be measured** — those columns are
genuinely unknown, not zero.

Fixing this would mean installing the MSVC C++ build tools and re-running from
a developer prompt. That is an environment change outside this sprint's scope;
it is recorded here as the concrete prerequisite for a future attempt.

## 3. Additional caveat found along the way

```
UserWarning: Torchinductor does not support code generation for complex operators.
```

`check_nonlinear_boundary_tensor` calls `inverse_poly_tensor`, which computes
cube roots through `complex64`. Even with a working MSVC, the complex section
would fall back to eager, so the realistic upside there is small.

## 4. Eager baselines (measured, useful independent of compile)

| candidate | n=100 | n=1024 |
|---|---|---|
| `check_corners_tensor` | 255.5 us | 717.7 us |
| `check_nonlinear_boundary_tensor` | 4 657.5 us | 9 015.4 us |
| `_max_objective_over_x_candidates` | 276.3 us | 832.6 us |
| `lin_reg_tensor_batched` | 3 251.8 us | 23 697.1 us |

Note the shape of these numbers: the two cheap kernels cost < 1 ms even at
n=1024, i.e. they are already a negligible slice of the ~19 s 3layer run (they
are called 960x -> under 1 s total). Compiling them could not move the
end-to-end number much even in the best case. The two expensive candidates are
dominated by `linalg_lstsq` and the complex root solve — exactly the parts
Inductor cannot generate code for.

## 5. Is torch.compile relevant for this workload?

**Not currently, and not obviously worth pursuing.** Reasons, in order of
weight:

1. The backend is unavailable (MSVC missing) — hard blocker today.
2. The regression kernel graph-breaks on `linalg_lstsq` anyway.
3. The complex-arithmetic kernel is unsupported by Inductor codegen.
4. The kernels that *do* capture cleanly are already too cheap to matter after
   Phase 3 vectorization.

The sprint's evidence says manual vectorization + batching (96 000 -> 960
lstsq calls, 64 738 -> 15 056 operator calls per forward) delivered the wins
that torch.compile is usually reached for. **This is an acceptable and
informative negative result**, consistent with the sprint brief.

## 6. If revisited later

Prerequisites, in order: (a) install MSVC Build Tools and verify `cl.exe`;
(b) target `get_linspace`-replacement kernels (pure elementwise, no complex, no
lstsq) — the current top bottleneck and the best-shaped compile candidate;
(c) measure first-call vs steady-state separately, as this script already does;
(d) on GPU, revisit with Inductor's Triton backend, where the calculus differs
(no MSVC dependency, and fusion matters more).
