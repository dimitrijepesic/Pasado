# Verification of the speedup ladder and of soundness (2026-11-15/16, M1 Pro, local)

A plain-language account in Serbian of how and why each check was done: [IZVESTAJ.md](IZVESTAJ.md).

Independent checks of the claims in [`../ladder_a100_40gb/README.md`](../ladder_a100_40gb/README.md) and the presentation. They recompute from scratch or read the archived files; none reuses the ladder's own report code.

- **Code:** transformer and batched sources unchanged since `91ebd88` (`git diff 91ebd88 -- forward_mode_tensorized_src Section_5_4` is empty).
- **Machine:** Apple M1 Pro, torch 2.12.1, float64.
- **Script:** [`experiments/verify_ladder_claims.py`](../../experiments/verify_ladder_claims.py), one JSON per check in this folder.

## Results

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | Existing test suite, 264-bound sweep vs. sampled max and vs. the 2026-10-22 archive | PASS | [`run_all_checks_summary.txt`](run_all_checks_summary.txt). The two compiled-path tests (`planar_cc`, `batched_cc`) failed in that run on a stale Inductor precompiled header from Nov 6 in the system temp folder (`CppCompileError`, the code fell back to eager). After deleting the cache both pass: [`compiled_tests_rerun.txt`](compiled_tests_rerun.txt). |
| 2 | Mutation check: 33 planted bugs, CPU | all 33 caught | Full run: [`mutation_check_cpu.log`](mutation_check_cpu.log). That run used duplicate ids for M16–M20 (fixed in `f89a5d5`), so the affected mutants and the new M30–M33 were rerun: [`mutation_recheck_M16-M20_M30-M33.log`](mutation_recheck_M16-M20_M30-M33.log). |
| 2b | Mutation check on CUDA (Colab A100-80GB, `f89a5d5`, `--device cuda` for the planar-fit and batched tests) | all 33 caught | [`mutation_check_cuda.log`](mutation_check_cuda.log). Baseline green. Same matrix as on the CPU except M10 (gelsy instead of gelsd): the `planar_fit` tests do not catch it on Colab, while `grid` and `lin_reg` do. The difference comes from the LAPACK library (macOS vs. Linux), not from CUDA: gelsd and gelsy agree on Linux on the degenerate-x boxes where they differ on macOS. |
| 2c | Unit tests on CUDA: batched splits (gelsd, closed form, compiled), planar fit (eager, compiled), automatic fit choice | PASS, 6/6 | [`gpu_tests_cuda.txt`](gpu_tests_cuda.txt), same Colab session as 2b. The batched tests compare the CUDA pass with the sequential CPU code. |
| 3 | `paths`: each ladder step runs the code it claims | PASS, 9/9 | [`paths.json`](paths.json); see [Code path per step](#code-path-per-step). |
| 4 | `gpu_time`: GPU busy time ≤ timed wall clock | PASS, 8/8 | [`gpu_time.json`](gpu_time.json); see [GPU busy time](#gpu-busy-time). |
| 5 | `dense`: bound ≥ a dense, refined search for max \|dF/dt\| | PASS, 72/72 | [`dense.json`](dense.json); see [Dense search](#dense-search). |
| 6 | `baselines`: plain zonotope and interval AD above the reference; Pasado below interval AD | PASS, 48/48 | [`baselines.json`](baselines.json); see [Baselines](#baselines). |
| 7 | `dplane`: D covers the exact max \|g − P\| of the plane each step used | PASS | [`dplane.json`](dplane.json); see [D for the plane actually used](#d-for-the-plane-actually-used). |
| 8 | Rounding study of D (`verify_d_shortfall.py`) rerun | identical | All 36 values equal the archive: [`d_shortfall_rerun.json`](d_shortfall_rerun.json) vs. [`../d_shortfall_verification_2026-11/result.json`](../d_shortfall_verification_2026-11/result.json). |
| 9 | The same bounds on M1 CPU and A100 GPU | PASS | 3layer and 5layer, 8 splits, steps 0, 3, 6, 7, 8, 4 pairs: largest relative difference 4.2e-12. |

### Code path per step

The check covers 3layer with 4 splits on the CPU. It counts calls to the selector-dependent functions:

- **Steps 0–4** use the per-neuron or batched gelsd, the loop or vectorized checks, and the complex or real cubic, exactly as the ladder table states.
- **Steps 5–7** use the batched pass with gelsd, the eager closed form, and the compiled closed form respectively.
- **The compiled fit really ran compiled:** `compile_fallback` is empty.
- **Step 8** runs one compiled graph.

### GPU busy time

The check reads `Self CUDA time total` from the kernel rerun's profiles.

| 64 splits | GPU busy, steps 5–8 | what it means |
|---|---|---|
| 5layer | 3–56% of the wall clock | launch- and host-bound |
| FFNNBig | 16% → 98% | after compilation the GPU is saturated |

### Dense search

100,001 uniform points on [0, ε], then 20 local refinements.

- **The 4,001-point reference was enough.** It is at most 0.0024% below the refined maximum.
- **Every bound clears the refined maximum.** The smallest bound / refined max is 1.000039.

### Baselines

- **Pasado is below interval AD** on all 48 (net, splits, pair).
- **Pasado is not always below the plain zonotope.** It is above it in 3 of 48 cases, by 0.02%, 0.63% and 0.09%. All three are at 8 splits and ε index 8, and the same holds for step 0, so the optimizations did not cause it.

### D for the plane actually used

The check covers steps 5, 6 and 7 with 8 splits on 3layer, 5layer and FFNNBig: 112,704 boxes.

- **What is computed.** The exact maximum of \|σ'(x)y − (Ax + By + C)\| over each box. It is linear in y, so the maximum lies on the y edges. It is found by a fine grid, and every box that looks short is redone in 60-digit arithmetic.
- **Largest shortfall: 1.99 ε·T.** T is the size of the terms D is computed from.
  - That is rounding: D is a small difference of terms near 1.
  - It is within the documented 2.36 ε·T.
  - The outward-rounded D covers every box.
- **Do not measure against D.** A float64 reference reported up to 4.7 ε·T; the 60-digit recheck shows that was the reference's own rounding. Measured relative to D, the same gap looks like 1e-9.

## CPU-only ladder on the M1 Pro ([`../ladder_m1pro_cpu/`](../ladder_m1pro_cpu/summary.md))

All steps on the M1 Pro CPU (steps 4–8 with `--device cpu`), commit `f89a5d5`. This answers how fast a strong CPU gets with the same code.

| 64 splits | original, M1 | best CPU step, M1 | best, A100 (3 runs) | A100 vs. M1 original | A100 vs. best M1 |
|---|---:|---:|---:|---:|---:|
| 5layer | 3,242 ms | 31.4 ms (step 8) | 4.6–5.8 ms | 560–700× | 5.4–6.8× |
| FFNNBig | 42,976 ms* | 7,320 ms* (step 8) | 231–239 ms | about 180× | about 31× |

- **The M1 Pro is 3–3.5× faster than the Colab Xeon** on the original code (3.4–3.6× on 5layer, 2.9–3.0× on FFNNBig). The ×-vs-original figures of the A100 ladder use that slower host.
- **Without a GPU the same optimizations give** 103× on 5layer and 5.9× on FFNNBig at 64 splits, on the M1 alone.
- **FFNNBig at 64 splits ran with fewer repeats** (`*`), limited by the time budget and 16 GB of RAM. Its batched steps 5 and 7 sit below steps 3 and 6, so read that row as an order of magnitude.
- **This is a cross-machine comparison** and is labelled as one.

## What these checks do not cover

- **Testing is not proof.** The checks cover 4 networks, 3 images, 2 ε indices and up to 64 splits.
- **Floating point.** D can be about 2 ε·T below the exact maximum, so without outward rounding the bound is not guaranteed sound in float arithmetic. The same holds for the original Pasado.
- **The transformer's mathematics** rests on the paper and the fix `b526762`; it was not re-derived.
- **`test_soundness.py` covers only the sequential path.** The batched-path mutants M31–M33 are caught by the batched-vs-sequential comparisons, and the batched bounds directly by checks 5 and 7 here.
- **`partial_rep3_unused/`** holds one cell (5layer, 1 split) of a third A100-40GB session that was stopped. It is too little to compare sessions with and is not used anywhere.
