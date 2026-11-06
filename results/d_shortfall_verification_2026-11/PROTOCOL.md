# Is it true that D falls short of the exact maximum by up to 2.4 ulp in 51% of boxes? (protocol)

Written 2026-11-06, before the check. Script: `experiments/verify_d_shortfall.py`. Local,
Apple M1 Pro CPU; no GPU needed.

## The claim

`results/compiled_sigmoid_v3_cuda_2026-10-29/README.md`, section 5: on the 4992 boxes of the
pre-registered D-contract sample (A100, S = 64, eps index 8, post-fix code `a57b560`) the eager
error term D is below the exact maximum of |sigma'(x) y - (A x + B y + C)| over the box on
2562 boxes (51.3%), the compiled D on 2383 (47.7%); measured against the size of the terms,
0.25 |y| + |A x| + |B y| + |C| over the box, every shortfall is at most 2.4 ulp (median 0.2).

## Checks

1. **Recount from the archived file.** From `gate_d_contract.csv` recompute the number of boxes
   with D < D_true and the shortfall in ulp, for eager and compiled, with two definitions:
   (a) shortfall / (eps_64 * T) and (b) shortfall / ulp(T), T = max over the box of
   0.25 |y| + |A x| + |B y| + |C| (the maximum is at a corner: every term is a convex function
   of (x, y) on the box except 0.25|y|, which is maximal at an endpoint too).
   Pass: the counts equal 2562 / 2383 and the maximum and median ulp equal 2.4 / 0.2 to the
   printed precision under one of the two definitions (the README does not say which).
2. **Independent oracle.** For every box with a nonzero eager or compiled shortfall that is in
   the top 200 by shortfall over W, plus 200 random boxes (seed 2026): maximise the error on
   both edges y = ly, y = uy by a dense grid (2001 points) followed by golden-section
   refinement of the best three brackets, in mpmath with 50 digits, without the cubic roots
   of `lstsq_oracle.true_max_error`. Pass: the independent maximum never exceeds D_true by more
   than 1e-25 relative (D_true is a maximum), and the shortfall classification of every box
   is the same.
3. **Same boxes, CPU.** Recompute D from the archived eager A, B, C and box with the library
   code on the CPU (`check_corners_tensor`, `check_nonlinear_boundary_tensor` with the real
   cubic, as in `batched_splits.precise_sigmoid`), and report how many CPU values equal the
   archived (CUDA) eager D bit for bit, how many fall short of D_true, and the largest ulp.
   Descriptive: shows whether the shortfall is a CUDA artefact or a property of the algorithm.

## Outcome

"Verified" if 1 and 2 pass; otherwise the document states which number does not reproduce.
