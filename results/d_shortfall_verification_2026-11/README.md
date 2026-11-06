# Check of the D-shortfall claim (2026-11-06)

Protocol: `PROTOCOL.md` (written before the check). Script: `experiments/verify_d_shortfall.py`,
raw result: `result.json`, console output: `../../logs/verify_d_shortfall.log`.
Machine: Apple M1 Pro CPU, torch 2.12.1, mpmath 1.4.1, float64. Input: the 4992 boxes of
`results/compiled_sigmoid_v3_cuda_2026-10-29/gate_d_contract.csv` (A100-SXM4-80GB, code `a57b560`,
post-fix, S = 64, eps index 8, 3 images per network).

## Verdict

**The claim holds, with two corrections to how it is stated.**

| quantity | claim (v3 README, section 5) | recomputed |
|---|---|---|
| eager D below the exact maximum | 2562 of 4992 (51.3%) | 2562 of 4992 (51.32%) |
| compiled D below the exact maximum | 2383 (47.7%) | 2383 (47.74%) |
| largest shortfall, eager / compiled, in units of eps x T | "at most 2.4 ulps" | 2.352 / 2.356 |
| same, in units of ulp(T) | - | 3.897 / 3.950 |
| median shortfall, eps x T units | "median 0.2 ulps" | 0.196 over the 2562 boxes with a shortfall; 0.004 over all 4992 boxes |
| largest absolute shortfall, eager / compiled | 2.486e-14 / 2.494e-14 | 2.486e-14 / 2.494e-14 |
| largest shortfall over W, eager / compiled | 3.44e-9 / 3.86e-9 | 3.44e-9 / 3.86e-9 |

T is the largest value over the box of 0.25 |y| + |A x| + |B y| + |C|, the size of the terms the
error is computed from; eps = 2^-52. Correct wording: *the shortfall is at most 2.4 eps T (up to
3.9 ulp of T); among the boxes that fall short, the median is 0.2 eps T.* "ulp" without the
definition overstates the precision by up to a factor 2 (ulp(T) <= eps T).

## Checks

1. **Recount (pass).** The counts, the maximum and the median reproduce exactly under the
   eps x T definition; see the table.
2. **Independent oracle (classification passes; the tolerance as written does not).** On 400
   boxes (the 200 with the largest shortfall over W and 200 random ones, seed 2026), each side,
   the maximum was recomputed by a dense float64 grid (2001 points) and golden-section
   refinement of the three best brackets in mpmath (50 digits), without the cubic roots that
   `lstsq_oracle.true_max_error` uses. The shortfall classification (D below the maximum or not)
   is identical on all 800 cases. The independent maximum exceeds the archived D_true by at most
   1.07e-16 relative, above the pre-registered 1e-25: the archived D_true is stored as a
   float64 (`float(best)` in `lstsq_oracle.py`), so it carries up to half an ulp (2^-53 =
   1.11e-16 relative) of rounding, and the 1e-25 criterion could not be met by construction.
   The criterion was wrong, not the oracle; recorded as a failed criterion with this cause.
3. **The same boxes on the CPU (descriptive).** D recomputed on the CPU from the archived eager
   A, B, C and box with the library checks (`check_corners_tensor`,
   `check_nonlinear_boundary_tensor`, real cubic): 4564 of 4992 values equal the CUDA eager D bit
   for bit, the largest relative difference is 4.7e-9 (where D is tiny); 2560 boxes (51.3%) fall
   short of the exact maximum, by at most 2.35 eps T. **The shortfall is a property of the
   floating-point evaluation of the error term, not a CUDA or compiler artefact.**

## What it means

Pasado's D is computed in round-to-nearest with cancellation and no outward rounding, so in
about half of these narrow boxes it is a few eps T below the exact maximum. The sampled autograd
checks of the bound still pass because the shortfall (<= 2.5e-14 absolute, <= 3.9e-9 of the
neuron's radius W) is far below the slack of the bound. It is a formal soundness gap in
floating point, not an observed wrong bound. A preliminary outward-rounding prototype is in
`results/outward_rounding_prototype_2026-11/`.
