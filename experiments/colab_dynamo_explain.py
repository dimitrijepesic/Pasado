"""Enumerate the graph breaks in sigmoid_prime_product_tensor.

Compiling `check_nonlinear_boundary_tensor` alone already gave 5-30x
(colab_compile_boundary.py) with `linalg_lstsq` simply outside the compiled
region. The open question is what compiling the WHOLE transformer would cost and
gain, and that hinges on where Dynamo breaks the graph. Two breaks are expected:

  * `torch.linalg.lstsq` -- dynamic output shape, documented in
    TORCH_COMPILE_ANALYSIS.md as the reason the whole function was not
    compilable.
  * `np.random.normal(...)` in the A-coefficient perturbation -- a numpy call in
    the hot path. This one matters beyond fusion: if Dynamo constant-folds it
    instead of breaking, the compiled function would reuse FIXED perturbations
    on every call, which is a correctness bug rather than a missed
    optimization. The script checks that explicitly by calling the compiled
    function twice and comparing.

Reports, per compiled callable: number of graphs, number of breaks, and the
reason + source location of each break.

Colab (after the setup cells used for compile_boundary.py):
    !python Pasado/experiments/colab_dynamo_explain.py
"""
import os
import sys

import torch

torch.set_default_dtype(torch.float64)


def find_src():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "forward_mode_tensorized_src"),
                 os.path.join(os.getcwd(), "Pasado", "forward_mode_tensorized_src"),
                 os.path.join(os.getcwd(), "forward_mode_tensorized_src")):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    raise FileNotFoundError("forward_mode_tensorized_src not found")


sys.path.insert(0, find_src())
import precise_transformer as pt          # noqa: E402
from SimpleZono import IntervalsToZonotope  # noqa: E402


def make_zonotopes(n, seed=17):
    g = torch.Generator().manual_seed(seed)
    x_lb = torch.rand(n, generator=g) * 6 - 3
    x_ub = x_lb + torch.rand(n, generator=g) * 2 + 1e-3
    y_lb = torch.rand(n, generator=g) * 4 - 2
    y_ub = y_lb + torch.rand(n, generator=g) * 2 + 1e-3
    return (IntervalsToZonotope(x_lb, x_ub), IntervalsToZonotope(y_lb, y_ub))


def report_breaks(label, fn, args):
    import torch._dynamo as dynamo
    dynamo.reset()
    print(f"\n=== {label} ===")
    try:
        expl = dynamo.explain(fn)(*args)
    except Exception as e:
        print(f"  explain FAILED: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        return
    print(f"  graphs: {expl.graph_count}   graph breaks: {expl.graph_break_count}   "
          f"ops captured: {expl.op_count}")
    reasons = getattr(expl, "break_reasons", []) or []
    for i, r in enumerate(reasons, 1):
        reason = getattr(r, "reason", str(r))
        # user_stack holds the frames inside our own code, which is what tells
        # us WHICH line to change.
        frames = getattr(r, "user_stack", []) or []
        where = ""
        for f in reversed(frames):
            fn_name = getattr(f, "filename", "")
            if "precise_transformer" in fn_name or "SimpleZono" in fn_name:
                where = f"  at {os.path.basename(fn_name)}:{getattr(f, 'lineno', '?')}"
                break
        print(f"  break {i}: {str(reason)[:150]}{where}")


def check_randomness_survives_compile(n=64):
    """Does the compiled transformer still draw fresh perturbations?

    sigmoid_prime_product_tensor perturbs coefficient A with np.random.normal.
    If Dynamo folds that into a constant, two calls would return identical
    results -- silently freezing a value the analysis expects to be random.
    Two eager calls differ, so two compiled calls must differ too.
    """
    print("\n=== does np.random survive compilation? ===")
    pt.USE_VECTORIZED_PRECISE = True
    pt.USE_BATCHED_LSTSQ = True
    pt.USE_BATCHED_GRID = True

    def run(fn):
        x, y = make_zonotopes(n)
        return fn(x, y).generators.clone()

    eager_a, eager_b = run(pt.sigmoid_prime_product_tensor), run(pt.sigmoid_prime_product_tensor)
    eager_differs = not torch.equal(eager_a, eager_b)

    compiled = torch.compile(pt.sigmoid_prime_product_tensor)
    try:
        comp_a, comp_b = run(compiled), run(compiled)
    except Exception as e:
        print(f"  compiled call FAILED: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:160]}")
        return
    comp_differs = not torch.equal(comp_a, comp_b)

    print(f"  two eager calls differ    : {eager_differs}")
    print(f"  two compiled calls differ : {comp_differs}")
    if eager_differs and not comp_differs:
        print("  -> WARNING: compilation froze the random perturbation. "
              "That is a correctness change, not a performance question.")
    elif eager_differs and comp_differs:
        print("  -> randomness preserved across compilation.")


def main():
    print(f"torch {torch.__version__}  "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    if not hasattr(pt, "inverse_poly_real_tensor"):
        print("fetched precise_transformer.py predates PASADO_REAL_CUBIC; "
              "re-download from colab-gpu-wip")
        return

    n = 100
    saved = (pt.USE_VECTORIZED_PRECISE, pt.USE_VECTORIZED_BOUNDARY,
             pt.USE_BATCHED_LSTSQ, pt.USE_BATCHED_GRID, pt.USE_REAL_CUBIC)
    try:
        pt.USE_VECTORIZED_PRECISE = True
        pt.USE_VECTORIZED_BOUNDARY = True
        pt.USE_BATCHED_LSTSQ = True
        pt.USE_BATCHED_GRID = True

        # The kernel already proven to compile cleanly, as a control: it should
        # report zero breaks, which validates that a nonzero count elsewhere is
        # a real finding and not a harness artifact.
        g = torch.Generator().manual_seed(3)
        lx = torch.rand(n, generator=g) * 6 - 3
        ux = lx + torch.rand(n, generator=g) * 2 + 1e-3
        ly = torch.rand(n, generator=g) * 2 - 1
        ly = torch.where(ly.abs() < 0.05, torch.full_like(ly, 0.05), ly)
        uy = ly + torch.rand(n, generator=g) + 1e-3
        ABCs = torch.stack([torch.rand(n, generator=g) * 2 - 1 for _ in range(3)], dim=1)

        for real_cubic in (False, True):
            pt.USE_REAL_CUBIC = real_cubic
            tag = "real cubic" if real_cubic else "complex64"
            report_breaks(f"check_nonlinear_boundary_tensor ({tag})",
                          pt.check_nonlinear_boundary_tensor,
                          (lx, ux, ly, uy, ABCs))

        pt.USE_REAL_CUBIC = True
        report_breaks("sigmoid_prime_product_tensor (whole transformer)",
                      pt.sigmoid_prime_product_tensor, make_zonotopes(n))

        check_randomness_survives_compile()
    finally:
        (pt.USE_VECTORIZED_PRECISE, pt.USE_VECTORIZED_BOUNDARY,
         pt.USE_BATCHED_LSTSQ, pt.USE_BATCHED_GRID, pt.USE_REAL_CUBIC) = saved


if __name__ == "__main__":
    main()
