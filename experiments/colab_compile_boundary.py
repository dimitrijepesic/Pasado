"""Times check_nonlinear_boundary_tensor in eager mode and with torch.compile, with
the complex64 root solver and with the real cubic solver. The first-call compile
time is reported separately from the steady-state time."""
import argparse
import os
import statistics
import sys
import time
import traceback

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
import precise_transformer as pt  # noqa: E402


def make_inputs(n, device, regime, seed=1234):
    """Boxes and planes shaped like a real sigmoid layer's inputs.

    `regime` matters more than it looks. check_nonlinear_boundary_tensor divides
    A by ly/uy and keeps only quotients inside the image of sigma''
    (|.| < 1/(6*sqrt3)); everything else becomes NaN *before* the cubic solve.
    So the fraction of live candidates decides which arithmetic the root solver
    actually executes, and transcendental functions can be markedly slower on
    NaN operands than on ordinary ones.

    An early version of this script only generated the "in_image" regime (A
    scaled so most quotients are live) and reported the real cubic as 1.54x
    FASTER, while the end-to-end run had measured it 8 % SLOWER. Both regimes are
    therefore measured, and the disagreement is the finding rather than a bug.

      in_image  - A scaled into the image: most candidates live
      realistic - A unscaled, as the regression produces it: most filtered away
    """
    g = torch.Generator().manual_seed(seed)
    lx = (torch.rand(n, generator=g) * 6 - 3)
    ux = lx + torch.rand(n, generator=g) * 2 + 1e-3
    ly = (torch.rand(n, generator=g) * 2 - 1)
    ly = torch.where(ly.abs() < 0.05, torch.full_like(ly, 0.05), ly)
    uy = ly + torch.rand(n, generator=g) * 1 + 1e-3
    lim = 1.0 / (6.0 * 3.0 ** 0.5)
    if regime == "in_image":
        A = (torch.rand(n, generator=g) * 2 - 1) * lim * ly.abs()
    else:
        # Planar-regression A coefficients are O(0.1-1); divided by ly they
        # mostly land outside the sigma'' image, exactly as in production.
        A = torch.rand(n, generator=g) * 2 - 1
    B = torch.rand(n, generator=g) * 2 - 1
    C = torch.rand(n, generator=g) * 2 - 1
    ABCs = torch.stack((A, B, C), dim=1)
    to = lambda t: t.to(device)
    return to(lx), to(ux), to(ly), to(uy), to(ABCs)


def live_fraction(args):
    """Share of A/ly quotients that survive the sigma''-image filter, i.e. how
    much real root-solving the timed call does."""
    lx, ux, ly, uy, ABCs = args
    lim = 1.0 / (6.0 * 3.0 ** 0.5)
    q = ABCs[:, 0] / ly
    return (q.abs() <= lim).double().mean().item()


def max_abs_diff(a, b):
    """NaN-aware: NaN in the same position counts as agreement (it is the
    filtered-out marker), NaN in only one position counts as total disagreement."""
    worst = 0.0
    for x, y in zip(a, b):
        nx, ny = torch.isnan(x), torch.isnan(y)
        if not torch.equal(nx, ny):
            return float("inf")
        m = ~nx
        if m.any():
            worst = max(worst, (x[m] - y[m]).abs().max().item())
    return worst


def bench(fn, args, device, warmup=5, reps=30):
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn(*args)
    sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_variant(label, real_cubic, compile_it, args, device, ref=None):
    pt.USE_REAL_CUBIC = real_cubic
    fn = pt.check_nonlinear_boundary_tensor
    compile_s = None
    if compile_it:
        fn = torch.compile(pt.check_nonlinear_boundary_tensor)
        t0 = time.perf_counter()
        try:
            out = fn(*args)
            if device == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"  {label:26s} COMPILE FAILED: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:120]}")
            return None, ref
        compile_s = time.perf_counter() - t0
    else:
        out = fn(*args)

    med = bench(fn, args, device)
    # Numerical check against the production variant: the real cubic is expected
    # to differ slightly (that is the point); compilation must not change it.
    dev = ""
    if ref is not None:
        d = max_abs_diff(out, ref)
        dev = ("  NaN-PATTERN DIFFERS" if d == float("inf")
               else f"  max|diff vs production|={d:.2e}")
    extra = f"  compile={compile_s:5.1f}s" if compile_s is not None else ""
    print(f"  {label:26s} median={med * 1e3:8.3f} ms{extra}{dev}")
    return med, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", nargs="+", type=int, default=[100, 1024],
                    help="neurons per layer; 100 = 3/4/5layer, 1024 = big")
    ap.add_argument("--regimes", nargs="+", default=["realistic", "in_image"],
                    choices=["realistic", "in_image"],
                    help="candidate density; see make_inputs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = ap.parse_args()

    dev = args_cli.device
    print(f"torch {torch.__version__}  device={dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    print(f"real-cubic support present: "
          f"{hasattr(pt, 'inverse_poly_real_tensor')}")
    if not hasattr(pt, "inverse_poly_real_tensor"):
        print("\nThe fetched precise_transformer.py predates PASADO_REAL_CUBIC; "
              "re-download it from the colab-gpu-wip branch.")
        return

    saved = pt.USE_REAL_CUBIC
    try:
        for n in args_cli.widths:
            for regime in args_cli.regimes:
                args = make_inputs(n, dev, regime)
                print(f"\n=== n = {n} neurons, regime={regime} "
                      f"(live candidates {live_fraction(args) * 100:.1f}%) ===")
                base, ref = run_variant("eager + complex64", False, False, args, dev)
                real, _ = run_variant("eager + real cubic", True, False, args, dev, ref)
                run_variant("compile + complex64", False, True, args, dev, ref)
                comp, _ = run_variant("compile + real cubic", True, True, args, dev, ref)

                if base and real:
                    print(f"  -> real cubic alone: {base / real:.3f}x "
                          f"({'faster' if real < base else 'slower'})")
                if base and comp:
                    print(f"  -> compile + real vs production: {base / comp:.3f}x")
                    print(f"     fusion must beat {real * 1e3:.3f} ms to be worth "
                          f"having on top of the real solver")
    finally:
        pt.USE_REAL_CUBIC = saved


if __name__ == "__main__":
    main()
