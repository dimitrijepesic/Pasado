"""Compares lstsq with gelsd (CPU) and gels (GPU) on rank-deficient systems, which
is what zero-width boxes produce in the real analysis. Needs a GPU runtime."""
import torch

torch.set_default_dtype(torch.float64)
SEED = 777
M = 25  # matches STEPS=5 -> 5x5 grid, same as the real per-neuron regression


def build_batch(n, g):
    """[n, M, 2] grids mimicking get_linspace's per-neuron 5x5 grid, with a
    controlled fraction forced rank-deficient the same way a degenerate box
    (lx==ux and/or ly==uy) would be in the real analysis. Returns the batch
    plus the (point-box, degenerate-x) boundary indices for reporting."""
    steps = int(round(M ** 0.5))
    n_point = int(n * 0.1)   # lx==ux and ly==uy -> rank 1
    n_xdeg = int(n * 0.3)    # lx==ux only        -> rank 2
    x = torch.empty(n, M)
    y = torch.empty(n, M)
    for i in range(n):
        lx, spanx = torch.rand(2, generator=g).tolist()
        ly, spany = torch.rand(2, generator=g).tolist()
        ux, uy = lx + spanx + 1e-3, ly + spany + 1e-3
        if i < n_point:
            ux, uy = lx, ly
        elif i < n_point + n_xdeg:
            ux = lx
        xs = torch.linspace(lx, ux, steps=steps)
        ys = torch.linspace(ly, uy, steps=steps)
        grid = torch.cartesian_prod(xs, ys)
        x[i], y[i] = grid[:, 0], grid[:, 1]
    return torch.stack((x, y), dim=-1), n_point, n_point + n_xdeg


def report(name, diff, lo, hi):
    d = diff[lo:hi]
    if d.numel() == 0:
        return
    over = (d > 1e-6).float().mean() * 100
    print(f"{name:22s} n={hi - lo:5d}  max|gelsd_cpu-gels_gpu| per row: "
          f"median={d.median():.3e}  max={d.max():.3e}  (>1e-6: {over:5.1f}%)")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available - nothing to compare, exiting.")
        return
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(SEED)
    n = 2000
    x_batch, n_point, n_deg_end = build_batch(n, g)
    zs = torch.rand(n, M, generator=g).unsqueeze(-1)

    ones = torch.ones(n, M, 1, dtype=x_batch.dtype)
    xplusone = torch.cat((ones, x_batch), dim=2)  # [n, 25, 3], intercept-first

    cpu_sol = torch.linalg.lstsq(xplusone, zs, driver="gelsd").solution.squeeze(-1)
    gpu_sol = torch.linalg.lstsq(
        xplusone.to(dev), zs.to(dev), driver="gels"
    ).solution.squeeze(-1).cpu()

    diff = (cpu_sol - gpu_sol).abs().max(dim=1).values  # [n], worst coeff per row

    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, n={n} rows")
    report("point box (rank 1)", diff, 0, n_point)
    report("degenerate x (rank 2)", diff, n_point, n_deg_end)
    report("full rank", diff, n_deg_end, n)


if __name__ == "__main__":
    main()
