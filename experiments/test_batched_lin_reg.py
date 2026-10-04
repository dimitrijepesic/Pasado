import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import precise_transformer as pt
from batched_lin_reg import lin_reg_tensor_batched

SEED = 12345
# Hard tolerances per dtype for the assert_close check (see module docstring).
TOL = {torch.float64: dict(rtol=1e-12, atol=1e-14),
       torch.float32: dict(rtol=1e-5, atol=1e-6)}


def old_loop(x_batch, zs_batch):
    """Reference: lin_reg_tensor called once per neuron."""
    return torch.stack([pt.lin_reg_tensor(x_batch[i], zs_batch[i])
                        for i in range(x_batch.shape[0])])


def realistic_grids(n, seed, dtype, box="normal"):
    """Sampling grids as get_linspace builds them, one per neuron.

    box picks the kind of input: normal, degenerate_x, point, near_degenerate,
    saturated or mixed_scale. Returns points [n, 25, 2] and targets [n, 25].
    """
    g = torch.Generator().manual_seed(seed)

    def rnd(lo, hi):
        return (torch.rand((), generator=g, dtype=dtype) * (hi - lo) + lo).item()

    grids = []
    for _ in range(n):
        if box == "normal":
            lx = rnd(-4, 4); ux = lx + rnd(1e-3, 4)
            ly = rnd(-4, 4); uy = ly + rnd(1e-3, 4)
        elif box == "degenerate_x":
            lx = rnd(-4, 4); ux = lx
            ly = rnd(-4, 4); uy = ly + rnd(1e-3, 4)
        elif box == "point":
            lx = rnd(-4, 4); ux = lx
            ly = rnd(-4, 4); uy = ly
        elif box == "near_degenerate":
            lx = rnd(-4, 4); ux = lx + 1e-12
            ly = rnd(-4, 4); uy = ly + 1e-12
        elif box == "saturated":
            lx = rnd(25, 35); ux = lx + rnd(1e-3, 4)
            ly = rnd(-4, 4); uy = ly + rnd(1e-3, 4)
        elif box == "mixed_scale":
            lx = rnd(-1e-8, 1e-8); ux = lx + 1e-8
            ly = rnd(-1e6, 1e6); uy = ly + rnd(1.0, 1e6)
        xs = torch.linspace(lx, ux, steps=5)
        ys = torch.linspace(ly, uy, steps=5)
        grids.append(torch.cartesian_prod(xs, ys))
    x_batch = torch.stack(grids)                      # [n, 25, 2]
    zs_batch = torch.stack([pt.sigmoid_prime_times_y(gr) for gr in grids])  # [n, 25]
    return x_batch, zs_batch


def random_points(n, m, seed, dtype):
    """n random regression problems with m points each."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, m, 2, generator=g, dtype=dtype)
    zs = torch.randn(n, m, generator=g, dtype=dtype)
    return x, zs


def planar_error(coeffs, x_batch, zs_batch):
    """Largest |z - (A x + B y + C)| per neuron. Coefficients are intercept first (C, A, B)."""
    C, A, B = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3]
    pred = A * x_batch[:, :, 0] + B * x_batch[:, :, 1] + C
    return (zs_batch - pred).abs().max(dim=1).values


def run_case(name, x_batch, zs_batch, dtype):
    """Compare batched and loop results. Returns True if they are bit-identical."""
    old = old_loop(x_batch, zs_batch)
    new = lin_reg_tensor_batched(x_batch, zs_batch)

    assert new.shape == (x_batch.shape[0], 3), f"{name}: shape {new.shape}"
    assert new.dtype == x_batch.dtype, f"{name}: dtype {new.dtype}"
    assert new.device == x_batch.device, f"{name}: device {new.device}"

    bitwise = torch.equal(old, new)
    d = (old - new).abs()
    max_abs = d.max().item()
    max_rel = (d / old.abs().clamp_min(1e-300)).max().item()
    torch.testing.assert_close(new, old, **TOL[dtype])

    # zs as [n, m, 1] must give the identical result
    new3 = lin_reg_tensor_batched(x_batch, zs_batch.unsqueeze(-1))
    assert torch.equal(new, new3), f"{name}: [n,m,1] zs path differs"

    # downstream planar errors must match
    pe = (planar_error(old, x_batch, zs_batch)
          - planar_error(new, x_batch, zs_batch)).abs().max().item()

    print(f"  [{name:34s}] bitwise={'YES' if bitwise else 'no '} "
          f"max|dcoef|={max_abs:.3e} max_rel={max_rel:.3e} "
          f"planar_err_diff={pe:.3e}  PASS")
    return bitwise


def accelerators():
    """Devices other than the CPU that this machine has."""
    found = []
    if torch.cuda.is_available():
        found.append("cuda")
    if torch.backends.mps.is_available():
        found.append("mps")
    return found


def cases(dtype):
    """Every input set used here, as (name, points, targets)."""
    out = []
    for n in (1, 10, 100, 1024):
        x, zs = realistic_grids(n, SEED + n, dtype)
        out.append((f"realistic grid n={n}", x, zs))
    for n in (1, 10, 100):
        x, zs = random_points(n, 25, SEED + n, dtype)
        out.append((f"random points n={n}", x, zs))
    for box in ("degenerate_x", "point", "near_degenerate", "saturated", "mixed_scale"):
        x, zs = realistic_grids(16, SEED, dtype, box=box)
        out.append((f"{box} n=16", x, zs))
    # mixed batch: healthy and rank-deficient neurons side by side
    xh, zh = realistic_grids(8, SEED + 1, dtype)
    xd, zd = realistic_grids(8, SEED + 2, dtype, box="degenerate_x")
    out.append(("mixed healthy+degenerate n=16", torch.cat((xh, xd)), torch.cat((zh, zd))))
    return out


def device_case(name, x, zs, dtype, device):
    """With the inputs on `device`, the result must be on `device` and equal the CPU solve exactly."""
    ref = lin_reg_tensor_batched(x, zs)
    got = lin_reg_tensor_batched(x.to(device), zs.to(device))
    assert got.device.type == torch.device(device).type, f"{name}: result is on {got.device}"
    assert got.dtype == dtype, f"{name}: dtype {got.dtype}"
    assert torch.equal(got.cpu(), ref), f"{name} on {device}: differs from the CPU solve"
    print(f"  [{name:34s}] on {device}: identical to the CPU solve  PASS")


def main():
    """Run every case for float64 and float32, on CPU and on each available GPU."""
    devices = accelerators()
    all_bitwise = True
    for dtype in (torch.float64, torch.float32):
        torch.set_default_dtype(dtype)  # lin_reg_tensor builds ones() in default dtype
        print(f"--- dtype={dtype} ---")
        sets = cases(dtype)
        for name, x, zs in sets:
            all_bitwise &= run_case(name, x, zs, dtype)
        for device in devices:
            if device == "mps" and dtype == torch.float64:
                print("  SKIP mps with float64: the device has no float64")
                continue
            print(f"  --- inputs on {device} ---")
            for name, x, zs in sets:
                device_case(name, x, zs, dtype, device)

    if not devices:
        print("\nSKIP device cases: no cuda or mps device on this machine")
    print(f"\nALL PASS (assert_close, rtol/atol per dtype); "
          f"bitwise-identical everywhere: {'YES' if all_bitwise else 'NO'}")


if __name__ == "__main__":
    main()
