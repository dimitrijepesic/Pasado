import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from benchmark_lipschitz import VARIANT_ENV   # noqa: E402

NAMES = ["PASADO_VECTORIZED_PRECISE", "PASADO_VEC_BOUNDARY", "PASADO_BATCHED_LSTSQ",
         "PASADO_BATCHED_GRID", "PASADO_REAL_CUBIC"]

# What precise_transformer should see for each variant: vec, bnd, lstsq, grid, cubic.
EXPECTED = {
    "original":           (0, 0, 0, 0, 0),
    "vec_corners":        (1, 0, 0, 0, 0),
    "vec_both":           (1, 1, 0, 0, 0),
    "vec_both_batched":   (1, 1, 1, 0, 0),
    "vec_batched_grid":   (1, 1, 1, 1, 0),
    "vec_grid_realcubic": (1, 1, 1, 1, 1),
}

PROBE = ("import sys; sys.path.insert(0, 'forward_mode_tensorized_src'); "
         "import precise_transformer as p; "
         "print(int(p.USE_VECTORIZED_PRECISE), int(p.USE_VECTORIZED_BOUNDARY), "
         "int(p.USE_BATCHED_LSTSQ), int(p.USE_BATCHED_GRID), int(p.USE_REAL_CUBIC))")


def flags_seen(variant, exported):
    """Flags precise_transformer sees when `exported` is in the shell and the
    variant is applied the way benchmark_lipschitz.run_once applies it."""
    env = dict(os.environ)
    env.update(exported)
    env.update(VARIANT_ENV[variant])
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout.split()
    return tuple(int(x) for x in out[-5:])


def main():
    ok = True
    # Export every flag to the opposite of the default, so any flag a variant
    # forgets to set would leak through.
    leaks = {"PASADO_VECTORIZED_PRECISE": "0", "PASADO_VEC_BOUNDARY": "0",
             "PASADO_BATCHED_LSTSQ": "1", "PASADO_BATCHED_GRID": "1", "PASADO_REAL_CUBIC": "1"}
    for variant, want in EXPECTED.items():
        if set(VARIANT_ENV[variant]) != set(NAMES):
            print(f"FAIL {variant}: does not set all five flags: {sorted(VARIANT_ENV[variant])}")
            ok = False
            continue
        for label, exported in (("clean shell", {}), ("flags exported", leaks)):
            got = flags_seen(variant, exported)
            status = "ok  " if got == want else "FAIL"
            ok &= got == want
            print(f"{status} {variant:20s} {label:15s} sees {got}  expected {want}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
