"""Why did the speedup ladder record no CUDA events? Each scenario runs in a fresh process.

    python experiments/diag_profiler_cuda.py            # runs every scenario in its own subprocess
    python experiments/diag_profiler_cuda.py <scenario> # one scenario in this process

Scenarios, each followed by one CPU+CUDA profile of a matmul:
    plain        nothing before
    cpu_first    a CPU-only torch profile first (the ladder profiles CPU steps 0-3 first)
    cprofile     a cProfile pass first (the ladder runs cProfile right before the torch profile)
    compiled     a torch.compile'd function run first
    ladder_like  cpu_first + cprofile + compiled, in that order
"""
import cProfile
import subprocess
import sys

SCENARIOS = ["plain", "cpu_first", "cprofile", "compiled", "ladder_like"]


def cuda_profile_count():
    import torch
    from torch.profiler import ProfilerActivity, profile
    x = torch.randn(512, 512, device="cuda")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            x = x @ x / 512
        torch.cuda.synchronize()
    dev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return len(dev)


def run(scenario):
    import torch
    from torch.profiler import ProfilerActivity, profile
    if scenario in ("cpu_first", "ladder_like"):
        with profile(activities=[ProfilerActivity.CPU]):
            torch.randn(256, 256) @ torch.randn(256, 256)
    if scenario in ("cprofile", "ladder_like"):
        pr = cProfile.Profile()
        pr.enable()
        torch.randn(256, 256, device="cuda").sum().item()
        pr.disable()
    if scenario in ("compiled", "ladder_like"):
        f = torch.compile(lambda a: (a.sin() * 2 + 1).sum(), fullgraph=True)
        f(torch.randn(1024, device="cuda")).item()
    print(f"{scenario:12s} cuda_events={cuda_profile_count()}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        for s in SCENARIOS:
            out = subprocess.run([sys.executable, __file__, s], capture_output=True, text=True)
            lines = [l for l in out.stdout.splitlines() if "cuda_events" in l]
            print(lines[-1] if lines else f"{s:12s} FAILED: {out.stderr.strip().splitlines()[-1:]}")
