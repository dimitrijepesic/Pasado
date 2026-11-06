"""Helpers shared by the checkpoint Colab notebooks.

Every measurement is a "run": one subprocess with a fixed command line whose outputs are
written to Google Drive. When it succeeds, a JSON record (command, seconds, return code,
environment, sha256 of the outputs) is written next to them, and a later call with the same
key is skipped. A notebook can therefore be re-run from the top after a disconnect and
continues where it stopped. Logs go to Drive as well.

Local test mode (PASADO_NB_LOCAL_TEST=1): no Drive, no clone, no pip, CPU instead of CUDA
and tiny grids, so the notebook logic can be checked on a laptop.
"""
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time

LOCAL_TEST = os.environ.get("PASADO_NB_LOCAL_TEST", "0") == "1"


def sh(cmd, check=True):
    """Runs a shell command and returns its stdout (stripped)."""
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{cmd}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout.strip()


def gpu_tag():
    """Short name of the GPU, used in folder names so different GPUs are never pooled."""
    if LOCAL_TEST:
        return "local-cpu"
    name = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)
    return name.split("\n")[0].replace(" ", "_").replace(",", "").replace("/", "-") or "unknown-gpu"


def environment(repo):
    """Everything that identifies the measurement environment."""
    env = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=platform.node(),
               python=platform.python_version(), machine=platform.machine(),
               commit=sh(f"git -C {shlex.quote(repo)} rev-parse HEAD", check=False),
               dirty=bool(sh(f"git -C {shlex.quote(repo)} status --porcelain --untracked-files=no",
                             check=False)))
    env["cpu"] = (sh("lscpu | grep 'Model name' | sed 's/.*: *//'", check=False)
                  or sh("sysctl -n machdep.cpu.brand_string", check=False) or platform.processor())
    env["cpu_count"] = os.cpu_count()
    if not LOCAL_TEST:
        env["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
                               "--format=csv,noheader", check=False)
    code = ("import json, torch; d = dict(torch=torch.__version__, cuda=torch.version.cuda, "
            "cuda_available=torch.cuda.is_available(), threads=torch.get_num_threads());\n"
            "if torch.cuda.is_available(): d['gpu'] = torch.cuda.get_device_name(0); "
            "d['gpu_mem_gib'] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)\n"
            "try:\n import jax; d['jax'] = jax.__version__\nexcept Exception: pass\n"
            "try:\n import triton; d['triton'] = triton.__version__\nexcept Exception: pass\n"
            "print(json.dumps(d))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        env.update(json.loads(out.stdout.strip().splitlines()[-1]))
    except Exception:  # noqa: BLE001 - record the failure instead of stopping
        env["torch_probe_error"] = out.stderr[-500:]
    return env


class Runner:
    def __init__(self, repo, results_dir, extra_env=None):
        self.repo = repo
        self.results = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.extra_env = dict(extra_env or {})
        self.env_record = environment(repo)
        with open(os.path.join(results_dir, "environment.json"), "w") as handle:
            json.dump(self.env_record, handle, indent=1)

    def path(self, *parts):
        p = os.path.join(self.results, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def done(self, key):
        p = os.path.join(self.results, "runs", key + ".json")
        if not os.path.exists(p):
            return False
        with open(p) as handle:
            return json.load(handle).get("returncode") == 0

    def run(self, key, args, outputs=(), env=None, estimate_min=None, cwd=None):
        """Runs `python args...` (args is a list) unless `key` already succeeded."""
        record_path = self.path("runs", key + ".json")
        if self.done(key):
            print(f"[skip] {key} (done)")
            return True
        cmd = [sys.executable, *map(str, args)]
        full_env = dict(os.environ, **self.extra_env, **(env or {}),
                        PASADO_COMMIT=self.env_record.get("commit", ""))
        log_path = self.path("logs", key + ".log")
        note = f" (planning estimate ~{estimate_min} min)" if estimate_min else ""
        print(f"[run ] {key}{note}\n       {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        start = time.time()
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, cwd=cwd or self.repo, env=full_env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                log.write(line)
                if "it/s]" not in line and "s/it]" not in line:  # keep tqdm noise out of the cell
                    print("       " + line, end="", flush=True)
            proc.wait()
        seconds = time.time() - start
        hashes = {}
        for out in outputs:
            if os.path.isfile(out):
                with open(out, "rb") as handle:
                    hashes[out] = hashlib.sha256(handle.read()).hexdigest()
        record = dict(key=key, cmd=cmd, env_overrides={**self.extra_env, **(env or {})},
                      returncode=proc.returncode, seconds=seconds, outputs=hashes,
                      environment=self.env_record)
        if proc.returncode == 0:
            with open(record_path, "w") as handle:
                json.dump(record, handle, indent=1)
            print(f"[ ok ] {key} in {seconds / 60:.1f} min")
        else:
            with open(self.path("runs", key + ".FAILED.json"), "w") as handle:
                json.dump(record, handle, indent=1)
            print(f"[FAIL] {key} (exit {proc.returncode}) after {seconds / 60:.1f} min; log: {log_path}")
        return proc.returncode == 0
