"""Profile: how much of the `big` precise pass is torch.linalg.lstsq(driver='gelsd')?

Answers Sasa's question (Slack, 22.8.2026): before building the CPU/GPU hybrid
lstsq dispatch, measure whether gelsd is actually a meaningful slice of
wall-clock time on `big`, or whether the big AffineZonotope matmul dominates
and gelsd optimization is secondary.

Method: wall-clock timers wrapped directly around the two calls we're
comparing (torch.linalg.lstsq and the big matmul), not cProfile -- we want a
clean side-by-side percentage of the SAME wall-clock run, not cProfile's
inflated per-call attribution.

Run:  .venv\\Scripts\\python.exe experiments\\profile_gelsd.py
"""
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEC = os.path.join(REPO, "Section_5_4")

# TODO 1: which variant are we profiling? Match profile_variants.py's
#   "vec_batched_grid" env vars (current best CPU state).

# TODO 2: wrap torch.linalg.lstsq so every call is timed and counted, without
#   changing what it returns.

# TODO 3: wrap the big matmul too (SimpleZono.AffineZonotope's
#   `generators @ layer`) -- same pattern, so both numbers come from ONE run.

# TODO 4: run get_lipschitz.py --network big --no-save in-process (its own
#   sys.path.insert / relative paths assume cwd == Section_5_4).

# TODO 5: report total wall time, lstsq time + %, matmul time + %, call counts.
