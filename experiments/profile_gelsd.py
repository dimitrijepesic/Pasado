"""Measures how much of the big network's precise pass goes to torch.linalg.lstsq
(gelsd) compared to the big matmul. Uses wall-clock timers around the two calls,
not cProfile.

Run: python experiments/profile_gelsd.py"""
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
#   `generators @ layer`) - same pattern, so both numbers come from ONE run.

# TODO 4: run get_lipschitz.py --network big --no-save in-process (its own
#   sys.path.insert / relative paths assume cwd == Section_5_4).

# TODO 5: report total wall time, lstsq time + %, matmul time + %, call counts.
