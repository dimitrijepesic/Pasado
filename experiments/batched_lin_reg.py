"""Re-export of the batched per-neuron planar regression.

The implementation lives in forward_mode_tensorized_src/precise_transformer.py
as lin_reg_tensor_batched, and is enabled with the PASADO_BATCHED_LSTSQ
selector (off by default). This module only imports it so the experiment
scripts and test_batched_lin_reg.py exercise the production code.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "forward_mode_tensorized_src"))

from precise_transformer import lin_reg_tensor_batched  # noqa: F401
