"""Batched per-neuron planar regression (Phase 6 of the 5.4 sprint).

The implementation now lives in
forward_mode_tensorized_src/precise_transformer.py (wired behind the
PASADO_BATCHED_LSTSQ selector, default OFF). This module re-exports it so the
experiment scripts keep working and so the tests exercise the production code.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "forward_mode_tensorized_src"))

from precise_transformer import lin_reg_tensor_batched  # noqa: F401
