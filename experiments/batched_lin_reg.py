"""Re-exports lin_reg_tensor_batched from precise_transformer for the experiment scripts."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "forward_mode_tensorized_src"))

from precise_transformer import lin_reg_tensor_batched  # noqa: F401
