"""pcn/config.py

`PcnConfig` -- config for the static (non-temporal) PCN baseline, in the
same "config vs. metadata" spirit as `tpch/config.py`'s `TpchConfig`. No
`control_layer_size`/`input_size`/`hidden_sizes`-as-a-separate-thing: a
static hierarchy is just a chain of layer widths from input to output,
`layer_sizes = (input_size, hidden_1, ..., hidden_k, output_size)`.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PcnConfig:
    layer_sizes: Tuple[int, ...]  # (input_size, hidden_1, ..., hidden_k, output_size); len >= 2
    act_fn: str = "tanh"  # Activation function name (registry key), applied at every layer except the output readout
    loss: str = "mse"  # Output-layer loss ('mse' | 'ce')
    weight_decay: float = 0.0
    orthogonal_penalty: float = 0.0
    activity_decay: float = 0.0
    activity_reg_type: str = "l1"
