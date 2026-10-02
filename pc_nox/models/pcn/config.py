"""pcn/config.py

`PcnConfig` -- config for the static (non-temporal) PCN baseline. Two
ways to describe the architecture, exactly one of which must be given
(validated in `PcnModel.__init__`):

  - `layer_sizes` (original, dense-only path): a flat chain of widths,
    `(input_size, hidden_1, ..., hidden_k, output_size)`.

  - `layer_specs` (new, heterogeneous path): a tuple of per-layer specs
    for a mixed conv/flatten/dense stack, each one of:
        ("conv", out_channels, kernel_size, stride, padding)
        ("flatten",)
        ("dense", out_size)
    `input_channels`/`input_hw` must also be set, since a conv trunk
    needs `x` shaped (input_channels, *input_hw) -- NOT flattened --
    unlike the `layer_sizes` path where `x` is a flat vector. See
    `model.py`'s `_build_layers_from_specs` for exactly how a spec list
    is turned into layer objects (it uses `jax.eval_shape` to work out
    each "dense" layer's input size after a flatten, rather than
    reimplementing conv output-size arithmetic here).

`act_fn` is either one name (broadcast to every layer) or a tuple with
one name per layer -- including "flatten" entries, where the name is
accepted but ignored (see `layers.PcnFlattenLayer`). This lets a conv
trunk use e.g. "relu" while the dense tail uses "tanh", and the readout
use "identity" for `loss="ce"` (see `model.py`'s bounded-activation
warning).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class PcnConfig:
    layer_sizes: Optional[Tuple[int, ...]] = None
    layer_specs: Optional[Tuple[Tuple, ...]] = None
    input_channels: Optional[int] = None  # required with layer_specs
    input_hw: Optional[Tuple[int, int]] = None  # required with layer_specs
    act_fn: Union[str, Tuple[str, ...]] = "tanh"  # one name, or one per layer (see module docstring)
    loss: str = "mse"  # Output-layer loss ('mse' | 'ce')
    weight_decay: float = 0.0
    orthogonal_penalty: float = 0.0
    activity_decay: float = 0.0
    activity_reg_type: str = "l1"
