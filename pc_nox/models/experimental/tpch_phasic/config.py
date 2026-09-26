"""kpch/config.py

`KPCHConfig` -- config for the phase-only, non-oscillatory variant: every
layer's state is a bare angle theta (real-valued), constrained to the
unit circle by construction (z = e^{i*theta}), with NO amplitude degree
of freedom and NO intrinsic rotation/oscillator dynamics -- see
`model.py`'s module docstring for the full design and how this differs
from `slpch` (which has both).

Structurally mirrors `tpch.config.TpchConfig`/`slpch.config.SLPCHConfig`
(`control_layer_size`, `hidden_sizes`, `obs_size`, `input_size`
unchanged). What replaces `slpch`'s oscillator params (`gamma_init`,
`omega_init_scale`, `amp_weight`/`sync_weight`) is a single per-node
LEARNABLE coupling strength `kappa` (init range `kappa_init`) -- there's
no amplitude term to weight against here, so there's nothing for a
second coefficient to trade off against; `kappa` is the only knob on how
strongly a node's phase is pulled toward its prediction.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class KPCHConfig:
    control_layer_size: int
    hidden_sizes: Tuple[int, ...]
    obs_size: int
    input_size: int = 0

    kappa_init: float = 1.0  # initial per-node coupling strength (> 0); learnable, see layers.py
    loss: str = "mse"  # Observation-layer loss ('mse' | 'ce'); same meaning as TpchConfig.loss

    def __post_init__(self):
        if self.kappa_init <= 0:
            raise ValueError(f"kappa_init must be > 0 (it's a coupling strength), got {self.kappa_init}")
        if self.loss not in ("mse", "ce"):
            raise ValueError(f"loss must be 'mse' or 'ce', got {self.loss!r}")
