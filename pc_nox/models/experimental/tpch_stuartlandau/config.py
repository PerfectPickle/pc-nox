"""slpch/config.py

`SLPCHConfig` -- everything that changes what an `SLPCHModel` computes (see
`ModelBase`'s "config vs. metadata" note in `model_base.py`). Structurally
mirrors `tpch.config.TpchConfig` (`control_layer_size`, `hidden_sizes`,
`obs_size`, `input_size` are unchanged), but swaps the real-valued-net
knobs it doesn't need (`act_fn`, the three regularisers) for the ones a
Stuart-Landau oscillator network needs instead:

  * every layer's states are complex now, and the pointwise nonlinearity
    `act_fn` used to provide is replaced by the oscillators' own intrinsic
    dynamics (see `layers.py`) -- `gamma_init`/`omega_init_scale` are that
    dynamics' *initialisation* ranges, not per-node values (those are
    learnable parameters living on the layers themselves).
  * `amp_weight`/`sync_weight` split the old single per-layer prediction
    error `0.5 * ||state - prediction||^2` into an amplitude-mismatch term
    and an explicit Kuramoto-style phase-synchrony term -- see
    `model.py`'s `slpch_energy_fn` docstring for the exact split and why
    it's an *exact* decomposition of the complex residual, not an ad hoc
    addition.

Regularisers (`weight_decay`, `orthogonal_penalty`, `activity_decay`) are
deliberately left out of this variant, per scope.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SLPCHConfig:
    control_layer_size: int  # Control layer width
    hidden_sizes: Tuple[int, ...]  # Ordered top (just below control) to bottom (just above obs)
    obs_size: int  # Observation / sensory layer width
    input_size: int = 0  # Control input (optional)

    # --- oscillator initialisation (per-node gamma/omega are learnable
    # parameters on the layers themselves -- see layers.py -- these just
    # set their initial values/ranges) ---
    gamma_init: float = 1.0  # initial supercritical-Hopf bifurcation param (> 0); limit-cycle radius = sqrt(gamma)
    omega_init_scale: float = 2.0  # intrinsic frequencies init ~ Uniform(-omega_init_scale, omega_init_scale)

    # --- energy: splits the old ||state - prediction||^2 term into an
    # amplitude part and an explicit synchrony (Kuramoto) part; see
    # model.py's slpch_energy_fn docstring for the exact identity ---
    amp_weight: float = 1.0  # coefficient on the amplitude/magnitude prediction-error term
    sync_weight: float = 1.0  # coefficient on the phase-synchrony (Kuramoto) term; 0 disables it

    loss: str = "mse"  # Observation-layer loss ('mse' | 'ce'); same meaning as TpchConfig.loss

    # --- adaptive omega (opt-in, off by default -- see model.py's
    # make_adaptive_vector_field/settle_scan_adaptive/settle_diffrax_adaptive).
    # When enabled, each dynamical layer's rotation rate is no longer the
    # fixed, layer-owned `omega` -- it becomes part of the settled state,
    # adapted every relaxation step toward whatever frequency reduces the
    # phase-torque the prediction-error force is currently applying, and
    # decaying back toward the layer's own `omega` (now playing the role
    # of a REST frequency) when unforced. This is a per-timestep dynamic
    # quantity threaded explicitly by the caller (like states_prev/
    # states_curr), NOT a model parameter -- param_grad still can't reach
    # it, for the same reason it can't reach the static omega.
    adapt_omega: bool = False
    omega_adapt_rate: float = 1.0  # kappa: how fast omega chases the phase-torque signal
    omega_decay_rate: float = 0.3  # lambda: how fast omega relaxes back to its rest value absent forcing

    def __post_init__(self):
        if self.gamma_init <= 0:
            raise ValueError(f"gamma_init must be > 0 (it's a limit-cycle radius^2), got {self.gamma_init}")
        if self.loss not in ("mse", "ce"):
            raise ValueError(f"loss must be 'mse' or 'ce', got {self.loss!r}")
        if self.omega_decay_rate < 0:
            raise ValueError(f"omega_decay_rate must be >= 0 (it's a decay rate), got {self.omega_decay_rate}")
