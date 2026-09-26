"""tpch/config.py

`TpchConfig` -- everything that changes what a `TpchModel` computes (see
`ModelBase`'s "config vs. metadata" note). Kept in its own module, separate
from `layers.py`/`model.py`, so other variants (e.g. bidirectional tPC-H)
can import and extend it -- or define their own sibling config -- without
pulling in the layer/model classes too.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


# frozen to ensure hashable for JAX compilation
@dataclass(frozen=True)
class TpchConfig:
    control_layer_size: int  # Control layer width
    hidden_sizes: Tuple[int]  # Ordered from top (just below control) to bottom (just above obs). Tuple is hashable, unlike List
    obs_size: int  # Observation / sensory layer width
    input_size: int = 0  # Control input (optional)
    act_fn: str = "tanh"  # Activation function name, used as lookup key in ACT_FN_REGISTRY
    loss: str = "mse"  # Observation-layer loss ('mse' | 'ce'); only changes the y_hat term, not the state-prediction terms
    weight_decay: float = 0.0  # Coefficient for L2 penalty on weights (0 = disabled)
    weight_decay_scope: str = "all"  # Which weights weight_decay applies to: 'all', 'rec' (recurrent, i.e. W_rec only), or 'ff' (everything except W_rec)
    orthogonal_penalty: float = 0.0  # Coefficient for orthogonality penalty ||I - W^T W||^2 on weights (0 = disabled)
    orthogonal_scope: str = "rec"  # Which weights orthogonal_penalty applies to: 'rec' (default, only weights guaranteed square), 'all', or 'ff'
    activity_decay: float = 0.0  # Coefficient for L1/L2 penalty on current states/activities (0 = disabled)
    activity_reg_type: str = "l1"  # Norm used by activity_decay ('l1' | 'l2')
    control_alpha: Optional[float] = None  # tPC-E leak rate for the control layer (see TpchControlLayer); None = disabled
    hidden_alphas: Optional[Tuple[Optional[float], ...]] = None  # tPC-E leak rate per hidden layer (same order as hidden_sizes); None = disabled for all, or per-entry None disables that one layer only
    observation_groups: Optional[Tuple[Tuple[str, int, str], ...]] = None  # (name, size, loss) per node group for a heterogeneous observation layer (see TpchHeterogeneousObservationLayer); None = plain homogeneous TpchObservationLayer (default, unchanged)
    trace_mode: str = "readout"  # Which convention places f' in the eligibility-trace weight update, for leaky layers. 'readout' (default): f' applied fresh at read-out time, uniformly over the trace's whole history (vector-shaped trace, cheaper). 'accumulate': each step's own f' baked into the trace's own accumulation (matrix-shaped trace, matching the weight's shape -- more memory, arguably more faithful to a literal extension of the source paper's derivation). Identical results at alpha=1 regardless of mode; NOT equivalent for alpha<1 spanning several steps. See eligibility.py and tPC-HE.md section 8.


def scopes_overlap(scope_a: str, scope_b: str) -> bool:
    """Whether two weight scopes ('all' | 'rec' | 'ff') can refer to any of
    the same weight matrices. Purely symbolic (doesn't need actual weight
    arrays): 'all' is a superset of both 'rec' and 'ff', which are disjoint
    from each other. Used to decide whether weight_decay and
    orthogonal_penalty could compete for the same weights (see the warning
    in TpchModel.__init__).
    """
    if scope_a == "all" or scope_b == "all":
        return True
    return scope_a == scope_b
