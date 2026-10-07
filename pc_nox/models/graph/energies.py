"""graph/energies.py

Energy strategies: how a node's state is scored against its
predictions. Chosen per node by registry name on the `NodeSpec`
(`energy="pc"`), independent of the node type, so any node can be paired
with any energy it is compatible with (compatibility is checked once, at
construction, by `validation.validate_pairings`).

A strategy is a pure function of the state, the node's `NodePrediction`,
and a `frozen` context built once per settle (empty for standard PC).
"""

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .nodes.base import NodePrediction


@dataclass(frozen=True)
class EnergyStrategy:
    """A registry entry: the energy function plus what it needs from a node.

    Attributes:
        fn: `energy(state, prediction: NodePrediction, frozen) -> scalar`.
        required_paths: prediction paths the node MUST return (e.g. "td").
        allowed_paths: if not None, the ONLY paths this strategy knows how
            to score; a node returning any other path is rejected rather
            than silently mis-scored. None means "any paths".
        needs_pre_activations: whether the strategy reads
            `NodePrediction.pre_activations` (e.g. Meta-PCN's D(h)).
    """
    fn: Callable
    required_paths: FrozenSet[str] = frozenset({"td"})
    allowed_paths: Optional[FrozenSet[str]] = None
    needs_pre_activations: bool = False


def pc_energy(state: Array, prediction: NodePrediction, frozen=None) -> Array:
    """Standard PC: 0.5 * ||state - prediction||^2, summed over every
    prediction path the node returns (one term for a plain node, two for a
    bidirectional one, and zero for a pure-data node such as `InputNode`).
    """
    return sum((0.5 * jnp.sum((state - p) ** 2) for p in prediction.predictions.values()), jnp.asarray(0.0))


def ce_energy(state: Array, prediction: NodePrediction, frozen=None) -> Array:
    """Cross-entropy of `state` (a target distribution, e.g. one-hot) under
    the node's "td" prediction treated as LOGITS. Intended for clamped
    observation nodes; identical to the `loss="ce"` observation term.
    """
    return -jnp.sum(state * jax.nn.log_softmax(prediction.predictions["td"]))


ENERGY_REGISTRY: Dict[str, EnergyStrategy] = {
    # any paths, including none (a pure-data node). That an INFERRED node returns a "td" prediction is
    # enforced separately, in validation, because initialisation needs it whatever the energy.
    "pc": EnergyStrategy(fn=pc_energy, required_paths=frozenset()),
    "ce": EnergyStrategy(fn=ce_energy, allowed_paths=frozenset({"td"})),
    # "meta": EnergyStrategy(fn=meta_energy, allowed_paths=frozenset({"td"}),
    #                        needs_pre_activations=True),   # added with Meta-PCN
}
