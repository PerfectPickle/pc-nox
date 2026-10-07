"""graph/nodes/pcn.py

`PcnNode`: the static (non-recurrent) dense node, the graph counterpart of
`PcnLayer`. Every quantity is "now" -- there is no previous time step, so
the node needs only its parents' CURRENT state (and, if bidirectional, its
children's).

    top-down  (td):  z_hat = f(W_td @ z_parent + b_td)
    bottom-up (bu):  z_hat = f(W_bu @ z_child  + b_bu)        [optional]

Both projections keep their biases, matching `PcnLayer` / the Meta-PCN
algorithm box (`W_l z + b_l`).

ROOT MODE. With no parents, the node has nothing to predict it from, yet a
generative graph needs an inferred top. Such a node predicts a learned
PRIOR: `z_hat = f(prior)`, with `prior` a trainable vector (initialised to
zeros). Its energy term is then a Gaussian prior on that state, and it is
settled from error flowing up from its children like any other node.

Input layout is "flat": several parents/children are concatenated, and a
conv parent/child is flattened (see `builders.py`).
"""

from typing import Callable, List, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from .base import NodeBase, NodeInputs, NodePrediction, WeightEntry


class PcnNode(NodeBase):
    """Static dense node with an optional bottom-up path.

    Fields:
        td_op: `Linear(parent_size -> own_size)` with bias; None in root mode.
        prior: trainable prior pre-activation `(own_size,)`; None unless root.
        bu_op: `Linear(child_size -> own_size)` with bias; None unless
            `bidirectional`.
        own_size, bidirectional, td_act_fn, bu_act_fn: static metadata.
    """

    td_op: Optional[eqx.nn.Linear]
    prior: Optional[Array]
    bu_op: Optional[eqx.nn.Linear]

    own_size: int = eqx.field(static=True)
    bidirectional: bool = eqx.field(static=True)
    td_act_fn: Callable = eqx.field(static=True)
    bu_act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        own_size: int,
        parent_size: Optional[int] = None,
        child_size: Optional[int] = None,
        td_act_fn: Callable = jnp.tanh,
        bu_act_fn: Callable = jnp.tanh,
        bidirectional: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        if own_size <= 0:
            raise ValueError(f"own_size must be positive, got {own_size}")
        if bidirectional and not child_size:
            raise ValueError("bidirectional=True requires child_size")

        k_td, k_bu = jr.split(key)
        self.own_size = own_size
        self.bidirectional = bidirectional
        self.td_act_fn = td_act_fn
        self.bu_act_fn = bu_act_fn

        if parent_size:
            self.td_op = eqx.nn.Linear(parent_size, own_size, use_bias=True, key=k_td)
            self.prior = None
        else:                                           # root mode: learned prior
            self.td_op = None
            self.prior = jnp.zeros(own_size)

        self.bu_op = eqx.nn.Linear(child_size, own_size, use_bias=True, key=k_bu) if bidirectional else None

    @classmethod
    def from_spec(
        cls,
        spec,                                           # NodeSpec
        parent_shape: Optional[Tuple[int, ...]],
        child_shape: Optional[Tuple[int, ...]],
        act_fn: Callable,
        *,
        key: PRNGKeyArray,
    ) -> "PcnNode":
        """Build from a `NodeSpec`; shapes are the builder's COMBINED
        (flattened) parent/child shapes. `act_fn` is used for both
        directions."""
        if len(spec.shape) != 1:
            raise ValueError(f"node {spec.name!r}: PcnNode shape must be 1-D, got {spec.shape}")
        if spec.options:
            raise ValueError(f"node {spec.name!r}: PcnNode takes no options, got {dict(spec.options)}")
        return cls(
            own_size=spec.shape[0],
            parent_size=parent_shape[0] if parent_shape else None,
            child_size=child_shape[0] if (spec.bidirectional and child_shape) else None,
            td_act_fn=act_fn, bu_act_fn=act_fn, bidirectional=spec.bidirectional, key=key,
        )

    # ------------------------------------------------------------------ #
    # NodeBase interface
    # ------------------------------------------------------------------ #

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.own_size,)

    @property
    def prediction_keys(self) -> Tuple[str, ...]:
        return ("td", "bu") if self.bidirectional else ("td",)

    @property
    def needs_parents_curr(self) -> bool:
        return self.td_op is not None                   # root mode needs no parents

    @property
    def needs_children(self) -> bool:
        return self.bidirectional

    @property
    def supports_bottom_up(self) -> bool:
        return True

    @property
    def provides_pre_activations(self) -> bool:
        return True

    def predict(self, inp: NodeInputs) -> NodePrediction:
        """Top-down prediction from `inp.parents_curr` (or the learned
        prior, in root mode), plus a bottom-up prediction from
        `inp.children_curr` if `bidirectional`."""
        self.check_inputs(inp)

        pre_td = self.td_op(inp.parents_curr) if self.td_op is not None else self.prior
        predictions = {"td": self.td_act_fn(pre_td)}
        pre_activations = {"td": pre_td}

        if self.bidirectional:
            pre_bu = self.bu_op(inp.children_curr)
            predictions["bu"] = self.bu_act_fn(pre_bu)
            pre_activations["bu"] = pre_bu

        return NodePrediction(predictions, pre_activations)

    def weight_entries(self) -> List[WeightEntry]:
        """Weight matrices tagged "ff" (nothing is recurrent). Biases and
        the root prior are not included."""
        entries = []
        if self.td_op is not None:
            entries.append(WeightEntry("td_op.weight", "ff", self.td_op.weight))
        if self.bu_op is not None:
            entries.append(WeightEntry("bu_op.weight", "ff", self.bu_op.weight))
        return entries
