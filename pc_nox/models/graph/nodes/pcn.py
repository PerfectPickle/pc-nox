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
from .activity_norm import apply_affine, make_activity_norms


class PcnNode(NodeBase):
    """Static dense node with an optional bottom-up path.

    Fields:
        op_td: `Linear(parent_size -> own_size)` with bias; None in root mode.
        prior: trainable prior pre-activation `(own_size,)`; None unless root.
        op_bu: `Linear(child_size -> own_size)` with bias; None unless
            `bidirectional`.
        own_size, bidirectional, td_act_fn, bu_act_fn: static metadata.
    """
    activity_norms: Dict[str, eqx.Module]          # activity normalisation targets (e.g. 'parents_curr') and Modules

    op_td: Optional[eqx.nn.Linear]
    prior: Optional[Array]
    op_bu: Optional[eqx.nn.Linear]


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
        has_prior: bool = True,
        activity_norm: Optional[str] = None,      # None | "rms": normalise incoming activities
        activity_norm_inputs: Optional[Tuple[str, ...]] = None,   #  e.g. 'parents_curr'
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
            self.op_td = eqx.nn.Linear(parent_size, own_size, use_bias=True, key=k_td)
            self.prior = None
        elif has_prior:                                 # root mode: learned prior
            self.op_td = None
            self.prior = jnp.zeros(own_size)
        else:                                           # parentless data root: bottom-up path only
            if not bidirectional:
                raise ValueError("a node with no parents and no prior predicts nothing: it needs bidirectional=True")
            self.op_td = None
            self.prior = None

        self.op_bu = eqx.nn.Linear(child_size, own_size, use_bias=True, key=k_bu) if bidirectional else None

        # the norm exists exactly when its projection exists
        self.activity_norms = make_activity_norms(activity_norm, activity_norm_inputs, {
            "parents_curr":  (parent_size,) if self.op_td is not None else None,
            "children_curr": (child_size,)  if self.op_bu is not None else None,
        })


    @classmethod
    def from_spec(
        cls,
        spec,                                           # NodeSpec
        parent_shape: Optional[Tuple[int, ...]],
        child_shape: Optional[Tuple[int, ...]],
        td_act_fn: Callable,
        bu_act_fn: Callable,
        *,
        key: PRNGKeyArray,
    ) -> "PcnNode":
        """Build from a `NodeSpec`; shapes are the builder's COMBINED
        (flattened) parent/child shapes. `act_fn` is used for both
        directions."""
        if len(spec.shape) != 1:
            raise ValueError(f"node {spec.name!r}: PcnNode shape must be 1-D, got {spec.shape}")
        opts = spec.opts
        unknown = set(opts) - {"prior", "activity_norm"}
        if unknown:
            raise ValueError(f"node {spec.name!r}: PcnNode takes no options other than 'prior' and/or 'activity norm', got {sorted(unknown)}")
        bi = bool(spec.bidirectional)
        return cls(
            own_size=spec.shape[0],
            parent_size=parent_shape[0] if parent_shape else None,
            child_size=child_shape[0] if (bi and child_shape) else None,
            td_act_fn=td_act_fn, bu_act_fn=bu_act_fn, bidirectional=bi, has_prior=bool(opts.get("prior", True)), activity_norm=opts.get("activity_norm"), key=key,
        )

    # ------------------------------------------------------------------ #
    # NodeBase interface
    # ------------------------------------------------------------------ #

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.own_size,)

    @property
    def prediction_keys(self) -> Tuple[str, ...]:
        keys = ("td",) if (self.op_td is not None or self.prior is not None) else ()
        return keys + ("bu",) if self.bidirectional else keys

    @property
    def needs_parents_curr(self) -> bool:
        return self.op_td is not None                   # root mode needs no parents

    @property
    def needs_children(self) -> bool:
        return self.bidirectional

    @property
    def supports_bottom_up(self) -> bool:
        return True

    @property
    def provides_pre_activations(self) -> bool:
        return True

    @property
    def normed_inputs(self): 
        return tuple(self.norms)


    def predict(self, inp: NodeInputs) -> NodePrediction:
        """Top-down prediction from `inp.parents_curr` (or the learned
        prior, in root mode), plus a bottom-up prediction from
        `inp.children_curr` if `bidirectional`."""
        self.check_inputs(inp)

        predictions, pre_activations = {}, {}
        if self.op_td is not None:  
            x_td = apply_affine(self.norms, "parents_curr", inp.parents_curr) # apply activity normalisation if specified
            pre_td = self.op_td(x_td)
        else:
            pre_td = self.prior                  # root mode: nothing to normalise

        if pre_td is not None:
            predictions["td"] = self.td_act_fn(pre_td)
            pre_activations["td"] = pre_td

        if self.bidirectional:
            x_bu = apply_affine(self.norms, "children_curr", inp.children_curr)  # apply activity normalisation if specified
            pre_bu = self.op_bu(x_bu)
            predictions["bu"] = self.bu_act_fn(pre_bu)
            pre_activations["bu"] = pre_bu

        return NodePrediction(predictions, pre_activations)

    def weight_entries(self) -> List[WeightEntry]:
        """Weight matrices tagged "ff" (nothing is recurrent). Biases and
        the root prior are not included."""
        entries = []
        if self.op_td is not None:
            entries.append(WeightEntry("op_td.weight", "ff", self.op_td.weight))
        if self.op_bu is not None:
            entries.append(WeightEntry("op_bu.weight", "ff", self.op_bu.weight))
        return entries
