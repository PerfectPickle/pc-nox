"""graph/nodes/tpch.py

`NodeBase` adapters around the existing tPC-H layers, which are reused
UNCHANGED (composition, not subclassing: each node holds a layer in its
`layer` field). That keeps `TpchModel` untouched, avoids two `predict`
signatures on one class, and keeps the weight names (`op_rec`,
`op_parents_prev`, ...) identical so weights can be copied between a
`TpchModel` and a `GraphModel` for equivalence testing.

All three expect FLAT (1-D) inputs, so they are registered with
`input_layout="flat"` (see `builders.py`).

Eligibility traces: `supports_trace` is left False for now. The trace
hooks arrive with the eligibility step (a `trace_inputs` method on the
recurrent nodes), at which point control/hidden nodes with `alpha` set
report True.
"""

from typing import List, Tuple, Callable, Optional, Tuple
from jaxtyping import Array, PRNGKeyArray
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from .base import NodeBase, NodeInputs, NodePrediction, WeightEntry


class TpchControlNode(NodeBase):
    """Top node: driven by the external control input and its own previous
    state. `s_hat_t = f(op_rec s_{t-1} + op_in x_t)` (optionally leaky)."""

    op_rec: eqx.nn.Linear  # recurrent weight (from self at t-1)
    op_in: eqx.nn.Linear = None  # optional control input
    op_bu: eqx.nn.Linear = None # optional bidirectional bottom up weight

    # static=True excludes this variable from the pytree, i.e. it's python metadata, not a leaf, and will be ignored during JAX autodifferentiation
    has_input: bool = eqx.field(static=True)
    td_act_fn: Callable = eqx.field(static=True) # not truly td because of rec self->self weight
    bu_act_fn: Callable = eqx.field(static=True)
    own_size: int = eqx.field(static=True)
    
    bidirectional: bool = eqx.field(static=True)
    alpha: Optional[float] = eqx.field(static=True, default=None)  # tPC-E leak rate; None = disabled

    def __init__(
        self,
        own_size: int,
        child_size: Optional[int] = None,
        input_size: Optional[int] = 0,
        td_act_fn: Callable = jnp.tanh,
        bu_act_fn: Callable = jnp.tanh,
        bidirectional: bool = False,
        alpha: Optional[float] = None,
        *,
        key: PRNGKeyArray,
    ):
        if own_size <= 0:
            raise ValueError(f"own_size must be positive, got {own_size}")
        if bidirectional and not child_size:
            raise ValueError("bidirectional=True requires child_size")

        k_rec, k_in, k_bu = jr.split(key, 3)
        self.own_size = own_size
        self.bidirectional = bidirectional
        self.td_act_fn = td_act_fn
        self.bu_act_fn = bu_act_fn
        self.alpha = alpha
        self.has_input = (input_size > 0)

        self.op_rec = eqx.nn.Linear(own_size, own_size, use_bias=False, key=k_rec)
        if self.has_input:
            self.op_in = eqx.nn.Linear(input_size, own_size, use_bias=True, key=k_in)
        if bidirectional:
            if child_size is None: # NOTE, handling of child_size wrong
                raise ValueError("A node with no child_size cannot be bidirectional")
            self.op_bu = eqx.nn.Linear(child_size[0], own_size, use_bias=True, key=k_bu)

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.op_rec.weight.shape[0],)
    
    @property
    def prediction_keys(self) -> Tuple[str, ...]:
        keys = ("td",) if (self.op_rec is not None) else ()
        return keys + ("bu",) if self.bidirectional else keys


    @property
    def needs_own_prev(self) -> bool:
        return True

    @property
    def needs_parents_curr(self) -> bool:
        return False

    @property
    def needs_control_input(self) -> bool:
        return self.has_input

    @property
    def provides_pre_activations(self) -> bool:
        return True

    @property
    def needs_children(self) -> bool:
        return self.bidirectional

    @property
    def supports_bottom_up(self) -> bool:
        return True

    # top down only
    def pre_activation(self, own_prev: Array, control_input: Optional[Array] = None) -> Array:
        """A @ s_{t-1} + B @ x_t, i.e. the argument to `act_fn` -- exposed
        separately from `predict` so `f'(pre)` can be computed (via
        `..eligibility.elementwise_deriv`) for the traced weight-update
        rule; see `TpchModel.param_grad_traced`.
        """
        pre_acts = self.op_rec(own_prev)
        if self.has_input and control_input is not None:
            pre_acts = pre_acts + self.op_in(control_input)
        return pre_acts

    def predict(self, inp: NodeInputs) -> NodePrediction:
        """s_hat_t = f(A @ s_{t-1} + B @ x_t), or its leaky-integrated
        variant (see class docstring) when `alpha` is set."""
        self.check_inputs(inp)

        predictions, pre_activations = {}, {}
        pre_td = self.pre_activation(inp.own_prev, inp.control_input)
        if pre_td is not None:
            pre_activations["td"] = pre_td
            instantaneous_td = self.td_act_fn(pre_td)
            if self.alpha is None or self.alpha == 0:
                predictions["td"] = instantaneous_td
            else:
                predictions["td"] = (1.0 - self.alpha) * state_prev + self.alpha * instantaneous_td

        if self.bidirectional:
            pre_bu = self.op_bu(inp.children_curr)
            predictions["bu"] = self.bu_act_fn(pre_bu)
            pre_activations["bu"] = pre_bu
            
        return NodePrediction(predictions, pre_activations)

    def weight_entries(self) -> List[WeightEntry]:
        entries = [WeightEntry("op_rec.weight", "rec", self.op_rec.weight)]
        if self.has_input:
            entries.append(WeightEntry("op_in.weight", "ff", self.op_in.weight))
        if self.op_bu is not None: # TODO -> consider adding more categories than 'rec' and 'ff'. 'ff' originally stood for feedforward, violted here.
            entries.append(WeightEntry("op_bu.weight", "ff", self.op_bu.weight))
        return entries


class TpchHiddenNode(NodeBase):
    """Middle node: `z_hat_t = f(P z_{t-1} + Q s_{t-1} + R s_t)` where `s`
    is the (combined) parent state; optionally leaky."""
    op_rec: eqx.nn.Linear  # recurrent weight (from self at t-1)
    op_parents_prev: eqx.nn.Linear # node can have multiple parents, treated as one inside the node
    op_parents_curr: eqx.nn.Linear
    op_bu: eqx.nn.Linear = None # optional bidirectional bottom up weight

    td_act_fn: Callable = eqx.field(static=True) # not truly td because of rec self->self weight
    bu_act_fn: Callable = eqx.field(static=True)
    own_size: int = eqx.field(static=True)
    
    bidirectional: bool = eqx.field(static=True)
    alpha: Optional[float] = eqx.field(static=True, default=None)  # tPC-E leak rate; None = disabled
    
    def __init__(
        self,
        own_size: int,
        parent_size: int,
        child_size: Optional[int] = None,
        td_act_fn: Callable = jnp.tanh,
        bu_act_fn: Callable = jnp.tanh,
        bidirectional: bool = False,
        alpha: Optional[float] = None,
        *,
        key: PRNGKeyArray,
    ):
        if parent_size <= 0:
            raise ValueError(f"TpchHiddenNode requires positive, nonzero parent_size")
        if own_size <= 0:
            raise ValueError(f"own_size must be positive, got {own_size}")
        if bidirectional and not child_size:
            raise ValueError("bidirectional=True requires child_size")

        k_rec, k_parent_prev, k_parent_curr, k_bu = jr.split(key, 4)
        self.own_size = own_size
        self.bidirectional = bidirectional
        self.td_act_fn = td_act_fn
        self.bu_act_fn = bu_act_fn
        self.alpha = alpha

        self.op_rec = eqx.nn.Linear(own_size, own_size, use_bias=False, key=k_rec)
        self.op_parents_prev = eqx.nn.Linear(parent_size, own_size, use_bias=False, key=k_parent_prev)
        self.op_parents_curr = eqx.nn.Linear(parent_size, own_size, use_bias=True, key=k_parent_curr)

        if bidirectional:
            if child_size is None or child_size[0] <= 0:
                raise ValueError("A node with no child_size cannot be bidirectional")
            self.op_bu = eqx.nn.Linear(child_size[0], own_size, use_bias=True, key=k_bu)


    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.op_rec.weight.shape[0],)

    @property
    def prediction_keys(self) -> Tuple[str, ...]:
        keys = ("td",) if not None in [self.op_parents_curr, self.op_parents_prev, self.op_rec] else ()
        return keys + ("bu",) if self.bidirectional else keys


    @property
    def needs_own_prev(self) -> bool:
        return True

    @property
    def needs_parents_prev(self) -> bool:
        return True

    @property
    def provides_pre_activations(self) -> bool:
        return True

    @property
    def needs_children(self) -> bool:
        return self.bidirectional

    @property
    def supports_bottom_up(self) -> bool:
        return True

    # top down only
    def pre_activation(self, state_prev: Array, parent_prev: Array, parent_curr: Array) -> Array:
        """P @ z_{t-1} + Q @ s_{t-1} + R @ s_t -- see
        `TpchControlNode.pre_activation` for why this is exposed."""
        return (
            self.op_rec(state_prev)
            + self.op_parents_prev(parent_prev)
            + self.op_parents_curr(parent_curr)
        )

    def predict(self, inp: NodeInputs) -> NodePrediction:
        self.check_inputs(inp)

        predictions, pre_activations = {}, {}
        pre_td = self.pre_activation(inp.own_prev, inp.parents_prev, inp.parents_curr)
        if pre_td is not None:
            pre_activations["td"] = pre_td
            instantaneous_td = self.td_act_fn(pre_td)
            if self.alpha is None or self.alpha == 0:
                predictions["td"] = instantaneous_td
            else:
                predictions["td"] = (1.0 - self.alpha) * state_prev + self.alpha * instantaneous_td

        if self.bidirectional:
            pre_bu = self.op_bu(inp.children_curr)
            predictions["bu"] = self.bu_act_fn(pre_bu)
            pre_activations["bu"] = pre_bu
            
        return NodePrediction(predictions, pre_activations)

    def weight_entries(self) -> List[WeightEntry]:
        entries= [
            WeightEntry("op_rec.weight", "rec", self.op_rec.weight),
            WeightEntry("op_parents_prev.weight", "ff", self.op_parents_prev.weight),
            WeightEntry("op_parents_curr.weight", "ff", self.op_parents_curr.weight),
        ]
        if self.op_bu is not None: 
            entries.append(WeightEntry("op_bu.weight", "ff", self.op_bu.weight))
        return entries


class TpchObservationNode(NodeBase):
    """Leaf node: a pure linear readout `y_hat = C s_t` of its (combined)
    parent's CURRENT state. No memory, no nonlinearity. Normally `clamped`
    (its target is observed data), with its loss chosen by the spec's
    `energy` ("pc" for squared error, "ce" for cross-entropy on logits).
    """
    op_parents_curr: eqx.nn.Linear

    def __init__(self, obs_size: int, parent_size: int, key: PRNGKeyArray):
        self.op_parents_curr = eqx.nn.Linear(parent_size, obs_size, use_bias=False, key=key)

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.op_parents_curr.weight.shape[0],)

    @property
    def prediction_keys(self) -> Tuple[str, ...]:
        keys = ("td",) if (self.op_parents_curr is not None) else ()
        return keys


    @property
    def provides_pre_activations(self) -> bool:
        return True

    def predict(self, inp: NodeInputs) -> NodePrediction:
        """y_hat_t = C @ z_t  (no activation function -- pure linear readout)"""
        self.check_inputs(inp)
        y_hat = self.op_parents_curr(inp.parents_curr)
        return NodePrediction({"td": y_hat}, {"td": y_hat})   # linear: pre-activation == prediction

    def weight_entries(self) -> List[WeightEntry]:
        return [WeightEntry("op_parents_curr.weight", "ff", self.op_parents_curr.weight)]
