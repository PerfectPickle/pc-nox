"""graph/nodes/tpch.py

`NodeBase` adapters around the existing tPC-H layers, which are reused
UNCHANGED (composition, not subclassing: each node holds a layer in its
`layer` field). That keeps `TpchModel` untouched, avoids two `predict`
signatures on one class, and keeps the weight names (`W_rec`,
`W_parent_prev`, ...) identical so weights can be copied between a
`TpchModel` and a `GraphModel` for equivalence testing.

All three expect FLAT (1-D) inputs, so they are registered with
`input_layout="flat"` (see `builders.py`).

Eligibility traces: `supports_trace` is left False for now. The trace
hooks arrive with the eligibility step (a `trace_inputs` method on the
recurrent nodes), at which point control/hidden nodes with `alpha` set
report True.
"""

from typing import List, Tuple

from jaxtyping import Array

from pc_nox.models.tpch.layers import TpchControlLayer, TpchHiddenLayer, TpchObservationLayer

from .base import NodeBase, NodeInputs, NodePrediction, WeightEntry


class TpchControlNode(NodeBase):
    """Top node: driven by the external control input and its own previous
    state. `s_hat_t = f(W_rec s_{t-1} + W_in x_t)` (optionally leaky)."""
    layer: TpchControlLayer

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.layer.W_rec.weight.shape[0],)

    @property
    def needs_own_prev(self) -> bool:
        return True

    @property
    def needs_parents_curr(self) -> bool:
        return False

    @property
    def needs_control_input(self) -> bool:
        return self.layer.has_input

    @property
    def provides_pre_activations(self) -> bool:
        return True

    def predict(self, inp: NodeInputs) -> NodePrediction:
        self.check_inputs(inp)
        pred = self.layer.predict(inp.own_prev, inp.control_input)
        pre = self.layer.pre_activation(inp.own_prev, inp.control_input)
        return NodePrediction({"td": pred}, {"td": pre})

    def weight_entries(self) -> List[WeightEntry]:
        entries = [WeightEntry("layer.W_rec.weight", "rec", self.layer.W_rec.weight)]
        if self.layer.has_input:
            entries.append(WeightEntry("layer.W_in.weight", "ff", self.layer.W_in.weight))
        return entries


class TpchHiddenNode(NodeBase):
    """Middle node: `z_hat_t = f(P z_{t-1} + Q s_{t-1} + R s_t)` where `s`
    is the (combined) parent state; optionally leaky."""
    layer: TpchHiddenLayer

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.layer.W_rec.weight.shape[0],)

    @property
    def needs_own_prev(self) -> bool:
        return True

    @property
    def needs_parents_prev(self) -> bool:
        return True

    @property
    def provides_pre_activations(self) -> bool:
        return True

    def predict(self, inp: NodeInputs) -> NodePrediction:
        self.check_inputs(inp)
        pred = self.layer.predict(inp.own_prev, inp.parents_prev, inp.parents_curr)
        pre = self.layer.pre_activation(inp.own_prev, inp.parents_prev, inp.parents_curr)
        return NodePrediction({"td": pred}, {"td": pre})

    def weight_entries(self) -> List[WeightEntry]:
        return [
            WeightEntry("layer.W_rec.weight", "rec", self.layer.W_rec.weight),
            WeightEntry("layer.W_parent_prev.weight", "ff", self.layer.W_parent_prev.weight),
            WeightEntry("layer.W_parent_curr.weight", "ff", self.layer.W_parent_curr.weight),
        ]


class TpchObservationNode(NodeBase):
    """Leaf node: a pure linear readout `y_hat = C s_t` of its (combined)
    parent's CURRENT state. No memory, no nonlinearity. Normally `clamped`
    (its target is observed data), with its loss chosen by the spec's
    `energy` ("pc" for squared error, "ce" for cross-entropy on logits).
    """
    layer: TpchObservationLayer

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.layer.W_parent.weight.shape[0],)

    @property
    def provides_pre_activations(self) -> bool:
        return True

    def predict(self, inp: NodeInputs) -> NodePrediction:
        self.check_inputs(inp)
        y_hat = self.layer.predict(inp.parents_curr)
        return NodePrediction({"td": y_hat}, {"td": y_hat})   # linear: pre-activation == prediction

    def weight_entries(self) -> List[WeightEntry]:
        return [WeightEntry("layer.W_parent.weight", "ff", self.layer.W_parent.weight)]
