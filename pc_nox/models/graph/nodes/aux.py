"""graph/nodes/aux.py

`AuxNode`: an auxilliary node to facilitate properly timed skip connections, 
after "Towards the Training of Deeper Predictive Coding Neural Networks" (Qi et al., 2025).
"""

from .base import NodeBase, NodePrediction
import equinox as eqx
from typing import Tuple

class AuxNode(NodeBase):
    shape: Tuple[int, ...] = eqx.field(static=True)
    bidirectional: bool = eqx.field(static=True)

    @property
    def state_shape(self): return self.shape
    @property
    def prediction_keys(self): return ("td", "bu") if self.bidirectional else ("td",)
    @property
    def needs_children(self): return self.bidirectional
    @property
    def supports_bottom_up(self): return True
    @property
    def provides_pre_activations(self): return True     # so Meta-PCN accepts it

    def predict(self, inp):
        self.check_inputs(inp)
        preds = {"td": inp.parents_curr}
        if self.bidirectional:
            preds["bu"] = inp.children_curr
        return NodePrediction(preds, dict(preds))        # identity: pre-activation == prediction

    def weight_entries(self): return []