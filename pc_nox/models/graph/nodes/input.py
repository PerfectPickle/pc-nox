"""graph/nodes/input.py

`InputNode`: a parameter-free, clamped data node with no parents and no
predictions -- the root of a classic SUPERVISED PCN, where the input is
clamped at the top and every other node predicts downward from it:

    image (InputNode, clamped) -> conv -> conv -> dense -> label (clamped in training)

It has no weights and returns no predictions, so it contributes zero
energy under any energy strategy that sums over predictions (including the
default "pc"), and it does not appear in `y_hat`. Its only job is to put
the data into the graph so children can name it as a parent.

Must be `clamped=True` (it is data, not an inferred state) and must be
supplied in every `observation`, since its children read it.

LIMITATION (by design, for now): because it has no weights and no
predictions, nothing can predict it from its children, and it can never be
a free (inferred) node. A graph with an `InputNode` root therefore runs in
ONE direction only (input -> ... -> label). Querying the reverse direction
(label clamped, input generated from the label up) needs a root node that
carries a bottom-up path, and a way to release a clamped node into the
inferred states; neither exists yet. See the design notes in the project
discussion; `pcn`-kind roots (dense) can already be clamped and carry a
bottom-up path for TRAINING, but cannot yet be released at test time.
"""

from typing import List, Tuple

import equinox as eqx

from .base import NodeBase, NodeInputs, NodePrediction, WeightEntry


class InputNode(NodeBase):
    """A clamped node with a state shape and nothing else."""
    shape: Tuple[int, ...] = eqx.field(static=True)

    def __init__(self, shape: Tuple[int, ...]):
        shape = tuple(shape)
        if not shape or not all(isinstance(d, int) and d > 0 for d in shape):
            raise ValueError(f"shape must be a non-empty tuple of positive ints, got {shape}")
        self.shape = shape

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return self.shape

    @property
    def prediction_keys(self) -> Tuple[str, ...]:
        return ()

    @property
    def needs_parents_curr(self) -> bool:
        return False

    def predict(self, inp: NodeInputs) -> NodePrediction:
        return NodePrediction({}, {})

    def weight_entries(self) -> List[WeightEntry]:
        return []
