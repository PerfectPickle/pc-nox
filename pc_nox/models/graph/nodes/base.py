"""graph/nodes/base.py

The abstract interface every node in a `GraphModel` satisfies,
plus the two small data containers that cross that interface.

Design rules (see the design discussion):

* A node is a pure computation unit: it owns weights and maps the inputs it
  is handed to predictions. It knows nothing about names, levels, parents,
  or other nodes. The model does the gathering (by name, from the spec) and
  hands the result in as a `NodeInputs`.
* A node never owns activities. States live in model-level dicts so that
  state-gradients (inference) and weight-gradients (learning) stay separate
  pytrees.
* A node does NOT own its energy. The energy is a strategy chosen on the
  `NodeSpec` by registry name (e.g. "pc", "meta"), and consumes a
  `NodePrediction`, so any node type can be paired with any energy.
* Everything the model branches on while tracing (`needs_*`,
  `prediction_keys`, `supports_trace`, `state_shape`) must be static and
  hashable -- plain Python values, never arrays -- so that branching
  resolves at trace time and the node stays JIT-compatible.
"""

from abc import abstractmethod
from typing import Dict, List, NamedTuple, Optional, Tuple

import equinox as eqx
from jaxtyping import Array


class NodeInputs(NamedTuple):
    """Everything the model may hand a node for one `predict` call.

    The model fills only the fields a node declares it needs (`needs_*`);
    the rest are `None` (an empty pytree subtree, so JIT-safe). A node reads
    only the fields it uses, so a static decoder can ignore all but
    `parents_curr`, while a recurrent node also reads `own_prev`.

    Multi-parent / multi-child gathers arrive already combined by the model
    (flatten-and-concatenate for dense consumers, channel-concatenate for
    conv consumers); a single parent or child passes through unchanged.

    Attributes:
        own_prev: This node's own state at t-1. Recurrent nodes only.
        parents_prev: Parent state(s) at t-1.
        parents_curr: Parent state(s) at the current time step (the live
            guess during inference, the parents' predictions during the
            feedforward initialisation sweep).
        children_curr: Child state(s) at the current time step. Only nodes
            with a bottom-up (child-driven) path need this.
        control_input: External input x_t. Control-type nodes only.
        skip_parents_curr: State of top-down skip connection source/s at the current time step.
        skip_children_curr: State of bottom-up skip connection source/s at the current time step.
    """
    own_prev: Optional[Array] = None
    parents_prev: Optional[Array] = None
    parents_curr: Optional[Array] = None
    children_curr: Optional[Array] = None
    control_input: Optional[Array] = None
    skip_parents_curr: Optional[Array] = None
    skip_children_curr: Optional[Array] = None


class NodePrediction(NamedTuple):
    """What a node's `predict` returns: one prediction per path, plus the
    pre-activations an energy strategy may need.

    Both fields are dicts keyed by PATH NAME. The conventional names are
    "td" (top-down: the node predicted from its PARENTS; REQUIRED for every
    unclamped node, since the initialisation sweep uses it) and "bu"
    (bottom-up: the node predicted from its CHILDREN; bidirectional nodes
    only). These are directions in the graph, not task roles -- see
    `NodeBase.predict`.
    A new path is just a new key.

    The energy strategy decides how to combine paths -- the standard "pc"
    strategy sums 0.5 * ||state - prediction||^2 over every key present --
    so adding a path to a node requires no change to the model or to
    existing strategies.

    Attributes:
        predictions: path name -> predicted state, same shape as the node's
            `state_shape`.
        pre_activations: path name -> the argument to the activation
            function for that path. May be empty (or omit paths) if the
            node has no separable pre-activation; strategies that need one
            (e.g. Meta-PCN's frozen derivative D(h)) must be rejected at
            config validation for such nodes, not discovered mid-trace.
    """
    predictions: Dict[str, Array]
    pre_activations: Dict[str, Array] = {}


class WeightEntry(NamedTuple):
    """One weight tensor, as seen by the regularisers.

    Attributes:
        name: Field name on the node (e.g. "W_rec"), for per-weight
            breakdowns and for locating the weight in a gradient pytree.
        tag: Scope tag the spec's regulariser scopes select on, e.g. "rec"
            (recurrent, self-to-self) or "ff" (everything else). Matches
            the existing `weight_decay_scope` / `orthogonal_scope` values.
        matrix: A 2-D view of the weight (a conv kernel is reshaped to
            `(out_channels, -1)`), so `penalties.py` -- which assumes 2-D
            arrays -- works unchanged. Return an empty list from
            `weight_entries` to exclude a node from weight regularisation
            entirely.
    """
    name: str
    tag: str
    matrix: Array


class NodeBase(eqx.Module):
    """Abstract interface for a node in a `GraphModel`.

    Subclasses implement `state_shape`, `predict` and `weight_entries`, and
    override the capability properties that differ from the defaults. The
    capability properties tell the model WHICH `NodeInputs` fields to
    gather, so a static node never pays for (or needs) recurrent inputs.
    They are properties rather than ClassVars so that an instance-level
    static setting (e.g. a node's `bidirectional` flag) can drive them;
    whatever they read must itself be an `eqx.field(static=True)`.
    """

    # ------------------------------------------------------------------ #
    # Required
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def state_shape(self) -> Tuple[int, ...]:
        """Shape of the state this node predicts, e.g. `(128,)` or
        `(C, H, W)`. The builder checks it against the node's `NodeSpec`
        at construction, so shape mismatches fail early and by name.
        """
        ...

    @abstractmethod
    def predict(self, inp: NodeInputs) -> NodePrediction:
        """Run every prediction path this node has, given `inp`.

        Must return a "td" entry in `predictions` -- the TOP-DOWN
        prediction, i.e. of this node's state from its PARENTS -- unless
        the node is clamped. `GraphModel.init_activities` starts every
        inferred (unclamped) node at its own "td" prediction, sweeping
        parents before children, and validation rejects an unclamped node
        that has none. Clamped nodes are data and are never initialised,
        so they may return no "td" (`InputNode` returns nothing at all).

        "td" and "bu" name graph DIRECTIONS (parent -> child, and child ->
        parent), not task roles. In a supervised chain (image -> ... ->
        label) "td" is the DISCRIMINATIVE direction; in a latent-on-top
        graph (latent -> ... -> image) it is the generative one. Only a
        top-down initialisation sweep exists; there is currently no
        bottom-up sweep, and the "bu" path is never used to initialise.

        Every returned array must have shape `state_shape`.
        Read only the `inp` fields this node declares it needs; call
        `check_inputs(inp)` first to fail with a clear message if the
        model failed to supply one.

        Must be a pure function of `(self, inp)` -- no hidden state -- so
        it can be traced, vmapped, scanned and differentiated freely.
        """
        ...

    @abstractmethod
    def weight_entries(self) -> List[WeightEntry]:
        """Every weight tensor the regularisers may act on, as 2-D views
        with a scope tag (see `WeightEntry`). Biases are included only if
        you want them regularised.
        """
        ...

    # ------------------------------------------------------------------ #
    # Capabilities (static; override as needed)
    # ------------------------------------------------------------------ #

    @property
    def prediction_keys(self) -> Tuple[str, ...]:
        """Path names `predict` returns. Used by config validation to
        check pairings (e.g. an energy that needs "bu") before tracing.
        """
        return ("td",)

    @property
    def needs_own_prev(self) -> bool:
        """Whether `predict` reads `inp.own_prev` (recurrent nodes)."""
        return False

    @property
    def needs_parents_prev(self) -> bool:
        """Whether `predict` reads `inp.parents_prev`."""
        return False

    @property
    def needs_parents_curr(self) -> bool:
        """Whether `predict` reads `inp.parents_curr`."""
        return True

    @property
    def needs_children(self) -> bool:
        """Whether `predict` reads `inp.children_curr` (bottom-up path)."""
        return False

    @property
    def needs_control_input(self) -> bool:
        """Whether `predict` reads `inp.control_input` (control nodes)."""
        return False

    @property
    def needs_skips_td(self) -> bool:
        """Whether `predict` reads `inp.skip_parents_curr` (skip connection source nodes)."""
        return False

    @property
    def supports_bottom_up(self) -> bool:
        """Whether this node type CAN have a bottom-up (child-driven) path.
        Whether it is switched on is a separate, per-instance setting
        (`NodeSpec.bidirectional`); `validate_pairings` rejects switching
        it on for a node type that cannot support it.
        """
        return False

    @property
    def needs_skips_bu(self) -> bool:
        """Whether `predict` reads `inp.skip_children_curr` (skip connection source nodes)."""
        return False

    @property
    def provides_pre_activations(self) -> bool:
        """Whether `predict` fills `NodePrediction.pre_activations`. Energy
        strategies that need them (e.g. Meta-PCN's D(h)) are rejected at
        validation for nodes that do not.
        """
        return False

    @property
    def normed_inputs(self) -> Tuple[str, ...]:
        """NodeInputs fields whose activity-norm STATISTIC the model applies in `_inputs`
        (this node holds the matching affine module). Default: none."""
        return ()

    @property
    def supports_trace(self) -> bool:
        """Whether this node supports tPC-E eligibility traces. Dense
        recurrent nodes can; conv nodes cannot (the `outer(error, input)`
        weight gradient is only valid for dense weights). A traceable node
        additionally implements the trace-input hook defined alongside the
        trace machinery; the model checks this flag before calling it.
        """
        return False

    @property
    def needs_skip_parents_curr(self): 
        return False

    @property
    def needs_skip_children_curr(self): 
        return False

    # ------------------------------------------------------------------ #
    # Concrete helpers
    # ------------------------------------------------------------------ #

    def check_inputs(self, inp: NodeInputs) -> None:
        """Raise a clear error if the model omitted an input this node
        declared it needs. Pure Python on static structure (is-None
        checks), so it runs at trace time and costs nothing at run time.
        """
        required = {
            "own_prev": self.needs_own_prev,
            "parents_prev": self.needs_parents_prev,
            "parents_curr": self.needs_parents_curr,
            "children_curr": self.needs_children,
            "control_input": self.needs_control_input,
            "skip_parents_curr": self.needs_skip_parents_curr,
            "skip_children_curr": self.needs_skip_children_curr,
        }
        missing = [f for f, need in required.items() if need and getattr(inp, f) is None]
        if missing:
            raise ValueError(
                f"{type(self).__name__} declared it needs {missing} but the "
                "model did not supply them"
            )
