"""graph/model.py

`GraphModel`: a mixed predictive-coding network whose structure is a
graph of named nodes (see `config.py`) instead of a fixed chain, e.g.
control -> hidden -> observation. Branches ("tails"), multi-parent
nodes and mixed dense/conv nodes all fall out of the config; there is no
tail-specific code. Two tails never interact because no node lists a node
from another tail as a parent, and a shared parent simply receives the
SUM of its children's error gradients from `jax.grad`. 
Both temporal and static nodes are supported and can coexist in the same graph.
Bidirectionality is supported.

Responsibilities (the split we designed):

  * nodes (`nodes/`)    own weights and compute predictions, nothing else
  * energies            score a node's state against its predictions
  * builders            turn specs into nodes (`builders.py`)
  * THIS class          data flow: gather each node's inputs by name,
                        call the nodes, sum the energies, settle, and
                        differentiate -- plus the `ModelBase` hooks

State layout (the same calling convention the shared runners use):

  states_prev / states_curr : {name: Array} for every UNCLAMPED node
  (A clamped node with `release=True` that is omitted from `observation`
   becomes an INFERRED state: it appears in `states`, is initialised by the
   sweep, and its energy is scored. Without `release` an omitted clamped
   node is dropped, as below.)
  observation (the runners' `y`) : {name: Array} for CLAMPED nodes. A clamped
                              node may be OMITTED (a missing modality): its
                              energy term is dropped but its prediction is
                              still returned in `y_hat` -- this is how a
                              label tail becomes a classifier at test time.
                              Keys that are not clamped nodes are an error.
  control_input             : Array or None, handed to nodes that need it

Extension points (each is one place, on purpose):

  * a new node kind        -> `builders.register_node`
  * a new energy           -> `energies.ENERGY_REGISTRY`; energies that need
                              a per-settle frozen context (Meta-PCN) declare an
                              `init_frozen` hook; `frozen_context` builds it once
                              per settle and `_node_energies` hands it to them
  * weight normalisation   -> `weight_norm.py`, switched on by the config's
                              `weight_norm`; runs via `postprocess_params`
  * eligibility traces     -> not implemented yet; they need a
                              `trace_inputs` hook on recurrent nodes and a
                              loop over `supports_trace` nodes in a
                              `param_grad_traced` (see `TpchModel`'s)
  * post-update transforms -> override `postprocess_params` (inherited
                              no-op from `ModelBase`)
"""

import functools
import warnings
from typing import ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from math import prod

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from pc_nox.core import penalties
from pc_nox.core.model_base import ModelBase
from pc_nox.engine import ode_settling

from .builders import build_nodes, combine, get_builder, combine_normed, node_key
from .config import GraphModelConfig, NodeSpec
from .energies import ENERGY_REGISTRY
from .nodes.base import NodeBase, NodeInputs, NodePrediction, WeightEntry
from .nodes.norm import resolve_activity_norm
from .validation import validate_pairings
from .weight_norm import normalise_weights, tag_in_scope

States = Dict[str, Array]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=None)
def _children_of(config: GraphModelConfig) -> Dict[str, Tuple[str, ...]]:
    """Cached `config.children()` (configs are hashable; this runs at trace time)."""
    return config.children()


def _get(table: Optional[Mapping[str, Array]], name: str, what: str, needed_by: str) -> Array:
    """Look up `name` in `table` with an error that says what is missing."""
    if table is None or name not in table:
        raise KeyError(f"node {needed_by!r} needs {what} of {name!r}, which was not provided")
    return table[name]


def _check_observation(config: GraphModelConfig, observation: Optional[Mapping[str, Array]]) -> None:
    """Reject observation keys that are not clamped nodes (almost always a typo)."""
    if observation is None:
        return
    clamped = {s.name for s in config.clamped}
    unknown = sorted(set(observation) - clamped)
    if unknown:
        raise ValueError(
            f"observation has keys {unknown} that are not clamped nodes; clamped nodes: {sorted(clamped)}"
        )


def select_weights(entries: Sequence[WeightEntry], scope: str) -> List[Array]:
    """2-D weight matrices whose tag matches `scope` ('all' | 'rec' | 'ff')."""
    return [e.matrix for e in entries if tag_in_scope(e.tag, scope)]


def _scopes_overlap(a: str, b: str) -> bool:
    return a == "all" or b == "all" or a == b


def _warn_if_regularisers_conflict(config: GraphModelConfig) -> None:
    """Per node: weight decay pulls singular values to 0, the orthogonal
    penalty to 1; with overlapping scopes and decay >= 2 * orthogonal there
    is no stable nonzero equilibrium (see `TpchModel`'s warning)."""
    if not _scopes_overlap(config.weight_decay_scope, config.orthogonal_scope):
        return
    for spec in config.node_specs:
        wd, orth = config.get(spec, "weight_decay"), config.get(spec, "orthogonal_penalty")
        if wd > 0.0 and orth > 0.0 and wd >= 2 * orth:
            warnings.warn(
                f"node {spec.name!r}: weight_decay ({wd}) >= 2 * orthogonal_penalty ({orth}) on "
                f"overlapping scopes ({config.weight_decay_scope!r}, {config.orthogonal_scope!r}); "
                "the weights will tend to collapse toward zero.",
                stacklevel=3,
            )


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

class GraphModel(eqx.Module, ModelBase):
    """A graph of named predictive-coding nodes.

    Build from a config:

        config = GraphModelConfig(node_specs=(
            NodeSpec("core", "tpch_control", 64),
            NodeSpec("vis", "tpch_hidden", 128, parents=("core",)),
            NodeSpec("aud", "tpch_hidden", 32,  parents=("core",)),
            NodeSpec("vis_obs", "tpch_obs", 4096, parents=("vis",), clamped=True,),
            NodeSpec("aud_obs", "tpch_obs", 16,   parents=("aud",), clamped=True),
        ))
        model = GraphModel(config, key=jax.random.PRNGKey(0))
        states = GraphModel.zero_activities(config)         # {name: zeros}
        y = {"vis_obs": ..., "aud_obs": ...}                # the runners' `y`

    Attributes:
        config: The static, hashable `GraphModelConfig`.
        nodes: name -> built node module, in canonical (level) order.
    """
    model_type: ClassVar[str] = "graph"
    config_cls: ClassVar[type] = GraphModelConfig
    config: GraphModelConfig = eqx.field(static=True)
    nodes: Dict[str, NodeBase]

    def __init__(self, config: GraphModelConfig, *, key: PRNGKeyArray):
        if not isinstance(config, GraphModelConfig):
            raise TypeError(f"config must be a GraphModelConfig, got {type(config).__name__}")
        self.config = config
        self.nodes = build_nodes(config, key)
        validate_pairings(config, self.nodes, ENERGY_REGISTRY)   # before anything is traced
        _warn_if_regularisers_conflict(config)

    # ------------------------------------------------------------------ #
    # Gathering a node's inputs (the model's core job)
    # ------------------------------------------------------------------ #

    def _inputs(
        self,
        spec: NodeSpec,
        node: NodeBase,
        states_prev: States,
        curr: Mapping[str, Array],
        control_input: Optional[Array],
        zero_fill: Optional[str] = None,
    ) -> NodeInputs:
        """Build the `NodeInputs` a node declared it needs. `curr` holds the
        current values of every node known so far (inferred states plus
        clamped observations). Multi-parent / multi-child inputs are
        combined per the node kind's layout (see `builders.combine`).

        `zero_fill` ("children" | "parents" | None): during an initialisation
        sweep, the neighbours on the side the sweep has not reached yet are
        replaced by zeros of the right shape. Their paths' results are
        discarded (only the sweep direction's prediction is used), so the
        stand-in is harmless; any OTHER missing neighbour is still an error."""
        layout = get_builder(spec.kind).input_layout
        norm = resolve_activity_norm(self.config.get(spec, "activity_norm"))
        mode = self.config.activity_norm_mode

        def gather(field, arrays):
            stat = norm.stat if (norm is not None and field in node.normed_inputs) else None
            return combine_normed(arrays, layout, stat, mode)

        def cur(p):   # same rule parents_curr already uses
            if p in curr: 
                return curr[p]
            if zero_fill == "parents": 
                return jnp.zeros(self.config.spec(p).shape)
            return _get(curr, p, "the current state (or observation)", spec.name)

        fields = {}
        if node.needs_own_prev:
            x = _get(states_prev, spec.name, "its previous state", spec.name)
            fields["own_prev"] = norm.stat(x) if (norm is not None and "own_prev" in node.normed_inputs) else x
        if node.needs_parents_prev:
            fields["parents_prev"] = gather(
                "parents_prev",
                [_get(states_prev, p, "the previous state", spec.name) for p in spec.parents])
        if node.needs_parents_curr:
            fields["parents_curr"] = gather(
                "parents_curr",
                [curr[p] if p in curr else (jnp.zeros(self.config.spec(p).shape) if zero_fill == "parents" else
                 _get(curr, p, "the current state (or observation)", spec.name)) for p in spec.parents])
        if node.needs_children:
            kids = []
            for c in _children_of(self.config)[spec.name]:
                if c in curr:
                    kids.append(curr[c])
                elif zero_fill == "children" or self.config.spec(c).clamped:
                    kids.append(jnp.zeros(self.config.spec(c).shape))
                else:
                    raise KeyError(f"node {spec.name!r} needs the current state of child {c!r}, which was not provided")
            fields["children_curr"] = gather("children_curr", kids)
        if node.needs_control_input:
            fields["control_input"] = control_input
        if node.needs_skips_td:
            fields["skip_parents_curr"] = combine([cur(p) for p in spec.skip_parents], layout)
        return NodeInputs(**fields)

    # ------------------------------------------------------------------ #
    # Prediction and initialisation
    # ------------------------------------------------------------------ #

    def predict(
        self,
        states_prev: States,
        states_curr: States,
        control_input: Optional[Array] = None,
        observation: Optional[Mapping[str, Array]] = None,
    ) -> Tuple[Dict[str, NodePrediction], Dict[str, Array]]:
        """Run every node's `predict` once.

        Returns `(predictions, y_hat)`: `predictions` maps EVERY node name
        to its `NodePrediction`; `y_hat` maps each CLAMPED node that makes a
        prediction to its "td" prediction (the runners' `y_hat`; a pure-data
        `InputNode` has none and is absent).
        """
        _check_observation(self.config, observation)
        curr = {**states_curr, **(observation or {})}
        predictions = {
            spec.name: self.nodes[spec.name].predict(
                self._inputs(spec, self.nodes[spec.name], states_prev, curr, control_input))
            for spec in self.config.node_specs
        }
        y_hat = {s.name: predictions[s.name].predictions["td"] for s in self.config.clamped
                 if "td" in predictions[s.name].predictions}        # pure-data nodes (InputNode) predict nothing
        return predictions, y_hat

    def inferred_specs(self, observation: Optional[Mapping[str, Array]] = None) -> Tuple[NodeSpec, ...]:
        """The nodes whose state is inferred in a call with this `observation`,
        in canonical order: every unclamped node, plus every clamped node that
        is `release`d and omitted from `observation`."""
        obs = observation or {}
        return tuple(s for s in self.config.node_specs if (not s.clamped) or (s.release and s.name not in obs))

    def init_activities(
        self,
        states_prev: States,
        control_input: Optional[Array] = None,
        observation: Optional[Mapping[str, Array]] = None,
        direction: Optional[str] = None,
    ) -> States:
        """Amortised initialisation sweep: each inferred node starts at its own
        prediction in the sweep direction, so settling begins from a good guess.

        direction="td" (default `config.init_direction`): parents before
            children, each node at its "td" prediction.
        direction="bu": children before parents, each node at its "bu"
            prediction. Every inferred node on the path must be
            bidirectional, and its children must be observed or already swept.

        A node with no prediction in the sweep direction (e.g. a released
        data root in a "td" sweep) starts at zero, as the paper initialises
        missing pixels. Clamped observations are used as given.
        """
        direction = direction or self.config.init_direction
        if direction not in ("td", "bu"):
            raise ValueError(f"direction must be 'td' or 'bu', got {direction!r}")
        _check_observation(self.config, observation)
        known: Dict[str, Array] = dict(observation or {})
        states_curr: States = {}
        inferred = self.inferred_specs(observation)
        for spec in (inferred if direction == "td" else tuple(reversed(inferred))):
            node = self.nodes[spec.name]
            if direction not in node.prediction_keys:
                value = jnp.zeros(spec.shape)
            else:
                inp = self._inputs(spec, node, states_prev, known, control_input,
                                   zero_fill="children" if direction == "td" else "parents")
                value = node.predict(inp).predictions[direction]
            states_curr[spec.name] = value
            known[spec.name] = value
        return states_curr

    # ------------------------------------------------------------------ #
    # Energy
    # ------------------------------------------------------------------ #

    def frozen_context(
        self,
        states_prev: States,
        states_ref: States,
        observation: Optional[Mapping[str, Array]],
        control_input: Optional[Array] = None,
        learning: bool = False,
    ) -> Dict[str, object]:
        """Build, once per settle, the context that strategies with an
        `init_frozen` hook (currently "meta") keep constant while states move.

        Plain English: evaluate every node's predictions at a REFERENCE set of
        states (`states_ref`, normally the initialisation sweep's output
        $z^{(0)}$) and let each such strategy record what it needs from them
        (for Meta-PCN: the frozen predictions $c$ and derivatives
        $\\phi'(h^{(0)})$). Settling then differentiates energies that use
        this fixed context w.r.t. the states only.

        Args:
            states_prev: Previous-step states (for recurrent nodes).
            states_ref: The linearisation point, e.g. `init_activities(...)`.
            observation: Clamped data, as everywhere.
            control_input: External input, as everywhere.
            learning: Build the LEARNING-mode context instead (Meta-PCN: keep
                the predictions live in the weights and drop the coupling
                term; used inside `param_grad`).

        Returns:
            `{node_name: context}` for every node whose energy strategy has an
            `init_frozen` hook; empty if none does (so it is free to call
            for a plain-PC graph).
        """
        hooked = [s for s in self.config.node_specs
                  if ENERGY_REGISTRY[self.config.get(s, "energy")].init_frozen is not None]
        if not hooked:
            return {}
        predictions, _ = self.predict(states_prev, states_ref, control_input, observation)
        return {
            s.name: ENERGY_REGISTRY[self.config.get(s, "energy")].init_frozen(
                self.nodes[s.name], predictions[s.name], learning)
            for s in hooked
        }


    def _node_energies(
        self,
        states_prev: States,
        states_curr: States,
        obs: Optional[Mapping[str, Array]], # observation
        control_input: Optional[Array],
        frozen: Optional[Mapping[str, object]] = None,
        highway_matrices: Optional[Mapping[Tuple[str, str], Array]] = None, 
    ) -> List[Array]:
        """Each node's precision-weighted energy, in canonical order. This
        is the ONE place energy strategies are called. `frozen` is the
        per-settle context from `frozen_context`; it is passed to each node's
        strategy (and required by those that declare an `init_frozen` hook)."""
        predictions, _ = self.predict(states_prev, states_curr, control_input, obs)
        curr = {**states_curr, **(obs or {})}
        energies = []

        # --- Highway Error Propagation: output errors of the observed SOURCE nodes ---
        # Only nodes that some checkpoint lists in `highway_from` are sources. This keeps
        # non-HEP graphs untouched and avoids calling energies that need a frozen context.
        # each element of highway_sources is a src tuple of the form: ("node_name", "pathway")
        highway_sources = sorted({src for s in self.config.node_specs for src in s.highway_sources})
        source_errors = {
            (src_name, path): self._highway_source_error(
                self.config.spec(src_name), obs[src_name], predictions[src_name],
                {"td": self.config.get(self.config.spec(src_name), "td_precision"),
                "bu": self.config.get(self.config.spec(src_name), "bu_precision")}, path)
            for src_name, path in highway_sources if src_name in obs and path in predictions[src_name].predictions
        }

        for spec in self.config.node_specs:
            if spec.clamped and spec.name not in curr:        # unobserved (missing modality): no energy term
                energies.append(jnp.asarray(0.0))
                continue
            strategy = ENERGY_REGISTRY[self.config.get(spec, "energy")]
            # pathway weighting, td vs bu
            weights = {"td": self.config.get(spec, "td_precision"), "bu": self.config.get(spec, "bu_precision")}

            node_frozen = None if frozen is None else frozen.get(spec.name)
            if strategy.init_frozen is not None and node_frozen is None:
                raise ValueError(
                    f"node {spec.name!r} uses energy {self.config.get(spec, 'energy')!r}, which needs a frozen "
                    "context: pass `frozen=model.frozen_context(states_prev, states_init, observation, control_input)` "
                    "(settle_scan and the runners already do this)")
            
            # ---------------------------------------------------------------------------
            #    per node in the loop: this node's highway term, a SCALAR.
            #    Implements  z_i^T  V_{L->i}  sg(eps_L), summed over this node's sources.
            # ---------------------------------------------------------------------------
            hw = jnp.asarray(0.0)
            if spec.name in states_curr:                      # only inferred nodes have a state to nudge
                z = states_curr[spec.name].ravel()
                for src, path in spec.highway_sources:
                    if (src, path) in source_errors:                  # observed => has an error
                        # V @ err : matrix-vector product, shape (n_node,). This is the
                        #           "nudge direction" delivered to the node in one hop.
                        # vdot(z, .) : dot product of the state with that nudge -> a scalar.
                        # Taking jax.grad of this scalar w.r.t. z gives exactly  V @ err, so
                        # each inference step moves z by (alpha * V @ eps), i.e. Eq. 8.
                        hw = hw + jnp.vdot(z, highway_matrices[(spec.name, src, path)] @ source_errors[(src, path)].ravel())

            energies.append(
                strategy.fn(curr[spec.name], predictions[spec.name], node_frozen, weights)
                + self.config.highway_alpha * hw
                )
        return energies


    def _weight_reg_by_node(self) -> Array:
        """Weight decay + orthogonal penalty per node (canonical order),
        using each node's resolved coefficients and the model-wide scopes."""
        cfg = self.config
        terms = []
        for spec in cfg.node_specs:
            entries = self.nodes[spec.name].weight_entries()
            l2 = penalties.l2_reg(select_weights(entries, cfg.weight_decay_scope), cfg.get(spec, "weight_decay"))
            orth = penalties.orthogonal_reg(
                select_weights(entries, cfg.orthogonal_scope), cfg.get(spec, "orthogonal_penalty"))
            terms.append(l2 + orth)
        return jnp.stack(terms)

    def _activity_reg_by_node(self, states_curr: States) -> Array:
        """Activity penalty per node (canonical order); nodes that are not inferred states get 0."""
        cfg = self.config
        terms = []
        for spec in cfg.node_specs:
            if spec.name not in states_curr:                  # clamped (or dropped): not an inferred state
                terms.append(jnp.asarray(0.0))
            else:
                terms.append(penalties.activity_reg(
                    [states_curr[spec.name]], cfg.get(spec, "activity_decay"), cfg.activity_reg_type))
        return jnp.stack(terms)


    def _highway_matrices(self):
        r"""Build the fixed random Highway Error Propagation (HEP) matrices $V_{src \to i}$.

        One matrix per highway edge, i.e. per (checkpoint node, source node,
        source path) triple listed in the specs' `highway_from_td` /
        `highway_from_bu`. Each matrix projects the source's output error
        $\epsilon$ (see `_highway_source_error`) onto the checkpoint's state, so
        the checkpoint's energy gets the term $z_i^\top V\,\mathrm{sg}(\epsilon)$
        (Eq. 7 of the HEP paper).

        The matrices are never trained and never stored. Each is a pure function of
        `config.highway_seed` and the node names, so the same matrices are
        regenerated after a checkpoint reload, and adding or removing other nodes
        does not change an existing edge's matrix. Because they are not model
        leaves, the optimiser, weight decay and `weight_norm` cannot touch them.

        Call once per settle (see `_energy_closures`), not per inference step:
        they are constants while the states settle.

        Returns:
            A dict mapping `(checkpoint_name, source_name, path)` to a matrix `V`,
            where:

            * `checkpoint_name`: the node whose STATE is nudged (the spec that
              lists the source).
            * `source_name`: the clamped, observed node that supplies the error.
            * `path`: "td" or "bu", which of the source's prediction errors is
              used. A source listed under both paths gets two independent matrices.
            * `V` has shape `(n_checkpoint, n_source)`: rows are the checkpoint's
              units, columns are the source's units, with `(C, H, W)` shapes
              flattened to `C*H*W`. `V @ eps.ravel()` is the nudge direction
              delivered to the checkpoint's flattened state.

            Entries are i.i.d. $\mathcal{N}(0, \sigma_v^2)$ with
            $\sigma_v$ = `config.highway_sigma`. The dict is empty if no node lists
            any highway source.
        """
        cfg = self.config
        # One base PRNG key, derived from an integer seed stored in the config.
        # Because the seed lives in the config, the same V is regenerated after a
        # checkpoint reload. V never needs to be saved or trained.
        base = jr.PRNGKey(cfg.highway_seed)

        # Dict comprehension: one matrix for every (checkpoint node, source node) pair.
        #   key   = (checkpoint name, source name), e.g. ("vis", "label_obs")
        #   value = V, a random matrix of shape (n_checkpoint, n_source)
        return {
            (s.name, src, path): cfg.highway_sigma * jr.normal(          # sigma_v * N(0, 1) entries
                # A key unique to this pair, derived from the NAMES (via crc32 in
                # node_key), so adding or removing other nodes never changes this V.
                node_key(base, f"{s.name}<-{src}:{path}"),
                # Rows = number of units in the checkpoint's state.
                # Columns = number of units in the source node's error vector.
                # prod() flattens a shape like (C, H, W) to C*H*W, so conv states work too.
                (prod(s.shape), prod(cfg.spec(src).shape)),
            )
            for s in cfg.node_specs          # every node is a potential checkpoint...
            for src, path in s.highway_sources        # ...for each source, path it lists (empty tuple = none)
        }


    # ---------------------------------------------------------------------------
    #    The output error eps = -dE_src/d(mu_src^path) for ONE source node and ONE
    #    of its prediction paths ("td" or "bu").
    #    This is the paper's eps_L: how the source's loss would change if that
    #    prediction moved, with the sign flipped, so it points "toward the target".
    #    It includes the source's precision weight for that path.
    # ---------------------------------------------------------------------------
    def _highway_source_error(self, spec, state, pred, weights, path):
        r"""Output error of one source node on one prediction path, for a highway.

        Computes the paper's $\epsilon_L = -\partial E_{src} / \partial \mu_{src}^{path}$
        (Eq. 2 of the HEP paper): how the source's own energy would change if its
        prediction on `path` moved, with the sign flipped so the result points toward
        the target. Differentiating the source's own energy, not a second hand-written
        formula, means the highway always uses the same loss the source is trained
        with. The result includes the source's precision weight for `path`.

        The error is returned stop-gradiented, so the highway term is strictly
        one-way: no gradient flows back through it into the source's prediction, and
        so none into the weights (the "sg" in Eq. 7). The local weight update (Eq. 9)
        is therefore unchanged.

        Worked examples (the data is the clamped `state`, written y):
            "pc":  energy = sum_k w_k * 0.5 * ||y - mu_k||^2
                   -> eps = w_path * (y - mu_path). The other paths' terms are
                   constant in `mu`, so they drop out.
            "ce":  energy = w_td * CE(y, softmax(mu_td))
                   -> eps = w_td * (y - softmax(mu_td)).

        Args:
            spec: `NodeSpec` of the SOURCE node (e.g. the label node, or a
                bidirectional data root).
            state: The source's observed data (e.g. the one-hot label). It must be
                an observation: inferred sources are not supported.
            pred: The source's full `NodePrediction`, with every path it returns
                (td and/or bu). Paths other than `path` are held fixed.
            weights: The `{"td": precision, "bu": precision}` dict for the source,
                as built in `_node_energies`.
            path: Which prediction path's error to return, "td" or "bu".

        Returns:
            An array shaped like `pred.predictions[path]`, stop-gradiented.

        Note:
            The "ce" energy never reads the "bu" prediction, so for `path="bu"` the
            gradient is exactly zero and the highway would silently do nothing.
            `validate_pairings` must reject that pairing (and energies that need a
            frozen context, such as "meta", since `None` is passed for it here).
        """
        # Look up whichever energy this source uses ("pc", "ce", ...).
        strat = ENERGY_REGISTRY[self.config.get(spec, "energy")]

        # The source's own energy as a scalar function of ONE path's prediction `mu`:
        # that path is replaced by `mu` and every other path is held fixed, so
        # jax.grad below is the partial derivative with respect to that path only.
        # `state` (the data) is also held fixed.
        def energy_of(mu):
            preds = {**pred.predictions, path: mu}
            return strat.fn(state, NodePrediction(preds, pred.pre_activations), None, weights)

        # stop_gradient freezes eps as a constant, so no gradient flows back through the
        # highway term into the source's prediction (and so into the weights).
        # This is the "sg(eps_L)" of Eq. 7: the highway is strictly one-way.
        return jax.lax.stop_gradient(-jax.grad(energy_of)(pred.predictions[path]))


    def energy_fn(
        self,
        states_prev: States,
        states_curr: States,
        observation: Optional[Mapping[str, Array]],
        control_input: Optional[Array] = None,
        weight_reg_total: Optional[Array] = None,
        return_layerwise: bool = False,
        frozen: Optional[Mapping[str, object]] = None,
        highway_matrices: Optional[Mapping[Tuple[str, str], Array]] = None, 
    ) -> Array:
        """Total free energy: every node's precision-weighted energy, plus
        weight and activity regularisation.

        `frozen`: per-settle context from `frozen_context`; only needed if some
        node's energy has an `init_frozen` hook (Meta-PCN), else leave None.

        `weight_reg_total`: a precomputed weight-regularisation scalar, for
        callers doing repeated inference steps (weights cannot change while
        states settle); `None` recomputes it from `self`.

        `return_layerwise`: if True, an unsummed Array with one entry per
        node in canonical order (matches `layer_labels`), each node's energy
        plus its own regularisation; its sum equals the scalar result. An
        unobserved clamped node contributes 0 to its entry's energy term.
        """
        if highway_matrices is None:
            highway_matrices = self._highway_matrices()
        node_energies = jnp.stack(self._node_energies(states_prev, states_curr, observation, control_input, frozen, highway_matrices))
        activity = self._activity_reg_by_node(states_curr)
        if return_layerwise:
            return node_energies + self._weight_reg_by_node() + activity
        if weight_reg_total is None:
            weight_reg_total = jnp.sum(self._weight_reg_by_node())
        return jnp.sum(node_energies) + weight_reg_total + jnp.sum(activity)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _energy_closures(self, states_prev, observation, control_input, frozen=None):
        """(energy(s), layerwise_energy(s)) closed over one trajectory, with
        the weight regularisation computed once and frozen, and the per-settle
        `frozen` context (if any) closed over so it is constant while the
        states move."""
        weight_reg_total = jnp.sum(self._weight_reg_by_node())
        highway_matrices = self._highway_matrices()
        energy = lambda s: self.energy_fn(
            states_prev, s, observation, control_input, weight_reg_total=weight_reg_total, frozen=frozen, highway_matrices=highway_matrices)
        layerwise = lambda s: self.energy_fn(
            states_prev, s, observation, control_input, return_layerwise=True, frozen=frozen)
        return energy, layerwise

    def make_activity_step(
        self,
        activity_optim: optax.GradientTransformation,
        states_prev: States,
        observation: Mapping[str, Array],
        control_input: Optional[Array] = None,
        frozen: Optional[Mapping[str, object]] = None,
    ):
        """A scan body running ONE inference update. carry:
        `(states_curr, activity_opt_state)`; output: `states_curr`."""
        energy, _ = self._energy_closures(states_prev, observation, control_input, frozen)

        def activity_step(carry, _):
            states_curr, opt_state = carry
            grads = jax.grad(energy)(states_curr)
            updates, opt_state = activity_optim.update(grads, opt_state, states_curr)
            states_curr = optax.apply_updates(states_curr, updates)
            return (states_curr, opt_state), states_curr

        return activity_step

    def settle_scan(
        self,
        activity_optim: optax.GradientTransformation,
        states_prev: States,
        observation: Mapping[str, Array],
        control_input: Optional[Array] = None,
        n_steps: int = 20,
        return_layerwise: bool = False,
        init_direction: Optional[str] = None,
        states_init: Optional[States] = None,
    ):
        """Feedforward init, then `n_steps` of optax-driven relaxation as one
        `jax.lax.scan`. With `return_layerwise`, also returns the
        `(n_steps, n_nodes)` layerwise energy trace.

        If any node's energy needs a frozen context (Meta-PCN), it is built
        here ONCE from the initialisation states (the linearisation point
        $z^{(0)}$) and held constant for all `n_steps`. `states_init` lets a
        caller that already ran the init sweep pass it in to avoid repeating it."""
        states_curr0 = states_init if states_init is not None else self.init_activities(
            states_prev, control_input, observation, init_direction)
        frozen = self.frozen_context(states_prev, states_curr0, observation, control_input)
        opt_state0 = activity_optim.init(states_curr0)
        step = self.make_activity_step(activity_optim, states_prev, observation, control_input, frozen)
        (states_curr, _), states_hist = jax.lax.scan(step, (states_curr0, opt_state0), xs=None, length=n_steps)
        if not return_layerwise:
            return states_curr
        _, layerwise = self._energy_closures(states_prev, observation, control_input, frozen)
        return states_curr, jax.vmap(layerwise)(states_hist)

    def make_vector_field(self, states_prev: States, observation: Mapping[str, Array], control_input: Optional[Array] = None,
                          frozen: Optional[Mapping[str, object]] = None):
        """The ODE vector field `ds/dt = -dE/ds` for one trajectory."""
        energy, _ = self._energy_closures(states_prev, observation, control_input, frozen)

        def vector_field(t, states_curr, args):
            return jax.tree_util.tree_map(jnp.negative, jax.grad(energy)(states_curr))

        return vector_field

    def make_steady_state_event(
        self,
        states_prev: States,
        observation: Mapping[str, Array],
        control_input: Optional[Array] = None,
        tol: Optional[float] = 1e-3,
        criterion: str = "rms",
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        frozen: Optional[Mapping[str, object]] = None,
    ) -> diffrax.Event:
        """Adapter over `ode_settling.make_steady_state_event`."""
        vector_field = self.make_vector_field(states_prev, observation, control_input, frozen)
        energy, _ = self._energy_closures(states_prev, observation, control_input, frozen)
        return ode_settling.make_steady_state_event(
            vector_field, tol=tol, criterion=criterion, rtol=rtol, atol=atol, energy_fn=energy)

    def settle_diffrax(
        self,
        states_prev: States,
        observation: Mapping[str, Array],
        control_input: Optional[Array] = None,
        max_t1: float = 20.0,
        dt0: Optional[float] = None,
        n_save: int = 20,
        solver: Optional[diffrax.AbstractSolver] = None,
        stepsize_controller: Optional[diffrax.AbstractStepSizeController] = None,
        steady_state_tol: Optional[float] = 1e-3,
        steady_state_criterion: str = "rms",
        steady_state_rtol: Optional[float] = None,
        steady_state_atol: Optional[float] = None,
        return_layerwise: bool = False,
        init_direction: Optional[str] = None,
    ) -> Union[States, Tuple[States, Array, Array]]:
        """Adapter over `ode_settling.settle_diffrax` (see its docstring for
        modes and returns)."""
        states_init = self.init_activities(states_prev, control_input, observation, init_direction)
        frozen = self.frozen_context(states_prev, states_init, observation, control_input)   # Meta-PCN etc.; {} otherwise
        energy, layerwise = self._energy_closures(states_prev, observation, control_input, frozen)
        return ode_settling.settle_diffrax(
            self.make_vector_field(states_prev, observation, control_input, frozen),
            states_init,
            energy_fn=energy, layerwise_energy_fn=layerwise,
            max_t1=max_t1, dt0=dt0, n_save=n_save, solver=solver, stepsize_controller=stepsize_controller,
            steady_state_tol=steady_state_tol, steady_state_criterion=steady_state_criterion,
            steady_state_rtol=steady_state_rtol, steady_state_atol=steady_state_atol,
            return_layerwise=return_layerwise,
        )

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def param_grad(
        self,
        states_prev: States,
        states_curr: States,
        observation: Mapping[str, Array],
        control_input: Optional[Array] = None,
        states_init: Optional[States] = None,
    ) -> PyTree:
        """dE/d(weights) at the settled states. `eqx.filter_grad` ignores
        the static, non-array node metadata.

        `states_init`: the initialisation states $z^{(0)}$ that settling started
        from. Only needed when some node's energy has a frozen context
        (Meta-PCN): its learning rule is plain PC against the frozen forward
        pass, $\\tfrac12\\lVert z^{(T)} - f(z^{(0)};\\theta)\\rVert^2$ (paper Eq. 27),
        so the context is rebuilt INSIDE the differentiated function, in
        learning mode, letting the weight gradient flow through the frozen
        prediction $c(\\theta)$."""
        def energy_of_weights(m):
            frozen = None
            if states_init is not None:
                frozen = m.frozen_context(states_prev, states_init, observation, control_input, learning=True)
            return m.energy_fn(states_prev, states_curr, observation, control_input, frozen=frozen)
        return eqx.filter_grad(energy_of_weights)(self)

    def update_params(self, grads: PyTree, optim: optax.GradientTransformation, opt_state: optax.OptState):
        """Optax-driven weight update; `opt_state` is carried by the caller."""
        updates, opt_state = optim.update(grads, opt_state, self)
        return eqx.apply_updates(self, updates).postprocess_params(), opt_state

    def postprocess_params(self) -> "GraphModel":
        """Hook run after every optimiser update (see `ModelBase`).

        Here: Meta-PCN's variance-based weight normalisation for every node
        whose resolved `weight_norm` is "variance" (see `weight_norm.py`):
        each weight matrix is divided by $(\\sqrt m + \\sqrt n)\\sigma_w$ so its
        spectral norm stays near 1. A no-op (returns `self`) when nothing opts in.
        """
        return normalise_weights(self)

    # ------------------------------------------------------------------ #
    # ModelBase hooks (saving / loading / labelling)
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config: GraphModelConfig, *, key) -> "GraphModel":
        return cls(config, key=key)

    @classmethod
    def zero_activities(cls, config: GraphModelConfig) -> States:
        """Activities skeleton: zeros for every inferred (unclamped) node."""
        return {s.name: jnp.zeros(s.shape) for s in config.unclamped}

    @classmethod
    def layer_labels(cls, config: GraphModelConfig) -> List[str]:
        """One label per node, in the order of `energy_fn(..., return_layerwise=True)`."""
        return list(config.names)
