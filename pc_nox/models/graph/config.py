"""graph/config.py

`NodeSpec` (one named node) and `GraphModelConfig` (the whole graph).
Vocabulary: a NODE is a named vertex holding a vector/tensor state (what
a conventional PC network calls a layer); `level` is its depth index, and
nodes sharing a level sit side by side. Individual neurons are called
UNITS in this package.

The config holds everything that changes what a `GraphModel` COMPUTES, and nothing else
(see `ModelBase`'s "config vs. metadata" note). No arrays, no callables, no
module instances -- only strings, numbers, bools, None, and tuples of
those -- so the config is:

  * hashable (it is a static field of the model, compared by jit),
  * JSON-serialisable (`save_checkpoint` does `json.dumps(asdict(config))`),
  * and rebuildable from its own JSON (`load_checkpoint` does
    `config_cls(**json_dict)`).

The last point is the subtle one: `asdict` turns nested dataclasses into
dicts and JSON turns tuples into lists. Both `NodeSpec` and
`GraphModelConfig` therefore COERCE their fields back in `__post_init__`
(dicts -> NodeSpec, lists -> tuples), so `config_cls(**loaded_dict)` just
works and `ModelBase.load_checkpoint` needs no changes.

What is deliberately NOT here: run state. Weights, optimiser state,
activities (and eligibility traces, for leaky nodes) are saved alongside
the config; epoch / data position / RNG key go in checkpoint `metadata`.

Evolution rule: only ever ADD fields, always with a default that
reproduces the old behaviour; never rename or repurpose one. Old
checkpoints then keep loading.

Stage-1 validation (everything that needs only the specs) runs in
`__post_init__`, so it also runs on every loaded config. Checks that need
the built nodes live in `validation.py`.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

_SCOPES = ("all", "rec", "ff")
_ACTIVITY_REG_TYPES = ("l1", "l2")
_TRACE_MODES = ("readout", "accumulate")
_WEIGHT_NORMS = ("none", "variance")     # see weight_norm.py
# valid activity normalisation constants
_NORM_INPUTS = ("own_prev", "parents_prev", "parents_curr", "children_curr")   # the NodeInputs fields a node may normalise
_NORM_MODES = ("joint", "per_source")

def _freeze(v: Any) -> Any:
    """Recursively turn lists/dicts into tuples (dicts -> sorted pairs)."""
    if isinstance(v, Mapping):
        return tuple(sorted((str(k), _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def _is_json_value(v: Any) -> bool:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return True
    return isinstance(v, tuple) and all(_is_json_value(x) for x in v)


@dataclass(frozen=True)
class NodeSpec:
    """Static description of one named node.

    Attributes:
        name: Unique identifier (must be a valid Python identifier). Used
            as the key in the node/state/observation dicts.
        kind: Key into the node-builder registry (e.g. a control, hidden,
            observation or conv node type). Resolved at build time.
        shape: State shape, e.g. `(128,)` or `(C, H, W)`. An int is
            accepted as shorthand for `(n,)`.
        parents: Names of parent nodes. ORDER MATTERS: it is the
            concatenation order, i.e. the weight-column order.
        level: Depth in the hierarchy; parents must have strictly lower
            levels. None = derived (1 + deepest parent) by the config.
        clamped: True if the state is observed data (fed in via
            `observation[name]`) rather than inferred.
        bidirectional: Whether the node also has a bottom-up path from its
            children.
        alpha: tPC-E leak rate; None = non-leaky.
        energy, td_act_fn, bu_act_fn, weight_decay, orthogonal_penalty, activity_decay:
            Per-node overrides; None = inherit the model-wide default.
        options: Node-specific constructor options (kernel size, stride,
            ...) as a hashable tuple of `(key, value)` pairs. A dict is
            accepted and canonicalised (sorted). Values must be
            JSON-compatible scalars or tuples of them.
        bidirectional: Whether the node also has a bottom-up path from its
            children. None = inherit `GraphModelConfig.bidirectional`, which
            the config resolves to an explicit bool (see there).
        td_precision, bu_precision: Per-node override of the weight on this
            node's top-down / bottom-up prediction-error terms (None =
            inherit the model-wide value). The paper's alpha_disc/alpha_gen.
        release: A CLAMPED node that is omitted from `observation` becomes an
            inferred state (initialised by the sweep, updated by settling)
            instead of being dropped. This is how a label is free at test
            time, or an image is generated from a label.
        weight_norm: Per-node override of the model-wide `weight_norm`
            ("none" | "variance"); None = inherit.
    """
    name: str
    kind: str
    shape: Tuple[int, ...]
    parents: Tuple[str, ...] = ()
    level: Optional[int] = None
    clamped: bool = False
    bidirectional: Optional[bool] = None
    energy: Optional[str] = None
    td_act_fn: Optional[str] = None
    bu_act_fn: Optional[str] = None
    options: Tuple[Tuple[str, Any], ...] = ()
    td_precision: Optional[float] = None
    bu_precision: Optional[float] = None
    release: bool = False
    weight_norm: Optional[str] = None
    highway_from: Tuple[str, ...] = ()   # clamped source nodes, who provide HEP error signal to the node.
    skip_parents: Tuple[str, ...] = ()  # parents used for 'outer' skip connections (where skip lives outside activation function, i.e. y = F(x) + x)
    skip_children: Tuple[str, ...] = () # as above, but for bottom up pathway
    
    weight_decay: Optional[float] = None
    orthogonal_penalty: Optional[float] = None
    activity_decay: Optional[float] = None
    alpha: Optional[float] = None

    def __post_init__(self):
        # --- coerce JSON-round-tripped / convenient forms to canonical ones
        shape = (self.shape,) if isinstance(self.shape, int) else self.shape
        object.__setattr__(self, "shape", tuple(shape))
        object.__setattr__(self, "parents", tuple(self.parents))
        object.__setattr__(self, "skip_parents", tuple(self.skip_parents))
        object.__setattr__(self, "highway_from", tuple(self.highway_from))

        opts = self.options
        items = opts.items() if isinstance(opts, Mapping) else [tuple(p) for p in opts]
        bad = [p for p in items if len(p) != 2]
        if bad:
            raise ValueError(f"node {self.name!r}: options must be (key, value) pairs, got {bad}")
        object.__setattr__(
            self, "options", tuple(sorted((str(k), _freeze(v)) for k, v in items))
        )

        # --- validate (collect everything, report once)
        errs: List[str] = []
        if not (isinstance(self.name, str) and self.name.isidentifier()):
            errs.append(f"name {self.name!r} must be a valid Python identifier")
        if not isinstance(self.kind, str) or not self.kind:
            errs.append("kind must be a non-empty string")
        if not self.shape or not all(isinstance(d, int) and d > 0 for d in self.shape):
            errs.append(f"shape {self.shape} must be a non-empty tuple of positive ints")
        if len(set(self.all_parents)) != len(self.all_parents):
            errs.append(f"duplicate parents {self.parents}")
        # Check for overlap between parents and skip_parents
        overlap = set(self.parents) & set(self.skip_parents)
        if overlap:
            raise ValueError(
                f"Invalid NodeSpec: 'skip_parents' overlaps with 'parents'. "
                f"Overlapping elements: {overlap}"
            )
        if self.alpha is not None and not (0.0 < self.alpha <= 1.0):
            errs.append(f"alpha must be in (0, 1], got {self.alpha}")
        if self.alpha is not None and self.clamped:
            errs.append("a clamped node cannot be leaky (alpha)")
        for f in ("weight_decay", "orthogonal_penalty", "activity_decay"):
            v = getattr(self, f)
            if v is not None and v < 0:
                errs.append(f"{f} must be >= 0, got {v}")
        for f in ("td_precision", "bu_precision"):
            v = getattr(self, f)
            if v is not None and v < 0:
                errs.append(f"{f} must be >= 0, got {v}")
        if self.weight_norm is not None and self.weight_norm not in _WEIGHT_NORMS:
            errs.append(f"weight_norm={self.weight_norm!r} not in {_WEIGHT_NORMS}")
        if self.release and not self.clamped:
            errs.append("release only applies to a clamped node (an unclamped node is always inferred)")
        for k, v in self.options:
            if not _is_json_value(v):
                errs.append(f"option {k!r}={v!r} is not a JSON-compatible scalar/tuple")
        if errs:
            raise ValueError(f"invalid NodeSpec {self.name!r}:\n  - " + "\n  - ".join(errs))

    @property
    def opts(self) -> Dict[str, Any]:
        """`options` as a plain dict, for builders."""
        return dict(self.options)

    @property
    def all_parents(self): 
        return self.parents + self.skip_parents  


def _resolve_levels(by_name: Mapping[str, NodeSpec]) -> Tuple[Dict[str, int], List[str]]:
    """Resolve every node's level; detect cycles; check strict increase.

    A node with `level=None` gets 1 + the deepest parent (0 for a root).
    An explicit level is kept, but must exceed every parent's level.
    """
    resolved: Dict[str, int] = {}
    visiting: set = set()
    errors: List[str] = []

    def level_of(name: str) -> Optional[int]:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            errors.append(f"cycle through node {name!r}")
            return None
        visiting.add(name)
        spec = by_name[name]
        parent_levels = [level_of(p) for p in spec.parents]
        visiting.discard(name)
        if any(pl is None for pl in parent_levels):
            return None
        derived = 1 + max(parent_levels) if parent_levels else 0
        resolved[name] = derived if spec.level is None else spec.level
        return resolved[name]

    for n in by_name:
        level_of(n)
    if not errors:
        for n, spec in by_name.items():
            for p in spec.all_parents:
                if resolved[p] >= resolved[n]:
                    errors.append(
                        f"node {n!r} (level {resolved[n]}) has parent {p!r} at level "
                        f"{resolved[p]}; parents must have strictly lower levels"
                    )
    return resolved, errors


@dataclass(frozen=True)
class GraphModelConfig:
    """Everything needed to rebuild a `GraphModel` (and to size its
    activities skeleton) from nothing but this object.

    Attributes:
        node_specs: The node specs. Normalised in `__post_init__` to canonical
            order -- sorted by (level, declaration order), with every level
            resolved -- so the layerwise-energy order, execution order and
            saved config are all deterministic.
        input_size: Width of the external control input (0 = none).
        td_act_fn, bu_act_fn, energy, weight_decay, orthogonal_penalty, activity_decay:
            Model-wide defaults that a `NodeSpec` may override.
        weight_decay_scope, orthogonal_scope, activity_reg_type, trace_mode:
            Model-wide settings with the same meaning as in `TpchConfig`.
            (Per-node scopes can be added later the same way as the
            coefficient overrides.)
        bidirectional: Model-wide default for `NodeSpec.bidirectional`: when
            True, every node that has children gets a bottom-up path unless it
            sets `bidirectional=False` itself. (A node kind that cannot be
            bidirectional then fails to build, naming the node.)
        td_precision, bu_precision: Model-wide weights on the top-down /
            bottom-up prediction-error terms (the paper's alpha_disc /
            alpha_gen, in whichever direction your graph makes them). Per-node
            overrides on `NodeSpec`. The paper finds the generative weight must
            be much smaller than the discriminative one (1e-3 .. 1e-8).
        init_direction: Default direction of the initialisation sweep: "td"
            (parents before children; the paper's bottom-up amortised sweep when
            the image is the root) or "bu" (children before parents; the
            generative sweep from a clamped label). Overridable per call.
        weight_norm: Model-wide weight normalisation applied after every
            parameter update: "none" (default) or "variance" (Meta-PCN's
            $W \\leftarrow W / ((\\sqrt m + \\sqrt n)\\sigma_w)$; see `weight_norm.py`).
            A `NodeSpec` may override it. Biases are never normalised.
        weight_norm_scope: Which weights `weight_norm` touches: "all" (default),
            "rec" (recurrent only) or "ff" (everything else); same tags as
            `weight_decay_scope`.
    """
    node_specs: Tuple[NodeSpec, ...]
    input_size: int = 0
    td_act_fn: str = "tanh"
    bu_act_fn: str = "tanh"
    energy: str = "pc"
    bidirectional: bool = False
    td_precision: float = 1.0
    bu_precision: float = 1.0
    init_direction: str = "td"
    weight_norm: str = "none"
    weight_norm_scope: str = "all"
    highway_alpha: float = 0.0  # strength of HEP error signal
    highway_sigma: float = 1e-3  # std of normal distribution random fixed HEP weights are initiased from
    highway_seed: int = 0
    weight_decay: float = 0.0
    weight_decay_scope: str = "all"
    orthogonal_penalty: float = 0.0
    orthogonal_scope: str = "rec"
    activity_decay: float = 0.0
    activity_reg_type: str = "l1"
    trace_mode: str = "readout"

    def __post_init__(self):
        specs = tuple(s if isinstance(s, NodeSpec) else NodeSpec(**s) for s in self.node_specs)
        errs: List[str] = []

        # --- model-wide settings
        if self.input_size < 0:
            errs.append(f"input_size must be >= 0, got {self.input_size}")
        for field, allowed in (
            ("weight_decay_scope", _SCOPES),
            ("orthogonal_scope", _SCOPES),
            ("activity_reg_type", _ACTIVITY_REG_TYPES),
            ("trace_mode", _TRACE_MODES),
            ("init_direction", ("td", "bu")),
            ("weight_norm", _WEIGHT_NORMS),
            ("weight_norm_scope", _SCOPES),
        ):
            if getattr(self, field) not in allowed:
                errs.append(f"{field}={getattr(self, field)!r} not in {allowed}")
        for field in ("weight_decay", "orthogonal_penalty", "activity_decay", "td_precision", "bu_precision"):
            if getattr(self, field) < 0:
                errs.append(f"{field} must be >= 0, got {getattr(self, field)}")

        # --- activity normalisation
        # JSON has no tuples: after save_checkpoint -> load_checkpoint this field comes back
        # as a LIST, which is unhashable and would break the config's use as a static jit
        # field. Coerce it back to a tuple first, so everything below sees the canonical form.
        # (object.__setattr__ is needed because the dataclass is frozen.)
        object.__setattr__(self, "activity_norm_inputs", tuple(self.activity_norm_inputs))

        # A typo in the scope (e.g. "parent_curr") would otherwise silently normalise
        # nothing. Check every entry against the fields a node can actually have.
        bad = [f for f in self.activity_norm_inputs if f not in _NORM_INPUTS]
        if bad:
            errs.append(f"activity_norm_inputs {bad} not in {_NORM_INPUTS}")

        # "joint": statistic over the whole concatenation of parents.
        # "per_source": statistic over each parent separately, before concatenating.
        if self.activity_norm_mode not in _NORM_MODES:
            errs.append(f"activity_norm_mode={self.activity_norm_mode!r} not in {_NORM_MODES}")
        # (Deliberately NOT checked here: whether `activity_norm` names a registered norm.
        #  The registry lives in nodes/norm.py and custom norms may be registered after
        #  the config is created, so that check happens at build time instead, like
        #  `resolve_act_fn` does for activation names.)

        # --- graph structure
        if not specs:
            errs.append("at least one node is required")
        names = [s.name for s in specs]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            errs.append(f"duplicate node names: {dupes}")
        by_name = {s.name: s for s in specs}
        for s in specs:
            for p in s.all_parents:
                if p == s.name:
                    errs.append(f"node {s.name!r} lists itself as a parent")
                elif p not in by_name:
                    errs.append(f"node {s.name!r} has unknown parent {p!r}")
                    
        if errs:  # structure is too broken to resolve levels
            raise ValueError("invalid GraphModelConfig:\n  - " + "\n  - ".join(errs))

        levels, level_errs = _resolve_levels(by_name)
        if level_errs:
            raise ValueError("invalid GraphModelConfig:\n  - " + "\n  - ".join(level_errs))

        order = sorted(range(len(specs)), key=lambda i: (levels[specs[i].name], i))
        canonical = tuple(replace(specs[i], level=levels[specs[i].name]) for i in order)
        # Resolve the model-wide `bidirectional` default structurally: a node inherits it if it HAS children, so
        # data roots get a bottom-up-only path and leaves stay top-down-only (the paper's x1 / xL asymmetry).
        n_children = {s.name: 0 for s in canonical}
        for s in canonical:
            for p in s.all_parents:
                n_children[p] += 1
        canonical = tuple(
            replace(s, bidirectional=bool(self.bidirectional and n_children[s.name] > 0)) if s.bidirectional is None else s
            for s in canonical
        )
        object.__setattr__(self, "node_specs", canonical)

        try:  # fail now, with a clear message, rather than inside jit
            hash(self)
        except TypeError as e:
            raise TypeError(f"GraphModelConfig must be hashable (static jit field): {e}") from None

    # ------------------------------------------------------------------ #
    # Derived views (computed on demand; nothing derived is stored, so
    # there is no second source of truth to drift out of sync)
    # ------------------------------------------------------------------ #

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self.node_specs)

    @property
    def unclamped(self) -> Tuple[NodeSpec, ...]:
        """Nodes whose state is inferred (the keys of `states`)."""
        return tuple(s for s in self.node_specs if not s.clamped)

    @property
    def clamped(self) -> Tuple[NodeSpec, ...]:
        """Nodes whose state is observed data (the keys of `observation`)."""
        return tuple(s for s in self.node_specs if s.clamped)

    @property
    def all_parents(self): 
        return self.parents + self.skip_parents  

    def spec(self, name: str) -> NodeSpec:
        for s in self.node_specs:
            if s.name == name:
                return s
        raise KeyError(f"no node named {name!r}; nodes: {self.names}")

    def children(self) -> Dict[str, Tuple[str, ...]]:
        """name -> names of nodes that list it as a parent (the inverse of
        `parents`), in canonical order."""
        out: Dict[str, List[str]] = {s.name: [] for s in self.node_specs}
        for s in self.node_specs:
            for p in s.all_parents:
                out[p].append(s.name)
        return {k: tuple(v) for k, v in out.items()}

    def get(self, spec: NodeSpec, field: str) -> Any:
        """A per-node setting with fallback to the model-wide default.
        Valid for: act_fn, energy, weight_decay, orthogonal_penalty,
        activity_decay, weight_norm, activity_norm.
        """
        v = getattr(spec, field)
        return getattr(self, field) if v is None else v
