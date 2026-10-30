"""utils/node_specs.py

Helpers that build `NodeSpec` tuples for `GraphModel`, so large networks do not
have to be written out node by node.

DESIGN. Four small layers, each usable on its own, each taking and returning
plain tuples of `NodeSpec` (specs are frozen, so nothing is ever mutated):

  1. `layer_widths`      -> a tuple of ints (the layer sizes).
  2. `make_chain`        -> a linear run of nodes of ONE kind, optionally attached
                            below an existing node (`parent=`). This is the unit
                            for hand-building "a big sequence of hidden layers",
                            and it works equally for a whole network or a piece.
  3. transforms          -> `replace_specs`, `add_highways`, `add_skips`: take
                            specs (and the names of a chain) and return new specs.
                            Orthogonal features are separate functions rather than
                            flags on one giant function.
  4. presets             -> `make_tpch_specs`, `make_pcn_classifier_specs` (whole
                            network types, written purely in terms of 1-3) and
                            `make_config` / `make_*_config`, which also set the
                            MODEL-WIDE settings that a NodeSpec cannot carry
                            (`input_size`, defaults, ...).

There is no "whole network vs partial" mode: a chain with `parent=None` is a root
chain, a chain with `parent="x"` hangs below node "x", and the transforms take
explicit names, so the same functions cover both uses.

CONVENTIONS
  * Names must be valid Python identifiers (NodeSpec enforces this), so layers are
    named "h0", "h1", ... and never "0", "1".
  * `None` for an act_fn / energy / bidirectional means INHERIT the model-wide
    default (the NodeSpec convention), not "off".
  * A chain of one node is named exactly `prefix` ("core", "obs"), not "core0".
  * Pass freshly built specs to the transforms, not `config.node_specs`: the
    latter have their levels resolved and pinned, which stops them re-deriving.

SKIPS (see `add_skips`). A skip block runs from a SOURCE node to a CONSUMER node
`span` hops further down the chain.
  top-down skip     the consumer's td prediction gets "+ skip(source)" outside its
                    activation. Listed in the consumer's `skip_parents`.
  bottom-up skip    the source's bu prediction gets "+ skip(consumer)" outside its
                    activation. Listed in the source's `skip_children`.
Each pathway gets its OWN aux chain: a td aux node is a delayed copy of the source
(predicted only from its skip parent), a bu aux node a delayed copy of the consumer
(predicted only from its skip child). They are never one shared bidirectional node:
a node predicted from both sides settles on the AVERAGE of its neighbours, so a
shared chain would interpolate between source and consumer instead of copying either.

REQUIREMENTS ON THE LIBRARY (these helpers only set spec fields):
  * NodeSpec fields `highway_from`, `skip_parents`, `skip_children`.
  * config: `skip_parents` counts as a parent edge for level resolution, unknown-name
    checks and `children()`. `X.skip_children` containing `Y` counts as X being a
    LEVEL-parent of Y, so bottom-up aux chains get aligned levels automatically.
  * model: skip edges NEVER feed `parents_curr` / `children_curr` (a td aux node is a
    copy of its source; as a regular child it would let the source's bottom-up path
    predict itself from that copy). They are gathered into `skip_parents_curr` /
    `skip_children_curr`.
  * an "aux" node kind with option pathway="td" (default; reads skip_parents_curr, or
    parents_curr when aux_wiring="parents") or pathway="bu" (bidirectional=True, no td
    prediction, reads skip_children_curr). `validate_pairings` must accept such a
    bu-only inferred node (it currently requires "td"); `init_activities` already
    starts a node with no prediction in the sweep direction at zero.
"""

import math
import warnings
from dataclasses import replace
from typing import Callable, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

from pc_nox.models.graph.config import GraphModelConfig, NodeSpec

Shape = Union[int, Tuple[int, ...]]
Selector = Union[str, Iterable[str], Callable[[NodeSpec], bool]]

WIDTH_PROFILES = ("linear", "hourglass", "barrel")
SKIP_VIA = ("aux", "direct")
AUX_WIRING = ("skip", "parents")
_RESERVED = ("name", "kind", "shape", "parents")


# --------------------------------------------------------------------------- #
# 1. Layer widths
# --------------------------------------------------------------------------- #

def layer_widths(first: int, last: int, num_layers: int, profile: str = "linear", min_width: int = 1) -> Tuple[int, ...]:
    """Layer sizes from `first` (the root side, no parents) to `last` (the leaf side).

    (Named first/last rather than highest/lowest: in this library a node's level
    GROWS towards the leaves, so "highest level" is ambiguous.)

    Args:
        first, last: Sizes of the two end layers (kept exactly).
        num_layers: Number of layers INCLUDING both ends.
        profile: How the interior is shaped.
            "linear"    straight-line interpolation from `first` to `last`.
            "hourglass" the linear value, shrunk smoothly to half in the middle.
            "barrel"    the linear value, grown smoothly to double in the middle.
        min_width: Floor applied to INTERIOR layers (end layers are never changed).
    """
    if profile not in WIDTH_PROFILES:
        raise ValueError(f"profile must be one of {WIDTH_PROFILES}, got {profile!r}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if first < 1 or last < 1 or min_width < 1:
        raise ValueError("first, last and min_width must be >= 1")
    if num_layers == 1:
        return (first,)
    widths: List[int] = []
    for i in range(num_layers):
        t = i / (num_layers - 1)                       # 0 at `first`, 1 at `last`
        w = first + (last - first) * t
        bump = math.sin(math.pi * t)                   # 0 at both ends, 1 in the middle
        if profile == "hourglass":
            w *= 1.0 - 0.5 * bump
        elif profile == "barrel":
            w *= 1.0 + bump
        widths.append(max(min_width, int(round(w))))
    widths[0], widths[-1] = first, last
    return tuple(widths)


# --------------------------------------------------------------------------- #
# 2. Chains
# --------------------------------------------------------------------------- #

class Chain(NamedTuple):
    """A run of connected nodes. `specs` is a tuple of NodeSpec; the properties
    are the main-path names, which the transforms below take as `names`."""
    specs: Tuple[NodeSpec, ...]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    @property
    def first(self) -> str:
        return self.specs[0].name

    @property
    def last(self) -> str:
        return self.specs[-1].name


def make_chain(kind: str, widths: Sequence[Shape], prefix: str = "h", parent: Optional[str] = None, **spec_kwargs) -> Chain:
    """A linear chain of nodes of one `kind`: node i has node i-1 as its only parent.

    Args:
        kind: A registered node kind ("pcn", "pcn_conv", "tpch_hidden", ...).
        widths: One shape per node (an int, or a tuple such as (C, H, W)).
        prefix: Name stem: "h" gives h0, h1, ... (a single node is just "h").
        parent: Name of an existing node to attach the first node below
            (None = the first node is a root).
        **spec_kwargs: Any other NodeSpec field, applied to EVERY node
            (clamped=, energy=, td_act_fn=, bidirectional=, alpha=, options=, ...).
            An unknown field raises the dataclass's own TypeError.
    """
    widths = list(widths)
    if not widths:
        raise ValueError("widths must not be empty")
    if not isinstance(prefix, str) or not f"{prefix}0".isidentifier():
        raise ValueError(f"prefix {prefix!r} must make valid identifiers (e.g. 'h' -> 'h0')")
    bad = [k for k in _RESERVED if k in spec_kwargs]
    if bad:
        raise TypeError(f"{bad} are set by make_chain; to choose the first node's parent use parent=")
    names = [prefix] if len(widths) == 1 else [f"{prefix}{i}" for i in range(len(widths))]
    specs = []
    for i, (name, w) in enumerate(zip(names, widths)):
        above = parent if i == 0 else names[i - 1]
        specs.append(NodeSpec(name=name, kind=kind, shape=w, parents=(above,) if above else (), **spec_kwargs))
    return Chain(tuple(specs))


# --------------------------------------------------------------------------- #
# 3. Transforms
# --------------------------------------------------------------------------- #

def _select(specs: Sequence[NodeSpec], where: Selector) -> frozenset:
    if callable(where):
        return frozenset(s.name for s in specs if where(s))
    wanted = frozenset([where] if isinstance(where, str) else where)
    unknown = wanted - {s.name for s in specs}
    if unknown:
        raise KeyError(f"no spec(s) named {sorted(unknown)}; have {[s.name for s in specs]}")
    return wanted


def replace_specs(specs: Sequence[NodeSpec], where: Selector, **changes) -> Tuple[NodeSpec, ...]:
    """Apply `changes` to the selected nodes (a name, names, or a predicate on NodeSpec).

    `options=` is MERGED into each node's existing options rather than replacing
    them. This is the escape hatch for anything the helpers do not cover:
        specs = replace_specs(specs, "h3", energy="meta")
        specs = replace_specs(specs, lambda s: s.kind == "pcn", weight_norm="variance")
    """
    chosen = _select(specs, where)
    new_opts = changes.pop("options", None)
    out = []
    for s in specs:
        if s.name not in chosen:
            out.append(s)
            continue
        kw = dict(changes)
        if new_opts is not None:
            kw["options"] = {**s.opts, **dict(new_opts)}
        out.append(replace(s, **kw))                    # re-runs NodeSpec validation
    return tuple(out)


def _split_source(src: str) -> Tuple[str, str]:
    name, _, path = src.partition(":")
    path = path or "td"
    if not name or path not in ("td", "bu"):
        raise ValueError(f"highway source {src!r} must look like 'name' or 'name:td' / 'name:bu'")
    return name, path


def add_highways(
    specs: Sequence[NodeSpec],
    names: Sequence[str],
    sources: Union[str, Sequence[str]],
    every: int = 1,
    offset: int = 0,
) -> Tuple[NodeSpec, ...]:
    """Make every `every`-th node of `names` (starting at index `offset`) a highway
    checkpoint (HEP) fed by `sources`.

    Args:
        names: The candidate nodes IN ORDER, normally `chain.names`. Counting
            `every` over this list (not over all specs) is what keeps aux nodes
            and clamped nodes from being counted.
        sources: Clamped node(s) whose error is broadcast to each checkpoint.
            "label" means its top-down error; "img:bu" its bottom-up error.
            May name a node that is not in `specs` (a partial network); the
            config validates it when the whole graph is assembled.
        every, offset: Checkpoint density, as in the paper's Table 2.
    """
    sources = (sources,) if isinstance(sources, str) else tuple(sources)
    if not sources:
        raise ValueError("sources must not be empty")
    if every < 1 or offset < 0:
        raise ValueError("every must be >= 1 and offset >= 0")
    parsed = [_split_source(s) for s in sources]
    by_name = {s.name: s for s in specs}
    chosen = list(names)[offset::every]
    if not chosen:
        raise ValueError(f"no checkpoint selected from {len(list(names))} names with every={every}, offset={offset}")
    for n in chosen:
        if n not in by_name:
            raise KeyError(f"checkpoint {n!r} is not in specs")
        if by_name[n].clamped and not by_name[n].release:
            raise ValueError(f"{n!r} is clamped: only inferred nodes can be highway checkpoints")
    for src, _ in parsed:
        if src in chosen:
            raise ValueError(f"{src!r} cannot be a checkpoint of its own highway")
        if src in by_name and not by_name[src].clamped:
            raise ValueError(f"highway source {src!r} must be a clamped (observed) node")
    out = []
    for s in specs:
        if s.name in chosen:
            merged = tuple(dict.fromkeys(s.highway_from + sources))       # order-preserving de-duplication
            s = replace(s, highway_from=merged)
        out.append(s)
    return tuple(out)


def _fresh(cur: dict, name: str) -> None:
    if name in cur:
        raise ValueError(f"node name {name!r} is already in use")


def add_skips(
    specs: Sequence[NodeSpec],
    names: Sequence[str],
    span: int = 2,
    every: Optional[int] = None,
    via: Optional[str] = "aux",
    bu_via: Optional[str] = None,
    mode: str = "auto",
    bu_mode: str = "auto",
    aux_kind: str = "aux",
    aux_wiring: str = "skip",
    match_precision: bool = True,
    start: int = 0,
) -> Tuple[NodeSpec, ...]:
    """Add residual skip blocks along the main path `names` (outside-activation skips).

    A block starts at `names[i]` (the SOURCE) and ends at `names[i + span]` (the
    CONSUMER), bypassing `span - 1` layers. Blocks start at i = start, start+every, ...
    (`every` defaults to `span`: back-to-back ResNet-style blocks; smaller values
    overlap them). `names` must be ONE connected path: for a network with parallel
    branches, call this once per branch.

    TOP-DOWN skip (`via`; None disables it):
        "aux"    `span - 1` identity nodes, one per bypassed layer:
                     source -> aux1 -> ... -> aux_{span-1} -> consumer
                 Each has the SOURCE's width and runs alongside the main node it
                 parallels, so the skip signal and the main signal reach the
                 consumer in the same number of steps. The consumer lists the last
                 one in `skip_parents`. Names: f"{consumer}_aux{k}".
        "direct" the consumer lists the source itself in `skip_parents`. The signal
                 arrives early (the failure the aux scheme fixes): a baseline.
    BOTTOM-UP skip (`bu_via`; None disables it): the mirror image.
        "aux"    `span - 1` nodes with the CONSUMER's width, each predicted only
                 from its `skip_children` entry: source <- bu_aux1 <- ... <- consumer.
                 Names: f"{source}_bu_aux{m}". The source lists the first in
                 `skip_children` (and is made bidirectional).
        "direct" the source lists the consumer itself in `skip_children`.

    Args:
        mode, bu_mode: The receiving node's skip-op mode ("auto" = node default;
            otherwise written to options as {"skip_td": mode} on the consumer /
            {"skip_bu": bu_mode} on the source). "identity" / "gain" need source
            and consumer to have the same shape.
        aux_kind: The registered node kind used for aux nodes.
        aux_wiring: How td aux nodes are linked. "skip" (recommended) uses
            `skip_parents`, so the aux copy never appears in its source's
            `children_curr`. "parents" uses ordinary `parents`, which needs no
            library change but, in a bidirectional graph, feeds the source a copy of
            itself through its bottom-up path.
        match_precision: Give each aux node the td/bu precision of the main-path
            node it runs alongside, i.e. `names[i + k]` for the k-th aux node of a
            block (the paper sets aux covariance equal to the main layer's). This
            is by POSITION on the path you pass, not by level: with several nodes on
            one level, pass each branch separately.
        start: Index in `names` of the first block's source.

    Returns the specs with consumers/sources modified and the aux nodes appended.
    """
    specs = tuple(specs)
    chain = list(names)
    every = span if every is None else every
    for label, v in (("via", via), ("bu_via", bu_via)):
        if v is not None and v not in SKIP_VIA:
            raise ValueError(f"{label} must be None or one of {SKIP_VIA}, got {v!r}")
    if via is None and bu_via is None:
        raise ValueError("nothing to add: via and bu_via are both None")
    if aux_wiring not in AUX_WIRING:
        raise ValueError(f"aux_wiring must be one of {AUX_WIRING}, got {aux_wiring!r}")
    if span < 2:
        raise ValueError(f"span must be >= 2 (a skip over zero layers is meaningless), got {span}")
    if every < 1 or start < 0:
        raise ValueError("every must be >= 1 and start >= 0")
    cur = {s.name: s for s in specs}
    if len(cur) != len(specs):
        raise ValueError("duplicate node names in specs")
    missing = [n for n in chain if n not in cur]
    if missing:
        raise KeyError(f"names not in specs: {missing}")
    for above, below in zip(chain, chain[1:]):
        if above not in cur[below].parents:
            raise ValueError(f"`names` must be a connected main path, but {above!r} is not a parent of {below!r}")
    blocks = [(i, i + span) for i in range(start, len(chain) - span, every)]
    if not blocks:
        raise ValueError(f"no skip block of span {span} fits in a path of {len(chain)} nodes (start={start})")

    new_aux: List[NodeSpec] = []
    for i, j in blocks:
        src, con = cur[chain[i]], cur[chain[j]]
        same_shape = tuple(src.shape) == tuple(con.shape)
        for active, m, label in ((via, mode, "mode"), (bu_via, bu_mode, "bu_mode")):
            if active and m in ("identity", "gain") and not same_shape:
                raise ValueError(f"skip {src.name!r}{src.shape} <-> {con.name!r}{con.shape}: {label}={m!r} cannot change "
                                 "shape; use 'auto', 'linear', 'pad' or 'random'")

        def precision(k: int) -> dict:                           # of the main-path node k hops below the source
            if not match_precision:
                return {}
            alongside = cur[chain[i + k]]
            return dict(td_precision=alongside.td_precision, bu_precision=alongside.bu_precision)

        # ---- top-down skip: source -> consumer
        if via is not None:
            if via == "direct":
                skip_src = src.name
            else:
                prev = src.name
                for k in range(1, span):
                    name = f"{con.name}_aux{k}"
                    _fresh(cur, name)
                    link = {"skip_parents": (prev,)} if aux_wiring == "skip" else {"parents": (prev,)}
                    # bidirectional=False EXPLICITLY: left as None it would inherit a model-wide True and the node
                    # would settle on the average of its neighbours instead of copying its parent.
                    aux = NodeSpec(name=name, kind=aux_kind, shape=src.shape, bidirectional=False, **link, **precision(k))
                    cur[name] = aux
                    new_aux.append(aux)
                    prev = name
                skip_src = prev
            changes = {"skip_parents": con.skip_parents + (skip_src,)}
            if mode != "auto":
                changes["options"] = {**con.opts, "skip_td": mode}
            cur[con.name] = replace(con, **changes)

        # ---- bottom-up skip: consumer -> source
        if bu_via is not None:
            if cur[src.name].bidirectional is False:
                raise ValueError(f"{src.name!r} has bidirectional=False but a bottom-up skip needs its bottom-up path")
            if bu_via == "direct":
                skip_child = con.name
            else:
                bu_names = [f"{src.name}_bu_aux{m}" for m in range(1, span)]
                for m, name in enumerate(bu_names, start=1):
                    _fresh(cur, name)
                    nxt = bu_names[m] if m < span - 1 else con.name       # predicted from the next one down, the last from the consumer
                    aux = NodeSpec(name=name, kind=aux_kind, shape=con.shape, bidirectional=True,
                                   skip_children=(nxt,), options={"pathway": "bu"}, **precision(m))
                    cur[name] = aux
                    new_aux.append(aux)
                skip_child = bu_names[0]
            s_now = cur[src.name]
            changes = {"skip_children": s_now.skip_children + (skip_child,), "bidirectional": True}
            if bu_mode != "auto":
                changes["options"] = {**s_now.opts, "skip_bu": bu_mode}
            cur[src.name] = replace(s_now, **changes)
    return tuple(cur[s.name] for s in specs) + tuple(new_aux)


def check_skip_sync(config: GraphModelConfig) -> List[str]:
    """Report skips that are not time-synchronised, as a list of messages (empty = fine).

    The condition is LEVEL equality: a skip's far end must sit at the same level as
    the near end's ordinary neighbours, so the skip and main signals take the same
    number of hops.
      top-down:  each of a consumer's `skip_parents` vs its `parents`.
      bottom-up: each of a node's `skip_children` vs its ordinary children.
    Aux chains built by `add_skips` satisfy it by construction. `via="direct"` skips
    do not, by design, and will be reported. Aux nodes themselves are not checked.
    """
    problems = []
    regular_children = {s.name: [x.name for x in config.node_specs if s.name in x.parents] for s in config.node_specs}
    for s in config.node_specs:
        if s.parents:
            main = sorted({config.spec(p).level for p in s.parents})
            for src in s.skip_parents:
                level = config.spec(src).level
                if main != [level]:
                    problems.append(f"node {s.name!r}: top-down skip source {src!r} is at level {level} but its parents are "
                                    f"at {main}; the two signals will not arrive together")
        if regular_children[s.name]:
            main = sorted({config.spec(c).level for c in regular_children[s.name]})
            for src in s.skip_children:
                level = config.spec(src).level
                if main != [level]:
                    problems.append(f"node {s.name!r}: bottom-up skip source {src!r} is at level {level} but its children are "
                                    f"at {main}; the two signals will not arrive together")
    return problems


def make_config(specs: Sequence[NodeSpec], *, warn_unsynced: bool = True, **config_kwargs) -> GraphModelConfig:
    """`GraphModelConfig(node_specs=specs, **config_kwargs)`, plus a warning (not an error)
    for any unsynchronised skip, since `via="direct"` skips are deliberate baselines.

    This is where MODEL-WIDE settings go: `input_size`, `bidirectional`, `td_act_fn`,
    `init_direction`, ... A `NodeSpec` describes one node and has no field for them.
    """
    config = GraphModelConfig(node_specs=tuple(specs), **config_kwargs)
    if warn_unsynced:
        problems = check_skip_sync(config)
        if problems:
            warnings.warn("unsynchronised skips:\n  - " + "\n  - ".join(problems), stacklevel=2)
    return config


# --------------------------------------------------------------------------- #
# 4. Presets
# --------------------------------------------------------------------------- #

def _apply_features(specs, names, highway_source, highway_every, highway_offset, skip_span, skip_kwargs):
    if skip_span:
        specs = add_skips(specs, names, span=skip_span, **skip_kwargs)
    if highway_every:
        specs = add_highways(specs, names, sources=highway_source, every=highway_every, offset=highway_offset)
    return specs


def make_tpch_specs(
    obs_dim: int,
    num_hidden: int,
    min_width: int = 4,
    width_profile: str = "linear",
    td_act_fn: Optional[str] = None,
    bu_act_fn: Optional[str] = None,
    bidirectional: Optional[bool] = None,
    alpha: Optional[float] = None,
    obs_energy: Optional[str] = None,
    highway_every: Optional[int] = None,
    highway_offset: int = 0,
    skip_span: Optional[int] = None,
    skip_every: Optional[int] = None,
    skip_via: Optional[str] = "aux",
    bu_skip_via: Optional[str] = None,
    skip_mode: str = "auto",
    bu_skip_mode: str = "auto",
    aux_wiring: str = "skip",
) -> Tuple[NodeSpec, ...]:
    """One-tail tPC-H network: core (control) -> h0 ... h{n-1} -> obs (clamped).

    Widths run from `min_width` at the core to `obs_dim` at the observation node
    (see `layer_widths`). Names: "core", "h0".., "obs". Highways, if enabled,
    broadcast the obs node's error to every `highway_every`-th hidden node. Skips
    (`skip_span` set) run over the hidden chain; the tPC-H node classes must
    implement them (see `add_skips`). `bidirectional` is applied to core and hidden
    nodes only: the observation readout has no bottom-up path.

    This returns specs only. The control input's width, `input_size`, belongs to the
    model-wide config: use `make_tpch_config` to set it.
    """
    if num_hidden < 1:
        raise ValueError("num_hidden must be >= 1")
    widths = layer_widths(min_width, obs_dim, num_hidden + 2, width_profile, min_width=min_width)
    shared = dict(td_act_fn=td_act_fn, bu_act_fn=bu_act_fn, bidirectional=bidirectional, alpha=alpha)
    core = make_chain("tpch_control", widths[:1], "core", **shared)
    hidden = make_chain("tpch_hidden", widths[1:-1], "h", parent=core.last, **shared)
    obs = make_chain("tpch_obs", widths[-1:], "obs", parent=hidden.last, clamped=True, energy=obs_energy)
    skip_kwargs = dict(every=skip_every, via=skip_via, bu_via=bu_skip_via, mode=skip_mode, bu_mode=bu_skip_mode, aux_wiring=aux_wiring)
    return _apply_features(core.specs + hidden.specs + obs.specs, hidden.names, obs.last,
                           highway_every, highway_offset, skip_span, skip_kwargs)


def make_pcn_classifier_specs(
    in_dim: int,
    num_classes: int,
    hidden_widths: Sequence[int],
    td_act_fn: Optional[str] = None,
    bu_act_fn: Optional[str] = None,
    bidirectional: Optional[bool] = None,
    highway_every: Optional[int] = None,
    highway_offset: int = 0,
    skip_span: Optional[int] = None,
    skip_every: Optional[int] = None,
    skip_via: Optional[str] = "aux",
    bu_skip_via: Optional[str] = None,
    skip_mode: str = "auto",
    bu_skip_mode: str = "auto",
    aux_wiring: str = "skip",
) -> Tuple[NodeSpec, ...]:
    """Supervised static-PCN classifier: x (clamped data) -> h0 ... -> y (clamped label, CE).

    The setting of the HEP and aux-neuron papers. Names: "x", "h0".., "y" (plus
    aux nodes, see `add_skips`). `skip_span=None` disables skips, `highway_every=None`
    disables highways. Skips run over the hidden chain only. The output node uses
    td_act_fn="identity" (a linear readout feeding the CE loss); make sure
    "identity" is in your ACT_FN_REGISTRY.
    """
    hidden_widths = list(hidden_widths)
    if not hidden_widths:
        raise ValueError("hidden_widths must not be empty")
    inp = make_chain("input", [in_dim], "x", clamped=True)
    hidden = make_chain("pcn", hidden_widths, "h", parent=inp.last,
                        td_act_fn=td_act_fn, bu_act_fn=bu_act_fn, bidirectional=bidirectional)
    out = make_chain("pcn", [num_classes], "y", parent=hidden.last, clamped=True, energy="ce", td_act_fn="identity")
    skip_kwargs = dict(every=skip_every, via=skip_via, bu_via=bu_skip_via, mode=skip_mode, bu_mode=bu_skip_mode, aux_wiring=aux_wiring)
    return _apply_features(inp.specs + hidden.specs + out.specs, hidden.names, out.last,
                           highway_every, highway_offset, skip_span, skip_kwargs)


def make_tpch_config(*, input_size: int = 0, config_kwargs: Optional[dict] = None, warn_unsynced: bool = True,
                     **spec_kwargs) -> GraphModelConfig:
    """`make_tpch_specs(**spec_kwargs)` wrapped in a GraphModelConfig with `input_size` set.

    `input_size` is the width of the external control input x_t that drives the core
    node (0 = none). Other model-wide settings go in `config_kwargs`
    (e.g. config_kwargs={"bidirectional": True, "init_direction": "td"}).
    Spec arguments must be passed by keyword: make_tpch_config(obs_dim=128, num_hidden=8, input_size=3).
    """
    return make_config(make_tpch_specs(**spec_kwargs), warn_unsynced=warn_unsynced,
                       input_size=input_size, **(config_kwargs or {}))


def make_pcn_classifier_config(*, config_kwargs: Optional[dict] = None, warn_unsynced: bool = True,
                               **spec_kwargs) -> GraphModelConfig:
    """`make_pcn_classifier_specs(**spec_kwargs)` wrapped in a GraphModelConfig."""
    return make_config(make_pcn_classifier_specs(**spec_kwargs), warn_unsynced=warn_unsynced, **(config_kwargs or {}))
