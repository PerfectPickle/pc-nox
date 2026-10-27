"""graph/builders.py

Everything that turns a `GraphModelConfig` into built node modules, kept
OUT of `GraphModel` so the model stays about data flow and so new node
families register here (or in their own module) without touching it:

  * the combine rules (`combined_shape` / `combine`): how several parent
    (or child) states become the single array a node consumes. The shape
    rule (used at build time) and the array rule (used at run time, in the
    model) are deliberately written side by side so they cannot drift.
  * the kind registry (`BUILDERS`, `register_node`): `NodeSpec.kind` ->
    how to build that node and which combine layout it wants.
  * `build_nodes`: the loop `GraphModel.__init__` calls.

Combine layouts (a property of the CONSUMING node kind):

  "flat"    flatten every input and concatenate -> 1-D. For dense nodes.
  "spatial" a single input passes through unchanged (any rank); several
            1-D inputs concatenate; several (C, H, W) inputs with equal
            (H, W) concatenate along channels. Anything else is an error.
            For conv nodes.

Adding a node kind = write the node (a `NodeBase` subclass), then register
a builder for it:

    @register_node("my_kind", input_layout="flat")
    def _build_my_kind(ctx: BuildContext) -> NodeBase: ...

Custom kinds must be registered (i.e. their module imported) before
loading a checkpoint that uses them.
"""

import zlib
from dataclasses import dataclass
from math import prod
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from pc_nox.core.activations import ACT_FN_REGISTRY

from .config import GraphModelConfig, NodeSpec
from .nodes.base import NodeBase
from .nodes.conv import PcnConvNode
from .nodes.input import InputNode
from .nodes.pcn import PcnNode
from .nodes.aux import AuxNode
# nodes/tpch.py is the ONLY file that imports the tPC-H layers from the tpch package
from .nodes.tpch import (
    TpchControlNode, TpchHiddenNode, TpchObservationNode,
)

LAYOUTS = ("flat", "spatial")


# --------------------------------------------------------------------------- #
# Combine rules
# --------------------------------------------------------------------------- #

def combined_shape(shapes: Sequence[Tuple[int, ...]], layout: str) -> Tuple[int, ...]:
    """Shape of the combined input for these parent (or child) shapes."""
    shapes = [tuple(s) for s in shapes]
    if not shapes:
        raise ValueError("cannot combine an empty list of shapes")
    if layout == "flat":
        return (sum(prod(s) for s in shapes),)
    if layout == "spatial":
        if len(shapes) == 1:
            return shapes[0]
        if all(len(s) == 1 for s in shapes):
            return (sum(s[0] for s in shapes),)
        if all(len(s) == 3 for s in shapes) and len({s[1:] for s in shapes}) == 1:
            return (sum(s[0] for s in shapes), *shapes[0][1:])
        raise ValueError(
            f"cannot combine shapes {shapes} for a spatial consumer: several inputs must be "
            "all 1-D, or all (C, H, W) with identical (H, W)"
        )
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def combine(arrays: Sequence[Array], layout: str) -> Array:
    """Array counterpart of `combined_shape` (same rules, run at trace time)."""
    if layout == "flat":
        return jnp.concatenate([jnp.reshape(a, (-1,)) for a in arrays])
    if layout == "spatial":
        if len(arrays) == 1:
            return arrays[0]
        return jnp.concatenate(list(arrays), axis=0)   # 1-D: concat; (C,H,W): channel-concat
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def combine_normed(arrays: Sequence[Array], layout: str, stat: Optional[Callable] = None,
                   mode: str = "joint") -> Array:
    """`combine`, plus an activity-norm statistic (see nodes/norm.py).

    stat=None   -> plain combine.
    "per_source" -> stat on each parent/child array BEFORE concatenation.
    "joint"      -> stat on the combined array AFTER concatenation.
    The statistic preserves shape, so `combined_shape` needs no counterpart."""
    if stat is None:
        return combine(arrays, layout)
    if mode == "per_source":
        return combine([stat(a) for a in arrays], layout)
    return stat(combine(arrays, layout))

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class BuildContext(NamedTuple):
    """Everything a node builder is given.

    Attributes:
        spec: The node's `NodeSpec`.
        parent_shape: Combined parent shape (per the kind's layout), or
            None if the node has no parents.
        skip_parent_shape: Combined parent shape (per the kind's layout), or
            None if the node has no parents.
        child_shape: Combined child shape, or None unless the spec is
            `bidirectional`.
        td_act_fn: Resolved top-down activation callable (spec override or model default).
        bu_act_fn: Resolved bottom-up activation callable (spec override or model default).
        input_size: The config's control-input width (0 = none).
        key: A PRNG key unique to this node (see `node_key`).
    """
    spec: NodeSpec
    parent_shape: Optional[Tuple[int, ...]]
    skip_parents_shape: Optional[Tuple[int, ...]]
    child_shape: Optional[Tuple[int, ...]]
    skip_children_shape: Optional[Tuple[int, ...]]
    td_act_fn: Callable
    bu_act_fn: Callable
    input_size: int
    activity_norm: str
    activity_norm_inputs: Tuple[str, ...]
    key: PRNGKeyArray


@dataclass(frozen=True)
class NodeBuilder:
    build: Callable[[BuildContext], NodeBase]
    input_layout: str = "flat"


BUILDERS: Dict[str, NodeBuilder] = {}


def register_node(kind: str, *, input_layout: str = "flat"):
    """Decorator registering a builder function for `NodeSpec.kind`."""
    if input_layout not in LAYOUTS:
        raise ValueError(f"input_layout must be one of {LAYOUTS}, got {input_layout!r}")

    def decorator(fn: Callable[[BuildContext], NodeBase]):
        if kind in BUILDERS:
            raise ValueError(f"node kind {kind!r} already registered")
        BUILDERS[kind] = NodeBuilder(build=fn, input_layout=input_layout)
        return fn

    return decorator


def get_builder(kind: str) -> NodeBuilder:
    try:
        return BUILDERS[kind]
    except KeyError:
        raise KeyError(
            f"unknown node kind {kind!r}; registered kinds: {sorted(BUILDERS)}. If this is a "
            "custom kind, import the module that registers it (see register_node) before building or loading."
        ) from None


# --------------------------------------------------------------------------- #
# Built-in kinds
# --------------------------------------------------------------------------- #

def _require(cond: bool, spec: NodeSpec, msg: str) -> None:
    if not cond:
        raise ValueError(f"node {spec.name!r} (kind {spec.kind!r}): {msg}")


def _no_options(spec: NodeSpec) -> None:
    _require(not spec.options, spec, f"takes no options, got {dict(spec.options)}")


@register_node("tpch_control", input_layout="flat")
def _build_tpch_control(ctx: BuildContext) -> NodeBase:
    s = ctx.spec
    _no_options(s)
    _require(not s.parents, s, "a control node has no parents")
    _require(len(s.shape) == 1, s, f"shape must be 1-D, got {s.shape}")
    bi = bool(s.bidirectional)
    return TpchControlNode(
        own_size=s.shape[0], 
        child_size=ctx.child_shape[0] if bi else None, 
        bidirectional=s.bidirectional, 
        input_size=ctx.input_size, 
        td_act_fn=ctx.td_act_fn, 
        bu_act_fn=ctx.bu_act_fn, 
        alpha=s.alpha, 
        activity_norm=ctx.activity_norm, activity_norm_inputs=ctx.activity_norm_inputs, key=ctx.key,
        skip_shape_td=ctx.skip_parents_shape,
        skip_shape_bu=ctx.skip_children_shape if bi else None,
    )


@register_node("tpch_hidden", input_layout="flat")
def _build_tpch_hidden(ctx: BuildContext) -> NodeBase:
    s = ctx.spec
    _no_options(s)
    _require(bool(s.parents), s, "a hidden node needs at least one parent")
    _require(len(s.shape) == 1, s, f"shape must be 1-D, got {s.shape}")
    bi = bool(s.bidirectional)
    _require(not bi or ctx.child_shape is not None, s, "bidirectional=True but the node has no children")
    return TpchHiddenNode(
        own_size=s.shape[0], 
        child_size=ctx.child_shape[0] if bi else None, 
        bidirectional=bi, 
        parent_size=ctx.parent_shape[0], 
        td_act_fn=ctx.td_act_fn, 
        bu_act_fn=ctx.bu_act_fn, 
        alpha=s.alpha, 
        activity_norm=ctx.activity_norm, activity_norm_inputs=ctx.activity_norm_inputs, key=ctx.key,
        skip_shape_td=ctx.skip_parents_shape,
        skip_shape_bu=ctx.skip_children_shape if bi else None,
    )


@register_node("tpch_obs", input_layout="flat")
def _build_tpch_obs(ctx: BuildContext) -> NodeBase:
    s = ctx.spec
    _no_options(s)
    _require(bool(s.parents), s, "an observation node needs at least one parent")
    _require(len(s.shape) == 1, s, f"shape must be 1-D, got {s.shape}")
    _require(s.alpha is None, s, "an observation node has no memory, so alpha must be None")
    # loss is chosen by the spec's `energy` ("pc" = squared error, "ce" = cross-entropy), not here
    return TpchObservationNode(
        obs_size=s.shape[0], parent_size=ctx.parent_shape[0], key=ctx.key,
        activity_norm=ctx.activity_norm, activity_norm_inputs=ctx.activity_norm_inputs, 
        skip_shape_td=ctx.skip_parents_shape, 
    )


@register_node("input", input_layout="spatial")
def _build_input(ctx: BuildContext) -> NodeBase:
    s = ctx.spec
    _require(not s.parents, s, "an input node has no parents")
    _require(s.clamped, s, "an input node is data, so it must be clamped=True")
    _require(s.alpha is None, s, "an input node has no memory, so alpha must be None")
    if not s.bidirectional:
        _no_options(s)                                   # the plain, weightless data node
        return InputNode(s.shape)
    # A bidirectional input is a data root that is PREDICTED from its children (the paper's generative term on x1):
    # a bottom-up-only conv or dense node, chosen by the declared shape.
    _require(ctx.child_shape is not None, s, "a bidirectional input node needs at least one child to predict it from")
    if len(s.shape) == 3:
        return _conv_node(ctx, None)
    _require(len(s.shape) == 1, s, f"shape must be (D,) or (C, H, W), got {s.shape}")
    return PcnNode(s.shape[0], None, ctx.child_shape[0], ctx.td_act_fn, ctx.bu_act_fn, True, False, key=ctx.key)


@register_node("pcn", input_layout="flat")
def _build_pcn(ctx: BuildContext) -> NodeBase:
    s, bi = ctx.spec, bool(ctx.spec.bidirectional)
    _require(s.alpha is None, s, "a static PCN node has no memory, so alpha must be None")
    _require(len(s.shape) == 1, s, f"shape must be 1-D, got {s.shape}")
    unknown = set(s.opts) - {"prior"}
    _require(not unknown, s, f"takes no options other than 'prior', got {sorted(unknown)}")
    return PcnNode(
        own_size=s.shape[0],
        parent_size=ctx.parent_shape[0] if ctx.parent_shape else None,
        child_size=ctx.child_shape[0] if (bi and ctx.child_shape) else None,
        td_act_fn=ctx.td_act_fn, bu_act_fn=ctx.bu_act_fn, bidirectional=bi,
        has_prior=bool(s.opts.get("prior", True)),
        activity_norm=ctx.activity_norm, activity_norm_inputs=ctx.activity_norm_inputs, 
        skip_shape_td=ctx.skip_parents_shape, skip_shape_bu=ctx.skip_children_shape,
        key=ctx.key,
    )

def _conv_node(ctx: BuildContext, parent_shape) -> PcnConvNode:
    s = ctx.spec
    #unknown = set(s.opts) - CONV_OPTION_KEYS          # keep the constant in conv.py and import it
    #_require(not unknown, s, f"unknown conv options {sorted(unknown)}; allowed: {sorted(CONV_OPTION_KEYS)}")
    bi = bool(s.bidirectional)
    return PcnConvNode(
        parent_shape=parent_shape, own_shape=s.shape,
        child_shape=ctx.child_shape if bi else None,
        td_act_fn=ctx.td_act_fn, bu_act_fn=ctx.bu_act_fn, bidirectional=bi,
        activity_norm=ctx.activity_norm, activity_norm_inputs=ctx.activity_norm_inputs, key=ctx.key,
        skip_shape_td=ctx.skip_parents_shape,
        skip_shape_bu=ctx.skip_children_shape if bi else None,
        **dict(s.opts),
    )
@register_node("pcn_conv", input_layout="spatial")
def _build_pcn_conv(ctx: BuildContext) -> NodeBase:
    s = ctx.spec
    _require(bool(s.parents) or s.bidirectional, s, "a conv node needs at least one parent (or bidirectional=True, for a bottom-up-only data root)")
    _require(s.alpha is None, s, "conv nodes are not leaky (no eligibility traces), alpha must be None")
    return _conv_node(ctx, ctx.parent_shape)


@register_node("aux", input_layout="spatial")
def _build_aux(ctx: BuildContext) -> NodeBase:
    s, shape = ctx.spec, tuple(ctx.spec.shape)
    _no_options(s)
    _require(not s.clamped, s, "an aux node is an inferred state, so it must be clamped=False")
    _require(s.alpha is None, s, "an aux node has no memory, so alpha must be None")
    _require(len(s.parents) == 1, s, f"an aux node copies exactly one parent, got {s.parents}")
    _require(ctx.parent_shape == shape, s, f"parent shape {ctx.parent_shape} must equal the aux shape {shape} (identity map)")
    if s.bidirectional:
        # more than one child would be concatenated, so the equality check below also catches that
        _require(ctx.child_shape == shape, s, f"a bidirectional aux node needs exactly one child of identical shape, got combined child shape {ctx.child_shape}")
    return AuxNode(shape=shape, bidirectional=bool(s.bidirectional))

# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def node_key(key: PRNGKeyArray, name: str) -> PRNGKeyArray:
    """A key unique to the node NAME (not its position), so adding or
    removing other nodes never changes this node's initial weights.
    (crc32, not `hash`, which is salted per Python process.)"""
    return jr.fold_in(key, zlib.crc32(name.encode()))

class InputShapes(NamedTuple):
    parent: Optional[Tuple[int, ...]]
    skip_parents: Optional[Tuple[int, ...]]
    child: Optional[Tuple[int, ...]]
    skip_children: Optional[Tuple[int, ...]]

def input_shapes(config: GraphModelConfig, spec: NodeSpec) -> InputShapes:
    """Combined shapes (per the kind's layout) of a node's parents, skip parents,
    children and skip children. The two child-side shapes only exist for
    bidirectional nodes."""
    layout = get_builder(spec.kind).input_layout

    def comb(names):                       # None if there are no such neighbours
        return combined_shape([config.spec(n).shape for n in names], layout) if names else None

    bi = bool(spec.bidirectional)
    return InputShapes(
        parent=comb(spec.parents),
        skip_parents=comb(spec.skip_parents),
        child=comb(config.children()[spec.name]) if bi else None,
        skip_children=comb(spec.skip_children) if bi else None,
    )


def resolve_act_fn(name: str) -> Callable:
    try:
        return ACT_FN_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"act_fn={name!r} not in ACT_FN_REGISTRY. If this is a custom activation, "
            f"register it before building or loading: ACT_FN_REGISTRY[{name!r}] = ..."
        ) from None


def build_nodes(config: GraphModelConfig, key: PRNGKeyArray) -> Dict[str, NodeBase]:
    """Build every node in canonical order. Errors are re-raised with the
    node's name and kind attached."""
    nodes: Dict[str, NodeBase] = {}
    for spec in config.node_specs:
        builder = get_builder(spec.kind)
        try:
            shapes = input_shapes(config, spec)
            ctx = BuildContext(
                spec=spec,
                parent_shape=shapes.parent, skip_parents_shape=shapes.skip_parents,
                child_shape=shapes.child,   skip_children_shape=shapes.skip_children,
                td_act_fn=resolve_act_fn(config.get(spec, "td_act_fn")),
                bu_act_fn=resolve_act_fn(config.get(spec, "bu_act_fn")),
                input_size=config.input_size,
                activity_norm=config.get(spec, "activity_norm"),
                activity_norm_inputs=config.activity_norm_inputs,     # model-wide only, see below
                key=node_key(key, spec.name),
            )
            nodes[spec.name] = builder.build(ctx)
        except ValueError as e:
            msg = str(e)
            if f"node {spec.name!r}" not in msg:
                msg = f"node {spec.name!r} (kind {spec.kind!r}): {msg}"
            raise ValueError(msg) from e
    return nodes
