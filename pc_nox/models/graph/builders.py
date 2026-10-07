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
# nodes/tpch.py is the ONLY file that imports the tPC-H layers from the tpch package
from .nodes.tpch import (
    TpchControlLayer, TpchControlNode, TpchHiddenLayer, TpchHiddenNode, TpchObservationLayer, TpchObservationNode,
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


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class BuildContext(NamedTuple):
    """Everything a node builder is given.

    Attributes:
        spec: The node's `NodeSpec`.
        parent_shape: Combined parent shape (per the kind's layout), or
            None if the node has no parents.
        child_shape: Combined child shape, or None unless the spec is
            `bidirectional`.
        act_fn: Resolved activation callable (spec override or model default).
        input_size: The config's control-input width (0 = none).
        key: A PRNG key unique to this node (see `node_key`).
    """
    spec: NodeSpec
    parent_shape: Optional[Tuple[int, ...]]
    child_shape: Optional[Tuple[int, ...]]
    act_fn: Callable
    input_size: int
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
    return TpchControlNode(TpchControlLayer(
        state_size=s.shape[0], input_size=ctx.input_size, act_fn=ctx.act_fn, alpha=s.alpha, key=ctx.key,
    ))


@register_node("tpch_hidden", input_layout="flat")
def _build_tpch_hidden(ctx: BuildContext) -> NodeBase:
    s = ctx.spec
    _no_options(s)
    _require(bool(s.parents), s, "a hidden node needs at least one parent")
    _require(len(s.shape) == 1, s, f"shape must be 1-D, got {s.shape}")
    return TpchHiddenNode(TpchHiddenLayer(
        state_size=s.shape[0], parent_size=ctx.parent_shape[0], act_fn=ctx.act_fn, alpha=s.alpha, key=ctx.key,
    ))


@register_node("tpch_obs", input_layout="flat")
def _build_tpch_obs(ctx: BuildContext) -> NodeBase:
    s = ctx.spec
    _no_options(s)
    _require(bool(s.parents), s, "an observation node needs at least one parent")
    _require(len(s.shape) == 1, s, f"shape must be 1-D, got {s.shape}")
    _require(s.alpha is None, s, "an observation node has no memory, so alpha must be None")
    # loss is chosen by the spec's `energy` ("pc" = squared error, "ce" = cross-entropy), not here
    return TpchObservationNode(TpchObservationLayer(
        obs_size=s.shape[0], parent_size=ctx.parent_shape[0], key=ctx.key,
    ))


@register_node("input", input_layout="flat")
def _build_input(ctx: BuildContext) -> NodeBase:
    s = ctx.spec
    _no_options(s)
    _require(not s.parents, s, "an input node has no parents")
    _require(s.clamped, s, "an input node is data, so it must be clamped=True")
    _require(s.alpha is None and not s.bidirectional, s, "an input node has no memory and no bottom-up path")
    return InputNode(s.shape)


@register_node("pcn", input_layout="flat")
def _build_pcn(ctx: BuildContext) -> NodeBase:
    s = ctx.spec
    _require(s.alpha is None, s, "a static PCN node has no memory, so alpha must be None")
    return PcnNode.from_spec(s, ctx.parent_shape, ctx.child_shape, ctx.act_fn, key=ctx.key)


@register_node("pcn_conv", input_layout="spatial")
def _build_pcn_conv(ctx: BuildContext) -> NodeBase:
    s = ctx.spec
    _require(bool(s.parents), s, "a conv node needs at least one parent")
    _require(s.alpha is None, s, "conv nodes are not leaky (no eligibility traces), alpha must be None")
    return PcnConvNode.from_spec(s, ctx.parent_shape, ctx.child_shape, ctx.act_fn, key=ctx.key)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def node_key(key: PRNGKeyArray, name: str) -> PRNGKeyArray:
    """A key unique to the node NAME (not its position), so adding or
    removing other nodes never changes this node's initial weights.
    (crc32, not `hash`, which is salted per Python process.)"""
    return jr.fold_in(key, zlib.crc32(name.encode()))


def input_shapes(config: GraphModelConfig, spec: NodeSpec) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """(combined parent shape, combined child shape) for a node, per its
    kind's layout. The child shape is only computed for bidirectional nodes."""
    layout = get_builder(spec.kind).input_layout
    parent_shape = (
        combined_shape([config.spec(p).shape for p in spec.parents], layout) if spec.parents else None
    )
    kids = config.children()[spec.name]
    child_shape = (
        combined_shape([config.spec(c).shape for c in kids], layout) if (spec.bidirectional and kids) else None
    )
    return parent_shape, child_shape


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
            parent_shape, child_shape = input_shapes(config, spec)
            ctx = BuildContext(
                spec=spec, parent_shape=parent_shape, child_shape=child_shape,
                act_fn=resolve_act_fn(config.get(spec, "act_fn")),
                input_size=config.input_size, key=node_key(key, spec.name),
            )
            nodes[spec.name] = builder.build(ctx)
        except ValueError as e:
            msg = str(e)
            if f"node {spec.name!r}" not in msg:
                msg = f"node {spec.name!r} (kind {spec.kind!r}): {msg}"
            raise ValueError(msg) from e
    return nodes
