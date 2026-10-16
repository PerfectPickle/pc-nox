"""graph/nodes/conv.py

`PcnConvNode`: a static (non-recurrent) convolutional node with an
independent top-down projection from its parents and an optional
bottom-up projection from its children (td/bu are graph directions, see
`NodeBase.predict`), after
bidirectional PC (Oliviers et al., 2026).

Conventions (see `NodeBase` / `NodeInputs`):

* State shape is `(C, H, W)` and stays shaped -- no flattening at node
  boundaries. A node operates on ONE example; batching is `vmap`'d outside.
* `inp.parents_curr` / `inp.children_curr` arrive already combined by the
  model: a single parent/child passes through unchanged; several are
  channel-concatenated for a conv consumer (so `parent_shape` below is the
  COMBINED shape, which the builder computes from the specs).
* DIRECTION decides the op, not a preference. A node's top-down projection
  maps parent -> own, so:
    - parent spatially SMALLER than own (generative decoder, e.g. latent ->
      image): `ConvTranspose2d`, which upsamples;
    - parent spatially LARGER than (or equal to) own (supervised
      classifier, e.g. image -> feature map): strided `Conv2d`, which
      downsamples. A transposed conv CANNOT downsample, so it is simply the
      wrong tool there, not a weaker one.
  The bottom-up projection (child -> own) is the mirror image. The default
  `"auto"` picks from the shapes; `td_mode` / `bu_mode` ("conv" |
  "transpose") override it. The shape check below catches any mismatch.
* A dense (1-D) parent is supported: a learned `Linear` projects it to
  `prod(own_shape)` and the result is reshaped to `own_shape`. This is the
  usual "dense core -> conv decoder tail" boundary. Likewise a dense child
  for the bottom-up path.
* Both directions are checked at construction: the op's actual output
  shape must equal `own_shape`, otherwise a `ValueError` names the
  mismatch (kernel/stride/padding/output_padding are easy to get wrong).
* Not traceable (`supports_trace` stays False): the eligibility-trace
  update assumes a dense `outer(error, input)` weight gradient.
* For a clamped image node, set `act_fn="identity"` on its `NodeSpec` so
  the readout is linear, as in the dense observation layer.
"""

from math import prod
from typing import Callable, List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from .base import NodeBase, NodeInputs, NodePrediction, WeightEntry

# Keys a NodeSpec.options dict may carry for this node (anything else is
# rejected by `from_spec`, so typos fail loudly instead of being ignored).
CONV_OPTION_KEYS = frozenset({
    "td_kernel_size", "td_stride", "td_padding", "td_output_padding", "td_mode",
    "bu_kernel_size", "bu_stride", "bu_padding", "bu_output_padding", "bu_mode",
})


_MODES = ("auto", "conv", "transpose")


def _resolve_mode(mode: str, src_shape: Tuple[int, ...], own_shape: Tuple[int, ...], which: str) -> str:
    """"transpose" if the op must UPSAMPLE (own is spatially larger than its
    source), else "conv". Explicit modes pass through."""
    if mode not in _MODES:
        raise ValueError(f"{which}_mode must be one of {_MODES}, got {mode!r}")
    if mode != "auto":
        return mode
    return "transpose" if own_shape[1] > src_shape[1] else "conv"


def _check_conv_output(op, in_shape: Tuple[int, ...], expected: Tuple[int, ...], which: str) -> None:
    """Abstractly evaluate `op` on `in_shape` (no compute) and require the
    output shape to equal `expected`."""
    out = eqx.filter_eval_shape(op, jax.ShapeDtypeStruct(in_shape, jnp.float32)).shape
    if tuple(out) != tuple(expected):
        raise ValueError(
            f"PcnConvNode {which} op maps input shape {tuple(in_shape)} to {tuple(out)}, "
            f"which does not match own_shape {tuple(expected)}; adjust kernel_size / stride / "
            "padding / output_padding, or set the op's mode explicitly (mode='conv' downsamples, "
            "mode='transpose' upsamples)"
        )


class PcnConvNode(NodeBase):
    """Static conv node with top-down ("td") and optional bottom-up ("bu")
    projections.

    td = top-down: this node predicted from its parents
    bu = bottom-up: this node predicted from its children (only if `bidirectional`)

    Fields:
        op_td: `Linear` (dense parent), or for a conv parent `ConvTranspose2d`
            (upsampling) / `Conv2d` (downsampling), chosen by direction.
        op_bu: `Linear` (dense child), or for a conv child `Conv2d`
            (child larger) / `ConvTranspose2d` (child smaller); None unless
            `bidirectional`.
        own_shape, is_td_dense, is_bu_dense, bidirectional, td_act_fn,
        bu_act_fn: static metadata (hashable, resolved at trace time).
    """

    op_td: Optional[Union[eqx.nn.Linear, eqx.nn.ConvTranspose2d, eqx.nn.Conv2d]]
    op_bu: Optional[Union[eqx.nn.Conv2d, eqx.nn.ConvTranspose2d, eqx.nn.Linear]]

    own_shape: Tuple[int, int, int] = eqx.field(static=True)
    is_td_dense: bool = eqx.field(static=True)
    is_bu_dense: bool = eqx.field(static=True)
    bidirectional: bool = eqx.field(static=True)
    td_act_fn: Callable = eqx.field(static=True)
    bu_act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        parent_shape: Tuple[int, ...],
        own_shape: Tuple[int, int, int],
        child_shape: Optional[Tuple[int, ...]] = None,
        # --- Parent (top-down / ConvTranspose) config ---
        td_kernel_size: int = 3,
        td_stride: int = 2,
        td_padding: Union[str, int] = 1,
        td_output_padding: Union[str, int] = 0,    # transpose mode only
        td_mode: str = "auto",                     # "auto" | "conv" | "transpose"
        # --- Child (bottom-up) config ---
        bu_kernel_size: int = 3,
        bu_stride: int = 2,
        bu_padding: Union[str, int] = 1,
        bu_output_padding: Union[str, int] = 0,    # transpose mode only
        bu_mode: str = "auto",
        td_act_fn: Callable = jnp.tanh,
        bu_act_fn: Callable = jnp.tanh,
        bidirectional: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        parent_shape = None if parent_shape is None else tuple(parent_shape)
        own_shape = tuple(own_shape)
        if len(own_shape) != 3:
            raise ValueError(f"own_shape must be (C, H, W), got {own_shape}")
        if parent_shape is None and not bidirectional:
            raise ValueError("a node with no parent is predicted only from its children: it needs bidirectional=True")
        if parent_shape is not None and len(parent_shape) not in (1, 3):
            raise ValueError(f"parent_shape must be (D,) or (C, H, W), got {parent_shape}")
        if bidirectional:
            if child_shape is None:
                raise ValueError("bidirectional=True requires child_shape")
            child_shape = tuple(child_shape)
            if len(child_shape) not in (1, 3):
                raise ValueError(f"child_shape must be (D,) or (C, H, W), got {child_shape}")

        k_td, k_bu = jr.split(key)
        self.own_shape = own_shape
        self.td_act_fn = td_act_fn
        self.bu_act_fn = bu_act_fn
        self.bidirectional = bidirectional

        # 1. PARENT PROJECTION (top-down)
        self.is_td_dense = parent_shape is not None and len(parent_shape) == 1
        if parent_shape is None:                       # a parentless data root: bottom-up path only
            self.op_td = None
        elif self.is_td_dense:
            self.op_td = eqx.nn.Linear(parent_shape[0], prod(own_shape), use_bias=True, key=k_td)
        else:
            if _resolve_mode(td_mode, parent_shape, own_shape, "td") == "transpose":
                self.op_td = eqx.nn.ConvTranspose2d(
                    in_channels=parent_shape[0],
                    out_channels=own_shape[0],
                    kernel_size=td_kernel_size,
                    stride=td_stride,
                    padding=td_padding,
                    output_padding=td_output_padding,
                    key=k_td,
                )
            else:
                if td_output_padding != 0:
                    raise ValueError("td_output_padding only applies to a transposed (upsampling) top-down op")
                self.op_td = eqx.nn.Conv2d(
                    in_channels=parent_shape[0],
                    out_channels=own_shape[0],
                    kernel_size=td_kernel_size,
                    stride=td_stride,
                    padding=td_padding,
                    key=k_td,
                )
            _check_conv_output(self.op_td, parent_shape, own_shape, "top-down")

        # 2. CHILD PROJECTION (bottom-up)
        self.is_bu_dense = False
        self.op_bu = None
        if bidirectional:
            self.is_bu_dense = len(child_shape) == 1
            if self.is_bu_dense:
                self.op_bu = eqx.nn.Linear(child_shape[0], prod(own_shape), use_bias=True, key=k_bu)
            else:
                if _resolve_mode(bu_mode, child_shape, own_shape, "bu") == "transpose":
                    self.op_bu = eqx.nn.ConvTranspose2d(
                        in_channels=child_shape[0],
                        out_channels=own_shape[0],
                        kernel_size=bu_kernel_size,
                        stride=bu_stride,
                        padding=bu_padding,
                        output_padding=bu_output_padding,
                        key=k_bu,
                    )
                else:
                    if bu_output_padding != 0:
                        raise ValueError("bu_output_padding only applies to a transposed (upsampling) bottom-up op")
                    self.op_bu = eqx.nn.Conv2d(
                        in_channels=child_shape[0],
                        out_channels=own_shape[0],
                        kernel_size=bu_kernel_size,
                        stride=bu_stride,
                        padding=bu_padding,
                        key=k_bu,
                    )
                _check_conv_output(self.op_bu, child_shape, own_shape, "bottom-up")

    # ------------------------------------------------------------------ #
    # Construction from a NodeSpec (called by the builder registry)
    # ------------------------------------------------------------------ #

    @classmethod
    def from_spec(
        cls,
        spec,                                   # NodeSpec
        parent_shape: Tuple[int, ...],
        child_shape: Optional[Tuple[int, ...]],
        td_act_fn: Callable,
        bu_act_fn: Callable,
        *,
        key: PRNGKeyArray,
    ) -> "PcnConvNode":
        """Build from a `NodeSpec`. `parent_shape`/`child_shape` are the
        COMBINED shapes the builder derived from the spec's parents and the
        config's children map; `act_fn` is the resolved activation (used for
        both directions). `spec.options` may carry any of `CONV_OPTION_KEYS`.
        """
        opts = spec.opts
        unknown = set(opts) - CONV_OPTION_KEYS
        if unknown:
            raise ValueError(
                f"node {spec.name!r}: unknown PcnConvNode options {sorted(unknown)}; "
                f"allowed: {sorted(CONV_OPTION_KEYS)}"
            )
        return cls(
            parent_shape=parent_shape,
            own_shape=spec.shape,
            child_shape=child_shape if bool(spec.bidirectional) else None,
            td_act_fn=td_act_fn,
            bu_act_fn=bu_act_fn,
            bidirectional=bool(spec.bidirectional),
            key=key,
            **opts,
        )

    # ------------------------------------------------------------------ #
    # NodeBase interface
    # ------------------------------------------------------------------ #

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return self.own_shape

    @property
    def prediction_keys(self) -> Tuple[str, ...]:
        keys = () if self.op_td is None else ("td",)
        return keys + ("bu",) if self.bidirectional else keys

    @property
    def needs_parents_curr(self) -> bool:
        return self.op_td is not None

    @property
    def needs_children(self) -> bool:
        return self.bidirectional

    @property
    def supports_bottom_up(self) -> bool:
        return True

    @property
    def provides_pre_activations(self) -> bool:
        return True

    def _project(self, op, x: Array, is_dense: bool) -> Array:
        """Apply `op` to `x`; a dense op's flat output is reshaped to
        `own_shape`."""
        pre = op(x)
        return jnp.reshape(pre, self.own_shape) if is_dense else pre

    def predict(self, inp: NodeInputs) -> NodePrediction:
        """Top-down prediction from `inp.parents_curr`, plus a bottom-up
        prediction from `inp.children_curr` if `bidirectional`. Each path
        also reports its pre-activation (for energies that need it)."""
        self.check_inputs(inp)

        predictions, pre_activations = {}, {}
        if self.op_td is not None:
            pre_td = self._project(self.op_td, inp.parents_curr, self.is_td_dense)
            predictions["td"] = self.td_act_fn(pre_td)
            pre_activations["td"] = pre_td

        if self.bidirectional:
            pre_bu = self._project(self.op_bu, inp.children_curr, self.is_bu_dense)
            predictions["bu"] = self.bu_act_fn(pre_bu)
            pre_activations["bu"] = pre_bu

        return NodePrediction(predictions, pre_activations)

    def weight_entries(self) -> List[WeightEntry]:
        """Each projection's weight as a 2-D `(out, -1)` view, tagged "ff"
        (nothing here is recurrent), so `penalties.py` works unchanged. The
        views are reshapes of the live weights, so penalty gradients reach
        the real parameters. Biases are not included.
        """
        ops = ([("op_td", self.op_td)] if self.op_td is not None else []) + ([("op_bu", self.op_bu)] if self.op_bu is not None else [])
        return [
            WeightEntry(f"{name}.weight", "ff", jnp.reshape(op.weight, (op.weight.shape[0], -1)))
            for name, op in ops
        ]
