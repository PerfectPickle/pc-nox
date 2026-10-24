"""graph/nodes/skip.py

The receiver-side map for an OUTER skip connection: how a skip source's state is carried onto the
receiving node's own state shape before being added to its prediction.

    pred = act_fn(W @ x + b) + apply_skip(op, skip_state, own_shape)
                                 ^ added AFTER the activation, so the skip bypasses it

The skip source is typically an auxiliary node: a weightless copy of an earlier state that delays the
error signal so it stays in step with the main path. Everything else about the skip (the copy, the
delay) lives in that node. This file only decides how the RECEIVER maps the copy onto its own shape,
which is the identity when the shapes match and something that changes shape when they differ.

This is the "outside the activation" form. The alternative, "inside", needs no code here: list the
source in the node's `parents`, and its state is concatenated into `parents_curr` and projected by the
node's ordinary `op_td` (see `builders.combine`).

Modes (`SKIP_MODES`), selected with `make_skip_op(mode=...)`:

    mode       parameters           changes shape?   notes
    ---------  -------------------  ---------------  ------------------------------------------------
    identity   none                 no               the default when shapes match; keeps the error path clean
    gain       learnable, per unit  no               starts at 1, so it begins as the identity
    linear     learnable            yes              Linear (dense side) or strided 1x1 Conv2d (conv to conv)
    pad        none                 yes              subsample H, W; zero-pad or truncate channels
    random     none (frozen)        yes              random projection regenerated from a seed each call

"pad" and "random" hold no arrays, so the optimiser, weight decay and checkpoints never touch them.
The random matrix is regenerated from a static integer seed on every call (a constant under jit)
instead of being stored, for that reason.

Shapes are `(D,)` or `(C, H, W)`. Spatial size may only SHRINK, by an integer factor per axis (the H and W
factors may differ): the skip source sits earlier in the network, where resolution is higher.
"""
from math import prod
from typing import Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from .norm import Gain

SKIP_MODES = ("identity", "gain", "linear", "pad", "random")


def _spatial_strides(skip_shape, own_shape):
    """Integer (height, width) strides that take a skip source's spatial grid down to this node's.

    Example: a skip of shape (64, 32, 32) feeding a node of shape (128, 16, 16) gives (2, 2); keeping every
    second row and column of the skip lands exactly on the node's 16 x 16 grid. The strides are
    returned separately because the two axes may shrink by different factors.

    Both arguments must be `(C, H, W)`; the channel counts are ignored here (channels are handled by
    the mode). Called once, at build time, so a bad shape fails with a clear message instead of
    inside a traced function.

    Args:
        skip_shape: `(C, H, W)` of the skip source.
        own_shape: `(C, H, W)` of the receiving node.

    Returns:
        `(stride_h, stride_w)`. `(1, 1)` means the spatial sizes already match.

    Raises:
        ValueError: if either spatial size of the skip is not an integer multiple of the node's, which
            includes any attempt to upsample (the node's grid larger than the skip's).
    """
    (_, h_in, w_in), (_, h_out, w_out) = skip_shape, own_shape
    if h_in % h_out or w_in % w_out:
        raise ValueError(f"skip {skip_shape} -> {own_shape}: H, W must shrink by integer factors (no upsampling)")
    return h_in // h_out, w_in // w_out


class PadTruncateSkip(eqx.Module):
    """No parameters. Conv: subsample H,W by integer strides, zero-pad or truncate channels (ResNet 'option A').
    Dense or mixed: flatten, zero-pad or truncate, reshape."""
    in_shape: Tuple[int, ...] = eqx.field(static=True)
    own_shape: Tuple[int, ...] = eqx.field(static=True)

    def __call__(self, x):
        if len(self.in_shape) == 3 and len(self.own_shape) == 3:
            sh, sw = _spatial_strides(self.in_shape, self.own_shape)
            x = x[:, ::sh, ::sw]
            c_in, c_out = self.in_shape[0], self.own_shape[0]
            return jnp.pad(x, ((0, c_out - c_in), (0, 0), (0, 0))) if c_out > c_in else x[:c_out]
        flat, n = jnp.reshape(x, (-1,)), prod(self.own_shape)
        flat = jnp.pad(flat, (0, n - flat.size)) if n > flat.size else flat[:n]
        return jnp.reshape(flat, self.own_shape)


class RandomProjectionSkip(eqx.Module):
    """Frozen random projection, never stored: regenerated from a static seed (so no leaf, no decay).
    Entries ~ N(0, 1/fan_in), which preserves RMS in expectation. Conv->conv uses a per-pixel
    channel mix (a frozen 1x1 conv) after subsampling; a full dense matrix over (C,H,W) would be enormous."""
    in_shape: Tuple[int, ...] = eqx.field(static=True)
    own_shape: Tuple[int, ...] = eqx.field(static=True)
    seed: int = eqx.field(static=True)

    def __call__(self, x):
        k = jr.PRNGKey(self.seed)
        if len(self.in_shape) == 3 and len(self.own_shape) == 3:
            sh, sw = _spatial_strides(self.in_shape, self.own_shape)
            w = jr.normal(k, (self.own_shape[0], self.in_shape[0])) / jnp.sqrt(self.in_shape[0])
            return jnp.einsum("oc,chw->ohw", w, x[:, ::sh, ::sw])
        n_in, n_out = prod(self.in_shape), prod(self.own_shape)
        w = jr.normal(k, (n_out, n_in)) / jnp.sqrt(n_in)
        return jnp.reshape(w @ jnp.reshape(x, (-1,)), self.own_shape)


def make_skip_op(
    skip_shape: Tuple[int, ...],
    own_shape: Tuple[int, ...],
    key: PRNGKeyArray,
    mode: str = "auto",
    seed: int = 0,
) -> Optional[eqx.Module]:
    """Build the receiver-side map that carries a skip source's state onto this node's state shape.

    The skip source (e.g. an auxiliary node) holds a plain copy of an earlier state. This op is what
    the RECEIVING node applies to it before adding it to its prediction (see `apply_skip`). When the
    shapes already match, no op is needed and the skip is a pure identity.

    Args:
        skip_shape: Shape of the (combined) skip source state, `(D,)` or `(C, H, W)`.
        own_shape: Shape of this node's state, the shape the skip must be mapped onto.
        key: PRNG key, used only by the "linear" mode (the other modes need no random init).
        mode: How to map the skip. One of `SKIP_MODES`:
            "auto"     -> "identity" if the shapes match, else "linear".
            "identity" -> no op (returns None). Shapes must match.
            "gain"     -> learnable per-unit (or per-channel, for `(C, H, W)`) scale, initialised at 1,
                          so the identity path is preserved at the start. Shapes must match.
            "linear"   -> learnable projection: a `Linear` (any dense side, flattened) or a 1x1
                          strided `Conv2d` (conv to conv, spatial size shrinking by an integer factor).
            "pad"      -> no parameters: subsample H, W by integer strides, then zero-pad or truncate
                          channels (the original ResNet "option A"); for dense or mixed shapes,
                          flatten and zero-pad or truncate. Truncating discards information.
            "random"   -> no learned parameters: a FROZEN random projection, regenerated from `seed`
                          on every call, so no array is stored and no optimiser or weight decay touches it.
        seed: Integer seed for "random". Should be derived deterministically (e.g. from the node name),
            not from `key`, so a reloaded checkpoint reproduces the same matrix.

    Returns:
        None for an identity skip, otherwise an `eqx.Module` callable on one example. Only
        "linear" and "gain" contain trainable arrays; "pad" and "random" have no leaves.

    Raises:
        ValueError: unknown mode; "identity" or "gain" with different shapes; or a conv-to-conv map
            whose spatial size does not shrink by an integer factor (raised here, at build time,
            rather than during tracing).
    """
    skip_shape, own_shape = tuple(skip_shape), tuple(own_shape)
    same = skip_shape == own_shape
    if mode == "auto":
        mode = "identity" if same else "linear"
    if mode not in SKIP_MODES:
        raise ValueError(f"skip mode {mode!r} not in {SKIP_MODES}")
    if mode in ("identity", "gain") and not same:
        raise ValueError(f"skip {skip_shape} -> {own_shape}: {mode!r} cannot change shape; use 'linear', 'pad' or 'random'")

    # Conv-to-conv: validate once, at build time, and keep the (sh, sw) strides for the branches below.
    # Raises unless H and W each shrink by an integer factor (no upsampling).
    conv_to_conv = len(skip_shape) == 3 == len(own_shape)
    strides = _spatial_strides(skip_shape, own_shape) if (conv_to_conv and mode in ("linear", "pad", "random")) else None

    if mode == "identity": return None
    if mode == "gain":     return Gain(own_shape)
    if mode == "pad":      return PadTruncateSkip(skip_shape, own_shape)
    if mode == "random":   return RandomProjectionSkip(skip_shape, own_shape, seed)

    # mode == "linear"
    if not conv_to_conv:                                     # any dense side: flatten, Linear, reshape in apply_skip
        return eqx.nn.Linear(prod(skip_shape), prod(own_shape), use_bias=False, key=key)
    return eqx.nn.Conv2d(skip_shape[0], own_shape[0], kernel_size=1, stride=strides, use_bias=False, key=key)


def apply_skip(op: Optional[eqx.Module], x: Array, own_shape: Tuple[int, ...]) -> Array:
    """Map a skip source state `x` onto `own_shape` using the op from `make_skip_op`.

    Called inside a node's `predict`, after the activation function, so the skip is added OUTSIDE it:
        pred = act_fn(pre) + apply_skip(self.op_skip, inp.skip_..._curr, self.state_shape)

    Args:
        op: The module from `make_skip_op`, or None for an identity skip.
        x: The skip source state for one example (already combined across skip sources by the model).
        own_shape: This node's state shape. Only needed for the `Linear` case, whose flat
            output is reshaped to it; every other op already returns `own_shape`.

    Returns:
        An array of shape `own_shape`, ready to be added to the node's prediction.
    """
    if op is None:
        return x
    if isinstance(op, eqx.nn.Linear):
        return jnp.reshape(op(jnp.reshape(x, (-1,))), own_shape)
    return op(x)