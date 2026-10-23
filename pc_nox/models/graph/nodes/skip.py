"""graph/nodes/skip.py: the learnable, fixed random, or identity map from a skip source onto a node's state shape."""
from math import prod
from typing import Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from .norm import Gain

SKIP_MODES = ("identity", "gain", "linear", "pad", "random")


def _spatial_strides(skip_shape, own_shape):
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


def make_skip_op(skip_shape: Tuple[int, ...], own_shape: Tuple[int, ...], key: PRNGKeyArray, mode: str = "auto", seed: int = 0):
    """None (identity) if the shapes match, else a learned map onto `own_shape`."""
    skip_shape, own_shape = tuple(skip_shape), tuple(own_shape)
    same = skip_shape == own_shape
    if mode == "auto":
        mode = "identity" if same else "linear"
    if mode not in SKIP_MODES:
        raise ValueError(f"skip mode {mode!r} not in {SKIP_MODES}")
    if mode in ("identity", "gain") and not same:
        raise ValueError(f"skip {skip_shape} -> {own_shape}: {mode!r} cannot change shape; use 'linear', 'pad' or 'random'")
    if not same and mode in ("linear", "pad", "random") and len(skip_shape) == 3 == len(own_shape):
        _spatial_strides(skip_shape, own_shape)                   # fail at build time, not trace time
    if mode == "identity": return None
    if mode == "gain":     return Gain(own_shape)
    if mode == "pad":      return PadTruncateSkip(skip_shape, own_shape)
    if mode == "random":   return RandomProjectionSkip(skip_shape, own_shape, seed)

    # handle linear / conv case (mode = "linear")
    if len(skip_shape) == 1 or len(own_shape) == 1:           # any dense side: flatten, Linear, reshape in apply
        return eqx.nn.Linear(prod(skip_shape), prod(own_shape), use_bias=False, key=key)
    (c_in, h_in, w_in), (c_out, h_out, w_out) = skip_shape, own_shape
    if h_in % h_out or w_in % w_out or h_in // h_out != w_in // w_out:
        raise ValueError(f"skip {skip_shape} -> {own_shape}: spatial size must shrink by an integer factor")
    return eqx.nn.Conv2d(c_in, c_out, kernel_size=1, stride=h_in // h_out, use_bias=False, key=key)  # 1x1 projection shortcut


def apply_skip(op: Optional[eqx.Module], x: Array, own_shape: Tuple[int, ...]) -> Array:
    if op is None:
        return x
    if isinstance(op, eqx.nn.Linear):
        return jnp.reshape(op(jnp.reshape(x, (-1,))), own_shape)
    return op(x)