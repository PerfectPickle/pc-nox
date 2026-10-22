"""graph/nodes/skip.py: the learnable, fixed random, or identity map from a skip source onto a node's state shape."""
from math import prod
from typing import Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


def make_skip_op(skip_shape: Tuple[int, ...], own_shape: Tuple[int, ...], key: PRNGKeyArray):
    """None (identity) if the shapes match, else a learned map onto `own_shape`."""
    skip_shape, own_shape = tuple(skip_shape), tuple(own_shape)
    if skip_shape == own_shape:
        return None
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