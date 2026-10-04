"""pcn/layers.py

Three layer roles for a static PCN, each implementing the same minimal
interface -- `predict(parent_curr) -> own_prediction` and
`weights() -> List[Array]` (possibly empty) -- so `PcnModel` can treat a
mixed conv/flatten/dense stack uniformly (see `PcnModel._all_weights` and
`PcnModel.__init__`'s `_build_layers_from_specs`).

    PcnLayer:       plain dense feedforward map (unchanged from before).
    PcnConvLayer:   convolutional counterpart of PcnLayer -- same
                    predict-interface, W is a conv kernel instead of a
                    weight matrix, and parent/own states are feature
                    maps (C, H, W) rather than flat vectors.
    PcnFlattenLayer: stateless transition from feature maps to a flat
                    vector -- no parameters, contributes nothing to
                    `weights()`. Its output is still a free interior
                    latent like any other layer's (see `PcnModel.predict`):
                    settling can nudge it away from a pure reshape of its
                    parent, it just isn't PRODUCED by a weight matrix.
                    (Minor inefficiency, not a correctness issue -- see
                    `model.py`'s module docstring for the note on this.)

Every prediction is still `f(W @ z_parent + b)` (dense) or
`f(Conv(z_parent))` (conv) -- no recurrence, no "previous timestep",
exactly as before.
"""

from typing import Callable, List, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray
from math import prod


class PcnLayer(eqx.Module):
    """One dense layer of a static hierarchical PCN.

        W_parrent_curr ("W_l" in the algorithm box): applied to the
            parent layer's current state

    Prediction:
        z_hat_l = f(W @ z_parent + b)
    """

    W_parrent_curr: eqx.nn.Linear  # holds both W and b (use_bias=True)
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        parent_size: int,
        own_size: int,
        act_fn: Callable = jnp.tanh,
        *,
        key: PRNGKeyArray,
    ):
        self.W_parrent_curr = eqx.nn.Linear(parent_size, own_size, use_bias=True, key=key)
        self.act_fn = act_fn

    def predict(self, parent_curr: Array) -> Array:
        """z_hat = f(W @ z_parent + b)"""
        return self.act_fn(self.W_parrent_curr(parent_curr))

    @property
    def weight(self) -> Array:
        """Kept for anything that still reaches for `.weight` directly;
        `weights()` below is what `PcnModel._all_weights` actually uses."""
        return self.W_parrent_curr.weight

    def weights(self) -> List[Array]:
        return [self.W_parrent_curr.weight]



class PcnConvLayer(eqx.Module):
    """Convolutional layer with independent spatial parameters 
    for parent (top-down, generative) and optional child (bottom-up, discriminative) 
    operations following bidirectional PC (Oliviers et al., 2026)

    td = top-down, generative, from parent
    bu = bottom-up, discriminative, from child
    """

    td_op: Union[eqx.nn.Linear, eqx.nn.ConvTranspose2d]
    bu_op: Union[eqx.nn.Conv2d, eqx.nn.Linear]

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
        child_shape: Tuple[int, ...],
        # --- Parent (Top-Down / ConvTranspose) Config ---
        td_kernel_size: int = 3,
        td_stride: int = 2,
        td_padding: Union[str, int] = 1,
        td_output_padding: Union[str, int] = 0,
        # --- Child (Bottom-Up / Conv2d) Config ---
        bu_kernel_size: int = 3,
        bu_stride: int = 2,
        bu_padding: Union[str, int] = 1,
        td_act_fn: Callable = jnp.tanh,
        bu_act_fn: Callable = jnp.tanh,
        bidirectional: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jr.split(key)
        self.own_shape = own_shape
        self.td_act_fn = td_act_fn
        self.bu_act_fn = bu_act_fn

        # -------------------------------------------------------------
        # 1. PARENT PROJECTION (Top-Down Generative)
        # -------------------------------------------------------------
        self.is_td_dense = (len(parent_shape) == 1)
        if self.is_td_dense:
            self.td_op = eqx.nn.Linear(
                in_features=parent_shape[0],
                out_features=prod(own_shape),
                use_bias=True,
                key=k1,
            )
        else:
            self.td_op = eqx.nn.ConvTranspose2d(
                in_channels=parent_shape[0],
                out_channels=own_shape[0],
                kernel_size=td_kernel_size,
                stride=td_stride,
                padding=td_padding,
                output_padding=td_output_padding,
                key=k1,
            )

        # -------------------------------------------------------------
        # 2. CHILD PROJECTION (Bottom-Up Discriminative Sweep)
        # -------------------------------------------------------------
        if bidirectional:
            self.is_bu_dense = (len(child_shape) == 1)
            if self.is_bu_dense:
                self.bu_op = eqx.nn.Linear(
                    in_features=child_shape[0],
                    out_features=int(jnp.prod(jnp.array(own_shape))),
                    use_bias=True,
                    key=k2,
                )
            else:
                self.bu_op = eqx.nn.Conv2d(
                    in_channels=bu_shape[0],
                    out_channels=own_shape[0],
                    kernel_size=bu_kernel_size,
                    stride=bu_stride,
                    padding=bu_padding,
                    key=k2,
                )
        else:
            self.bu_op = None


    # top-down, generative
    def predict_td(self, parent_curr: Array) -> Array:
        pre_act = self.parent_op(parent_curr)
        if self.is_td_dense:
            pre_act = jnp.reshape(pre_act, self.own_shape)
        return self.td_act_fn(pre_act)

    # bottom-up, discriminative
    def predict_bu(self, child_curr: Array) -> Array:
        if self.bu_op is None:
            raise RuntimeError("Bottom-up prediction is disabled.")
        pre_act = self.bu_op(child_curr)
        if self.is_bu_dense:
            pre_act = jnp.reshape(pre_act, self.own_shape)
        return self.bu_act_fn(pre_act)

    # little wrapper for testing
    def predict(self, parent_curr: Array) -> Array:
        self.predict_td(parent_curr)

    def weights(self) -> Tuple[Array, ...]:
        if self.bu_op is None:
            return (self.td_op.weight,)
        return (self.td_op.weight, self.bu_op.weight)


# class PcnConvLayer(eqx.Module):
#     """Convolutional counterpart of `PcnLayer`. Parent/own states are
#     feature maps shaped (C, H, W) -- no batch axis, matching every other
#     layer here (batching is `jax.vmap` over `per_example`, see
#     `runners_static.py`).

#     Prediction:
#         z_hat_l = f(Conv(z_parent))

#     `padding` follows `equinox.nn.Conv2d`'s convention: an int, a string
#     ("SAME"/"VALID"), or an explicit per-dimension spec.
#     """

#     parent_conv: eqx.nn.Conv2d
#     act_fn: Callable = eqx.field(static=True)

#     def __init__(
#         self,
#         parent_channels: int,
#         own_channels: int,
#         kernel_size: int = 3,
#         stride: int = 1,
#         padding: Union[str, int] = "SAME",
#         act_fn: Callable = jnp.tanh,
#         *,
#         key: PRNGKeyArray,
#     ):
#         self.parent_conv = eqx.nn.Conv2d(
#             in_channels, out_channels, kernel_size, stride=stride, padding=padding, key=key
#         )
#         self.act_fn = act_fn

#     def predict(self, parent_curr: Array) -> Array:
#         """z_hat = f(Conv(z_parent)) -- parent_curr shaped (C_in, H, W),
#         result shaped (C_out, H', W')."""
#         return self.act_fn(self.parent_conv(parent_curr))

#     @property
#     def weight(self) -> Array:
#         return self.parent_conv.weight

#     def weights(self) -> List[Array]:
#         return [self.parent_conv.weight]


# class PcnFlattenLayer(eqx.Module):
#     """Stateless transition from a feature map to a flat vector. No
#     parameters: `weights()` returns `[]`, so it contributes nothing to
#     L2/orthogonal regularisation. `act_fn` is accepted-and-ignored so it
#     can sit at a normal position in a `layer_specs`/`act_fn` list without
#     special-casing the broadcast logic in `PcnModel.__init__` -- see that
#     module's docstring.
#     """

#     def __init__(self, act_fn: Optional[Callable] = None, *, key: Optional[PRNGKeyArray] = None):
#         del act_fn, key  # unused; accepted for constructor-signature uniformity with the other layer types

#     def predict(self, parent_curr: Array) -> Array:
#         return jnp.ravel(parent_curr)

#     def weights(self) -> List[Array]:
#         return []
