"""tpch/layers.py

The three layer roles that compose a `TpchModel` -- see `model.py`'s
module docstring for the shapes/equations. Split out on their own so a
sibling variant that reuses one or two of these unchanged (e.g. a variant
that only changes the observation layer) can import just those, and so
`model.py` isn't 250 lines longer than it needs to be.
"""

from typing import Callable, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class TpchControlLayer(eqx.Module):
    """Top layer of a tPC-H network ("s" in the paper).

    This is the only layer driven by the raw external input x_t. It has two
    sets of weights:

        W_rec  ("A" in the paper): applied to its own previous state s_{t-1}
        W_in   ("B" in the paper): applied to the current control input x_t

    Prediction (eq. 19's first term, generalised):
        s_hat_t = f(W_rec @ s_{t-1} + W_in @ x_t)
    """

    W_rec: eqx.nn.Linear  # recurrent weight (from self at t-1)
    W_in: eqx.nn.Linear = None  # optional control input

    # static=True excludes this variable from the pytree, i.e. it's python metadata, not a leaf, and will be ignored during JAX autodifferentiation
    has_input: bool = eqx.field(static=True)
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        state_size: int,  # nodes / width dimension
        input_size: Optional[int] = 0,
        act_fn: Callable = jnp.tanh,
        *,
        key: PRNGKeyArray,
    ):
        key_rec, key_in = jr.split(key)
        # use_bias=False to match the paper exactly (eq. 19 has no bias terms)
        self.W_rec = eqx.nn.Linear(state_size, state_size, use_bias=False, key=key_rec)
        self.act_fn = act_fn
        self.has_input = (input_size > 0)
        if self.has_input:  # create input 'control' weights if there is any input dim
            self.W_in = eqx.nn.Linear(input_size, state_size, use_bias=False, key=key_in)

    def predict(self, state_prev: Array, control_input: Optional[Array] = None) -> Array:
        """s_hat_t = f(A @ s_{t-1} + B @ x_t)"""
        pre_acts = self.W_rec(state_prev)
        if self.has_input and control_input is not None:
            pre_acts = pre_acts + self.W_in(control_input)
        return self.act_fn(pre_acts)


class TpchHiddenLayer(eqx.Module):
    """Middle layer of a tPC-H network ("z" in the paper). Any number of
    these can be stacked between the control layer and the observation
    layer.

        W_rec ("P"): applied to its own previous state, z_{t-1}
        W_parent_prev ("Q"): applied to its parent's previous state, s_{t-1}
        W_parent_curr ("R"): applied to its parent's CURRENT state, s_t

    Prediction (eq. 19's second term, generalised):
        z_hat_t = f(W_rec @ z_{t-1} + W_parent_prev @ s_{t-1} + W_parent_curr @ s_t)

    The W_parent_curr / "R" pathway is what lets information flow down the
    hierarchy within a single time step, instead of only across time steps.
    """

    W_rec: eqx.nn.Linear
    W_parent_prev: eqx.nn.Linear
    W_parent_curr: eqx.nn.Linear
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        state_size: int,
        parent_size: int,
        act_fn: Callable = jnp.tanh,
        *,
        key: PRNGKeyArray,
    ):
        key_rec, key_parent_prev, key_parent_curr = jr.split(key, 3)
        self.W_rec = eqx.nn.Linear(state_size, state_size, use_bias=False, key=key_rec)
        self.W_parent_prev = eqx.nn.Linear(parent_size, state_size, use_bias=False, key=key_parent_prev)
        self.W_parent_curr = eqx.nn.Linear(parent_size, state_size, use_bias=False, key=key_parent_curr)
        self.act_fn = act_fn

    def predict(self, state_prev: Array, parent_prev: Array, parent_curr: Array) -> Array:
        """z_hat_t = f(P @ z_{t-1} + Q @ s_{t-1} + R @ s_t)"""
        return self.act_fn(
            self.W_rec(state_prev)
            + self.W_parent_prev(parent_prev)
            + self.W_parent_curr(parent_curr)
        )


class TpchObservationLayer(eqx.Module):
    """Bottom layer of a tPC-H network ("y" in the paper).

    Unlike the other two layer types above, the observation layer has no
    memory of its own and no nonlinearity -- it is a pure linear emission
    of whatever the lowest hidden layer's CURRENT state is. One set of
    weights:

        W_parent ("C"): applied to the parent layer's current state, z_t

    Prediction (eq. 19's third term):
        y_hat_t = C @ z_t
    """

    W_parent: eqx.nn.Linear

    def __init__(self, obs_size: int, parent_size: int, *, key: PRNGKeyArray):
        self.W_parent = eqx.nn.Linear(parent_size, obs_size, use_bias=False, key=key)

    def predict(self, parent_curr: Array) -> Array:
        """y_hat_t = C @ z_t  (no activation function -- pure linear readout)"""
        return self.W_parent(parent_curr)
