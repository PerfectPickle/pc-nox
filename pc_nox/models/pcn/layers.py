"""pcn/layers.py

The one layer role a static PCN needs: a plain feedforward prediction map
from a parent layer's current state to this layer's current state,
`z_hat_l = f(W_l @ z_{l-1} + b_l)` -- eq. (Algorithm 1, lines 6/8 of the
Meta-PCN paper). No recurrence, no "previous timestep": every quantity is
"now", since there's no time axis at all.
"""

from typing import Callable, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class PcnLayer(eqx.Module):
    """One layer of a static hierarchical PCN.

        W ("W_l" in the algorithm box): applied to the parent layer's
            current state
        b ("b_l"): bias (kept, unlike tPC-H's layers, to match the
            algorithm box's W_l z_l + b_l exactly)

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
        return self.W_parrent_curr.weight
