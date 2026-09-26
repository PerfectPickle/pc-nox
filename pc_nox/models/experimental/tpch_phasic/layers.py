"""kpch/layers.py

The three layer roles that compose a `KPCHModel` -- phase-only analogues
of `tpch.layers`'/`slpch.layers`'s layer classes. Same top-to-bottom
roles and the same temporal-recurrence + hierarchical-prediction wiring
as both of those (own previous state, parent's previous state, parent's
CURRENT state), but the thing being predicted is now purely a PHASE:

  * a layer's actual state is a real angle `theta` (one per node) --
    `z = e^{i*theta}` is always EXACTLY unit modulus, by construction,
    for every value `theta` can take. There is no amplitude equation to
    solve, no potential to keep it bounded, because there's nothing that
    could push it off the unit circle in the first place -- contrast
    `slpch`, where staying near a particular amplitude is itself
    something the dynamics have to achieve (and can fail to, see the
    numerical-stability issues that came up there).
  * predictions are still formed the same way as `slpch` -- complex-
    linear combinations of the (exponentiated) own-previous, parent-
    previous and parent-current states via `ComplexLinear` -- but the
    prediction target `z_hat` this produces is a GENERAL complex number,
    not unit modulus. Its phase is the predicted phase; its magnitude is
    incidental (roughly "how aligned/confident the inputs being combined
    are" -- like a Kuramoto order parameter falling out of the weighted
    sum), not itself a prediction target.
  * each control/hidden layer additionally owns a per-node LEARNABLE
    coupling strength `kappa` (`> 0` via softplus) -- how strongly that
    node's actual phase is pulled toward the predicted one. This is the
    "separate weight purely for coupling strength" -- distinct from the
    prediction weights, which determine WHAT phase is predicted, not how
    hard the model tries to match it.
"""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class ComplexLinear(eqx.Module):
    """Bias-free linear map with complex weights -- identical role to
    `slpch.layers.ComplexLinear` (duplicated here rather than imported,
    so this package stays self-contained, matching how `tpch` and
    `slpch` don't depend on each other either). See that module's
    docstring for the init convention.
    """

    weight: Array

    def __init__(self, in_size: int, out_size: int, *, key: PRNGKeyArray):
        key_re, key_im = jr.split(key)
        scale = 1.0 / jnp.sqrt(in_size)
        real = jr.uniform(key_re, (out_size, in_size), minval=-scale, maxval=scale)
        imag = jr.uniform(key_im, (out_size, in_size), minval=-scale, maxval=scale)
        self.weight = (real + 1j * imag).astype(jnp.complex64)

    def __call__(self, x: Array) -> Array:
        return self.weight @ x.astype(self.weight.dtype)


def _init_kappa_raw(kappa_init: float, shape) -> Array:
    """Unconstrained per-node param s.t. softplus(kappa_raw) == kappa_init
    exactly at init -- same inverse-softplus convention as `slpch`'s
    `gamma_raw` (see that module for why: guarantees kappa > 0 for any
    finite kappa_raw, including after gradient updates).
    """
    return jnp.full(shape, jnp.log(jnp.expm1(jnp.asarray(kappa_init, dtype=jnp.float32))))


class PhaseControlLayer(eqx.Module):
    """Top layer -- phase analogue of `TpchControlLayer`/`SLControlLayer`.
    Same two weight sets (W_rec on its own previous state, W_in on the
    control input), now complex and applied to `e^{i*theta}` rather than
    to `theta` directly (angles don't combine linearly -- the whole point
    of routing through the complex exponential is that IT does, via
    ordinary complex arithmetic, respect the circular topology).
    """

    W_rec: ComplexLinear
    W_in: Optional[ComplexLinear]
    kappa_raw: Array

    has_input: bool = eqx.field(static=True)

    def __init__(
        self,
        state_size: int,
        input_size: Optional[int] = 0,
        kappa_init: float = 1.0,
        *,
        key: PRNGKeyArray,
    ):
        key_rec, key_in = jr.split(key)
        self.W_rec = ComplexLinear(state_size, state_size, key=key_rec)
        self.has_input = input_size > 0
        self.W_in = ComplexLinear(input_size, state_size, key=key_in) if self.has_input else None
        self.kappa_raw = _init_kappa_raw(kappa_init, (state_size,))

    def predict(self, theta_prev: Array, control_input: Optional[Array] = None) -> Array:
        """z_hat_t = W_rec @ e^{i*theta_prev} + W_in @ x_t -- a general
        complex number (not unit modulus, see module docstring)."""
        z_hat = self.W_rec(jnp.exp(1j * theta_prev))
        if self.has_input and control_input is not None:
            z_hat = z_hat + self.W_in(control_input)
        return z_hat


class PhaseHiddenLayer(eqx.Module):
    """Middle layer -- phase analogue of `TpchHiddenLayer`/`SLHiddenLayer`.
    Same three weight sets (own recurrence, parent's previous, parent's
    CURRENT), applied to exponentiated angles."""

    W_rec: ComplexLinear
    W_parent_prev: ComplexLinear
    W_parent_curr: ComplexLinear
    kappa_raw: Array

    def __init__(
        self,
        state_size: int,
        parent_size: int,
        kappa_init: float = 1.0,
        *,
        key: PRNGKeyArray,
    ):
        key_rec, key_pp, key_pc = jr.split(key, 3)
        self.W_rec = ComplexLinear(state_size, state_size, key=key_rec)
        self.W_parent_prev = ComplexLinear(parent_size, state_size, key=key_pp)
        self.W_parent_curr = ComplexLinear(parent_size, state_size, key=key_pc)
        self.kappa_raw = _init_kappa_raw(kappa_init, (state_size,))

    def predict(self, theta_prev: Array, theta_parent_prev: Array, theta_parent_curr: Array) -> Array:
        """z_hat_t = W_rec @ e^{i theta_prev} + W_parent_prev @ e^{i theta_parent_prev}
        + W_parent_curr @ e^{i theta_parent_curr}"""
        return (
            self.W_rec(jnp.exp(1j * theta_prev))
            + self.W_parent_prev(jnp.exp(1j * theta_parent_prev))
            + self.W_parent_curr(jnp.exp(1j * theta_parent_curr))
        )


class PhaseObservationLayer(eqx.Module):
    """Bottom layer -- phase analogue of `TpchObservationLayer`/
    `SLObservationLayer`. Same role: pure linear emission of the parent's
    CURRENT state, no memory/dynamics/kappa of its own (never was a free
    node to begin with). Sensory data is real, so the complex prediction
    is projected with `jnp.real`, exactly as in `slpch`.
    """

    W_parent: ComplexLinear

    def __init__(self, obs_size: int, parent_size: int, *, key: PRNGKeyArray):
        self.W_parent = ComplexLinear(parent_size, obs_size, key=key)

    def predict(self, theta_parent_curr: Array) -> Array:
        """y_hat_t = Re(C @ e^{i*theta_parent_curr})"""
        return jnp.real(self.W_parent(jnp.exp(1j * theta_parent_curr)))
