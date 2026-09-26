"""slpch/layers.py

The three layer roles that compose an `SLPCHModel` -- complex-valued
analogues of `tpch.layers`' `TpchControlLayer`/`TpchHiddenLayer`/
`TpchObservationLayer`. Same top-to-bottom roles and the same
temporal-recurrence + hierarchical-prediction wiring (own previous state,
parent's previous state, parent's CURRENT state), just:

  * every weight matrix is complex (`ComplexLinear`, below, standing in
    for `eqx.nn.Linear(..., use_bias=False)`), and
  * the control/hidden layers additionally own a supercritical-Hopf
    bifurcation parameter `gamma` and intrinsic frequency `omega` per
    node -- the oscillator parameters that replace `act_fn`.

No activation function anywhere here (contrast `TpchControlLayer`/
`TpchHiddenLayer`'s `tanh`): the nonlinearity that used to come from
`act_fn` now comes from the oscillators' own intrinsic dynamics -- the
`(gamma - |z|^2) * z` amplitude-saturating term folded into
`model.py`'s `slpch_energy_fn` (see that docstring), plus the purely
rotational `i * omega * z` term added in `rotation_term` below. Every
`predict()` here is therefore a *plain complex-linear* prediction target,
exactly like the observation layer's linear readout already was in the
original tPC-H.
"""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class ComplexLinear(eqx.Module):
    """Bias-free linear map with complex weights -- stands in for
    `eqx.nn.Linear(in_size, out_size, use_bias=False)` everywhere below,
    since `eqx.nn.Linear` itself only initialises real weights.

    Independent real/imag parts, each Uniform(-1/sqrt(fan_in),
    1/sqrt(fan_in)) -- the same fan-in scaling `eqx.nn.Linear` uses,
    applied to each part separately (standard complex-weight init
    convention, e.g. Trabelsi et al. 2018's independent-Glorot variant).
    Casts its input to the weight's (complex) dtype, so a real vector
    (e.g. a real control_input) can be passed straight in.
    """

    weight: Array  # complex, shape (out_size, in_size)

    def __init__(self, in_size: int, out_size: int, *, key: PRNGKeyArray):
        key_re, key_im = jr.split(key)
        scale = 1.0 / jnp.sqrt(in_size)
        real = jr.uniform(key_re, (out_size, in_size), minval=-scale, maxval=scale)
        imag = jr.uniform(key_im, (out_size, in_size), minval=-scale, maxval=scale)
        self.weight = (real + 1j * imag).astype(jnp.complex64)

    def __call__(self, x: Array) -> Array:
        return self.weight @ x.astype(self.weight.dtype)


def _init_gamma_raw(gamma_init: float, shape) -> Array:
    """Unconstrained per-node param s.t. softplus(gamma_raw) == gamma_init
    exactly at init (inverse-softplus). `gamma = softplus(gamma_raw)` is
    how every layer below reads it back out, guaranteeing gamma > 0
    (supercritical) for any finite gamma_raw -- including after gradient
    updates push it around during learning. A unit whose gamma_raw is
    driven very negative just fades toward a non-oscillating fixed point
    at the origin rather than becoming unstable -- learning can "turn
    off" oscillation for a node if that's what fits the data, it can't
    blow the dynamics up.
    """
    return jnp.full(shape, jnp.log(jnp.expm1(jnp.asarray(gamma_init, dtype=jnp.float32))))


class SLControlLayer(eqx.Module):
    """Top layer of an SL-tPC-H network -- complex analogue of
    `TpchControlLayer`. Same two weight sets (W_rec on its own previous
    state, W_in on the current control input), now complex; plus the
    per-node oscillator parameters gamma_raw/omega that `TpchControlLayer`
    doesn't have (it used a fixed `act_fn` instead).
    """

    W_rec: ComplexLinear
    W_in: Optional[ComplexLinear]
    gamma_raw: Array  # unconstrained; gamma = softplus(gamma_raw), see _init_gamma_raw
    omega: Array  # intrinsic angular frequency, one per node; learnable, no sign constraint

    has_input: bool = eqx.field(static=True)

    def __init__(
        self,
        state_size: int,
        input_size: Optional[int] = 0,
        gamma_init: float = 1.0,
        omega_init_scale: float = 2.0,
        *,
        key: PRNGKeyArray,
    ):
        key_rec, key_in, key_omega = jr.split(key, 3)
        self.W_rec = ComplexLinear(state_size, state_size, key=key_rec)
        self.has_input = input_size > 0
        self.W_in = ComplexLinear(input_size, state_size, key=key_in) if self.has_input else None
        self.gamma_raw = _init_gamma_raw(gamma_init, (state_size,))
        self.omega = jr.uniform(key_omega, (state_size,), minval=-omega_init_scale, maxval=omega_init_scale)

    def predict(self, state_prev: Array, control_input: Optional[Array] = None) -> Array:
        """z_hat_t = W_rec @ s_{t-1} + W_in @ x_t -- plain complex-linear
        prediction target (no activation, see module docstring)."""
        z_hat = self.W_rec(state_prev)
        if self.has_input and control_input is not None:
            z_hat = z_hat + self.W_in(control_input)
        return z_hat

    def rotation_term(self, z: Array, omega: Optional[Array] = None) -> Array:
        """i * omega ⊙ z -- the purely rotational (Hamiltonian, non-
        gradient) piece of the Hopf normal form. Unlike the amplitude
        term `(gamma - |z|^2) * z`, this is NOT the descent direction of
        any real scalar energy, so it can't be obtained via `jax.grad` on
        `slpch_energy_fn` -- it's added directly in `model.py`'s
        `make_vector_field`/`make_activity_step` instead. See this
        module's docstring for where the amplitude term comes from.

        `omega`: defaults to this layer's own learnable/static `self.omega`.
        Callers using the ADAPTIVE-omega path (see `model.py`'s
        `make_adaptive_vector_field`) pass in a per-timestep, dynamically
        adapted frequency instead -- `self.omega` then plays the role of
        the baseline/rest frequency that dynamic value decays toward,
        not the rotation rate actually used.
        """
        return 1j * (self.omega if omega is None else omega) * z


class SLHiddenLayer(eqx.Module):
    """Middle layer of an SL-tPC-H network -- complex analogue of
    `TpchHiddenLayer`. Same three weight sets (own recurrence, parent's
    previous state, parent's CURRENT state), now complex, plus
    gamma_raw/omega as on `SLControlLayer`.
    """

    W_rec: ComplexLinear
    W_parent_prev: ComplexLinear
    W_parent_curr: ComplexLinear
    gamma_raw: Array
    omega: Array

    def __init__(
        self,
        state_size: int,
        parent_size: int,
        gamma_init: float = 1.0,
        omega_init_scale: float = 2.0,
        *,
        key: PRNGKeyArray,
    ):
        key_rec, key_pp, key_pc, key_omega = jr.split(key, 4)
        self.W_rec = ComplexLinear(state_size, state_size, key=key_rec)
        self.W_parent_prev = ComplexLinear(parent_size, state_size, key=key_pp)
        self.W_parent_curr = ComplexLinear(parent_size, state_size, key=key_pc)
        self.gamma_raw = _init_gamma_raw(gamma_init, (state_size,))
        self.omega = jr.uniform(key_omega, (state_size,), minval=-omega_init_scale, maxval=omega_init_scale)

    def predict(self, state_prev: Array, parent_prev: Array, parent_curr: Array) -> Array:
        """z_hat_t = W_rec @ z_{t-1} + W_parent_prev @ s_{t-1} + W_parent_curr @ s_t"""
        return self.W_rec(state_prev) + self.W_parent_prev(parent_prev) + self.W_parent_curr(parent_curr)

    def rotation_term(self, z: Array, omega: Optional[Array] = None) -> Array:
        """See `SLControlLayer.rotation_term` -- identical role, including
        the optional dynamic-`omega` override used by the adaptive-omega
        path."""
        return 1j * (self.omega if omega is None else omega) * z


class SLObservationLayer(eqx.Module):
    """Bottom layer of an SL-tPC-H network -- complex analogue of
    `TpchObservationLayer`. Same role: a pure linear emission of the
    parent's CURRENT state, no memory/dynamics of its own -- which is why,
    unlike the two layers above, this one has no gamma/omega: it was
    never a free-running oscillator in the real-valued model either, it's
    a static readout, and stays one here.

    The one real change: sensory data `y_t` is real-valued, not complex,
    so the complex prediction is projected down with `jnp.real` before
    comparison -- see `model.py`'s `slpch_energy_fn` for the observation
    loss this feeds.
    """

    W_parent: ComplexLinear

    def __init__(self, obs_size: int, parent_size: int, *, key: PRNGKeyArray):
        self.W_parent = ComplexLinear(parent_size, obs_size, key=key)

    def predict(self, parent_curr: Array) -> Array:
        """y_hat_t = Re(C @ z_t) -- linear complex readout, real part taken
        as the observable (the convention that lets a real sensory signal
        be "read off" an oscillator's real axis)."""
        return jnp.real(self.W_parent(parent_curr))
