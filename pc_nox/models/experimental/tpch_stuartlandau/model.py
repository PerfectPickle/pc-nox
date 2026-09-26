"""slpch/model.py

Equinox implementation of Stuart-Landau tPC-H (SL-tPC-H): the same
hierarchical-temporal predictive-coding architecture as `tpch/model.py`'s
`TpchModel`, but with every layer's state replaced by a complex-valued
Stuart-Landau (supercritical Hopf) oscillator, so that inference settles
onto a phase-locked limit cycle instead of a fixed point, and the
per-layer prediction error decomposes into an explicit amplitude term and
an explicit Kuramoto-style phase-synchrony term (see `slpch_energy_fn`).

--------------------------------------------------------------------------
Two things this is NOT a naive port of a flat 2-layer Hopf-PC sketch
--------------------------------------------------------------------------
1. It keeps tPC-H's actual wiring: each state is predicted from its OWN
   previous value (temporal recurrence, W_rec) AND its parent's previous
   AND current value (hierarchical, W_parent_prev/W_parent_curr) -- eq.
   19's structure, not just "layer l+1 predicts layer l".
2. It gets JAX's complex-gradient convention right, which a direct port
   of `ds/dt = -dE/ds` silently does NOT: for a real-valued energy E(z)
   of a complex state z, `jax.grad(E)(z)` returns 2*dE/dz (Wirtinger,
   holding z-bar fixed) -- NOT the steepest-descent direction. Naively
   doing `z - lr * jax.grad(E)(z)` *increases* E in general (checked
   empirically: for E(z)=|z|^2 at z=1+2j, one small SGD-convention step
   raises E from 5.0 to 5.122). The correct descent direction is
   `-conj(jax.grad(E)(z))` -- see `_descent_direction` below, used
   everywhere the original `TpchModel` just negated the grad directly.

--------------------------------------------------------------------------
Where the oscillator dynamics come from, precisely
--------------------------------------------------------------------------
The full per-node Stuart-Landau/Hopf normal form is

    dz/dt = (gamma + i*omega - |z|^2) * z  +  (PC error-correction terms)

The first piece splits into a radial/amplitude part `(gamma - |z|^2) * z`
and a purely rotational part `i*omega * z`. The amplitude part turns out
to be *exactly* the steepest-descent direction (see the conjugation note
above) of a quartic Ginzburg-Landau potential

    V(z) = 0.25 * |z|^4  -  0.5 * gamma * |z|^2

so rather than hand-adding it to the vector field, `slpch_energy_fn`
below adds V(z) as an extra term in the energy itself -- which means
gamma is learned by the ordinary `param_grad`/`update_params` machinery,
completely unchanged from `TpchModel`'s. The rotational part `i*omega*z`
is NOT the gradient of any real scalar (it's the Hamiltonian/symplectic
piece, not the dissipative piece), so it genuinely can't be folded into
the energy -- it's added directly in `make_vector_field`/
`make_activity_step`. One consequence, stated plainly: `omega` is NOT fit
by `param_grad` as written here (it never appears in `slpch_energy_fn`),
only initialised and then held fixed by ordinary weight learning. Making
omega adaptive would need a supplementary (e.g. Kuramoto-style
frequency-adaptation) learning rule this file doesn't implement.

What's model-agnostic and lives elsewhere: exactly as in `tpch/model.py`,
`..inference`'s diffrax integration engine (`settle_diffrax`) doesn't
know this is SL-tPC-H -- see that module. One real caveat specific to
this variant, though: `..inference.make_steady_state_event`'s criteria
all assume the settled trajectory approaches `ds/dt ~ 0` (a fixed
point). A phase-locked oscillator never stops moving -- `ds/dt` stays
bounded away from zero forever (the `i*omega*z` rotation persists even
once amplitude/phase-offset have converged) -- so Mode-1 event-based
early stopping generally will NOT fire here, and if it's forced to
(e.g. a very loose `tol`) it isn't measuring what it measures for
`TpchModel`. `settle_diffrax` below therefore defaults to Mode 2
(fixed-horizon integration over `max_t1`, long enough to cover several
intrinsic periods) rather than `TpchModel`'s event-based default -- see
its docstring.
"""

from dataclasses import asdict
from typing import ClassVar, List, Optional, Sequence, Tuple, Union

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from pc_nox.models import inference
from pc_nox.models.model_base import Activities, ModelBase, Predictions
from .config import SLPCHConfig
from .layers import SLControlLayer, SLHiddenLayer, SLObservationLayer

_EPS = 1e-8  # inside every sqrt below, so |z| stays differentiable at z=0 (see _safe_abs)


def _safe_abs(z: Array) -> Array:
    """|z| with a gradient that stays finite at z=0 (plain jnp.abs's
    gradient is singular there -- amplitude and sync terms below both
    hit z=0 legitimately, e.g. right after init or for a unit whose
    gamma has decayed toward the non-oscillating regime, see
    layers.py's `_init_gamma_raw`).
    """
    return jnp.sqrt(jnp.real(z) ** 2 + jnp.imag(z) ** 2 + _EPS)


def _descent_direction(grads: PyTree) -> PyTree:
    """Steepest-descent direction for a pytree of (possibly complex)
    gradients, as returned by `jax.grad` of a real-valued function. For a
    real leaf this is just `-grad` (conjugation is a no-op); for a
    complex leaf it's `-conj(grad)` -- see this module's docstring for
    why the conjugate is required. Applying `conj` uniformly is safe for
    every leaf regardless of dtype, so this one function replaces the
    plain `tree_map(jnp.negative, ...)` `TpchModel` uses everywhere.
    """
    return jax.tree_util.tree_map(lambda g: -jnp.conj(g), grads)


def _complex_to_real(states: PyTree) -> PyTree:
    """Pack a pytree's complex leaves into real ones (real/imag stacked
    on a new leading axis); leaves that are already real pass through
    unchanged. Used ONLY around calls into `inference.settle_diffrax` --
    diffrax's own solvers/step-size controllers currently warn that
    complex-dtype support is "a work in progress and may not yet produce
    correct results" (confirmed against installed diffrax 0.7.2), since
    PID error norms, `<` comparisons, etc. aren't well-defined on complex
    arrays. Splitting real/imag into a real ODE of twice the dimension is
    the textbook fix (and diffrax's own suggested workaround) and is
    exact: for z = x + iy, dz/dt = f(z) is precisely dx/dt = Re(f),
    dy/dt = Im(f) treated as two independent real ODEs.

    Leaving real leaves untouched (rather than forcing them through the
    same stack-of-two convention) is what lets this same pair of
    functions serve BOTH `settle_diffrax` (a pytree of complex `z`'s
    only) AND `settle_diffrax_adaptive` (a pytree of `(z's, omega's)` --
    the omega's are already real, and don't need or want packing).
    """
    return jax.tree_util.tree_map(
        lambda x: jnp.stack([jnp.real(x), jnp.imag(x)], axis=0) if jnp.iscomplexobj(x) else x,
        states,
    )


def _real_to_complex(real_states: PyTree, template: PyTree) -> PyTree:
    """Inverse of `_complex_to_real`. Needs `template` -- a pytree with
    the same structure as the ORIGINAL (pre-packing) states -- to know
    which leaves to reassemble into complex numbers vs. pass through;
    that information isn't recoverable from `real_states` alone once
    everything is real arrays of various shapes.
    """
    return jax.tree_util.tree_map(
        lambda t, r: (r[0] + 1j * r[1]) if jnp.iscomplexobj(t) else r,
        template, real_states,
    )


# =============================================================================
# SLPCHModel: composes the three oscillator layer types into a full hierarchy
# =============================================================================

class SLPCHModel(eqx.Module, ModelBase):
    """A full SL-tPC-H hierarchy chained top to bottom -- complex analogue
    of `TpchModel`. One control layer, N >= 0 hidden layers, one
    observation layer; see `layers.py` for each role.

    Args:
        control_layer_size, hidden_sizes, obs_size, input_size: same
            meaning as `TpchModel`'s.
        key: PRNG key for layer initialisation.
        gamma_init, omega_init_scale: oscillator init ranges, see
            `SLPCHConfig`.
        amp_weight, sync_weight: energy-term coefficients, see
            `SLPCHConfig` and `slpch_energy_fn`.
        loss: `"mse"` or `"ce"` on the observation term, same meaning as
            `TpchModel`'s.
    """
    model_type: ClassVar[str] = "slpch"
    config_cls: ClassVar[type] = SLPCHConfig
    config: SLPCHConfig = eqx.field(static=True)

    control_layer: SLControlLayer
    hidden_layers: List[SLHiddenLayer]
    observation_layer: SLObservationLayer

    def __init__(
        self,
        control_layer_size: int,
        hidden_sizes: Sequence[int],
        obs_size: int,
        key: PRNGKeyArray,
        input_size: Optional[int] = 0,
        gamma_init: float = 1.0,
        omega_init_scale: float = 2.0,
        amp_weight: float = 1.0,
        sync_weight: float = 1.0,
        loss: str = "mse",
        adapt_omega: bool = False,
        omega_adapt_rate: float = 1.0,
        omega_decay_rate: float = 0.3,
    ):
        self.config = SLPCHConfig(
            control_layer_size=control_layer_size,
            hidden_sizes=tuple(hidden_sizes),
            obs_size=obs_size,
            input_size=input_size,
            gamma_init=gamma_init,
            omega_init_scale=omega_init_scale,
            amp_weight=amp_weight,
            sync_weight=sync_weight,
            loss=loss,
            adapt_omega=adapt_omega,
            omega_adapt_rate=omega_adapt_rate,
            omega_decay_rate=omega_decay_rate,
        )

        n_hidden = len(hidden_sizes)
        key_control, *hidden_keys, key_obs = jr.split(key, 2 + n_hidden)

        self.control_layer = SLControlLayer(
            state_size=control_layer_size, input_size=input_size,
            gamma_init=gamma_init, omega_init_scale=omega_init_scale, key=key_control,
        )

        hidden_layers = []
        parent_size = control_layer_size
        for size, hkey in zip(hidden_sizes, hidden_keys):
            hidden_layers.append(
                SLHiddenLayer(
                    state_size=size, parent_size=parent_size,
                    gamma_init=gamma_init, omega_init_scale=omega_init_scale, key=hkey,
                )
            )
            parent_size = size
        self.hidden_layers = hidden_layers

        self.observation_layer = SLObservationLayer(obs_size=obs_size, parent_size=parent_size, key=key_obs)

    # -------------------------------------------------------------------
    # tiny helper: every dynamical (state-holding) layer, top-to-bottom --
    # i.e. everything EXCEPT the observation layer, which has no state of
    # its own. Same list `states_curr`/`states_prev` are indexed against.
    # -------------------------------------------------------------------
    def _dynamical_layers(self) -> List:
        return [self.control_layer, *self.hidden_layers]

    def predict(
        self,
        states_prev: Activities,
        states_curr: Activities,
        control_input: Optional[Array] = None,
        observation: Optional[Array] = None,
    ) -> Tuple[Predictions, Array]:
        """Run every layer's `predict` once -- identical structure to
        `TpchModel.predict`, just complex-valued. `observation` is unused
        (accepted for calling-convention parity, see `TpchModel.predict`).
        """
        predictions = [self.control_layer.predict(states_prev[0], control_input)]

        for i, layer in enumerate(self.hidden_layers):
            own_prev = states_prev[i + 1]
            parent_prev = states_prev[i]
            parent_curr = states_curr[i]
            predictions.append(layer.predict(own_prev, parent_prev, parent_curr))

        y_hat = self.observation_layer.predict(states_curr[-1])
        return predictions, y_hat

    def init_activities(
        self,
        states_prev: Activities,
        control_input: Optional[Array] = None,
        observation: Optional[Array] = None,
    ) -> Activities:
        """Feedforward kick-start, identical structure to
        `TpchModel.init_activities`. Note this seeds `states_curr` exactly
        at each layer's linear prediction target -- since that's a
        perfectly ordinary complex number (not the origin in general),
        it's a fine starting point for the oscillators despite z=0 being
        an unstable equilibrium of the intrinsic dynamics (see
        `layers.py`): only a state that lands exactly on 0 would need the
        error-correction terms to kick it off-center, and a random
        `states_prev`/weights combination essentially never does.
        """
        control_pred = self.control_layer.predict(states_prev[0], control_input)
        states_curr = [control_pred]

        parent_pred = control_pred
        for i, layer in enumerate(self.hidden_layers):
            own_prev = states_prev[i + 1]
            parent_prev = states_prev[i]
            prediction = layer.predict(own_prev, parent_prev, parent_pred)
            states_curr.append(prediction)
            parent_pred = prediction

        return states_curr

    # =========================================================================
    # Free energy -- amplitude + synchrony split, plus the intrinsic
    # Ginzburg-Landau potential (see module docstring for the derivation
    # of why this last piece reproduces the Hopf amplitude term exactly).
    # =========================================================================

    def _amp_energy(self, z: Array, z_hat: Array) -> Array:
        """0.5 * sum (|z| - |z_hat|)^2 -- amplitude-mismatch term."""
        return 0.5 * jnp.sum((_safe_abs(z) - _safe_abs(z_hat)) ** 2)

    def _sync_energy(self, z: Array, z_hat: Array) -> Array:
        """sum |z| |z_hat| (1 - cos(phase(z) - phase(z_hat))) -- the
        Kuramoto-Sakaguchi phase-coupling potential, weighted by the
        predicted*actual amplitude product (standard Kuramoto convention:
        near-zero-amplitude nodes contribute no synchrony pressure).
        Computed WITHOUT calling `jnp.angle` (whose gradient is singular
        at 0): `|z||z_hat|(1-cos(dphi)) == |z||z_hat| - Re(z * conj(z_hat))`
        is an algebraic identity, and the right-hand side is smooth
        everywhere `_safe_abs` is.

        Together, `_amp_energy + _sync_energy` at weight 1 each exactly
        reproduce the plain complex residual `0.5*|z - z_hat|^2`
        `TpchModel` would compute if you just swapped its states for
        complex ones (algebraic identity: `|z-z_hat|^2 = (|z|-|z_hat|)^2
        + 2|z||z_hat|(1-cos(dphi))`). `amp_weight`/`sync_weight` let you
        pull the two apart instead of always weighting them equally.
        """
        return jnp.sum(_safe_abs(z) * _safe_abs(z_hat) - jnp.real(z * jnp.conj(z_hat)))

    def _intrinsic_potential(self, z: Array, layer) -> Array:
        """0.25*sum|z|^4 - 0.5*sum(gamma * |z|^2) -- see module docstring:
        this is the term whose steepest-descent direction (correctly
        conjugated, see `_descent_direction`) is exactly the Hopf normal
        form's amplitude-saturating pull `(gamma - |z|^2) * z`.
        """
        gamma = jax.nn.softplus(layer.gamma_raw)
        abs2 = jnp.real(z) ** 2 + jnp.imag(z) ** 2
        return jnp.sum(0.25 * abs2 ** 2 - 0.5 * gamma * abs2)

    def slpch_energy_fn(
        self,
        states_prev: Activities,
        states_curr: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        return_layerwise: bool = False,
    ) -> Array:
        """Total free energy: for every dynamical layer (control + hidden),
        `amp_weight * amp_energy + sync_weight * sync_energy +
        intrinsic_potential`, plus the observation term (`mse` or `ce`,
        exactly as `TpchModel.tpch_energy_fn`). No weight/activity
        regularisers (out of scope for this variant, see config.py).

        `return_layerwise`: as in `TpchModel.tpch_energy_fn`, an unsummed
        Array instead of the scalar total, one entry per dynamical layer
        (control, then each hidden layer) followed by the observation
        term -- order matches `layer_labels()`.
        """
        predictions, y_hat = self.predict(states_prev, states_curr, control_input, observation)

        layer_energies = []
        for z, z_hat, layer in zip(states_curr, predictions, self._dynamical_layers()):
            e = (
                self.config.amp_weight * self._amp_energy(z, z_hat)
                + self.config.sync_weight * self._sync_energy(z, z_hat)
                + self._intrinsic_potential(z, layer)
            )
            layer_energies.append(e)

        if self.config.loss == "mse":
            y_error = observation - y_hat
            obs_energy = 0.5 * jnp.sum(y_error ** 2)
        else:  # "ce", validated in SLPCHConfig.__post_init__
            obs_energy = -jnp.sum(observation * jax.nn.log_softmax(y_hat))
        layer_energies.append(obs_energy)

        if return_layerwise:
            return jnp.stack(layer_energies)
        return sum(layer_energies)

    def energy_fn(self, *args, **kwargs):
        """Alias for `slpch_energy_fn`, under the shared-runner name -- see
        `TpchModel.energy_fn`."""
        return self.slpch_energy_fn(*args, **kwargs)

    def prediction_energy_fn(
        self,
        states_prev: Activities,
        states_curr: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        return_layerwise: bool = False,
    ) -> Array:
        """`slpch_energy_fn` MINUS the intrinsic Ginzburg-Landau potential
        term -- i.e. just amp_energy + sync_energy per dynamical layer,
        plus the observation term. Use this (not `slpch_energy_fn`) as
        your diagnostic/step-count-tuning signal: the intrinsic term
        measures "has each unit settled onto its own comfortable
        amplitude," which is real but has nothing to do with prediction
        quality, and dominates `slpch_energy_fn` enough to make it a
        misleading proxy for how well inference is actually doing (this
        is WHY total VFE went negative and didn't track your perceptible
        prediction quality -- confirmed empirically: on a representative
        settle run, the intrinsic term saturated around -1.5 to -1.7
        within ~20 steps while amp+sync kept climbing from ~0 to ~1.8
        over the next 280).

        Also doubles as the "prediction-error force" the adaptive-omega
        machinery below adapts frequency against -- see
        `make_adaptive_vector_field`.
        """
        predictions, y_hat = self.predict(states_prev, states_curr, control_input, observation)
        layer_energies = []
        for z, z_hat in zip(states_curr, predictions):
            e = self.config.amp_weight * self._amp_energy(z, z_hat) + self.config.sync_weight * self._sync_energy(z, z_hat)
            layer_energies.append(e)
        if self.config.loss == "mse":
            obs_energy = 0.5 * jnp.sum((observation - y_hat) ** 2)
        else:
            obs_energy = -jnp.sum(observation * jax.nn.log_softmax(y_hat))
        layer_energies.append(obs_energy)
        if return_layerwise:
            return jnp.stack(layer_energies)
        return sum(layer_energies)

    # =========================================================================
    # Inference -- gradient part of the dynamics (the energy-derived
    # piece; the rotational piece is added on top in the two sections
    # below, since it isn't a gradient -- see module docstring).
    # =========================================================================

    def _rotation_terms(self, states_curr: Activities) -> Activities:
        """i*omega ⊙ z for every dynamical layer's current state."""
        return [layer.rotation_term(z) for layer, z in zip(self._dynamical_layers(), states_curr)]

    def neg_activity_grad(
        self,
        states_curr: Activities,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ) -> Activities:
        """Steepest-descent direction of the free energy w.r.t. states_curr
        (the *gradient-derived* part of the dynamics only -- does NOT
        include the `i*omega*z` rotation, see `neg_activity_grad_full`
        below for the complete vector field). Complex analogue of
        `TpchModel.neg_activity_grad`: uses `_descent_direction`, not a
        plain negation, since these states are complex (see module
        docstring for why that matters).
        """
        energy_of_states = lambda s: self.slpch_energy_fn(states_prev, s, observation, control_input)
        return _descent_direction(jax.grad(energy_of_states)(states_curr))

    def neg_activity_grad_full(
        self,
        states_curr: Activities,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ) -> Activities:
        """Full `ds/dt`: gradient-derived descent direction PLUS the
        rotational `i*omega*z` term. This is the complete Stuart-Landau
        + predictive-coding vector field, evaluated at a single point (as
        opposed to `make_vector_field`, below, which returns the closure
        `diffrax`/`settle_scan` actually integrate).
        """
        descent = self.neg_activity_grad(states_curr, states_prev, observation, control_input)
        rotation = self._rotation_terms(states_curr)
        return jax.tree_util.tree_map(lambda d, r: d + r, descent, rotation)

    def infer_step(
        self,
        states_curr: Activities,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        state_lr: float = 0.1,
    ) -> Activities:
        """One Euler step of the full dynamics. Complex analogue of
        `TpchModel.infer_step`."""

        if self.config.adapt_omega:
            raise ValueError(
                "infer_step ignores config.adapt_omega entirely -- it always uses each "
                "layer's fixed, static omega, regardless of this config flag. That "
                "mismatch is exactly what silently produced \"no effect at all\" "
                "results if you set adapt_omega=True and kept calling infer_step: the "
                "flag was doing nothing. Use `infer_step_adaptive (build your own -- not provided; it's a one-line Euler step of make_adaptive_vector_field, see settle_scan_adaptive's activity_step for the pattern)` instead."
            )

        dstates = self.neg_activity_grad_full(states_curr, states_prev, observation, control_input)
        return jax.tree_util.tree_map(lambda s, d: s + state_lr * d, states_curr, dstates)

    def settle(
        self,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        n_steps: int = 20,
        state_lr: float = 0.1,
    ) -> Activities:
        """Plain Python-loop settling -- complex analogue of `TpchModel.settle`.
        Note the caveat from the module docstring applies: unlike
        `TpchModel`, there's no fixed point to "arrive at" in general --
        `n_steps` controls how many periods of relaxation-toward-the-
        limit-cycle you get, not convergence to a resting state.
        """

        if self.config.adapt_omega:
            raise ValueError(
                "settle ignores config.adapt_omega entirely -- it always uses each "
                "layer's fixed, static omega, regardless of this config flag. That "
                "mismatch is exactly what silently produced \"no effect at all\" "
                "results if you set adapt_omega=True and kept calling settle: the "
                "flag was doing nothing. Use `a python loop around infer_step_adaptive, or settle_scan_adaptive` instead."
            )

        states_curr = self.init_activities(states_prev, control_input, observation)
        for _ in range(n_steps):
            states_curr = self.infer_step(states_curr, states_prev, observation, control_input, state_lr)
        return states_curr

    # =========================================================================
    # Learning -- unchanged in form from TpchModel: because gamma is
    # folded into the energy (see module docstring), grad-of-energy
    # w.r.t. weights reaches every learnable parameter EXCEPT omega (see
    # module docstring for why omega specifically is excluded).
    # =========================================================================

    def param_grad(
        self,
        states_prev: Activities,
        states_curr: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ) -> PyTree:
        """dE/d(weights) at the settled states -- identical role to
        `TpchModel.param_grad`. `eqx.filter_grad` returns each leaf's
        *raw* Wirtinger-style gradient (JAX's own convention: 2*dE/dz
        for a complex leaf, plain dE/dw for a real one -- see this
        module's docstring). Left as-is, `update_params`'s
        `optim.update(grads, ...)` would perform correct SGD on the
        real leaves (gamma_raw) but silently ASCEND on the complex
        weight leaves -- exactly the same bug `_descent_direction` fixes
        for activities, just easy to miss here because `update_params`'s
        code is unchanged from `TpchModel`'s. (Caught empirically, not
        just by inspection: a toy training loop's energy was rising
        step over step until this conjugation was added.) `jnp.conj` is
        a no-op on real leaves, so applying it uniformly is safe and
        keeps `update_params` itself identical to `TpchModel`'s.
        """
        energy_of_weights = lambda m: m.slpch_energy_fn(states_prev, states_curr, observation, control_input)
        raw_grads = eqx.filter_grad(energy_of_weights)(self)
        return jax.tree_util.tree_map(jnp.conj, raw_grads)

    def update_params(
        self,
        grads: PyTree,
        optim: optax.GradientTransformation,
        opt_state: optax.OptState,
    ) -> Tuple[eqx.Module, optax.OptState]:
        """Identical to `TpchModel.update_params`. One caveat worth
        stating plainly: adaptive optimisers (adam and friends) form a
        second-moment estimate from `grad**2`; for a COMPLEX grad that's
        not the same thing as `|grad|**2` (magnitude-squared), so their
        adaptive scaling isn't quite doing what it does for real
        parameters. `optax.sgd` doesn't have this issue (it's a linear
        transform of the grad) and is the safe default for this model's
        complex weights; an adaptive optimiser will very likely still
        train, just without a fully principled complex-aware second
        moment.
        """
        updates, opt_state = optim.update(grads, opt_state, self)
        updated_model = eqx.apply_updates(self, updates)
        return updated_model, opt_state

    # =========================================================================
    # Scan-fused inference (optax-driven activity optimiser)
    # =========================================================================

    def make_activity_step(
        self,
        activity_optim: optax.GradientTransformation,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ):
        """Complex analogue of `TpchModel.make_activity_step`. The grad
        handed to `activity_optim.update` is `conj(dE/dz) - rotation`, so
        that `optax.sgd`'s own `updates = -lr * grad` convention produces
        exactly `lr * (descent + rotation)` -- one Euler step of the full
        vector field (matching `make_vector_field`, below, term for
        term). This is the ONE place in this file the raw (unconjugated)
        `jax.grad` output and the conjugated descent direction both
        appear side by side -- see `_descent_direction`'s docstring if
        the sign here looks surprising.
        """

        if self.config.adapt_omega:
            raise ValueError(
                "make_activity_step ignores config.adapt_omega entirely -- it always uses each "
                "layer's fixed, static omega, regardless of this config flag. That "
                "mismatch is exactly what silently produced \"no effect at all\" "
                "results if you set adapt_omega=True and kept calling make_activity_step: the "
                "flag was doing nothing. Use `make_adaptive_activity_step` instead."
            )

        energy_fn = lambda s: self.slpch_energy_fn(states_prev, s, observation, control_input)

        def activity_step(carry, _):
            states_curr, opt_state = carry
            grads = jax.grad(energy_fn)(states_curr)
            rotation = self._rotation_terms(states_curr)
            combined_grad = jax.tree_util.tree_map(lambda g, r: jnp.conj(g) - r, grads, rotation)
            updates, opt_state = activity_optim.update(combined_grad, opt_state, states_curr)
            states_curr = optax.apply_updates(states_curr, updates)
            return (states_curr, opt_state), states_curr

        return activity_step

    def settle_scan(
        self,
        activity_optim: optax.GradientTransformation,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        n_steps: int = 20,
        return_layerwise: bool = False,
    ) -> Activities:
        """Scan-fused settling -- complex analogue of `TpchModel.settle_scan`.
        Use `optax.sgd` for `activity_optim` unless you've specifically
        accounted for the complex-second-moment caveat on adaptive
        optimisers (see `update_params`'s docstring) -- it applies here
        too, and matters more for activities than weights since they're
        being stepped every relaxation iteration.
        """

        if self.config.adapt_omega:
            raise ValueError(
                "settle_scan ignores config.adapt_omega entirely -- it always uses each "
                "layer's fixed, static omega, regardless of this config flag. That "
                "mismatch is exactly what silently produced \"no effect at all\" "
                "results if you set adapt_omega=True and kept calling settle_scan: the "
                "flag was doing nothing. Use `settle_scan_adaptive` instead."
            )

        states_curr0 = self.init_activities(states_prev, control_input, observation)
        opt_state0 = activity_optim.init(states_curr0)

        activity_step = self.make_activity_step(activity_optim, states_prev, observation, control_input)
        (states_curr, _), states_hist = jax.lax.scan(activity_step, (states_curr0, opt_state0), xs=None, length=n_steps)

        if not return_layerwise:
            return states_curr

        energy_trace_fn = lambda s: self.slpch_energy_fn(states_prev, s, observation, control_input, return_layerwise=True)
        energy_trace = jax.vmap(energy_trace_fn)(states_hist)
        return states_curr, energy_trace

    # =========================================================================
    # Diffrax-fused inference -- thin adapters over the model-agnostic
    # engine in `..inference`, exactly as `TpchModel.settle_diffrax` is.
    # See module docstring for why the *default* steady-state behaviour
    # differs from `TpchModel`'s.
    # =========================================================================

    def make_vector_field(
        self,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ):
        """Builds `ds/dt` = (conjugated) energy descent + rotation, as a
        closure over the fixed trajectory context -- diffrax analogue of
        `make_activity_step`, complex analogue of `TpchModel.make_vector_field`.
        """

        if self.config.adapt_omega:
            raise ValueError(
                "make_vector_field ignores config.adapt_omega entirely -- it always uses each "
                "layer's fixed, static omega, regardless of this config flag. That "
                "mismatch is exactly what silently produced \"no effect at all\" "
                "results if you set adapt_omega=True and kept calling make_vector_field: the "
                "flag was doing nothing. Use `make_adaptive_vector_field` instead."
            )

        energy_fn = lambda s: self.slpch_energy_fn(states_prev, s, observation, control_input)

        def vector_field(t, states_curr, args):
            descent = _descent_direction(jax.grad(energy_fn)(states_curr))
            rotation = self._rotation_terms(states_curr)
            return jax.tree_util.tree_map(lambda d, r: d + r, descent, rotation)

        return vector_field

    def make_steady_state_event(
        self,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        tol: Optional[float] = 1e-3,
        criterion: str = "rms",
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> diffrax.Event:
        """Adapter over `inference.make_steady_state_event` -- provided for
        API parity with `TpchModel`, but see this module's docstring
        before using it: none of `"rms"`/`"relative_rms"`/`"energy_rate"`
        are measuring "reached the limit cycle" the way they measure
        "reached the fixed point" for `TpchModel` -- a converged
        oscillator still has `ds/dt = i*omega*z_converged != 0` forever.

        There's a second, sharper problem if you call this directly (as
        opposed to through `settle_diffrax`, which sidesteps it): every
        criterion in `inference.make_steady_state_event` computes
        `jnp.square(leaf)` on the raw vector field, which for a COMPLEX
        leaf is the complex square `z*z`, not `|z|^2` -- e.g.
        `jnp.square(1+2j) == (-3+4j)`, not `5.0`. The resulting "rms"
        is then a complex number compared against a real `tol`, which
        silently does something (JAX doesn't raise), just not anything
        meaningful. `settle_diffrax` never hits this because it hands
        `inference.settle_diffrax` an already real-packed (Re, Im)
        vector field (see `_complex_to_real`), so its internally-built
        event -- a DIFFERENT event object from whatever this method
        returns -- squares real numbers correctly. If you need an Event
        for your own `diffeqsolve` call, build it from a real-packed
        vector field the same way `settle_diffrax` does, not straight
        from `make_vector_field`.
        """
        vector_field = self.make_vector_field(states_prev, observation, control_input)
        energy_fn = lambda s: self.slpch_energy_fn(states_prev, s, observation, control_input)
        return inference.make_steady_state_event(
            vector_field, tol=tol, criterion=criterion, rtol=rtol, atol=atol, energy_fn=energy_fn,
        )

    def settle_diffrax(
        self,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        max_t1: float = 20.0,
        dt0: Optional[float] = None,
        n_save: int = 20,
        solver: Optional[diffrax.AbstractSolver] = None,
        stepsize_controller: Optional[diffrax.AbstractStepSizeController] = None,
        steady_state_tol: Optional[float] = None,
        steady_state_criterion: str = "rms",
        steady_state_rtol: Optional[float] = None,
        steady_state_atol: Optional[float] = None,
        return_layerwise: bool = False,
    ) -> Union[Activities, Tuple[Activities, Array, Array]]:
        """Adapter over `inference.settle_diffrax` -- see that docstring
        for Mode 1 vs Mode 2 and the `return_layerwise` inf-padding
        caveat. The ONE default that differs from `TpchModel.settle_diffrax`:
        `steady_state_tol=None` here (Mode 2, fixed-horizon), vs `1e-3`
        there (Mode 1, event-based) -- see this module's docstring for
        why event-based early stopping isn't a good fit for oscillatory
        dynamics. Pass an explicit `steady_state_tol` to opt back into
        Mode 1 anyway; `max_t1` should cover several intrinsic periods
        (roughly `2*pi / min(|omega|)`) for the fixed-horizon default to
        actually reach the phase-locked regime.

        Internally packs/unpacks complex states to/from real (Re, Im)
        pairs around the `inference.settle_diffrax` call -- see
        `_complex_to_real`'s docstring for why. This is invisible from
        the outside: arguments in, and the returned `Activities`, are
        ordinary complex arrays exactly as everywhere else in this class.
        """

        if self.config.adapt_omega:
            raise ValueError(
                "settle_diffrax ignores config.adapt_omega entirely -- it always uses each "
                "layer's fixed, static omega, regardless of this config flag. That "
                "mismatch is exactly what silently produced \"no effect at all\" "
                "results if you set adapt_omega=True and kept calling settle_diffrax: the "
                "flag was doing nothing. Use `settle_diffrax_adaptive` instead."
            )

        states_curr0 = self.init_activities(states_prev, control_input, observation)
        vector_field = self.make_vector_field(states_prev, observation, control_input)
        energy_fn = lambda s: self.slpch_energy_fn(states_prev, s, observation, control_input)
        layerwise_energy_fn = lambda s: self.slpch_energy_fn(
            states_prev, s, observation, control_input, return_layerwise=True
        )

        # diffrax sees only real arithmetic from here down (see
        # _complex_to_real's docstring) -- wrap the complex closures
        # above accordingly.
        def vector_field_real(t, states_real, args):
            return _complex_to_real(vector_field(t, _real_to_complex(states_real, states_curr0), args))

        energy_fn_real = lambda s: energy_fn(_real_to_complex(s, states_curr0))
        layerwise_energy_fn_real = lambda s: layerwise_energy_fn(_real_to_complex(s, states_curr0))

        result = inference.settle_diffrax(
            vector_field_real,
            _complex_to_real(states_curr0),
            energy_fn=energy_fn_real,
            layerwise_energy_fn=layerwise_energy_fn_real,
            max_t1=max_t1,
            dt0=dt0,
            n_save=n_save,
            solver=solver,
            stepsize_controller=stepsize_controller,
            steady_state_tol=steady_state_tol,
            steady_state_criterion=steady_state_criterion,
            steady_state_rtol=steady_state_rtol,
            steady_state_atol=steady_state_atol,
            return_layerwise=return_layerwise,
        )

        if return_layerwise:
            states_curr_real, energy_trace, ts = result
            return _real_to_complex(states_curr_real, states_curr0), energy_trace, ts
        return _real_to_complex(result, states_curr0)

    # =========================================================================
    # Adaptive omega (opt-in: config.adapt_omega=True) -- promotes each
    # dynamical layer's rotation rate from a fixed layer parameter to a
    # per-timestep DYNAMIC quantity, adapted during settling itself. See
    # `SLPCHConfig`'s docstring for the on/off switch and rate constants,
    # and the module docstring's discussion of why static omega leaves
    # phase a "neutral" direction with no restoring force toward
    # locking -- this section is the fix for that, modelled on
    # "adaptive-frequency oscillators" (Righetti, Buchli & Ijspeert,
    # 2006, used for learning to entrain Hopf oscillators to periodic
    # driving signals in robotics/CPG contexts -- this is an ADAPTATION
    # of that idea to a multi-layer PC setting, not a transcription of
    # their exact equations, and hasn't been validated beyond the
    # empirical comparison in this repo's demo -- treat it as a
    # promising but unproven design, not a settled result.
    #
    # The adaptation rule, per dynamical layer:
    #
    #   torque   = Im(conj(z) * F) / |z|      -- tangential component of
    #                                             the prediction-error
    #                                             force F, i.e. how hard
    #                                             F is pushing z's PHASE
    #                                             (as opposed to its
    #                                             amplitude) forward/back
    #   domega/dt = kappa * torque  -  lambda * (omega - omega_rest)
    #
    # where F is the steepest-descent direction of `prediction_energy_fn`
    # ONLY (amp+sync+obs -- deliberately excluding the intrinsic
    # potential, which is radial and produces no torque anyway), and
    # `omega_rest` is the layer's own static `omega` field -- now playing
    # the role of a rest frequency, not the rotation rate actually used.
    # Intuition: if the prediction-error force is persistently pushing a
    # unit's phase forward, its own rotation should speed up to
    # anticipate that, reducing the correction needed each step; absent
    # forcing, `lambda` pulls it back to a fixed baseline rather than
    # letting it drift or blow up (this is the same decay-to-baseline
    # idea from your own question about instability).
    #
    # omega itself is NOT a model parameter -- like states_prev, it's a
    # per-timestep quantity the CALLER threads across timesteps
    # (`omega_prev` below), defaulting to each layer's rest omega on the
    # first call of a sequence. `param_grad` still can't reach it, for
    # the same structural reason it can't reach the static omega (see
    # module docstring): it never appears inside a differentiable energy.
    # =========================================================================

    def init_omegas(self) -> List[Array]:
        """Default starting point for a fresh sequence: each dynamical
        layer's own rest omega."""
        return [layer.omega for layer in self._dynamical_layers()]

    def _prediction_force(
        self,
        states_curr: Activities,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ) -> Activities:
        """Steepest-descent direction of `prediction_energy_fn` alone
        (amp+sync+obs, no intrinsic potential) -- the "F" the omega
        torque is computed from, see this section's docstring above.
        """
        energy_fn = lambda s: self.prediction_energy_fn(states_prev, s, observation, control_input)
        return _descent_direction(jax.grad(energy_fn)(states_curr))

    def _omega_dynamics(
        self,
        states_curr: Activities,
        omegas: List[Array],
        force: Activities,
    ) -> List[Array]:
        """domega/dt for every dynamical layer -- see this section's
        docstring for the rule."""
        domegas = []
        for layer, z, om, f in zip(self._dynamical_layers(), states_curr, omegas, force):
            torque = jnp.imag(jnp.conj(z) * f) / _safe_abs(z)
            domegas.append(self.config.omega_adapt_rate * torque - self.config.omega_decay_rate * (om - layer.omega))
        return domegas

    def make_adaptive_vector_field(
        self,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ):
        """Adaptive-omega analogue of `make_vector_field`: operates on the
        combined state `(states_curr, omegas)` -- a 2-tuple of two
        parallel `Activities`-shaped lists -- instead of `states_curr`
        alone. `dstates` is computed exactly as in `make_vector_field`
        (same energy, same conjugation fix), just using each layer's
        CURRENT dynamic `omega` for the rotation term instead of its
        fixed one; `domegas` follows the rule in this section's
        docstring.
        """
        energy_fn = lambda s: self.slpch_energy_fn(states_prev, s, observation, control_input)

        def vector_field(t, state, args):
            states_curr, omegas = state
            descent = _descent_direction(jax.grad(energy_fn)(states_curr))
            rotation = [layer.rotation_term(z, om) for layer, z, om in zip(self._dynamical_layers(), states_curr, omegas)]
            dstates = jax.tree_util.tree_map(lambda d, r: d + r, descent, rotation)

            force = self._prediction_force(states_curr, states_prev, observation, control_input)
            domegas = self._omega_dynamics(states_curr, omegas, force)
            return dstates, domegas

        return vector_field

    def make_adaptive_activity_step(
        self,
        activity_optim: optax.GradientTransformation,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ):
        """Adaptive-omega analogue of `make_activity_step`. `activity_optim`
        is applied to the COMBINED `(states_curr, omegas)` pytree in one
        `optim.update` call, so an omega leaf gets the same optimiser
        (e.g. `optax.sgd`) as the z leaves -- there's no separate
        omega-specific learning rate here beyond `omega_adapt_rate`/
        `omega_decay_rate` already scaling `domega/dt` itself. Real
        leaves (omega) don't need the `jnp.conj` correction z leaves
        do -- conj is a no-op on them, so applying it uniformly (as
        `_descent_direction` already does inside `_prediction_force`) is
        safe without a special case.
        """
        energy_fn = lambda s: self.slpch_energy_fn(states_prev, s, observation, control_input)

        def activity_step(carry, _):
            (states_curr, omegas), opt_state = carry
            grads = jax.grad(energy_fn)(states_curr)
            rotation = [layer.rotation_term(z, om) for layer, z, om in zip(self._dynamical_layers(), states_curr, omegas)]
            combined_grad_states = jax.tree_util.tree_map(lambda g, r: jnp.conj(g) - r, grads, rotation)

            force = self._prediction_force(states_curr, states_prev, observation, control_input)
            domegas = self._omega_dynamics(states_curr, omegas, force)
            combined_grad_omegas = jax.tree_util.tree_map(jnp.negative, domegas)  # optax sign convention, see make_activity_step

            combined_grad = (combined_grad_states, combined_grad_omegas)
            updates, opt_state = activity_optim.update(combined_grad, opt_state, (states_curr, omegas))
            states_curr, omegas = optax.apply_updates((states_curr, omegas), updates)
            return ((states_curr, omegas), opt_state), (states_curr, omegas)

        return activity_step

    def settle_scan_adaptive(
        self,
        activity_optim: optax.GradientTransformation,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        omega_prev: Optional[List[Array]] = None,
        n_steps: int = 20,
        return_layerwise: bool = False,
    ) -> Union[Tuple[Activities, List[Array]], Tuple[Activities, List[Array], Array]]:
        """Adaptive-omega analogue of `settle_scan`. `omega_prev` is the
        adapted omega carried in from the previous EXTERNAL timestep
        (the same role `states_prev` plays for z) -- pass `None` on the
        first call of a sequence to start from each layer's rest omega
        (`init_omegas()`), or thread the returned `omega_curr` forward
        yourself to warm-start each frame from where the last one left
        off (recommended -- resetting every frame throws away the
        adaptation).

        Returns `(states_curr, omega_curr)`, or `(states_curr,
        omega_curr, energy_trace)` if `return_layerwise` -- note
        `energy_trace` is from `slpch_energy_fn` (not
        `prediction_energy_fn`), for consistency with `settle_scan`.
        """
        states_curr0 = self.init_activities(states_prev, control_input, observation)
        omegas0 = self.init_omegas() if omega_prev is None else omega_prev
        opt_state0 = activity_optim.init((states_curr0, omegas0))

        activity_step = self.make_adaptive_activity_step(activity_optim, states_prev, observation, control_input)
        (final_state, _), hist = jax.lax.scan(
            activity_step, ((states_curr0, omegas0), opt_state0), xs=None, length=n_steps,
        )
        states_curr, omega_curr = final_state

        if not return_layerwise:
            return states_curr, omega_curr

        states_hist, _ = hist
        energy_trace_fn = lambda s: self.slpch_energy_fn(states_prev, s, observation, control_input, return_layerwise=True)
        energy_trace = jax.vmap(energy_trace_fn)(states_hist)
        return states_curr, omega_curr, energy_trace

    def settle_diffrax_adaptive(
        self,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        omega_prev: Optional[List[Array]] = None,
        max_t1: float = 20.0,
        dt0: Optional[float] = None,
        n_save: int = 20,
        solver: Optional[diffrax.AbstractSolver] = None,
        stepsize_controller: Optional[diffrax.AbstractStepSizeController] = None,
    ) -> Tuple[Activities, List[Array]]:
        """Adaptive-omega analogue of `settle_diffrax`, Mode 2 (fixed-
        horizon) only -- event-based early stopping (Mode 1) is even
        less meaningful here than for the fixed-omega case (see module
        docstring), since omega itself is now moving too. Uses
        `diffrax.diffeqsolve` directly rather than going through
        `inference.settle_diffrax` (which assumes a single `Activities`
        pytree, not the combined `(states, omegas)` one) -- still packs
        through `_complex_to_real`/`_real_to_complex` for the same
        reason `settle_diffrax` does.
        """
        if solver is None:
            solver = diffrax.Heun()
        if stepsize_controller is None:
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)

        states_curr0 = self.init_activities(states_prev, control_input, observation)
        omegas0 = self.init_omegas() if omega_prev is None else omega_prev
        y0 = (states_curr0, omegas0)

        vector_field = self.make_adaptive_vector_field(states_prev, observation, control_input)

        def vector_field_real(t, y_real, args):
            return _complex_to_real(vector_field(t, _real_to_complex(y_real, y0), args))

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field_real), solver, t0=0.0, t1=max_t1, dt0=dt0,
            y0=_complex_to_real(y0), saveat=diffrax.SaveAt(t1=True), stepsize_controller=stepsize_controller,
        )
        # sol.ys carries an extra leading (save-point) axis on every leaf --
        # strip it BEFORE _real_to_complex, whose real/imag unpacking
        # assumes each leaf's leading axis is the [Re, Im] pair, not time.
        final_real = jax.tree_util.tree_map(lambda x: x[-1], sol.ys)
        states_curr, omega_curr = _real_to_complex(final_real, y0)
        return states_curr, omega_curr

    # =========================================================================
    # Saving and loading
    # =========================================================================

    @classmethod
    def from_config(cls, config: SLPCHConfig, *, key) -> "SLPCHModel":
        return cls(
            control_layer_size=config.control_layer_size,
            hidden_sizes=config.hidden_sizes,
            obs_size=config.obs_size,
            key=key,
            input_size=config.input_size,
            gamma_init=config.gamma_init,
            omega_init_scale=config.omega_init_scale,
            amp_weight=config.amp_weight,
            sync_weight=config.sync_weight,
            loss=config.loss,
            adapt_omega=config.adapt_omega,
            omega_adapt_rate=config.omega_adapt_rate,
            omega_decay_rate=config.omega_decay_rate,
        )

    @classmethod
    def layer_labels(cls, config: SLPCHConfig) -> List[str]:
        """Matches `slpch_energy_fn(..., return_layerwise=True)`'s output
        order -- identical convention to `TpchModel.layer_labels`."""
        return (
            ["Control"]
            + [f"Hidden {i + 1}" for i in range(len(config.hidden_sizes))]
            + ["Observation"]
        )

    @classmethod
    def zero_activities(cls, config: SLPCHConfig) -> Activities:
        """Complex analogue of `TpchModel.zero_activities`. Used only as a
        fixed `states_prev` context (e.g. the start of a sequence), never
        itself integrated -- so landing exactly on the unstable z=0
        equilibrium (see layers.py) is harmless here, unlike it would be
        as an initial condition for `settle`/`settle_scan`/`settle_diffrax`
        (which all seed from `init_activities`'s feedforward prediction
        instead, not from this).
        """
        sizes = [config.control_layer_size, *config.hidden_sizes]
        return [jnp.zeros(s, dtype=jnp.complex64) for s in sizes]


def make_train_step(
    param_optim: optax.GradientTransformation,
    activity_optim: optax.GradientTransformation,
    n_infer_steps: int,
    control_input: Optional[Array] = None,
):
    """Builds one fully-jitted training step using adaptive omegas: settle -> log-quantities -> weight update.

    Args:
        param_optim: Optax transform used for the weight update.
        activity_optim: Optax transform used during activity settling.
        n_infer_steps: Number of inference settling steps.
        control_input: Optional array for exogenous control signals.

    Returns:
        train_step: A function with signature
            `train_step(model, param_opt_state, states_prev, y, omega_prev=None, return_layerwise=False)`
            -> `(model, param_opt_state, states_curr, omega_curr, y_hat_before, y_hat_after,
            energy_before, energy_after, energy_trace)`.
    """
    @eqx.filter_jit
    def train_step(
        model,
        param_opt_state,
        states_prev,
        y,
        omega_prev: Optional[List[Array]] = None,
        return_layerwise: bool = False,
    ):
        states_curr_init = model.init_activities(states_prev, control_input, y)
        _, y_hat_before = model.predict(states_prev, states_curr_init, control_input, y)
        energy_before = model.energy_fn(states_prev, states_curr_init, y, control_input)

        settle_result = model.settle_scan_adaptive(
            activity_optim,
            states_prev,
            y,
            control_input=control_input,
            omega_prev=omega_prev,
            n_steps=n_infer_steps,
            return_layerwise=return_layerwise,
        )

        if return_layerwise:
            states_curr, omega_curr, energy_trace = settle_result
        else:
            states_curr, omega_curr = settle_result
            energy_trace = None

        _, y_hat_after = model.predict(states_prev, states_curr, control_input, y)
        energy_after = model.energy_fn(states_prev, states_curr, y, control_input)

        grads = model.param_grad(states_prev, states_curr, y, control_input)
        updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
        model = eqx.apply_updates(model, updates)
        model = model.postprocess_params()

        return (
            model,
            param_opt_state,
            states_curr,
            omega_curr,
            y_hat_before,
            y_hat_after,
            energy_before,
            energy_after,
            energy_trace,
        )

    return train_step