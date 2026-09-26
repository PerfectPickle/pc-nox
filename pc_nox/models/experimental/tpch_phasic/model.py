"""kpch/model.py

Equinox implementation of KPCH: the phase-only, non-oscillatory sibling
of `slpch`. Same hierarchical-temporal predictive-coding architecture as
`tpch.model.TpchModel`/`slpch.model.SLPCHModel`, but the representational
content at every layer is now PURELY a phase -- there's no amplitude
degree of freedom at all, and no intrinsic dynamics (no Hopf term, no
rotation, no adaptive omega). States settle to a genuine FIXED POINT,
exactly like `TpchModel`'s real-valued states do -- just a fixed point
on a circle (a phase configuration) rather than in R^n.

--------------------------------------------------------------------------
Why this is simpler and more robust than slpch, not just "smaller"
--------------------------------------------------------------------------
The representational state at each node is a real angle `theta`;
`z = e^{i*theta}` is unit modulus for EVERY value `theta` can take, by
construction -- there is nothing analogous to `slpch`'s intrinsic
Ginzburg-Landau potential needed to keep it there, because it can never
leave. Two direct consequences:

1. No stiff nonlinearity. `slpch`'s `0.25|z|^4` term has a gradient that
   scales as `|z|^3`, which is exactly what made explicit-Euler settling
   blow up at moderately aggressive step sizes (confirmed empirically in
   that model). There is no such term here -- `kpch_energy_fn` is a
   bounded, smooth function of `theta` (built entirely from `cos`/`sin`,
   which are bounded everywhere), so there's no analogous blow-up mode.

2. No complex-gradient conjugation gotcha. `theta` is REAL. Every energy
   here is a real-valued function of a real variable, so plain
   `jax.grad` already gives the correct steepest-descent direction --
   none of `slpch`'s `_descent_direction`/conjugation machinery is
   needed anywhere in this file, and `..inference.settle_diffrax` can be
   called directly with no complex/real packing workaround either (no
   diffrax complex-dtype warning to route around, since nothing here is
   complex-dtyped at the state level -- only the WEIGHTS are).

The tradeoff for that robustness is exactly what you'd expect: no
amplitude channel means no amplitude information can be represented or
predicted at all -- if your task needs "how strongly is this feature
present," not just "what phase is it at," that information has nowhere
to live in this model. `slpch` is the version that keeps both channels
available (at the cost of the numerical issues above); this is the
version that commits fully to phase as the only content.

--------------------------------------------------------------------------
What's predicted, and where the coupling strength comes in
--------------------------------------------------------------------------
Each layer's `predict()` (see layers.py) forms `z_hat` as a complex-
linear combination of its neighbours' EXPONENTIATED states -- own
previous, parent's previous, parent's current, exactly `TpchModel`'s eq.
19 structure. `z_hat` itself is a general complex number, not unit
modulus: its phase is the predicted phase; its magnitude is an
incidental by-product of how aligned the combined inputs happen to be
(structurally similar to a Kuramoto order parameter), not something
directly optimised.

The energy per dynamical layer is a per-node, kappa-weighted
Kuramoto-Sakaguchi term:

    E_j = kappa_j * (1 - cos(theta_j - phase(z_hat_j)))

computed WITHOUT calling `jnp.angle` (whose gradient is singular at 0,
same concern as in `slpch`): `cos(theta_j - phase(z_hat_j)) ==
Re(z_j * conj(z_hat_j)) / |z_hat_j|` since `|z_j| = 1` exactly, and the
division uses `_safe_abs` for the same reason `slpch` does. `kappa_j` is
a LEARNABLE per-node parameter distinct from the prediction weights --
the prediction weights decide WHAT phase is predicted, `kappa_j` decides
how strongly that node is pulled toward matching it. Unlike `slpch`'s
`omega`, `kappa` sits directly inside the energy, so it's learned by the
ordinary `param_grad` machinery with no caveats.
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
from .config import KPCHConfig
from .layers import PhaseControlLayer, PhaseHiddenLayer, PhaseObservationLayer

_EPS = 1e-8


def _safe_abs(z: Array) -> Array:
    """|z| with a gradient that stays finite at z=0 -- see `slpch`'s
    identically-named helper for why (z_hat legitimately can be exactly
    or near 0 here too: a node whose upstream inputs are currently
    completely out of phase with each other sums to a near-zero
    "confidence", and that's a real, expected state, not a corner case
    to special-case away).
    """
    return jnp.sqrt(jnp.real(z) ** 2 + jnp.imag(z) ** 2 + _EPS)


# =============================================================================
# KPCHModel
# =============================================================================

class KPCHModel(eqx.Module, ModelBase):
    """A full KPCH hierarchy chained top to bottom -- phase-only analogue
    of `TpchModel`/`SLPCHModel`. One control layer, N >= 0 hidden layers,
    one observation layer.

    Args:
        control_layer_size, hidden_sizes, obs_size, input_size: same
            meaning as `TpchModel`'s/`SLPCHModel`'s.
        key: PRNG key for layer initialisation.
        kappa_init: initial per-node coupling strength, see `KPCHConfig`.
        loss: `"mse"` or `"ce"` on the observation term.
    """
    model_type: ClassVar[str] = "kpch"
    config_cls: ClassVar[type] = KPCHConfig
    config: KPCHConfig = eqx.field(static=True)

    control_layer: PhaseControlLayer
    hidden_layers: List[PhaseHiddenLayer]
    observation_layer: PhaseObservationLayer

    def __init__(
        self,
        control_layer_size: int,
        hidden_sizes: Sequence[int],
        obs_size: int,
        key: PRNGKeyArray,
        input_size: Optional[int] = 0,
        kappa_init: float = 1.0,
        loss: str = "mse",
    ):
        self.config = KPCHConfig(
            control_layer_size=control_layer_size,
            hidden_sizes=tuple(hidden_sizes),
            obs_size=obs_size,
            input_size=input_size,
            kappa_init=kappa_init,
            loss=loss,
        )

        n_hidden = len(hidden_sizes)
        key_control, *hidden_keys, key_obs = jr.split(key, 2 + n_hidden)

        self.control_layer = PhaseControlLayer(
            state_size=control_layer_size, input_size=input_size, kappa_init=kappa_init, key=key_control,
        )

        hidden_layers = []
        parent_size = control_layer_size
        for size, hkey in zip(hidden_sizes, hidden_keys):
            hidden_layers.append(
                PhaseHiddenLayer(state_size=size, parent_size=parent_size, kappa_init=kappa_init, key=hkey)
            )
            parent_size = size
        self.hidden_layers = hidden_layers

        self.observation_layer = PhaseObservationLayer(obs_size=obs_size, parent_size=parent_size, key=key_obs)

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
        `TpchModel.predict`/`SLPCHModel.predict`; `states_prev`/
        `states_curr` are now lists of real angle arrays, not complex.
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
        """Feedforward kick-start: seeds each layer's angle at the PHASE
        of its feedforward prediction (`jnp.angle(z_hat)`), cascading
        down exactly as `TpchModel.init_activities`/`SLPCHModel.
        init_activities` do. Using `jnp.angle` here is safe despite its
        gradient singularity at 0 -- this value is only ever used as a
        concrete numeric starting point for the iterative settle that
        follows, never differentiated through (this model, like
        `TpchModel`/`SLPCHModel`, learns via `param_grad` at the ALREADY-
        SETTLED state, not by backpropagating through settling itself).
        """
        control_pred = self.control_layer.predict(states_prev[0], control_input)
        states_curr = [jnp.angle(control_pred)]

        parent_theta = states_curr[0]
        for i, layer in enumerate(self.hidden_layers):
            own_prev = states_prev[i + 1]
            parent_prev = states_prev[i]
            prediction = layer.predict(own_prev, parent_prev, parent_theta)
            theta = jnp.angle(prediction)
            states_curr.append(theta)
            parent_theta = theta

        return states_curr

    # =========================================================================
    # Free energy
    # =========================================================================

    def _phase_energy(self, theta: Array, z_hat: Array, layer) -> Array:
        """kappa_j * (1 - cos(theta_j - phase(z_hat_j))), see module
        docstring for the angle-free derivation."""
        kappa = jax.nn.softplus(layer.kappa_raw)
        z = jnp.exp(1j * theta)
        cos_approx = jnp.real(z * jnp.conj(z_hat)) / _safe_abs(z_hat)
        return jnp.sum(kappa * (1.0 - cos_approx))

    def kpch_energy_fn(
        self,
        states_prev: Activities,
        states_curr: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        return_layerwise: bool = False,
    ) -> Array:
        """Total free energy: for every dynamical layer, the kappa-
        weighted phase-alignment term above; plus the observation term
        (`mse`/`ce`, unchanged from `TpchModel`/`SLPCHModel`). No
        amplitude term (nothing to weight it against), no regularisers
        (out of scope, per the original SL variant's scope too).

        Bounded below by construction (each `(1-cos(...))` term lies in
        `[0, 2]`, `kappa > 0`) -- unlike `slpch`'s VFE, this one can't go
        negative, and total energy IS a faithful proxy for "how well is
        inference doing" with nothing else mixed in to confound it.
        """
        predictions, y_hat = self.predict(states_prev, states_curr, control_input, observation)

        layer_energies = []
        for theta, z_hat, layer in zip(states_curr, predictions, self._dynamical_layers()):
            layer_energies.append(self._phase_energy(theta, z_hat, layer))

        if self.config.loss == "mse":
            y_error = observation - y_hat
            obs_energy = 0.5 * jnp.sum(y_error ** 2)
        else:
            obs_energy = -jnp.sum(observation * jax.nn.log_softmax(y_hat))
        layer_energies.append(obs_energy)

        if return_layerwise:
            return jnp.stack(layer_energies)
        return sum(layer_energies)

    def energy_fn(self, *args, **kwargs):
        """Alias for `kpch_energy_fn`, under the shared-runner name."""
        return self.kpch_energy_fn(*args, **kwargs)

    # =========================================================================
    # Inference -- plain real-valued gradient descent, no conjugation
    # trick needed (see module docstring): dtheta/dt = -dE/dtheta.
    # =========================================================================

    def neg_activity_grad(
        self,
        states_curr: Activities,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ) -> Activities:
        """-dE/dtheta -- ordinary negated gradient, exactly `TpchModel`'s
        form (theta is real, so no `slpch`-style conjugation is needed)."""
        energy_of_states = lambda s: self.kpch_energy_fn(states_prev, s, observation, control_input)
        return jax.tree_util.tree_map(jnp.negative, jax.grad(energy_of_states)(states_curr))

    def infer_step(
        self,
        states_curr: Activities,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        state_lr: float = 0.1,
    ) -> Activities:
        """One Euler step. No stiffness concern here (see module
        docstring), so this is safe at a wider range of `state_lr` than
        the `slpch` equivalent -- though "wider" isn't "unlimited";
        `kappa`/prediction-weight magnitudes still set an effective
        curvature this has to stay under, same as any gradient descent.
        """
        dstates = self.neg_activity_grad(states_curr, states_prev, observation, control_input)
        return jax.tree_util.tree_map(lambda s, d: s + state_lr * d, states_curr, dstates)

    def settle(
        self,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        n_steps: int = 20,
        state_lr: float = 0.1,
    ) -> Activities:
        """Plain Python-loop settling -- complex analogue of
        `TpchModel.settle`. Unlike `slpch`'s equivalent, there IS a
        genuine fixed point to arrive at here (no persistent rotation),
        so -- given enough steps/an appropriate `state_lr` -- this
        should behave like `TpchModel`'s version: converge and stay,
        rather than `slpch`'s bounded-oscillation behaviour.
        """
        states_curr = self.init_activities(states_prev, control_input, observation)
        for _ in range(n_steps):
            states_curr = self.infer_step(states_curr, states_prev, observation, control_input, state_lr)
        return states_curr

    # =========================================================================
    # Learning -- identical in form to TpchModel's; kappa is inside the
    # energy (see module docstring), so it's learned like any weight,
    # with none of slpch's omega caveat.
    # =========================================================================

    def param_grad(
        self,
        states_prev: Activities,
        states_curr: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ) -> PyTree:
        """dE/d(weights) at the settled states. Plain `eqx.filter_grad`,
        no post-hoc conjugation needed anywhere in this pytree -- every
        leaf here (including `kappa_raw`) is real."""
        energy_of_weights = lambda m: m.kpch_energy_fn(states_prev, states_curr, observation, control_input)
        return eqx.filter_grad(energy_of_weights)(self)

    def update_params(
        self,
        grads: PyTree,
        optim: optax.GradientTransformation,
        opt_state: optax.OptState,
    ) -> Tuple[eqx.Module, optax.OptState]:
        """Identical to `TpchModel.update_params`. Weight leaves here are
        complex (the `ComplexLinear`s), which brings back `slpch`'s
        adaptive-optimiser caveat for the WEIGHTS specifically (`adam`'s
        `grad**2` second-moment isn't `|grad|**2` for a complex leaf) --
        `optax.sgd` remains the safe default; this doesn't apply to
        `kappa_raw`, which is real.
        """
        updates, opt_state = optim.update(grads, opt_state, self)
        updated_model = eqx.apply_updates(self, updates)
        return updated_model, opt_state

    # =========================================================================
    # Scan-fused inference
    # =========================================================================

    def make_activity_step(
        self,
        activity_optim: optax.GradientTransformation,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ):
        """Complex analogue of `TpchModel.make_activity_step` -- but
        "complex" only in the sense of using complex WEIGHTS internally;
        the grad handed to `activity_optim` is the plain real
        `jax.grad(energy_fn)(states_curr)`, exactly `TpchModel`'s
        convention, no `slpch`-style rotation term or conjugation.
        """
        energy_fn = lambda s: self.kpch_energy_fn(states_prev, s, observation, control_input)

        def activity_step(carry, _):
            states_curr, opt_state = carry
            grads = jax.grad(energy_fn)(states_curr)
            updates, opt_state = activity_optim.update(grads, opt_state, states_curr)
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
        """Scan-fused settling -- identical structure to `TpchModel.settle_scan`."""
        states_curr0 = self.init_activities(states_prev, control_input, observation)
        opt_state0 = activity_optim.init(states_curr0)

        activity_step = self.make_activity_step(activity_optim, states_prev, observation, control_input)
        (states_curr, _), states_hist = jax.lax.scan(activity_step, (states_curr0, opt_state0), xs=None, length=n_steps)

        if not return_layerwise:
            return states_curr

        energy_trace_fn = lambda s: self.kpch_energy_fn(states_prev, s, observation, control_input, return_layerwise=True)
        energy_trace = jax.vmap(energy_trace_fn)(states_hist)
        return states_curr, energy_trace

    # =========================================================================
    # Diffrax-fused inference -- straight reuse of `..inference`, exactly
    # like `TpchModel`: no complex/real packing needed (see module
    # docstring), and the standard steady-state criteria are meaningful
    # again (there's an actual fixed point to detect convergence to),
    # so this defaults to Mode 1 (event-based), matching `TpchModel`
    # rather than `slpch`.
    # =========================================================================

    def make_vector_field(
        self,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ):
        energy_fn = lambda s: self.kpch_energy_fn(states_prev, s, observation, control_input)

        def vector_field(t, states_curr, args):
            return jax.tree_util.tree_map(jnp.negative, jax.grad(energy_fn)(states_curr))

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
        vector_field = self.make_vector_field(states_prev, observation, control_input)
        energy_fn = lambda s: self.kpch_energy_fn(states_prev, s, observation, control_input)
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
        steady_state_tol: Optional[float] = 1e-3,
        steady_state_criterion: str = "rms",
        steady_state_rtol: Optional[float] = None,
        steady_state_atol: Optional[float] = None,
        return_layerwise: bool = False,
    ) -> Union[Activities, Tuple[Activities, Array, Array]]:
        """Thin adapter over `inference.settle_diffrax`, called DIRECTLY
        with no packing step -- see module docstring for why that's safe
        here but wasn't for `slpch`. Default `steady_state_tol=1e-3`
        (Mode 1, event-based), matching `TpchModel`, not `slpch`.
        """
        states_curr0 = self.init_activities(states_prev, control_input, observation)
        vector_field = self.make_vector_field(states_prev, observation, control_input)
        energy_fn = lambda s: self.kpch_energy_fn(states_prev, s, observation, control_input)
        layerwise_energy_fn = lambda s: self.kpch_energy_fn(
            states_prev, s, observation, control_input, return_layerwise=True
        )

        return inference.settle_diffrax(
            vector_field,
            states_curr0,
            energy_fn=energy_fn,
            layerwise_energy_fn=layerwise_energy_fn,
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

    # =========================================================================
    # Saving and loading
    # =========================================================================

    @classmethod
    def from_config(cls, config: KPCHConfig, *, key) -> "KPCHModel":
        return cls(
            control_layer_size=config.control_layer_size,
            hidden_sizes=config.hidden_sizes,
            obs_size=config.obs_size,
            key=key,
            input_size=config.input_size,
            kappa_init=config.kappa_init,
            loss=config.loss,
        )

    @classmethod
    def layer_labels(cls, config: KPCHConfig) -> List[str]:
        return (
            ["Control"]
            + [f"Hidden {i + 1}" for i in range(len(config.hidden_sizes))]
            + ["Observation"]
        )

    @classmethod
    def zero_activities(cls, config: KPCHConfig) -> Activities:
        """theta=0 for every node (i.e. z=1+0j) -- an arbitrary but
        harmless "null" phase, used only as a fixed `states_prev`
        context (e.g. sequence start), never itself integrated. Real-
        valued now, not complex -- the one dtype difference from
        `SLPCHModel.zero_activities`.
        """
        sizes = [config.control_layer_size, *config.hidden_sizes]
        return [jnp.zeros(s, dtype=jnp.float32) for s in sizes]
