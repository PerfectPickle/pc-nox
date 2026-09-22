"""tpch/model.py

Equinox implementation of temporal Hierarchical Predictive Coding (tPC-H),
following eqs. (19)-(27) of the tPC-H paper:

Ng-Kee-Kwong, J., Tang, M., Akam, T., & Bogacz, R. (2026).
Learning complex temporal dependencies via local synaptic plasticity.
bioRxiv. https://doi.org/10.64898/2026.07.09.737423

--------------------------------------------------------------------------
What tPC-H is, in one paragraph
--------------------------------------------------------------------------
A standard (non-hierarchical) temporal PC network has a single chain of
latent states s_{t-1} -> s_t -> ... driven by an optional top-down external
input x_t, plus an output y_t = C @ s_t. tPC-H stacks several such chains
on top of each other -- see `layers.py` for the three layer roles.

--------------------------------------------------------------------------
How inference & learning are implemented
--------------------------------------------------------------------------
Rather than hand-coding the closed-form gradients of eqs. (20)-(27), we
write down the scalar free energy F_t (eq. 19, generalised) as a plain
JAX-differentiable function of the states and weights, and let
`jax.grad` do the differentiation:

    * grad of F_t w.r.t. the current states  ==  eqs. (20)-(21) (inference)
    * grad of F_t w.r.t. the weights         ==  eqs. (22)-(27) (learning)

Every function here operates on a *single, unbatched* time step (all
arrays are 1-D). Use `jax.vmap` over a leading batch axis, and
`jax.lax.scan` over a leading time axis, in the calling code that loops
over a batch of sequences -- or use `runners_temporal.py`, which does this
for you and is shared across every temporal model variant, not just this
one.

--------------------------------------------------------------------------
What's model-agnostic and lives elsewhere
--------------------------------------------------------------------------
- `..inference`: the diffrax integration / early-termination / save-grid
  engine `settle_diffrax` below is a thin adapter over. Doesn't know this
  is tPC-H.
- `..runners_temporal`: the eval/train step/run `jax.lax.scan` factories
  (`make_eval_step`, `make_train_run`, their diffrax analogues, etc.).
  Drives any temporal model exposing the small interface documented at
  the top of that module (`predict`, `init_activities`, `energy_fn`,
  `settle_scan`, `settle_diffrax`, `param_grad`) -- this file provides
  that interface, it doesn't hard-code being called by anything specific.
"""

import warnings
from dataclasses import asdict
from typing import Callable, List, ClassVar, Optional, Sequence, Tuple, Union

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from .. import inference, regularisers
from ..model_base import ACT_FN_REGISTRY, Activities, ModelBase, Predictions
from .config import TpchConfig, scopes_overlap
from .layers import TpchControlLayer, TpchHiddenLayer, TpchObservationLayer


# =============================================================================
# TpchModel: composes the three layer types into a full tPC-H hierarchy
# =============================================================================

class TpchModel(eqx.Module, ModelBase):
    """A full tPC-H hierarchy chained top to bottom.

    Consists of: one control layer, N >= 0 hidden layers, and one observation layer.

    Attributes:
        control_layer: The control layer module at the top of the network.
        hidden_layers: List of hidden layer modules.
        observation_layer: The observation layer module at the bottom of the network.

    Args:
        control_layer_size: Width of the top/control layer.
        hidden_sizes: Sequence of hidden layer widths, number of elements determines the number of hidden layers.
        obs_size: Width of the observation / output.
        act_fn: Name of activation function used by the control and hidden layers ('tanh').
        key: Jax pseudo random number generator key used for layer initilizations.
        input_size: Width of input provided to control layer, defaults to 0.
        loss: Loss used for the observation term of the free energy. Either
            `"mse"` (default) or `"ce"` (cross-entropy, treating the
            observation layer's linear output as logits). Only affects the
            observation term -- state-prediction terms are always squared error.
        weight_decay: Coefficient for an L2 penalty on the weights (0 =
            disabled, the default). Applied to `weight_decay_scope` weights.
        weight_decay_scope: Which weights `weight_decay` applies to: `"all"`
            (default), `"rec"` (only the recurrent, self-to-self weights,
            i.e. W_rec on the control and hidden layers), or `"ff"` (every
            weight except W_rec -- use this to keep weight_decay from
            competing with an `orthogonal_penalty` also targeting W_rec).
        orthogonal_penalty: Coefficient for an orthogonality-promoting penalty
            of the form ||I - W^T @ W||^2 on the weights (0 = disabled, the
            default). Applied to `orthogonal_scope` weights.
        orthogonal_scope: Which weights `orthogonal_penalty` applies to: `"rec"`
            (default, the only weights guaranteed square), `"all"`, or `"ff"`.
        activity_decay: Coefficient for an L1 or L2 penalty on the current
            states/activities (0 = disabled, the default).
        activity_reg_type: Either `"l1"` (default) or `"l2"`, selecting the
            form of the `activity_decay` penalty.
    """
    model_type: ClassVar[str] = "tpch"
    config_cls: ClassVar[type] = TpchConfig
    config: TpchConfig = eqx.field(static=True)

    control_layer: TpchControlLayer
    hidden_layers: List[TpchHiddenLayer]
    observation_layer: TpchObservationLayer

    def __init__(
        self,
        control_layer_size: int,
        hidden_sizes: Sequence[int],
        obs_size: int,
        key: PRNGKeyArray,
        act_fn: str = "tanh",
        input_size: Optional[int] = 0,  # optional control input
        loss: str = "mse",
        weight_decay: float = 0.0,
        weight_decay_scope: str = "all",
        orthogonal_penalty: float = 0.0,
        orthogonal_scope: str = "rec",
        activity_decay: float = 0.0,
        activity_reg_type: str = "l1",
    ):
        if loss not in ("mse", "ce"):
            raise ValueError(f"loss must be 'mse' or 'ce', got {loss!r}")
        if weight_decay_scope not in ("all", "rec", "ff"):
            raise ValueError(f"weight_decay_scope must be 'all', 'rec' or 'ff', got {weight_decay_scope!r}")
        if orthogonal_scope not in ("all", "rec", "ff"):
            raise ValueError(f"orthogonal_scope must be 'all', 'rec' or 'ff', got {orthogonal_scope!r}")
        if activity_reg_type not in ("l1", "l2"):
            raise ValueError(f"activity_reg_type must be 'l1' or 'l2', got {activity_reg_type!r}")

        # If weight_decay and orthogonal_penalty can touch the same weights,
        # they pull those weights' singular values in opposite directions
        # (decay -> 0, orthogonal -> 1). See the equilibrium-singular-value
        # derivation this warning is based on: sigma* = sqrt(1 -
        # weight_decay / (2 * orthogonal_penalty)) only exists when
        # weight_decay < 2 * orthogonal_penalty.
        if (
            weight_decay > 0. and orthogonal_penalty > 0.
            and scopes_overlap(weight_decay_scope, orthogonal_scope)
            and weight_decay >= 2 * orthogonal_penalty
        ):
            warnings.warn(
                f"weight_decay ({weight_decay}) >= 2 * orthogonal_penalty ({orthogonal_penalty}) "
                f"on overlapping scopes (weight_decay_scope={weight_decay_scope!r}, "
                f"orthogonal_scope={orthogonal_scope!r}): the two regularisers have no stable "
                "nonzero equilibrium on the shared weights and will tend to collapse them toward "
                "zero. Lower weight_decay, raise orthogonal_penalty, or use non-overlapping "
                "scopes (e.g. weight_decay_scope='ff' with orthogonal_scope='rec').",
                stacklevel=2,
            )

        self.config = TpchConfig(
            control_layer_size=control_layer_size,
            hidden_sizes=tuple(hidden_sizes),  # cast hidden_sizes to tuple to ensure config hashability
            obs_size=obs_size,
            input_size=input_size,
            act_fn=act_fn,
            loss=loss,
            weight_decay=weight_decay,
            weight_decay_scope=weight_decay_scope,
            orthogonal_penalty=orthogonal_penalty,
            orthogonal_scope=orthogonal_scope,
            activity_decay=activity_decay,
            activity_reg_type=activity_reg_type,
        )

        try:
            act_fn_callable = ACT_FN_REGISTRY[act_fn]  # Get Callable act_fn. This is the ONE place this str -> callable lookup happens.
        except KeyError:
            raise KeyError(
                f"act_fn={act_fn!r} not in ACT_FN_REGISTRY. If this is a custom "
                f"activation, register it before loading: ACT_FN_REGISTRY[{act_fn!r}] = ..."
            ) from None

        n_hidden = len(hidden_sizes)
        key_control, *hidden_keys, key_obs = jr.split(key, 2 + n_hidden)

        self.control_layer = TpchControlLayer(
            state_size=control_layer_size, input_size=input_size, act_fn=act_fn_callable, key=key_control
        )

        hidden_layers = []
        parent_size = control_layer_size  # the first hidden layer's parent is the control layer
        for size, hkey in zip(hidden_sizes, hidden_keys):
            hidden_layers.append(
                TpchHiddenLayer(state_size=size, parent_size=parent_size, act_fn=act_fn_callable, key=hkey)
            )
            parent_size = size  # each subsequent hidden layer's parent is the one above it
        self.hidden_layers = hidden_layers

        # the observation layer's parent is the lowest hidden layer (or the
        # control layer itself, if there are no hidden layers at all)
        self.observation_layer = TpchObservationLayer(obs_size=obs_size, parent_size=parent_size, key=key_obs)

    def predict(
        self,
        states_prev: Activities,
        states_curr: Activities,
        control_input: Optional[Array] = None,
        observation: Optional[Array] = None,
    ) -> Tuple[Predictions, Array]:
        """Run every layer's `predict` once, given:
          - states_prev: every layer's state at t-1 (fixed, "memory")
          - states_curr: every layer's CURRENT guess for its state at t (this
            is what inference iteratively refines -- see `settle` below)
          - control_input: x_t
          - observation: unused by tPC-H itself, accepted (and ignored) so
            every temporal model variant's `predict` shares one calling
            convention -- see `runners_temporal.py`'s module docstring.
            A variant with a discriminative/backward pathway (e.g.
            bidirectional tPC-H) uses this argument for real.

        Returns (predictions, y_hat) where `predictions` is a list aligned
        with `states_curr`, i.e. predictions[i] is what layer i "expected"
        its own states_curr[i] to be.
        """
        predictions = [self.control_layer.predict(states_prev[0], control_input)]

        for i, layer in enumerate(self.hidden_layers):
            own_prev = states_prev[i + 1]  # to account for control layer state at index 0
            parent_prev = states_prev[i]  # layer above, one step ago
            parent_curr = states_curr[i]  # layer above, right now
            predictions.append(layer.predict(own_prev, parent_prev, parent_curr))

        y_hat = self.observation_layer.predict(states_curr[-1])
        return predictions, y_hat

    def init_activities(
        self,
        states_prev: Activities,
        control_input: Optional[Array] = None,
        observation: Optional[Array] = None,
    ) -> Activities:
        """Feedforward "kick-start" pass: produces an initial prediction for every
        layer's current-time-step state using only states_prev and
        control_input, sweeping top-to-bottom as every non-control layer depends on its parent.
        `settle` (below) then refines this prediction by descending the free energy.

        `observation`: unused by tPC-H itself, accepted (and ignored) for
        the same cross-variant calling-convention reason as `predict`'s.
        """
        control_pred = self.control_layer.predict(states_prev[0], control_input)
        states_curr = [control_pred]

        parent_pred = control_pred
        for i, layer in enumerate(self.hidden_layers):
            own_prev = states_prev[i + 1]  # to account for control layer state at index 0
            parent_prev = states_prev[i]
            prediction = layer.predict(own_prev, parent_prev, parent_pred)
            states_curr.append(prediction)
            parent_pred = prediction

        return states_curr

    # =========================================================================
    # Weight enumeration / regularisation
    # =========================================================================

    def _rec_weights(self) -> List[Array]:
        """The recurrent, self-to-self weights: control_layer.W_rec and every
        hidden layer's W_rec. These are the only weights guaranteed to be
        square (state_size x state_size), which makes them the natural
        default target for the orthogonality/spectral penalty below.
        """
        weights = [self.control_layer.W_rec.weight]
        weights += [layer.W_rec.weight for layer in self.hidden_layers]
        return weights

    # feedforward weights only
    def _ff_weights(self) -> List[Array]:
        """Every weight EXCEPT the recurrent ones: W_in (if present), each
        hidden layer's W_parent_prev/W_parent_curr, and the observation
        layer's W_parent. Useful when you want weight_decay to leave the
        recurrent weights alone entirely -- e.g. so it can't fight an
        orthogonal_penalty also targeting W_rec regardless of how the two
        coefficients are tuned.
        """
        weights = []
        if self.control_layer.has_input:
            weights.append(self.control_layer.W_in.weight)
        for layer in self.hidden_layers:
            weights += [layer.W_parent_prev.weight, layer.W_parent_curr.weight]
        weights.append(self.observation_layer.W_parent.weight)
        return weights

    def _all_weights(self) -> List[Array]:
        """Every weight matrix in the model (recurrent + feedforward/emission)."""
        weights = [self.control_layer.W_rec.weight]
        if self.control_layer.has_input:
            weights.append(self.control_layer.W_in.weight)
        for layer in self.hidden_layers:
            weights += [
                layer.W_rec.weight,
                layer.W_parent_prev.weight,
                layer.W_parent_curr.weight,
            ]
        weights.append(self.observation_layer.W_parent.weight)
        return weights

    def _weights_for_scope(self, scope: str) -> List[Array]:
        """Resolves a scope string ('all' | 'rec' | 'ff') to the weight
        arrays it refers to. Shared by `_weight_l2_reg` and `_weight_orthogonal_reg`
        so both regularisers pick their targets the same way.
        """
        if scope == "rec":
            return self._rec_weights()
        elif scope == "ff":
            return self._ff_weights()
        else:  # "all", validated in __init__
            return self._all_weights()

    # extra energy (penalty) from L2 weight decay / regularisation.
    # The arithmetic itself (`regularisers.l2_reg` etc.) is generic --
    # every variant wants the same Frobenius-norm / Gram-matrix / L1-L2
    # math. Only *which weights are selected* (`_weights_for_scope`,
    # above) is tPC-H-specific, and stays here.
    def _weight_l2_reg(self) -> Array:
        """0.5 * weight_decay * sum ||W||_F^2 (squared Frobenius norm) over
        `weight_decay_scope` weights."""
        weights = self._weights_for_scope(self.config.weight_decay_scope)
        return regularisers.l2_reg(weights, self.config.weight_decay)

    def _weight_orthogonal_reg(self) -> Array:
        """0.5 * orthogonal_penalty * sum ||I - W^T W||_F^2 (or W W^T for a
        'wide' W) over `orthogonal_scope` weights -- an orthogonality penalty
        that discourages the corresponding linear map from expanding or
        contracting its input, which is particularly useful on recurrent
        weights to keep the state dynamics well-conditioned over time.
        """
        weights = self._weights_for_scope(self.config.orthogonal_scope)
        return regularisers.orthogonal_reg(weights, self.config.orthogonal_penalty)

    def _activity_reg(self, states_curr: Activities) -> Array:
        """0.5 * activity_decay * sum ||z||_p^p over every current state,
        with p=1 (sparsity-promoting) or p=2 (soft bound on activity norm)
        depending on `activity_reg_type`.
        """
        return regularisers.activity_reg(states_curr, self.config.activity_decay, self.config.activity_reg_type)

    # -------------------------------------------------------------------
    # Layerwise regularisation (for `return_layerwise=True`) -- additive to
    # everything above; `_weight_l2_reg`/`_weight_orthogonal_reg`/
    # `_activity_reg` (the scalar totals used by the default, cached
    # `weight_reg_total` path) are untouched, deliberately, since they're
    # covered by strict atol=0.0 tests and float summation order matters
    # for those.
    # -------------------------------------------------------------------

    def _weight_entries(self) -> List[Tuple[int, bool, Array]]:
        """(owning_layer_idx, is_recurrent, weight) for every weight matrix
        in the model. owning_layer_idx matches `layer_labels()`'s order:
        0 = control, 1..num_hidden = hidden layers top-to-bottom,
        num_hidden+1 = observation. Single source of truth for the
        layerwise regularisers below, so "which weights count as
        rec/ff" can't drift from `_rec_weights`/`_ff_weights`/`_all_weights`
        (which remain independent and untouched -- this doesn't replace them).
        """
        entries = [(0, True, self.control_layer.W_rec.weight)]
        if self.control_layer.has_input:
            entries.append((0, False, self.control_layer.W_in.weight))
        for i, layer in enumerate(self.hidden_layers, start=1):
            entries += [
                (i, True, layer.W_rec.weight),
                (i, False, layer.W_parent_prev.weight),
                (i, False, layer.W_parent_curr.weight),
            ]
        entries.append((len(self.hidden_layers) + 1, False, self.observation_layer.W_parent.weight))
        return entries

    def _weight_groups_for_scope(self, scope: str) -> List[List[Array]]:
        """Weights selected by `scope` ('all' | 'rec' | 'ff'), grouped by
        owning layer -- one sub-list per `layer_labels()` entry, possibly
        empty. Same selection semantics as `_weights_for_scope`, just
        partitioned instead of flattened.
        """
        num_groups = len(self.hidden_layers) + 2
        groups = [[] for _ in range(num_groups)]
        for layer_idx, is_rec, weight in self._weight_entries():
            selected = scope == "all" or (scope == "rec" and is_rec) or (scope == "ff" and not is_rec)
            if selected:
                groups[layer_idx].append(weight)
        return groups

    def _weight_l2_reg_by_layer(self) -> Array:
        """Per-layer breakdown of `_weight_l2_reg()`: shape (num_hidden+2,),
        same order as `layer_labels()`. `jnp.sum(...)` of this exactly
        equals `_weight_l2_reg()`'s scalar (same terms, just partitioned).
        """
        groups = self._weight_groups_for_scope(self.config.weight_decay_scope)
        return regularisers.l2_reg_by_group(groups, self.config.weight_decay)

    def _weight_orthogonal_reg_by_layer(self) -> Array:
        """Per-layer breakdown of `_weight_orthogonal_reg()`, same Gram-matrix
        convention (narrow-side identity) applied per matrix, grouped by
        owning layer instead of summed across the whole scope.
        """
        groups = self._weight_groups_for_scope(self.config.orthogonal_scope)
        return regularisers.orthogonal_reg_by_group(groups, self.config.orthogonal_penalty)

    def _activity_reg_by_layer(self, states_curr: Activities) -> Array:
        """Per-layer breakdown of the activity-regularisation term: shape
        (len(states_curr),) -- NOT padded to num_hidden+2, since activity
        regularisation has no observation-layer term. Callers combining
        this with the weight-reg breakdowns above must pad with one
        trailing zero themselves (see `tpch_energy_fn`).
        """
        return regularisers.activity_reg_by_layer(states_curr, self.config.activity_decay, self.config.activity_reg_type)

    # =========================================================================
    # Free energy -- eq. (19), generalised to an arbitrary number of layers
    # =========================================================================

    def tpch_energy_fn(
        self,
        states_prev: Activities,
        states_curr: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        weight_reg_total: Optional[Array] = None,
        return_layerwise: bool = False,
    ) -> Array:
        """
        Sum over every layer's energy, calculated with prediction errors.

        F_t = sum over every layer of 1/2 ||actual state - predicted state||^2,
        plus the observation term 1/2 ||y_t - y_hat_t||^2 for `loss="mse"`,
        or the cross-entropy of `y_t` under logits `y_hat_t` for
        `loss="ce"`), plus any weight/orthogonal/activity regularisation
        configured on `self.config`.

        This is exactly eq. (19), just written for however many layers `model`
        happens to have instead of being hard-coded to the 2-layer (s, z) case.

        `weight_reg_total`: internal-use only. `_weight_l2_reg() + _weight_orthogonal_reg()`
        is a pure function of `self`'s weights -- it cannot change while
        `states_curr` is being relaxed towards a fixed point during
        inference, since weights are frozen until `update_params` runs
        between time steps. Callers doing repeated inference steps (e.g.
        `settle`, `make_activity_step`) compute it ONCE and pass it in here
        to avoid recomputing the same matrix products on every one of the
        `n_steps` relaxation iterations. Leave as `None` (the default) to
        recompute fresh from `self` -- this is what `param_grad` and any
        one-off energy readout should do.

        `return_layerwise`: if True, returns an unsummed Array of per-layer
        energies instead of the total scalar -- one entry per element of
        `states_curr` (top-to-bottom: control, then each hidden layer),
        followed by one final entry for the observation term. Order matches
        `TpchModel.layer_labels()`. `jnp.sum(...)` of this array exactly
        equals the summed-scalar output (`return_layerwise=False`) on the
        same inputs.

        Args:
            states_prev: Activities from previous state.
            states_curr: Activities from current state.
            observation: Current observation.
            control_input: Optional control layer input.
            weight_reg_total: Precalculated regularisation penalty to be added, else calculates weight reg automatically.
            return_layerwise: Whether or not to return jnp stack of per layer energies, instead of the default total energy sum.
        """
        predictions, y_hat = self.predict(states_prev, states_curr, control_input, observation)

        layer_energies = []
        for state, prediction in zip(states_curr, predictions):
            error = state - prediction
            layer_energies.append(0.5 * jnp.sum(error ** 2))

        if self.config.loss == "mse":
            y_error = observation - y_hat
            obs_energy = 0.5 * jnp.sum(y_error ** 2)
        else:  # "ce", validated in __init__
            obs_energy = -jnp.sum(observation * jax.nn.log_softmax(y_hat))
        layer_energies.append(obs_energy)

        if return_layerwise:
            weight_l2_by_layer = self._weight_l2_reg_by_layer()
            weight_orth_by_layer = self._weight_orthogonal_reg_by_layer()
            activity_by_layer = self._activity_reg_by_layer(states_curr)
            # activity has no observation-layer term (states_curr never
            # includes one) -- pad with one trailing zero to line up with
            # the weight-reg breakdowns and layer_energies, which both do.
            activity_by_layer = jnp.concatenate([activity_by_layer, jnp.zeros((1,), dtype=activity_by_layer.dtype)])
            return jnp.stack(layer_energies) + weight_l2_by_layer + weight_orth_by_layer + activity_by_layer

        if weight_reg_total is None:
            weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()

        return sum(layer_energies) + weight_reg_total + self._activity_reg(states_curr)

    def energy_fn(self, *args, **kwargs):
        """Alias for `tpch_energy_fn`, under the name shared runners (see
        `runners_temporal.py`) call so they can drive any temporal model
        variant without knowing its specific energy function's name. A
        sibling variant is free to keep its own descriptively-named energy
        method (e.g. `bpc_energy_fn`) as the "real" one and just alias it
        the same way -- the name `tpch_energy_fn` itself isn't special,
        only `energy_fn` is, and only for models that want to plug into
        the shared runners.
        """
        return self.tpch_energy_fn(*args, **kwargs)

    # =========================================================================
    # Inference -- discretised eqs. (20)-(21): settle the current states by
    # gradient-descending the free energy while holding weights fixed
    # =========================================================================

    def neg_activity_grad(
        self,
        states_curr: Activities,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        weight_reg_total: Optional[Array] = None,
    ) -> Activities:
        """-dF_t/d(states_curr), i.e. the direction each state should move in to
        reduce the free energy. This is the generalised form of eqs. (20)-(21).

        `weight_reg_total`: see the docstring on `tpch_energy_fn` -- passed
        straight through so repeated calls (from `settle`) don't recompute
        the weight/orthogonal regularisation on every inference step.
        """
        energy_of_states = lambda s: self.tpch_energy_fn(
            states_prev, s, observation, control_input, weight_reg_total=weight_reg_total
        )
        return jax.tree_util.tree_map(jnp.negative, jax.grad(energy_of_states)(states_curr))

    def infer_step(
        self,
        states_curr: Activities,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        state_lr: float = 0.1,
        weight_reg_total: Optional[Array] = None,
    ) -> Activities:
        """One Euler step of the continuous-time inference dynamics in eqs.
        (20)-(21): states_curr <- states_curr + state_lr * (-dF_t/d(states_curr)).
        `state_lr` plays the role of the (dt / tau) discretisation step.

        `weight_reg_total` is the weight regularisation penalty held static during state settling.
        """
        grad_step = self.neg_activity_grad(
            states_curr, states_prev, observation, control_input, weight_reg_total=weight_reg_total
        )
        return jax.tree_util.tree_map(lambda s, g: s + state_lr * g, states_curr, grad_step)

    def settle(
        self,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        n_steps: int = 20,
        state_lr: float = 0.1,
    ) -> Activities:
        """Full per-time-step inference: start from the feedforward guess
        (`init_activities`) and take `n_steps` of gradient-descent inference to
        let the states relax towards a local minimum of F_t before learning.
        """
        weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()

        states_curr = self.init_activities(states_prev, control_input, observation)
        for _ in range(n_steps):
            states_curr = self.infer_step(
                states_curr, states_prev, observation, control_input, state_lr, weight_reg_total=weight_reg_total
            )
        return states_curr

    # =========================================================================
    # Learning -- eqs. (22)-(27): local, Hebbian weight updates at the
    # settled states
    # =========================================================================

    def param_grad(
        self,
        states_prev: Activities,
        states_curr: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ) -> PyTree:
        """dF_t/d(weights), evaluated at the settled states. Because F_t is a sum
        of per-layer squared-error terms and each weight matrix appears in
        exactly one of those terms, this reproduces eqs. (22)-(27) exactly,
        without us ever writing f' by hand.

        Uses `eqx.filter_grad` (rather than plain `jax.grad`) so that the
        non-array fields on the layers, like `act_fn`, are safely ignored.
        """
        energy_of_weights = lambda m: m.tpch_energy_fn(states_prev, states_curr, observation, control_input)
        return eqx.filter_grad(energy_of_weights)(self)

    def update_params(
        self,
        grads: PyTree,
        optim: optax.GradientTransformation,
        opt_state: optax.OptState,
    ) -> Tuple[eqx.Module, optax.OptState]:
        """Optax-driven weight update. `optim` can be optax.sgd(lr) to recover
        plain gradient descent, or optax.adam(lr), etc. `opt_state` must be
        carried forward by the caller between steps.
        """
        updates, opt_state = optim.update(grads, opt_state, self)
        updated_model = eqx.apply_updates(self, updates)
        return updated_model, opt_state

    # =========================================================================
    # Scan-fused inference, for performance
    # =========================================================================
    #
    # Kept per-model rather than in `inference.py`: `make_activity_step` +
    # `settle_scan` is ~50 lines, cheap to duplicate, and different models
    # may reasonably want a different scan body (e.g. one that fixes some
    # activities during inference). `settle_diffrax`, below, is the one
    # that's factored out -- see that method's docstring for why.

    def make_activity_step(
        self,
        activity_optim: optax.GradientTransformation,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ):
        """Builds a scan body that runs ONE inference-relaxation update at a
        single, fixed time step, using `tpch_energy_fn` (eq. 19) in place of
        a generic `pc_energy_fn`.

        carry: (states_curr, activity_opt_state)
        scan output: states_curr at every step, so you can inspect the full
            relaxation trajectory (e.g. to check/plot convergence) if you want.
        """
        weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        energy_fn = lambda s: self.tpch_energy_fn(
            states_prev, s, observation, control_input, weight_reg_total=weight_reg_total
        )

        def activity_step(carry, _):
            states_curr, opt_state = carry
            grads = jax.grad(energy_fn)(states_curr)  # positive grad -- see eqs. (20)-(21)
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
        """Scan-fused equivalent of `settle`: same feedforward init, but the
        relaxation loop runs as a single `jax.lax.scan` instead of a Python
        `for` loop.

        `return_layerwise`: if True, the recorded trajectory is additionally
        run back through `tpch_energy_fn(..., return_layerwise=True)` via
        `vmap`, returning `(states_curr, energy_trace)` instead, with
        `energy_trace` shape (n_steps, num_layers+1).
        """
        states_curr0 = self.init_activities(states_prev, control_input, observation)
        opt_state0 = activity_optim.init(states_curr0)

        activity_step = self.make_activity_step(activity_optim, states_prev, observation, control_input)
        (states_curr, _), states_hist = jax.lax.scan(activity_step, (states_curr0, opt_state0), xs=None, length=n_steps)

        if not return_layerwise:
            return states_curr

        energy_trace_fn = lambda s: self.tpch_energy_fn(
            states_prev, s, observation, control_input, return_layerwise=True
        )
        energy_trace = jax.vmap(energy_trace_fn)(states_hist)  # (n_steps, num_layers + 1)
        return states_curr, energy_trace

    def make_tpch_sequence_step(
        self,
        activity_optim: optax.GradientTransformation,
        n_infer_steps: int = 20,
    ):
        """Builds a scan body that processes ONE time step of a sequence: settles
        that step's activities (itself a fused inner scan, via `settle_scan`)
        and hands the result on as `states_prev` for the next time step.
        """
        @eqx.filter_jit
        def sequence_step(states_prev, xy_t):
            control_input_t, observation_t = xy_t
            states_curr = self.settle_scan(
                activity_optim, states_prev, observation_t, control_input_t, n_steps=n_infer_steps
            )
            energy_t = self.tpch_energy_fn(states_prev, states_curr, observation_t, control_input_t)
            return states_curr, (states_curr, energy_t)

        return sequence_step

    # =========================================================================
    # Diffrax-fused inference -- thin adapters over the model-agnostic
    # engine in `..inference`. tPC-H builds the closures (which know about
    # `tpch_energy_fn`); `..inference` runs the integration (which doesn't
    # know or care what the energy function computes).
    # =========================================================================

    def make_vector_field(
        self,
        states_prev: "Activities",
        observation: "Array",
        control_input: Optional["Array"] = None,
    ):
        """Builds the ODE vector field for one inference-relaxation trajectory.

        Diffrax analogue of `make_activity_step`: instead of returning a
        function that performs ONE discrete gradient-descent update, this
        returns the continuous-time vector field `ds/dt = -dE/ds` itself,
        which `diffrax.diffeqsolve` then integrates. Uses `tpch_energy_fn`
        (eq. 19) exactly as `make_activity_step` does, including freezing
        and closing over the weight/orthogonal regularisation once per
        trajectory rather than recomputing it at every solver step/stage.
        """
        weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        energy_fn = lambda s: self.tpch_energy_fn(
            states_prev, s, observation, control_input, weight_reg_total=weight_reg_total
        )

        def vector_field(t, states_curr, args):
            grads = jax.grad(energy_fn)(states_curr)  # positive grad -- see eqs. (20)-(21)
            return jax.tree_util.tree_map(lambda g: -g, grads)

        return vector_field

    def make_steady_state_event(
        self,
        states_prev: "Activities",
        observation: "Array",
        control_input: Optional["Array"] = None,
        tol: Optional[float] = 1e-3,
        criterion: str = "rms",
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> diffrax.Event:
        """tPC-H adapter over `inference.make_steady_state_event` -- builds
        this trajectory's vector field and (for `criterion="energy_rate"`)
        energy closure, then delegates. See `inference.make_steady_state_event`
        for what each `criterion` measures and the full Args/Returns.
        """
        vector_field = self.make_vector_field(states_prev, observation, control_input)
        energy_fn = lambda s: self.tpch_energy_fn(states_prev, s, observation, control_input)
        return inference.make_steady_state_event(
            vector_field, tol=tol, criterion=criterion, rtol=rtol, atol=atol, energy_fn=energy_fn,
        )

    def settle_diffrax(
        self,
        states_prev: "Activities",
        observation: "Array",
        control_input: Optional["Array"] = None,
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
    ) -> Union["Activities", Tuple["Activities", "Array", "Array"]]:
        """tPC-H adapter over `inference.settle_diffrax`: builds the
        feedforward init and the trajectory's closures (vector field, plain
        energy, layerwise energy), then delegates the actual integration.
        Behaviour, args and returns are unchanged from before this file was
        split -- see `inference.settle_diffrax`'s docstring for the full
        description (Mode 1 / Mode 2, `return_layerwise`'s inf-padding
        caveat, etc.), which now lives there since it no longer describes
        anything tPC-H-specific.
        """
        states_curr0 = self.init_activities(states_prev, control_input, observation)
        vector_field = self.make_vector_field(states_prev, observation, control_input)
        energy_fn = lambda s: self.tpch_energy_fn(states_prev, s, observation, control_input)
        layerwise_energy_fn = lambda s: self.tpch_energy_fn(
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
    def from_config(cls, config, *, key) -> "TpchModel":
        """
        (Re)build model from config
        """
        return cls(
            control_layer_size=config.control_layer_size,
            hidden_sizes=config.hidden_sizes,
            obs_size=config.obs_size,
            key=key,
            act_fn=config.act_fn,  # plaintext name of act_fn, used with registry
            input_size=config.input_size,
            loss=config.loss,
            weight_decay=config.weight_decay,
            weight_decay_scope=config.weight_decay_scope,
            orthogonal_penalty=config.orthogonal_penalty,
            orthogonal_scope=config.orthogonal_scope,
            activity_decay=config.activity_decay,
            activity_reg_type=config.activity_reg_type,
        )

    @classmethod
    def layer_labels(cls, config) -> List[str]:
        """
        Labels matching `tpch_energy_fn(..., return_layerwise=True)`'s output
        order: control layer, then each hidden layer top-to-bottom, then
        the observation/output term.
        """
        return (
            ["Control"]
            + [f"Hidden {i + 1}" for i in range(len(config.hidden_sizes))]
            + ["Observation"]
        )

    @classmethod
    def zero_activities(cls, config: TpchConfig) -> Activities:
        """
        Builds activities skeleton for TpchModel loading.
        """
        sizes = [config.control_layer_size, *config.hidden_sizes]
        return [jnp.zeros(s) for s in sizes]
