"""pcn/model.py

Static (non-temporal) hierarchical PCN, in the same shape as
`tpch/model.py` but with the temporal axis removed entirely -- there is
no `states_prev`, no recurrent weights, no `control_input`; every
quantity in the model is "now". Follows Algorithm 1's structure directly
(input z_1 = x clamped, output z_L = y clamped, interior z_2..z_{L-1}
free latents settled by inference), minus the Meta-PE objective,
frozen-prediction trick, and weight normalisation specific to Meta-PCN --
this is the plain PC baseline `PcnConfig`/`PcnModel` mean to be an
EXTENSION POINT for, not Meta-PCN itself.

--------------------------------------------------------------------------
How a Meta-PCN variant would extend this
--------------------------------------------------------------------------
Same relationship as `tpch_bi` to `tpch`: copy this package as a sibling
(`pcn_meta/`), not a subclass -- concrete variants that change energy/weights/settling wholesale get
little from inheriting a concrete `eqx.Module`, and real equinox/MRO
pitfalls). What would change, concretely:

  - `init_activities` (below) already computes exactly `c_l` (Algorithm 1
    lines 5-9) as its feedforward pass -- Meta-PCN's frozen predictions
    are these values, held fixed for the whole inference loop rather than
    recomputed each iteration.
  - `pcn_energy_fn` here is used for BOTH inference (`settle`/`settle_scan`)
    and learning (`param_grad`) -- one scalar, same as tPC-H. Meta-PCN
    needs two: a meta-PE objective J (with `jax.lax.stop_gradient` on the
    incoming delta_{l+1} term) for inference, and a plain sum-of-squared-
    settled-errors L(theta) for learning. Two internal methods instead of
    one `energy_fn`, wired the same way into `settle_scan`/`param_grad`.
  - Weight normalisation (Algorithm 1 lines 21-24) is a post-update
    transform, not a gradient penalty -- use `ModelBase.postprocess_params`
    (see `model_base.py`), not a new energy term.

--------------------------------------------------------------------------
What's shared with the temporal side, and what isn't
--------------------------------------------------------------------------
`regularisers.py`'s L2/orthogonal/activity math is reused as-is (weight
selection is trivial here -- there's no rec/ff distinction, so `_all_weights()`
is the only scope). `runners_static.py` (batch via `jax.vmap`, not `jax.lax.scan`
across time -- see that module's docstring) is this model's counterpart
to `runners_temporal.py`, and is NOT shared with it: there's no
`states_prev` for a batch of independent examples to carry, so there's
nothing for a `lax.scan`-based runner to do here that `vmap` doesn't do
more naturally.
"""

from dataclasses import asdict
from typing import ClassVar, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from .. import regularisers
from ..model_base import ACT_FN_REGISTRY, Activities, ModelBase, Predictions
from .config import PcnConfig
from .layers import PcnLayer


class PcnModel(eqx.Module, ModelBase):
    """A static hierarchical PCN: z_1 (=x, clamped) -> z_2 -> ... ->
    z_{L-1} (free latents) -> z_L (=y, clamped).

    Attributes:
        layers: L-1 `PcnLayer`s, layers[i] predicting z_{i+2} from z_{i+1}
            (layers[0] predicts z_2 from z_1=x; layers[-1] predicts z_L
            from z_{L-1}, compared against y).

    Args:
        layer_sizes: (input_size, hidden_1, ..., hidden_k, output_size).
        act_fn: Name of activation function, applied at every layer
            (including the output readout, matching Algorithm 1 line 8 --
            unlike tPC-H's purely-linear observation layer).
        loss: Output-layer loss, `"mse"` (default) or `"ce"`.
        weight_decay, orthogonal_penalty, activity_decay, activity_reg_type:
            same meaning as the like-named `TpchConfig` fields -- see
            `regularisers.py`. No `_scope` variants: there's no rec/ff
            distinction in a static PCN, every penalty applies to every
            weight/state uniformly.
    """
    model_type: ClassVar[str] = "pcn"
    config_cls: ClassVar[type] = PcnConfig
    config: PcnConfig = eqx.field(static=True)

    layers: List[PcnLayer]

    def __init__(
        self,
        layer_sizes: Sequence[int],
        key: PRNGKeyArray,
        act_fn: str = "tanh",
        loss: str = "mse",
        weight_decay: float = 0.0,
        orthogonal_penalty: float = 0.0,
        activity_decay: float = 0.0,
        activity_reg_type: str = "l1",
    ):
        if len(layer_sizes) < 2:
            raise ValueError(f"layer_sizes needs at least 2 entries (input, output), got {layer_sizes!r}")
        if loss not in ("mse", "ce"):
            raise ValueError(f"loss must be 'mse' or 'ce', got {loss!r}")
        if activity_reg_type not in ("l1", "l2"):
            raise ValueError(f"activity_reg_type must be 'l1' or 'l2', got {activity_reg_type!r}")

        self.config = PcnConfig(
            layer_sizes=tuple(layer_sizes),
            act_fn=act_fn,
            loss=loss,
            weight_decay=weight_decay,
            orthogonal_penalty=orthogonal_penalty,
            activity_decay=activity_decay,
            activity_reg_type=activity_reg_type,
        )

        try:
            act_fn_callable = ACT_FN_REGISTRY[act_fn]
        except KeyError:
            raise KeyError(
                f"act_fn={act_fn!r} not in ACT_FN_REGISTRY. If this is a custom "
                f"activation, register it before loading: ACT_FN_REGISTRY[{act_fn!r}] = ..."
            ) from None

        n_layers = len(layer_sizes) - 1
        keys = jr.split(key, n_layers)
        self.layers = [
            PcnLayer(parent_size=layer_sizes[i], own_size=layer_sizes[i + 1], act_fn=act_fn_callable, key=k)
            for i, k in enumerate(keys)
        ]

    # =========================================================================
    # Predict / init -- states_curr here means the INTERIOR latents only,
    # z_2..z_{L-1}; x (=z_1) and y (=z_L) are passed separately, exactly
    # like tPC-H passes control_input/observation alongside states_prev.
    # =========================================================================

    def predict(self, states_curr: Activities, x: Array) -> Tuple[Predictions, Array]:
        """Run every layer's `predict` once. `states_curr` is the current
        guess for z_2..z_{L-1} (len = len(self.layers) - 1); `x` is z_1.

        Returns (predictions, y_hat): predictions[i] is what layer i
        expected z_{i+2} to be, aligned with `states_curr` (so
        predictions[:-1] compares to states_curr, mirroring tPC-H's
        `predict`); y_hat = predictions[-1] is the final layer's
        prediction of y.
        """
        chain_in = [x] + list(states_curr)  # z_1..z_{L-1}
        predictions = [layer.predict(z) for layer, z in zip(self.layers, chain_in)]  # -> z_2..z_L
        y_hat = predictions[-1]
        return predictions, y_hat

    def init_activities(self, x: Array) -> Activities:
        """Feedforward pass: z_2..z_{L-1} computed by sweeping the input
        through every layer except the last (the last layer's prediction
        is y_hat, not a free latent). Exactly Algorithm 1 lines 4-8's `c_l`
        for l=2..L-1 -- see this module's docstring for how a Meta-PCN
        variant would freeze exactly these values.
        """
        states_curr = []
        z = x
        for layer in self.layers[:-1]:
            z = layer.predict(z)
            states_curr.append(z)
        return states_curr

    # =========================================================================
    # Weight enumeration / regularisation -- no rec/ff distinction (no
    # recurrent weights at all), so `regularisers.py`'s math is applied to
    # every weight uniformly.
    # =========================================================================

    def _all_weights(self) -> List[Array]:
        return [layer.weight for layer in self.layers]

    def _weight_l2_reg(self) -> Array:
        return regularisers.l2_reg(self._all_weights(), self.config.weight_decay)

    def _weight_orthogonal_reg(self) -> Array:
        return regularisers.orthogonal_reg(self._all_weights(), self.config.orthogonal_penalty)

    def _activity_reg(self, states_curr: Activities) -> Array:
        return regularisers.activity_reg(states_curr, self.config.activity_decay, self.config.activity_reg_type)

    # =========================================================================
    # Free energy
    # =========================================================================

    def pcn_energy_fn(
        self,
        states_curr: Activities,
        x: Array,
        y: Array,
        weight_reg_total: Optional[Array] = None,
        return_layerwise: bool = False,
    ) -> Array:
        """Sum over every interior layer's squared prediction error, plus
        the output term (mse or ce against y), plus configured
        regularisation. Same shape as `tpch_energy_fn`, minus the temporal
        terms.
        """
        predictions, y_hat = self.predict(states_curr, x)
        targets = list(states_curr) + [y]  # z_2..z_{L-1}, y  (aligned with `predictions`)

        layer_energies = []
        for target, pred in zip(targets[:-1], predictions[:-1]):
            layer_energies.append(0.5 * jnp.sum((target - pred) ** 2))

        if self.config.loss == "mse":
            obs_energy = 0.5 * jnp.sum((y - y_hat) ** 2)
        else:  # "ce"
            obs_energy = -jnp.sum(y * jax.nn.log_softmax(y_hat))
        layer_energies.append(obs_energy)

        if return_layerwise:
            return jnp.stack(layer_energies)  # regularisation NOT broken out per-layer here (see TpchModel for that pattern if needed)

        if weight_reg_total is None:
            weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        return sum(layer_energies) + weight_reg_total + self._activity_reg(states_curr)

    def energy_fn(self, *args, **kwargs):
        """Alias for `pcn_energy_fn` -- see `TpchModel.energy_fn`'s
        docstring for why this name exists (shared-runner convention)."""
        return self.pcn_energy_fn(*args, **kwargs)

    # =========================================================================
    # Inference
    # =========================================================================

    def neg_activity_grad(self, states_curr, x, y, weight_reg_total=None) -> Activities:
        energy_of_states = lambda s: self.pcn_energy_fn(s, x, y, weight_reg_total=weight_reg_total)
        return jax.tree_util.tree_map(jnp.negative, jax.grad(energy_of_states)(states_curr))

    def infer_step(self, states_curr, x, y, state_lr: float = 0.1, weight_reg_total=None) -> Activities:
        grad_step = self.neg_activity_grad(states_curr, x, y, weight_reg_total=weight_reg_total)
        return jax.tree_util.tree_map(lambda s, g: s + state_lr * g, states_curr, grad_step)

    def settle(self, x: Array, y: Array, n_steps: int = 20, state_lr: float = 0.1) -> Activities:
        weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        states_curr = self.init_activities(x)
        for _ in range(n_steps):
            states_curr = self.infer_step(states_curr, x, y, state_lr, weight_reg_total=weight_reg_total)
        return states_curr

    def make_activity_step(self, activity_optim: optax.GradientTransformation, x: Array, y: Array):
        weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        energy_fn = lambda s: self.pcn_energy_fn(s, x, y, weight_reg_total=weight_reg_total)

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
        x: Array,
        y: Array,
        n_steps: int = 20,
        return_layerwise: bool = False,
    ) -> Activities:
        states_curr0 = self.init_activities(x)
        opt_state0 = activity_optim.init(states_curr0)
        activity_step = self.make_activity_step(activity_optim, x, y)
        (states_curr, _), states_hist = jax.lax.scan(activity_step, (states_curr0, opt_state0), xs=None, length=n_steps)

        if not return_layerwise:
            return states_curr
        energy_trace_fn = lambda s: self.pcn_energy_fn(s, x, y, return_layerwise=True)
        energy_trace = jax.vmap(energy_trace_fn)(states_hist)
        return states_curr, energy_trace

    # =========================================================================
    # Learning
    # =========================================================================

    def param_grad(self, states_curr: Activities, x: Array, y: Array) -> PyTree:
        energy_of_weights = lambda m: m.pcn_energy_fn(states_curr, x, y)
        return eqx.filter_grad(energy_of_weights)(self)

    def update_params(self, grads: PyTree, optim: optax.GradientTransformation, opt_state: optax.OptState):
        updates, opt_state = optim.update(grads, opt_state, self)
        updated_model = eqx.apply_updates(self, updates)
        return updated_model, opt_state

    # =========================================================================
    # Saving and loading
    # =========================================================================

    @classmethod
    def from_config(cls, config, *, key) -> "PcnModel":
        return cls(
            layer_sizes=config.layer_sizes,
            key=key,
            act_fn=config.act_fn,
            loss=config.loss,
            weight_decay=config.weight_decay,
            orthogonal_penalty=config.orthogonal_penalty,
            activity_decay=config.activity_decay,
            activity_reg_type=config.activity_reg_type,
        )

    @classmethod
    def layer_labels(cls, config: PcnConfig) -> List[str]:
        n_interior = len(config.layer_sizes) - 2
        return [f"Hidden {i + 1}" for i in range(n_interior)] + ["Output"]

    @classmethod
    def zero_activities(cls, config: PcnConfig) -> Activities:
        return [jnp.zeros(s) for s in config.layer_sizes[1:-1]]
