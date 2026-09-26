"""runners_static.py

Batch train/eval step factories for the static PCN family (`PcnModel` and
its siblings, e.g. a future `pcn_meta` package) -- the non-temporal
counterpart to `runners_temporal.py`.

Deliberately a SEPARATE module, not a generalisation of
`runners_temporal.py`: that module's whole shape is built around
`states_prev` threading through a `jax.lax.scan` across TIME, because
each frame genuinely depends on the previous one. A batch of static PCN
examples has no such dependency -- every `(x, y)` pair is independent --
so batching here is `jax.vmap` over independent per-example settling,
followed by averaging the resulting weight gradients before one optax
step. There's no carry, and therefore nothing for `lax.scan` to do that
`vmap` doesn't do more directly.

--------------------------------------------------------------------------
The interface these expect from `model`
--------------------------------------------------------------------------
    model.predict(states_curr, x) -> (predictions, y_hat)
    model.init_activities(x) -> states_curr
    model.energy_fn(states_curr, x, y, weight_reg_total=None, return_layerwise=False) -> scalar or Array
    model.settle_scan(activity_optim, x, y, n_steps, return_layerwise) -> states_curr, or (states_curr, energy_trace)
    model.param_grad(states_curr, x, y) -> weight gradients, same pytree shape as `model`

`PcnModel` implements all of these (see `pcn/model.py`); any sibling
variant that keeps this shape (e.g. a future Meta-PCN model swapping in
its own `pcn_energy_fn`/`param_grad` internals) can use these runners
unmodified, same "small shared interface" pattern as `runners_temporal.py`.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array


def make_eval_step(activity_optim: optax.GradientTransformation, n_infer_steps: int):
    """Builds one fully-jitted, batched eval/inference step: per-example
    settle (via `jax.vmap`) -> log-quantities. Zero weight updates.

    Returns:
        eval_step: `eval_step(model, xs, ys, return_layerwise=False)` ->
            `(states_curr, y_hat_before, y_hat_after, energy_before,
            energy_after, energy_trace)`, every leading axis batched.
    """
    @eqx.filter_jit
    def eval_step(model, xs: Array, ys: Array, return_layerwise: bool = False):
        def per_example(x, y):
            states_curr_init = model.init_activities(x)
            _, y_hat_before = model.predict(states_curr_init, x)
            energy_before = model.energy_fn(states_curr_init, x, y)

            settle_result = model.settle_scan(activity_optim, x, y, n_steps=n_infer_steps, return_layerwise=return_layerwise)
            states_curr, energy_trace = settle_result if return_layerwise else (settle_result, None)

            _, y_hat_after = model.predict(states_curr, x)
            energy_after = model.energy_fn(states_curr, x, y)
            return states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace

        return jax.vmap(per_example)(xs, ys)

    return eval_step


def make_train_step(param_optim: optax.GradientTransformation, activity_optim: optax.GradientTransformation, n_infer_steps: int):
    """Builds one fully-jitted, batched training step: per-example settle
    (via `jax.vmap`) -> per-example weight gradient -> gradients averaged
    across the batch -> ONE optax weight update.

    The averaging is the one genuinely batch-specific piece here (there's
    no analogue in `runners_temporal.py`, where every step already
    operates on a single sequence): each example's `param_grad` is
    computed independently under `vmap`, then combined with
    `jax.tree_util.tree_map(jnp.mean, ...)` before `param_optim.update`,
    matching ordinary minibatch SGD/Adam semantics.

    Returns:
        train_step: `train_step(model, param_opt_state, xs, ys, return_layerwise=False)`
            -> `(model, param_opt_state, states_curr, y_hat_before,
            y_hat_after, energy_before, energy_after, energy_trace)` --
            `states_curr`/`y_hat_*`/`energy_*` are batched (one entry per
            example); `model`/`param_opt_state` are not (one update for
            the whole batch).
    """
    @eqx.filter_jit
    def train_step(model, param_opt_state, xs: Array, ys: Array, return_layerwise: bool = False):
        def per_example(x, y):
            states_curr_init = model.init_activities(x)
            _, y_hat_before = model.predict(states_curr_init, x)
            energy_before = model.energy_fn(states_curr_init, x, y)

            settle_result = model.settle_scan(activity_optim, x, y, n_steps=n_infer_steps, return_layerwise=return_layerwise)
            states_curr, energy_trace = settle_result if return_layerwise else (settle_result, None)

            _, y_hat_after = model.predict(states_curr, x)
            energy_after = model.energy_fn(states_curr, x, y)

            grads = model.param_grad(states_curr, x, y)
            return grads, states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace

        grads_batch, states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace = jax.vmap(
            per_example
        )(xs, ys)

        grads = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads_batch)
        updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
        model = eqx.apply_updates(model, updates)
        model = model.postprocess_params()

        return model, param_opt_state, states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace

    return train_step
