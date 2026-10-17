"""graph/runners.py

Train/eval factories for `GraphModel` (and any model with the same
dict-state calling convention), covering BOTH axes the two older runner
modules each fix one of:

    runners_temporal.py   time  axis: `lax.scan` carrying `states_prev`
    runners_static.py     batch axis: `jax.vmap` + mean gradients, no carry

A graph may mix static nodes (no memory) and temporal nodes (leaky,
recurrent), so neither module alone fits. The two axes are orthogonal and
compose, which is all this module does:

    one FRAME   = settle + (optionally) weight gradient for ONE example at
                  ONE time step                                  (_frame)
    batch axis  = `jax.vmap` over frames, mean the gradients, ONE optax
                  update                                    (*_step factories)
    time axis   = `jax.lax.scan` over batched steps, carrying each step's
                  `states_curr` as the next `states_prev`    (*_run factories)

A purely static graph is simply the case with no time axis: use the
`*_step` factories and never pass `states_prev` (static nodes ignore it).
A purely temporal model is the case with batch size 1. Neither is special.

Data layout (leading axes are batch, then time for the `*_run` factories):

    observation   {clamped_node_name: Array}  leaves (B, ...)   [(T, B, ...) for runs]
                  A clamped node may be omitted (a missing modality); see
                  `GraphModel`. Evaluating a classifier = omit the label node.
    states_prev   {node_name: Array} leaves (B, ...) or None. None is fine
                  when no node is recurrent. For runs, None starts from zeros.
    control_input Array (B, ...) or None            [(T, B, ...) for runs]

Outputs mirror the older runners: `(states_curr, y_hat_before, y_hat_after,
energy_before, energy_after, energy_trace)` for eval, with `model` and
`param_opt_state` prepended for training. `y_hat_*` are dicts keyed by
clamped node. Everything except `model`/`param_opt_state` carries the batch
axis (and a leading time axis for runs).
"""

from typing import Mapping, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree

from .model import GraphModel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def zero_states(model_or_config, batch_size: Optional[int] = None) -> PyTree:
    """All-zero `states_prev` for a model (or config): unbatched if
    `batch_size` is None, else with a leading batch axis. The usual initial
    `states_prev` for a sequence."""
    config = getattr(model_or_config, "config", model_or_config)
    zeros = GraphModel.zero_activities(config)
    if batch_size is None:
        return zeros
    return jax.tree_util.tree_map(lambda z: jnp.zeros((batch_size,) + z.shape, z.dtype), zeros)


def _leading_size(tree: PyTree, axis: int) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("observation is empty; at least one clamped node must be observed")
    return leaves[0].shape[axis]


# --------------------------------------------------------------------------- #
# One frame, then one batched step (shared by the *_step and *_run factories)
# --------------------------------------------------------------------------- #

def _frame(model, activity_optim, n_infer_steps, return_layerwise, with_grads, init_direction=None, before_init_dir=None):
    """The per-example computation: init -> settle -> predictions/energies
    (before and after) -> optionally the weight gradient."""
    # This flag exists so that in a bidirectional context, 
    # a purely generative (td init) y_hat_before and energy_before can be returned
    if before_init_dir is None:
        before_init_dir = init_direction
    def per_example(states_prev, observation, control_input): # TODO, consider moving y_hat_before earlier? or separate y_hat_before init dir option
        states_curr_init = model.init_activities(states_prev, control_input, observation, before_init_dir)
        _, y_hat_before = model.predict(states_prev, states_curr_init, control_input, observation)

        # The states settling starts from (the linearisation point z^(0)). Energies with a frozen context
        # (Meta-PCN) are defined relative to it, so the before/after energies and the weight gradient all
        # use it. `frozen` is {} (and free) for plain-PC graphs.
        states_settle_init = model.init_activities(states_prev, control_input, observation, init_direction)
        frozen = model.frozen_context(states_prev, states_settle_init, observation, control_input)
        energy_before = model.energy_fn(states_prev, states_curr_init, observation, control_input, frozen=frozen)

        settle_result = model.settle_scan(
            activity_optim, states_prev, observation, control_input,
            n_steps=n_infer_steps, return_layerwise=return_layerwise, init_direction=init_direction,
            states_init=states_settle_init,
        )
        states_curr, energy_trace = settle_result if return_layerwise else (settle_result, None)

        _, y_hat_after = model.predict(states_prev, states_curr, control_input, observation)
        energy_after = model.energy_fn(states_prev, states_curr, observation, control_input, frozen=frozen)

        grads = (model.param_grad(states_prev, states_curr, observation, control_input, states_init=states_settle_init)
                 if with_grads else None)
        return grads, states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace

    return per_example


def _eval_batch(model, activity_optim, n_infer_steps, observation, states_prev, control_input, return_layerwise, init_direction=None, before_init_dir=None):
    frame = _frame(model, activity_optim, n_infer_steps, return_layerwise, with_grads=False, init_direction=init_direction, before_init_dir=before_init_dir)
    _, *outputs = jax.vmap(frame)(states_prev, observation, control_input)
    return tuple(outputs)        # states_curr, y_hat_before, y_hat_after, e_before, e_after, trace


def _train_batch(model, param_optim, activity_optim, n_infer_steps, param_opt_state,
                 observation, states_prev, control_input, return_layerwise, init_direction=None, before_init_dir=None):
    frame = _frame(model, activity_optim, n_infer_steps, return_layerwise, with_grads=True, init_direction=init_direction, before_init_dir=before_init_dir)
    grads_batch, *outputs = jax.vmap(frame)(states_prev, observation, control_input)

    grads = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads_batch)   # minibatch semantics
    updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
    model = eqx.apply_updates(model, updates)
    model = model.postprocess_params()
    return (model, param_opt_state, *outputs)


# --------------------------------------------------------------------------- #
# Batch axis only
# --------------------------------------------------------------------------- #

def make_eval_step(activity_optim: optax.GradientTransformation, n_infer_steps: int):
    """Batched eval step (zero weight updates).

    Returns `eval_step(model, observation, *, states_prev=None,
    control_input=None, return_layerwise=False)` ->
    `(states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace)`.
    """
    @eqx.filter_jit
    def eval_step(model, observation: Mapping[str, Array], *, states_prev=None, control_input=None,
                  return_layerwise: bool = False, init_direction: Optional[str] = None, before_init_dir=None):
        return _eval_batch(model, activity_optim, n_infer_steps, observation, states_prev, control_input, return_layerwise, init_direction, before_init_dir)

    return eval_step


def make_train_step(param_optim: optax.GradientTransformation, activity_optim: optax.GradientTransformation,
                    n_infer_steps: int):
    """Batched training step: per-example settle and weight gradient under
    `vmap`, gradients averaged over the batch, ONE optax update.

    Returns `train_step(model, param_opt_state, observation, *,
    states_prev=None, control_input=None, return_layerwise=False)` ->
    `(model, param_opt_state, states_curr, y_hat_before, y_hat_after,
    energy_before, energy_after, energy_trace)`. `model`/`param_opt_state`
    are unbatched (one update for the whole batch); the rest carry the batch axis.
    """
    @eqx.filter_jit
    def train_step(model, param_opt_state, observation: Mapping[str, Array], *, states_prev=None,
                   control_input=None, return_layerwise: bool = False, init_direction: Optional[str] = None, before_init_dir: Optional[str] = None):
        return _train_batch(model, param_optim, activity_optim, n_infer_steps, param_opt_state,
                            observation, states_prev, control_input, return_layerwise, init_direction, before_init_dir)

    return train_step


# --------------------------------------------------------------------------- #
# Batch axis + time axis
# --------------------------------------------------------------------------- #

def make_eval_run(activity_optim: optax.GradientTransformation, n_infer_steps: int, run_length: int):
    """`run_length` consecutive batched eval frames fused into one `lax.scan`,
    each frame's `states_curr` becoming the next frame's `states_prev`.

    Returns `eval_run(model, observations, *, states_prev=None,
    control_inputs=None, return_layerwise=False)`; `observations` leaves have
    shape `(run_length, B, ...)`. Outputs as `eval_step`, each with a leading
    time axis, except `states_curr` which is the FINAL step's `(B, ...)`.
    `states_prev=None` starts from zeros.
    """
    @eqx.filter_jit
    def eval_run(model, observations: Mapping[str, Array], *, states_prev=None, control_inputs=None,
                 return_layerwise: bool = False, init_direction: Optional[str] = None, before_init_dir: Optional[str] = None):
        if states_prev is None:
            states_prev = zero_states(model, _leading_size(observations, axis=1))

        def step(states_prev, xs_t):
            obs_t, ci_t = xs_t
            states_curr, *logs = _eval_batch(
                model, activity_optim, n_infer_steps, obs_t, states_prev, ci_t, return_layerwise, init_direction, before_init_dir)
            return states_curr, tuple(logs)

        states_curr, logs = jax.lax.scan(step, states_prev, xs=(observations, control_inputs), length=run_length)
        return (states_curr, *logs)

    return eval_run


def make_train_run(param_optim: optax.GradientTransformation, activity_optim: optax.GradientTransformation,
                   n_infer_steps: int, run_length: int):
    """`run_length` consecutive batched training frames fused into one
    `lax.scan` (settle -> learn -> carry states forward).

    Returns `train_run(model, param_opt_state, observations, *, states_prev=None,
    control_inputs=None, return_layerwise=False)`; `observations` leaves have
    shape `(run_length, B, ...)`. Outputs as `train_step`, with a leading
    time axis on the logs and the FINAL step's `states_curr`.
    """
    @eqx.filter_jit
    def train_run(model, param_opt_state, observations: Mapping[str, Array], *, states_prev=None,
                  control_inputs=None, return_layerwise: bool = False, init_direction: Optional[str] = None, before_init_dir: Optional[str] = None):
        if states_prev is None:
            states_prev = zero_states(model, _leading_size(observations, axis=1))

        def step(carry, xs_t):
            model, param_opt_state, states_prev = carry
            obs_t, ci_t = xs_t
            model, param_opt_state, states_curr, *logs = _train_batch(
                model, param_optim, activity_optim, n_infer_steps, param_opt_state,
                obs_t, states_prev, ci_t, return_layerwise, init_direction, before_init_dir)
            return (model, param_opt_state, states_curr), tuple(logs)

        (model, param_opt_state, states_curr), logs = jax.lax.scan(
            step, (model, param_opt_state, states_prev), xs=(observations, control_inputs), length=run_length)
        return (model, param_opt_state, states_curr, *logs)

    return train_run
