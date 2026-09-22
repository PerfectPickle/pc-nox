"""runners_temporal.py

`jax.lax.scan`-fused eval/train step and run factories, shared by every
temporal PC model variant (tPC-H, bidirectional tPC-H, and future ones),
not just `TpchModel`.

This is a straight extraction of `tpch.py`'s section 8/9/10 helpers
(`make_eval_step`, `make_eval_run`, `make_train_step`, `make_train_run`,
`_train_frame_pre`/`_train_frame_post`, and their diffrax analogues).
Behaviour is unchanged -- the only thing that changed is that these
functions used to call tPC-H-specific method names
(`model.tpch_energy_fn`) and a narrower `predict`/`init_activities`
signature; they now call the small shared interface below, so any model
implementing it can use these runners as-is, with zero edits here.

--------------------------------------------------------------------------
The interface these runners expect from `model`
--------------------------------------------------------------------------
    model.predict(states_prev, states_curr, control_input, observation)
        -> (predictions, y_hat)
    model.init_activities(states_prev, control_input, observation)
        -> states_curr
    model.energy_fn(states_prev, states_curr, observation, control_input,
                     weight_reg_total=None, return_layerwise=False)
        -> scalar, or an Array if return_layerwise
    model.settle_scan(activity_optim, states_prev, observation,
                       control_input, n_steps, return_layerwise)
        -> states_curr, or (states_curr, energy_trace) if return_layerwise
    model.settle_diffrax(states_prev, observation, control_input, **kwargs)
        -> states_curr, or (states_curr, energy_trace, ts) if return_layerwise
    model.param_grad(states_prev, states_curr, observation, control_input)
        -> weight gradients, in the same pytree shape as `model`

`TpchModel` (see `tpch/model.py`) implements all of these; `observation`
is accepted but unused by `predict`/`init_activities` there. A variant
whose `predict`/`init_activities` genuinely need `observation` (e.g. a
bidirectional model's discriminative/backward pathway) can use these
runners completely unmodified -- the runners already thread `observation`
(named `y`/`y_t` at this layer, since that's the actual data value) into
both calls, they just weren't doing so before this split because only
tPC-H existed and tPC-H didn't need it.

Everything below operates on a *single, unbatched* time step per scan
iteration, exactly as in the original module -- see that module's
docstring convention (`jax.vmap` over batch, `jax.lax.scan` over time, in
the calling code).
"""

from typing import Optional

import diffrax
import equinox as eqx
import jax
import optax
from jaxtyping import Array

from .model_base import Activities

# =============================================================================
# lax.scan inference and/or training helpers (optax activity optimiser)
# =============================================================================


def make_eval_step(activity_optim: optax.GradientTransformation, n_infer_steps: int, control_input: Optional[Array] = None):
    """Builds one fully-jitted eval/inference step (zero weight updates): settle -> log-quantities

    The returned `eval_step` is traced once per distinct `return_layerwise`
    value on first use, then reused for every subsequent call with that same
    value -- not re-traced per eval iteration.

    Args:
        activity_optim: Optax transform used for the inference/settling loop,
            passed straight through to `model.settle_scan`.
        n_infer_steps: Number of relaxation steps per call, i.e. `settle_scan`'s
            `n_steps`. Fixed at build time because it becomes `jax.lax.scan`'s
            `length=` internally, which must be a concrete Python int known
            at trace time.
        control_input: Optional control-layer input, constant for the whole
            eval run and closed over here rather than passed to `eval_step`
            each call.

    Returns:
        eval_step: A function with signature
            `eval_step(model, states_prev, y, return_layerwise=False)`
            -> `(states_curr, y_hat_before, y_hat_after,
            energy_before, energy_after, energy_trace)`.
    """
    @eqx.filter_jit
    def eval_step(model, states_prev, y, return_layerwise: bool = False):
        states_curr_init = model.init_activities(states_prev, control_input, y)
        _, y_hat_before = model.predict(states_prev, states_curr_init, control_input, y)
        energy_before = model.energy_fn(states_prev, states_curr_init, y, control_input)

        settle_result = model.settle_scan(
            activity_optim, states_prev, y, control_input, n_steps=n_infer_steps, return_layerwise=return_layerwise
        )
        states_curr, energy_trace = settle_result if return_layerwise else (settle_result, None)

        _, y_hat_after = model.predict(states_prev, states_curr, control_input, y)
        energy_after = model.energy_fn(states_prev, states_curr, y, control_input)

        return states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace

    return eval_step


def make_eval_run(activity_optim, n_infer_steps, run_length, control_input=None):
    """Builds one fully-jitted, multi-frame eval/inference run: settle,
    repeated for `run_length` consecutive frames, fused into a single
    `jax.lax.scan` (and hence one JIT compile for the whole block).

    Args:
        activity_optim: Optax transform used for the inference/settling
            loop at every frame, passed straight through to `model.settle_scan`.
        n_infer_steps: Number of relaxation steps per frame. Fixed at build
            time -- becomes part of a `jax.lax.scan` `length=` internally.
        run_length: Number of frames processed per call to the returned
            `eval_run`. Fixed at build time, same reasoning.
        control_input: Optional control-layer input, constant for every
            frame and closed over here.

    Returns:
        eval_run: A function with signature
            `eval_run(model, states_prev, ys, return_layerwise=False)`
            -> `(states_curr, y_hat_before, y_hat_after, energies_before,
            energies_after, energy_traces)`.
    """
    @eqx.filter_jit
    def eval_run(model, states_prev: Activities, ys: Array, return_layerwise: bool = False):
        def step(states_prev, y_t):
            states_curr_init = model.init_activities(states_prev, control_input, y_t)
            _, y_hat_before = model.predict(states_prev, states_curr_init, control_input, y_t)
            energy_before_t = model.energy_fn(states_prev, states_curr_init, y_t, control_input)

            settle_result = model.settle_scan(
                activity_optim, states_prev, y_t, control_input, n_steps=n_infer_steps, return_layerwise=return_layerwise
            )
            states_curr, energy_trace_t = settle_result if return_layerwise else (settle_result, None)

            _, y_hat_after = model.predict(states_prev, states_curr, control_input, y_t)
            energy_after_t = model.energy_fn(states_prev, states_curr, y_t, control_input)

            return states_curr, (
                y_hat_before, y_hat_after, energy_before_t, energy_after_t, energy_trace_t
            )

        states_curr, (y_hat_before, y_hat_after, energies_before, energies_after, energy_traces) = jax.lax.scan(
            step, states_prev, xs=ys, length=run_length
        )
        return states_curr, y_hat_before, y_hat_after, energies_before, energies_after, energy_traces

    return eval_run


def make_train_step(param_optim: optax.GradientTransformation, activity_optim: optax.GradientTransformation, n_infer_steps: int, control_input: Optional[Array] = None):
    """Builds one fully-jitted training step: settle -> log-quantities -> weight update.

    Args: see `make_eval_step`, plus:
        param_optim: Optax transform used for the weight update
            (`param_grad` -> `param_optim.update` -> `eqx.apply_updates`).

    Returns:
        train_step: A function with signature
            `train_step(model, param_opt_state, states_prev, y, return_layerwise=False)`
            -> `(model, param_opt_state, states_curr, y_hat_before, y_hat_after,
            energy_before, energy_after, energy_trace)`.
    """
    @eqx.filter_jit
    def train_step(model, param_opt_state, states_prev, y, return_layerwise: bool = False):
        states_curr_init = model.init_activities(states_prev, control_input, y)
        _, y_hat_before = model.predict(states_prev, states_curr_init, control_input, y)
        energy_before = model.energy_fn(states_prev, states_curr_init, y, control_input)

        settle_result = model.settle_scan(
            activity_optim, states_prev, y, control_input, n_steps=n_infer_steps, return_layerwise=return_layerwise
        )
        states_curr, energy_trace = settle_result if return_layerwise else (settle_result, None)

        _, y_hat_after = model.predict(states_prev, states_curr, control_input, y)
        energy_after = model.energy_fn(states_prev, states_curr, y, control_input)

        grads = model.param_grad(states_prev, states_curr, y, control_input)
        updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, param_opt_state, states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace

    return train_step


def make_train_run(param_optim, activity_optim, n_infer_steps, run_length, control_input=None):
    """Builds one fully-jitted, multi-frame training run: settle -> learn,
    repeated for `run_length` consecutive frames, fused into a single
    `jax.lax.scan`.

    Args: see `make_eval_run`, plus `param_optim` (see `make_train_step`).

    Returns:
        train_run: A function with signature
            `train_run(model, param_opt_state, states_prev, ys, return_layerwise=False)`
            -> `(model, param_opt_state, states_curr, y_hat_before,
            y_hat_after, energies_before, energies_after, energy_traces)`.
    """
    @eqx.filter_jit
    def train_run(model, param_opt_state, states_prev: Activities, ys: Array, return_layerwise: bool = False):
        def step(carry, y_t):
            model, param_opt_state, states_prev = carry

            states_curr_init = model.init_activities(states_prev, control_input, y_t)
            _, y_hat_before = model.predict(states_prev, states_curr_init, control_input, y_t)
            energy_before_t = model.energy_fn(states_prev, states_curr_init, y_t, control_input)

            settle_result = model.settle_scan(
                activity_optim, states_prev, y_t, control_input, n_steps=n_infer_steps, return_layerwise=return_layerwise
            )
            states_curr, energy_trace_t = settle_result if return_layerwise else (settle_result, None)

            _, y_hat_after = model.predict(states_prev, states_curr, control_input, y_t)
            energy_after_t = model.energy_fn(states_prev, states_curr, y_t, control_input)

            grads = model.param_grad(states_prev, states_curr, y_t, control_input)
            updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
            model = eqx.apply_updates(model, updates)

            return (model, param_opt_state, states_curr), (
                y_hat_before, y_hat_after, energy_before_t, energy_after_t, energy_trace_t
            )

        (model, param_opt_state, states_curr), (y_hat_before, y_hat_after, energies_before, energies_after, energy_traces) = jax.lax.scan(
            step, (model, param_opt_state, states_prev), xs=ys, length=run_length
        )
        return model, param_opt_state, states_curr, y_hat_before, y_hat_after, energies_before, energies_after, energy_traces

    return train_run


# =============================================================================
# Diffrax training/eval helpers
# =============================================================================
# `_train_frame_pre` / `_eval_frame_post` factor out the parts of a frame
# that are IDENTICAL regardless of which settling mechanism produced
# `states_curr`: the pre-inference prediction/energy, and (for training)
# the post-inference prediction/energy/weight-update. Confirmed (in the
# original tPC-H-only version of this module) that neither half
# differentiates through the settling process, so `settle_scan` and
# `settle_diffrax` really are drop-in-different only in the middle.
# ---------------------------------------------------------------------------

def _train_frame_pre(model, states_prev, y, control_input):
    """Pre-inference half of one training/eval frame: feedforward init,
    then the prediction/energy of that raw (pre-settling) guess. Identical
    for every settling mechanism, since it runs entirely before
    `settle_scan`/`settle_diffrax` is even called.
    """
    states_curr_init = model.init_activities(states_prev, control_input, y)
    _, y_hat_before = model.predict(states_prev, states_curr_init, control_input, y)
    energy_before = model.energy_fn(states_prev, states_curr_init, y, control_input)
    return y_hat_before, energy_before


def _train_frame_post(model, param_optim, param_opt_state, states_prev, states_curr, y, control_input):
    """Post-inference half of one training frame: prediction/energy of the
    settled state, weight gradient at that state, and the optax weight
    update. Identical for every settling mechanism, since `param_grad` is
    evaluated at `states_curr` as a plain value.
    """
    _, y_hat_after = model.predict(states_prev, states_curr, control_input, y)
    energy_after = model.energy_fn(states_prev, states_curr, y, control_input)

    grads = model.param_grad(states_prev, states_curr, y, control_input)
    updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, param_opt_state, y_hat_after, energy_after


def _eval_frame_post(model, states_prev, states_curr, y, control_input):
    """Post-inference half of one EVAL frame: prediction/energy of the
    settled state, and nothing else -- i.e. `_train_frame_post` minus the
    `param_grad` / optax weight update.
    """
    _, y_hat_after = model.predict(states_prev, states_curr, control_input, y)
    energy_after = model.energy_fn(states_prev, states_curr, y, control_input)
    return y_hat_after, energy_after


def make_train_step_diffrax(
    param_optim: optax.GradientTransformation,
    max_t1: float = 20.0,
    dt0: Optional[float] = None,
    n_save: int = 20,
    solver: Optional[diffrax.AbstractSolver] = None,
    stepsize_controller: Optional[diffrax.AbstractStepSizeController] = None,
    steady_state_tol: Optional[float] = 1e-3,
    steady_state_criterion: str = "rms",
    steady_state_rtol: Optional[float] = None,
    steady_state_atol: Optional[float] = None,
    control_input: Optional["Array"] = None,
):
    """Diffrax analogue of `make_train_step`: builds one fully-jitted
    training step using `model.settle_diffrax` in place of `model.settle_scan`.

    There's no `activity_optim` here -- relaxation is governed by `solver`/
    `stepsize_controller`/`steady_state_tol` (passed straight through to
    `model.settle_diffrax`; see `inference.settle_diffrax`'s docstring for
    what each one does).

    Returns:
        train_step: A function with signature
            `train_step(model, param_opt_state, states_prev, y, return_layerwise=False)`
            -> `(model, param_opt_state, states_curr, y_hat_before,
            y_hat_after, energy_before, energy_after, energy_trace, ts)`.
    """
    if solver is None:
        solver = diffrax.Heun()
    if stepsize_controller is None:
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)

    @eqx.filter_jit
    def train_step(model, param_opt_state, states_prev, y, return_layerwise: bool = False):
        y_hat_before, energy_before = _train_frame_pre(model, states_prev, y, control_input)

        settle_result = model.settle_diffrax(
            states_prev, y, control_input,
            max_t1=max_t1, dt0=dt0, n_save=n_save,
            solver=solver, stepsize_controller=stepsize_controller,
            steady_state_tol=steady_state_tol, steady_state_criterion=steady_state_criterion,
            steady_state_rtol=steady_state_rtol, steady_state_atol=steady_state_atol,
            return_layerwise=return_layerwise,
        )
        if return_layerwise:
            states_curr, energy_trace, ts = settle_result
        else:
            states_curr, energy_trace, ts = settle_result, None, None

        model, param_opt_state, y_hat_after, energy_after = _train_frame_post(
            model, param_optim, param_opt_state, states_prev, states_curr, y, control_input
        )

        return (
            model, param_opt_state, states_curr, y_hat_before, y_hat_after,
            energy_before, energy_after, energy_trace, ts,
        )

    return train_step


def make_train_run_diffrax(
    param_optim: optax.GradientTransformation,
    run_length: int,
    max_t1: float = 20.0,
    dt0: Optional[float] = None,
    n_save: int = 20,
    solver: Optional[diffrax.AbstractSolver] = None,
    stepsize_controller: Optional[diffrax.AbstractStepSizeController] = None,
    steady_state_tol: Optional[float] = 1e-3,
    steady_state_criterion: str = "rms",
    steady_state_rtol: Optional[float] = None,
    steady_state_atol: Optional[float] = None,
    control_input: Optional["Array"] = None,
):
    """Diffrax analogue of `make_train_run`. See `make_train_step_diffrax`
    for what each argument does; `run_length` frames are fused into a
    single outer `jax.lax.scan`, same as `make_train_run`.

    Returns:
        train_run: A function with signature
            `train_run(model, param_opt_state, states_prev, ys, return_layerwise=False)`
            -> `(model, param_opt_state, states_curr, y_hat_before,
            y_hat_after, energies_before, energies_after, energy_traces,
            ts_traces)`.
    """
    if solver is None:
        solver = diffrax.Heun()
    if stepsize_controller is None:
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)

    @eqx.filter_jit
    def train_run(model, param_opt_state, states_prev: "Activities", ys: "Array", return_layerwise: bool = False):
        def step(carry, y_t):
            model, param_opt_state, states_prev = carry

            y_hat_before, energy_before_t = _train_frame_pre(model, states_prev, y_t, control_input)

            settle_result = model.settle_diffrax(
                states_prev, y_t, control_input,
                max_t1=max_t1, dt0=dt0, n_save=n_save,
                solver=solver, stepsize_controller=stepsize_controller,
                steady_state_tol=steady_state_tol, steady_state_criterion=steady_state_criterion,
                steady_state_rtol=steady_state_rtol, steady_state_atol=steady_state_atol,
                return_layerwise=return_layerwise,
            )
            if return_layerwise:
                states_curr, energy_trace_t, ts_t = settle_result
            else:
                states_curr, energy_trace_t, ts_t = settle_result, None, None

            model, param_opt_state, y_hat_after, energy_after_t = _train_frame_post(
                model, param_optim, param_opt_state, states_prev, states_curr, y_t, control_input
            )

            return (model, param_opt_state, states_curr), (
                y_hat_before, y_hat_after, energy_before_t, energy_after_t, energy_trace_t, ts_t
            )

        (model, param_opt_state, states_curr), (
            y_hat_before, y_hat_after, energies_before, energies_after, energy_traces, ts_traces
        ) = jax.lax.scan(step, (model, param_opt_state, states_prev), xs=ys, length=run_length)

        return (
            model, param_opt_state, states_curr, y_hat_before, y_hat_after,
            energies_before, energies_after, energy_traces, ts_traces,
        )

    return train_run


def make_eval_step_diffrax(
    max_t1: float = 20.0,
    dt0: Optional[float] = None,
    n_save: int = 20,
    solver: Optional[diffrax.AbstractSolver] = None,
    stepsize_controller: Optional[diffrax.AbstractStepSizeController] = None,
    steady_state_tol: Optional[float] = 1e-3,
    steady_state_criterion: str = "rms",
    steady_state_rtol: Optional[float] = None,
    steady_state_atol: Optional[float] = None,
    control_input: Optional["Array"] = None,
):
    """Diffrax analogue of `make_eval_step`, and the inference-only
    counterpart of `make_train_step_diffrax`.

    Returns:
        eval_step: A function with signature
            `eval_step(model, states_prev, y, return_layerwise=False)`
            -> `(states_curr, y_hat_before, y_hat_after, energy_before,
            energy_after, energy_trace, ts)`.
    """
    if solver is None:
        solver = diffrax.Heun()
    if stepsize_controller is None:
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)

    @eqx.filter_jit
    def eval_step(model, states_prev, y, return_layerwise: bool = False):
        y_hat_before, energy_before = _train_frame_pre(model, states_prev, y, control_input)

        settle_result = model.settle_diffrax(
            states_prev, y, control_input,
            max_t1=max_t1, dt0=dt0, n_save=n_save,
            solver=solver, stepsize_controller=stepsize_controller,
            steady_state_tol=steady_state_tol, steady_state_criterion=steady_state_criterion,
            steady_state_rtol=steady_state_rtol, steady_state_atol=steady_state_atol,
            return_layerwise=return_layerwise,
        )
        if return_layerwise:
            states_curr, energy_trace, ts = settle_result
        else:
            states_curr, energy_trace, ts = settle_result, None, None

        y_hat_after, energy_after = _eval_frame_post(model, states_prev, states_curr, y, control_input)

        return states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace, ts

    return eval_step


def make_eval_run_diffrax(
    run_length: int,
    max_t1: float = 20.0,
    dt0: Optional[float] = None,
    n_save: int = 20,
    solver: Optional[diffrax.AbstractSolver] = None,
    stepsize_controller: Optional[diffrax.AbstractStepSizeController] = None,
    steady_state_tol: Optional[float] = 1e-3,
    steady_state_criterion: str = "rms",
    steady_state_rtol: Optional[float] = None,
    steady_state_atol: Optional[float] = None,
    control_input: Optional["Array"] = None,
):
    """Diffrax analogue of `make_eval_run`, and the inference-only
    counterpart of `make_train_run_diffrax`.

    Returns:
        eval_run: A function with signature
            `eval_run(model, states_prev, ys, return_layerwise=False)`
            -> `(states_curr, y_hat_before, y_hat_after, energies_before,
            energies_after, energy_traces, ts_traces)`.
    """
    if solver is None:
        solver = diffrax.Heun()
    if stepsize_controller is None:
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)

    @eqx.filter_jit
    def eval_run(model, states_prev: "Activities", ys: "Array", return_layerwise: bool = False):
        def step(states_prev, y_t):
            y_hat_before, energy_before_t = _train_frame_pre(model, states_prev, y_t, control_input)

            settle_result = model.settle_diffrax(
                states_prev, y_t, control_input,
                max_t1=max_t1, dt0=dt0, n_save=n_save,
                solver=solver, stepsize_controller=stepsize_controller,
                steady_state_tol=steady_state_tol, steady_state_criterion=steady_state_criterion,
                steady_state_rtol=steady_state_rtol, steady_state_atol=steady_state_atol,
                return_layerwise=return_layerwise,
            )
            if return_layerwise:
                states_curr, energy_trace_t, ts_t = settle_result
            else:
                states_curr, energy_trace_t, ts_t = settle_result, None, None

            y_hat_after, energy_after_t = _eval_frame_post(model, states_prev, states_curr, y_t, control_input)

            return states_curr, (
                y_hat_before, y_hat_after, energy_before_t, energy_after_t, energy_trace_t, ts_t
            )

        states_curr, (
            y_hat_before, y_hat_after, energies_before, energies_after, energy_traces, ts_traces
        ) = jax.lax.scan(step, states_prev, xs=ys, length=run_length)

        return (
            states_curr, y_hat_before, y_hat_after,
            energies_before, energies_after, energy_traces, ts_traces,
        )

    return eval_run
