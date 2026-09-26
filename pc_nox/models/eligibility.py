"""eligibility.py

Generic (hierarchy-depth-agnostic) leaky-integration + eligibility-trace
machinery for tPC-E-style temporal credit assignment (Ng-Kee-Kwong et al.,
eqs. 28-35 of the tPC-H/tPC-E paper, generalised per S4 Appendix's own
derivation to nonlinear internal dynamics -- see tPC-HE.md for the full
worked derivation). Pure functions on plain arrays -- knows nothing about
control/hidden/observation layers, the same split `regularisers.py` and
`inference.py` already use: the MATH is generic, only which
weights/predictions use it, and how a model wires it into its outer
training scan carry, is model-specific (see `tpch/model.py`'s
`EligibilityState`/`zero_eligibility_state`/`update_eligibility_state`/
`param_grad_traced`, and `runners_temporal.py`'s `make_train_step_traced`/
`make_train_run_traced`).

Three independent pieces:

  - `leaky_predict`: blends a layer's usual instantaneous prediction with
    its own previous state (eq. 29), giving the state itself persistent
    memory across time. See `TpchControlLayer`'s `alpha` parameter.

  - `elementwise_deriv`: f'(pre), for an elementwise activation, via
    autodiff -- needed once tPC-H's nonlinearity stays inside the leaky
    blend (this codebase's choice) rather than moving to the readout (the
    source paper's choice, made specifically to avoid this).

  - TWO trace-mode implementations, differing in WHERE f' gets applied --
    an open modeling choice the source paper doesn't resolve, since its
    own linear internal dynamics make f'=1 identically, so the choice
    never arises for them:

    * "readout" (`trace_update` + `traced_weight_grad`): the trace is
      VECTOR-shaped (matching the traced input), carries NO f', and f' is
      applied once, fresh, at read-out time, uniformly across the whole
      accumulated trace history. Cheaper (O(input_size) per trace).
    * "accumulate" (`matrix_trace_update` + `traced_weight_grad_from_matrix`):
      the trace is MATRIX-shaped (matching the weight itself), and each
      accumulation step bakes in THAT step's own f'(pre) via an outer
      product before decaying -- no separate f' at read-out. More
      faithful to a literal extension of the source paper's own eq. 80-81
      derivation steps, at the cost of O(own_size * input_size) memory per
      trace instead of O(input_size).

    These are NOT equivalent for alpha < 1 spanning several steps (see
    tPC-HE.md section 8); they coincide only in the trivial alpha=1 case.
    `TpchModel`'s `trace_mode` config field selects between them.

Caveat shared by BOTH modes, not derived in this codebase (see the tPC-HE
derivation doc): the update rule assumes weights change slowly relative to
the trace's own decay -- the same approximation the original tPC-E
derivation makes (S4 Appendix: dropping the "non-local" recursive
sensitivity term), inherited here unchanged by either mode.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array


def leaky_predict(state_prev: Array, instantaneous_pred: Array, alpha: float) -> Array:
    """eq. 29: z_tilde_t = (1 - alpha) * z_{t-1} + alpha * instantaneous_pred.

    `instantaneous_pred` is whatever a layer's ordinary `predict()` would
    have returned (already through its own activation function, if any --
    the published derivation is for the linear case; applying the leak to
    the post-activation prediction is the natural generalisation used
    here, not verified against the paper's nonlinear appendix). See
    `TpchControlLayer.predict` for this exact expression wired into a
    real layer.
    """
    return (1.0 - alpha) * state_prev + alpha * instantaneous_pred


def trace_update(trace_prev: Array, input_now: Array, alpha: float) -> Array:
    """eqs. 32-33: e_t = (1 - alpha) * e_{t-1} + alpha * input_now.

    `input_now` is whatever activity feeds a leaky-integrated weight this
    step (e.g. `state_prev` for a recurrent weight like W_rec, or
    `control_input` for an input weight like W_in) -- use the SAME alpha
    as the `leaky_predict` call this trace is paired with, since the
    trace is meant to track the same persistence the state itself has.
    """
    return (1.0 - alpha) * trace_prev + alpha * input_now


def traced_weight_grad(error: Array, trace: Array) -> Array:
    """eqs. 34-35: dW = eta * outer(error, trace) is the paper's own
    convention: an ASCENT step meant to be ADDED to the weight directly
    (A <- A + dW), not subtracted. This function returns just the
    `outer(error, trace)` factor (caller applies `eta`), in that same
    ADDITIVE convention.

    CAUTION -- sign convention: this is the OPPOSITE sign to a
    `jax.grad(energy)`-style descent gradient (the convention
    `param_grad`/`eqx.filter_grad` use everywhere else in this codebase,
    where `optax`/`eqx.apply_updates` SUBTRACT the returned value to
    descend). If you're assembling a gradient pytree meant to be
    consumed by `update_params`/optax (as `TpchModel.param_grad_traced`
    does), negate this before inserting it: `-traced_weight_grad(error,
    trace)`. If you're applying the paper's update directly and
    additively instead, use it as returned. `error`/`trace` play exactly
    the roles epsilon^z/e^A_t do in the paper; shape is
    (len(error), len(trace)), matching the weight matrix it updates.

    This is the "readout" trace-mode's read-out step (see
    `matrix_trace_update`/`traced_weight_grad_from_matrix` for the
    "accumulate" mode's very differently-shaped counterpart, and
    `TpchModel.param_grad_traced`'s `trace_mode` for how a model chooses
    between them). `trace` here is VECTOR-shaped (matching the traced
    input, not the weight) -- the caller (`param_grad_traced`) supplies
    `error` already combined with `f'(pre)` (as `delta_l`) before calling
    this, since "readout" mode applies `f'` once, fresh, at read-out time.
    """
    return jnp.outer(error, trace)


def matrix_trace_update(trace_prev: Array, deriv_now: Array, input_now: Array, alpha: float) -> Array:
    """"Accumulate" trace-mode's own recurrence -- the alternative to
    `trace_update` discussed in the tPC-H+E derivation doc's "open
    modeling choice" section. Unlike `trace_update`'s trace (vector-shaped,
    matching the traced INPUT), this trace is MATRIX-shaped, matching the
    WEIGHT it updates -- because `deriv_now` (that step's own `f'(pre)`,
    shape matching the layer's OWN output) gets folded in at accumulation
    time, one outer product per step, rather than pulled out and applied
    once at read-out:

        e_t = (1 - alpha) * e_{t-1} + alpha * outer(deriv_now, input_now)

    `deriv_now`: `f'(pre)` at THIS step, from `elementwise_deriv` -- NOT
    multiplied by any error here; error is combined at read-out via
    `traced_weight_grad_from_matrix`, not here. `input_now`: the traced
    input (e.g. `state_prev`, `control_input`), same role as
    `trace_update`'s `input_now`. `alpha`: same layer leak rate as always.

    Memory cost: O(own_size * input_size), i.e. the same size as the
    weight matrix itself -- versus `trace_update`'s O(input_size). This is
    the real, practical cost of this mode, not just an algorithmic choice.
    """
    return (1.0 - alpha) * trace_prev + alpha * jnp.outer(deriv_now, input_now)


def traced_weight_grad_from_matrix(error: Array, trace_matrix: Array) -> Array:
    """"Accumulate" trace-mode's read-out step -- the counterpart to
    `traced_weight_grad` for a matrix-shaped trace built with
    `matrix_trace_update`. Since `f'(pre)` is already baked into
    `trace_matrix` at every accumulation step, only `error` needs
    combining here, as a per-ROW broadcast scale (matching eq. 82's own
    `epsilon^z e^A_t` notation, read as row-wise broadcasting rather than
    a matrix product -- `trace_matrix` is already weight-shaped, so there
    is no outer product left to take):

        dW = outer_free_scale = error[:, None] * trace_matrix

    Same ADDITIVE sign convention as `traced_weight_grad` -- negate before
    inserting into a `jax.grad`-convention gradient pytree. Shape:
    (len(error), trace_matrix.shape[1]), matching the weight matrix.
    """
    return error[:, None] * trace_matrix


def zero_matrix_trace_like(own_size: int, input_size: int) -> Array:
    """Zero-initialised matrix trace for "accumulate" mode, shaped like
    the weight matrix it will update -- see `matrix_trace_update`.
    """
    return jnp.zeros((own_size, input_size))


def zero_trace_like(input_example: Array) -> Array:
    """Zero-initialised trace matching one leaky-integrated input's shape
    -- e.g. `zero_trace_like(control_input)` for W_in's trace, or
    `zero_trace_like(state_prev)` for W_rec's -- for building the initial
    trace state at the start of a sequence.
    """
    return jnp.zeros_like(input_example)


def elementwise_deriv(act_fn: Callable, pre: Array) -> Array:
    """f'(pre), for an ELEMENTWISE activation function, computed via
    autodiff (a single `jax.jvp` with an all-ones tangent) rather than a
    hand-coded derivative -- works for any elementwise `act_fn` (tanh,
    relu, identity, ...) without needing a closed-form derivative
    registered anywhere. Valid specifically because an elementwise
    function's Jacobian is diagonal, so its directional derivative in the
    all-ones direction IS the elementwise derivative; do not use this for
    a non-elementwise `act_fn` (e.g. softmax).

    This is the `f'(pre_l)` factor in the traced weight-update rule
    (`δ_l = f'(pre_l) ⊙ ε_l`, then `traced_weight_grad(δ_l, trace)`) --
    the correction needed for a nonlinear `f`; the plain-`outer(error,
    trace)` form (paper's own linear-case eqs. 34-35) implicitly assumes
    f'=1 everywhere.
    """
    _, deriv = jax.jvp(act_fn, (pre,), (jnp.ones_like(pre),))
    return deriv
