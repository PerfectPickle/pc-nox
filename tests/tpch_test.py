from models.tpch import TpchModel, TpchControlLayer, TpchHiddenLayer, TpchObservationLayer
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
import pytest
from typing import Callable, List, Sequence, Tuple, Optional
import optax

# =============================================================================
# manual gradient functions implemented by me, to test energy function correctness
# =============================================================================

### Activity update gradients

# Activity update gradient calculation is the same for control and hidden layers in general, but update differs for the parent of the observation layer. This function handles both cases.
def activity_update(layer: TpchControlLayer | TpchHiddenLayer, layer_error, state_prev, state_curr, child_layer: TpchHiddenLayer | TpchObservationLayer, child_error, child_state_prev):
    if isinstance(child_layer, TpchHiddenLayer):
        # vmap allows the derivative to be applied element-wise on a vector
        f_prime = jax.vmap(jax.grad(layer.act_fn))
        child_preactivations = child_layer.W_rec(child_state_prev) + child_layer.W_parent_prev(state_prev) + child_layer.W_parent_curr(state_curr)
        return -layer_error + jnp.transpose(child_layer.W_parent_curr.weight) @ (child_error * f_prime(child_preactivations))

    elif isinstance(child_layer, TpchObservationLayer):
        return -layer_error + jnp.transpose(child_layer.W_parent.weight) @ child_error


### Param update gradients

# Control layer

def control_rec_weight_update_grad(layer: TpchControlLayer, layer_error, state_prev: Array, control_input: Optional[Array] = None):
    # vmap allows the derivative to be applied element-wise on a vector
    f_prime = jax.vmap(jax.grad(layer.act_fn))
    weighted_inputs = layer.W_rec(state_prev)
    if layer.has_input and control_input is not None:
            weighted_inputs += layer.W_in(control_input)
    # transpose of state_prev is redundant here, where jnp.outer is used
    # we divide both sides of equation (e.g. 22) by -lr to get the raw partial derivative on LHS
    return -jnp.outer(f_prime(weighted_inputs) * layer_error, state_prev)

def control_input_weight_update_grad(layer: TpchControlLayer, lr, layer_error, state_prev: Array, control_input: Optional[Array] = None):
    # vmap allows the derivative to be applied element-wise on a vector
    f_prime = jax.vmap(jax.grad(layer.act_fn))
    weighted_inputs = layer.W_rec(state_prev) + layer.W_in(control_input)
    # we divide both sides of equation (e.g. 22) by -lr to get the raw partial derivative on LHS
    return -jnp.outer(f_prime(weighted_inputs) * layer_error, control_input)


# Hidden layers (these are almost identical, just multiplied by different state transposes)

def hidden_rec_weight_update_grad(layer: TpchHiddenLayer, layer_error, state_prev: Array, parent_prev: Array, parent_curr: Array):
    # vmap allows the derivative to be applied element-wise on a vector
    f_prime = jax.vmap(jax.grad(layer.act_fn))
    weighted_inputs = layer.W_rec(state_prev) + layer.W_parent_prev(parent_prev) + layer.W_parent_curr(parent_curr)
    # we divide both sides of equation (e.g. 22) by -lr to get the raw partial derivative on LHS
    return -jnp.outer(f_prime(weighted_inputs) * layer_error, state_prev)

def parent_prev_weight_update_grad(layer: TpchHiddenLayer, layer_error, state_prev: Array, parent_prev: Array, parent_curr: Array):
    # vmap allows the derivative to be applied element-wise on a vector
    f_prime = jax.vmap(jax.grad(layer.act_fn))
    weighted_inputs = layer.W_rec(state_prev) + layer.W_parent_prev(parent_prev) + layer.W_parent_curr(parent_curr)
    # we divide both sides of equation (e.g. 22) by -lr to get the raw partial derivative on LHS
    return -jnp.outer(f_prime(weighted_inputs) * layer_error, parent_prev)

def parent_curr_weight_update_grad(layer: TpchHiddenLayer, layer_error, state_prev: Array, parent_prev: Array, parent_curr: Array):
    # vmap allows the derivative to be applied element-wise on a vector
    f_prime = jax.vmap(jax.grad(layer.act_fn))
    weighted_inputs = layer.W_rec(state_prev) + layer.W_parent_prev(parent_prev) + layer.W_parent_curr(parent_curr)
    # we divide both sides of equation (e.g. 22) by -lr to get the raw partial derivative on LHS
    return -jnp.outer(f_prime(weighted_inputs) * layer_error, parent_curr)


# Observation layer

def obs_weight_update_grad(layer: TpchObservationLayer, layer_error, parent_curr: Array):
    # we divide both sides of equation 27 by -lr to get the raw partial derivative on LHS
    return -jnp.outer(layer_error, parent_curr)
    



# =============================================================================
# pytests by Claude: correctness & sanity test suite for tPC-H
# =============================================================================
# Nothing above this banner has been changed -- this section only ADDS
# fixtures, helpers, and test functions below the existing code (plus two
# new import lines up top: `Optional` and `optax`, which the code above
# already needed but never imported).
#
#
# We deliberately give every layer in the fixtures below a DIFFERENT width
# (4, 3, 5, 6, input 2) specifically so a shape bug can't hide
# behind an accidental broadcast or a coincidental size match -- it raises
# loudly instead of silently returning something the wrong shape. Worth
# keeping this habit for future PCN-variant tests.
#
# For future PCN variants: the fixtures/helpers/tests below are written to
# depend only on the public TpchModel API (predict, tpch_energy_fn,
# neg_activity_grad, param_grad, settle, settle_scan, make_tpch_sequence_step)
# plus equinox/optax, so most of this file should copy-paste with only the
# model constructor call and the manual formula calls needing to change.
# =============================================================================

from jax.test_util import check_grads


# ---- fixtures ---------------------------------------------------------------

FX_CONTROL_SIZE = 4
FX_HIDDEN_SIZES = [3, 5]
FX_OBS_SIZE = 6
FX_INPUT_SIZE = 2


@pytest.fixture
def fx_model():
    return TpchModel(
        control_layer_size=FX_CONTROL_SIZE,
        hidden_sizes=FX_HIDDEN_SIZES,
        obs_size=FX_OBS_SIZE,
        key=jr.key(0),
        input_size=FX_INPUT_SIZE,
    )


@pytest.fixture
def fx_states_prev():
    sizes = [FX_CONTROL_SIZE] + list(FX_HIDDEN_SIZES)
    keys = jr.split(jr.key(1), len(sizes))
    return [jr.normal(k, (n,)) for k, n in zip(keys, sizes)]


@pytest.fixture
def fx_control_input():
    return jr.normal(jr.key(2), (FX_INPUT_SIZE,))


@pytest.fixture
def fx_observation():
    return jr.normal(jr.key(3), (FX_OBS_SIZE,))


@pytest.fixture
def fx_states_curr():
    """A genuinely arbitrary point in state-space -- deliberately NOT
    model.init_activities(...). The feedforward init makes every layer's
    prediction error exactly zero by construction (each state literally
    equals its own prediction), which trivially satisfies almost any
    gradient formula, correct or not. Random states give every comparison
    below real, non-zero errors to actually test against."""
    sizes = [FX_CONTROL_SIZE] + list(FX_HIDDEN_SIZES)
    keys = jr.split(jr.key(4), len(sizes))
    return [jr.normal(k, (n,)) for k, n in zip(keys, sizes)]


def assert_allclose(actual, expected, name, atol=1e-4, rtol=1e-4):
    """Reusable comparison helper with an informative failure message.
    Worth keeping this in future PCN-variant test files too."""
    actual, expected = jnp.asarray(actual), jnp.asarray(expected)
    assert actual.shape == expected.shape, (
        f"{name}: shape mismatch -- got {actual.shape}, expected {expected.shape}"
    )
    max_abs_diff = float(jnp.max(jnp.abs(actual - expected))) if actual.size else 0.0
    assert jnp.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name}: values differ, max abs diff = {max_abs_diff}"
    )


def layer_energies(model, states_prev, states_curr, observation, control_input=None):
    """Per-term breakdown of tpch_energy_fn's sum: one entry per layer that
    contributes an error term (control, each hidden layer, observation), in
    top-to-bottom order. sum(layer_energies(...)) == tpch_energy_fn(...) by
    construction -- handy for isolating which layer's prediction is off."""
    predictions, y_hat = model.predict(states_prev, states_curr, control_input)
    energies = [0.5 * jnp.sum((s - p) ** 2) for s, p in zip(states_curr, predictions)]
    energies.append(0.5 * jnp.sum((observation - y_hat) ** 2))
    return energies


# =============================================================================
# A. Model construction & structural sanity
# =============================================================================

def test_model_predict_shapes(fx_model, fx_states_prev, fx_control_input):
    """predict() should return one prediction per state (matching
    states_curr's shapes) plus a y_hat matching obs_size."""
    states_curr = fx_model.init_activities(fx_states_prev, fx_control_input)
    predictions, y_hat = fx_model.predict(fx_states_prev, states_curr, fx_control_input)
    assert len(predictions) == len(states_curr)
    for pred, state in zip(predictions, states_curr):
        assert pred.shape == state.shape
    assert y_hat.shape == (FX_OBS_SIZE,)


def test_zero_hidden_layers_edge_case():
    """A degenerate hierarchy (control -> observation directly, no hidden
    layers) should still build and run predict/energy/settle cleanly, since
    the docstring explicitly claims this "generalises for free" to any
    number of hidden layers, including zero."""
    model = TpchModel(control_layer_size=4, hidden_sizes=[], obs_size=6, key=jr.key(10), input_size=2)
    states_prev = [jr.normal(jr.key(11), (4,))]
    control_input = jr.normal(jr.key(12), (2,))
    observation = jr.normal(jr.key(13), (6,))

    states_curr = model.init_activities(states_prev, control_input)
    assert len(states_curr) == 1
    energy = model.tpch_energy_fn(states_prev, states_curr, observation, control_input)
    assert jnp.isfinite(energy) and energy >= 0

    settled = model.settle(states_prev, observation, control_input, n_steps=5, state_lr=0.05)
    assert len(settled) == len(states_prev)


def test_no_control_input_edge_case():
    """input_size=0 (has_input=False) should work throughout with
    control_input left as None."""
    model = TpchModel(control_layer_size=4, hidden_sizes=[3], obs_size=5, key=jr.key(20))
    assert model.control_layer.has_input is False

    states_prev = [jr.normal(k, (n,)) for k, n in zip(jr.split(jr.key(21), 2), [4, 3])]
    observation = jr.normal(jr.key(22), (5,))

    states_curr = model.init_activities(states_prev, control_input=None)
    energy = model.tpch_energy_fn(states_prev, states_curr, observation, control_input=None)
    assert jnp.isfinite(energy)

    grad = model.neg_activity_grad(states_curr, states_prev, observation, control_input=None)
    assert len(grad) == len(states_curr)
    for g, s in zip(grad, states_curr):
        assert g.shape == s.shape
        assert jnp.all(jnp.isfinite(g))


# =============================================================================
# B. Free-energy properties (from the requested sanity checklist)
# =============================================================================

def test_energy_is_finite_and_nonnegative(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    """assert jnp.isfinite(energy); assert energy >= 0
    True for ANY inputs, not just settled ones: F_t is a sum of squared
    terms (eq. 19), so it's bounded below by zero by construction."""
    energy = fx_model.tpch_energy_fn(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)
    assert jnp.isfinite(energy)
    assert energy >= 0


def test_layer_energies_length_and_consistency(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    """assert len(energies) == len(model); assert all(jnp.isfinite(e) for e in energies)
    `len(model)` isn't meaningful for an eqx.Module (no __len__), so we use
    the natural equivalent here: one energy term per error-contributing
    component (control layer + every hidden layer + observation layer)."""
    energies = layer_energies(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input)
    n_components = 1 + len(fx_model.hidden_layers) + 1
    assert len(energies) == n_components
    assert all(jnp.isfinite(e) for e in energies)
    assert all(e >= 0 for e in energies)
    # the breakdown should sum back up to tpch_energy_fn's scalar output
    total = fx_model.tpch_energy_fn(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)
    assert_allclose(sum(energies), total, "sum(layer_energies) vs tpch_energy_fn")


def test_energy_is_a_pure_function(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    """Calling tpch_energy_fn twice on identical inputs should give
    bit-for-bit the same result -- catches accidental hidden randomness/state."""
    e1 = fx_model.tpch_energy_fn(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)
    e2 = fx_model.tpch_energy_fn(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)
    assert e1 == e2


# =============================================================================
# C. Activity/param-gradient structural sanity, plus an independent
#    finite-difference correctness check that does NOT depend on the
#    hand-derived manual formulas in section D -- a good first thing to
#    trust if D's comparisons ever disagree.
# =============================================================================

def test_neg_activity_grad_shapes_and_finiteness(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    """assert len(grads) == len(activities)
    for grad, act in zip(grads, activities):
        assert grad.shape == act.shape
        assert jnp.all(jnp.isfinite(grad))"""
    grads = fx_model.neg_activity_grad(fx_states_curr, fx_states_prev, fx_observation, fx_control_input)
    activities = fx_states_curr
    assert len(grads) == len(activities)
    for grad, act in zip(grads, activities):
        assert grad.shape == act.shape
        assert jnp.all(jnp.isfinite(grad))


def test_neg_activity_grad_matches_finite_differences(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    """Cross-checks jax.grad's gradient of tpch_energy_fn w.r.t. states_curr
    against numerical (finite-difference) gradients -- independent of any
    hand-written formula. This really tests whether tpch_energy_fn itself is
    a faithful, cleanly-differentiable implementation of eq. (19); if this
    ever fails, look at tpch_energy_fn/predict before anything else."""
    energy_fn = lambda s: fx_model.tpch_energy_fn(fx_states_prev, s, fx_observation, fx_control_input)
    check_grads(energy_fn, (fx_states_curr,), order=1, modes=("rev",), atol=1e-2, rtol=1e-2)


def test_param_grad_matches_finite_differences(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    """Same idea, but for the WEIGHT gradients (eqs. 22-27) rather than the
    activity gradients (eqs. 20-21). eqx.partition/combine splits the model
    into its array leaves (what we differentiate w.r.t.) and its static
    metadata (act_fn, has_input, ...), which check_grads needs."""
    params, static = eqx.partition(fx_model, eqx.is_array)

    def energy_of_params(p):
        m = eqx.combine(p, static)
        return m.tpch_energy_fn(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)

    check_grads(energy_of_params, (params,), order=1, modes=("rev",), atol=1e-2, rtol=1e-2)


# =============================================================================
# D. Manual (paper-derived) update rules vs. autograd -- the comparison you
#    actually asked for. 
# =============================================================================

def test_activity_update_matches_autograd_control_layer(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    predictions, _ = fx_model.predict(fx_states_prev, fx_states_curr, fx_control_input)
    layer_errors = [s - p for s, p in zip(fx_states_curr, predictions)]

    manual = activity_update(
        fx_model.control_layer, layer_errors[0], fx_states_prev[0], fx_states_curr[0],
        fx_model.hidden_layers[0], layer_errors[1], fx_states_prev[1],
    )
    autograd = fx_model.neg_activity_grad(fx_states_curr, fx_states_prev, fx_observation, fx_control_input)
    assert_allclose(manual, autograd[0], "control layer activity grad (manual vs autograd)")


def test_activity_update_matches_autograd_hidden_to_hidden(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    predictions, _ = fx_model.predict(fx_states_prev, fx_states_curr, fx_control_input)
    layer_errors = [s - p for s, p in zip(fx_states_curr, predictions)]

    manual = activity_update(
        fx_model.hidden_layers[0], layer_errors[1], fx_states_prev[1], fx_states_curr[1],
        fx_model.hidden_layers[1], layer_errors[2], fx_states_prev[2],
    )
    autograd = fx_model.neg_activity_grad(fx_states_curr, fx_states_prev, fx_observation, fx_control_input)
    assert_allclose(manual, autograd[1], "hidden[0] activity grad (manual vs autograd)")


def test_activity_update_matches_autograd_last_hidden_to_observation(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    predictions, y_hat = fx_model.predict(fx_states_prev, fx_states_curr, fx_control_input)
    layer_errors = [s - p for s, p in zip(fx_states_curr, predictions)]
    y_error = fx_observation - y_hat

    manual = activity_update(
        fx_model.hidden_layers[-1], layer_errors[-1], fx_states_prev[-1], fx_states_curr[-1],
        fx_model.observation_layer, y_error, None,
    )
    autograd = fx_model.neg_activity_grad(fx_states_curr, fx_states_prev, fx_observation, fx_control_input)
    assert_allclose(manual, autograd[-1], "hidden[-1] activity grad (manual vs autograd)")


def test_control_weight_grads_match_autograd(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    predictions, _ = fx_model.predict(fx_states_prev, fx_states_curr, fx_control_input)
    layer_error = fx_states_curr[0] - predictions[0]
    autograd = fx_model.param_grad(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)

    manual_rec = control_rec_weight_update_grad(fx_model.control_layer, layer_error, fx_states_prev[0], fx_control_input)
    assert_allclose(manual_rec, autograd.control_layer.W_rec.weight, "control W_rec grad")

    manual_in = control_input_weight_update_grad(fx_model.control_layer, None, layer_error, fx_states_prev[0], fx_control_input)
    assert_allclose(manual_in, autograd.control_layer.W_in.weight, "control W_in grad")


def test_hidden_weight_grads_match_autograd(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    predictions, _ = fx_model.predict(fx_states_prev, fx_states_curr, fx_control_input)
    layer_error = fx_states_curr[1] - predictions[1]
    autograd = fx_model.param_grad(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)

    manual_rec = hidden_rec_weight_update_grad(fx_model.hidden_layers[0], layer_error, fx_states_prev[1], fx_states_prev[0], fx_states_curr[0])
    assert_allclose(manual_rec, autograd.hidden_layers[0].W_rec.weight, "hidden[0] W_rec grad")

    manual_pp = parent_prev_weight_update_grad(fx_model.hidden_layers[0], layer_error, fx_states_prev[1], fx_states_prev[0], fx_states_curr[0])
    assert_allclose(manual_pp, autograd.hidden_layers[0].W_parent_prev.weight, "hidden[0] W_parent_prev grad")

    manual_pc = parent_curr_weight_update_grad(fx_model.hidden_layers[0], layer_error, fx_states_prev[1], fx_states_prev[0], fx_states_curr[0])
    assert_allclose(manual_pc, autograd.hidden_layers[0].W_parent_curr.weight, "hidden[0] W_parent_curr grad")


def test_observation_weight_grad_matches_autograd(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    _, y_hat = fx_model.predict(fx_states_prev, fx_states_curr, fx_control_input)
    y_error = fx_observation - y_hat
    autograd = fx_model.param_grad(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)

    manual = obs_weight_update_grad(fx_model.observation_layer, y_error, fx_states_curr[-1])
    assert_allclose(manual, autograd.observation_layer.W_parent.weight, "observation W_parent grad")


# =============================================================================
# E. Weight-gradient PyTree structure -- needed for eqx.apply_updates/optax
#    to work at all, so worth its own (non-numeric) regression test.
# =============================================================================

def test_param_grad_structure_matches_model(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    grads = fx_model.param_grad(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)
    model_arrays = eqx.filter(fx_model, eqx.is_array)
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(model_arrays)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.all(jnp.isfinite(leaf))


def test_one_adam_update_keeps_params_finite(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    optim = optax.adam(learning_rate=1e-3)
    opt_state = optim.init(eqx.filter(fx_model, eqx.is_array))
    grads = fx_model.param_grad(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)
    updated_model, _ = fx_model.update_params(grads, optim, opt_state)
    for leaf in jax.tree_util.tree_leaves(eqx.filter(updated_model, eqx.is_array)):
        assert jnp.all(jnp.isfinite(leaf))


# =============================================================================
# F. Inference (settle) behavior -- operational checks that the gradient
#    step actually descends the energy, complementing the analytic checks
#    above with a "does it behave correctly when you run it" check.
# =============================================================================

def test_infer_step_does_not_increase_energy(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input):
    e_before = fx_model.tpch_energy_fn(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)
    states_next = fx_model.infer_step(fx_states_curr, fx_states_prev, fx_observation, fx_control_input, state_lr=0.01)
    e_after = fx_model.tpch_energy_fn(fx_states_prev, states_next, fx_observation, fx_control_input)
    assert jnp.isfinite(e_after)
    assert e_after <= e_before + 1e-6


def test_settle_reduces_energy_below_feedforward_init(fx_model, fx_states_prev, fx_control_input, fx_observation):
    init_guess = fx_model.init_activities(fx_states_prev, fx_control_input)
    energy_before = fx_model.tpch_energy_fn(fx_states_prev, init_guess, fx_observation, fx_control_input)

    settled = fx_model.settle(fx_states_prev, fx_observation, fx_control_input, n_steps=20, state_lr=0.1)
    energy_after = fx_model.tpch_energy_fn(fx_states_prev, settled, fx_observation, fx_control_input)

    assert jnp.isfinite(energy_after)
    assert energy_after <= energy_before


def test_settle_output_length_and_shapes(fx_model, fx_states_prev, fx_control_input, fx_observation):
    """assert len(solution) == len(activities)"""
    activities = fx_model.init_activities(fx_states_prev, fx_control_input)
    solution = fx_model.settle(fx_states_prev, fx_observation, fx_control_input, n_steps=10, state_lr=0.1)
    assert len(solution) == len(activities)
    for s, a in zip(solution, activities):
        assert s.shape == a.shape


def test_settle_and_settle_scan_agree(fx_model, fx_states_prev, fx_control_input, fx_observation):
    """The Euler for-loop (settle) and the optax/scan-fused version
    (settle_scan) should converge to essentially the same energy given a
    matching learning rate and step count -- catches the two inference
    implementations silently drifting apart from each other."""
    settled = fx_model.settle(fx_states_prev, fx_observation, fx_control_input, n_steps=30, state_lr=0.05)
    settled_scan = fx_model.settle_scan(optax.sgd(learning_rate=0.05), fx_states_prev, fx_observation, fx_control_input, n_steps=30)

    e1 = fx_model.tpch_energy_fn(fx_states_prev, settled, fx_observation, fx_control_input)
    e2 = fx_model.tpch_energy_fn(fx_states_prev, settled_scan, fx_observation, fx_control_input)
    assert_allclose(e1, e2, "settle() vs settle_scan() final energy", atol=1e-3, rtol=1e-3)


# =============================================================================
# G. Sequence-level scan (make_tpch_sequence_step) sanity
# =============================================================================

def test_sequence_step_over_short_synthetic_sequence(fx_model):
    seq_len = 4
    xkey, ykey, skey = jr.split(jr.key(30), 3)
    x_seq = jr.normal(xkey, (seq_len, FX_INPUT_SIZE))
    y_seq = jr.normal(ykey, (seq_len, FX_OBS_SIZE))
    sizes = [FX_CONTROL_SIZE] + list(FX_HIDDEN_SIZES)
    states_prev_0 = [jr.normal(k, (n,)) for k, n in zip(jr.split(skey, len(sizes)), sizes)]

    activity_optim = optax.sgd(learning_rate=0.05)
    sequence_step = fx_model.make_tpch_sequence_step(activity_optim, n_infer_steps=5)
    final_states, (states_history, energies) = jax.lax.scan(sequence_step, states_prev_0, xs=(x_seq, y_seq))

    assert len(final_states) == len(states_prev_0)
    assert energies.shape == (seq_len,)
    assert jnp.all(jnp.isfinite(energies))
    for hist_leaf, prev_leaf in zip(states_history, states_prev_0):
        assert hist_leaf.shape == (seq_len,) + prev_leaf.shape


# =============================================================================
# H. Checkpointing (ModelBase.save_checkpoint / load_checkpoint) applied to
#    a real TpchModel. The generic save/load *contract* (opt_state/activities
#    flags, registry dispatch, the ValueError/NotImplementedError cases,
#    etc.) is covered once, model-agnostically, in model_base_test.py against
#    a throwaway dummy model -- these two just confirm nothing about tPC-H's
#    actual shape (nested hidden_layers list, optional W_in, act_fn as a
#    static field) breaks that generic round trip.
# =============================================================================

@pytest.fixture
def fx_config():
    return TpchConfig(
        control_layer_size=FX_CONTROL_SIZE,
        hidden_sizes=FX_HIDDEN_SIZES,
        obs_size=FX_OBS_SIZE,
        input_size=FX_INPUT_SIZE,
        act_fn="tanh",
    )


def test_checkpoint_round_trip_predict_matches(fx_model, fx_config, fx_states_prev, fx_control_input, tmp_path):
    out_dir = fx_model.save_checkpoint(fx_config, path=tmp_path / "tpch_ckpt")
    loaded = TpchModel.load_checkpoint(out_dir)

    states_curr_orig = fx_model.init_activities(fx_states_prev, fx_control_input)
    states_curr_loaded = loaded.model.init_activities(fx_states_prev, fx_control_input)
    for o, l in zip(states_curr_orig, states_curr_loaded):
        assert_allclose(o, l, "init_activities before vs after checkpoint round trip")

    preds_orig, y_hat_orig = fx_model.predict(fx_states_prev, states_curr_orig, fx_control_input)
    preds_loaded, y_hat_loaded = loaded.model.predict(fx_states_prev, states_curr_loaded, fx_control_input)
    assert_allclose(y_hat_orig, y_hat_loaded, "y_hat before vs after checkpoint round trip")
    for o, l in zip(preds_orig, preds_loaded):
        assert_allclose(o, l, "layer prediction before vs after checkpoint round trip")


def test_checkpoint_round_trip_with_activities_and_opt_state(fx_model, fx_config, tmp_path):
    """The 'everything at once' path: config + metadata + opt_state +
    activities all saved and reloaded together, using TpchModel's real
    zero_activities implementation."""
    optim = optax.adam(learning_rate=1e-3)
    opt_state = optim.init(eqx.filter(fx_model, eqx.is_array))
    activities = fx_model.zero_activities(fx_config)

    out_dir = fx_model.save_checkpoint(
        fx_config,
        path=tmp_path / "tpch_ckpt_full",
        metadata={"epoch": 3},
        opt_state=opt_state,
        activities=activities,
    )
    loaded = TpchModel.load_checkpoint(out_dir, optim=optim)

    assert loaded.metadata == {"epoch": 3}
    assert loaded.opt_state is not None
    assert loaded.activities is not None
    assert len(loaded.activities) == len(activities)
    for a in loaded.activities:
        assert jnp.all(a == 0.0)