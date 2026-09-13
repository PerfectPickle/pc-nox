from models.tpch import TpchModel, TpchConfig, TpchControlLayer, TpchHiddenLayer, TpchObservationLayer, make_train_step, make_train_run
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
    construction -- handy for isolating which layer's prediction is off.

    Mirrors tpch_energy_fn's own loss switch (mse vs ce) via model.config.loss,
    so this stays a valid ground truth for both loss types, not just mse.
    """
    predictions, y_hat = model.predict(states_prev, states_curr, control_input)
    energies = [0.5 * jnp.sum((s - p) ** 2) for s, p in zip(states_curr, predictions)]
    if model.config.loss == "mse":
        energies.append(0.5 * jnp.sum((observation - y_hat) ** 2))
    else:  # "ce"
        energies.append(-jnp.sum(observation * jax.nn.log_softmax(y_hat)))
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


def test_checkpoint_round_trip_predict_matches(fx_model, fx_states_prev, fx_control_input, tmp_path):
    # No config= needed: TpchModel stores its config as self.config (set in
    # __init__), and save_checkpoint falls back to that automatically.
    out_dir = fx_model.save_checkpoint(path=tmp_path / "tpch_ckpt")
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


# =============================================================================
# I. Regularisation -- explicit, isolated correctness tests
#
# The regularisers live inside tpch_energy_fn(), so the strongest tests are
# deliberately independent of the task-loss terms:
#
#   1. Make a zero-task-energy state by starting from init_activities() and
#      choosing observation == y_hat.
#   2. Compare an otherwise identical regularised model against a
#      zero-regularisation model.
#   3. Check the exact energy delta and the exact gradient delta.
#
# This prevents a broken regulariser from "passing" merely because the total
# energy/gradient is finite or because the task gradient happens to dominate.
#
# These tests also exercise every public scope:
#   weight_decay:       all / rec / ff
#   orthogonal_penalty: all / rec / ff
#   activity_decay:     l1 / l2
#
# The gradient checks below are analytic and are deliberately supplemented by
# finite-difference checks for the *regularised* total energy, so they do not
# rely exclusively on our hand-derived regulariser formulas.
# =============================================================================


def _weight_entries(model):
    """Return all model weight matrices as (stable_name, array) pairs."""
    entries = [("control.W_rec", model.control_layer.W_rec.weight)]
    if model.control_layer.has_input:
        entries.append(("control.W_in", model.control_layer.W_in.weight))
    for i, layer in enumerate(model.hidden_layers):
        entries.extend([
            (f"hidden[{i}].W_rec", layer.W_rec.weight),
            (f"hidden[{i}].W_parent_prev", layer.W_parent_prev.weight),
            (f"hidden[{i}].W_parent_curr", layer.W_parent_curr.weight),
        ])
    entries.append(("observation.W_parent", model.observation_layer.W_parent.weight))
    return entries


def _recurrent_names(model):
    return {
        "control.W_rec",
        *(f"hidden[{i}].W_rec" for i in range(len(model.hidden_layers))),
    }


def _flatten_model_param_grads(model, grads):
    """Return parameter-gradient leaves using the same stable names as _weight_entries."""
    result = {}
    result["control.W_rec"] = grads.control_layer.W_rec.weight
    if model.control_layer.has_input:
        result["control.W_in"] = grads.control_layer.W_in.weight
    for i, layer in enumerate(model.hidden_layers):
        result[f"hidden[{i}].W_rec"] = grads.hidden_layers[i].W_rec.weight
        result[f"hidden[{i}].W_parent_prev"] = grads.hidden_layers[i].W_parent_prev.weight
        result[f"hidden[{i}].W_parent_curr"] = grads.hidden_layers[i].W_parent_curr.weight
    result["observation.W_parent"] = grads.observation_layer.W_parent.weight
    return result


def _regularised_model(*, weight_decay=0.0, weight_decay_scope="all",
                       orthogonal_penalty=0.0, orthogonal_scope="rec",
                       activity_decay=0.0, activity_reg_type="l1"):
    """Build the same small, intentionally rectangular network used by the tests."""
    return TpchModel(
        control_layer_size=FX_CONTROL_SIZE,
        hidden_sizes=FX_HIDDEN_SIZES,
        obs_size=FX_OBS_SIZE,
        key=jr.key(101),
        input_size=FX_INPUT_SIZE,
        weight_decay=weight_decay,
        weight_decay_scope=weight_decay_scope,
        orthogonal_penalty=orthogonal_penalty,
        orthogonal_scope=orthogonal_scope,
        activity_decay=activity_decay,
        activity_reg_type=activity_reg_type,
    )


@pytest.fixture
def fx_zero_task_point():
    """Construct a state where the unregularised task energy is exactly zero."""
    model = _regularised_model()
    states_prev = [jr.normal(k, (n,)) for k, n in
                   zip(jr.split(jr.key(102), len([FX_CONTROL_SIZE] + FX_HIDDEN_SIZES)),
                       [FX_CONTROL_SIZE] + FX_HIDDEN_SIZES)]
    control_input = jr.normal(jr.key(103), (FX_INPUT_SIZE,))

    states_curr = model.init_activities(states_prev, control_input)
    _, y_hat = model.predict(states_prev, states_curr, control_input)
    observation = y_hat

    base_energy = model.tpch_energy_fn(
        states_prev, states_curr, observation, control_input
    )
    # This is an especially useful invariant for these tests: any non-zero
    # result means the test point itself is not actually isolating the
    # regularisers.
    assert base_energy == 0.0

    return model, states_prev, states_curr, observation, control_input


def _expected_orthogonal_grad(weight, coefficient):
    """Gradient of 0.5 * coefficient * ||I - Gram(W)||_F^2.

    tpch.py uses W.T @ W when the matrix is tall/square and W @ W.T when it
    is wide, choosing the smaller Gram matrix. This helper mirrors that
    mathematical definition rather than reaching into the implementation.
    """
    out_dim, in_dim = weight.shape
    if in_dim <= out_dim:
        gram = weight.T @ weight
        identity = jnp.eye(in_dim, dtype=weight.dtype)
        return 2.0 * coefficient * weight @ (gram - identity)
    else:
        gram = weight @ weight.T
        identity = jnp.eye(out_dim, dtype=weight.dtype)
        return 2.0 * coefficient * (gram - identity) @ weight


def test_regularisation_zero_coefficients_are_exact_noops(
    fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    """Setting every regularisation coefficient to zero must reproduce the
    original energy, activity gradient, and parameter gradient exactly."""
    base = _regularised_model()
    explicit_zero = _regularised_model(
        weight_decay=0.0,
        orthogonal_penalty=0.0,
        activity_decay=0.0,
    )

    e_base = base.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input
    )
    e_zero = explicit_zero.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input
    )
    assert e_base == e_zero

    g_base = _flatten_model_param_grads(
        base,
        base.param_grad(fx_states_prev, fx_states_curr, fx_observation, fx_control_input),
    )
    g_zero = _flatten_model_param_grads(
        explicit_zero,
        explicit_zero.param_grad(
            fx_states_prev, fx_states_curr, fx_observation, fx_control_input
        ),
    )
    for name in g_base:
        assert_allclose(g_zero[name], g_base[name], f"zero-reg param grad: {name}",
                        atol=0.0, rtol=0.0)

    a_base = base.neg_activity_grad(
        fx_states_curr, fx_states_prev, fx_observation, fx_control_input
    )
    a_zero = explicit_zero.neg_activity_grad(
        fx_states_curr, fx_states_prev, fx_observation, fx_control_input
    )
    for i, (a0, ab) in enumerate(zip(a_zero, a_base)):
        assert_allclose(a0, ab, f"zero-reg activity grad[{i}]",
                        atol=0.0, rtol=0.0)


@pytest.mark.parametrize("scope", ["rec", "ff", "all"])
def test_weight_decay_energy_matches_exact_frobenius_penalty(
    fx_zero_task_point, scope
):
    """weight_decay must add exactly 0.5 * lambda * sum ||W||_F^2 over the
    requested weight scope, and nothing else."""
    _, states_prev, states_curr, observation, control_input = fx_zero_task_point
    reg = _regularised_model(weight_decay=0.37, weight_decay_scope=scope)
    base = _regularised_model()

    expected = 0.5 * 0.37 * sum(
        jnp.sum(weight ** 2)
        for name, weight in _weight_entries(reg)
        if (
            scope == "all"
            or (scope == "rec" and name in _recurrent_names(reg))
            or (scope == "ff" and name not in _recurrent_names(reg))
        )
    )

    e_reg = reg.tpch_energy_fn(
        states_prev, states_curr, observation, control_input
    )
    e_base = base.tpch_energy_fn(
        states_prev, states_curr, observation, control_input
    )
    assert_allclose(
        e_reg - e_base, expected, f"weight_decay energy ({scope})",
        atol=2e-6, rtol=2e-6,
    )


@pytest.mark.parametrize("scope", ["rec", "ff", "all"])
def test_weight_decay_parameter_gradient_targets_only_requested_scope(
    fx_states_prev, fx_states_curr, fx_observation, fx_control_input, scope
):
    """The parameter-gradient delta from weight decay must be lambda*W on
    selected matrices and exactly zero on excluded matrices."""
    coefficient = 0.23
    base = _regularised_model()
    reg = _regularised_model(
        weight_decay=coefficient,
        weight_decay_scope=scope,
    )

    base_grads = _flatten_model_param_grads(
        base, base.param_grad(
            fx_states_prev, fx_states_curr, fx_observation, fx_control_input
        )
    )
    reg_grads = _flatten_model_param_grads(
        reg, reg.param_grad(
            fx_states_prev, fx_states_curr, fx_observation, fx_control_input
        )
    )

    recurrent = _recurrent_names(reg)
    for name, weight in _weight_entries(reg):
        selected = (
            scope == "all"
            or (scope == "rec" and name in recurrent)
            or (scope == "ff" and name not in recurrent)
        )
        expected_delta = coefficient * weight if selected else jnp.zeros_like(weight)
        actual_delta = reg_grads[name] - base_grads[name]
        assert_allclose(
            actual_delta, expected_delta,
            f"weight_decay gradient delta ({scope}, {name})",
            atol=2e-6, rtol=2e-6,
        )


def test_weight_decay_does_not_change_activity_gradient(
    fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    """Weight-only regularisation depends only on parameters, so its gradient
    w.r.t. current activities must be exactly zero."""
    base = _regularised_model()
    reg = _regularised_model(weight_decay=0.41, weight_decay_scope="all")

    base_grad = base.neg_activity_grad(
        fx_states_curr, fx_states_prev, fx_observation, fx_control_input
    )
    reg_grad = reg.neg_activity_grad(
        fx_states_curr, fx_states_prev, fx_observation, fx_control_input
    )

    for i, (a, b) in enumerate(zip(base_grad, reg_grad)):
        assert_allclose(
            b, a, f"weight_decay activity gradient unchanged [{i}]",
            atol=0.0, rtol=0.0,
        )


@pytest.mark.parametrize("scope", ["rec", "ff", "all"])
def test_orthogonal_penalty_energy_matches_exact_definition(
    fx_zero_task_point, scope
):
    """orthogonal_penalty must add exactly
    0.5 * mu * sum ||I - Gram(W)||_F^2, with the same narrow/tall Gram
    convention documented by tpch.py."""
    _, states_prev, states_curr, observation, control_input = fx_zero_task_point
    coefficient = 0.19
    reg = _regularised_model(
        orthogonal_penalty=coefficient,
        orthogonal_scope=scope,
    )
    base = _regularised_model()

    recurrent = _recurrent_names(reg)
    expected = 0.0
    for name, weight in _weight_entries(reg):
        selected = (
            scope == "all"
            or (scope == "rec" and name in recurrent)
            or (scope == "ff" and name not in recurrent)
        )
        if selected:
            out_dim, in_dim = weight.shape
            if in_dim <= out_dim:
                gram = weight.T @ weight
                identity = jnp.eye(in_dim, dtype=weight.dtype)
            else:
                gram = weight @ weight.T
                identity = jnp.eye(out_dim, dtype=weight.dtype)
            expected = expected + 0.5 * coefficient * jnp.sum((identity - gram) ** 2)

    e_reg = reg.tpch_energy_fn(
        states_prev, states_curr, observation, control_input
    )
    e_base = base.tpch_energy_fn(
        states_prev, states_curr, observation, control_input
    )
    assert_allclose(
        e_reg - e_base, expected, f"orthogonal energy ({scope})",
        atol=2e-6, rtol=2e-6,
    )


@pytest.mark.parametrize("scope", ["rec", "ff", "all"])
def test_orthogonal_penalty_gradient_targets_only_requested_scope(
    fx_states_prev, fx_states_curr, fx_observation, fx_control_input, scope
):
    """The parameter-gradient delta from the orthogonality penalty must match
    the derivative of the implemented Gram-matrix objective, and be exactly
    zero on excluded weights."""
    coefficient = 0.13
    base = _regularised_model()
    reg = _regularised_model(
        orthogonal_penalty=coefficient,
        orthogonal_scope=scope,
    )

    base_grads = _flatten_model_param_grads(
        base, base.param_grad(
            fx_states_prev, fx_states_curr, fx_observation, fx_control_input
        )
    )
    reg_grads = _flatten_model_param_grads(
        reg, reg.param_grad(
            fx_states_prev, fx_states_curr, fx_observation, fx_control_input
        )
    )

    recurrent = _recurrent_names(reg)
    for name, weight in _weight_entries(reg):
        selected = (
            scope == "all"
            or (scope == "rec" and name in recurrent)
            or (scope == "ff" and name not in recurrent)
        )
        expected_delta = (
            _expected_orthogonal_grad(weight, coefficient)
            if selected else jnp.zeros_like(weight)
        )
        actual_delta = reg_grads[name] - base_grads[name]
        assert_allclose(
            actual_delta, expected_delta,
            f"orthogonal gradient delta ({scope}, {name})",
            atol=2e-5, rtol=2e-5,
        )


def test_orthogonal_penalty_is_zero_for_identity_recurrent_weights():
    """An exactly orthogonal recurrent matrix must have zero orthogonal
    penalty and zero orthogonal gradient."""
    model = _regularised_model(
        orthogonal_penalty=1.0,
        orthogonal_scope="rec",
    )

    # Replace the control recurrent matrix with identity and the first hidden
    # recurrent matrix with identity. These are square by construction.
    identity_control = jnp.eye(FX_CONTROL_SIZE)
    identity_hidden = jnp.eye(FX_HIDDEN_SIZES[0])
    model = eqx.tree_at(
        lambda m: (m.control_layer.W_rec.weight, m.hidden_layers[0].W_rec.weight),
        model,
        (identity_control, identity_hidden),
    )

    selected = model._rec_weights()
    expected_energy = 0.5 * sum(
        jnp.sum((jnp.eye(w.shape[0]) - w.T @ w) ** 2) for w in selected
    )
    actual_energy = model._weight_orthogonal_reg()
    assert_allclose(actual_energy, expected_energy, "identity orthogonal penalty")
    assert actual_energy >= 0.0

    # Make the remaining terms irrelevant to this check by inspecting the
    # direct weight-only energy contribution.
    reg_grads = _flatten_model_param_grads(
        model,
        model.param_grad(
            [jnp.zeros(n) for n in [FX_CONTROL_SIZE] + FX_HIDDEN_SIZES],
            [jnp.zeros(n) for n in [FX_CONTROL_SIZE] + FX_HIDDEN_SIZES],
            jnp.zeros(FX_OBS_SIZE),
            jnp.zeros(FX_INPUT_SIZE),
        ),
    )
    assert_allclose(
        reg_grads["control.W_rec"],
        jnp.zeros_like(identity_control),
        "identity control W_rec orthogonal gradient",
        atol=2e-6, rtol=2e-6,
    )
    assert_allclose(
        reg_grads["hidden[0].W_rec"],
        jnp.zeros_like(identity_hidden),
        "identity hidden W_rec orthogonal gradient",
        atol=2e-6, rtol=2e-6,
    )


@pytest.mark.parametrize("reg_type", ["l1", "l2"])
def test_activity_regularisation_energy_matches_exact_norm(
    fx_states_prev, fx_states_curr, fx_observation, fx_control_input, reg_type
):
    """activity_decay must add exactly 0.5 * lambda * sum |s| for L1 or
    0.5 * lambda * sum s^2 for L2 over every current activity."""
    coefficient = 0.29
    base = _regularised_model()
    reg = _regularised_model(
        activity_decay=coefficient,
        activity_reg_type=reg_type,
    )

    e_base = base.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input
    )
    e_reg = reg.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input
    )

    if reg_type == "l1":
        expected = 0.5 * coefficient * sum(
            jnp.sum(jnp.abs(state)) for state in fx_states_curr
        )
    else:
        expected = 0.5 * coefficient * sum(
            jnp.sum(state ** 2) for state in fx_states_curr
        )

    assert_allclose(
        e_reg - e_base, expected,
        f"activity {reg_type} energy",
        atol=2e-6, rtol=2e-6,
    )


@pytest.mark.parametrize("reg_type", ["l1", "l2"])
def test_activity_regularisation_changes_only_activity_gradients(
    fx_states_prev, fx_observation, fx_control_input, reg_type
):
    """Activity regularisation must contribute to inference gradients but not
    parameter gradients. L1 is tested away from zero so its sign derivative
    is unambiguous."""
    coefficient = 0.31
    base = _regularised_model()
    reg = _regularised_model(
        activity_decay=coefficient,
        activity_reg_type=reg_type,
    )

    states_curr = [
        jnp.array([0.4, -0.7, 0.9, -1.1]),
        jnp.array([-0.3, 0.8, -1.2]),
        jnp.array([0.6, -0.5, 1.0, -0.9, 0.2]),
    ]

    base_act = base.neg_activity_grad(
        states_curr, fx_states_prev, fx_observation, fx_control_input
    )
    reg_act = reg.neg_activity_grad(
        states_curr, fx_states_prev, fx_observation, fx_control_input
    )

    for i, (actual, state) in enumerate(zip(
        [r - b for r, b in zip(reg_act, base_act)], states_curr
    )):
        if reg_type == "l1":
            expected = -0.5 * coefficient * jnp.sign(state)
        else:
            expected = -coefficient * state
        assert_allclose(
            actual, expected,
            f"activity {reg_type} negative-gradient delta [{i}]",
            atol=2e-6, rtol=2e-6,
        )

    base_param = _flatten_model_param_grads(
        base, base.param_grad(
            fx_states_prev, states_curr, fx_observation, fx_control_input
        )
    )
    reg_param = _flatten_model_param_grads(
        reg, reg.param_grad(
            fx_states_prev, states_curr, fx_observation, fx_control_input
        )
    )
    for name in base_param:
        assert_allclose(
            reg_param[name], base_param[name],
            f"activity {reg_type} parameter gradient unchanged: {name}",
            atol=0.0, rtol=0.0,
        )


def test_activity_l1_gradient_matches_jax_subgradient_at_zero(
    fx_states_prev, fx_observation, fx_control_input
):
    """At exactly zero, the L1 contribution matches the subgradient chosen
    by JAX for jnp.abs(). The L1 derivative is not uniquely defined at zero,
    so the test checks the actual autodiff convention rather than assuming
    a particular subgradient.
    """
    decay = 0.7

    model = _regularised_model(
        activity_decay=decay,
        activity_reg_type="l1",
    )
    base = _regularised_model()

    states_curr = [
        jnp.zeros(FX_CONTROL_SIZE),
        jnp.zeros(FX_HIDDEN_SIZES[0]),
        jnp.zeros(FX_HIDDEN_SIZES[1]),
    ]

    regularised_grad = model.neg_activity_grad(
        states_curr,
        fx_states_prev,
        fx_observation,
        fx_control_input,
    )
    base_grad = base.neg_activity_grad(
        states_curr,
        fx_states_prev,
        fx_observation,
        fx_control_input,
    )

    # JAX's chosen derivative of abs(x) at x=0.
    abs_grad_at_zero = jax.grad(lambda x: jnp.abs(x))(0.0)

    # _activity_reg = 0.5 * decay * sum(abs(state))
    # neg_activity_grad therefore receives:
    #     -0.5 * decay * d|x|/dx
    expected_contribution = -0.5 * decay * abs_grad_at_zero

    for i, (reg_grad, base_g) in enumerate(zip(regularised_grad, base_grad)):
        actual_contribution = reg_grad - base_g

        assert jnp.all(jnp.isfinite(reg_grad))

        assert_allclose(
            actual_contribution,
            jnp.full_like(actual_contribution, expected_contribution),
            f"L1 zero-point subgradient [{i}]",
            atol=1e-6,
            rtol=1e-6,
        )


def test_combined_regularisation_energy_is_additive(
    fx_zero_task_point
):
    """Weight L2 + orthogonal + activity regularisation must equal the sum of
    the three individually-measured energy contributions, with no accidental
    interaction or double counting."""
    _, states_prev, states_curr, observation, control_input = fx_zero_task_point

    wd = 0.17
    op = 0.11
    ad = 0.23

    base = _regularised_model()
    weight_only = _regularised_model(weight_decay=wd, weight_decay_scope="all")
    orth_only = _regularised_model(orthogonal_penalty=op, orthogonal_scope="all")
    activity_only = _regularised_model(activity_decay=ad, activity_reg_type="l2")
    combined = _regularised_model(
        weight_decay=wd,
        weight_decay_scope="all",
        orthogonal_penalty=op,
        orthogonal_scope="all",
        activity_decay=ad,
        activity_reg_type="l2",
    )

    e0 = base.tpch_energy_fn(states_prev, states_curr, observation, control_input)
    ew = weight_only.tpch_energy_fn(states_prev, states_curr, observation, control_input)
    eo = orth_only.tpch_energy_fn(states_prev, states_curr, observation, control_input)
    ea = activity_only.tpch_energy_fn(states_prev, states_curr, observation, control_input)
    ec = combined.tpch_energy_fn(states_prev, states_curr, observation, control_input)

    assert_allclose(
        ec - e0,
        (ew - e0) + (eo - e0) + (ea - e0),
        "combined regularisation energy additivity",
        atol=4e-6, rtol=4e-6,
    )


def test_combined_regularisation_gradient_matches_finite_differences(
    fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    """The complete regularised energy remains correctly differentiable:
    jax.grad should agree with numerical finite differences when all three
    regulariser families are enabled together.

    This is intentionally a whole-model check rather than three isolated
    formula checks; it catches plumbing mistakes such as accidentally
    computing a regulariser but failing to add it to tpch_energy_fn().
    """
    model = _regularised_model(
        weight_decay=0.07,
        weight_decay_scope="all",
        orthogonal_penalty=0.05,
        orthogonal_scope="all",
        activity_decay=0.09,
        activity_reg_type="l2",
    )

    def energy_of_states(s):
        return model.tpch_energy_fn(
            fx_states_prev, s, fx_observation, fx_control_input
        )

    check_grads(
        energy_of_states,
        (fx_states_curr,),
        order=1,
        modes=("rev",),
        atol=2e-2,
        rtol=2e-2,
    )

    params, static = eqx.partition(model, eqx.is_array)

    def energy_of_params(p):
        m = eqx.combine(p, static)
        return m.tpch_energy_fn(
            fx_states_prev, fx_states_curr, fx_observation, fx_control_input
        )

    check_grads(
        energy_of_params,
        (params,),
        order=1,
        modes=("rev",),
        atol=2e-2,
        rtol=2e-2,
    )


def test_weight_reg_total_fast_path_is_exactly_equivalent(
    fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    """Passing the precomputed weight_reg_total must not change the energy
    relative to the ordinary tpch_energy_fn path."""
    model = _regularised_model(
        weight_decay=0.12,
        weight_decay_scope="all",
        orthogonal_penalty=0.08,
        orthogonal_scope="rec",
        activity_decay=0.21,
        activity_reg_type="l2",
    )
    cached = model._weight_l2_reg() + model._weight_orthogonal_reg()

    normal = model.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input
    )
    fast = model.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input,
        weight_reg_total=cached,
    )
    assert_allclose(normal, fast, "cached weight_reg_total vs normal energy",
                    atol=0.0, rtol=0.0)


def test_regularised_settle_and_settle_scan_agree(
    fx_states_prev, fx_observation, fx_control_input
):
    """The two inference implementations must include activity regularisation
    identically and remain numerically equivalent when weight regularisation
    is also active."""
    model = _regularised_model(
        weight_decay=0.05,
        weight_decay_scope="ff",
        orthogonal_penalty=0.03,
        orthogonal_scope="rec",
        activity_decay=0.04,
        activity_reg_type="l2",
    )

    settled = model.settle(
        fx_states_prev, fx_observation, fx_control_input,
        n_steps=15, state_lr=0.03,
    )
    settled_scan = model.settle_scan(
        optax.sgd(learning_rate=0.03),
        fx_states_prev, fx_observation, fx_control_input,
        n_steps=15,
    )

    for i, (a, b) in enumerate(zip(settled, settled_scan)):
        assert_allclose(
            a, b, f"regularised settle vs settle_scan state[{i}]",
            atol=3e-5, rtol=3e-5,
        )

    e1 = model.tpch_energy_fn(
        fx_states_prev, settled, fx_observation, fx_control_input
    )
    e2 = model.tpch_energy_fn(
        fx_states_prev, settled_scan, fx_observation, fx_control_input
    )
    assert_allclose(
        e1, e2, "regularised settle vs settle_scan energy",
        atol=3e-5, rtol=3e-5,
    )


def test_regularisation_config_round_trips_through_from_config_and_checkpoint(
    tmp_path
):
    """All regularisation settings are computational config and therefore
    must survive from_config() and ModelBase checkpoint round trips."""
    config = TpchConfig(
        control_layer_size=FX_CONTROL_SIZE,
        hidden_sizes=tuple(FX_HIDDEN_SIZES),
        obs_size=FX_OBS_SIZE,
        input_size=FX_INPUT_SIZE,
        act_fn="tanh",
        loss="mse",
        weight_decay=0.12,
        weight_decay_scope="ff",
        orthogonal_penalty=0.09,
        orthogonal_scope="rec",
        activity_decay=0.07,
        activity_reg_type="l2",
    )
    original = TpchModel.from_config(config, key=jr.key(104))
    rebuilt = TpchModel.from_config(config, key=jr.key(999))

    assert original.config == rebuilt.config
    assert original.config.weight_decay == 0.12
    assert original.config.weight_decay_scope == "ff"
    assert original.config.orthogonal_penalty == 0.09
    assert original.config.orthogonal_scope == "rec"
    assert original.config.activity_decay == 0.07
    assert original.config.activity_reg_type == "l2"

    out_dir = original.save_checkpoint(path=tmp_path / "tpch_reg_ckpt")
    loaded = TpchModel.load_checkpoint(out_dir)

    assert loaded.model.config == config
    assert loaded.model.config.weight_decay_scope == "ff"
    assert loaded.model.config.orthogonal_scope == "rec"
    assert loaded.model.config.activity_reg_type == "l2"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_decay_scope": "bad"},
        {"orthogonal_scope": "bad"},
        {"activity_reg_type": "bad"},
    ],
)
def test_regularisation_enum_validation(kwargs):
    """Unsupported regularisation scope/type strings must fail at model
    construction rather than being silently treated as another option."""
    with pytest.raises(ValueError):
        _regularised_model(**kwargs)


def test_overlapping_weight_and_orthogonal_regularisation_warns_at_collapse_threshold():
    """The documented no-nonzero-equilibrium threshold must emit a warning
    exactly when positive, overlapping regularisers satisfy
    weight_decay >= 2 * orthogonal_penalty."""
    with pytest.warns(UserWarning, match="no stable nonzero equilibrium"):
        _regularised_model(
            weight_decay=0.2,
            weight_decay_scope="rec",
            orthogonal_penalty=0.1,
            orthogonal_scope="rec",
        )


def test_non_overlapping_weight_and_orthogonal_regularisation_does_not_warn():
    """The same coefficients are harmless when the two regularisers operate on
    disjoint parameter sets."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _regularised_model(
            weight_decay=0.2,
            weight_decay_scope="ff",
            orthogonal_penalty=0.1,
            orthogonal_scope="rec",
        )
    assert not caught


def test_regularised_param_grad_structure_still_matches_model(
    fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    """Adding regularisation must not alter the PyTree structure required by
    eqx.apply_updates/Optax."""
    model = _regularised_model(
        weight_decay=0.1,
        orthogonal_penalty=0.07,
        activity_decay=0.05,
        activity_reg_type="l1",
    )
    grads = model.param_grad(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input
    )
    model_arrays = eqx.filter(model, eqx.is_array)
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(model_arrays)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.all(jnp.isfinite(leaf))



# =============================================================================
# J. layer_labels() -- ModelBase's optional labelling hook. The generic
#    default-raises / implemented-subclass contract is tested once,
#    model-agnostically, in model_base_test.py against a dummy model (same
#    split as save_checkpoint/load_checkpoint in section H). These just
#    check TpchModel's actual labels and their relationship to
#    return_layerwise's output (section K).
# =============================================================================

def test_layer_labels_matches_fx_model_hidden_layer_count(fx_model):
    assert TpchModel.layer_labels(fx_model.config) == ["Control", "Hidden 1", "Hidden 2", "Observation"]


def test_layer_labels_length_matches_return_layerwise_breakdown(
    fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    labels = TpchModel.layer_labels(fx_model.config)
    breakdown = fx_model.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    assert len(labels) == breakdown.shape[0]


def test_layer_labels_zero_hidden_layers():
    config = TpchConfig(control_layer_size=4, hidden_sizes=(), obs_size=6, input_size=2)
    assert TpchModel.layer_labels(config) == ["Control", "Observation"]


@pytest.mark.parametrize("n_hidden", [0, 1, 4])
def test_layer_labels_various_hidden_layer_counts(n_hidden):
    config = TpchConfig(control_layer_size=4, hidden_sizes=tuple(range(3, 3 + n_hidden)), obs_size=6)
    labels = TpchModel.layer_labels(config)
    assert labels[0] == "Control"
    assert labels[-1] == "Observation"
    assert labels[1:-1] == [f"Hidden {i + 1}" for i in range(n_hidden)]
    assert len(labels) == n_hidden + 2


def test_layer_labels_depends_only_on_config_not_a_built_model():
    """layer_labels is a classmethod taking `config`, not `self` -- it must
    work from a bare TpchConfig with no TpchModel ever constructed."""
    config = TpchConfig(control_layer_size=4, hidden_sizes=(3, 5), obs_size=6)
    assert TpchModel.layer_labels(config) == ["Control", "Hidden 1", "Hidden 2", "Observation"]


# =============================================================================
# K. Layerwise energy (tpch_energy_fn(..., return_layerwise=True),
#    settle_scan(..., return_layerwise=True))
#
# Same isolation philosophy as section I: build ground truth independently
# of the implementation wherever possible (this file's own _weight_entries/
# _recurrent_names, grouped by owning layer, rather than calling the
# implementation's _weight_l2_reg_by_layer/_weight_orthogonal_reg_by_layer
# directly) so a broken attribution can't "pass" just because it's
# internally self-consistent.
#
# Unlike an earlier version of tpch_energy_fn, return_layerwise=True now
# includes regularisation, attributed EXACTLY (not approximated) to
# whichever layer's own weights/activities produced it -- these tests are
# the ground-truth check on that exactness claim specifically.
# =============================================================================


def _owning_layer_idx(name: str, model) -> int:
    """Maps a _weight_entries() name (this test file's own, independent of
    the implementation) to the owning-layer index used by layer_labels()
    and return_layerwise's breakdown: 0=control, 1..n=hidden, n+1=observation.
    """
    if name.startswith("control"):
        return 0
    if name.startswith("hidden["):
        i = int(name[len("hidden["):name.index("]")])
        return i + 1
    if name.startswith("observation"):
        return len(model.hidden_layers) + 1
    raise ValueError(f"unrecognised weight name: {name}")


# ---- K1. Basic shape/consistency -------------------------------------------

def test_return_layerwise_shape_matches_states_plus_one(
    fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    breakdown = fx_model.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    assert breakdown.shape == (len(fx_states_curr) + 1,)
    assert jnp.all(jnp.isfinite(breakdown))


def test_return_layerwise_is_a_stacked_array_not_a_python_list(
    fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    """Regression test: an earlier draft of tpch_energy_fn returned a plain
    Python list under return_layerwise=True, which breaks settle_scan's
    jax.vmap(energy_trace_fn)(states_hist) (vmap needs a stackable array
    back, not a list)."""
    breakdown = fx_model.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    assert isinstance(breakdown, jax.Array)


# ---- K2. Task-only breakdown matches the independent layer_energies() helper

def test_return_layerwise_matches_manual_layer_energies_helper_at_zero_reg(
    fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    """With all regularisation off, return_layerwise's breakdown must equal
    this file's own hand-written per-layer task-error formula exactly."""
    manual = jnp.stack(layer_energies(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input))
    actual = fx_model.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    assert_allclose(actual, manual, "return_layerwise vs manual layer_energies (zero reg)", atol=1e-5, rtol=1e-5)


def test_return_layerwise_matches_manual_layer_energies_helper_with_ce_loss(
    fx_states_prev, fx_states_curr, fx_control_input
):
    """The loss switch (section on loss, if present) must also be reflected
    correctly in the layerwise breakdown's observation entry, not just the
    scalar total."""
    model = TpchModel(
        control_layer_size=FX_CONTROL_SIZE, hidden_sizes=FX_HIDDEN_SIZES, obs_size=FX_OBS_SIZE,
        key=jr.key(210), input_size=FX_INPUT_SIZE, loss="ce",
    )
    observation = jax.nn.one_hot(2, FX_OBS_SIZE)
    manual = jnp.stack(layer_energies(model, fx_states_prev, fx_states_curr, observation, fx_control_input))
    actual = model.tpch_energy_fn(fx_states_prev, fx_states_curr, observation, fx_control_input, return_layerwise=True)
    assert_allclose(actual, manual, "return_layerwise vs manual layer_energies (ce loss)", atol=1e-5, rtol=1e-5)


# ---- K3. sum(return_layerwise) == scalar total, across regularisation configs

@pytest.mark.parametrize(
    "reg_kwargs",
    [
        {},
        {"weight_decay": 0.2, "weight_decay_scope": "all"},
        {"weight_decay": 0.2, "weight_decay_scope": "rec"},
        {"weight_decay": 0.2, "weight_decay_scope": "ff"},
        {"orthogonal_penalty": 0.15, "orthogonal_scope": "rec"},
        {"orthogonal_penalty": 0.15, "orthogonal_scope": "all"},
        {"activity_decay": 0.1, "activity_reg_type": "l1"},
        {"activity_decay": 0.1, "activity_reg_type": "l2"},
        {
            "weight_decay": 0.13, "weight_decay_scope": "ff",
            "orthogonal_penalty": 0.09, "orthogonal_scope": "rec",
            "activity_decay": 0.07, "activity_reg_type": "l2",
        },
    ],
)
def test_return_layerwise_sum_equals_scalar_total(
    fx_states_prev, fx_states_curr, fx_observation, fx_control_input, reg_kwargs
):
    """jnp.sum(return_layerwise output) must exactly equal the scalar
    (return_layerwise=False) total on identical inputs -- this is the core
    contract that makes the breakdown trustworthy for diagnostics: it's a
    genuine partition of the real total, not a different, smaller quantity."""
    model = _regularised_model(**reg_kwargs)
    scalar_total = model.tpch_energy_fn(fx_states_prev, fx_states_curr, fx_observation, fx_control_input)
    breakdown = model.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    assert_allclose(jnp.sum(breakdown), scalar_total, f"sum(breakdown) vs scalar total {reg_kwargs}", atol=1e-4, rtol=1e-4)


# ---- K4. Exact per-layer weight_decay attribution (independent formula) ----

@pytest.mark.parametrize("scope", ["rec", "ff", "all"])
def test_return_layerwise_weight_decay_attributed_to_correct_layer(fx_zero_task_point, scope):
    """Each entry of the return_layerwise breakdown must equal exactly the
    weight_decay penalty owed by THAT layer's own weights -- computed here
    by grouping this test file's independent _weight_entries() by owning
    layer, not by calling the implementation's internal grouping helpers.
    """
    _, states_prev, states_curr, observation, control_input = fx_zero_task_point
    coefficient = 0.29
    reg = _regularised_model(weight_decay=coefficient, weight_decay_scope=scope)

    n_groups = len(reg.hidden_layers) + 2
    expected = [0.0] * n_groups
    recurrent = _recurrent_names(reg)
    for name, weight in _weight_entries(reg):
        selected = (
            scope == "all"
            or (scope == "rec" and name in recurrent)
            or (scope == "ff" and name not in recurrent)
        )
        if selected:
            expected[_owning_layer_idx(name, reg)] += 0.5 * coefficient * float(jnp.sum(weight ** 2))

    breakdown = reg.tpch_energy_fn(states_prev, states_curr, observation, control_input, return_layerwise=True)
    for i, exp in enumerate(expected):
        assert_allclose(
            breakdown[i], exp, f"weight_decay attribution, layer {i} ({TpchModel.layer_labels(reg.config)[i]}), scope={scope}",
            atol=2e-6, rtol=2e-6,
        )


@pytest.mark.parametrize("scope", ["rec", "ff", "all"])
def test_return_layerwise_orthogonal_penalty_attributed_to_correct_layer(fx_zero_task_point, scope):
    """Same exactness check as above, for orthogonal_penalty, reusing the
    same Gram-matrix ground truth as test_orthogonal_penalty_energy_matches_exact_definition."""
    _, states_prev, states_curr, observation, control_input = fx_zero_task_point
    coefficient = 0.17
    reg = _regularised_model(orthogonal_penalty=coefficient, orthogonal_scope=scope)

    n_groups = len(reg.hidden_layers) + 2
    expected = [0.0] * n_groups
    recurrent = _recurrent_names(reg)
    for name, weight in _weight_entries(reg):
        selected = (
            scope == "all"
            or (scope == "rec" and name in recurrent)
            or (scope == "ff" and name not in recurrent)
        )
        if selected:
            out_dim, in_dim = weight.shape
            if in_dim <= out_dim:
                gram = weight.T @ weight
                identity = jnp.eye(in_dim, dtype=weight.dtype)
            else:
                gram = weight @ weight.T
                identity = jnp.eye(out_dim, dtype=weight.dtype)
            expected[_owning_layer_idx(name, reg)] += float(0.5 * coefficient * jnp.sum((identity - gram) ** 2))

    breakdown = reg.tpch_energy_fn(states_prev, states_curr, observation, control_input, return_layerwise=True)
    for i, exp in enumerate(expected):
        assert_allclose(
            breakdown[i], exp, f"orthogonal attribution, layer {i} ({TpchModel.layer_labels(reg.config)[i]}), scope={scope}",
            atol=2e-5, rtol=2e-5,
        )


# ---- K5. Exact per-layer activity_decay attribution (trivially exact:
#      states_curr is already one-per-layer, no grouping needed) ----------

@pytest.mark.parametrize("reg_type", ["l1", "l2"])
def test_return_layerwise_activity_decay_attributed_to_correct_layer(
    fx_states_prev, fx_states_curr, fx_observation, fx_control_input, reg_type
):
    coefficient = 0.31
    reg = _regularised_model(activity_decay=coefficient, activity_reg_type=reg_type)

    if reg_type == "l1":
        expected_per_state = [0.5 * coefficient * float(jnp.sum(jnp.abs(s))) for s in fx_states_curr]
    else:
        expected_per_state = [0.5 * coefficient * float(jnp.sum(s ** 2)) for s in fx_states_curr]

    breakdown = reg.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    manual_task = layer_energies(reg, fx_states_prev, fx_states_curr, fx_observation, fx_control_input)

    for i, expected_activity in enumerate(expected_per_state):
        expected_total = float(manual_task[i]) + expected_activity
        assert_allclose(
            breakdown[i], expected_total, f"activity {reg_type} attribution, layer {i}",
            atol=2e-6, rtol=2e-6,
        )


def test_return_layerwise_activity_decay_observation_entry_is_task_only(
    fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    """states_curr has no observation-layer entry (the observation is
    external ground truth y, not a state), so activity_decay must
    contribute exactly zero to the breakdown's LAST (observation) entry --
    that entry should be pure task/loss error, unaffected by activity_decay."""
    coefficient = 0.5
    base = _regularised_model()
    reg = _regularised_model(activity_decay=coefficient, activity_reg_type="l2")

    breakdown_base = base.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    breakdown_reg = reg.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    assert_allclose(
        breakdown_reg[-1], breakdown_base[-1], "observation entry unaffected by activity_decay",
        atol=0.0, rtol=0.0,
    )
    # sanity: some OTHER entry must actually have changed, or this test
    # would trivially pass even if activity_decay were completely broken
    assert not bool(jnp.allclose(breakdown_reg[:-1], breakdown_base[:-1]))


# ---- K6. return_layerwise under jit / grad / vmap ---------------------------
#
# tpch_energy_fn builds return_layerwise's output via a plain Python
# list.append() inside a for loop over states_curr. That's only safe under
# jax transforms because len(states_curr) (the number of layers) is a
# static property of the model's architecture, not a traced value -- the
# loop unrolls at trace time, same as every other Python loop in tpch.py
# (_all_weights, _weight_entries, predict, etc). These are regression tests
# for that property specifically, not just "does return_layerwise work".

def test_return_layerwise_compiles_under_eqx_filter_jit(
    fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    energy_fn = eqx.filter_jit(
        lambda m, sp, sc, y, x: m.tpch_energy_fn(sp, sc, y, x, return_layerwise=True)
    )
    out = energy_fn(fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input)
    direct = fx_model.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    assert_allclose(out, direct, "jit vs eager return_layerwise", atol=1e-5, rtol=1e-5)


def test_return_layerwise_differentiable_wrt_states(
    fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    grad_out = jax.grad(
        lambda sc: jnp.sum(fx_model.tpch_energy_fn(
            fx_states_prev, sc, fx_observation, fx_control_input, return_layerwise=True
        ))
    )(fx_states_curr)
    assert len(grad_out) == len(fx_states_curr)
    for g, s in zip(grad_out, fx_states_curr):
        assert g.shape == s.shape
        assert jnp.all(jnp.isfinite(g))


def test_return_layerwise_vmaps_over_a_batch_of_states(
    fx_model, fx_states_prev, fx_states_curr, fx_observation, fx_control_input
):
    """This is exactly the mechanism settle_scan(return_layerwise=True) relies
    on internally (jax.vmap over the recorded states_hist trajectory) --
    tested directly here, independent of settle_scan, to isolate the
    property from settle_scan's own scan/optax machinery."""
    batched = [jnp.stack([s, s * 1.5]) for s in fx_states_curr]
    vmap_out = jax.vmap(
        lambda sc: fx_model.tpch_energy_fn(fx_states_prev, sc, fx_observation, fx_control_input, return_layerwise=True)
    )(batched)
    assert vmap_out.shape == (2, len(fx_states_curr) + 1)
    direct_0 = fx_model.tpch_energy_fn(
        fx_states_prev, fx_states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    assert_allclose(vmap_out[0], direct_0, "vmap row 0 vs direct call", atol=1e-5, rtol=1e-5)


# ---- K7. settle_scan(return_layerwise=True) --------------------------------

def test_settle_scan_return_layerwise_false_returns_states_only(
    fx_model, fx_states_prev, fx_control_input, fx_observation
):
    """The default (False) must return exactly what it always returned --
    just states_curr, not a tuple -- regardless of what return_layerwise=True
    now does."""
    opt = optax.sgd(learning_rate=0.05)
    result = fx_model.settle_scan(opt, fx_states_prev, fx_observation, fx_control_input, n_steps=10)
    assert isinstance(result, list)
    assert len(result) == len(fx_states_prev)


def test_settle_scan_return_layerwise_true_states_match_false(
    fx_model, fx_states_prev, fx_control_input, fx_observation
):
    opt = optax.sgd(learning_rate=0.05)
    states_default = fx_model.settle_scan(opt, fx_states_prev, fx_observation, fx_control_input, n_steps=10)
    states_curr, trace = fx_model.settle_scan(
        opt, fx_states_prev, fx_observation, fx_control_input, n_steps=10, return_layerwise=True
    )
    for a, b in zip(states_default, states_curr):
        assert_allclose(a, b, "settle_scan states identical with/without return_layerwise", atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "reg_kwargs",
    [
        {},
        {"weight_decay": 0.1, "weight_decay_scope": "all"},
        {"orthogonal_penalty": 0.1, "orthogonal_scope": "rec"},
        {"activity_decay": 0.1, "activity_reg_type": "l1"},
    ],
)
def test_settle_scan_return_layerwise_trace_shape_and_sum_matches_scalar(
    fx_states_prev, fx_control_input, fx_observation, reg_kwargs
):
    model = _regularised_model(**reg_kwargs)
    opt = optax.sgd(learning_rate=0.05)
    n_steps = 12
    states_curr, trace = model.settle_scan(
        opt, fx_states_prev, fx_observation, fx_control_input, n_steps=n_steps, return_layerwise=True
    )
    assert trace.shape == (n_steps, len(fx_states_prev) + 1)
    assert jnp.all(jnp.isfinite(trace))

    # every row's sum must equal a scalar tpch_energy_fn call on that
    # row's own (recorded) states -- checked here for just the last row,
    # which we can reconstruct directly (see next test for interior rows)
    direct_final_breakdown = model.tpch_energy_fn(
        fx_states_prev, states_curr, fx_observation, fx_control_input, return_layerwise=True
    )
    assert_allclose(trace[-1], direct_final_breakdown, "trace last row vs direct recomputation", atol=1e-4, rtol=1e-4)


def test_settle_scan_return_layerwise_regularisation_rows_are_constant_across_steps(
    fx_states_prev, fx_control_input, fx_observation
):
    """Weights are frozen for the entire duration of one settle_scan call
    (they only change via param_grad + an optimiser step, between calls) --
    so the REGULARISATION contribution to the trace must be bit-identical
    at every one of the n_steps rows, even though the TASK-error part
    varies as the states relax. This is a structural property of the
    physics, not an implementation detail -- if this ever fails, weights
    are leaking into the trace somehow."""
    model = _regularised_model(weight_decay=0.15, weight_decay_scope="all", orthogonal_penalty=0.08, orthogonal_scope="rec")
    base = _regularised_model()  # zero regularisation, for isolating the reg-only contribution
    opt = optax.sgd(learning_rate=0.05)
    n_steps = 10

    _, trace_reg = model.settle_scan(opt, fx_states_prev, fx_observation, fx_control_input, n_steps=n_steps, return_layerwise=True)
    _, trace_base = base.settle_scan(opt, fx_states_prev, fx_observation, fx_control_input, n_steps=n_steps, return_layerwise=True)

    # NOTE: task-error rows differ between model/base because activity
    # regularisation (none here) would change the inference trajectory --
    # with only weight-only regularisers on, the trajectories are IDENTICAL
    # (weight_decay/orthogonal contribute zero gradient to states), so the
    # difference isolates purely the regularisation contribution at each step.
    reg_only_per_step = trace_reg - trace_base
    for t in range(1, n_steps):
        assert_allclose(
            reg_only_per_step[t], reg_only_per_step[0], f"regularisation contribution constant across steps (row {t})",
            atol=1e-5, rtol=1e-5,
        )
    # and it should be strictly nonzero (a broken/no-op regulariser would
    # trivially satisfy the "constant" check above with all zeros)
    assert not bool(jnp.allclose(reg_only_per_step[0], jnp.zeros_like(reg_only_per_step[0])))


# =============================================================================
# L. make_train_step -- module-level factory (not a TpchModel method) that
#    fuses settle_scan + logging quantities + one param_grad/optax weight
#    update into a single eqx.filter_jit-compiled function, built once per
#    training run and called once per frame.
#
# The strongest tests here (L4/L5) manually recompose the exact same
# sequence of calls WITHOUT make_train_step and check for agreement --
# this is what actually validates the fused version matches its documented
# behaviour, rather than just "it runs and returns finite numbers".
# =============================================================================

def _manual_train_step(model, param_optim, activity_optim, param_opt_state, states_prev, y, control_input, n_infer_steps):
    """Direct (unfused, eager) re-implementation of make_train_step's
    train_step body, built from public TpchModel methods only. Used as an
    independent ground truth for the tests below."""
    states_curr_init = model.init_activities(states_prev, control_input)
    _, y_hat_before = model.predict(states_prev, states_curr_init, control_input)
    energy_before = model.tpch_energy_fn(states_prev, states_curr_init, y, control_input)

    states_curr = model.settle_scan(activity_optim, states_prev, y, control_input, n_steps=n_infer_steps)

    _, y_hat_after = model.predict(states_prev, states_curr, control_input)
    energy_after = model.tpch_energy_fn(states_prev, states_curr, y, control_input)

    grads = model.param_grad(states_prev, states_curr, y, control_input)
    updates, new_opt_state = param_optim.update(grads, param_opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_opt_state, states_curr, y_hat_before, y_hat_after, energy_before, energy_after


@pytest.fixture
def fx_train_step_setup(fx_model, fx_states_prev, fx_observation, fx_control_input):
    """Common setup for make_train_step tests: a param optimiser + its
    initial state, matched to fx_model."""
    param_optim = optax.adam(learning_rate=1e-3)
    activity_optim = optax.adam(learning_rate=1e-2)
    param_opt_state = param_optim.init(eqx.filter(fx_model, eqx.is_array))
    return param_optim, activity_optim, param_opt_state


# ---- L1. Output contract ----------------------------------------------------

def test_make_train_step_default_returns_none_energy_trace(
    fx_model, fx_states_prev, fx_observation, fx_control_input, fx_train_step_setup
):
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    n_infer_steps = 8
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps, fx_control_input)

    result = train_step(fx_model, param_opt_state, fx_states_prev, fx_observation)
    assert len(result) == 8
    model, opt_state, states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace = result
    assert energy_trace is None
    assert len(states_curr) == len(fx_states_prev)
    assert y_hat_before.shape == fx_observation.shape
    assert y_hat_after.shape == fx_observation.shape
    assert jnp.isfinite(energy_before)
    assert jnp.isfinite(energy_after)


def test_make_train_step_return_layerwise_true_gives_correctly_shaped_trace(
    fx_model, fx_states_prev, fx_observation, fx_control_input, fx_train_step_setup
):
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    n_infer_steps = 8
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps, fx_control_input)

    result = train_step(fx_model, param_opt_state, fx_states_prev, fx_observation, return_layerwise=True)
    energy_trace = result[-1]
    assert energy_trace is not None
    assert energy_trace.shape == (n_infer_steps, len(fx_states_prev) + 1)
    assert jnp.all(jnp.isfinite(energy_trace))


def test_make_train_step_weight_update_actually_changes_weights(
    fx_model, fx_states_prev, fx_observation, fx_control_input, fx_train_step_setup
):
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    train_step = make_train_step(param_optim, activity_optim, 8, fx_control_input)

    new_model, *_ = train_step(fx_model, param_opt_state, fx_states_prev, fx_observation)

    before_leaves = jax.tree_util.tree_leaves(eqx.filter(fx_model, eqx.is_array))
    after_leaves = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))
    assert len(before_leaves) == len(after_leaves)
    assert any(not bool(jnp.array_equal(b, a)) for b, a in zip(before_leaves, after_leaves))
    for leaf in after_leaves:
        assert jnp.all(jnp.isfinite(leaf))


# ---- L4. Manual recomposition -- the strongest correctness check ----------

def test_make_train_step_matches_manual_recomposition(
    fx_model, fx_states_prev, fx_observation, fx_control_input, fx_train_step_setup
):
    """train_step's fused/jitted computation must agree with calling the
    exact same sequence of public methods by hand, eagerly, outside any
    make_train_step machinery. This is the test that actually validates
    the fusion didn't silently change what gets computed."""
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    n_infer_steps = 8
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps, fx_control_input)

    fused_model, fused_opt_state, fused_states, fused_y_before, fused_y_after, fused_e_before, fused_e_after, _ = train_step(
        fx_model, param_opt_state, fx_states_prev, fx_observation
    )
    manual_model, manual_opt_state, manual_states, manual_y_before, manual_y_after, manual_e_before, manual_e_after = _manual_train_step(
        fx_model, param_optim, activity_optim, param_opt_state, fx_states_prev, fx_observation, fx_control_input, n_infer_steps
    )

    for f, m in zip(fused_states, manual_states):
        assert_allclose(f, m, "states_curr: fused vs manual", atol=1e-4, rtol=1e-4)
    assert_allclose(fused_y_before, manual_y_before, "y_hat_before: fused vs manual", atol=1e-4, rtol=1e-4)
    assert_allclose(fused_y_after, manual_y_after, "y_hat_after: fused vs manual", atol=1e-4, rtol=1e-4)
    assert_allclose(fused_e_before, manual_e_before, "energy_before: fused vs manual", atol=1e-4, rtol=1e-4)
    assert_allclose(fused_e_after, manual_e_after, "energy_after: fused vs manual", atol=1e-4, rtol=1e-4)

    fused_arrays = jax.tree_util.tree_leaves(eqx.filter(fused_model, eqx.is_array))
    manual_arrays = jax.tree_util.tree_leaves(eqx.filter(manual_model, eqx.is_array))
    for f, m in zip(fused_arrays, manual_arrays):
        assert_allclose(f, m, "updated model weights: fused vs manual", atol=1e-4, rtol=1e-4)


# ---- L5. return_layerwise must not perturb training dynamics --------------

def test_make_train_step_return_layerwise_does_not_change_training_dynamics(
    fx_model, fx_states_prev, fx_observation, fx_control_input, fx_train_step_setup
):
    """The two compiled traces (return_layerwise True/False) must agree on
    everything except the trace itself -- return_layerwise is purely an
    additional diagnostic readout, not a different computation path."""
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    train_step = make_train_step(param_optim, activity_optim, 8, fx_control_input)

    result_false = train_step(fx_model, param_opt_state, fx_states_prev, fx_observation, return_layerwise=False)
    result_true = train_step(fx_model, param_opt_state, fx_states_prev, fx_observation, return_layerwise=True)

    for i, label in enumerate([
        "model", "param_opt_state", "states_curr", "y_hat_before", "y_hat_after", "energy_before", "energy_after"
    ]):
        a = jax.tree_util.tree_leaves(result_false[i])
        b = jax.tree_util.tree_leaves(result_true[i])
        for la, lb in zip(a, b):
            assert_allclose(la, lb, f"{label}: return_layerwise=False vs True", atol=1e-5, rtol=1e-5)

    assert result_false[-1] is None
    assert result_true[-1] is not None


# ---- L6. Multi-step training-loop integration ------------------------------

def test_make_train_step_multi_step_loop_threads_state_correctly(
    fx_model, fx_control_input, fx_train_step_setup
):
    """Simulates a real training loop: repeatedly call train_step, threading
    model/param_opt_state/states_curr through as the next call's inputs,
    exactly as the documented usage pattern describes."""
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    train_step = make_train_step(param_optim, activity_optim, 6, fx_control_input)

    model = fx_model
    sizes = [FX_CONTROL_SIZE] + list(FX_HIDDEN_SIZES)
    states_prev = [jr.normal(k, (n,)) for k, n in zip(jr.split(jr.key(500), len(sizes)), sizes)]

    weight_snapshots = []
    for i in range(5):
        y = jr.normal(jr.fold_in(jr.key(501), i), (FX_OBS_SIZE,))
        do_record = i % 2 == 0
        model, param_opt_state, states_curr, y_hat_before, y_hat_after, e_before, e_after, e_trace = train_step(
            model, param_opt_state, states_prev, y, return_layerwise=do_record
        )
        assert jnp.isfinite(e_before)
        assert jnp.isfinite(e_after)
        assert (e_trace is not None) == do_record
        for s in states_curr:
            assert jnp.all(jnp.isfinite(s))
        weight_snapshots.append(jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))[0].copy())
        states_prev = states_curr  # thread forward, as the real loop does

    # weights should have moved at every step (not stuck/frozen)
    for a, b in zip(weight_snapshots[:-1], weight_snapshots[1:]):
        assert not bool(jnp.array_equal(a, b))


# ---- L7. n_infer_steps is actually wired to settle_scan --------------------

def test_make_train_step_n_infer_steps_matches_settle_scan_n_steps(
    fx_model, fx_states_prev, fx_observation, fx_control_input, fx_train_step_setup
):
    """states_curr returned by train_step (computed BEFORE the weight update)
    must exactly match a direct model.settle_scan(..., n_steps=n_infer_steps)
    call on the same, still-unmodified model -- confirms n_infer_steps is
    genuinely passed through, not silently ignored or hardcoded."""
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    n_infer_steps = 11
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps, fx_control_input)

    _, _, states_curr, *_ = train_step(fx_model, param_opt_state, fx_states_prev, fx_observation)
    direct = fx_model.settle_scan(activity_optim, fx_states_prev, fx_observation, fx_control_input, n_steps=n_infer_steps)

    for a, b in zip(states_curr, direct):
        assert_allclose(a, b, "train_step's states_curr vs direct settle_scan(n_steps=n_infer_steps)", atol=1e-4, rtol=1e-4)


def test_make_train_step_different_n_infer_steps_give_different_settled_states(
    fx_model, fx_states_prev, fx_observation, fx_control_input, fx_train_step_setup
):
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    train_step_short = make_train_step(param_optim, activity_optim, 2, fx_control_input)
    train_step_long = make_train_step(param_optim, activity_optim, 40, fx_control_input)

    _, _, states_short, *_ = train_step_short(fx_model, param_opt_state, fx_states_prev, fx_observation)
    _, _, states_long, *_ = train_step_long(fx_model, param_opt_state, fx_states_prev, fx_observation)

    assert any(
        not bool(jnp.allclose(s, l, atol=1e-4, rtol=1e-4)) for s, l in zip(states_short, states_long)
    )


# ---- L8. control_input is genuinely used, not silently dropped ------------

def test_make_train_step_control_input_actually_affects_predictions(
    fx_model, fx_states_prev, fx_observation, fx_train_step_setup
):
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    control_input = jr.normal(jr.key(502), (FX_INPUT_SIZE,))

    train_step_with_input = make_train_step(param_optim, activity_optim, 8, control_input)
    train_step_without_input = make_train_step(param_optim, activity_optim, 8, None)

    _, _, _, y_before_with, *_ = train_step_with_input(fx_model, param_opt_state, fx_states_prev, fx_observation)
    _, _, _, y_before_without, *_ = train_step_without_input(fx_model, param_opt_state, fx_states_prev, fx_observation)

    assert not bool(jnp.allclose(y_before_with, y_before_without, atol=1e-4, rtol=1e-4))


# ---- L9/L10. Edge cases: zero hidden layers, ce loss -----------------------

def test_make_train_step_zero_hidden_layers():
    model = TpchModel(control_layer_size=4, hidden_sizes=[], obs_size=5, key=jr.key(503), input_size=2)
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    param_opt_state = param_optim.init(eqx.filter(model, eqx.is_array))
    control_input = jr.normal(jr.key(504), (2,))
    train_step = make_train_step(param_optim, activity_optim, 6, control_input)

    states_prev = [jr.normal(jr.key(505), (4,))]
    y = jr.normal(jr.key(506), (5,))
    new_model, new_opt_state, states_curr, y_before, y_after, e_before, e_after, trace = train_step(
        model, param_opt_state, states_prev, y, return_layerwise=True
    )
    assert len(states_curr) == 1
    assert trace.shape == (6, 2)  # control + observation only
    assert jnp.isfinite(e_after)


def test_make_train_step_with_ce_loss():
    model = TpchModel(
        control_layer_size=FX_CONTROL_SIZE, hidden_sizes=FX_HIDDEN_SIZES, obs_size=FX_OBS_SIZE,
        key=jr.key(507), input_size=FX_INPUT_SIZE, loss="ce",
    )
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    param_opt_state = param_optim.init(eqx.filter(model, eqx.is_array))
    control_input = jr.normal(jr.key(508), (FX_INPUT_SIZE,))
    train_step = make_train_step(param_optim, activity_optim, 8, control_input)

    sizes = [FX_CONTROL_SIZE] + list(FX_HIDDEN_SIZES)
    states_prev = [jr.normal(k, (n,)) for k, n in zip(jr.split(jr.key(509), len(sizes)), sizes)]
    y = jax.nn.one_hot(2, FX_OBS_SIZE)

    new_model, new_opt_state, states_curr, y_before, y_after, e_before, e_after, trace = train_step(
        model, param_opt_state, states_prev, y
    )
    assert jnp.isfinite(e_before)
    assert jnp.isfinite(e_after)
    weight_before = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))[0]
    weight_after = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))[0]
    assert not bool(jnp.array_equal(weight_before, weight_after))





# =============================================================================
# M. make_train_run -- module-level factory (same reasoning as
#    make_train_step: model/param_opt_state must be call-time arguments,
#    not closed over, since both change every frame). Fuses settle+learn
#    for `run_length` consecutive frames into ONE jax.lax.scan.
#
# The strongest tests here (M3, M5) check agreement against completely
# independent references: M3 against make_train_step looped by hand (no
# make_train_run involved at all), M5 against calling make_train_run
# itself with a different run_length/call pattern over the same data --
# i.e. the "goldilocks blocks == one big run" equivalence this function's
# whole design rests on.
# =============================================================================

@pytest.fixture
def fx_train_run_setup(fx_model):
    param_optim = optax.adam(learning_rate=1e-3)
    activity_optim = optax.adam(learning_rate=1e-2)
    param_opt_state = param_optim.init(eqx.filter(fx_model, eqx.is_array))
    return param_optim, activity_optim, param_opt_state


def _fresh_states_prev(seed):
    sizes = [FX_CONTROL_SIZE] + list(FX_HIDDEN_SIZES)
    return [jr.normal(k, (n,)) for k, n in zip(jr.split(jr.key(seed), len(sizes)), sizes)]


# ---- M1. Output contract ----------------------------------------------------

def test_make_train_run_default_returns_none_energy_traces(fx_model, fx_control_input, fx_train_run_setup):
    param_optim, activity_optim, param_opt_state = fx_train_run_setup
    run_length = 5
    train_run = make_train_run(param_optim, activity_optim, n_infer_steps=6, run_length=run_length, control_input=fx_control_input)

    states_prev = _fresh_states_prev(600)
    ys = jr.normal(jr.key(601), (run_length, FX_OBS_SIZE))

    result = train_run(fx_model, param_opt_state, states_prev, ys)
    assert len(result) == 8
    model, opt_state, states_curr, y_before, y_after, energies_before, energies_after, energy_traces = result
    assert energy_traces is None
    assert len(states_curr) == len(states_prev)
    assert y_before.shape == (run_length, FX_OBS_SIZE)
    assert y_after.shape == (run_length, FX_OBS_SIZE)
    assert energies_before.shape == (run_length,)
    assert energies_after.shape == (run_length,)
    assert jnp.all(jnp.isfinite(energies_before))
    assert jnp.all(jnp.isfinite(energies_after))


def test_make_train_run_return_layerwise_true_gives_correctly_shaped_traces(fx_model, fx_control_input, fx_train_run_setup):
    param_optim, activity_optim, param_opt_state = fx_train_run_setup
    run_length, n_infer_steps = 5, 7
    train_run = make_train_run(param_optim, activity_optim, n_infer_steps, run_length=run_length, control_input=fx_control_input)

    states_prev = _fresh_states_prev(602)
    ys = jr.normal(jr.key(603), (run_length, FX_OBS_SIZE))

    *_, energy_traces = train_run(fx_model, param_opt_state, states_prev, ys, return_layerwise=True)
    assert energy_traces.shape == (run_length, n_infer_steps, len(FX_HIDDEN_SIZES) + 2)
    assert jnp.all(jnp.isfinite(energy_traces))


def test_make_train_run_weight_update_actually_changes_weights(fx_model, fx_control_input, fx_train_run_setup):
    param_optim, activity_optim, param_opt_state = fx_train_run_setup
    train_run = make_train_run(param_optim, activity_optim, n_infer_steps=6, run_length=4, control_input=fx_control_input)

    states_prev = _fresh_states_prev(604)
    ys = jr.normal(jr.key(605), (4, FX_OBS_SIZE))
    new_model, *_ = train_run(fx_model, param_opt_state, states_prev, ys)

    before = jax.tree_util.tree_leaves(eqx.filter(fx_model, eqx.is_array))
    after = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))
    assert any(not bool(jnp.array_equal(b, a)) for b, a in zip(before, after))
    for leaf in after:
        assert jnp.all(jnp.isfinite(leaf))


# ---- M3. Manual recomposition via make_train_step, looped by hand ---------

def test_make_train_run_matches_make_train_step_looped_manually(fx_model, fx_control_input, fx_train_run_setup):
    """make_train_run's fused multi-frame scan must agree with calling
    make_train_step (itself already independently verified) once per
    frame in an ordinary Python loop -- completely independent of
    make_train_run's own internals."""
    param_optim, activity_optim, param_opt_state = fx_train_run_setup
    n_infer_steps, run_length = 6, 5

    states_prev = _fresh_states_prev(606)
    ys = jr.normal(jr.key(607), (run_length, FX_OBS_SIZE))

    # reference: make_train_step, looped by hand
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps, fx_control_input)
    m, ops, sp = fx_model, param_opt_state, states_prev
    ref_energies_before, ref_energies_after, ref_y_before, ref_y_after = [], [], [], []
    for i in range(run_length):
        m, ops, sp, y_bef, y_aft, e_before, e_after, _ = train_step(m, ops, sp, ys[i])
        ref_energies_before.append(e_before)
        ref_energies_after.append(e_after)
        ref_y_before.append(y_bef)
        ref_y_after.append(y_aft)

    # fused
    train_run = make_train_run(param_optim, activity_optim, n_infer_steps, run_length=run_length, control_input=fx_control_input)
    fused_model, fused_ops, fused_sp, fused_y_before, fused_y_after, fused_energies_before, fused_energies_after, _ = train_run(
        fx_model, param_opt_state, states_prev, ys
    )

    assert_allclose(fused_energies_before, jnp.stack(ref_energies_before), "energies_before: make_train_run vs looped make_train_step", atol=1e-4, rtol=1e-4)
    assert_allclose(fused_energies_after, jnp.stack(ref_energies_after), "energies_after: make_train_run vs looped make_train_step", atol=1e-4, rtol=1e-4)
    assert_allclose(fused_y_before, jnp.stack(ref_y_before), "y_hat_before: make_train_run vs looped make_train_step", atol=1e-4, rtol=1e-4)
    assert_allclose(fused_y_after, jnp.stack(ref_y_after), "y_hat_after: make_train_run vs looped make_train_step", atol=1e-4, rtol=1e-4)
    for f, r in zip(fused_sp, sp):
        assert_allclose(f, r, "final states: make_train_run vs looped make_train_step", atol=1e-4, rtol=1e-4)

    fused_weights = jax.tree_util.tree_leaves(eqx.filter(fused_model, eqx.is_array))
    ref_weights = jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
    for f, r in zip(fused_weights, ref_weights):
        assert_allclose(f, r, "final weights: make_train_run vs looped make_train_step", atol=1e-4, rtol=1e-4)


# ---- M4. return_layerwise must not perturb training dynamics --------------

def test_make_train_run_return_layerwise_does_not_change_training_dynamics(fx_model, fx_control_input, fx_train_run_setup):
    param_optim, activity_optim, param_opt_state = fx_train_run_setup
    train_run = make_train_run(param_optim, activity_optim, n_infer_steps=6, run_length=4, control_input=fx_control_input)

    states_prev = _fresh_states_prev(608)
    ys = jr.normal(jr.key(609), (4, FX_OBS_SIZE))

    out_false = train_run(fx_model, param_opt_state, states_prev, ys, return_layerwise=False)
    out_true = train_run(fx_model, param_opt_state, states_prev, ys, return_layerwise=True)

    for i, label in enumerate(["model", "param_opt_state", "states_curr", "y_hat_before", "y_hat_after", "energies_before", "energies_after"]):
        a = jax.tree_util.tree_leaves(out_false[i])
        b = jax.tree_util.tree_leaves(out_true[i])
        for la, lb in zip(a, b):
            assert_allclose(la, lb, f"{label}: return_layerwise False vs True", atol=1e-5, rtol=1e-5)

    assert out_false[-1] is None
    assert out_true[-1] is not None


# ---- M5. Goldilocks blocks == one big run (the core design equivalence) ---

def test_make_train_run_goldilocks_blocks_match_one_big_run(fx_model, fx_control_input, fx_train_run_setup):
    """Calling train_run 3 times with run_length=3 (threading model/
    opt_state/states_prev through, exactly the goldilocks usage pattern)
    must produce the same final model/states as ONE call with
    run_length=9 over the same 9 frames -- this is the actual property
    that makes the goldilocks pattern trustworthy: splitting a run into
    blocks for side effects must not change what gets computed."""
    param_optim, activity_optim, param_opt_state = fx_train_run_setup
    n_infer_steps = 5
    block_len, n_blocks = 3, 3
    total_len = block_len * n_blocks

    states_prev0 = _fresh_states_prev(610)
    all_ys = jr.normal(jr.key(611), (total_len, FX_OBS_SIZE))

    # one big run
    train_run_full = make_train_run(param_optim, activity_optim, n_infer_steps, run_length=total_len, control_input=fx_control_input)
    full_model, full_opt_state, full_states, full_y_before, full_y_after, full_energies_before, full_energies_after, _ = train_run_full(
        fx_model, param_opt_state, states_prev0, all_ys
    )

    # goldilocks blocks
    train_run_block = make_train_run(param_optim, activity_optim, n_infer_steps, run_length=block_len, control_input=fx_control_input)
    model, opt_state, states_prev = fx_model, param_opt_state, states_prev0
    block_energies_before, block_energies_after = [], []
    for b in range(n_blocks):
        ys_block = all_ys[b * block_len:(b + 1) * block_len]
        model, opt_state, states_prev, y_before, y_after, energies_before, energies_after, _ = train_run_block(model, opt_state, states_prev, ys_block)
        block_energies_before.append(energies_before)
        block_energies_after.append(energies_after)
    block_energies_before = jnp.concatenate(block_energies_before)
    block_energies_after = jnp.concatenate(block_energies_after)

    assert_allclose(block_energies_before, full_energies_before, "energies_before: goldilocks blocks vs one big run", atol=1e-4, rtol=1e-4)
    assert_allclose(block_energies_after, full_energies_after, "energies_after: goldilocks blocks vs one big run", atol=1e-4, rtol=1e-4)
    for a, b in zip(states_prev, full_states):
        assert_allclose(a, b, "final states: goldilocks blocks vs one big run", atol=1e-4, rtol=1e-4)
    block_weights = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    full_weights = jax.tree_util.tree_leaves(eqx.filter(full_model, eqx.is_array))
    for a, b in zip(block_weights, full_weights):
        assert_allclose(a, b, "final weights: goldilocks blocks vs one big run", atol=1e-4, rtol=1e-4)


# ---- M6. n_infer_steps and run_length are genuinely wired, not hardcoded --

def test_make_train_run_run_length_matches_scan_length(fx_model, fx_control_input, fx_train_run_setup):
    param_optim, activity_optim, param_opt_state = fx_train_run_setup
    states_prev = _fresh_states_prev(612)

    for run_length in [1, 3, 7]:
        train_run = make_train_run(param_optim, activity_optim, n_infer_steps=5, run_length=run_length, control_input=fx_control_input)
        ys = jr.normal(jr.fold_in(jr.key(613), run_length), (run_length, FX_OBS_SIZE))
        _, _, _, y_before, y_after, energies_before, energies_after, _ = train_run(fx_model, param_opt_state, states_prev, ys)
        assert energies_before.shape == (run_length,)
        assert energies_after.shape == (run_length,)
        assert y_before.shape == (run_length, FX_OBS_SIZE)


def test_make_train_run_different_n_infer_steps_give_different_settled_states(fx_model, fx_control_input, fx_train_run_setup):
    param_optim, activity_optim, param_opt_state = fx_train_run_setup
    states_prev = _fresh_states_prev(614)
    ys = jr.normal(jr.key(615), (3, FX_OBS_SIZE))

    train_run_short = make_train_run(param_optim, activity_optim, n_infer_steps=2, run_length=3, control_input=fx_control_input)
    train_run_long = make_train_run(param_optim, activity_optim, n_infer_steps=30, run_length=3, control_input=fx_control_input)

    _, _, states_short, *_ = train_run_short(fx_model, param_opt_state, states_prev, ys)
    _, _, states_long, *_ = train_run_long(fx_model, param_opt_state, states_prev, ys)

    assert any(not bool(jnp.allclose(s, l, atol=1e-4, rtol=1e-4)) for s, l in zip(states_short, states_long))


# ---- M7. control_input is genuinely used ------------------------------------

def test_make_train_run_control_input_actually_affects_predictions(fx_model, fx_train_run_setup):
    param_optim, activity_optim, param_opt_state = fx_train_run_setup
    states_prev = _fresh_states_prev(616)
    ys = jr.normal(jr.key(617), (3, FX_OBS_SIZE))
    control_input = jr.normal(jr.key(618), (FX_INPUT_SIZE,))

    train_run_with = make_train_run(param_optim, activity_optim, n_infer_steps=5, run_length=3, control_input=control_input)
    train_run_without = make_train_run(param_optim, activity_optim, n_infer_steps=5, run_length=3, control_input=None)

    _, _, _, y_before_with, *_ = train_run_with(fx_model, param_opt_state, states_prev, ys)
    _, _, _, y_before_without, *_ = train_run_without(fx_model, param_opt_state, states_prev, ys)

    assert not bool(jnp.allclose(y_before_with, y_before_without, atol=1e-4, rtol=1e-4))


# ---- M8/M9. Edge cases: zero hidden layers, ce loss ------------------------

def test_make_train_run_zero_hidden_layers():
    model = TpchModel(control_layer_size=4, hidden_sizes=[], obs_size=5, key=jr.key(619), input_size=2)
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    param_opt_state = param_optim.init(eqx.filter(model, eqx.is_array))
    control_input = jr.normal(jr.key(620), (2,))
    run_length, n_infer_steps = 4, 6
    train_run = make_train_run(param_optim, activity_optim, n_infer_steps, run_length=run_length, control_input=control_input)

    states_prev = [jr.normal(jr.key(621), (4,))]
    ys = jr.normal(jr.key(622), (run_length, 5))
    new_model, new_opt_state, states_curr, y_before, y_after, energies_before, energies_after, energy_traces = train_run(
        model, param_opt_state, states_prev, ys, return_layerwise=True
    )
    assert len(states_curr) == 1
    assert energy_traces.shape == (run_length, n_infer_steps, 2)  # control + observation only
    assert jnp.all(jnp.isfinite(energies_before))
    assert jnp.all(jnp.isfinite(energies_after))


def test_make_train_run_with_ce_loss():
    model = TpchModel(
        control_layer_size=FX_CONTROL_SIZE, hidden_sizes=FX_HIDDEN_SIZES, obs_size=FX_OBS_SIZE,
        key=jr.key(623), input_size=FX_INPUT_SIZE, loss="ce",
    )
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    param_opt_state = param_optim.init(eqx.filter(model, eqx.is_array))
    control_input = jr.normal(jr.key(624), (FX_INPUT_SIZE,))
    run_length = 4
    train_run = make_train_run(param_optim, activity_optim, n_infer_steps=6, run_length=run_length, control_input=control_input)

    states_prev = _fresh_states_prev(625)
    ys = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), FX_OBS_SIZE)

    new_model, new_opt_state, states_curr, y_before, y_after, energies_before, energies_after, _ = train_run(
        model, param_opt_state, states_prev, ys
    )
    assert jnp.all(jnp.isfinite(energies_before))
    assert jnp.all(jnp.isfinite(energies_after))
    weight_before = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))[0]
    weight_after = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))[0]
    assert not bool(jnp.array_equal(weight_before, weight_after))
