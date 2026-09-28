"""tests/eligibility_test.py

Pytest coverage for the tPC-E eligibility-trace machinery: the pure
functions in `pc_nox/models/tpch/eligibility.py`, and their wiring into
`TpchModel` (`zero_eligibility_state`, `update_eligibility_state`,
`param_grad_traced`, and the `trace_mode` switch between "readout" and
"accumulate"), plus the eligibility-traced training runners
(`make_train_step_traced`, `make_train_run_traced`) in
`runners_temporal.py`.

None of this had any test coverage before this file: `tpch_test.py` only
imports `TpchHeterogeneousObservationLayer` for typing purposes and never
touches `control_alpha`/`hidden_alphas`/`trace_mode`/`EligibilityState`/
`param_grad_traced`/`zero_eligibility_state`/`update_eligibility_state`/
`make_train_step_traced`/`make_train_run_traced` anywhere.

Style follows `tpch_test.py`: plain pytest functions + fixtures, no
classes, `assert_allclose` for numeric comparisons, deliberately
different layer widths so shape bugs can't hide behind a broadcast.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from pc_nox.models.tpch import eligibility as elig
from pc_nox.models.tpch.model import TpchModel, EligibilityState
from pc_nox.engine.runners_temporal import make_train_step, make_train_run, make_train_step_traced, make_train_run_traced


def assert_allclose(actual, expected, name, atol=1e-4, rtol=1e-4):
    actual, expected = jnp.asarray(actual), jnp.asarray(expected)
    assert actual.shape == expected.shape, (
        f"{name}: shape mismatch -- got {actual.shape}, expected {expected.shape}"
    )
    max_abs_diff = float(jnp.max(jnp.abs(actual - expected))) if actual.size else 0.0
    assert jnp.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name}: values differ, max abs diff = {max_abs_diff}"
    )


# =============================================================================
# Fixtures -- deliberately different widths (control=3, hidden=4, obs=5,
# input=2) from tpch_test.py's fixtures, so this file's failures can never
# be confused with a leaked shape from that one.
# =============================================================================

EL_CONTROL_SIZE = 3
EL_HIDDEN_SIZES = [4]
EL_OBS_SIZE = 5
EL_INPUT_SIZE = 2


@pytest.fixture
def fx_leaky_model():
    """Both the control layer and the (single) hidden layer are leaky."""
    return TpchModel(
        control_layer_size=EL_CONTROL_SIZE,
        hidden_sizes=EL_HIDDEN_SIZES,
        obs_size=EL_OBS_SIZE,
        key=jr.key(100),
        input_size=EL_INPUT_SIZE,
        control_alpha=0.4,
        hidden_alphas=(0.35,),
    )


@pytest.fixture
def fx_leaky_model_accumulate():
    """Same weights as fx_leaky_model (same PRNGKey), but trace_mode='accumulate'."""
    return TpchModel(
        control_layer_size=EL_CONTROL_SIZE,
        hidden_sizes=EL_HIDDEN_SIZES,
        obs_size=EL_OBS_SIZE,
        key=jr.key(100),
        input_size=EL_INPUT_SIZE,
        control_alpha=0.4,
        hidden_alphas=(0.35,),
        trace_mode="accumulate",
    )


@pytest.fixture
def fx_nonleaky_model():
    """Same weights (same PRNGKey), no leaky layers at all -- alpha=None everywhere."""
    return TpchModel(
        control_layer_size=EL_CONTROL_SIZE,
        hidden_sizes=EL_HIDDEN_SIZES,
        obs_size=EL_OBS_SIZE,
        key=jr.key(100),
        input_size=EL_INPUT_SIZE,
    )


@pytest.fixture
def fx_states_prev():
    sizes = [EL_CONTROL_SIZE] + list(EL_HIDDEN_SIZES)
    keys = jr.split(jr.key(101), len(sizes))
    return [jr.normal(k, (n,)) for k, n in zip(keys, sizes)]


@pytest.fixture
def fx_control_input():
    return jr.normal(jr.key(102), (EL_INPUT_SIZE,))


@pytest.fixture
def fx_observation():
    return jr.normal(jr.key(103), (EL_OBS_SIZE,))


def _leaves(pytree):
    return jax.tree_util.tree_leaves(eqx.filter(pytree, eqx.is_array))


# =============================================================================
# Part 1: pure functions in eligibility.py -- no TpchModel involved at all
# =============================================================================

# ---- leaky_predict -----------------------------------------------------------

def test_leaky_predict_matches_formula():
    state_prev = jnp.array([1.0, -2.0, 0.5])
    instantaneous = jnp.array([0.3, 0.3, 0.3])
    alpha = 0.25
    out = elig.leaky_predict(state_prev, instantaneous, alpha)
    manual = (1 - alpha) * state_prev + alpha * instantaneous
    assert_allclose(out, manual, "leaky_predict")


def test_leaky_predict_alpha_1_returns_instantaneous_exactly():
    state_prev = jnp.array([1.0, -2.0, 0.5])
    instantaneous = jnp.array([9.0, 9.0, 9.0])
    out = elig.leaky_predict(state_prev, instantaneous, 1.0)
    assert_allclose(out, instantaneous, "leaky_predict at alpha=1")


def test_leaky_predict_alpha_0_returns_state_prev_exactly():
    state_prev = jnp.array([1.0, -2.0, 0.5])
    instantaneous = jnp.array([9.0, 9.0, 9.0])
    out = elig.leaky_predict(state_prev, instantaneous, 0.0)
    assert_allclose(out, state_prev, "leaky_predict at alpha=0")


# ---- trace_update ("readout" mode's own recurrence) --------------------------

def test_trace_update_matches_manual_recurrence_over_many_steps():
    alpha = 0.3
    inputs = [jnp.array([float(i), -float(i), 0.5 * i]) for i in range(1, 8)]
    trace = elig.zero_trace_like(inputs[0])
    manual = jnp.zeros_like(inputs[0])
    for x in inputs:
        trace = elig.trace_update(trace, x, alpha)
        manual = (1 - alpha) * manual + alpha * x
        assert_allclose(trace, manual, "trace_update step-by-step")


def test_trace_update_alpha_1_always_equals_current_input():
    """At alpha=1 the trace has no memory at all -- verified regardless of
    what came before, per tPC-HE.md section 9."""
    trace = jnp.array([100.0, -100.0])
    x = jnp.array([1.0, 2.0])
    out = elig.trace_update(trace, x, 1.0)
    assert_allclose(out, x, "trace_update at alpha=1")


def test_trace_update_alpha_0_never_changes():
    trace = jnp.array([1.0, 2.0, 3.0])
    x = jnp.array([99.0, -99.0, 0.0])
    out = elig.trace_update(trace, x, 0.0)
    assert_allclose(out, trace, "trace_update at alpha=0")


def test_trace_survives_temporal_gap_after_input_vanishes():
    """Canonical eligibility-trace property: even once the driving input
    has gone back to zero for many steps, the trace (and hence the traced
    gradient) still carries a nonzero memory of an earlier spike, unlike
    the raw instantaneous outer product which is exactly zero once the
    spike is gone."""
    alpha = 0.3
    n_steps = 15
    spike = jnp.array([5.0, -5.0, 0.0])
    seq = [spike] + [jnp.zeros(3)] * (n_steps - 1)

    trace = elig.zero_trace_like(spike)
    traces_over_time = []
    for x in seq:
        trace = elig.trace_update(trace, x, alpha)
        traces_over_time.append(trace)

    error = jnp.ones(3)
    instantaneous_grad_t10 = jnp.outer(error, seq[10])
    traced_grad_t10 = elig.traced_weight_grad(error, traces_over_time[10])

    assert_allclose(instantaneous_grad_t10, jnp.zeros((3, 3)), "instantaneous grad at t=10 (input long gone)")
    assert bool(jnp.any(jnp.abs(traced_grad_t10) > 1e-6)), "traced grad at t=10 should still carry the spike"

    norms = [float(jnp.linalg.norm(t)) for t in traces_over_time]
    assert all(norms[i] >= norms[i + 1] - 1e-9 for i in range(1, len(norms) - 1)), (
        "trace should decay monotonically once the driving input is gone"
    )


# ---- traced_weight_grad (readout mode's read-out step) ------------------------

def test_traced_weight_grad_is_plain_outer_product():
    error = jnp.array([1.0, 2.0])
    trace = jnp.array([0.5, -0.5, 2.0])
    out = elig.traced_weight_grad(error, trace)
    assert_allclose(out, jnp.outer(error, trace), "traced_weight_grad")
    assert out.shape == (2, 3)


def test_traced_weight_grad_additive_ascent_sign_convention():
    """The function itself returns the paper's ADDITIVE-ascent convention
    (no built-in negation) -- callers wiring this into a jax.grad-style
    descent gradient (as param_grad_traced does) must negate it
    themselves. This just locks down that traced_weight_grad itself does
    NOT do that negation."""
    error = jnp.array([1.0, -1.0])
    trace = jnp.array([2.0, 3.0])
    out = elig.traced_weight_grad(error, trace)
    assert float(out[0, 0]) == pytest.approx(2.0)  # error[0]*trace[0], not negated


# ---- matrix_trace_update / traced_weight_grad_from_matrix ("accumulate") -----

def test_matrix_trace_update_matches_manual_recurrence():
    alpha = 0.4
    own_size, input_size = 2, 3
    derivs = [jr.uniform(jr.key(200 + i), (own_size,)) for i in range(5)]
    inputs = [jr.normal(jr.key(300 + i), (input_size,)) for i in range(5)]

    trace = elig.zero_matrix_trace_like(own_size, input_size)
    manual = jnp.zeros((own_size, input_size))
    for d, x in zip(derivs, inputs):
        trace = elig.matrix_trace_update(trace, d, x, alpha)
        manual = (1 - alpha) * manual + alpha * jnp.outer(d, x)
        assert_allclose(trace, manual, "matrix_trace_update step-by-step")
    assert trace.shape == (own_size, input_size)


def test_matrix_trace_update_alpha_1_equals_current_outer_product():
    d = jnp.array([1.0, 2.0])
    x = jnp.array([3.0, 4.0, 5.0])
    prev = jnp.ones((2, 3)) * 100.0
    out = elig.matrix_trace_update(prev, d, x, 1.0)
    assert_allclose(out, jnp.outer(d, x), "matrix_trace_update at alpha=1")


def test_traced_weight_grad_from_matrix_is_row_broadcast_scale():
    error = jnp.array([2.0, -3.0])
    trace_matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = elig.traced_weight_grad_from_matrix(error, trace_matrix)
    manual = jnp.stack([error[i] * trace_matrix[i] for i in range(2)])
    assert_allclose(out, manual, "traced_weight_grad_from_matrix")


# ---- zero_trace_like / zero_matrix_trace_like ---------------------------------

def test_zero_trace_like_shape_and_all_zero():
    example = jnp.zeros((7,))
    out = elig.zero_trace_like(example)
    assert out.shape == (7,)
    assert_allclose(out, jnp.zeros(7), "zero_trace_like")


def test_zero_matrix_trace_like_shape_and_all_zero():
    out = elig.zero_matrix_trace_like(4, 6)
    assert out.shape == (4, 6)
    assert_allclose(out, jnp.zeros((4, 6)), "zero_matrix_trace_like")


# ---- elementwise_deriv ---------------------------------------------------------

def test_elementwise_deriv_tanh_matches_analytic_derivative():
    pre = jnp.array([-1.0, 0.0, 0.5, 2.0])
    out = elig.elementwise_deriv(jnp.tanh, pre)
    manual = 1.0 - jnp.tanh(pre) ** 2
    assert_allclose(out, manual, "elementwise_deriv(tanh)")


def test_elementwise_deriv_identity_is_all_ones():
    pre = jnp.array([-3.0, 0.0, 5.0])
    out = elig.elementwise_deriv(lambda x: x, pre)
    assert_allclose(out, jnp.ones_like(pre), "elementwise_deriv(identity)")


def test_elementwise_deriv_matches_per_element_jax_grad():
    """Cross-check against a completely independent per-element jax.grad
    loop, rather than a hand-typed closed form, for extra confidence."""
    act_fn = jnp.tanh
    pre = jr.normal(jr.key(5), (6,))
    out = elig.elementwise_deriv(act_fn, pre)
    manual = jnp.array([jax.grad(act_fn)(p) for p in pre])
    assert_allclose(out, manual, "elementwise_deriv vs per-element jax.grad")


# =============================================================================
# Part 2: TpchModel wiring -- zero_eligibility_state
# =============================================================================

def test_zero_eligibility_state_non_leaky_model_is_all_none(fx_nonleaky_model, fx_control_input):
    e = fx_nonleaky_model.zero_eligibility_state(control_input_example=fx_control_input)
    assert e.e_A is None
    assert e.e_B is None
    assert len(e.e_hidden) == 1
    assert e.e_hidden[0] == (None, None, None)


def test_zero_eligibility_state_readout_shapes(fx_leaky_model, fx_control_input):
    e = fx_leaky_model.zero_eligibility_state(control_input_example=fx_control_input)
    assert e.e_A.shape == (EL_CONTROL_SIZE,)
    assert e.e_B.shape == (EL_INPUT_SIZE,)
    e_P, e_Q, e_R = e.e_hidden[0]
    assert e_P.shape == (EL_HIDDEN_SIZES[0],)
    assert e_Q.shape == (EL_CONTROL_SIZE,)  # parent (control layer) size
    assert e_R.shape == (EL_CONTROL_SIZE,)
    for t in (e.e_A, e.e_B, e_P, e_Q, e_R):
        assert_allclose(t, jnp.zeros_like(t), "zero_eligibility_state readout trace should start at zero")


def test_zero_eligibility_state_accumulate_shapes_match_weights(fx_leaky_model_accumulate, fx_control_input):
    m = fx_leaky_model_accumulate
    e = m.zero_eligibility_state(control_input_example=fx_control_input)
    assert e.e_A.shape == m.control_layer.W_rec.weight.shape
    assert e.e_B.shape == m.control_layer.W_in.weight.shape
    e_P, e_Q, e_R = e.e_hidden[0]
    layer = m.hidden_layers[0]
    assert e_P.shape == layer.W_rec.weight.shape
    assert e_Q.shape == layer.W_parent_prev.weight.shape
    assert e_R.shape == layer.W_parent_curr.weight.shape
    for t in (e.e_A, e.e_B, e_P, e_Q, e_R):
        assert_allclose(t, jnp.zeros_like(t), "zero_eligibility_state accumulate trace should start at zero")


def test_zero_eligibility_state_raises_when_control_leaky_with_input_and_no_example():
    m = TpchModel(
        control_layer_size=3, hidden_sizes=(4,), obs_size=5, key=jr.key(0),
        input_size=2, control_alpha=0.5,
    )
    with pytest.raises(ValueError):
        m.zero_eligibility_state()  # control layer is leaky AND has_input, no example given


def test_zero_eligibility_state_no_input_pathway_leaves_e_B_none():
    """Leaky control layer but input_size=0 -- e_B should stay None (no
    W_in to trace at all), and no exception should be raised even without
    control_input_example."""
    m = TpchModel(control_layer_size=3, hidden_sizes=(), obs_size=3, key=jr.key(1), control_alpha=0.5)
    e = m.zero_eligibility_state()
    assert e.e_A is not None
    assert e.e_B is None


def test_zero_eligibility_state_mixed_leaky_and_non_leaky_hidden_layers():
    """hidden_alphas with a mix of a leak rate and None: only the leaky
    layer should get real traces, the other stays (None, None, None)."""
    m = TpchModel(
        control_layer_size=3, hidden_sizes=(4, 5), obs_size=6, key=jr.key(2),
        hidden_alphas=(0.3, None),
    )
    e = m.zero_eligibility_state()
    assert e.e_hidden[0] != (None, None, None)
    assert e.e_hidden[0][0].shape == (4,)
    assert e.e_hidden[1] == (None, None, None)


# =============================================================================
# Part 3: TpchModel wiring -- update_eligibility_state
# =============================================================================

def test_update_eligibility_state_control_layer_matches_manual_recurrence(fx_leaky_model, fx_control_input):
    """Drive the control layer's trace through 3 steps by hand (via
    eligibility.trace_update directly) and via update_eligibility_state,
    and confirm they agree exactly -- this is the same formula
    TpchModel's wiring is supposed to be calling."""
    m = fx_leaky_model
    alpha = m.control_layer.alpha
    s_seq = [jr.normal(jr.key(400 + i), (EL_CONTROL_SIZE,)) for i in range(4)]
    c_seq = [jr.normal(jr.key(500 + i), (EL_INPUT_SIZE,)) for i in range(3)]

    e = m.zero_eligibility_state(control_input_example=fx_control_input)
    e_A_manual = jnp.zeros(EL_CONTROL_SIZE)
    e_B_manual = jnp.zeros(EL_INPUT_SIZE)
    hidden_prev = jr.normal(jr.key(9), (EL_HIDDEN_SIZES[0],))
    for t in range(3):
        states_prev = [s_seq[t], hidden_prev]
        states_curr = [s_seq[t + 1], hidden_prev]  # hidden held fixed; only checking control trace here
        e = m.update_eligibility_state(e, states_prev, states_curr, c_seq[t])
        e_A_manual = (1 - alpha) * e_A_manual + alpha * s_seq[t]
        e_B_manual = (1 - alpha) * e_B_manual + alpha * c_seq[t]
        assert_allclose(e.e_A, e_A_manual, f"e_A after step {t}")
        assert_allclose(e.e_B, e_B_manual, f"e_B after step {t}")


def test_update_eligibility_state_uses_states_curr_for_parent_curr_trace(fx_leaky_model, fx_control_input, fx_states_prev):
    """e_R (the hidden layer's W_parent_curr trace) must be built from
    THIS frame's settled parent state (states_curr[0], i.e. the control
    layer's *current* state), NOT states_prev[0] -- see
    update_eligibility_state's docstring on why ordering matters. We
    prove this by making states_curr[0] wildly different from
    states_prev[0] and checking e_R tracks the CURR value."""
    m = fx_leaky_model
    e0 = m.zero_eligibility_state(control_input_example=fx_control_input)
    states_prev = fx_states_prev
    parent_curr_distinctive = jnp.array([100.0, -100.0, 50.0])
    states_curr = [parent_curr_distinctive, states_prev[1]]

    e1 = m.update_eligibility_state(e0, states_prev, states_curr, fx_control_input)
    _, _, e_R = e1.e_hidden[0]
    alpha = m.hidden_layers[0].alpha
    manual_e_R = (1 - alpha) * jnp.zeros(EL_CONTROL_SIZE) + alpha * parent_curr_distinctive
    assert_allclose(e_R, manual_e_R, "e_R should track states_curr's parent value, not states_prev's")


def test_update_eligibility_state_accumulate_mode_matches_manual_recurrence(fx_leaky_model_accumulate, fx_control_input):
    m = fx_leaky_model_accumulate
    alpha = m.control_layer.alpha
    s_seq = [jr.normal(jr.key(600 + i), (EL_CONTROL_SIZE,)) for i in range(3)]
    c_seq = [jr.normal(jr.key(700 + i), (EL_INPUT_SIZE,)) for i in range(2)]
    hidden_fixed = jr.normal(jr.key(19), (EL_HIDDEN_SIZES[0],))

    e = m.zero_eligibility_state(control_input_example=fx_control_input)
    manual = jnp.zeros((EL_CONTROL_SIZE, EL_CONTROL_SIZE))
    for t in range(2):
        states_prev = [s_seq[t], hidden_fixed]
        states_curr = [s_seq[t + 1], hidden_fixed]
        pre_s = m.control_layer.pre_activation(s_seq[t], c_seq[t])
        deriv_s = elig.elementwise_deriv(m.control_layer.act_fn, pre_s)
        manual = (1 - alpha) * manual + alpha * jnp.outer(deriv_s, s_seq[t])
        e = m.update_eligibility_state(e, states_prev, states_curr, c_seq[t])
        assert_allclose(e.e_A, manual, f"accumulate-mode e_A after step {t}")


def test_update_eligibility_state_preserves_none_for_nonleaky_control(fx_control_input):
    """A model whose hidden layer is leaky but whose control layer is NOT
    should keep e_A/e_B as None after update, while the hidden layer's
    traces become real arrays."""
    m = TpchModel(control_layer_size=3, hidden_sizes=(4,), obs_size=5, key=jr.key(3), input_size=2, hidden_alphas=(0.3,))
    e0 = m.zero_eligibility_state(control_input_example=fx_control_input)
    sp = [jnp.zeros(3), jnp.zeros(4)]
    sc = [jnp.ones(3), jnp.ones(4)]
    e1 = m.update_eligibility_state(e0, sp, sc, fx_control_input)
    assert e1.e_A is None
    assert e1.e_B is None
    assert e1.e_hidden[0][0] is not None


# =============================================================================
# Part 4: param_grad_traced -- correctness against param_grad and against
# independent hand derivations
# =============================================================================

def test_param_grad_traced_equals_param_grad_when_nothing_is_leaky(fx_nonleaky_model, fx_states_prev, fx_control_input, fx_observation):
    m = fx_nonleaky_model
    sc = m.settle(fx_states_prev, fx_observation, fx_control_input, n_steps=8)
    e0 = m.zero_eligibility_state(control_input_example=fx_control_input)

    g_plain = m.param_grad(fx_states_prev, sc, fx_observation, fx_control_input)
    g_traced = m.param_grad_traced(e0, fx_states_prev, sc, fx_observation, fx_control_input)

    for a, b in zip(_leaves(g_plain), _leaves(g_traced)):
        assert_allclose(a, b, "param_grad_traced vs param_grad, no leaky layers", atol=1e-6, rtol=1e-6)


def test_param_grad_traced_structure_matches_model(fx_leaky_model, fx_states_prev, fx_control_input, fx_observation):
    m = fx_leaky_model
    sc = m.settle(fx_states_prev, fx_observation, fx_control_input, n_steps=5)
    e0 = m.zero_eligibility_state(control_input_example=fx_control_input)
    e1 = m.update_eligibility_state(e0, fx_states_prev, sc, fx_control_input)
    g = m.param_grad_traced(e1, fx_states_prev, sc, fx_observation, fx_control_input)
    assert jax.tree_util.tree_structure(eqx.filter(g, eqx.is_array)) == jax.tree_util.tree_structure(eqx.filter(m, eqx.is_array))
    for leaf, model_leaf in zip(_leaves(g), _leaves(m)):
        assert leaf.shape == model_leaf.shape


def test_param_grad_traced_control_layer_matches_hand_derivation_readout(fx_leaky_model, fx_control_input):
    """Independent hand derivation of the readout-mode traced gradient for
    the control layer's W_rec, including f' and the additive-vs-descent
    sign flip -- mirrors the property new_features_test.py checked for
    the pre-refactor code, ported to this package's actual API."""
    m = fx_leaky_model
    s_seq = [jnp.array([1.0, -1.0, 0.5]), jnp.array([0.2, 0.1, -0.3]), jnp.array([0.0, 0.0, 0.0])]
    c_seq = [jnp.array([0.5, -0.2]), jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])]
    hidden_seq = [jr.normal(jr.key(800 + i), (EL_HIDDEN_SIZES[0],)) for i in range(3)]

    e = m.zero_eligibility_state(control_input_example=fx_control_input)
    for t in range(2):
        states_prev = [s_seq[t], hidden_seq[t]]
        states_curr = [s_seq[t + 1], hidden_seq[t + 1]]
        e = m.update_eligibility_state(e, states_prev, states_curr, c_seq[t])

    y = jr.normal(jr.key(11), (EL_OBS_SIZE,))
    states_prev_final = [s_seq[2], hidden_seq[2]]
    states_curr_final = m.settle(states_prev_final, y, c_seq[2], n_steps=1)

    g = m.param_grad_traced(e, states_prev_final, states_curr_final, y, c_seq[2])

    pre_s = m.control_layer.pre_activation(s_seq[2], c_seq[2])
    pred_s = m.control_layer.predict(s_seq[2], c_seq[2])
    eps_s = states_curr_final[0] - pred_s
    delta_s = (1 - jnp.tanh(pre_s) ** 2) * eps_s
    manual_grad_A = -jnp.outer(delta_s, e.e_A)  # additive-ascent -> negate to descend; NO alpha multiplier

    assert_allclose(g.control_layer.W_rec.weight, manual_grad_A, "traced control W_rec grad vs hand derivation", atol=1e-5, rtol=1e-5)


def test_param_grad_traced_hidden_layer_W_parent_curr_matches_hand_derivation(fx_control_input):
    """W_parent_curr (R) is the layer that's genuinely non-trivial to get
    right: it multiplies a SAME-timestep value but is still traced via
    the layer's own leaky memory. Verify against an independent formula."""
    m = TpchModel(control_layer_size=3, hidden_sizes=(4,), obs_size=5, key=jr.key(4), hidden_alphas=(0.35,))
    sp0, zp0 = jnp.array([0.4, -0.2, 0.1]), jnp.array([0.1, 0.2, -0.1, 0.3])
    y = jr.normal(jr.key(12), (5,))

    sc = m.settle([sp0, zp0], y, None, n_steps=1)
    e0 = m.zero_eligibility_state()
    e1 = m.update_eligibility_state(e0, [sp0, zp0], sc, None)
    g = m.param_grad_traced(e1, [sp0, zp0], sc, y, None)

    layer = m.hidden_layers[0]
    s_curr_real = sc[0]
    pre_l = layer.pre_activation(zp0, sp0, s_curr_real)
    pred_l = layer.predict(zp0, sp0, s_curr_real)
    eps_l = sc[1] - pred_l
    delta_l = (1 - jnp.tanh(pre_l) ** 2) * eps_l
    _, _, e_R = e1.e_hidden[0]
    manual_R = -jnp.outer(delta_l, e_R)

    assert_allclose(g.hidden_layers[0].W_parent_curr.weight, manual_R, "traced hidden W_parent_curr (R) grad vs hand derivation", atol=1e-5, rtol=1e-5)


def test_param_grad_traced_accumulate_mode_matches_hand_derivation(fx_leaky_model_accumulate, fx_control_input):
    m = fx_leaky_model_accumulate
    s_seq = [jnp.array([1.0, -1.0, 0.5]), jnp.array([0.2, 0.1, -0.3]), jnp.array([0.0, 0.0, 0.0])]
    c_seq = [jnp.array([0.5, -0.2]), jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])]
    hidden_seq = [jr.normal(jr.key(900 + i), (EL_HIDDEN_SIZES[0],)) for i in range(3)]

    e = m.zero_eligibility_state(control_input_example=fx_control_input)
    for t in range(2):
        states_prev = [s_seq[t], hidden_seq[t]]
        states_curr = [s_seq[t + 1], hidden_seq[t + 1]]
        e = m.update_eligibility_state(e, states_prev, states_curr, c_seq[t])

    y = jr.normal(jr.key(13), (EL_OBS_SIZE,))
    states_prev_final = [s_seq[2], hidden_seq[2]]
    states_curr_final = m.settle(states_prev_final, y, c_seq[2], n_steps=1)
    g = m.param_grad_traced(e, states_prev_final, states_curr_final, y, c_seq[2])

    # accumulate mode: no separate f' at read-out (already baked into e_A)
    pred_s = m.control_layer.predict(s_seq[2], c_seq[2])
    eps_s = states_curr_final[0] - pred_s
    manual_grad_A = -(eps_s[:, None] * e.e_A)

    assert_allclose(g.control_layer.W_rec.weight, manual_grad_A, "accumulate-mode traced control W_rec grad vs hand derivation", atol=1e-5, rtol=1e-5)


def test_param_grad_traced_leaves_nonleaky_observation_weight_untouched(fx_leaky_model, fx_states_prev, fx_control_input, fx_observation):
    """The observation layer is never leaky (see class docstrings) -- its
    weight gradient under param_grad_traced must be numerically identical
    to param_grad's, even though other layers in the same model ARE
    leaky and traced."""
    m = fx_leaky_model
    sc = m.settle(fx_states_prev, fx_observation, fx_control_input, n_steps=5)
    e0 = m.zero_eligibility_state(control_input_example=fx_control_input)
    e1 = m.update_eligibility_state(e0, fx_states_prev, sc, fx_control_input)

    g_traced = m.param_grad_traced(e1, fx_states_prev, sc, fx_observation, fx_control_input)
    g_plain = m.param_grad(fx_states_prev, sc, fx_observation, fx_control_input)

    assert_allclose(
        g_traced.observation_layer.W_parent.weight,
        g_plain.observation_layer.W_parent.weight,
        "observation layer weight grad: traced vs plain (never leaky, must match exactly)",
        atol=1e-6, rtol=1e-6,
    )


def test_param_grad_traced_finite_and_all_weights_present(fx_leaky_model, fx_states_prev, fx_control_input, fx_observation):
    m = fx_leaky_model
    sc = m.settle(fx_states_prev, fx_observation, fx_control_input, n_steps=5)
    e0 = m.zero_eligibility_state(control_input_example=fx_control_input)
    e1 = m.update_eligibility_state(e0, fx_states_prev, sc, fx_control_input)
    g = m.param_grad_traced(e1, fx_states_prev, sc, fx_observation, fx_control_input)
    for leaf in _leaves(g):
        assert jnp.all(jnp.isfinite(leaf))


def test_param_grad_traced_regularisation_matches_param_grad_reg_component():
    """weight_decay/orthogonal_penalty gradients don't depend on
    states_prev/states_curr/traces at all -- they should be IDENTICAL
    between param_grad and param_grad_traced regardless of tracing, since
    both add the same reg_grad on top. We isolate this by comparing a
    leaky model against a zero-regularisation twin: reg-driven part of
    the gradient must vanish identically in both param_grad and
    param_grad_traced when weight_decay=0, and be present and IDENTICAL
    in both when weight_decay>0 and the correlation part is neutralised
    by using the exact same (settled) states for both calls."""
    key = jr.key(55)
    m = TpchModel(
        control_layer_size=3, hidden_sizes=(4,), obs_size=5, key=key, input_size=2,
        control_alpha=0.4, weight_decay=0.1, weight_decay_scope="all",
    )
    sp = [jnp.array([0.2, -0.1, 0.3]), jnp.array([0.1, 0.0, -0.2, 0.4])]
    ci = jnp.array([0.3, -0.5])
    y = jnp.array([0.1, 0.2, -0.1, 0.0, 0.3])
    sc = m.settle(sp, y, ci, n_steps=3)
    e0 = m.zero_eligibility_state(control_input_example=ci)
    e1 = m.update_eligibility_state(e0, sp, sc, ci)

    g_plain = m.param_grad(sp, sc, y, ci)
    g_traced = m.param_grad_traced(e1, sp, sc, y, ci)

    # The observation layer's weight is never traced (no leak there ever)
    # so, since weight_decay_scope="all" adds the SAME reg term regardless
    # of tracing, and the correlation term for this untraced weight is
    # identical too, the two must match exactly here.
    assert_allclose(
        g_traced.observation_layer.W_parent.weight,
        g_plain.observation_layer.W_parent.weight,
        "reg + correlation on untraced observation weight: traced vs plain",
        atol=1e-6, rtol=1e-6,
    )


# =============================================================================
# Part 5: alpha=1 sanity checks -- the paper's own explicit property (S4
# appendix): at alpha=1 the whole leaky/traced machinery collapses to
# plain tPC-H exactly, in BOTH trace modes. This is the exact regression
# the docstring says an earlier, buggy version of param_grad_traced
# failed (an extra spurious alpha_l multiplier).
# =============================================================================

def test_predictions_match_exactly_at_alpha_1(fx_control_input):
    key = jr.key(20)
    m_alpha1 = TpchModel(control_layer_size=3, hidden_sizes=(4,), obs_size=5, input_size=2, key=key, control_alpha=1.0, hidden_alphas=(1.0,))
    m_plain = TpchModel(control_layer_size=3, hidden_sizes=(4,), obs_size=5, input_size=2, key=key)  # identical weights

    sp = [jr.normal(jr.key(21), (3,)), jr.normal(jr.key(22), (4,))]
    ci = jr.normal(jr.key(23), (2,))
    y = jr.normal(jr.key(24), (5,))

    sc1 = m_alpha1.settle(sp, y, ci, n_steps=1)
    sc2 = m_plain.settle(sp, y, ci, n_steps=1)
    for a, b in zip(sc1, sc2):
        assert_allclose(a, b, "settled states at alpha=1 vs plain (leaky blend degenerates to instantaneous)")


def test_trace_after_one_step_equals_raw_input_at_alpha_1():
    key = jr.key(20)
    m_alpha1 = TpchModel(control_layer_size=3, hidden_sizes=(4,), obs_size=5, input_size=2, key=key, control_alpha=1.0, hidden_alphas=(1.0,))
    sp = [jr.normal(jr.key(21), (3,)), jr.normal(jr.key(22), (4,))]
    ci = jr.normal(jr.key(23), (2,))
    y = jr.normal(jr.key(24), (5,))
    sc = m_alpha1.settle(sp, y, ci, n_steps=1)

    e0 = m_alpha1.zero_eligibility_state(control_input_example=ci)
    e1 = m_alpha1.update_eligibility_state(e0, sp, sc, ci)
    assert_allclose(e1.e_A, sp[0], "trace at alpha=1 should equal raw states_prev[0] exactly, no memory")


@pytest.mark.parametrize("trace_mode", ["readout", "accumulate"])
def test_param_grad_traced_matches_param_grad_exactly_at_alpha_1(trace_mode):
    key = jr.key(20)
    m_alpha1 = TpchModel(
        control_layer_size=3, hidden_sizes=(4,), obs_size=5, input_size=2, key=key,
        control_alpha=1.0, hidden_alphas=(1.0,), trace_mode=trace_mode,
    )
    m_plain = TpchModel(control_layer_size=3, hidden_sizes=(4,), obs_size=5, input_size=2, key=key)

    sp = [jr.normal(jr.key(21), (3,)), jr.normal(jr.key(22), (4,))]
    ci = jr.normal(jr.key(23), (2,))
    y = jr.normal(jr.key(24), (5,))

    sc1 = m_alpha1.settle(sp, y, ci, n_steps=1)
    sc2 = m_plain.settle(sp, y, ci, n_steps=1)

    e0 = m_alpha1.zero_eligibility_state(control_input_example=ci)
    e1 = m_alpha1.update_eligibility_state(e0, sp, sc1, ci)
    g_traced = m_alpha1.param_grad_traced(e1, sp, sc1, y, ci)
    g_plain = m_plain.param_grad(sp, sc2, y, ci)

    leaves_t, leaves_p = _leaves(g_traced), _leaves(g_plain)
    assert len(leaves_t) == len(leaves_p)
    for a, b in zip(leaves_t, leaves_p):
        assert_allclose(a, b, f"param_grad_traced (trace_mode={trace_mode}) vs param_grad at alpha=1", atol=1e-5, rtol=1e-5)


# =============================================================================
# Part 6: readout vs accumulate genuinely differ at alpha<1 over multiple
# steps (NOT equivalent, per tPC-HE.md section 8), but produce IDENTICAL
# predictions (trace_mode only affects the weight update, never inference)
# =============================================================================

def test_trace_mode_does_not_affect_predictions_or_settling():
    key = jr.key(21)
    m_readout = TpchModel(control_layer_size=3, hidden_sizes=(), obs_size=3, input_size=2, key=key, control_alpha=0.4, trace_mode="readout")
    m_accum = TpchModel(control_layer_size=3, hidden_sizes=(), obs_size=3, input_size=2, key=key, control_alpha=0.4, trace_mode="accumulate")

    sp = [jr.normal(jr.key(30), (3,))]
    ci = jr.normal(jr.key(31), (2,))
    y = jr.normal(jr.key(32), (3,))

    sc_r = m_readout.settle(sp, y, ci, n_steps=5)
    sc_a = m_accum.settle(sp, y, ci, n_steps=5)
    for a, b in zip(sc_r, sc_a):
        assert_allclose(a, b, "settled states should be identical regardless of trace_mode")


def test_readout_and_accumulate_genuinely_diverge_at_alpha_below_1_multistep():
    key = jr.key(21)
    m_readout = TpchModel(control_layer_size=3, hidden_sizes=(), obs_size=3, input_size=2, key=key, control_alpha=0.4, trace_mode="readout")
    m_accum = TpchModel(control_layer_size=3, hidden_sizes=(), obs_size=3, input_size=2, key=key, control_alpha=0.4, trace_mode="accumulate")

    s_seq = [jnp.array([1.0, -1.0, 0.5]), jnp.array([2.0, 0.3, -0.2]), jnp.array([0.1, 0.1, 0.1])]
    c_seq = [jnp.array([0.5, -0.2]), jnp.array([-0.3, 0.7]), jnp.array([1.0, 1.0])]

    def run_two_steps(model):
        e = model.zero_eligibility_state(control_input_example=c_seq[0])
        sprev = s_seq[0]
        for t in range(2):
            scurr = [s_seq[t + 1]]
            e = model.update_eligibility_state(e, [sprev], scurr, c_seq[t])
            sprev = s_seq[t + 1]
        return e

    e_r = run_two_steps(m_readout)
    e_a = run_two_steps(m_accum)

    y = jnp.array([0.3, -0.1, 0.2])
    sc_r = m_readout.settle([s_seq[2]], y, c_seq[2], n_steps=1)
    sc_a = m_accum.settle([s_seq[2]], y, c_seq[2], n_steps=1)
    assert_allclose(sc_r[0], sc_a[0], "predictions must still agree even though traces are structurally different")

    g_r = m_readout.param_grad_traced(e_r, [s_seq[2]], sc_r, y, c_seq[2])
    g_a = m_accum.param_grad_traced(e_a, [s_seq[2]], sc_a, y, c_seq[2])
    diff = float(jnp.max(jnp.abs(g_r.control_layer.W_rec.weight - g_a.control_layer.W_rec.weight)))
    assert diff > 1e-4, "readout and accumulate trace modes should genuinely differ at alpha<1 over multiple steps"


# =============================================================================
# Part 7: construction-time validation for the new config knobs
# =============================================================================

def test_invalid_trace_mode_raises_value_error():
    with pytest.raises(ValueError):
        TpchModel(control_layer_size=2, hidden_sizes=(), obs_size=2, key=jr.key(0), trace_mode="bogus")


def test_trace_mode_default_is_readout():
    m = TpchModel(control_layer_size=2, hidden_sizes=(), obs_size=2, key=jr.key(0))
    assert m.config.trace_mode == "readout"


def test_hidden_alphas_length_mismatch_raises():
    with pytest.raises(ValueError):
        TpchModel(control_layer_size=2, hidden_sizes=(3, 4), obs_size=2, key=jr.key(0), hidden_alphas=(0.5,))


def test_control_alpha_default_is_none_and_layer_unaffected():
    m = TpchModel(control_layer_size=3, hidden_sizes=(4,), obs_size=5, key=jr.key(0), input_size=2)
    assert m.config.control_alpha is None
    assert m.control_layer.alpha is None
    s_prev = jnp.array([0.1, -0.2, 0.5])
    ci = jnp.array([0.1, 0.2])
    # with alpha=None, predict() must equal the bare instantaneous prediction
    instantaneous = m.control_layer.act_fn(m.control_layer.pre_activation(s_prev, ci))
    assert_allclose(m.control_layer.predict(s_prev, ci), instantaneous, "alpha=None leaves prediction unaffected")


# =============================================================================
# Part 8: control_alpha/hidden_alphas/trace_mode config round-trip through
# from_config and full checkpoint save/load -- these are computational
# config (see ModelBase's config-vs-metadata note) and were completely
# untested before this file.
# =============================================================================

def test_eligibility_config_round_trips_through_from_config():
    from pc_nox.models.tpch.config import TpchConfig
    cfg = TpchConfig(
        control_layer_size=3, hidden_sizes=(4, 5), obs_size=6, input_size=2,
        control_alpha=0.4, hidden_alphas=(0.3, None), trace_mode="accumulate",
    )
    m = TpchModel.from_config(cfg, key=jr.key(0))
    assert m.config.control_alpha == 0.4
    assert m.config.hidden_alphas == (0.3, None)
    assert m.config.trace_mode == "accumulate"
    assert m.control_layer.alpha == 0.4
    assert m.hidden_layers[0].alpha == 0.3
    assert m.hidden_layers[1].alpha is None


def test_eligibility_config_round_trips_through_checkpoint(tmp_path):
    m = TpchModel(
        control_layer_size=3, hidden_sizes=(4,), obs_size=5, key=jr.key(0), input_size=2,
        control_alpha=0.4, hidden_alphas=(0.3,), trace_mode="accumulate",
    )
    out_dir = m.save_checkpoint(path=tmp_path / "elig_ckpt")
    loaded = TpchModel.load_checkpoint(out_dir)

    assert loaded.model.config.control_alpha == 0.4
    assert loaded.model.config.hidden_alphas == (0.3,)
    assert loaded.model.config.trace_mode == "accumulate"
    assert loaded.model.control_layer.alpha == 0.4
    assert loaded.model.hidden_layers[0].alpha == 0.3

    # weights round-trip exactly too
    for a, b in zip(_leaves(m), _leaves(loaded.model)):
        assert bool(jnp.array_equal(a, b))


# =============================================================================
# Part 9: eligibility-traced training runners -- make_train_step_traced,
# make_train_run_traced (runners_temporal.py). Zero coverage previously.
# =============================================================================

@pytest.fixture
def fx_traced_step_setup(fx_leaky_model):
    param_optim = optax.adam(learning_rate=1e-3)
    param_opt_state = param_optim.init(eqx.filter(fx_leaky_model, eqx.is_array))
    return param_optim, param_opt_state


def test_make_train_step_traced_output_contract(fx_leaky_model, fx_states_prev, fx_observation, fx_control_input, fx_traced_step_setup):
    param_optim, param_opt_state = fx_traced_step_setup
    activity_optim = optax.adam(learning_rate=1e-2)
    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=fx_control_input)
    train_step = make_train_step_traced(param_optim, activity_optim, n_infer_steps=6, control_input=fx_control_input)

    result = train_step(fx_leaky_model, param_opt_state, elig0, fx_states_prev, fx_observation)
    assert len(result) == 9
    model, opt_state, elig1, states_curr, y_before, y_after, e_before, e_after, trace = result
    assert trace is None
    assert isinstance(elig1, EligibilityState)
    assert jnp.isfinite(e_before) and jnp.isfinite(e_after)
    assert len(states_curr) == len(fx_states_prev)


def test_make_train_step_traced_weight_update_actually_changes_weights(fx_leaky_model, fx_states_prev, fx_observation, fx_control_input, fx_traced_step_setup):
    param_optim, param_opt_state = fx_traced_step_setup
    activity_optim = optax.adam(learning_rate=1e-2)
    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=fx_control_input)
    train_step = make_train_step_traced(param_optim, activity_optim, n_infer_steps=6, control_input=fx_control_input)

    new_model, *_ = train_step(fx_leaky_model, param_opt_state, elig0, fx_states_prev, fx_observation)
    before, after = _leaves(fx_leaky_model), _leaves(new_model)
    assert any(not bool(jnp.array_equal(b, a)) for b, a in zip(before, after))
    for leaf in after:
        assert jnp.all(jnp.isfinite(leaf))


def test_make_train_step_traced_matches_manual_recomposition(fx_leaky_model, fx_states_prev, fx_observation, fx_control_input, fx_traced_step_setup):
    """Recompose train_step's body from public TpchModel methods by hand
    and confirm the factory-built version agrees exactly."""
    param_optim, param_opt_state = fx_traced_step_setup
    activity_optim = optax.adam(learning_rate=1e-2)
    n_infer_steps = 6

    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=fx_control_input)
    train_step = make_train_step_traced(param_optim, activity_optim, n_infer_steps, control_input=fx_control_input)
    out = train_step(fx_leaky_model, param_opt_state, elig0, fx_states_prev, fx_observation)
    model_f, opt_state_f, elig_f, states_curr_f, y_before_f, y_after_f, e_before_f, e_after_f, _ = out

    m = fx_leaky_model
    states_curr_init = m.init_activities(fx_states_prev, fx_control_input, fx_observation)
    _, y_before_manual = m.predict(fx_states_prev, states_curr_init, fx_control_input, fx_observation)
    e_before_manual = m.energy_fn(fx_states_prev, states_curr_init, fx_observation, fx_control_input)

    states_curr_manual = m.settle_scan(activity_optim, fx_states_prev, fx_observation, fx_control_input, n_steps=n_infer_steps)
    _, y_after_manual = m.predict(fx_states_prev, states_curr_manual, fx_control_input, fx_observation)
    e_after_manual = m.energy_fn(fx_states_prev, states_curr_manual, fx_observation, fx_control_input)

    elig_manual = m.update_eligibility_state(elig0, fx_states_prev, states_curr_manual, fx_control_input)
    g = m.param_grad_traced(elig_manual, fx_states_prev, states_curr_manual, fx_observation, fx_control_input)
    updates, opt_state_manual = param_optim.update(g, param_opt_state, m)
    model_manual = eqx.apply_updates(m, updates)
    model_manual = model_manual.postprocess_params()

    assert_allclose(y_before_f, y_before_manual, "y_hat_before: factory vs manual", atol=1e-4, rtol=1e-4)
    assert_allclose(y_after_f, y_after_manual, "y_hat_after: factory vs manual", atol=1e-4, rtol=1e-4)
    assert_allclose(e_before_f, e_before_manual, "energy_before: factory vs manual", atol=1e-4, rtol=1e-4)
    assert_allclose(e_after_f, e_after_manual, "energy_after: factory vs manual", atol=1e-4, rtol=1e-4)
    # Looser tolerance than the y_hat/energy checks above: this comparison is
    # jit-compiled factory output vs. an eager re-run of the same formula, and
    # the eligibility-traced path chains enough extra float32 products (traces,
    # outer products, adam moments) that jit-vs-eager operation reordering
    # produces a small but expected numerical drift -- still tight enough to
    # catch a genuinely wrong formula (which would differ by orders of
    # magnitude more than this).
    for f, r in zip(_leaves(model_f), _leaves(model_manual)):
        assert_allclose(f, r, "final weights: factory vs manual", atol=1e-2, rtol=1e-2)
    for f, r in zip(states_curr_f, states_curr_manual):
        assert_allclose(f, r, "final states: factory vs manual", atol=1e-2, rtol=1e-2)


def test_make_train_step_traced_eligibility_state_actually_advances_across_calls(fx_leaky_model, fx_states_prev, fx_observation, fx_control_input, fx_traced_step_setup):
    """Threading eligibility_state through two successive calls must
    genuinely accumulate -- the trace after 2 steps should differ from
    the trace after 1 step (not silently reset each call)."""
    param_optim, param_opt_state = fx_traced_step_setup
    activity_optim = optax.adam(learning_rate=1e-2)
    train_step = make_train_step_traced(param_optim, activity_optim, n_infer_steps=5, control_input=fx_control_input)

    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=fx_control_input)
    m, ops = fx_leaky_model, param_opt_state
    sp = fx_states_prev

    m, ops, elig1, sp, *_ = train_step(m, ops, elig0, sp, fx_observation)
    m, ops, elig2, sp, *_ = train_step(m, ops, elig1, sp, fx_observation)

    assert not bool(jnp.allclose(elig1.e_A, elig2.e_A, atol=1e-6))


def test_make_train_step_traced_control_input_override_works(fx_leaky_model, fx_states_prev, fx_observation, fx_traced_step_setup):
    param_optim, param_opt_state = fx_traced_step_setup
    activity_optim = optax.adam(learning_rate=1e-2)
    train_step = make_train_step_traced(param_optim, activity_optim, n_infer_steps=5, control_input=None)

    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=jnp.zeros(EL_INPUT_SIZE))
    override_ci = jnp.array([9.0, -9.0])
    _, _, _, _, y_before_override, *_ = train_step(fx_leaky_model, param_opt_state, elig0, fx_states_prev, fx_observation, control_input=override_ci)
    _, _, _, _, y_before_default, *_ = train_step(fx_leaky_model, param_opt_state, elig0, fx_states_prev, fx_observation)

    assert not bool(jnp.allclose(y_before_override, y_before_default, atol=1e-4))


# ---- make_train_run_traced ----------------------------------------------------

def _fresh_states_prev(seed):
    sizes = [EL_CONTROL_SIZE] + list(EL_HIDDEN_SIZES)
    return [jr.normal(k, (n,)) for k, n in zip(jr.split(jr.key(seed), len(sizes)), sizes)]


def test_make_train_run_traced_matches_make_train_step_traced_looped_manually(fx_leaky_model, fx_control_input):
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    n_infer_steps, run_length = 5, 4

    states_prev = _fresh_states_prev(1000)
    ys = jr.normal(jr.key(1001), (run_length, EL_OBS_SIZE))
    param_opt_state = param_optim.init(eqx.filter(fx_leaky_model, eqx.is_array))
    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=fx_control_input)

    train_step = make_train_step_traced(param_optim, activity_optim, n_infer_steps, control_input=fx_control_input)
    m, ops, e, sp = fx_leaky_model, param_opt_state, elig0, states_prev
    ref_e_before, ref_e_after = [], []
    for i in range(run_length):
        m, ops, e, sp, y_bef, y_aft, e_bef, e_aft, _ = train_step(m, ops, e, sp, ys[i])
        ref_e_before.append(e_bef)
        ref_e_after.append(e_aft)

    train_run = make_train_run_traced(param_optim, activity_optim, n_infer_steps, run_length=run_length, control_input=fx_control_input)
    fused_model, fused_ops, fused_elig, fused_sp, y_before, y_after, e_before, e_after, _ = train_run(
        fx_leaky_model, param_opt_state, elig0, states_prev, ys
    )

    assert_allclose(e_before, jnp.stack(ref_e_before), "energies_before: make_train_run_traced vs looped make_train_step_traced", atol=1e-4, rtol=1e-4)
    assert_allclose(e_after, jnp.stack(ref_e_after), "energies_after: make_train_run_traced vs looped make_train_step_traced", atol=1e-4, rtol=1e-4)

    for f, r in zip(_leaves(fused_model), _leaves(m)):
        assert_allclose(f, r, "final weights: make_train_run_traced vs looped", atol=1e-4, rtol=1e-4)
    assert_allclose(fused_elig.e_A, e.e_A, "final eligibility state e_A: fused vs looped", atol=1e-4, rtol=1e-4)
    for fp, rp in zip(fused_sp, sp):
        assert_allclose(fp, rp, "final states_prev: fused vs looped", atol=1e-4, rtol=1e-4)


def test_make_train_run_traced_reduces_energy_over_epochs_readout_mode():
    key = jr.key(3)
    m = TpchModel(control_layer_size=4, hidden_sizes=(5,), obs_size=6, key=key, control_alpha=0.3, hidden_alphas=(0.3,))
    sp = [jnp.zeros(4), jnp.zeros(5)]
    T = 15
    ys = jr.normal(jr.key(0), (T, 6)) * 0.3

    train_run = make_train_run_traced(optax.adam(1e-2), optax.sgd(0.1), n_infer_steps=8, run_length=T)
    opt_state = optax.adam(1e-2).init(eqx.filter(m, eqx.is_array))
    e = m.zero_eligibility_state()

    energies = []
    for _ in range(20):
        m, opt_state, e, sc, yhb, yha, eb, ea, _ = train_run(m, opt_state, e, sp, ys)
        energies.append(float(jnp.mean(ea)))

    assert energies[-1] < energies[0]
    assert all(jnp.isfinite(jnp.array(energies)))


def test_make_train_run_traced_reduces_energy_over_epochs_accumulate_mode():
    key = jr.key(3)
    m = TpchModel(control_layer_size=4, hidden_sizes=(5,), obs_size=6, key=key, control_alpha=0.3, hidden_alphas=(0.3,), trace_mode="accumulate")
    sp = [jnp.zeros(4), jnp.zeros(5)]
    T = 15
    ys = jr.normal(jr.key(0), (T, 6)) * 0.3

    train_run = make_train_run_traced(optax.adam(1e-2), optax.sgd(0.1), n_infer_steps=8, run_length=T)
    opt_state = optax.adam(1e-2).init(eqx.filter(m, eqx.is_array))
    e = m.zero_eligibility_state()

    energies = []
    for _ in range(20):
        m, opt_state, e, sc, yhb, yha, eb, ea, _ = train_run(m, opt_state, e, sp, ys)
        energies.append(float(jnp.mean(ea)))

    assert energies[-1] < energies[0]
    assert all(jnp.isfinite(jnp.array(energies)))


def test_make_train_run_traced_with_no_leaky_layers_matches_make_train_run(fx_control_input):
    """When nothing is leaky, param_grad_traced == param_grad exactly, so
    make_train_run_traced should produce results identical to the plain
    (non-traced) make_train_run on the same non-leaky model."""
    key = jr.key(77)
    m = TpchModel(control_layer_size=3, hidden_sizes=(4,), obs_size=5, key=key, input_size=2)
    sp = _fresh_states_prev(1002)
    ys = jr.normal(jr.key(1003), (4, EL_OBS_SIZE))
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)

    opt_state_a = param_optim.init(eqx.filter(m, eqx.is_array))
    elig0 = m.zero_eligibility_state(control_input_example=fx_control_input)
    train_run_traced = make_train_run_traced(param_optim, activity_optim, n_infer_steps=5, run_length=4, control_input=fx_control_input)
    model_a, ops_a, elig_a, sp_a, y_before_a, y_after_a, e_before_a, e_after_a, _ = train_run_traced(m, opt_state_a, elig0, sp, ys)

    opt_state_b = param_optim.init(eqx.filter(m, eqx.is_array))
    train_run_plain = make_train_run(param_optim, activity_optim, n_infer_steps=5, run_length=4, control_input=fx_control_input)
    model_b, ops_b, sp_b, y_before_b, y_after_b, e_before_b, e_after_b, _ = train_run_plain(m, opt_state_b, sp, ys)

    assert_allclose(e_before_a, e_before_b, "energies_before: traced vs plain runner, no leaky layers", atol=1e-4, rtol=1e-4)
    assert_allclose(e_after_a, e_after_b, "energies_after: traced vs plain runner, no leaky layers", atol=1e-4, rtol=1e-4)
    for a, b in zip(_leaves(model_a), _leaves(model_b)):
        assert_allclose(a, b, "final weights: traced vs plain runner, no leaky layers", atol=1e-4, rtol=1e-4)


def test_make_train_run_traced_run_length_matches_scan_length(fx_leaky_model, fx_control_input):
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    param_opt_state = param_optim.init(eqx.filter(fx_leaky_model, eqx.is_array))
    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=fx_control_input)
    sp = _fresh_states_prev(1004)

    for run_length in [1, 3, 6]:
        train_run = make_train_run_traced(param_optim, activity_optim, n_infer_steps=4, run_length=run_length, control_input=fx_control_input)
        ys = jr.normal(jr.fold_in(jr.key(1005), run_length), (run_length, EL_OBS_SIZE))
        _, _, _, _, y_before, y_after, e_before, e_after, _ = train_run(fx_leaky_model, param_opt_state, elig0, sp, ys)
        assert e_before.shape == (run_length,)
        assert y_before.shape == (run_length, EL_OBS_SIZE)


def test_make_train_run_traced_return_layerwise_true_gives_correctly_shaped_traces(fx_leaky_model, fx_control_input):
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    param_opt_state = param_optim.init(eqx.filter(fx_leaky_model, eqx.is_array))
    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=fx_control_input)
    sp = _fresh_states_prev(1006)
    run_length, n_infer_steps = 3, 5
    ys = jr.normal(jr.key(1007), (run_length, EL_OBS_SIZE))

    train_run = make_train_run_traced(param_optim, activity_optim, n_infer_steps, run_length=run_length, control_input=fx_control_input)
    *_, energy_traces = train_run(fx_leaky_model, param_opt_state, elig0, sp, ys, return_layerwise=True)
    assert energy_traces.shape == (run_length, n_infer_steps, len(EL_HIDDEN_SIZES) + 2)
    assert jnp.all(jnp.isfinite(energy_traces))
