"""tests/control_input_threading_test.py

Regression coverage for a real bug that was fixed in
`pc_nox/engine/runners_temporal.py`: every multi-frame `*_run` factory
(`make_train_run`, `make_eval_run`, `make_train_run_diffrax`,
`make_eval_run_diffrax`, `make_train_run_traced`) accepts a per-frame
`control_inputs=` array (leading axis length `run_length`, scanned
alongside `ys`), meant to let each frame of a sequence see its OWN
control input rather than a single value held constant for the whole
run. Before the fix, the scan body silently ignored the per-frame slice
and used the factory-level constant `control_input` for every frame
instead -- and no existing test caught it, because every existing
`*_control_input_actually_affects_predictions` test only ever compares
"a single constant control_input" against "no control_input at all"; it
never actually varies the control input BETWEEN frames of the same run.

Every test below builds a `control_inputs` array where each frame is
DELIBERATELY different from every other frame, then checks the fused
`jax.lax.scan` run against an eager, frame-by-frame loop over the
corresponding single-frame `*_step` factory (already independently
verified in `tpch_test.py`), passing each frame's own control input as
that step's `control_input=` override. This is exactly the comparison
that would have failed under the pre-fix behaviour (the fused run would
have used the constant `control_input`/`None` for every frame instead),
and passes now.

A second group of tests locks down the complementary regression: when
`control_inputs=` is *omitted* (the old, single-input calling
convention), every frame must still fall back to the factory-level
constant `control_input`, unchanged from before this feature existed.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import diffrax
import pytest

from pc_nox.models.tpch.model import TpchModel
from pc_nox.engine.runners_temporal import (
    make_train_step, make_train_run,
    make_eval_step, make_eval_run,
    make_train_step_diffrax, make_train_run_diffrax,
    make_eval_step_diffrax, make_eval_run_diffrax,
    make_train_step_traced, make_train_run_traced,
)


def assert_allclose(actual, expected, name, atol=1e-4, rtol=1e-4):
    actual, expected = jnp.asarray(actual), jnp.asarray(expected)
    assert actual.shape == expected.shape, (
        f"{name}: shape mismatch -- got {actual.shape}, expected {expected.shape}"
    )
    max_abs_diff = float(jnp.max(jnp.abs(actual - expected))) if actual.size else 0.0
    assert jnp.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name}: values differ, max abs diff = {max_abs_diff}"
    )


def _leaves(pytree):
    return jax.tree_util.tree_leaves(eqx.filter(pytree, eqx.is_array))


CI_CONTROL_SIZE = 3
CI_HIDDEN_SIZES = [4]
CI_OBS_SIZE = 5
CI_INPUT_SIZE = 2
RUN_LENGTH = 4


@pytest.fixture
def fx_model():
    return TpchModel(
        control_layer_size=CI_CONTROL_SIZE,
        hidden_sizes=CI_HIDDEN_SIZES,
        obs_size=CI_OBS_SIZE,
        key=jr.key(200),
        input_size=CI_INPUT_SIZE,
    )


@pytest.fixture
def fx_leaky_model():
    """A leaky model too -- make_train_run_traced has the exact same
    per-frame-control-input scan-body pattern as the plain runners, so it
    needs the same regression coverage."""
    return TpchModel(
        control_layer_size=CI_CONTROL_SIZE,
        hidden_sizes=CI_HIDDEN_SIZES,
        obs_size=CI_OBS_SIZE,
        key=jr.key(200),
        input_size=CI_INPUT_SIZE,
        control_alpha=0.4,
        hidden_alphas=(0.3,),
    )


@pytest.fixture
def fx_states_prev():
    sizes = [CI_CONTROL_SIZE] + list(CI_HIDDEN_SIZES)
    keys = jr.split(jr.key(201), len(sizes))
    return [jr.normal(k, (n,)) for k, n in zip(keys, sizes)]


@pytest.fixture
def fx_ys():
    return jr.normal(jr.key(202), (RUN_LENGTH, CI_OBS_SIZE))


@pytest.fixture
def fx_varying_control_inputs():
    """One genuinely different control input per frame -- NOT drawn from
    a single broadcastable value, so a bug that silently uses only
    frame 0 (or the factory constant) for every frame cannot pass by
    accident."""
    return jr.normal(jr.key(203), (RUN_LENGTH, CI_INPUT_SIZE)) * jnp.array([[1.0], [3.0], [-2.0], [5.0]])


_DIFFRAX_KWARGS = dict(
    max_t1=5.0, n_save=8, solver=diffrax.Heun(),
    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3), steady_state_tol=1e-2,
)


# =============================================================================
# 1. make_train_run
# =============================================================================

def test_make_train_run_per_frame_control_inputs_match_looped_train_step(fx_model, fx_states_prev, fx_ys, fx_varying_control_inputs):
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    n_infer_steps = 5
    opt_state = param_optim.init(eqx.filter(fx_model, eqx.is_array))

    # Reference: make_train_step, looped by hand, giving EACH frame its own
    # control_input override (the ground truth for "per-frame varying input").
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps, control_input=None)
    m, ops, sp = fx_model, opt_state, fx_states_prev
    ref_energies_after = []
    for i in range(RUN_LENGTH):
        m, ops, sp, y_bef, y_aft, e_bef, e_aft, _ = train_step(m, ops, sp, fx_ys[i], control_input=fx_varying_control_inputs[i])
        ref_energies_after.append(e_aft)

    # Fused: make_train_run with control_inputs= (per-frame array).
    train_run = make_train_run(param_optim, activity_optim, n_infer_steps, run_length=RUN_LENGTH, control_input=None)
    fused_model, fused_ops, fused_sp, y_before, y_after, e_before, e_after, _ = train_run(
        fx_model, opt_state, fx_states_prev, fx_ys, control_inputs=fx_varying_control_inputs
    )

    assert_allclose(e_after, jnp.stack(ref_energies_after), "energies_after: per-frame control_inputs vs looped per-frame control_input")
    for f, r in zip(_leaves(fused_model), _leaves(m)):
        assert_allclose(f, r, "final weights: per-frame control_inputs vs looped per-frame control_input")
    for fp, rp in zip(fused_sp, sp):
        assert_allclose(fp, rp, "final states: per-frame control_inputs vs looped per-frame control_input")


def test_make_train_run_per_frame_control_inputs_differ_from_using_the_first_frame_constantly(fx_model, fx_states_prev, fx_ys, fx_varying_control_inputs):
    """The exact shape of the historical bug: silently reusing a single
    (e.g. the first frame's, or the factory-level) control input for
    every frame instead of each frame's own. Confirm the correctly-wired
    per-frame run genuinely differs from the "constant" run -- if these
    matched, it would mean control_inputs was being ignored."""
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)

    train_run_varying = make_train_run(param_optim, activity_optim, n_infer_steps=5, run_length=RUN_LENGTH, control_input=None)
    opt_state = param_optim.init(eqx.filter(fx_model, eqx.is_array))
    _, _, _, _, _, _, e_after_varying, _ = train_run_varying(fx_model, opt_state, fx_states_prev, fx_ys, control_inputs=fx_varying_control_inputs)

    train_run_constant = make_train_run(param_optim, activity_optim, n_infer_steps=5, run_length=RUN_LENGTH, control_input=fx_varying_control_inputs[0])
    opt_state2 = param_optim.init(eqx.filter(fx_model, eqx.is_array))
    _, _, _, _, _, _, e_after_constant, _ = train_run_constant(fx_model, opt_state2, fx_states_prev, fx_ys)

    assert not bool(jnp.allclose(e_after_varying, e_after_constant, atol=1e-4)), (
        "per-frame varying control_inputs must give a genuinely different result "
        "than holding the first frame's control input constant across the whole run"
    )


def test_make_train_run_omitted_control_inputs_falls_back_to_constant(fx_model, fx_states_prev, fx_ys):
    """Regression: the OLD calling convention (no control_inputs= at
    all) must still work exactly as before -- every frame uses the
    factory-level constant `control_input`."""
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    constant_ci = jr.normal(jr.key(204), (CI_INPUT_SIZE,))
    n_infer_steps = 5

    train_step = make_train_step(param_optim, activity_optim, n_infer_steps, control_input=constant_ci)
    opt_state = param_optim.init(eqx.filter(fx_model, eqx.is_array))
    m, ops, sp = fx_model, opt_state, fx_states_prev
    ref_energies_after = []
    for i in range(RUN_LENGTH):
        m, ops, sp, y_bef, y_aft, e_bef, e_aft, _ = train_step(m, ops, sp, fx_ys[i])
        ref_energies_after.append(e_aft)

    train_run = make_train_run(param_optim, activity_optim, n_infer_steps, run_length=RUN_LENGTH, control_input=constant_ci)
    opt_state2 = param_optim.init(eqx.filter(fx_model, eqx.is_array))
    _, _, _, _, _, _, e_after, _ = train_run(fx_model, opt_state2, fx_states_prev, fx_ys)  # control_inputs omitted

    assert_allclose(e_after, jnp.stack(ref_energies_after), "energies_after: control_inputs omitted, must match constant-control_input loop")


# =============================================================================
# 2. make_eval_run
# =============================================================================

def test_make_eval_run_per_frame_control_inputs_match_looped_eval_step(fx_model, fx_states_prev, fx_ys, fx_varying_control_inputs):
    activity_optim = optax.adam(1e-2)
    n_infer_steps = 5

    eval_step = make_eval_step(activity_optim, n_infer_steps, control_input=None)
    sp = fx_states_prev
    ref_energies_after = []
    for i in range(RUN_LENGTH):
        sp, y_bef, y_aft, e_bef, e_aft, _ = eval_step(fx_model, sp, fx_ys[i], control_input=fx_varying_control_inputs[i])
        ref_energies_after.append(e_aft)

    eval_run = make_eval_run(activity_optim, n_infer_steps, run_length=RUN_LENGTH, control_input=None)
    fused_sp, y_before, y_after, e_before, e_after, _ = eval_run(
        fx_model, fx_states_prev, fx_ys, control_inputs=fx_varying_control_inputs
    )

    assert_allclose(e_after, jnp.stack(ref_energies_after), "eval_run: energies_after per-frame control_inputs vs looped eval_step")
    for fp, rp in zip(fused_sp, sp):
        assert_allclose(fp, rp, "eval_run: final states per-frame control_inputs vs looped eval_step")


def test_make_eval_run_per_frame_control_inputs_differ_from_constant(fx_model, fx_states_prev, fx_ys, fx_varying_control_inputs):
    activity_optim = optax.adam(1e-2)
    eval_run = make_eval_run(activity_optim, n_infer_steps=5, run_length=RUN_LENGTH, control_input=None)
    _, _, _, _, e_after_varying, _ = eval_run(fx_model, fx_states_prev, fx_ys, control_inputs=fx_varying_control_inputs)

    eval_run_constant = make_eval_run(activity_optim, n_infer_steps=5, run_length=RUN_LENGTH, control_input=fx_varying_control_inputs[0])
    _, _, _, _, e_after_constant, _ = eval_run_constant(fx_model, fx_states_prev, fx_ys)

    assert not bool(jnp.allclose(e_after_varying, e_after_constant, atol=1e-4))


def test_make_eval_run_omitted_control_inputs_falls_back_to_constant(fx_model, fx_states_prev, fx_ys):
    activity_optim = optax.adam(1e-2)
    constant_ci = jr.normal(jr.key(205), (CI_INPUT_SIZE,))
    n_infer_steps = 5

    eval_step = make_eval_step(activity_optim, n_infer_steps, control_input=constant_ci)
    sp = fx_states_prev
    ref_energies_after = []
    for i in range(RUN_LENGTH):
        sp, y_bef, y_aft, e_bef, e_aft, _ = eval_step(fx_model, sp, fx_ys[i])
        ref_energies_after.append(e_aft)

    eval_run = make_eval_run(activity_optim, n_infer_steps, run_length=RUN_LENGTH, control_input=constant_ci)
    _, _, _, _, e_after, _ = eval_run(fx_model, fx_states_prev, fx_ys)

    assert_allclose(e_after, jnp.stack(ref_energies_after), "eval_run: energies_after, control_inputs omitted, must match constant loop")


# =============================================================================
# 3. make_train_run_diffrax
# =============================================================================

def test_make_train_run_diffrax_per_frame_control_inputs_match_looped_train_step_diffrax(fx_model, fx_states_prev, fx_ys, fx_varying_control_inputs):
    param_optim = optax.adam(1e-3)
    opt_state = param_optim.init(eqx.filter(fx_model, eqx.is_array))

    train_step = make_train_step_diffrax(param_optim, control_input=None, **_DIFFRAX_KWARGS)
    m, ops, sp = fx_model, opt_state, fx_states_prev
    ref_energies_after = []
    for i in range(RUN_LENGTH):
        m, ops, sp, y_bef, y_aft, e_bef, e_aft, trace, ts = train_step(m, ops, sp, fx_ys[i], control_input=fx_varying_control_inputs[i])
        ref_energies_after.append(e_aft)

    train_run = make_train_run_diffrax(param_optim, run_length=RUN_LENGTH, control_input=None, **_DIFFRAX_KWARGS)
    fused_model, fused_ops, fused_sp, y_before, y_after, e_before, e_after, _, _ = train_run(
        fx_model, opt_state, fx_states_prev, fx_ys, control_inputs=fx_varying_control_inputs
    )

    assert_allclose(e_after, jnp.stack(ref_energies_after), "train_run_diffrax: energies_after per-frame control_inputs vs looped")
    for f, r in zip(_leaves(fused_model), _leaves(m)):
        assert_allclose(f, r, "train_run_diffrax: final weights per-frame control_inputs vs looped")


def test_make_train_run_diffrax_per_frame_control_inputs_differ_from_constant(fx_model, fx_states_prev, fx_ys, fx_varying_control_inputs):
    param_optim = optax.adam(1e-3)
    train_run = make_train_run_diffrax(param_optim, run_length=RUN_LENGTH, control_input=None, **_DIFFRAX_KWARGS)
    opt_state = param_optim.init(eqx.filter(fx_model, eqx.is_array))
    _, _, _, _, _, _, e_after_varying, _, _ = train_run(fx_model, opt_state, fx_states_prev, fx_ys, control_inputs=fx_varying_control_inputs)

    train_run_constant = make_train_run_diffrax(param_optim, run_length=RUN_LENGTH, control_input=fx_varying_control_inputs[0], **_DIFFRAX_KWARGS)
    opt_state2 = param_optim.init(eqx.filter(fx_model, eqx.is_array))
    _, _, _, _, _, _, e_after_constant, _, _ = train_run_constant(fx_model, opt_state2, fx_states_prev, fx_ys)

    assert not bool(jnp.allclose(e_after_varying, e_after_constant, atol=1e-4))


# =============================================================================
# 4. make_eval_run_diffrax
# =============================================================================

def test_make_eval_run_diffrax_per_frame_control_inputs_match_looped_eval_step_diffrax(fx_model, fx_states_prev, fx_ys, fx_varying_control_inputs):
    eval_step = make_eval_step_diffrax(control_input=None, **_DIFFRAX_KWARGS)
    sp = fx_states_prev
    ref_energies_after = []
    for i in range(RUN_LENGTH):
        sp, y_bef, y_aft, e_bef, e_aft, trace, ts = eval_step(fx_model, sp, fx_ys[i], control_input=fx_varying_control_inputs[i])
        ref_energies_after.append(e_aft)

    eval_run = make_eval_run_diffrax(run_length=RUN_LENGTH, control_input=None, **_DIFFRAX_KWARGS)
    fused_sp, y_before, y_after, e_before, e_after, _, _ = eval_run(
        fx_model, fx_states_prev, fx_ys, control_inputs=fx_varying_control_inputs
    )

    assert_allclose(e_after, jnp.stack(ref_energies_after), "eval_run_diffrax: energies_after per-frame control_inputs vs looped")
    for fp, rp in zip(fused_sp, sp):
        assert_allclose(fp, rp, "eval_run_diffrax: final states per-frame control_inputs vs looped")


def test_make_eval_run_diffrax_per_frame_control_inputs_differ_from_constant(fx_model, fx_states_prev, fx_ys, fx_varying_control_inputs):
    eval_run = make_eval_run_diffrax(run_length=RUN_LENGTH, control_input=None, **_DIFFRAX_KWARGS)
    _, _, _, _, e_after_varying, _, _ = eval_run(fx_model, fx_states_prev, fx_ys, control_inputs=fx_varying_control_inputs)

    eval_run_constant = make_eval_run_diffrax(run_length=RUN_LENGTH, control_input=fx_varying_control_inputs[0], **_DIFFRAX_KWARGS)
    _, _, _, _, e_after_constant, _, _ = eval_run_constant(fx_model, fx_states_prev, fx_ys)

    assert not bool(jnp.allclose(e_after_varying, e_after_constant, atol=1e-4))


# =============================================================================
# 5. make_train_run_traced (eligibility-traced runner) -- same scan-body
# pattern as the plain runners, so it's exposed to exactly the same class
# of bug, on top of which it's also the newest/least-tested runner.
# =============================================================================

def test_make_train_run_traced_per_frame_control_inputs_match_looped_train_step_traced(fx_leaky_model, fx_states_prev, fx_ys, fx_varying_control_inputs):
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    n_infer_steps = 5
    opt_state = param_optim.init(eqx.filter(fx_leaky_model, eqx.is_array))
    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=fx_varying_control_inputs[0])

    train_step = make_train_step_traced(param_optim, activity_optim, n_infer_steps, control_input=None)
    m, ops, e, sp = fx_leaky_model, opt_state, elig0, fx_states_prev
    ref_energies_after = []
    for i in range(RUN_LENGTH):
        m, ops, e, sp, y_bef, y_aft, e_bef, e_aft, _ = train_step(m, ops, e, sp, fx_ys[i], control_input=fx_varying_control_inputs[i])
        ref_energies_after.append(e_aft)

    train_run = make_train_run_traced(param_optim, activity_optim, n_infer_steps, run_length=RUN_LENGTH, control_input=None)
    fused_model, fused_ops, fused_elig, fused_sp, y_before, y_after, e_before, e_after, _ = train_run(
        fx_leaky_model, opt_state, elig0, fx_states_prev, fx_ys, control_inputs=fx_varying_control_inputs
    )

    assert_allclose(e_after, jnp.stack(ref_energies_after), "train_run_traced: energies_after per-frame control_inputs vs looped")
    for f, r in zip(_leaves(fused_model), _leaves(m)):
        assert_allclose(f, r, "train_run_traced: final weights per-frame control_inputs vs looped")
    assert_allclose(fused_elig.e_A, e.e_A, "train_run_traced: final eligibility trace e_A per-frame control_inputs vs looped")


def test_make_train_run_traced_per_frame_control_inputs_differ_from_constant(fx_leaky_model, fx_states_prev, fx_ys, fx_varying_control_inputs):
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=fx_varying_control_inputs[0])

    train_run = make_train_run_traced(param_optim, activity_optim, n_infer_steps=5, run_length=RUN_LENGTH, control_input=None)
    opt_state = param_optim.init(eqx.filter(fx_leaky_model, eqx.is_array))
    _, _, _, _, _, _, _, e_after_varying, _ = train_run(fx_leaky_model, opt_state, elig0, fx_states_prev, fx_ys, control_inputs=fx_varying_control_inputs)

    train_run_constant = make_train_run_traced(param_optim, activity_optim, n_infer_steps=5, run_length=RUN_LENGTH, control_input=fx_varying_control_inputs[0])
    opt_state2 = param_optim.init(eqx.filter(fx_leaky_model, eqx.is_array))
    _, _, _, _, _, _, _, e_after_constant, _ = train_run_constant(fx_leaky_model, opt_state2, elig0, fx_states_prev, fx_ys)

    assert not bool(jnp.allclose(e_after_varying, e_after_constant, atol=1e-4))


def test_make_train_run_traced_omitted_control_inputs_falls_back_to_constant(fx_leaky_model, fx_states_prev, fx_ys):
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    constant_ci = jr.normal(jr.key(206), (CI_INPUT_SIZE,))
    n_infer_steps = 5
    elig0 = fx_leaky_model.zero_eligibility_state(control_input_example=constant_ci)

    train_step = make_train_step_traced(param_optim, activity_optim, n_infer_steps, control_input=constant_ci)
    opt_state = param_optim.init(eqx.filter(fx_leaky_model, eqx.is_array))
    m, ops, e, sp = fx_leaky_model, opt_state, elig0, fx_states_prev
    ref_energies_after = []
    for i in range(RUN_LENGTH):
        m, ops, e, sp, y_bef, y_aft, e_bef, e_aft, _ = train_step(m, ops, e, sp, fx_ys[i])
        ref_energies_after.append(e_aft)

    train_run = make_train_run_traced(param_optim, activity_optim, n_infer_steps, run_length=RUN_LENGTH, control_input=constant_ci)
    opt_state2 = param_optim.init(eqx.filter(fx_leaky_model, eqx.is_array))
    _, _, _, _, _, _, _, e_after, _ = train_run(fx_leaky_model, opt_state2, elig0, fx_states_prev, fx_ys)  # control_inputs omitted

    assert_allclose(e_after, jnp.stack(ref_energies_after), "train_run_traced: energies_after, control_inputs omitted, must match constant loop")


# =============================================================================
# 6. Cross-run consistency: goldilocks blocks (calling a *_run factory
# repeatedly with successive slices of a longer control_inputs array)
# must equal one big call over the whole array -- the same "splitting
# for side effects must not change what's computed" property
# tpch_test.py already checks for the constant-control_input case, now
# extended to the per-frame-varying case specifically.
# =============================================================================

def test_make_train_run_goldilocks_blocks_match_one_big_run_with_varying_control_inputs(fx_model, fx_states_prev):
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    n_infer_steps = 4
    block_len, n_blocks = 3, 3
    total_len = block_len * n_blocks

    ys = jr.normal(jr.key(210), (total_len, CI_OBS_SIZE))
    control_inputs = jr.normal(jr.key(211), (total_len, CI_INPUT_SIZE)) * jnp.arange(1, total_len + 1)[:, None]

    opt_state0 = param_optim.init(eqx.filter(fx_model, eqx.is_array))
    train_run_full = make_train_run(param_optim, activity_optim, n_infer_steps, run_length=total_len, control_input=None)
    full_model, full_opt_state, full_states, *_, full_e_before, full_e_after, _ = train_run_full(
        fx_model, opt_state0, fx_states_prev, ys, control_inputs=control_inputs
    )

    train_run_block = make_train_run(param_optim, activity_optim, n_infer_steps, run_length=block_len, control_input=None)
    model, opt_state, states_prev = fx_model, opt_state0, fx_states_prev
    block_e_after = []
    for b in range(n_blocks):
        ys_block = ys[b * block_len:(b + 1) * block_len]
        ci_block = control_inputs[b * block_len:(b + 1) * block_len]
        model, opt_state, states_prev, y_before, y_after, e_before, e_after, _ = train_run_block(
            model, opt_state, states_prev, ys_block, control_inputs=ci_block
        )
        block_e_after.append(e_after)
    block_e_after = jnp.concatenate(block_e_after)

    assert_allclose(block_e_after, full_e_after, "goldilocks blocks vs one big run, with genuinely varying per-frame control_inputs")
    for a, b in zip(_leaves(model), _leaves(full_model)):
        assert_allclose(a, b, "goldilocks blocks vs one big run: final weights, varying control_inputs")
