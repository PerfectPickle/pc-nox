"""tests/heterogeneous_observation_test.py

Pytest coverage for `TpchHeterogeneousObservationLayer` (node-level
heterogeneous observation layer, `pc_nox/models/tpch/layers.py`) both as
a standalone `eqx.Module` and wired into `TpchModel` via
`observation_groups=`. Previously untested: `tpch_test.py` only imports
the class name.

Style follows `tpch_test.py`: plain pytest functions + fixtures,
deliberately mismatched group/layer widths so a broadcast bug can't hide.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from pc_nox.models.tpch.layers import TpchHeterogeneousObservationLayer, TpchObservationLayer
from pc_nox.models.tpch.model import TpchModel
from pc_nox.models.tpch.config import TpchConfig
from pc_nox.engine.runners_temporal import make_train_step, make_eval_step


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


# =============================================================================
# Part 1: TpchHeterogeneousObservationLayer standalone -- no TpchModel
# =============================================================================

HO_PARENT_SIZE = 4
HO_GROUP_SPECS = (("sensory", 5, "mse"), ("action", 3, "ce"), ("scalar", 1, "mse"))
HO_TOTAL_SIZE = sum(size for _, size, _ in HO_GROUP_SPECS)  # 9


@pytest.fixture
def fx_hetero_layer():
    return TpchHeterogeneousObservationLayer(parent_size=HO_PARENT_SIZE, group_specs=HO_GROUP_SPECS, key=jr.key(50))


@pytest.fixture
def fx_parent_curr():
    return jr.normal(jr.key(51), (HO_PARENT_SIZE,))


def test_group_metadata_stored_correctly(fx_hetero_layer):
    assert fx_hetero_layer.group_names == ("sensory", "action", "scalar")
    assert fx_hetero_layer.group_sizes == (5, 3, 1)
    assert fx_hetero_layer.group_losses == ("mse", "ce", "mse")


def test_weights_returns_one_matrix_per_group_with_correct_shapes(fx_hetero_layer):
    ws = fx_hetero_layer.weights()
    assert len(ws) == 3
    expected_shapes = [(5, HO_PARENT_SIZE), (3, HO_PARENT_SIZE), (1, HO_PARENT_SIZE)]
    for w, shape in zip(ws, expected_shapes):
        assert w.shape == shape


def test_predict_output_width_matches_total_size(fx_hetero_layer, fx_parent_curr):
    y_hat = fx_hetero_layer.predict(fx_parent_curr)
    assert y_hat.shape == (HO_TOTAL_SIZE,)


def test_predict_is_concatenation_of_each_groups_own_linear_readout(fx_hetero_layer, fx_parent_curr):
    y_hat = fx_hetero_layer.predict(fx_parent_curr)
    manual = jnp.concatenate([g(fx_parent_curr) for g in fx_hetero_layer.groups])
    assert_allclose(y_hat, manual, "predict() concatenation order")

    # and slice-by-slice against each group's own weight matrix directly
    idx = 0
    for w, (name, size, loss) in zip(fx_hetero_layer.weights(), HO_GROUP_SPECS):
        assert_allclose(y_hat[idx:idx + size], w @ fx_parent_curr, f"group {name!r} slice of predict()")
        idx += size


def test_energy_mse_group_matches_manual_half_squared_error():
    layer = TpchHeterogeneousObservationLayer(parent_size=3, group_specs=[("only", 4, "mse")], key=jr.key(52))
    parent = jr.normal(jr.key(53), (3,))
    y_hat = layer.predict(parent)
    obs = jr.normal(jr.key(54), (4,))
    e = layer.energy(y_hat, obs)
    manual = 0.5 * jnp.sum((obs - y_hat) ** 2)
    assert_allclose(e, manual, "single-mse-group energy")


def test_energy_ce_group_matches_manual_cross_entropy():
    layer = TpchHeterogeneousObservationLayer(parent_size=3, group_specs=[("only", 4, "ce")], key=jr.key(55))
    parent = jr.normal(jr.key(56), (3,))
    y_hat = layer.predict(parent)
    onehot = jnp.array([0.0, 1.0, 0.0, 0.0])
    e = layer.energy(y_hat, onehot)
    manual = -jnp.sum(onehot * jax.nn.log_softmax(y_hat))
    assert_allclose(e, manual, "single-ce-group energy")


def test_energy_mixed_groups_sums_each_groups_own_loss(fx_hetero_layer, fx_parent_curr):
    y_hat = fx_hetero_layer.predict(fx_parent_curr)
    sensory_target = jr.normal(jr.key(57), (5,))
    action_onehot = jnp.array([0.0, 1.0, 0.0])
    scalar_target = jr.normal(jr.key(58), (1,))
    obs = jnp.concatenate([sensory_target, action_onehot, scalar_target])

    e = fx_hetero_layer.energy(y_hat, obs)

    y_sensory, y_action, y_scalar = y_hat[:5], y_hat[5:8], y_hat[8:9]
    manual = (
        0.5 * jnp.sum((sensory_target - y_sensory) ** 2)
        + -jnp.sum(action_onehot * jax.nn.log_softmax(y_action))
        + 0.5 * jnp.sum((scalar_target - y_scalar) ** 2)
    )
    assert_allclose(e, manual, "mixed-group energy sum")


def test_energy_is_additive_across_groups_independently():
    """Perturbing only one group's slice of y_hat should only change that
    group's own contribution to the total energy -- proves groups really
    are computed independently, not cross-contaminated."""
    layer = TpchHeterogeneousObservationLayer(parent_size=3, group_specs=[("a", 2, "mse"), ("b", 2, "mse")], key=jr.key(59))
    parent = jr.normal(jr.key(60), (3,))
    y_hat = layer.predict(parent)
    obs = jr.normal(jr.key(61), (4,))

    e_base = layer.energy(y_hat, obs)
    y_hat_perturbed = y_hat.at[0].add(1.0)  # only group "a"'s first element
    e_perturbed = layer.energy(y_hat_perturbed, obs)

    manual_delta = 0.5 * ((obs[0] - y_hat_perturbed[0]) ** 2 - (obs[0] - y_hat[0]) ** 2)
    assert_allclose(e_perturbed - e_base, manual_delta, "energy delta from perturbing only group a")


def test_single_group_degenerates_to_homogeneous_layer_behaviour():
    """A heterogeneous layer with exactly one mse group should behave
    identically to a plain TpchObservationLayer carrying the SAME weight
    matrix -- same predict() and same energy(). (Built from two
    independently-keyed layers, then forced to share one weight via
    eqx.tree_at, since TpchHeterogeneousObservationLayer's internal
    jr.split means the "same PRNGKey" trick used elsewhere in this file
    does not, by itself, give identical weights here.)"""
    homo = TpchObservationLayer(obs_size=6, parent_size=4, loss="mse", key=jr.key(62))
    hetero = TpchHeterogeneousObservationLayer(parent_size=4, group_specs=[("only", 6, "mse")], key=jr.key(65))
    hetero = eqx.tree_at(lambda h: h.groups[0].weight, hetero, homo.W_parent.weight)

    parent = jr.normal(jr.key(63), (4,))
    obs = jr.normal(jr.key(64), (6,))

    assert_allclose(hetero.predict(parent), homo.predict(parent), "predict(): single-group hetero vs homogeneous")
    assert_allclose(hetero.energy(hetero.predict(parent), obs), homo.energy(homo.predict(parent), obs), "energy(): single-group hetero vs homogeneous")
    assert_allclose(hetero.weights()[0], homo.weights()[0], "weight matrix: single-group hetero vs homogeneous")


# =============================================================================
# Part 2: TpchModel integration via observation_groups=
# =============================================================================

TM_CONTROL_SIZE = 3
TM_HIDDEN_SIZES = [4]
TM_GROUP_SPECS = (("sensory", 5, "mse"), ("action", 3, "ce"))
TM_OBS_SIZE = sum(size for _, size, _ in TM_GROUP_SPECS)  # 8


@pytest.fixture
def fx_hetero_model():
    return TpchModel(
        control_layer_size=TM_CONTROL_SIZE,
        hidden_sizes=TM_HIDDEN_SIZES,
        obs_size=TM_OBS_SIZE,
        key=jr.key(70),
        observation_groups=TM_GROUP_SPECS,
    )


@pytest.fixture
def fx_states_prev():
    sizes = [TM_CONTROL_SIZE] + list(TM_HIDDEN_SIZES)
    keys = jr.split(jr.key(71), len(sizes))
    return [jr.normal(k, (n,)) for k, n in zip(keys, sizes)]


@pytest.fixture
def fx_observation():
    sensory = jr.normal(jr.key(72), (5,))
    action = jnp.array([0.0, 0.0, 1.0])
    return jnp.concatenate([sensory, action])


def test_observation_groups_installs_heterogeneous_layer(fx_hetero_model):
    assert isinstance(fx_hetero_model.observation_layer, TpchHeterogeneousObservationLayer)
    assert fx_hetero_model.observation_layer.group_names == ("sensory", "action")


def test_observation_groups_and_observation_layer_are_mutually_exclusive():
    custom = TpchHeterogeneousObservationLayer(parent_size=4, group_specs=[("x", 3, "mse")], key=jr.key(73))
    with pytest.raises(ValueError):
        TpchModel(
            control_layer_size=3, hidden_sizes=[4], obs_size=3, key=jr.key(0),
            observation_layer=custom, observation_groups=[("x", 3, "mse")],
        )


def test_observation_groups_sizes_must_sum_to_obs_size():
    with pytest.raises(ValueError):
        TpchModel(
            control_layer_size=3, hidden_sizes=[4], obs_size=10, key=jr.key(0),
            observation_groups=[("a", 5, "mse"), ("b", 3, "ce")],  # sums to 8, not 10
        )


def test_predict_output_width_matches_obs_size(fx_hetero_model, fx_states_prev, fx_observation):
    states_curr = fx_hetero_model.init_activities(fx_states_prev, None, fx_observation)
    _, y_hat = fx_hetero_model.predict(fx_states_prev, states_curr, None, fx_observation)
    assert y_hat.shape == (TM_OBS_SIZE,)


def test_energy_is_finite_with_heterogeneous_layer(fx_hetero_model, fx_states_prev, fx_observation):
    states_curr = fx_hetero_model.init_activities(fx_states_prev, None, fx_observation)
    e = fx_hetero_model.tpch_energy_fn(fx_states_prev, states_curr, fx_observation, None)
    assert jnp.isfinite(e)


def test_energy_matches_manual_group_energy_plus_state_errors(fx_hetero_model, fx_states_prev, fx_observation):
    """tpch_energy_fn's total must equal the sum of the control/hidden
    state-prediction error terms plus the heterogeneous observation
    layer's own energy() -- exactly what layer_energies() computes in
    tpch_test.py, ported here for the heterogeneous layer."""
    m = fx_hetero_model
    states_curr = [jr.normal(jr.key(74 + i), (n,)) for i, n in enumerate([TM_CONTROL_SIZE] + TM_HIDDEN_SIZES)]
    predictions, y_hat = m.predict(fx_states_prev, states_curr, None, fx_observation)
    manual_state_terms = sum(0.5 * jnp.sum((s - p) ** 2) for s, p in zip(states_curr, predictions))
    manual_obs_term = m.observation_layer.energy(y_hat, fx_observation)
    manual_total = manual_state_terms + manual_obs_term

    e = m.tpch_energy_fn(fx_states_prev, states_curr, fx_observation, None)
    assert_allclose(e, manual_total, "tpch_energy_fn vs manual group-energy recomposition")


def test_param_grad_structure_matches_model_with_heterogeneous_layer(fx_hetero_model, fx_states_prev, fx_observation):
    m = fx_hetero_model
    states_curr = m.settle(fx_states_prev, fx_observation, None, n_steps=5)
    g = m.param_grad(fx_states_prev, states_curr, fx_observation, None)
    assert jax.tree_util.tree_structure(eqx.filter(g, eqx.is_array)) == jax.tree_util.tree_structure(eqx.filter(m, eqx.is_array))
    assert len(g.observation_layer.groups) == 2
    for leaf, model_leaf in zip(_leaves(g), _leaves(m)):
        assert leaf.shape == model_leaf.shape


def test_param_grad_matches_finite_differences_for_heterogeneous_obs_weight(fx_hetero_model, fx_states_prev, fx_observation):
    """Numeric gradient check (central differences) on one entry of the
    'action' (ce-loss) group's weight matrix -- confirms autodiff through
    the heterogeneous layer's per-group loss switch is actually correct,
    not just shaped correctly."""
    m = fx_hetero_model
    states_curr = m.settle(fx_states_prev, fx_observation, None, n_steps=5)
    analytic = m.param_grad(fx_states_prev, states_curr, fx_observation, None)

    eps = 1e-4
    i, j = 0, 0
    action_group_idx = 1  # "action" is group index 1

    def energy_with_perturbation(delta):
        new_weight = m.observation_layer.groups[action_group_idx].weight.at[i, j].add(delta)
        new_group = eqx.tree_at(lambda g: g.weight, m.observation_layer.groups[action_group_idx], new_weight)
        new_groups = list(m.observation_layer.groups)
        new_groups[action_group_idx] = new_group
        new_obs_layer = eqx.tree_at(lambda o: o.groups, m.observation_layer, new_groups)
        perturbed_model = eqx.tree_at(lambda mm: mm.observation_layer, m, new_obs_layer)
        return perturbed_model.tpch_energy_fn(fx_states_prev, states_curr, fx_observation, None)

    numeric = (energy_with_perturbation(eps) - energy_with_perturbation(-eps)) / (2 * eps)
    analytic_val = analytic.observation_layer.groups[action_group_idx].weight[i, j]
    assert_allclose(analytic_val, numeric, "finite-difference check on ce-group weight gradient", atol=1e-2, rtol=1e-2)


def test_settle_reduces_energy_with_heterogeneous_layer(fx_hetero_model, fx_states_prev, fx_observation):
    m = fx_hetero_model
    init = m.init_activities(fx_states_prev, None, fx_observation)
    e0 = m.tpch_energy_fn(fx_states_prev, init, fx_observation, None)
    settled = m.settle(fx_states_prev, fx_observation, None, n_steps=25, state_lr=0.1)
    e1 = m.tpch_energy_fn(fx_states_prev, settled, fx_observation, None)
    assert float(e1) < float(e0)


def test_all_weights_includes_every_group_matrix(fx_hetero_model):
    """_all_weights: control W_rec + hidden(W_rec,W_parent_prev,W_parent_curr) + 2 obs groups."""
    m = fx_hetero_model
    assert len(m._all_weights()) == 1 + 3 * len(TM_HIDDEN_SIZES) + len(TM_GROUP_SPECS)


def test_ff_weights_includes_every_group_matrix_none_are_recurrent(fx_hetero_model):
    """None of the heterogeneous layer's group weights are 'recurrent' --
    they must all show up in _ff_weights, none in _rec_weights."""
    m = fx_hetero_model
    ff = m._ff_weights()
    rec = m._rec_weights()
    group_weights = m.observation_layer.weights()
    for gw in group_weights:
        assert any(jnp.array_equal(gw, w) for w in ff)
        assert not any(w.shape == gw.shape and jnp.array_equal(gw, w) for w in rec if w.shape == gw.shape)


def test_weight_decay_applies_to_heterogeneous_group_weights():
    """weight_decay_scope='all' (or 'ff') must penalise the heterogeneous
    layer's own group weights, exactly like any other feedforward weight."""
    m_reg = TpchModel(
        control_layer_size=3, hidden_sizes=[4], obs_size=8, key=jr.key(80),
        observation_groups=TM_GROUP_SPECS, weight_decay=0.5, weight_decay_scope="all",
    )
    m_bare = TpchModel(
        control_layer_size=3, hidden_sizes=[4], obs_size=8, key=jr.key(80),
        observation_groups=TM_GROUP_SPECS,
    )
    manual_l2 = 0.5 * 0.5 * sum(jnp.sum(w ** 2) for w in m_reg._all_weights())
    assert_allclose(m_reg._weight_l2_reg(), manual_l2, "weight_decay over heterogeneous groups")
    assert_allclose(m_bare._weight_l2_reg(), jnp.asarray(0.0), "no weight_decay configured -> zero penalty")


def test_return_layerwise_observation_entry_equals_heterogeneous_energy(fx_hetero_model, fx_states_prev, fx_observation):
    m = fx_hetero_model
    states_curr = m.init_activities(fx_states_prev, None, fx_observation)
    per_layer = m.tpch_energy_fn(fx_states_prev, states_curr, fx_observation, None, return_layerwise=True)
    _, y_hat = m.predict(fx_states_prev, states_curr, None, fx_observation)
    manual_obs_term = m.observation_layer.energy(y_hat, fx_observation)
    assert_allclose(per_layer[-1], manual_obs_term, "return_layerwise observation-term entry vs manual")
    assert_allclose(jnp.sum(per_layer), m.tpch_energy_fn(fx_states_prev, states_curr, fx_observation, None), "return_layerwise sum vs scalar total")


def test_layer_labels_unaffected_by_heterogeneous_observation_layer():
    """layer_labels() is driven purely by hidden_sizes in the config, not
    by the observation layer's internal group structure -- confirm it
    still has exactly num_hidden+2 entries."""
    cfg = TpchConfig(control_layer_size=3, hidden_sizes=(4,), obs_size=8, observation_groups=TM_GROUP_SPECS)
    labels = TpchModel.layer_labels(cfg)
    assert labels == ["Control", "Hidden 1", "Observation"]


def test_config_round_trips_observation_groups_via_from_config():
    cfg = TpchConfig(control_layer_size=3, hidden_sizes=(4,), obs_size=8, observation_groups=TM_GROUP_SPECS)
    m = TpchModel.from_config(cfg, key=jr.key(81))
    assert isinstance(m.observation_layer, TpchHeterogeneousObservationLayer)
    assert m.observation_layer.group_names == ("sensory", "action")
    assert m.observation_layer.group_sizes == (5, 3)
    assert m.observation_layer.group_losses == ("mse", "ce")


def test_checkpoint_round_trip_preserves_heterogeneous_layer(fx_hetero_model, fx_states_prev, fx_observation, tmp_path):
    m = fx_hetero_model
    out_dir = m.save_checkpoint(path=tmp_path / "hetero_ckpt")
    loaded = TpchModel.load_checkpoint(out_dir)

    assert isinstance(loaded.model.observation_layer, TpchHeterogeneousObservationLayer)
    assert loaded.model.observation_layer.group_names == m.observation_layer.group_names
    assert loaded.model.observation_layer.group_sizes == m.observation_layer.group_sizes
    assert loaded.model.observation_layer.group_losses == m.observation_layer.group_losses

    for a, b in zip(_leaves(m), _leaves(loaded.model)):
        assert bool(jnp.array_equal(a, b))

    states_curr = m.init_activities(fx_states_prev, None, fx_observation)
    _, y_hat_orig = m.predict(fx_states_prev, states_curr, None, fx_observation)
    _, y_hat_loaded = loaded.model.predict(fx_states_prev, states_curr, None, fx_observation)
    assert_allclose(y_hat_orig, y_hat_loaded, "y_hat before vs after checkpoint round trip (heterogeneous layer)")


def test_custom_observation_layer_instance_is_a_documented_checkpoint_limitation():
    """observation_layer= (a fully custom instance, as opposed to
    observation_groups=) is explicitly documented as NOT round-tripping
    through from_config/checkpointing -- from_config has no way to know
    what instance was passed. This test locks in that documented
    behaviour: the reloaded model falls back to the DEFAULT homogeneous
    TpchObservationLayer, rather than (silently, worse) crashing or
    (silently, worse still) claiming to preserve the custom layer."""
    custom = TpchHeterogeneousObservationLayer(parent_size=4, group_specs=[("x", 3, "mse"), ("y", 2, "ce")], key=jr.key(82))
    m = TpchModel(control_layer_size=3, hidden_sizes=[4], obs_size=5, key=jr.key(83), observation_layer=custom)
    assert isinstance(m.observation_layer, TpchHeterogeneousObservationLayer)

    # from_config alone (no checkpoint needed) already can't know about it:
    rebuilt = TpchModel.from_config(m.config, key=jr.key(84))
    assert isinstance(rebuilt.observation_layer, TpchObservationLayer)
    assert not isinstance(rebuilt.observation_layer, TpchHeterogeneousObservationLayer)


# =============================================================================
# Part 3: heterogeneous layer through the shared training/eval runners --
# confirms "no special-casing needed" (the design goal stated in the
# layer's own docstring) actually holds for the real runner factories.
# =============================================================================

def test_make_train_step_runs_with_heterogeneous_layer(fx_hetero_model, fx_states_prev, fx_observation):
    param_optim = optax.adam(1e-3)
    activity_optim = optax.sgd(0.1)
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps=5)
    opt_state = param_optim.init(eqx.filter(fx_hetero_model, eqx.is_array))

    new_model, new_opt_state, states_curr, y_before, y_after, e_before, e_after, _ = train_step(
        fx_hetero_model, opt_state, fx_states_prev, fx_observation
    )
    assert jnp.isfinite(e_after)
    assert y_after.shape == (TM_OBS_SIZE,)
    for leaf in _leaves(new_model):
        assert jnp.all(jnp.isfinite(leaf))


def test_make_eval_step_runs_with_heterogeneous_layer_and_does_not_mutate_weights(fx_hetero_model, fx_states_prev, fx_observation):
    activity_optim = optax.adam(1e-2)
    eval_step = make_eval_step(activity_optim, n_infer_steps=5)
    states_curr, y_before, y_after, e_before, e_after, _ = eval_step(fx_hetero_model, fx_states_prev, fx_observation)
    assert jnp.isfinite(e_after)
    assert float(e_after) <= float(e_before) + 1e-4


def test_heterogeneous_layer_with_zero_hidden_layers():
    m = TpchModel(control_layer_size=3, hidden_sizes=[], obs_size=8, key=jr.key(90), observation_groups=TM_GROUP_SPECS)
    states_prev = [jr.normal(jr.key(91), (3,))]
    obs = jnp.concatenate([jr.normal(jr.key(92), (5,)), jnp.array([1.0, 0.0, 0.0])])
    states_curr = m.init_activities(states_prev, None, obs)
    e = m.tpch_energy_fn(states_prev, states_curr, obs, None)
    assert jnp.isfinite(e)
    settled = m.settle(states_prev, obs, None, n_steps=5)
    assert len(settled) == 1


def test_heterogeneous_layer_single_ce_group_only():
    m = TpchModel(control_layer_size=3, hidden_sizes=[4], obs_size=4, key=jr.key(93), observation_groups=[("action", 4, "ce")])
    states_prev = [jr.normal(jr.key(94), (3,)), jr.normal(jr.key(95), (4,))]
    onehot = jnp.array([0.0, 0.0, 1.0, 0.0])
    states_curr = m.init_activities(states_prev, None, onehot)
    e = m.tpch_energy_fn(states_prev, states_curr, onehot, None)
    assert jnp.isfinite(e)
    manual_energy_fn = lambda s: -jnp.sum(onehot * jax.nn.log_softmax(m.observation_layer.predict(s[-1])))
    # cross-check the observation term alone against a from-scratch ce computation
    obs_term_only = m.observation_layer.energy(m.observation_layer.predict(states_curr[-1]), onehot)
    assert_allclose(obs_term_only, manual_energy_fn(states_curr), "single-ce-group observation term")
