"""Per-node learning rates (graph/optim_utils.py), including the checkpoint question."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from pc_nox.models.graph import GraphModel, GraphModelConfig, NodeSpec
from pc_nox.models.graph.optim_utils import grouped_optimiser, resolve_groups
from pc_nox.models.graph.runners import make_train_step

KEY = jr.PRNGKey(0)


def config():
    return GraphModelConfig(node_specs=(
        NodeSpec("image", "input", (2, 8, 8), clamped=True),
        NodeSpec("c1", "pcn_conv", (4, 4, 4), parents=("image",)),
        NodeSpec("c2", "pcn_conv", (6, 2, 2), parents=("c1",)),
        NodeSpec("fc", "pcn", 10, parents=("c2",)),
        NodeSpec("label", "pcn", 3, parents=("fc",), act_fn="identity", energy="ce", clamped=True)))


def test_name_rules_beat_kind_rules_and_unmatched_nodes_use_the_default():
    group_of, scale_of = resolve_groups(config(), {"kind:pcn_conv": 0.1, "name:c2": 0.5})
    assert group_of == {"image": "default", "c1": "kind:pcn_conv", "c2": "name:c2", "fc": "default", "label": "default"}
    assert scale_of == {"default": 1.0, "kind:pcn_conv": 0.1, "name:c2": 0.5}


@pytest.mark.parametrize("rules, message", [
    ({"kind:nope": 0.1}, "matches no node"), ({"name:zzz": 0.1}, "matches no node"),
    ({"c1": 0.1}, "must look like"), ({"layer:c1": 0.1}, "must look like"), ({"name:c1": 0.0}, "scale must be > 0")])
def test_bad_rules_are_rejected_with_a_clear_message(rules, message):
    with pytest.raises(ValueError, match=message):
        resolve_groups(config(), rules)


def test_each_node_is_updated_at_its_own_rate():
    cfg = config()
    model = GraphModel(cfg, key=KEY)
    optim = grouped_optimiser(cfg, optax.sgd, 1.0, {"kind:pcn_conv": 0.1, "name:fc": 0.5})
    params = eqx.filter(model, eqx.is_array)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)                    # unit gradient everywhere
    updates, _ = optim.update(grads, optim.init(params), params)
    step = lambda name: float(-jax.tree_util.tree_leaves(updates.nodes[name])[0].reshape(-1)[0])
    assert np.isclose(step("c1"), 0.1) and np.isclose(step("c2"), 0.1)       # kind rule
    assert np.isclose(step("fc"), 0.5)                                         # name rule
    assert np.isclose(step("label"), 1.0)                                      # default


def test_grouped_activity_optimiser_scales_per_node_states():
    cfg = config()
    optim = grouped_optimiser(cfg, optax.sgd, 1.0, {"name:fc": 0.25}, target="states")
    states = {"c1": jnp.ones((4, 4, 4)), "c2": jnp.ones((6, 2, 2)), "fc": jnp.ones(10)}
    updates, _ = optim.update(jax.tree_util.tree_map(jnp.ones_like, states), optim.init(states), states)
    assert np.isclose(float(updates["fc"][0]), -0.25) and np.isclose(float(updates["c1"][0, 0, 0]), -1.0)


def test_grouped_optimisers_drive_the_shared_train_step():
    cfg = config()
    model = GraphModel(cfg, key=KEY)
    p_opt = grouped_optimiser(cfg, optax.adam, 1e-3, {"kind:pcn_conv": 0.1})
    a_opt = grouped_optimiser(cfg, optax.sgd, 0.1, {"kind:pcn_conv": 0.5}, target="states")
    state = p_opt.init(eqx.filter(model, eqx.is_array))
    obs = {"image": jr.uniform(KEY, (4, 2, 8, 8)), "label": jax.nn.one_hot(jnp.arange(4) % 3, 3)}
    new_model, state, *logs = make_train_step(p_opt, a_opt, n_infer_steps=5)(model, state, obs)
    assert all(np.isfinite(np.asarray(l)).all() for l in jax.tree_util.tree_leaves(logs))
    assert not jnp.array_equal(new_model.nodes["fc"].td_op.weight, model.nodes["fc"].td_op.weight)


def test_a_grouped_optimiser_state_round_trips_through_the_existing_checkpoint_code(tmp_path):
    """The question: do multiple learning rates need multiple opt_states in the checkpoint? No -- one state."""
    cfg = config()
    model = GraphModel(cfg, key=KEY)
    p_opt = grouped_optimiser(cfg, optax.adam, 1e-3, {"kind:pcn_conv": 0.1})
    a_opt = grouped_optimiser(cfg, optax.sgd, 0.1, {}, target="states")
    state = p_opt.init(eqx.filter(model, eqx.is_array))
    obs = {"image": jr.uniform(KEY, (4, 2, 8, 8)), "label": jax.nn.one_hot(jnp.arange(4) % 3, 3)}
    step = make_train_step(p_opt, a_opt, n_infer_steps=3)
    for _ in range(3):
        model, state, *_ = step(model, state, obs)                           # a non-trivial (non-zero) optimiser state

    path = model.save_checkpoint(path=tmp_path / "ck", opt_state=state,
                                 metadata={"lr_scales": {"kind:pcn_conv": 0.1}, "base_lr": 1e-3})
    loaded = GraphModel.load_checkpoint(path, optim=grouped_optimiser(cfg, optax.adam, 1e-3, {"kind:pcn_conv": 0.1}))
    for a, b in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(loaded.opt_state)):
        assert jnp.array_equal(a, b)
    assert loaded.metadata["lr_scales"] == {"kind:pcn_conv": 0.1}
    # training resumes identically from the restored state
    m1, s1, *_ = step(model, state, obs)
    m2, s2, *_ = step(loaded.model, loaded.opt_state, obs)
    assert all(jnp.allclose(x, y, atol=1e-6) for x, y in zip(jax.tree_util.tree_leaves(eqx.filter(m1, eqx.is_array)),
                                                              jax.tree_util.tree_leaves(eqx.filter(m2, eqx.is_array))))
