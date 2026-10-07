"""Tests for graph/runners.py: batch (vmap) and time (scan) axes, alone and composed."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from pc_nox.models.graph import GraphModel, GraphModelConfig, NodeSpec
from pc_nox.models.graph.runners import (make_eval_run, make_eval_step, make_train_run, make_train_step, zero_states)
from graph_model_test import make_pair

KEY = jr.PRNGKey(0)
ACT = optax.sgd(0.1)


def allclose(a, b, rtol=1e-4, atol=1e-5):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


def leaves_close(a, b, **kw):
    la = jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_array))
    return len(la) == len(lb) and all(allclose(x, z, **kw) for x, z in zip(la, lb))


# ---- fixtures -------------------------------------------------------------- #

def static_two_tail():
    """Purely static: learned-prior root, two clamped tails (one is the 'label')."""
    return GraphModelConfig(node_specs=(
        NodeSpec("core", "pcn", 6),
        NodeSpec("feat", "pcn", 5, parents=("core",)),
        NodeSpec("data", "pcn", 4, parents=("feat",), clamped=True, act_fn="identity"),
        NodeSpec("label", "pcn", 3, parents=("core",), clamped=True, act_fn="identity", energy="ce"),
    ))


def static_batch(B=3):
    k = jr.split(jr.PRNGKey(5), 2)
    return {"data": jr.normal(k[0], (B, 4)), "label": jax.nn.one_hot(jnp.arange(B) % 3, 3)}


def mixed_config():
    """Temporal control node (leaky, recurrent) feeding STATIC dense nodes."""
    return GraphModelConfig(node_specs=(
        NodeSpec("core", "tpch_control", 6, alpha=0.5),
        NodeSpec("h", "pcn", 5, parents=("core",)),
        NodeSpec("y", "pcn", 3, parents=("h",), clamped=True, act_fn="identity"),
    ))


def mixed_sequence(T=4, B=2):
    t = jnp.arange(T)[:, None, None]
    b = jnp.arange(B)[None, :, None]
    return {"y": jnp.sin(0.5 * t + b + jnp.arange(3)[None, None, :])}        # (T, B, 3)


# ---- batch axis ------------------------------------------------------------ #

def test_zero_states_shapes():
    cfg = static_two_tail()
    assert {k: v.shape for k, v in zero_states(cfg).items()} == {"core": (6,), "feat": (5,)}
    assert {k: v.shape for k, v in zero_states(cfg, 7).items()} == {"core": (7, 6), "feat": (7, 5)}


def test_static_graph_trains_with_no_states_prev_and_energy_falls():
    model = GraphModel(static_two_tail(), key=KEY)
    optim = optax.adam(1e-2)
    state = optim.init(eqx.filter(model, eqx.is_array))
    step = make_train_step(optim, ACT, n_infer_steps=10)
    obs = static_batch()
    e_first = None
    for _ in range(30):
        model, state, states, yb, ya, e_before, e_after, trace = step(model, state, obs)
        e_first = e_first if e_first is not None else float(jnp.mean(e_after))
    assert float(jnp.mean(e_after)) < e_first
    assert states["core"].shape == (3, 6) and e_after.shape == (3,) and trace is None
    assert set(ya) == {"data", "label"} and ya["label"].shape == (3, 3)


def test_batch_gradient_is_the_mean_of_per_example_gradients():
    model = GraphModel(static_two_tail(), key=KEY)
    obs = static_batch(B=3)
    lr = 0.05
    step = make_train_step(optax.sgd(lr), ACT, n_infer_steps=8)
    new_model, *_ = step(model, optax.sgd(lr).init(eqx.filter(model, eqx.is_array)), obs)

    grads = []
    for i in range(3):
        o = {k: v[i] for k, v in obs.items()}
        s = model.settle_scan(ACT, {}, o, n_steps=8)
        grads.append(model.param_grad({}, s, o))
    mean_grad = jax.tree_util.tree_map(lambda *g: sum(g) / 3, *grads)
    expected = eqx.apply_updates(model, jax.tree_util.tree_map(lambda g: -lr * g, mean_grad))
    assert leaves_close(new_model, expected)


def test_eval_step_with_the_label_omitted_still_predicts_it():
    model = GraphModel(static_two_tail(), key=KEY)
    obs = static_batch()
    eval_step = make_eval_step(ACT, n_infer_steps=8)
    states, yb, ya, e_before, e_after, trace = eval_step(model, {"data": obs["data"]})
    assert ya["label"].shape == (3, 3)                                    # label predicted from the settled core
    # the label tail contributes nothing to the energy when unobserved
    only_data = jax.vmap(lambda s, d: model.energy_fn({}, s, {"data": d}))(states, obs["data"])
    assert allclose(e_after, only_data)


def test_return_layerwise_gives_a_per_example_trace():
    model = GraphModel(static_two_tail(), key=KEY)
    eval_step = make_eval_step(ACT, n_infer_steps=6)
    *_, trace = eval_step(model, static_batch(), return_layerwise=True)
    assert trace.shape == (3, 6, 4)                                       # (batch, steps, nodes)


def test_temporal_graph_without_states_prev_fails_with_a_clear_message():
    model = GraphModel(mixed_config(), key=KEY)
    step = make_eval_step(ACT, n_infer_steps=3)
    obs = {"y": mixed_sequence()["y"][0]}
    with pytest.raises(KeyError, match="previous state"):
        step(model, obs)


# ---- regression: batch of one == the existing temporal runner ---------------- #

def test_batch_of_one_matches_the_temporal_runner():
    runners_temporal = pytest.importorskip("pc_nox.engine.runners_temporal",
                                           reason="adjust this import to where runners_temporal.py lives")
    _, model = make_pair(weight_decay=0.05, weight_decay_scope="ff")
    k = jr.split(jr.PRNGKey(3), 4)
    states_prev = {"core": jr.normal(k[0], (8,)), "h1": jr.normal(k[1], (6,)), "h2": jr.normal(k[2], (5,))}
    y = {"y": jr.normal(k[3], (4,))}
    x = jnp.array([0.3, -0.2, 0.9])

    optim = optax.sgd(0.02)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    ref = runners_temporal.make_train_step(optim, ACT, n_infer_steps=7)
    m_ref, _, s_ref, _, ya_ref, eb_ref, ea_ref, _ = ref(model, opt_state, states_prev, y, control_input=x)

    add_batch = lambda t: jax.tree_util.tree_map(lambda a: a[None], t)
    new = make_train_step(optim, ACT, n_infer_steps=7)
    m_new, _, s_new, _, ya_new, eb_new, ea_new, _ = new(
        model, opt_state, add_batch(y), states_prev=add_batch(states_prev), control_input=x[None])

    assert leaves_close(m_new, m_ref)
    assert all(allclose(s_new[k][0], s_ref[k]) for k in s_ref)
    assert allclose(eb_new[0], eb_ref) and allclose(ea_new[0], ea_ref)
    assert allclose(ya_new["y"][0], ya_ref["y"] if not isinstance(ya_ref, dict) else ya_ref["y"])


# ---- time axis ------------------------------------------------------------- #

def test_eval_run_equals_sequential_eval_steps():
    model = GraphModel(mixed_config(), key=KEY)
    obs = mixed_sequence(T=4, B=2)
    run = make_eval_run(ACT, n_infer_steps=5, run_length=4)
    states_final, yb, ya, eb, ea, trace = run(model, obs)

    step = make_eval_step(ACT, n_infer_steps=5)
    states_prev = zero_states(model, 2)
    for t in range(4):
        states_prev, yb_t, ya_t, eb_t, ea_t, _ = step(model, {"y": obs["y"][t]}, states_prev=states_prev)
        assert allclose(ea[t], ea_t) and allclose(ya["y"][t], ya_t["y"])
    assert all(allclose(states_final[k], states_prev[k]) for k in states_prev)
    assert ea.shape == (4, 2)


def test_train_run_equals_sequential_train_steps_on_a_mixed_graph():
    model = GraphModel(mixed_config(), key=KEY)
    obs = mixed_sequence(T=4, B=2)
    optim = optax.sgd(0.02)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    run = make_train_run(optim, ACT, n_infer_steps=5, run_length=4)
    m_run, _, states_run, _, _, _, ea_run, _ = run(model, opt_state, obs)

    step = make_train_step(optim, ACT, n_infer_steps=5)
    m_seq, st, states_prev = model, opt_state, zero_states(model, 2)
    ea_seq = []
    for t in range(4):
        m_seq, st, states_prev, _, _, _, ea_t, _ = step(m_seq, st, {"y": obs["y"][t]}, states_prev=states_prev)
        ea_seq.append(ea_t)

    assert leaves_close(m_run, m_seq)
    assert allclose(ea_run, jnp.stack(ea_seq))
    assert all(allclose(states_run[k], states_prev[k]) for k in states_prev)


def test_states_are_actually_carried_through_time():
    # the leaky control node makes frame t depend on frame t-1: a carried run must
    # differ from running every frame from zero states
    model = GraphModel(mixed_config(), key=KEY)
    obs = mixed_sequence(T=3, B=2)
    run = make_eval_run(ACT, n_infer_steps=5, run_length=3)
    _, _, _, _, e_carried, _ = run(model, obs)
    step = make_eval_step(ACT, n_infer_steps=5)
    e_fresh = jnp.stack([step(model, {"y": obs["y"][t]}, states_prev=zero_states(model, 2))[4] for t in range(3)])
    assert np.allclose(e_carried[0], e_fresh[0]) and not np.allclose(e_carried[2], e_fresh[2])


def test_mixed_graph_learns_over_runs():
    model = GraphModel(mixed_config(), key=KEY)
    obs = mixed_sequence(T=6, B=4)
    optim = optax.adam(1e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    run = make_train_run(optim, optax.sgd(0.1), n_infer_steps=8, run_length=6)
    energies = []
    for _ in range(15):
        model, opt_state, _, _, _, _, e_after, _ = run(model, opt_state, obs)
        energies.append(float(jnp.mean(e_after)))
    assert np.all(np.isfinite(energies)) and energies[-1] < energies[0]
