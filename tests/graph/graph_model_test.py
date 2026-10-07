"""Tests for graph/model.py and graph/builders.py.

Run from the repo root (the directory containing the package that contains
`graph/`), e.g.:   python -m pytest <package>/graph/tests -v

The first section is the regression gate: a CHAIN `GraphModel` must match
`TpchModel` numerically (energy, layerwise energy, gradients, settling) when
given the same weights. The rest cover what is new: branching/isolation,
multi-parent gathering, mixed dense/conv/bidirectional graphs, per-node
settings, checkpointing, and the shared runners.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from pc_nox.core.model_base import MODEL_REGISTRY
from pc_nox.models.tpch.model import TpchModel

from pc_nox.models.graph import GraphModel, GraphModelConfig, NodeSpec
from pc_nox.models.graph.builders import build_nodes, combine, combined_shape, node_key
from pc_nox.models.graph.energies import ENERGY_REGISTRY
from pc_nox.models.graph.nodes.base import NodeBase, NodeInputs, NodePrediction
from pc_nox.models.graph.validation import validate_pairings

KEY = jr.PRNGKey(0)


def allclose(a, b, rtol=1e-5, atol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


# =========================================================================== #
# 1. Regression gate: chain GraphModel == TpchModel
# =========================================================================== #

REG_CASES = {
    "none": {},
    "decay-all": dict(weight_decay=0.1, weight_decay_scope="all"),
    "decay-rec": dict(weight_decay=0.1, weight_decay_scope="rec"),
    "decay-ff+orth-rec": dict(weight_decay=0.1, weight_decay_scope="ff", orthogonal_penalty=0.2, orthogonal_scope="rec"),
    "orth-all": dict(orthogonal_penalty=0.05, orthogonal_scope="all"),
    "activity-l1": dict(activity_decay=0.05, activity_reg_type="l1"),
    "activity-l2": dict(activity_decay=0.05, activity_reg_type="l2"),
}


def make_pair(loss="mse", **reg):
    """A TpchModel and a chain GraphModel with identical structure, leaky
    nodes included, and (after copy_weights) identical weights."""
    tpch = TpchModel(8, (6, 5), 4, key=jr.PRNGKey(1), input_size=3, loss=loss,
                     control_alpha=0.5, hidden_alphas=(0.3, None), **reg)
    config = GraphModelConfig(
        node_specs=(
            NodeSpec("core", "tpch_control", 8, alpha=0.5),
            NodeSpec("h1", "tpch_hidden", 6, parents=("core",), alpha=0.3),
            NodeSpec("h2", "tpch_hidden", 5, parents=("h1",)),
            NodeSpec("y", "tpch_obs", 4, parents=("h2",), clamped=True, energy="ce" if loss == "ce" else None),
        ),
        input_size=3, **reg,
    )
    graph = copy_weights(GraphModel(config, key=jr.PRNGKey(2)), tpch)
    return tpch, graph


def weight_paths(graph_or_grads, tpch_or_grads):
    """(graph weight, tpch weight) pairs, works for models and gradient pytrees."""
    g, t = graph_or_grads.nodes, tpch_or_grads
    pairs = [
        (g["core"].layer.W_rec.weight, t.control_layer.W_rec.weight),
        (g["core"].layer.W_in.weight, t.control_layer.W_in.weight),
        (g["y"].layer.W_parent.weight, t.observation_layer.W_parent.weight),
    ]
    for name, layer in zip(("h1", "h2"), t.hidden_layers):
        gl = g[name].layer
        pairs += [(gl.W_rec.weight, layer.W_rec.weight),
                  (gl.W_parent_prev.weight, layer.W_parent_prev.weight),
                  (gl.W_parent_curr.weight, layer.W_parent_curr.weight)]
    return pairs


def copy_weights(graph, tpch):
    getters = [
        lambda m: m.nodes["core"].layer.W_rec.weight, lambda m: m.nodes["core"].layer.W_in.weight,
        lambda m: m.nodes["y"].layer.W_parent.weight,
        lambda m: m.nodes["h1"].layer.W_rec.weight, lambda m: m.nodes["h1"].layer.W_parent_prev.weight,
        lambda m: m.nodes["h1"].layer.W_parent_curr.weight,
        lambda m: m.nodes["h2"].layer.W_rec.weight, lambda m: m.nodes["h2"].layer.W_parent_prev.weight,
        lambda m: m.nodes["h2"].layer.W_parent_curr.weight,
    ]
    for get, (_, src) in zip(getters, weight_paths(graph, tpch)):
        graph = eqx.tree_at(get, graph, src)
    return graph


def chain_inputs(one_hot=False):
    k = jr.split(KEY, 5)
    sizes = (8, 6, 5)
    prev_list = [jr.normal(kk, (s,)) for kk, s in zip(k, sizes)]
    curr_list = [jr.normal(kk, (s,)) * 0.5 for kk, s in zip(jr.split(k[4], 3), sizes)]
    y = jax.nn.one_hot(2, 4) if one_hot else jr.normal(k[0], (4,))
    x = jr.normal(k[1], (3,))
    to_dict = lambda lst: dict(zip(("core", "h1", "h2"), lst))
    return prev_list, curr_list, to_dict(prev_list), to_dict(curr_list), y, x


def test_chain_init_activities_matches_tpch():
    tpch, graph = make_pair()
    prev_l, _, prev_d, _, y, x = chain_inputs()
    expected = tpch.init_activities(prev_l, x, y)
    got = graph.init_activities(prev_d, x, {"y": y})
    for name, e in zip(("core", "h1", "h2"), expected):
        assert allclose(got[name], e)


def test_chain_predict_matches_tpch():
    tpch, graph = make_pair()
    prev_l, curr_l, prev_d, curr_d, y, x = chain_inputs()
    preds_t, y_hat_t = tpch.predict(prev_l, curr_l, x, y)
    preds_g, y_hat_g = graph.predict(prev_d, curr_d, x, {"y": y})
    assert allclose(y_hat_g["y"], y_hat_t)
    for name, e in zip(("core", "h1", "h2"), preds_t):
        assert allclose(preds_g[name].predictions["td"], e)


@pytest.mark.parametrize("reg", REG_CASES.values(), ids=REG_CASES.keys())
def test_chain_energy_total_and_layerwise_match_tpch(reg):
    tpch, graph = make_pair(**reg)
    prev_l, curr_l, prev_d, curr_d, y, x = chain_inputs()
    assert allclose(graph.energy_fn(prev_d, curr_d, {"y": y}, x), tpch.energy_fn(prev_l, curr_l, y, x))
    lw_g = graph.energy_fn(prev_d, curr_d, {"y": y}, x, return_layerwise=True)
    lw_t = tpch.energy_fn(prev_l, curr_l, y, x, return_layerwise=True)
    assert allclose(lw_g, lw_t)
    assert allclose(jnp.sum(lw_g), graph.energy_fn(prev_d, curr_d, {"y": y}, x))   # layerwise sums to total


def test_chain_cross_entropy_matches_tpch():
    tpch, graph = make_pair(loss="ce")
    prev_l, curr_l, prev_d, curr_d, y, x = chain_inputs(one_hot=True)
    assert allclose(graph.energy_fn(prev_d, curr_d, {"y": y}, x), tpch.energy_fn(prev_l, curr_l, y, x))


@pytest.mark.parametrize("reg", [REG_CASES["none"], REG_CASES["decay-ff+orth-rec"]], ids=["none", "decay+orth"])
def test_chain_param_grad_matches_tpch(reg):
    tpch, graph = make_pair(**reg)
    prev_l, curr_l, prev_d, curr_d, y, x = chain_inputs()
    g_t = tpch.param_grad(prev_l, curr_l, y, x)
    g_g = graph.param_grad(prev_d, curr_d, {"y": y}, x)
    for got, expected in weight_paths(g_g, g_t):
        assert allclose(got, expected, rtol=1e-4, atol=1e-6)


def test_chain_settle_scan_matches_tpch():
    tpch, graph = make_pair()
    prev_l, _, prev_d, _, y, x = chain_inputs()
    out_t = tpch.settle_scan(optax.sgd(0.1), prev_l, y, x, n_steps=15)
    out_g = graph.settle_scan(optax.sgd(0.1), prev_d, {"y": y}, x, n_steps=15)
    for name, e in zip(("core", "h1", "h2"), out_t):
        assert allclose(out_g[name], e, rtol=1e-4, atol=1e-5)


def test_chain_settle_scan_layerwise_trace_matches_tpch():
    tpch, graph = make_pair(weight_decay=0.1)
    prev_l, _, prev_d, _, y, x = chain_inputs()
    _, trace_t = tpch.settle_scan(optax.sgd(0.1), prev_l, y, x, n_steps=8, return_layerwise=True)
    _, trace_g = graph.settle_scan(optax.sgd(0.1), prev_d, {"y": y}, x, n_steps=8, return_layerwise=True)
    assert trace_g.shape == trace_t.shape == (8, 4)
    assert allclose(trace_g, trace_t, rtol=1e-4, atol=1e-5)


def test_chain_settle_diffrax_matches_tpch():
    tpch, graph = make_pair()
    prev_l, _, prev_d, _, y, x = chain_inputs()
    kw = dict(max_t1=2.0, steady_state_tol=None)
    out_t = tpch.settle_diffrax(prev_l, y, x, **kw)
    out_g = graph.settle_diffrax(prev_d, {"y": y}, x, **kw)
    for name, e in zip(("core", "h1", "h2"), out_t):
        assert allclose(out_g[name], e, rtol=1e-3, atol=1e-4)


# =========================================================================== #
# 2. Branching: tails are isolated, the shared parent sums its children
# =========================================================================== #

def tails_config(**kw):
    return GraphModelConfig(node_specs=(
        NodeSpec("core", "tpch_control", 8),
        NodeSpec("a", "tpch_hidden", 6, parents=("core",)),
        NodeSpec("b", "tpch_hidden", 5, parents=("core",)),
        NodeSpec("a_obs", "tpch_obs", 4, parents=("a",), clamped=True),
        NodeSpec("b_obs", "tpch_obs", 3, parents=("b",), clamped=True),
    ), **kw)


def tails_setup(cfg=None):
    cfg = cfg or tails_config()
    model = GraphModel(cfg, key=KEY)
    k = jr.split(jr.PRNGKey(7), 4)
    prev = {n: jr.normal(kk, cfg.spec(n).shape) for kk, n in zip(k, ("core", "a", "b"))}
    curr = {n: jr.normal(kk, cfg.spec(n).shape) * 0.5 for kk, n in zip(jr.split(k[3], 3), ("core", "a", "b"))}
    y = {"a_obs": jr.normal(k[0], (4,)), "b_obs": jr.normal(k[1], (3,))}
    return model, prev, curr, y


def test_tail_state_gradient_ignores_the_other_tails_data():
    model, prev, curr, y = tails_setup()
    grad_states = lambda y_: jax.grad(lambda s: model.energy_fn(prev, s, y_))(curr)
    g1 = grad_states(y)
    g2 = grad_states({**y, "b_obs": y["b_obs"] + 5.0})          # change only the b tail's data
    assert jnp.array_equal(g1["a"], g2["a"])                       # tail a is untouched
    assert not allclose(g1["b"], g2["b"])                          # tail b responds
    # with the other states held fixed, b's data only enters through b's own error term
    assert jnp.array_equal(g1["core"], g2["core"])


def test_shared_parent_receives_error_from_every_tail():
    model, prev, curr, y = tails_setup()
    grad_states = lambda c: jax.grad(lambda s: model.energy_fn(prev, s, y))(c)
    g = grad_states(curr)
    g_b_moved = grad_states({**curr, "b": curr["b"] + 1.0})
    g_a_moved = grad_states({**curr, "a": curr["a"] + 1.0})
    assert not allclose(g["core"], g_b_moved["core"])              # the core hears tail b ...
    assert not allclose(g["core"], g_a_moved["core"])              # ... and tail a
    assert jnp.array_equal(g["a"], g_b_moved["a"])                 # but a's own gradient ignores b's state
    assert jnp.array_equal(g["b"], g_a_moved["b"])


def test_tail_weight_gradient_is_zero_for_the_other_tail():
    model, prev, curr, y = tails_setup()
    idx = model.config.names.index("a_obs")
    grads = eqx.filter_grad(lambda m: m.energy_fn(prev, curr, y, return_layerwise=True)[idx])(model)
    b_tail = [grads.nodes["b"], grads.nodes["b_obs"]]
    assert all(jnp.all(leaf == 0) for leaf in jax.tree_util.tree_leaves(b_tail))
    assert any(jnp.any(leaf != 0) for leaf in jax.tree_util.tree_leaves(grads.nodes["a_obs"]))


def test_two_tails_settle_and_learn():
    model, prev, _, y = tails_setup()
    before = model.energy_fn(prev, model.init_activities(prev, None, y), y)
    states = model.settle_scan(optax.sgd(0.1), prev, y, n_steps=30)
    assert model.energy_fn(prev, states, y) < before
    grads = model.param_grad(prev, states, y)
    assert all(jnp.all(jnp.isfinite(l)) for l in jax.tree_util.tree_leaves(grads))


# =========================================================================== #
# 3. Multi-parent gathering
# =========================================================================== #

def fuse_config(parents=("a", "b")):
    return GraphModelConfig(node_specs=(
        NodeSpec("core", "tpch_control", 8),
        NodeSpec("a", "tpch_hidden", 6, parents=("core",)),
        NodeSpec("b", "tpch_hidden", 5, parents=("core",)),
        NodeSpec("fuse", "tpch_hidden", 7, parents=parents),
        NodeSpec("y", "tpch_obs", 2, parents=("fuse",), clamped=True),
    ))


def test_multi_parent_node_sees_concatenated_parents_in_config_order():
    for parents, expected_order in ((("a", "b"), ("a", "b")), (("b", "a"), ("b", "a"))):
        cfg = fuse_config(parents)
        model = GraphModel(cfg, key=KEY)
        curr = {"core": jnp.ones(8), "a": jnp.full(6, 1.0), "b": jnp.full(5, 2.0), "fuse": jnp.zeros(7)}
        spec = cfg.spec("fuse")
        inp = model._inputs(spec, model.nodes["fuse"], {k: jnp.zeros_like(v) for k, v in curr.items()}, curr, None)
        expected = jnp.concatenate([curr[p] for p in expected_order])
        assert jnp.array_equal(inp.parents_curr, expected)
        assert model.nodes["fuse"].layer.W_parent_curr.weight.shape == (7, 11)


def test_multi_parent_error_reaches_both_parents():
    cfg = fuse_config()
    model = GraphModel(cfg, key=KEY)
    prev = GraphModel.zero_activities(cfg)
    curr = {k: jr.normal(jr.PRNGKey(i), v.shape) for i, (k, v) in enumerate(prev.items())}
    y = {"y": jnp.ones(2)}
    g = jax.grad(lambda s: model.energy_fn(prev, s, y))(curr)
    assert jnp.any(g["a"] != 0) and jnp.any(g["b"] != 0)


# =========================================================================== #
# 4. Mixed dense / conv / bidirectional
# =========================================================================== #

def conv_config():
    return GraphModelConfig(node_specs=(
        NodeSpec("core", "tpch_control", 16),
        NodeSpec("feat", "pcn_conv", (2, 4, 4), parents=("core",), bidirectional=True,
                 options={"bu_kernel_size": 4, "bu_stride": 2, "bu_padding": 1}),
        NodeSpec("img", "pcn_conv", (1, 8, 8), parents=("feat",), clamped=True, act_fn="identity",
                 options={"td_output_padding": 1}),
        NodeSpec("vec", "tpch_hidden", 8, parents=("core",)),
        NodeSpec("vec_obs", "tpch_obs", 3, parents=("vec",), clamped=True),
    ))


def conv_setup():
    cfg = conv_config()
    model = GraphModel(cfg, key=KEY)
    prev = GraphModel.zero_activities(cfg)
    y = {"img": jr.normal(jr.PRNGKey(3), (1, 8, 8)), "vec_obs": jr.normal(jr.PRNGKey(4), (3,))}
    return cfg, model, prev, y


def test_dense_core_feeds_conv_tail_and_dense_tail():
    cfg, model, prev, y = conv_setup()
    states = model.init_activities(prev, None, y)
    assert states["feat"].shape == (2, 4, 4) and states["vec"].shape == (8,)
    preds, y_hat = model.predict(prev, states, None, y)
    assert y_hat["img"].shape == (1, 8, 8) and y_hat["vec_obs"].shape == (3,)
    assert set(preds["feat"].predictions) == {"td", "bu"}          # bidirectional node scores both paths


def test_conv_graph_settles_and_learns():
    cfg, model, prev, y = conv_setup()
    before = model.energy_fn(prev, model.init_activities(prev, None, y), y)
    states = model.settle_scan(optax.sgd(0.05), prev, y, n_steps=30)
    assert model.energy_fn(prev, states, y) < before
    grads = model.param_grad(prev, states, y)
    assert all(jnp.all(jnp.isfinite(l)) for l in jax.tree_util.tree_leaves(grads))


def test_bottom_up_path_adds_a_second_energy_term():
    cfg, model, prev, y = conv_setup()
    states = model.init_activities(prev, None, y)
    preds, _ = model.predict(prev, states, None, y)
    p = preds["feat"].predictions
    expected = sum(0.5 * jnp.sum((states["feat"] - p[k]) ** 2) for k in ("td", "bu"))
    idx = cfg.names.index("feat")
    assert allclose(model.energy_fn(prev, states, y, return_layerwise=True)[idx], expected)


# =========================================================================== #
# 5. Per-node settings
# =========================================================================== #

def test_precision_scales_a_nodes_energy():
    base_cfg = tails_config()
    scaled_cfg = GraphModelConfig(node_specs=tuple(
        s if s.name != "b_obs" else NodeSpec("b_obs", "tpch_obs", 3, parents=("b",), clamped=True, precision=4.0)
        for s in base_cfg.node_specs))
    m1, prev, curr, y = tails_setup(base_cfg)
    m2 = GraphModel(scaled_cfg, key=KEY)
    i = base_cfg.names.index("b_obs")
    e1 = m1.energy_fn(prev, curr, y, return_layerwise=True)
    e2 = m2.energy_fn(prev, curr, y, return_layerwise=True)
    assert allclose(e2[i], 4.0 * e1[i])
    assert allclose(jnp.delete(e2, i), jnp.delete(e1, i))


def test_per_node_weight_decay_override_only_charges_that_node():
    specs = list(tails_config().node_specs)
    specs[1] = NodeSpec("a", "tpch_hidden", 6, parents=("core",), weight_decay=0.5)
    cfg = GraphModelConfig(node_specs=tuple(specs))                     # model-wide decay stays 0
    model = GraphModel(cfg, key=KEY)
    reg = model._weight_reg_by_node()
    assert reg[cfg.names.index("a")] > 0
    assert jnp.sum(jnp.delete(reg, cfg.names.index("a"))) == 0


def test_conflicting_regularisers_warn():
    with pytest.warns(UserWarning, match="weight_decay .* >= 2 \\* orthogonal_penalty"):
        GraphModel(tails_config(weight_decay=1.0, orthogonal_penalty=0.1,
                                weight_decay_scope="all", orthogonal_scope="rec"), key=KEY)


def test_no_warning_for_compatible_regularisers():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GraphModel(tails_config(weight_decay=0.1, weight_decay_scope="ff",
                                orthogonal_penalty=0.2, orthogonal_scope="rec"), key=KEY)


# =========================================================================== #
# 6. ModelBase hooks, checkpointing, jit, runners
# =========================================================================== #

def test_model_is_registered_and_hooks_agree_with_the_config():
    cfg = tails_config()
    assert MODEL_REGISTRY["graph"] is GraphModel
    assert GraphModel.layer_labels(cfg) == list(cfg.names)
    zero = GraphModel.zero_activities(cfg)
    assert set(zero) == {"core", "a", "b"}                       # clamped nodes are not states
    model, prev, curr, y = tails_setup(cfg)
    assert model.energy_fn(prev, curr, y, return_layerwise=True).shape == (len(cfg.names),)


def test_checkpoint_round_trip(tmp_path):
    cfg = conv_config()
    model = GraphModel(cfg, key=jr.PRNGKey(11))
    states = {k: jr.normal(jr.PRNGKey(i), v.shape) for i, (k, v) in enumerate(GraphModel.zero_activities(cfg).items())}
    path = model.save_checkpoint(path=tmp_path / "ckpt", activities=states, metadata={"epoch": 3})
    # no key= passed: ModelBase.load_checkpoint does `key or jr.PRNGKey(0)`, which raises on a JAX key array.
    # The default init key (0) differs from the saved model's (11), so weights must come from disk.
    loaded = GraphModel.load_checkpoint(path)
    assert isinstance(loaded.model, GraphModel) and loaded.model.config == cfg
    assert loaded.metadata == {"epoch": 3}
    for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)),
                    jax.tree_util.tree_leaves(eqx.filter(loaded.model, eqx.is_array))):
        assert jnp.array_equal(a, b)
    assert all(jnp.array_equal(states[k], loaded.activities[k]) for k in states)


def test_jit_compiles_energy_and_settling():
    model, prev, curr, y = tails_setup()
    energy = eqx.filter_jit(lambda m: m.energy_fn(prev, curr, y))(model)
    assert allclose(energy, model.energy_fn(prev, curr, y))
    settled = eqx.filter_jit(lambda m: m.settle_scan(optax.sgd(0.1), prev, y, n_steps=5))(model)
    assert set(settled) == {"core", "a", "b"}


def test_shared_runner_train_step_reduces_energy():
    runners = pytest.importorskip("pc_nox.engine.runners_temporal",
                                  reason="adjust this import to where runners_temporal.py lives")
    cfg = tails_config()
    model = GraphModel(cfg, key=KEY)
    param_optim = optax.adam(1e-2)
    opt_state = param_optim.init(eqx.filter(model, eqx.is_array))
    train_step = runners.make_train_step(param_optim, optax.sgd(0.1), n_infer_steps=10)
    y = {"a_obs": jnp.ones(4), "b_obs": -jnp.ones(3)}
    energies = []
    for _ in range(25):
        prev = GraphModel.zero_activities(cfg)
        model, opt_state, _, _, _, _, e_after, _ = train_step(model, opt_state, prev, y)
        energies.append(float(e_after))
    assert np.all(np.isfinite(energies)) and energies[-1] < energies[0]


# =========================================================================== #
# 7. Builders and validation
# =========================================================================== #

@pytest.mark.parametrize("shapes, layout, expected", [
    ([(3, 4), (5,)], "flat", (17,)),
    ([(2, 4, 4)], "spatial", (2, 4, 4)),
    ([(3,), (4,)], "spatial", (7,)),
    ([(2, 4, 4), (3, 4, 4)], "spatial", (5, 4, 4)),
])
def test_combined_shape_agrees_with_runtime_combine(shapes, layout, expected):
    assert combined_shape(shapes, layout) == expected
    assert combine([jnp.zeros(s) for s in shapes], layout).shape == expected


@pytest.mark.parametrize("shapes", [[(2, 4, 4), (3, 8, 8)], [(3,), (2, 4, 4)]])
def test_spatial_combine_rejects_mismatched_inputs(shapes):
    with pytest.raises(ValueError, match="cannot combine"):
        combined_shape(shapes, "spatial")


def test_adding_a_tail_does_not_change_other_nodes_initial_weights():
    small = GraphModelConfig(node_specs=tails_config().node_specs[:2] + (
        NodeSpec("a_obs", "tpch_obs", 4, parents=("a",), clamped=True),))
    big = tails_config()
    n_small, n_big = build_nodes(small, KEY), build_nodes(big, KEY)
    for name in ("core", "a", "a_obs"):
        for x, z in zip(jax.tree_util.tree_leaves(eqx.filter(n_small[name], eqx.is_array)),
                        jax.tree_util.tree_leaves(eqx.filter(n_big[name], eqx.is_array))):
            assert jnp.array_equal(x, z)
    assert not jnp.array_equal(node_key(KEY, "a"), node_key(KEY, "b"))


def test_unknown_kind_lists_registered_kinds():
    cfg = GraphModelConfig(node_specs=(NodeSpec("n", "no_such_kind", 4),))
    with pytest.raises(KeyError, match="registered kinds"):
        GraphModel(cfg, key=KEY)


def test_builder_errors_name_the_node():
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("core", "tpch_control", 4),
        NodeSpec("oops", "tpch_control", 4, parents=("core",)),
    ))
    with pytest.raises(ValueError, match="node 'oops'.*no parents"):
        GraphModel(cfg, key=KEY)


def test_unknown_act_fn_is_reported():
    cfg = GraphModelConfig(node_specs=(NodeSpec("core", "tpch_control", 4, act_fn="nope"),))
    with pytest.raises(KeyError, match="ACT_FN_REGISTRY"):
        GraphModel(cfg, key=KEY)


def test_recurrent_node_under_a_clamped_parent_is_rejected():
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("core", "tpch_control", 4),
        NodeSpec("obs", "tpch_obs", 3, parents=("core",), clamped=True),
        NodeSpec("h", "tpch_hidden", 2, parents=("obs",)),
    ))
    with pytest.raises(ValueError, match="previous states but parent 'obs' is clamped"):
        GraphModel(cfg, key=KEY)


class _Stub(NodeBase):
    shape: tuple = eqx.field(static=True)
    wants_control: bool = eqx.field(static=True, default=False)

    @property
    def state_shape(self):
        return self.shape

    @property
    def needs_parents_curr(self):
        return False

    @property
    def needs_control_input(self):
        return self.wants_control

    def predict(self, inp):
        return NodePrediction({"td": jnp.zeros(self.shape)})

    def weight_entries(self):
        return []


def test_input_size_without_a_consumer_is_rejected():
    cfg = GraphModelConfig(node_specs=(NodeSpec("n", "stub", (4,)),), input_size=2)
    with pytest.raises(ValueError, match="no node consumes the control input"):
        validate_pairings(cfg, {"n": _Stub((4,))}, ENERGY_REGISTRY)


def test_node_needing_control_input_without_input_size_is_rejected():
    cfg = GraphModelConfig(node_specs=(NodeSpec("n", "stub", (4,)),), input_size=0)
    with pytest.raises(ValueError, match="needs the control input but config.input_size is 0"):
        validate_pairings(cfg, {"n": _Stub((4,), wants_control=True)}, ENERGY_REGISTRY)


def test_node_ignoring_its_listed_parents_is_rejected():
    cfg = GraphModelConfig(node_specs=(NodeSpec("p", "stub", (4,)), NodeSpec("n", "stub", (4,), parents=("p",))))
    with pytest.raises(ValueError, match="this node type ignores them"):
        validate_pairings(cfg, {"p": _Stub((4,)), "n": _Stub((4,))}, ENERGY_REGISTRY)
