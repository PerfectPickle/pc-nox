"""Bidirectional-PC support: per-path precisions, the model-wide `bidirectional` default, bottom-up-only data
roots, released (free-when-omitted) clamped nodes, and the two initialisation sweeps.

Reference energy = the paper's Eq. 3 written out by hand:
    E = sum_{l=1}^{L-1} a_gen/2 ||x_l - W_{l+1} f(x_{l+1})||^2  +  sum_{l=2}^{L} a_disc/2 ||x_l - V_{l-1} f(x_{l-1})||^2
(with this library's convention that the activation is applied to the OUTPUT of each projection, as in `PcnLayer`).
In graph terms: image = root, so the paper's discriminative V is `td` and the generative W is `bu`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from pc_nox.models.graph import GraphModel, GraphModelConfig, NodeSpec
from pc_nox.models.graph.nodes.conv import PcnConvNode
from pc_nox.models.graph.nodes.pcn import PcnNode
from pc_nox.models.graph.runners import make_eval_step, make_train_step

KEY = jr.PRNGKey(0)


def allclose(a, b, rtol=1e-5, atol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


# ---- a small dense bPC graph: x (data root, bu only) <-> h <-> y (label, td only) ----------------------------- #

def dense_config(td=1.0, bu=1e-2, **kw):
    return GraphModelConfig(node_specs=(
        NodeSpec("x", "input", (6,), clamped=True, release=True, act_fn="identity"),
        NodeSpec("h", "pcn", 5, parents=("x",), act_fn="tanh"),
        NodeSpec("y", "pcn", 3, parents=("h",), act_fn="identity", clamped=True, release=True)),
        bidirectional=True, td_precision=td, bu_precision=bu, **kw)


def ref_energy(m, x, h, y, td, bu):
    n, sq = m.nodes, (lambda a, b: 0.5 * jnp.sum((a - b) ** 2))
    gen = sq(x, n["x"].bu_op(h)) + sq(h, jnp.tanh(n["h"].bu_op(y)))               # W-path:  x <- h,  h <- y
    disc = sq(h, jnp.tanh(n["h"].td_op(x))) + sq(y, n["y"].td_op(h))              # V-path:  h <- x,  y <- h
    return bu * gen + td * disc


def example(m, seed=1):
    k = jr.split(jr.PRNGKey(seed), 3)
    return jr.normal(k[0], (6,)), jr.normal(k[1], (5,)) * 0.5, jr.normal(k[2], (3,))


# =========================================================================== #
# Structure: model-wide bidirectionality, data roots with bu weights, validation
# =========================================================================== #

def test_model_wide_bidirectional_gives_roots_bu_only_leaves_td_only_and_hidden_both():
    cfg = dense_config()
    assert [s.bidirectional for s in cfg.node_specs] == [True, True, False]       # leaf has no children -> no bu path
    m = GraphModel(cfg, key=KEY)
    assert m.nodes["x"].prediction_keys == ("bu",)                                 # root: generative term only
    assert m.nodes["h"].prediction_keys == ("td", "bu")
    assert m.nodes["y"].prediction_keys == ("td",)                                 # leaf: discriminative term only


def test_a_node_can_opt_out_of_the_model_wide_default_and_the_resolution_survives_json():
    import json
    from dataclasses import asdict
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("x", "input", (6,), clamped=True),
        NodeSpec("h", "pcn", 5, parents=("x",), bidirectional=False),
        NodeSpec("y", "pcn", 3, parents=("h",), clamped=True)), bidirectional=True)
    assert [s.bidirectional for s in cfg.node_specs] == [True, False, False]
    assert GraphModelConfig(**json.loads(json.dumps(asdict(cfg)))) == cfg


def test_a_node_kind_that_cannot_be_bidirectional_fails_naming_the_node():
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("core", "tpch_control", 4),
        NodeSpec("h", "tpch_hidden", 3, parents=("core",)),
        NodeSpec("y", "pcn", 2, parents=("h",), clamped=True)), bidirectional=True)
    with pytest.raises(ValueError, match="node 'core'.*no bottom-up path"):
        GraphModel(cfg, key=KEY)


def test_bidirectional_input_builds_a_conv_or_dense_data_root_by_shape():
    conv = GraphModel(GraphModelConfig(node_specs=(
        NodeSpec("img", "input", (2, 8, 8), clamped=True, bidirectional=True, options={"bu_output_padding": 1}),
        NodeSpec("c1", "pcn_conv", (4, 4, 4), parents=("img",)))), key=KEY)
    assert isinstance(conv.nodes["img"], PcnConvNode) and conv.nodes["img"].prediction_keys == ("bu",)
    assert not conv.nodes["img"].needs_parents_curr and conv.nodes["img"].td_op is None
    assert isinstance(GraphModel(dense_config(), key=KEY).nodes["x"], PcnNode)


def test_a_parentless_node_with_nothing_to_predict_it_is_rejected():
    with pytest.raises(ValueError, match="needs bidirectional=True"):
        PcnConvNode(parent_shape=None, own_shape=(2, 4, 4), key=KEY)
    with pytest.raises(ValueError, match="needs bidirectional=True"):
        PcnNode(own_size=4, has_prior=False, key=KEY)
    with pytest.raises(ValueError, match="needs at least one child"):
        GraphModel(GraphModelConfig(node_specs=(NodeSpec("img", "input", (6,), clamped=True, bidirectional=True),)), key=KEY)


def test_release_validation():
    with pytest.raises(ValueError, match="release only applies to a clamped node"):
        NodeSpec("a", "pcn", 3, release=True)
    ce_label = GraphModelConfig(node_specs=(
        NodeSpec("x", "input", (6,), clamped=True), NodeSpec("y", "pcn", 3, parents=("x",), energy="ce", clamped=True, release=True)))
    with pytest.raises(ValueError, match="loss against data"):
        GraphModel(ce_label, key=KEY)
    weightless = GraphModelConfig(node_specs=(
        NodeSpec("x", "input", (6,), clamped=True, release=True), NodeSpec("h", "pcn", 3, parents=("x",))))
    with pytest.raises(ValueError, match="predicts nothing in either direction"):
        GraphModel(weightless, key=KEY)


# =========================================================================== #
# Per-path precisions
# =========================================================================== #

@pytest.mark.parametrize("td, bu", [(1.0, 1.0), (1.0, 1e-2), (0.5, 3.0), (1.0, 0.0), (0.0, 1.0)])
def test_energy_matches_the_papers_eq3_with_separate_alphas(td, bu):
    m = GraphModel(dense_config(td, bu), key=KEY)
    x, h, y = example(m)
    got = m.energy_fn({}, {"h": h}, {"x": x, "y": y})
    assert allclose(got, ref_energy(m, x, h, y, td, bu))


def test_the_alphas_scale_each_direction_separately_and_a_node_override_wins():
    base = GraphModel(dense_config(1.0, 1.0), key=KEY)
    x, h, y = example(base)
    obs, st = {"x": x, "y": y}, {"h": h}
    lw = lambda m: np.asarray(m.energy_fn({}, st, obs, return_layerwise=True))
    only_bu_down = lw(GraphModel(dense_config(1.0, 0.1), key=KEY))
    assert allclose(only_bu_down[0], 0.1 * lw(base)[0])                           # x has only a bu term
    assert allclose(only_bu_down[2], lw(base)[2])                                 # y has only a td term: untouched
    over = GraphModelConfig(node_specs=tuple(
        NodeSpec("h", "pcn", 5, parents=("x",), act_fn="tanh", bu_precision=0.0) if s.name == "h" else s
        for s in dense_config(1.0, 1.0).node_specs), bidirectional=True)
    m = GraphModel(over, key=KEY)
    h_ref = 0.5 * jnp.sum((h - jnp.tanh(m.nodes["h"].td_op(x))) ** 2)             # h's bu term switched off by the override
    assert allclose(np.asarray(m.energy_fn({}, st, obs, return_layerwise=True))[1], h_ref)


def test_negative_precisions_are_rejected():
    with pytest.raises(ValueError, match="bu_precision must be >= 0"):
        NodeSpec("a", "pcn", 3, bu_precision=-1.0)
    with pytest.raises(ValueError, match="td_precision"):
        dense_config(td=-1.0)


# =========================================================================== #
# Release + the two initialisation sweeps
# =========================================================================== #

def test_an_omitted_released_node_becomes_an_inferred_state_and_an_observed_one_does_not():
    m = GraphModel(dense_config(), key=KEY)
    x, h, y = example(m)
    assert set(m.init_activities({}, None, {"x": x, "y": y})) == {"h"}
    assert set(m.init_activities({}, None, {"x": x})) == {"h", "y"}               # label released (classification)
    assert set(m.init_activities({}, None, {"y": y})) == {"h", "x"}               # image released (generation)


def test_the_td_sweep_is_the_papers_bottom_up_amortised_initialisation():
    m = GraphModel(dense_config(), key=KEY)
    x, *_ = example(m)
    s = m.init_activities({}, None, {"x": x}, direction="td")
    h0 = jnp.tanh(m.nodes["h"].td_op(x))
    assert allclose(s["h"], h0) and allclose(s["y"], m.nodes["y"].td_op(h0))
    assert allclose(m.init_activities({}, None, {"x": x})["y"], s["y"])          # "td" is the config default


def test_the_bu_sweep_is_the_generative_initialisation_from_a_clamped_label():
    m = GraphModel(dense_config(), key=KEY)
    *_, y = example(m)
    s = m.init_activities({}, None, {"y": y}, direction="bu")
    h0 = jnp.tanh(m.nodes["h"].bu_op(y))
    assert allclose(s["h"], h0) and allclose(s["x"], m.nodes["x"].bu_op(h0))


def test_a_node_without_a_prediction_in_the_sweep_direction_starts_at_zero():
    m = GraphModel(dense_config(), key=KEY)
    s = m.init_activities({}, None, {}, direction="bu")                            # nothing observed: leaf y has no bu path
    assert jnp.all(s["y"] == 0) and set(s) == {"x", "h", "y"}


def test_init_direction_default_comes_from_the_config_and_bad_values_are_rejected():
    m = GraphModel(dense_config(init_direction="bu"), key=KEY)
    *_, y = example(m)
    assert allclose(m.init_activities({}, None, {"y": y})["h"], jnp.tanh(m.nodes["h"].bu_op(y)))
    with pytest.raises(ValueError, match="init_direction"):
        dense_config(init_direction="sideways")
    with pytest.raises(ValueError, match="direction must be"):
        m.init_activities({}, None, {"y": y}, direction="sideways")


def test_a_bu_sweep_needs_the_children_it_reads():
    # y is clamped but NOT released: if it is also missing, h has no child to be predicted from
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("x", "input", (6,), clamped=True, release=True, act_fn="identity"),
        NodeSpec("h", "pcn", 5, parents=("x",)),
        NodeSpec("y", "pcn", 3, parents=("h",), act_fn="identity", clamped=True)), bidirectional=True)
    m = GraphModel(cfg, key=KEY)
    with pytest.raises(KeyError, match="child 'y'"):
        m.init_activities({}, None, {}, direction="bu")


def test_a_released_label_is_scored_and_settling_lowers_the_energy():
    m = GraphModel(dense_config(), key=KEY)
    x, *_ = example(m)
    obs = {"x": x}
    s0 = m.init_activities({}, None, obs)
    e0 = m.energy_fn({}, s0, obs)
    s1 = m.settle_scan(optax.sgd(0.05), {}, obs, n_steps=30)
    assert set(s1) == {"h", "y"} and m.energy_fn({}, s1, obs) <= e0 + 1e-6


def test_conv_data_root_generates_an_image_shaped_state_from_a_label():
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("img", "input", (2, 8, 8), clamped=True, release=True, options={"bu_output_padding": 1}),
        NodeSpec("c1", "pcn_conv", (4, 4, 4), parents=("img",), options={"bu_output_padding": 1}),
        NodeSpec("c2", "pcn_conv", (6, 2, 2), parents=("c1",)),
        NodeSpec("lab", "pcn", 3, parents=("c2",), act_fn="identity", clamped=True, release=True)),
        bidirectional=True, bu_precision=1e-2)
    m = GraphModel(cfg, key=KEY)
    s = m.settle_scan(optax.sgd(0.05), {}, {"lab": jax.nn.one_hot(1, 3)}, n_steps=5, init_direction="bu")
    assert s["img"].shape == (2, 8, 8) and set(s) == {"img", "c1", "c2"}
    assert np.isfinite(np.asarray(s["img"])).all()


# =========================================================================== #
# End to end: train a dense bPC, classify with a FREE label, generate from a label
# =========================================================================== #

def bars(n_per_class=24, size=16):
    rng = np.random.default_rng(0)
    imgs, labels = [], []
    for c in range(3):
        for _ in range(n_per_class):
            img = rng.normal(0.0, 0.1, (size, size)); r0 = 2 + 5 * c; img[r0:r0 + 3, 2:size - 2] += 1.0
            imgs.append(img.reshape(-1)); labels.append(c)
    p = rng.permutation(len(labels))
    return jnp.array(np.stack(imgs)[p], dtype=jnp.float32), jnp.array(labels)[p]


def test_a_dense_bpc_classifies_with_a_free_label_and_generates_class_images():
    X, L = bars(); Y = jax.nn.one_hot(L, 3)
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("x", "input", (256,), clamped=True, release=True, act_fn="identity"),
        NodeSpec("h", "pcn", 48, parents=("x",), act_fn="tanh"),
        NodeSpec("y", "pcn", 3, parents=("h",), act_fn="identity", clamped=True, release=True)),
        bidirectional=True, bu_precision=1e-2)
    model = GraphModel(cfg, key=KEY)
    p_opt = optax.adam(3e-3); a_opt = optax.sgd(0.1)
    state = p_opt.init(eqx.filter(model, eqx.is_array))
    train = make_train_step(p_opt, a_opt, n_infer_steps=15)
    ev = make_eval_step(a_opt, n_infer_steps=30)
    for _ in range(40):
        for i in range(0, len(X), 24):
            model, state, *_ = train(model, state, {"x": X[i:i + 24], "y": Y[i:i + 24]})

    states, *_ = ev(model, {"x": X})                                               # label omitted -> released
    assert float(jnp.mean(jnp.argmax(states["y"], -1) == L)) > 0.9

    gen, *_ = ev(model, {"y": jnp.eye(3)}, init_direction="bu")                    # image omitted, label clamped
    means = jnp.stack([X[L == c].mean(0) for c in range(3)])
    corr = lambda a, b: float(jnp.corrcoef(a, b)[0, 1])
    for c in range(3):
        own = corr(gen["x"][c], means[c])
        others = max(corr(gen["x"][c], means[o]) for o in range(3) if o != c)
        assert own > others, f"class {c}: generated image resembles another class more ({own:.2f} <= {others:.2f})"


def test_a_conv_bpc_classifies_with_a_free_label_and_generates_class_images():
    """The conv generative path (a bottom-up-only conv data root fed by transposed convs) learns to generate."""
    rng = np.random.default_rng(0)
    imgs, labels = [], []
    for c in range(3):
        for _ in range(24):
            img = rng.normal(0.0, 0.1, (1, 16, 16)); r0 = 2 + 5 * c; img[0, r0:r0 + 3, 2:14] += 1.0
            imgs.append(img); labels.append(c)
    p = rng.permutation(len(labels))
    X, L = jnp.array(np.stack(imgs)[p], dtype=jnp.float32), jnp.array(labels)[p]
    Y = jax.nn.one_hot(L, 3)
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("image", "input", (1, 16, 16), clamped=True, release=True, act_fn="identity", options={"bu_output_padding": 1}),
        NodeSpec("c1", "pcn_conv", (4, 8, 8), parents=("image",), act_fn="tanh", options={"bu_output_padding": 1}),
        NodeSpec("c2", "pcn_conv", (6, 4, 4), parents=("c1",), act_fn="tanh"),
        NodeSpec("fc", "pcn", 16, parents=("c2",), act_fn="tanh"),
        NodeSpec("label", "pcn", 3, parents=("fc",), act_fn="identity", clamped=True, release=True)),
        bidirectional=True, bu_precision=1e-2)
    model = GraphModel(cfg, key=KEY)
    p_opt, a_opt = optax.adam(3e-3), optax.sgd(0.1)
    state = p_opt.init(eqx.filter(model, eqx.is_array))
    train, ev = make_train_step(p_opt, a_opt, 15), make_eval_step(a_opt, 40)
    for _ in range(40):
        for i in range(0, len(X), 24):
            model, state, *_ = train(model, state, {"image": X[i:i + 24], "label": Y[i:i + 24]})

    states, *_ = ev(model, {"image": X})
    assert float(jnp.mean(jnp.argmax(states["label"], -1) == L)) > 0.9
    gen, *_ = ev(model, {"label": jnp.eye(3)}, init_direction="bu")
    means = jnp.stack([X[L == c].mean(0).reshape(-1) for c in range(3)])
    corr = lambda a, b: float(jnp.corrcoef(a.reshape(-1), b)[0, 1])
    for c in range(3):
        own = corr(gen["image"][c], means[c])
        assert own > 0.6 and own > max(corr(gen["image"][c], means[o]) for o in range(3) if o != c) + 0.4
