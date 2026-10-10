"""Is the graph machinery adding anything of its own?

Part 1: a DENSE chain GraphModel must reproduce the user's static `PcnModel` exactly (same weights in, same
energy / layerwise energy / gradients / settled states / trained weights out), including a full train step
through the user's own `runners_static`.

Part 2: a CONV chain GraphModel (uni- and bidirectional) must reproduce a hand-written PC energy that uses only
the nodes' own conv/linear ops and `jax.grad` -- no graph, gather, or energy-registry code.

If these pass, any sensitivity of a conv graph to learning rates is a property of predictive coding on that
network, not of the graph implementation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from pc_nox.models.graph import GraphModel, GraphModelConfig, NodeSpec
from pc_nox.models.graph.runners import make_train_step as graph_train_step

KEY = jr.PRNGKey(0)


def allclose(a, b, rtol=1e-4, atol=1e-5):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


def tree_close(a, b, **kw):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(allclose(x, y, **kw) for x, y in zip(la, lb))


# =========================================================================== #
# Part 1: dense chain == PcnModel
# =========================================================================== #

PcnModel = pytest.importorskip("pc_nox.models.pcn.model", reason="needs the user's PcnModel").PcnModel
SIZES = (20, 16, 12, 5)
NAMES = ("h1", "h2", "y")


def dense_pair(**reg):
    pcn = PcnModel(SIZES, jr.PRNGKey(1), act_fn=["tanh", "tanh", "identity"], loss="ce", **reg)
    config = GraphModelConfig(
        node_specs=(
            NodeSpec("x", "input", (SIZES[0],), clamped=True),
            NodeSpec("h1", "pcn", SIZES[1], parents=("x",), act_fn="tanh"),
            NodeSpec("h2", "pcn", SIZES[2], parents=("h1",), act_fn="tanh"),
            NodeSpec("y", "pcn", SIZES[3], parents=("h2",), act_fn="identity", energy="ce", clamped=True),
        ),
        weight_decay=reg.get("weight_decay", 0.0), weight_decay_scope="all",
        orthogonal_penalty=reg.get("orthogonal_penalty", 0.0), orthogonal_scope="all",
        activity_decay=reg.get("activity_decay", 0.0), activity_reg_type=reg.get("activity_reg_type", "l1"),
    )
    graph = GraphModel(config, key=jr.PRNGKey(2))
    for name, layer in zip(NAMES, pcn.layers):                                  # identical weights AND biases
        graph = eqx.tree_at(lambda g, n=name: g.nodes[n].td_op, graph, layer.W_parrent_curr)
    return pcn, graph


def example(seed=3):
    k = jr.split(jr.PRNGKey(seed), 4)
    x, y = jr.normal(k[0], (SIZES[0],)), jax.nn.one_hot(2, SIZES[3])
    states_list = [jr.normal(k[1], (SIZES[1],)) * 0.5, jr.normal(k[2], (SIZES[2],)) * 0.5]
    return x, y, states_list, dict(zip(("h1", "h2"), states_list))


REGS = {"none": {}, "decay+orth": dict(weight_decay=0.05, orthogonal_penalty=0.1),
        "activity-l1": dict(activity_decay=0.05), "activity-l2": dict(activity_decay=0.05, activity_reg_type="l2")}


@pytest.mark.parametrize("reg", REGS.values(), ids=REGS.keys())
def test_dense_chain_energy_matches_pcn_model(reg):
    pcn, graph = dense_pair(**reg)
    x, y, s_list, s_dict = example()
    assert allclose(graph.energy_fn({}, s_dict, {"x": x, "y": y}), pcn.energy_fn(s_list, x, y))
    assert allclose(graph.energy_fn({}, s_dict, {"x": x}), pcn.energy_fn(s_list, x, None))     # label free


def test_dense_chain_init_layerwise_and_gradients_match():
    pcn, graph = dense_pair()
    x, y, s_list, s_dict = example()
    init_p, init_g = pcn.init_activities(x), graph.init_activities({}, None, {"x": x})
    assert all(allclose(init_g[n], a) for n, a in zip(("h1", "h2"), init_p))
    lw_g = graph.energy_fn({}, s_dict, {"x": x, "y": y}, return_layerwise=True)
    assert allclose(lw_g[1:], pcn.energy_fn(s_list, x, y, return_layerwise=True)) and lw_g[0] == 0

    gp, gg = pcn.param_grad(s_list, x, y), graph.param_grad({}, s_dict, {"x": x, "y": y})
    for name, layer_grads in zip(NAMES, gp.layers):
        assert allclose(gg.nodes[name].td_op.weight, layer_grads.W_parrent_curr.weight)
        assert allclose(gg.nodes[name].td_op.bias, layer_grads.W_parrent_curr.bias)

    sp = jax.grad(lambda s: pcn.energy_fn(s, x, y))(s_list)
    sg = jax.grad(lambda s: graph.energy_fn({}, s, {"x": x, "y": y}))(s_dict)
    assert all(allclose(sg[n], a) for n, a in zip(("h1", "h2"), sp))


@pytest.mark.parametrize("optim, name", [(optax.adam(0.01), "adam"), (optax.sgd(0.2), "sgd")])
def test_dense_chain_settling_matches_pcn_model(optim, name):
    pcn, graph = dense_pair()
    x, y, *_ = example()
    out_p = pcn.settle_scan(optim, x, y, n_steps=15)
    out_g = graph.settle_scan(optim, {}, {"x": x, "y": y}, n_steps=15)
    assert all(allclose(out_g[n], a, rtol=1e-3, atol=1e-4) for n, a in zip(("h1", "h2"), out_p))


def test_dense_chain_trains_identically_through_the_users_static_runner():
    runners_static = pytest.importorskip("pc_nox.engine.runners_static")
    pcn, graph = dense_pair()
    xs = jr.normal(jr.PRNGKey(5), (6, SIZES[0]))
    ys = jax.nn.one_hot(jnp.arange(6) % SIZES[3], SIZES[3])
    param_optim, act_optim = optax.adam(1e-3), optax.adam(0.01)
    step_p = runners_static.make_train_step(param_optim, act_optim, n_infer_steps=10)
    step_g = graph_train_step(param_optim, act_optim, n_infer_steps=10)
    st_p, st_g = param_optim.init(eqx.filter(pcn, eqx.is_array)), param_optim.init(eqx.filter(graph, eqx.is_array))
    for _ in range(3):
        pcn, st_p, *out_p = step_p(pcn, st_p, xs, ys)
        graph, st_g, *out_g = step_g(graph, st_g, {"x": xs, "y": ys})
    for name, layer in zip(NAMES, pcn.layers):
        assert allclose(graph.nodes[name].td_op.weight, layer.W_parrent_curr.weight, rtol=1e-3, atol=1e-4)
        assert allclose(graph.nodes[name].td_op.bias, layer.W_parrent_curr.bias, rtol=1e-3, atol=1e-4)
    assert allclose(out_g[3], out_p[3], rtol=1e-3) and allclose(out_g[4], out_p[4], rtol=1e-3)   # energy before / after


# =========================================================================== #
# Part 2: conv chain == hand-written PC energy
# =========================================================================== #

def conv_config(bidirectional: bool):
    bi = dict(bidirectional=bidirectional)
    return GraphModelConfig(node_specs=(
        NodeSpec("image", "input", (2, 8, 8), clamped=True),
        NodeSpec("c1", "pcn_conv", (4, 4, 4), parents=("image",), act_fn="tanh", options={"bu_output_padding": 1}, **bi),
        NodeSpec("c2", "pcn_conv", (6, 2, 2), parents=("c1",), act_fn="tanh", **bi),
        NodeSpec("fc", "pcn", 10, parents=("c2",), act_fn="tanh"),
        NodeSpec("label", "pcn", 3, parents=("fc",), act_fn="identity", energy="ce", clamped=True),
    ))


def reference_energy(m, s, image, label, bidirectional):
    """PC energy written out by hand from the nodes' own ops. No gather / registry / graph code."""
    n, tanh = m.nodes, jnp.tanh
    sq = lambda a, b: 0.5 * jnp.sum((a - b) ** 2)
    c1, c2, fc = s["c1"], s["c2"], s["fc"]
    e = sq(c1, tanh(n["c1"].td_op(image))) + sq(c2, tanh(n["c2"].td_op(c1))) + sq(fc, tanh(n["fc"].td_op(c2.reshape(-1))))
    if bidirectional:
        e += sq(c1, tanh(n["c1"].bu_op(c2))) + sq(c2, tanh(n["c2"].bu_op(fc).reshape(6, 2, 2)))
    if label is not None:
        e += -jnp.sum(label * jax.nn.log_softmax(n["label"].td_op(fc)))
    return e


def conv_example(model):
    k = jr.split(jr.PRNGKey(7), 4)
    s = {"c1": jr.normal(k[0], (4, 4, 4)) * 0.3, "c2": jr.normal(k[1], (6, 2, 2)) * 0.3, "fc": jr.normal(k[2], (10,)) * 0.3}
    return jr.uniform(k[3], (2, 8, 8)), jax.nn.one_hot(1, 3), s


@pytest.mark.parametrize("bidirectional", [False, True], ids=["uni", "bi"])
def test_conv_chain_energy_and_gradients_match_the_hand_written_reference(bidirectional):
    model = GraphModel(conv_config(bidirectional), key=KEY)
    image, label, s = conv_example(model)
    obs = {"image": image, "label": label}
    assert allclose(model.energy_fn({}, s, obs), reference_energy(model, s, image, label, bidirectional))
    assert allclose(model.energy_fn({}, s, {"image": image}), reference_energy(model, s, image, None, bidirectional))

    gs_ref = jax.grad(lambda s_: reference_energy(model, s_, image, label, bidirectional))(s)
    gs = jax.grad(lambda s_: model.energy_fn({}, s_, obs))(s)
    assert all(allclose(gs[k], gs_ref[k]) for k in s)

    gw_ref = eqx.filter_grad(lambda m: reference_energy(m, s, image, label, bidirectional))(model)
    assert tree_close(model.param_grad({}, s, obs).nodes, gw_ref.nodes)


@pytest.mark.parametrize("bidirectional", [False, True], ids=["uni", "bi"])
def test_conv_chain_settling_matches_a_hand_rolled_inference_loop(bidirectional):
    model = GraphModel(conv_config(bidirectional), key=KEY)
    image, label, _ = conv_example(model)
    n, tanh = model.nodes, jnp.tanh
    c1 = tanh(n["c1"].td_op(image)); c2 = tanh(n["c2"].td_op(c1)); fc = tanh(n["fc"].td_op(c2.reshape(-1)))
    s = {"c1": c1, "c2": c2, "fc": fc}                                         # the feedforward starting point
    opt = optax.adam(0.01); st = opt.init(s)
    for _ in range(12):
        g = jax.grad(lambda s_: reference_energy(model, s_, image, label, bidirectional))(s)
        u, st = opt.update(g, st, s); s = optax.apply_updates(s, u)
    got = model.settle_scan(optax.adam(0.01), {}, {"image": image, "label": label}, n_steps=12)
    assert all(allclose(got[k], s[k], rtol=1e-3, atol=1e-4) for k in s)
