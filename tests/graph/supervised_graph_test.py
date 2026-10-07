"""Classic supervised PCNs in the graph: a clamped InputNode at the top, a
straight chain of nodes below it, and a label node that is clamped in
training and simply omitted from `observation` at test time.

Also covers the direction-aware conv node (Conv2d when a node predicts
DOWN in resolution, ConvTranspose2d when it predicts UP).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from pc_nox.models.graph import GraphModel, GraphModelConfig, NodeSpec
from pc_nox.models.graph.energies import ENERGY_REGISTRY, pc_energy
from pc_nox.models.graph.nodes.base import NodeBase, NodeInputs, NodePrediction
from pc_nox.models.graph.nodes.conv import PcnConvNode
from pc_nox.models.graph.nodes.input import InputNode
from pc_nox.models.graph.runners import make_eval_step, make_train_step
from pc_nox.models.graph.validation import validate_pairings

KEY = jr.PRNGKey(0)


def allclose(a, b, rtol=1e-5, atol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


# =========================================================================== #
# Direction-aware conv: down -> Conv2d, up -> ConvTranspose2d
# =========================================================================== #

def conv(parent, own, **kw):
    return PcnConvNode(parent_shape=parent, own_shape=own, key=KEY, **kw)


def test_predicting_down_in_resolution_uses_a_regular_strided_conv():
    node = conv((3, 32, 32), (8, 16, 16))                   # defaults: k3 s2 p1
    assert isinstance(node.td_op, eqx.nn.Conv2d) and not isinstance(node.td_op, eqx.nn.ConvTranspose2d)
    out = node.predict(NodeInputs(parents_curr=jnp.ones((3, 32, 32))))
    assert out.predictions["td"].shape == (8, 16, 16)


def test_predicting_up_in_resolution_uses_a_transposed_conv():
    node = conv((8, 16, 16), (3, 32, 32), td_output_padding=1)
    assert isinstance(node.td_op, eqx.nn.ConvTranspose2d)


def test_same_resolution_uses_a_regular_conv():
    node = conv((8, 16, 16), (4, 16, 16), td_stride=1)
    assert isinstance(node.td_op, eqx.nn.Conv2d)


def test_a_transposed_conv_cannot_downsample():
    # This is the shape-level reason image-at-the-top needs Conv2d: it is not a performance question.
    with pytest.raises(ValueError, match=r"maps input shape \(3, 32, 32\) to \(8, 63, 63\).*does not match"):
        conv((3, 32, 32), (8, 16, 16), td_mode="transpose")


def test_a_regular_conv_cannot_upsample():
    with pytest.raises(ValueError, match="does not match own_shape"):
        conv((8, 16, 16), (3, 32, 32), td_mode="conv")


def test_output_padding_is_rejected_where_it_does_not_apply():
    with pytest.raises(ValueError, match="td_output_padding only applies"):
        conv((3, 32, 32), (8, 16, 16), td_output_padding=1)
    with pytest.raises(ValueError, match="bu_output_padding only applies"):
        conv((3, 8, 8), (8, 16, 16), child_shape=(4, 32, 32), bidirectional=True, td_output_padding=1,
             bu_output_padding=1, bu_kernel_size=4)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="td_mode must be one of"):
        conv((3, 32, 32), (8, 16, 16), td_mode="sideways")


def test_bottom_up_op_mirrors_the_direction():
    # child LARGER than own -> Conv2d (own is built by shrinking its child), as before
    big_child = conv((4, 8, 8), (2, 16, 16), td_output_padding=1, child_shape=(3, 32, 32), bidirectional=True,
                     bu_kernel_size=4)
    assert isinstance(big_child.bu_op, eqx.nn.Conv2d) and not isinstance(big_child.bu_op, eqx.nn.ConvTranspose2d)
    # child SMALLER than own (the supervised direction) -> ConvTranspose2d
    small_child = conv((3, 32, 32), (2, 16, 16), child_shape=(5, 8, 8), bidirectional=True, bu_output_padding=1)
    assert isinstance(small_child.bu_op, eqx.nn.ConvTranspose2d)
    out = small_child.predict(NodeInputs(parents_curr=jnp.ones((3, 32, 32)), children_curr=jnp.ones((5, 8, 8))))
    assert out.predictions["td"].shape == out.predictions["bu"].shape == (2, 16, 16)


def test_conv_options_are_accepted_from_a_spec():
    spec = NodeSpec("c", "pcn_conv", (8, 16, 16), parents=("x",), options={"td_mode": "conv", "bu_mode": "auto"})
    node = PcnConvNode.from_spec(spec, (3, 32, 32), None, jnp.tanh, key=KEY)
    assert isinstance(node.td_op, eqx.nn.Conv2d)


# =========================================================================== #
# InputNode
# =========================================================================== #

def test_input_node_is_parameter_free_and_predicts_nothing():
    node = InputNode((3, 32, 32))
    assert node.state_shape == (3, 32, 32) and node.prediction_keys == ()
    assert node.weight_entries() == [] and not jax.tree_util.tree_leaves(eqx.filter(node, eqx.is_array))
    assert node.predict(NodeInputs()).predictions == {}
    with pytest.raises(ValueError, match="positive ints"):
        InputNode((3, 0))


def test_pc_energy_of_a_node_with_no_predictions_is_zero():
    e = pc_energy(jnp.ones(5), NodePrediction({}, {}), None)
    assert e.shape == () and e == 0
    assert ENERGY_REGISTRY["pc"].required_paths == frozenset()


def test_input_node_must_be_clamped_and_parentless():
    with pytest.raises(ValueError, match="must be clamped=True"):
        GraphModel(GraphModelConfig(node_specs=(NodeSpec("x", "input", (4,)),)), key=KEY)
    with pytest.raises(ValueError, match="no parents"):
        GraphModel(GraphModelConfig(node_specs=(
            NodeSpec("a", "input", (4,), clamped=True), NodeSpec("x", "input", (4,), parents=("a",), clamped=True))), key=KEY)


def test_an_unclamped_node_without_a_td_prediction_is_rejected():
    class Silent(NodeBase):
        @property
        def state_shape(self): return (3,)
        @property
        def prediction_keys(self): return ()
        @property
        def needs_parents_curr(self): return False
        def predict(self, inp): return NodePrediction({})
        def weight_entries(self): return []

    cfg = GraphModelConfig(node_specs=(NodeSpec("n", "silent", (3,)),))
    with pytest.raises(ValueError, match="must return a 'td' prediction"):
        validate_pairings(cfg, {"n": Silent()}, ENERGY_REGISTRY)


# =========================================================================== #
# The classic supervised chain: image -> conv -> conv -> dense -> label
# =========================================================================== #

def supervised_config(n_classes=3):
    return GraphModelConfig(node_specs=(
        NodeSpec("image", "input", (1, 16, 16), clamped=True),
        NodeSpec("c1", "pcn_conv", (4, 8, 8), parents=("image",), act_fn="tanh"),
        NodeSpec("c2", "pcn_conv", (6, 4, 4), parents=("c1",), act_fn="tanh"),
        NodeSpec("fc", "pcn", 12, parents=("c2",), act_fn="tanh"),
        NodeSpec("label", "pcn", n_classes, parents=("fc",), act_fn="identity", energy="ce", clamped=True),
    ))


def feedforward(model, image):
    """The same network evaluated as a plain feedforward pass, by hand."""
    n = model.nodes
    h1 = jnp.tanh(n["c1"].td_op(image))
    h2 = jnp.tanh(n["c2"].td_op(h1))
    h3 = jnp.tanh(n["fc"].td_op(h2.reshape(-1)))
    return n["label"].td_op(h3)


def test_straight_chain_builds_with_downsampling_convs_and_a_flattened_dense_layer():
    model = GraphModel(supervised_config(), key=KEY)
    assert isinstance(model.nodes["c1"].td_op, eqx.nn.Conv2d) and isinstance(model.nodes["c2"].td_op, eqx.nn.Conv2d)
    assert model.nodes["fc"].td_op.weight.shape == (12, 6 * 4 * 4)
    assert model.config.names == ("image", "c1", "c2", "fc", "label")


def test_initial_state_is_the_feedforward_pass_and_only_the_label_has_energy():
    model = GraphModel(supervised_config(), key=KEY)
    image = jr.uniform(KEY, (1, 16, 16))
    obs = {"image": image, "label": jax.nn.one_hot(1, 3)}
    states = model.init_activities({}, None, obs)
    assert set(states) == {"c1", "c2", "fc"}
    preds, y_hat = model.predict({}, states, None, obs)
    assert set(y_hat) == {"label"}                                       # the input node is not in y_hat
    assert allclose(y_hat["label"], feedforward(model, image))
    lw = model.energy_fn({}, states, obs, return_layerwise=True)
    assert np.all(np.asarray(lw[:4]) == 0) and lw[4] > 0                 # hidden errors are zero at the feedforward point


def test_at_test_time_the_label_is_omitted_and_settling_leaves_the_prediction_unchanged():
    model = GraphModel(supervised_config(), key=KEY)
    image = jr.uniform(KEY, (1, 16, 16))
    eval_step = make_eval_step(optax.adam(0.01), n_infer_steps=10)
    _, y_before, y_after, e_before, e_after, _ = eval_step(model, {"image": image[None]})
    assert allclose(y_before["label"][0], feedforward(model, image))
    assert allclose(y_after["label"], y_before["label"], atol=1e-5)      # nothing pushes the states: classic PC test time
    assert float(e_after[0]) < 1e-6


def test_label_clamped_inference_moves_the_states_and_lowers_the_energy():
    model = GraphModel(supervised_config(), key=KEY)
    obs = {"image": jr.uniform(KEY, (1, 16, 16)), "label": jax.nn.one_hot(2, 3)}
    e0 = model.energy_fn({}, model.init_activities({}, None, obs), obs)
    settled = model.settle_scan(optax.sgd(0.1), {}, obs, n_steps=40)
    assert model.energy_fn({}, settled, obs) < e0
    grads = model.param_grad({}, settled, obs)
    assert jnp.any(grads.nodes["c1"].td_op.weight != 0) and jnp.any(grads.nodes["label"].td_op.weight != 0)
    assert all(jnp.all(jnp.isfinite(l)) for l in jax.tree_util.tree_leaves(grads))


def make_toy_images(n_per_class=24, size=16):
    """Three classes distinguished by WHERE a bright bar sits: a task a conv net must learn."""
    rng = np.random.default_rng(0)
    imgs, labels = [], []
    for c in range(3):
        for _ in range(n_per_class):
            img = rng.normal(0.0, 0.1, (1, size, size))
            r0 = 2 + 5 * c
            img[0, r0:r0 + 3, 2:size - 2] += 1.0
            imgs.append(img); labels.append(c)
    imgs, labels = np.stack(imgs).astype(np.float32), np.array(labels)
    perm = rng.permutation(len(labels))
    return jnp.array(imgs[perm]), jax.nn.one_hot(jnp.array(labels[perm]), 3)


def test_the_supervised_conv_graph_learns_a_toy_task_end_to_end():
    X, Y = make_toy_images()
    model = GraphModel(supervised_config(), key=KEY)
    param_optim = optax.adam(3e-3)
    opt_state = param_optim.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(param_optim, optax.adam(0.05), n_infer_steps=15)
    eval_step = make_eval_step(optax.adam(0.05), n_infer_steps=15)

    acc = lambda m: float(jnp.mean(jnp.argmax(eval_step(m, {"image": X})[2]["label"], -1) == jnp.argmax(Y, -1)))
    chance = acc(model)
    for epoch in range(25):
        for i in range(0, len(X), 24):
            model, opt_state, *_ = train_step(model, opt_state, {"image": X[i:i + 24], "label": Y[i:i + 24]})
    assert acc(model) > 0.9 and acc(model) > chance + 0.4
