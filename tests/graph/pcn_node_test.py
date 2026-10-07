"""Tests for graph/nodes/pcn.py (PcnNode) and its "pcn" builder kind."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from pc_nox.models.graph import GraphModel, GraphModelConfig, NodeSpec
from pc_nox.models.graph.nodes.base import NodeInputs
from pc_nox.models.graph.nodes.pcn import PcnNode

KEY = jr.PRNGKey(0)


def allclose(a, b, rtol=1e-5, atol=1e-6):
    return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# The node on its own
# --------------------------------------------------------------------------- #

def test_top_down_prediction_is_act_of_linear_with_bias():
    node = PcnNode(own_size=5, parent_size=7, key=KEY)
    x = jr.normal(KEY, (7,))
    out = node.predict(NodeInputs(parents_curr=x))
    expected_pre = node.td_op.weight @ x + node.td_op.bias
    assert allclose(out.pre_activations["td"], expected_pre)
    assert allclose(out.predictions["td"], jnp.tanh(expected_pre))
    assert node.state_shape == (5,) and node.prediction_keys == ("td",)
    assert node.needs_parents_curr and not node.needs_children and not node.needs_own_prev


def test_root_mode_predicts_a_learned_prior():
    node = PcnNode(own_size=4, key=KEY)
    assert node.td_op is None and not node.needs_parents_curr
    out = node.predict(NodeInputs())
    assert out.predictions["td"].shape == (4,) and jnp.all(out.predictions["td"] == 0)   # tanh(0 prior)
    assert node.weight_entries() == []

    # the prior is trainable: the gradient of a state-vs-prediction energy reaches it
    state = jnp.ones(4)
    g = eqx.filter_grad(lambda n: 0.5 * jnp.sum((state - n.predict(NodeInputs()).predictions["td"]) ** 2))(node)
    assert jnp.any(g.prior != 0)


def test_bidirectional_adds_a_bottom_up_path_with_its_own_weights():
    node = PcnNode(own_size=5, parent_size=7, child_size=3, bidirectional=True, key=KEY)
    x, c = jr.normal(KEY, (7,)), jr.normal(jr.PRNGKey(1), (3,))
    out = node.predict(NodeInputs(parents_curr=x, children_curr=c))
    assert set(out.predictions) == {"td", "bu"} == set(out.pre_activations)
    assert allclose(out.predictions["bu"], jnp.tanh(node.bu_op.weight @ c + node.bu_op.bias))
    assert node.prediction_keys == ("td", "bu") and node.needs_children and node.supports_bottom_up
    assert not allclose(out.predictions["td"], out.predictions["bu"])


def test_bidirectional_validation_and_missing_input_errors():
    with pytest.raises(ValueError, match="requires child_size"):
        PcnNode(own_size=5, parent_size=7, bidirectional=True, key=KEY)
    node = PcnNode(own_size=5, parent_size=7, child_size=3, bidirectional=True, key=KEY)
    with pytest.raises(ValueError, match=r"PcnNode declared it needs \['children_curr'\]"):
        node.predict(NodeInputs(parents_curr=jnp.ones(7)))


def test_weight_entries_are_ff_matrices_without_biases():
    node = PcnNode(own_size=5, parent_size=7, child_size=3, bidirectional=True, key=KEY)
    entries = {e.name: e for e in node.weight_entries()}
    assert set(entries) == {"td_op.weight", "bu_op.weight"}
    assert all(e.tag == "ff" for e in entries.values())
    assert entries["td_op.weight"].matrix.shape == (5, 7) and entries["bu_op.weight"].matrix.shape == (5, 3)


def test_from_spec_rejects_non_1d_shapes_and_options():
    with pytest.raises(ValueError, match="must be 1-D"):
        PcnNode.from_spec(NodeSpec("n", "pcn", (2, 4)), (7,), None, jnp.tanh, key=KEY)
    with pytest.raises(ValueError, match="takes no options"):
        PcnNode.from_spec(NodeSpec("n", "pcn", 4, options={"x": 1}), (7,), None, jnp.tanh, key=KEY)


# --------------------------------------------------------------------------- #
# In a graph (the "pcn" kind)
# --------------------------------------------------------------------------- #

def static_chain():
    return GraphModelConfig(node_specs=(
        NodeSpec("z0", "pcn", 6),                                       # root: learned prior
        NodeSpec("z1", "pcn", 5, parents=("z0",)),
        NodeSpec("y", "pcn", 3, parents=("z1",), clamped=True, act_fn="identity"),
    ))


def test_static_chain_needs_no_previous_state_and_learns():
    cfg = static_chain()
    model = GraphModel(cfg, key=KEY)
    y = {"y": jnp.array([1.0, -1.0, 0.5])}
    states_prev = {}                                                     # nothing is recurrent
    states = model.init_activities(states_prev, None, y)
    e0 = model.energy_fn(states_prev, states, y)
    settled = model.settle_scan(optax.sgd(0.1), states_prev, y, n_steps=40)
    assert model.energy_fn(states_prev, settled, y) < e0

    grads = model.param_grad(states_prev, settled, y)
    assert jnp.any(grads.nodes["z0"].prior != 0)                         # the prior learns
    assert jnp.any(grads.nodes["z1"].td_op.bias != 0)                    # biases learn
    assert jnp.any(grads.nodes["y"].td_op.weight != 0)


def test_dense_node_below_a_conv_parent_gets_it_flattened():
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("core", "pcn", 16),
        NodeSpec("feat", "pcn_conv", (2, 4, 4), parents=("core",)),
        NodeSpec("head", "pcn", 5, parents=("feat",)),
    ))
    model = GraphModel(cfg, key=KEY)
    assert model.nodes["head"].td_op.weight.shape == (5, 32)             # 2*4*4 flattened
    states = model.init_activities({}, None, {})
    assert states["head"].shape == (5,)


def test_bidirectional_dense_node_with_a_conv_child_flattens_the_child():
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("core", "pcn", 16, bidirectional=True),
        NodeSpec("img", "pcn_conv", (2, 4, 4), parents=("core",), clamped=True, act_fn="identity"),
    ))
    model = GraphModel(cfg, key=KEY)
    assert model.nodes["core"].bu_op.weight.shape == (16, 32)
    y = {"img": jr.normal(KEY, (2, 4, 4))}
    states = model.init_activities({}, None, y)
    preds, _ = model.predict({}, states, None, y)
    assert set(preds["core"].predictions) == {"td", "bu"}
    grads = model.param_grad({}, states, y)
    assert jnp.any(grads.nodes["core"].bu_op.weight != 0)


def test_pcn_node_rejects_alpha():
    cfg = GraphModelConfig(node_specs=(NodeSpec("n", "pcn", 4, alpha=0.5),))
    with pytest.raises(ValueError, match="no memory"):
        GraphModel(cfg, key=KEY)
