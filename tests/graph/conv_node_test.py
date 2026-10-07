"""Tests for graph/nodes/conv.py, and (via a real conv node) for
graph/validation.py's pairing checks.

Run from the directory that CONTAINS the `graph/` package:

    python -m pytest graph/tests -v
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from pc_nox.models.graph.config import GraphModelConfig, NodeSpec
from pc_nox.models.graph.energies import ENERGY_REGISTRY, EnergyStrategy
from pc_nox.models.graph.nodes.base import NodeBase, NodeInputs, NodePrediction
from pc_nox.models.graph.nodes.conv import PcnConvNode
from pc_nox.models.graph.validation import validate_pairings

KEY = jr.PRNGKey(0)

# k=3, s=2, p=1, op=1 doubles H and W (the default op=0 gives 2H-1)
DOUBLE = dict(td_kernel_size=3, td_stride=2, td_padding=1, td_output_padding=1)
# k=4, s=2, p=1 halves H and W for the bottom-up Conv2d
HALVE = dict(bu_kernel_size=4, bu_stride=2, bu_padding=1)


def conv_node(**kw) -> PcnConvNode:
    kw.setdefault("key", KEY)
    return PcnConvNode(**kw)


# --------------------------------------------------------------------------- #
# Top-down path
# --------------------------------------------------------------------------- #

def test_dense_parent_is_projected_and_reshaped():
    node = conv_node(parent_shape=(16,), own_shape=(4, 8, 8))
    out = node.predict(NodeInputs(parents_curr=jnp.ones(16)))
    assert isinstance(out, NodePrediction)
    assert set(out.predictions) == {"td"}
    assert out.predictions["td"].shape == (4, 8, 8)
    assert jnp.allclose(out.predictions["td"], jnp.tanh(out.pre_activations["td"]))


def test_conv_parent_uses_transpose_conv():
    node = conv_node(parent_shape=(4, 8, 8), own_shape=(2, 16, 16), **DOUBLE)
    out = node.predict(NodeInputs(parents_curr=jnp.ones((4, 8, 8))))
    assert out.predictions["td"].shape == (2, 16, 16)
    assert not node.is_td_dense


def test_default_conv_options_that_do_not_hit_own_shape_raise_clearly():
    # defaults (k3, s2, p1, op0) map 8 -> 15, not 16
    with pytest.raises(ValueError, match=r"top-down op maps input shape \(4, 8, 8\) to \(2, 15, 15\)"):
        conv_node(parent_shape=(4, 8, 8), own_shape=(2, 16, 16))


def test_state_shape_and_non_bidirectional_capabilities():
    node = conv_node(parent_shape=(16,), own_shape=(4, 8, 8))
    assert node.state_shape == (4, 8, 8)
    assert node.prediction_keys == ("td",)
    assert node.needs_children is False and node.needs_parents_curr is True
    assert node.supports_bottom_up and node.provides_pre_activations
    assert node.supports_trace is False and node.needs_own_prev is False


# --------------------------------------------------------------------------- #
# Bottom-up path
# --------------------------------------------------------------------------- #

def bidirectional_conv_child() -> PcnConvNode:
    return conv_node(parent_shape=(4, 8, 8), own_shape=(2, 16, 16), child_shape=(3, 32, 32),
                     bidirectional=True, **DOUBLE, **HALVE)


def test_bidirectional_with_conv_child():
    node = bidirectional_conv_child()
    out = node.predict(NodeInputs(parents_curr=jnp.ones((4, 8, 8)), children_curr=jnp.ones((3, 32, 32))))
    assert set(out.predictions) == {"td", "bu"} == set(out.pre_activations)
    assert out.predictions["td"].shape == out.predictions["bu"].shape == (2, 16, 16)
    assert node.prediction_keys == ("td", "bu") and node.needs_children


def test_bidirectional_with_dense_child():
    node = conv_node(parent_shape=(16,), own_shape=(2, 16, 16), child_shape=(10,), bidirectional=True)
    out = node.predict(NodeInputs(parents_curr=jnp.ones(16), children_curr=jnp.ones(10)))
    assert out.predictions["bu"].shape == (2, 16, 16)
    assert node.is_bu_dense


def test_bidirectional_requires_child_shape():
    with pytest.raises(ValueError, match="requires child_shape"):
        conv_node(parent_shape=(16,), own_shape=(4, 8, 8), bidirectional=True)


def test_bottom_up_shape_mismatch_raises():
    with pytest.raises(ValueError, match="bottom-up op maps"):
        conv_node(parent_shape=(16,), own_shape=(2, 16, 16), child_shape=(3, 40, 40), bidirectional=True, **HALVE)


def test_missing_children_input_fails_with_named_error():
    node = bidirectional_conv_child()
    with pytest.raises(ValueError, match=r"PcnConvNode declared it needs \['children_curr'\]"):
        node.predict(NodeInputs(parents_curr=jnp.ones((4, 8, 8))))


@pytest.mark.parametrize("own, parent", [((4, 8), (16,)), ((4, 8, 8), (4, 8)), ((4, 8, 8), ())])
def test_bad_shapes_rejected(own, parent):
    with pytest.raises(ValueError):
        conv_node(parent_shape=parent, own_shape=own)


# --------------------------------------------------------------------------- #
# Weights (what the regularisers see)
# --------------------------------------------------------------------------- #

def test_weight_entries_are_2d_ff_views():
    node = bidirectional_conv_child()
    entries = {e.name: e for e in node.weight_entries()}
    assert set(entries) == {"td_op.weight", "bu_op.weight"}
    assert all(e.tag == "ff" and e.matrix.ndim == 2 for e in entries.values())
    assert entries["td_op.weight"].matrix.shape == (2, 4 * 3 * 3)   # (out_ch, in_ch*kh*kw)
    assert entries["bu_op.weight"].matrix.shape == (2, 3 * 4 * 4)


def test_weight_entries_without_bottom_up_has_only_td():
    node = conv_node(parent_shape=(16,), own_shape=(4, 8, 8))
    assert [e.name for e in node.weight_entries()] == ["td_op.weight"]
    assert node.weight_entries()[0].matrix.shape == (256, 16)


def test_regulariser_gradient_reaches_the_real_weights_through_the_view():
    node = bidirectional_conv_child()
    loss = lambda n: sum(jnp.sum(e.matrix ** 2) for e in n.weight_entries())
    grads = eqx.filter_grad(loss)(node)
    assert jnp.allclose(grads.td_op.weight, 2 * node.td_op.weight)
    assert jnp.allclose(grads.bu_op.weight, 2 * node.bu_op.weight)


# --------------------------------------------------------------------------- #
# JAX behaviour
# --------------------------------------------------------------------------- #

def test_jit_matches_eager_and_state_gradients_are_finite():
    node = bidirectional_conv_child()
    parents, children = jr.normal(KEY, (4, 8, 8)), jr.normal(KEY, (3, 32, 32))

    def energy(state, parents):
        out = node.predict(NodeInputs(parents_curr=parents, children_curr=children))
        return sum(0.5 * jnp.sum((state - p) ** 2) for p in out.predictions.values())

    state = jnp.zeros((2, 16, 16))
    eager = energy(state, parents)
    assert jnp.allclose(jax.jit(energy)(state, parents), eager, atol=1e-5)
    g_state, g_parents = jax.grad(energy, argnums=(0, 1))(state, parents)
    assert jnp.all(jnp.isfinite(g_state)) and jnp.any(g_parents != 0)   # error reaches the parent


def test_static_metadata_is_hashable_for_jit():
    hash(bidirectional_conv_child().own_shape)
    jax.tree_util.tree_structure(bidirectional_conv_child())            # flattens without error


# --------------------------------------------------------------------------- #
# from_spec
# --------------------------------------------------------------------------- #

def spec(**kw) -> NodeSpec:
    base = dict(name="c", kind="conv", shape=(2, 16, 16), parents=("p",), options=DOUBLE)
    base.update(kw)
    return NodeSpec(**base)


def test_from_spec_builds_and_applies_options():
    node = PcnConvNode.from_spec(spec(), (4, 8, 8), None, jnp.tanh, key=KEY)
    assert node.state_shape == (2, 16, 16) and not node.bidirectional


def test_from_spec_bidirectional_uses_child_shape():
    s = spec(bidirectional=True, options={**DOUBLE, **HALVE})
    node = PcnConvNode.from_spec(s, (4, 8, 8), (3, 32, 32), jnp.tanh, key=KEY)
    assert node.bidirectional and node.bu_op is not None


def test_from_spec_ignores_child_shape_when_not_bidirectional():
    node = PcnConvNode.from_spec(spec(), (4, 8, 8), (3, 32, 32), jnp.tanh, key=KEY)
    assert node.bu_op is None


def test_from_spec_rejects_unknown_option():
    with pytest.raises(ValueError, match=r"unknown PcnConvNode options \['td_kernal_size'\]"):
        PcnConvNode.from_spec(spec(options={"td_kernal_size": 3}), (4, 8, 8), None, jnp.tanh, key=KEY)


# --------------------------------------------------------------------------- #
# validation.py against a real node
# --------------------------------------------------------------------------- #

class _Stub(NodeBase):
    """Minimal node: no bottom-up path, no pre-activations. A root stub
    (the default) takes no parents; pass has_parents=True for a child."""
    shape: tuple = eqx.field(static=True)
    has_parents: bool = eqx.field(static=True, default=False)

    @property
    def state_shape(self):
        return self.shape

    @property
    def needs_parents_curr(self):
        return self.has_parents

    def predict(self, inp):
        return NodePrediction({"td": jnp.zeros(self.shape)})

    def weight_entries(self):
        return []


def graph(bidirectional: bool):
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("p", "stub", (16,)),
        NodeSpec("c", "conv", (4, 8, 8), parents=("p",), bidirectional=bidirectional),
        NodeSpec("d", "stub", (3, 16, 16), parents=("c",)),
    ))
    c = conv_node(parent_shape=(16,), own_shape=(4, 8, 8), child_shape=(3, 16, 16),
                  bidirectional=bidirectional, bu_kernel_size=4, bu_stride=2, bu_padding=1)
    nodes = {"p": _Stub((16,)), "c": c, "d": _Stub((3, 16, 16), has_parents=True)}
    return cfg, nodes


@pytest.mark.parametrize("bidirectional", [False, True])
def test_valid_graph_passes_pairing_validation(bidirectional):
    cfg, nodes = graph(bidirectional)
    validate_pairings(cfg, nodes, ENERGY_REGISTRY)


def test_spec_bidirectional_but_node_not_is_caught():
    cfg, nodes = graph(True)
    nodes["c"] = graph(False)[1]["c"]                      # node built without a bottom-up path
    with pytest.raises(ValueError, match="disagrees with the node's prediction paths"):
        validate_pairings(cfg, nodes, ENERGY_REGISTRY)


def test_bidirectional_on_a_node_type_that_cannot_is_caught():
    cfg = GraphModelConfig(node_specs=(NodeSpec("p", "stub", (4,), bidirectional=True),))
    with pytest.raises(ValueError, match="no bottom-up path"):
        validate_pairings(cfg, {"p": _Stub((4,))}, ENERGY_REGISTRY)


def test_energy_that_needs_pre_activations_is_rejected_for_nodes_without_them():
    registry = {"pc": EnergyStrategy(fn=None, needs_pre_activations=True)}
    cfg = GraphModelConfig(node_specs=(NodeSpec("p", "stub", (4,)),))
    with pytest.raises(ValueError, match="needs pre-activations"):
        validate_pairings(cfg, {"p": _Stub((4,))}, registry)


def test_energy_that_cannot_score_a_bottom_up_path_is_rejected():
    registry = {"pc": EnergyStrategy(fn=None, allowed_paths=frozenset({"td"}))}
    cfg, nodes = graph(True)
    with pytest.raises(ValueError, match=r"paths \['bu'\] that this energy cannot score"):
        validate_pairings(cfg, nodes, registry)


def test_all_pairing_errors_reported_together_with_node_names():
    cfg = GraphModelConfig(node_specs=(
        NodeSpec("p", "stub", (4,), bidirectional=True, energy="nope"),
        NodeSpec("q", "stub", (5,)),
    ))
    nodes = {"p": _Stub((4,)), "q": _Stub((6,))}           # q has the wrong state_shape
    with pytest.raises(ValueError) as exc:
        validate_pairings(cfg, nodes, ENERGY_REGISTRY)
    msg = str(exc.value)
    assert "layer 'p'" not in msg and "node 'p'" in msg and "node 'q'" in msg
    assert "unknown energy" in msg and "state_shape" in msg


def test_default_energy_is_resolved_from_the_config_not_read_off_the_spec():
    # spec.energy is None here; the model-wide default must be used for the lookup
    cfg = GraphModelConfig(node_specs=(NodeSpec("p", "stub", (4,)),), energy="pc")
    assert cfg.node_specs[0].energy is None
    validate_pairings(cfg, {"p": _Stub((4,))}, ENERGY_REGISTRY)
