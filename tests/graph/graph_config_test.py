"""Tests for graph/config.py (NodeSpec, GraphModelConfig).

Run from the directory that CONTAINS the `graph/` package:

    python -m pytest graph/tests -v

These cover the parts `ModelBase` checkpointing depends on: canonical
ordering, the asdict -> JSON -> `config_cls(**dict)` round trip, hashability
(the config is a static jit field), and early, named validation errors.
"""

import json
from dataclasses import asdict

import pytest

from pc_nox.models.graph.config import GraphModelConfig, NodeSpec


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def make_config(**overrides) -> GraphModelConfig:
    """A branching, multi-level config, declared deliberately out of order."""
    specs = (
        NodeSpec("vis_obs", "obs", (4096,), parents=("vis1",), clamped=True),
        NodeSpec("core", "control", 64),
        NodeSpec("vis0", "hidden", 128, parents=("core",), alpha=0.5,
                 options={"kernel": (3, 3), "stride": 2}),
        NodeSpec("aud0", "hidden", 32, parents=("core",), weight_decay=0.1),
        NodeSpec("vis1", "hidden", 256, parents=("vis0",), level=5),
        NodeSpec("aud_obs", "obs", 16, parents=("aud0",), clamped=True, precision=4.0),
    )
    return GraphModelConfig(node_specs=specs, input_size=3, **overrides)


def json_round_trip(cfg: GraphModelConfig) -> GraphModelConfig:
    """Exactly what save_checkpoint / load_checkpoint do to a config."""
    loaded = json.loads(json.dumps(asdict(cfg), indent=4))
    return GraphModelConfig(**loaded)


# --------------------------------------------------------------------------- #
# Canonical form
# --------------------------------------------------------------------------- #

def test_canonical_order_is_by_level_then_declaration():
    cfg = make_config()
    assert cfg.names == ("core", "vis0", "aud0", "aud_obs", "vis1", "vis_obs")


def test_levels_are_resolved_and_explicit_levels_kept():
    cfg = make_config()
    levels = {s.name: s.level for s in cfg.node_specs}
    # derived: root 0, children of root 1, aud_obs 2. Explicit vis1=5 kept,
    # so vis_obs (derived from vis1) becomes 6.
    assert levels == {"core": 0, "vis0": 1, "aud0": 1, "aud_obs": 2, "vis1": 5, "vis_obs": 6}


def test_no_node_has_unresolved_level():
    assert all(s.level is not None for s in make_config().node_specs)


def test_config_construction_is_idempotent():
    cfg = make_config()
    assert GraphModelConfig(node_specs=cfg.node_specs, input_size=3) == cfg


def test_ties_within_a_level_keep_declaration_order():
    a = GraphModelConfig(node_specs=(
        NodeSpec("r", "k", 4), NodeSpec("x", "k", 4, parents=("r",)), NodeSpec("y", "k", 4, parents=("r",)),
    ))
    b = GraphModelConfig(node_specs=(
        NodeSpec("r", "k", 4), NodeSpec("y", "k", 4, parents=("r",)), NodeSpec("x", "k", 4, parents=("r",)),
    ))
    assert a.names == ("r", "x", "y") and b.names == ("r", "y", "x")  # ties keep declaration order


# --------------------------------------------------------------------------- #
# Derived views
# --------------------------------------------------------------------------- #

def test_clamped_unclamped_split():
    cfg = make_config()
    assert [s.name for s in cfg.unclamped] == ["core", "vis0", "aud0", "vis1"]
    assert [s.name for s in cfg.clamped] == ["aud_obs", "vis_obs"]


def test_children_is_inverse_of_parents():
    assert make_config().children() == {
        "core": ("vis0", "aud0"),
        "vis0": ("vis1",),
        "aud0": ("aud_obs",),
        "aud_obs": (),
        "vis1": ("vis_obs",),
        "vis_obs": (),
    }


def test_spec_lookup_and_unknown_name():
    cfg = make_config()
    assert cfg.spec("vis0").shape == (128,)
    with pytest.raises(KeyError, match="no node named 'nope'"):
        cfg.spec("nope")


def test_get_falls_back_to_model_default_and_respects_override():
    cfg = make_config(weight_decay=0.01)
    assert cfg.get(cfg.spec("aud0"), "weight_decay") == 0.1     # per-node override
    assert cfg.get(cfg.spec("vis0"), "weight_decay") == 0.01    # inherits model default
    assert cfg.get(cfg.spec("vis0"), "energy") == "pc"


def test_explicit_zero_override_is_not_treated_as_missing():
    cfg = make_config(weight_decay=0.5)
    spec = NodeSpec("z", "k", 4, weight_decay=0.0)
    assert cfg.get(spec, "weight_decay") == 0.0                  # 0.0 disables; it must not fall back


# --------------------------------------------------------------------------- #
# Spec coercion
# --------------------------------------------------------------------------- #

def test_int_shape_shorthand():
    assert NodeSpec("a", "k", 64).shape == (64,)


def test_options_are_canonical_and_order_independent():
    a = NodeSpec("a", "k", 4, options={"stride": 2, "kernel": [3, 3]})
    b = NodeSpec("a", "k", 4, options=(("kernel", (3, 3)), ("stride", 2)))
    assert a.options == b.options == (("kernel", (3, 3)), ("stride", 2))
    assert a.opts == {"kernel": (3, 3), "stride": 2}


def test_list_fields_are_coerced_to_tuples():
    s = NodeSpec("a", "k", [4, 4], parents=["p"], options=[["k", [1, 2]]])
    assert s.shape == (4, 4) and s.parents == ("p",) and s.options == (("k", (1, 2)),)


# --------------------------------------------------------------------------- #
# Hashing / serialisation (what jit and ModelBase checkpointing rely on)
# --------------------------------------------------------------------------- #

def test_config_is_hashable():
    cfg = make_config()
    assert isinstance(hash(cfg), int)
    assert {cfg: 1}[make_config()] == 1                        # equal configs are interchangeable


def test_json_round_trip_preserves_equality_and_hash():
    cfg = make_config()
    cfg2 = json_round_trip(cfg)
    assert cfg2 == cfg
    assert hash(cfg2) == hash(cfg)


def test_json_round_trip_restores_tuples_and_specs():
    cfg2 = json_round_trip(make_config())
    assert isinstance(cfg2.node_specs, tuple)
    assert all(isinstance(s, NodeSpec) for s in cfg2.node_specs)
    s = cfg2.spec("vis0")
    assert isinstance(s.shape, tuple) and isinstance(s.parents, tuple)
    assert s.opts == {"kernel": (3, 3), "stride": 2}


def test_raw_loaded_json_really_is_lists_and_dicts():
    # Guards the premise of the coercion: if this ever fails, the round-trip tests above prove less.
    raw = json.loads(json.dumps(asdict(make_config())))
    assert isinstance(raw["node_specs"], list) and isinstance(raw["node_specs"][0], dict)
    assert isinstance(raw["node_specs"][0]["shape"], list)


def test_old_checkpoint_missing_newer_defaulted_field_still_loads():
    # Evolution rule: new fields are added with defaults, so older saved
    # configs (which lack them) must keep loading.
    raw = json.loads(json.dumps(asdict(make_config())))
    del raw["trace_mode"]
    del raw["node_specs"][0]["precision"]
    cfg = GraphModelConfig(**raw)
    assert cfg.trace_mode == "readout" and cfg.spec("vis_obs").precision == 1.0


GOLDEN_CONFIG_JSON = """
{"node_specs": [
  {"name": "core", "kind": "control", "shape": [8], "parents": [], "level": 0},
  {"name": "h", "kind": "hidden", "shape": [4], "parents": ["core"], "level": 1, "alpha": 0.5},
  {"name": "y", "kind": "obs", "shape": [2], "parents": ["h"], "level": 2, "clamped": true}
 ],
 "input_size": 1, "act_fn": "tanh", "energy": "pc"}
"""

def test_golden_config_still_loads():
    """A frozen config from today's schema. If a future change breaks this,
    it would also break every checkpoint saved with today's schema."""
    cfg = GraphModelConfig(**json.loads(GOLDEN_CONFIG_JSON))
    assert cfg.names == ("core", "h", "y")
    assert cfg.spec("h").alpha == 0.5 and cfg.spec("y").clamped


# --------------------------------------------------------------------------- #
# Validation: each rule is one deliberately bad config
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("build, message", [
    (lambda: GraphModelConfig(node_specs=(NodeSpec("a", "k", 4), NodeSpec("b", "k", 4, parents=("zzz",)))),
     "node 'b' has unknown parent 'zzz'"),
    (lambda: GraphModelConfig(node_specs=(NodeSpec("a", "k", 4, parents=("a",)),)),
     "lists itself as a parent"),
    (lambda: GraphModelConfig(node_specs=(NodeSpec("a", "k", 4, parents=("b",)), NodeSpec("b", "k", 4, parents=("a",)))),
     "cycle through node"),
    (lambda: GraphModelConfig(node_specs=(NodeSpec("a", "k", 4, level=2), NodeSpec("b", "k", 4, parents=("a",), level=2))),
     "strictly lower levels"),
    (lambda: GraphModelConfig(node_specs=(NodeSpec("a", "k", 4), NodeSpec("a", "k", 4))),
     "duplicate node names"),
    (lambda: GraphModelConfig(node_specs=()),
     "at least one node"),
    (lambda: GraphModelConfig(node_specs=(NodeSpec("a", "k", 4),), weight_decay_scope="most"),
     "weight_decay_scope"),
    (lambda: GraphModelConfig(node_specs=(NodeSpec("a", "k", 4),), trace_mode="sometimes"),
     "trace_mode"),
    (lambda: GraphModelConfig(node_specs=(NodeSpec("a", "k", 4),), input_size=-1),
     "input_size"),
    (lambda: GraphModelConfig(node_specs=(NodeSpec("a", "k", 4),), weight_decay=-0.1),
     "weight_decay must be >= 0"),
], ids=["unknown-parent", "self-parent", "cycle", "level-not-lower", "duplicate-names",
        "empty", "bad-scope", "bad-trace-mode", "negative-input-size", "negative-decay"])
def test_invalid_config_raises_with_named_message(build, message):
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize("kwargs, message", [
    (dict(name="my-node"), "valid Python identifier"),
    (dict(name="1abc"), "valid Python identifier"),
    (dict(shape=()), "non-empty tuple of positive ints"),
    (dict(shape=(4, 0)), "non-empty tuple of positive ints"),
    (dict(parents=("p", "p")), "duplicate parents"),
    (dict(alpha=0.0), "alpha must be in"),
    (dict(alpha=1.5), "alpha must be in"),
    (dict(clamped=True, alpha=0.5), "clamped node cannot be leaky"),
    (dict(precision=0.0), "precision must be > 0"),
    (dict(weight_decay=-1.0), "weight_decay must be >= 0"),
    (dict(options={"fn": print}), "not a JSON-compatible"),
    (dict(options=[("only_key",)]), "must be \\(key, value\\) pairs"),
], ids=["bad-identifier", "leading-digit", "empty-shape", "zero-dim", "dup-parents", "alpha-zero",
        "alpha-too-big", "leaky-clamped", "zero-precision", "negative-wd", "non-json-option", "bad-pair"])
def test_invalid_spec_raises(kwargs, message):
    base = dict(name="a", kind="k", shape=(4,))
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        NodeSpec(**base)


def test_all_structural_errors_are_reported_together():
    # Fixing a bad config should not be one-error-at-a-time.
    with pytest.raises(ValueError) as exc:
        GraphModelConfig(
            node_specs=(NodeSpec("a", "k", 4, parents=("ghost",)), NodeSpec("a", "k", 4)),
            weight_decay_scope="most", input_size=-1,
        )
    msg = str(exc.value)
    assert "duplicate node names" in msg and "unknown parent 'ghost'" in msg
    assert "weight_decay_scope" in msg and "input_size" in msg


def test_config_is_immutable():
    cfg = make_config()
    with pytest.raises(Exception):                              # dataclasses.FrozenInstanceError
        cfg.input_size = 99
