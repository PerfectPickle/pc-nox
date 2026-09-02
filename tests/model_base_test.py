"""model_base_test.py

Correctness & sanity tests for ModelBase.save_checkpoint / load_checkpoint
(pc_nox/models/model_base.py).

These are deliberately written against a minimal DUMMY model rather than
TpchModel: save/load lives on ModelBase and should work identically for any
subclass, so testing it through the smallest possible concrete model keeps
these tests from silently depending on tPC-H-specific behaviour. A couple of
integration tests at the bottom re-run the key round-trip checks against the
real TpchModel, to make sure nothing about the actual model breaks the
generic contract (e.g. unusual pytree structure, act_fn as a static field).

Mirrors the fixture/helper conventions in tpch_test.py: small, mismatched
layer sizes; a reusable assert_allclose; docstrings stating the property
under test up front.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from models.model_base import ModelBase, MODEL_REGISTRY
from models.tpch import TpchModel  # only used for the cross-subclass dispatch test below


def assert_allclose(actual, expected, name, atol=1e-6, rtol=1e-6):
    actual, expected = jnp.asarray(actual), jnp.asarray(expected)
    assert actual.shape == expected.shape, (
        f"{name}: shape mismatch -- got {actual.shape}, expected {expected.shape}"
    )
    assert jnp.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name}: values differ, max abs diff = {float(jnp.max(jnp.abs(actual - expected)))}"
    )


# =============================================================================
# Minimal dummy models -- smallest possible ModelBase subclasses, just
# enough surface area to exercise save/load in isolation from tPC-H.
# =============================================================================

@dataclass(frozen=True)  # frozen -> hashable, needed wherever a config is stored as an eqx static field
class DummyConfig:
    size: int


class DummyModel(eqx.Module, ModelBase):
    """A trivial single-linear-layer model. Supports activities checkpointing.

    Deliberately does NOT store `self.config` -- this is what exercises
    save_checkpoint's "config must be passed explicitly" branch. See
    DummyModelWithSelfConfig below for the opposite case."""

    model_type = "dummy_model_base_test"
    config_cls = DummyConfig

    W: eqx.nn.Linear

    def __init__(self, size: int, *, key):
        self.W = eqx.nn.Linear(size, size, key=key)

    def predict(self, x):
        return self.W(x)

    @classmethod
    def from_config(cls, config: DummyConfig, *, key) -> "DummyModel":
        return cls(config.size, key=key)

    @classmethod
    def zero_activities(cls, config: DummyConfig):
        return [jnp.zeros(config.size)]


class DummyModelNoActivities(eqx.Module, ModelBase):
    """Same as DummyModel but deliberately does NOT override zero_activities,
    to exercise ModelBase's default (NotImplementedError) fallback."""

    model_type = "dummy_model_base_test_no_activities"
    config_cls = DummyConfig

    W: eqx.nn.Linear

    def __init__(self, size: int, *, key):
        self.W = eqx.nn.Linear(size, size, key=key)

    def predict(self, x):
        return self.W(x)

    @classmethod
    def from_config(cls, config: DummyConfig, *, key) -> "DummyModelNoActivities":
        return cls(config.size, key=key)


class DummyModelWithSelfConfig(eqx.Module, ModelBase):
    """Same as DummyModel, but stores its config as `self.config`, mirroring
    the pattern real subclasses (e.g. TpchModel) now use. Exists to test
    save_checkpoint's `config = config if config is not None else
    getattr(self, "config", None)` fallback in isolation from DummyModel,
    which deliberately has no self.config."""

    model_type = "dummy_model_base_test_with_self_config"
    config_cls = DummyConfig

    W: eqx.nn.Linear
    config: DummyConfig = eqx.field(static=True)

    def __init__(self, size: int, *, key):
        self.W = eqx.nn.Linear(size, size, key=key)
        self.config = DummyConfig(size)

    def predict(self, x):
        return self.W(x)

    @classmethod
    def from_config(cls, config: DummyConfig, *, key) -> "DummyModelWithSelfConfig":
        return cls(config.size, key=key)


DUMMY_SIZE = 5


@pytest.fixture
def fx_config():
    return DummyConfig(size=DUMMY_SIZE)


@pytest.fixture
def fx_model(fx_config):
    return DummyModel.from_config(fx_config, key=jr.key(0))


@pytest.fixture
def fx_model_with_self_config(fx_config):
    return DummyModelWithSelfConfig.from_config(fx_config, key=jr.key(0))


# =============================================================================
# A. save_checkpoint -- files written, no more / no less than requested
# =============================================================================

def test_save_checkpoint_writes_expected_files_with_no_optional_state(fx_model, fx_config, tmp_path):
    """Base case: no opt_state, no activities. Only checkpoint.json and
    weights.eqx should be written -- opt_state.eqx/activities.eqx must NOT
    appear when their inputs are None."""
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt")
    assert out_dir == tmp_path / "ckpt"
    assert (out_dir / "checkpoint.json").exists()
    assert (out_dir / "weights.eqx").exists()
    assert not (out_dir / "opt_state.eqx").exists()
    assert not (out_dir / "activities.eqx").exists()


def test_save_checkpoint_accepts_str_path(fx_model, fx_config, tmp_path):
    """path is typed `str | Path`; make sure the str branch actually works,
    not just the Path branch every other test exercises."""
    out_dir = fx_model.save_checkpoint(config=fx_config, path=str(tmp_path / "ckpt_str"))
    assert isinstance(out_dir, type(tmp_path))
    assert (out_dir / "checkpoint.json").exists()


def test_save_checkpoint_default_path_is_timestamped_under_checkpoints(fx_model, fx_config, tmp_path, monkeypatch):
    """No `path` given -> falls back to checkpoints/<timestamp>, created
    relative to cwd. Isolated via monkeypatch.chdir so this never touches
    the real filesystem."""
    monkeypatch.chdir(tmp_path)
    out_dir = fx_model.save_checkpoint(config=fx_config)
    assert out_dir.parent.name == "checkpoints"
    assert out_dir.exists()
    # "%Y-%m-%d_%H-%M-%S"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", out_dir.name), out_dir.name


def test_save_checkpoint_creates_parent_directories(fx_model, fx_config, tmp_path):
    """path.mkdir(parents=True, ...) should create arbitrarily nested,
    not-yet-existing directories in one call."""
    nested = tmp_path / "a" / "b" / "c" / "ckpt"
    out_dir = fx_model.save_checkpoint(config=fx_config, path=nested)
    assert out_dir.exists()


# =============================================================================
# B. checkpoint.json contents
# =============================================================================

def test_checkpoint_json_default_fields(fx_model, fx_config, tmp_path):
    """schema_version, model_type, has_opt_state, has_activities, and a
    default empty metadata dict, all as expected when nothing optional is
    passed in."""
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt")
    data = json.loads((out_dir / "checkpoint.json").read_text())

    assert data["schema_version"] == DummyModel.SCHEMA_VERSION == 1
    assert data["model_type"] == "dummy_model_base_test"
    assert data["metadata"] == {}
    assert data["has_opt_state"] is False
    assert data["has_activities"] is False
    assert data["config"] == {"size": DUMMY_SIZE}


def test_checkpoint_json_created_at_is_valid_isoformat(fx_model, fx_config, tmp_path):
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt")
    data = json.loads((out_dir / "checkpoint.json").read_text())
    # should not raise
    datetime.fromisoformat(data["created_at"])


def test_checkpoint_json_custom_metadata_round_trips(fx_model, fx_config, tmp_path):
    metadata = {"epoch": 12, "note": "pre-annealing", "nested": {"lr": 0.01}}
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt", metadata=metadata)
    loaded = DummyModel.load_checkpoint(out_dir)
    assert loaded.metadata == metadata


# =============================================================================
# B2. config resolution -- explicit config= vs self.config fallback vs
#     neither being available. This is the contract that changed when
#     TpchModel started storing its config as `self.config`: save_checkpoint
#     now does `config = config if config is not None else getattr(self,
#     "config", None)`, so callers of a model that stores self.config no
#     longer need to pass config= at all.
# =============================================================================

def test_save_checkpoint_uses_self_config_when_not_passed(fx_model_with_self_config, tmp_path):
    """config= can be omitted entirely when the model stores self.config --
    save_checkpoint should fall back to getattr(self, "config", None)
    rather than requiring it explicitly."""
    out_dir = fx_model_with_self_config.save_checkpoint(path=tmp_path / "ckpt")
    data = json.loads((out_dir / "checkpoint.json").read_text())
    assert data["config"] == {"size": DUMMY_SIZE}

    loaded = DummyModelWithSelfConfig.load_checkpoint(out_dir)
    assert isinstance(loaded.model, DummyModelWithSelfConfig)


def test_save_checkpoint_explicit_config_overrides_self_config(fx_model_with_self_config, tmp_path):
    """An explicitly-passed config= should win over self.config, per
    save_checkpoint's documented `config if config is not None else
    self.config` precedence."""
    override = DummyConfig(size=DUMMY_SIZE * 2)
    out_dir = fx_model_with_self_config.save_checkpoint(config=override, path=tmp_path / "ckpt")
    data = json.loads((out_dir / "checkpoint.json").read_text())
    assert data["config"] == {"size": DUMMY_SIZE * 2}


def test_save_checkpoint_raises_when_no_config_available(fx_model, tmp_path):
    """DummyModel has neither self.config nor an explicit config= here --
    save_checkpoint should fail with a clear ValueError rather than
    crashing deeper in the save path (e.g. on asdict(None))."""
    with pytest.raises(ValueError):
        fx_model.save_checkpoint(path=tmp_path / "ckpt")


# =============================================================================
# C. Weight round-trip -- the actual point of checkpointing
# =============================================================================

def test_load_checkpoint_weights_match_saved_model(fx_model, fx_config, tmp_path):
    """The array leaves of the loaded model should be bit-for-bit identical
    to the saved model's -- not just 'close', since this is a lossless
    binary serialisation round trip."""
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt")
    loaded = DummyModel.load_checkpoint(out_dir)

    orig_leaves = jax.tree_util.tree_leaves(eqx.filter(fx_model, eqx.is_array))
    loaded_leaves = jax.tree_util.tree_leaves(eqx.filter(loaded.model, eqx.is_array))
    assert len(orig_leaves) == len(loaded_leaves)
    for o, l in zip(orig_leaves, loaded_leaves):
        assert_allclose(o, l, "checkpoint weight leaf", atol=0.0, rtol=0.0)


def test_load_checkpoint_reproduces_identical_predictions(fx_model, fx_config, tmp_path):
    """End-to-end behavioural check: calling predict() on the loaded model
    with the same input gives the same output as the original -- this is
    what actually matters to a caller, independent of pytree internals."""
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt")
    loaded = DummyModel.load_checkpoint(out_dir)

    x = jr.normal(jr.key(99), (DUMMY_SIZE,))
    assert_allclose(fx_model.predict(x), loaded.model.predict(x), "predict() before vs after round trip")


def test_load_checkpoint_returns_named_tuple_fields(fx_model, fx_config, tmp_path):
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt")
    result = DummyModel.load_checkpoint(out_dir)
    assert isinstance(result.model, DummyModel)
    assert result.metadata == {}
    assert result.opt_state is None
    assert result.activities is None


def test_load_checkpoint_dispatches_via_registry_regardless_of_caller_class(fx_model, fx_config, tmp_path):
    """load_checkpoint reassigns `cls` from MODEL_REGISTRY[checkpoint['model_type']]
    internally, so it should return the correct concrete subclass even when
    called on a totally unrelated ModelBase subclass (or ModelBase itself).
    This is what lets a generic 'load whatever is at this path' caller work
    without knowing the model type ahead of time."""
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt")

    via_dummy = DummyModel.load_checkpoint(out_dir)
    via_unrelated = TpchModel.load_checkpoint(out_dir)
    via_base = ModelBase.load_checkpoint(out_dir)

    assert isinstance(via_dummy.model, DummyModel)
    assert isinstance(via_unrelated.model, DummyModel)
    assert isinstance(via_base.model, DummyModel)


# =============================================================================
# D. opt_state save/load
# =============================================================================

def test_save_load_opt_state_round_trips(fx_model, fx_config, tmp_path):
    optim = optax.adam(learning_rate=1e-3)
    opt_state = optim.init(eqx.filter(fx_model, eqx.is_array))

    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt", opt_state=opt_state)
    assert (out_dir / "opt_state.eqx").exists()

    data = json.loads((out_dir / "checkpoint.json").read_text())
    assert data["has_opt_state"] is True

    loaded = DummyModel.load_checkpoint(out_dir, optim=optim)
    assert loaded.opt_state is not None
    # opt_state pytree structure should match a freshly-initialised one
    assert jax.tree_util.tree_structure(loaded.opt_state) == jax.tree_util.tree_structure(opt_state)


def test_load_checkpoint_with_saved_opt_state_but_no_optim_raises(fx_model, fx_config, tmp_path):
    """Documented contract: if the checkpoint has opt_state, load_checkpoint
    must be given the matching optax transform via `optim=`, or it should
    fail loudly rather than silently dropping the optimiser state."""
    optim = optax.sgd(learning_rate=0.1)
    opt_state = optim.init(eqx.filter(fx_model, eqx.is_array))
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt", opt_state=opt_state)

    with pytest.raises(ValueError):
        DummyModel.load_checkpoint(out_dir)


# =============================================================================
# E. activities save/load
# =============================================================================

def test_save_load_activities_round_trip(fx_model, fx_config, tmp_path):
    activities = [jr.normal(jr.key(5), (DUMMY_SIZE,))]
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt", activities=activities)
    assert (out_dir / "activities.eqx").exists()

    loaded = DummyModel.load_checkpoint(out_dir)
    assert loaded.activities is not None
    assert_allclose(loaded.activities[0], activities[0], "loaded activities")


def test_load_checkpoint_activities_accepts_explicit_skeleton_override(fx_model, fx_config, tmp_path):
    """`activities_skeleton=` should be honoured instead of falling back to
    `cls.zero_activities(config)` when the caller supplies one."""
    activities = [jr.normal(jr.key(6), (DUMMY_SIZE,))]
    out_dir = fx_model.save_checkpoint(config=fx_config, path=tmp_path / "ckpt", activities=activities)

    custom_skeleton = [jnp.zeros(DUMMY_SIZE)]
    loaded = DummyModel.load_checkpoint(out_dir, activities_skeleton=custom_skeleton)
    assert_allclose(loaded.activities[0], activities[0], "loaded activities via explicit skeleton")


def test_save_checkpoint_activities_fails_fast_for_unsupported_model(tmp_path):
    """save_checkpoint should call zero_activities BEFORE touching the
    filesystem, so an unsupported model raises NotImplementedError and
    leaves no partial checkpoint directory behind."""
    config = DummyConfig(size=DUMMY_SIZE)
    model = DummyModelNoActivities.from_config(config, key=jr.key(1))
    target = tmp_path / "ckpt"

    with pytest.raises(NotImplementedError):
        model.save_checkpoint(config=config, path=target, activities=[jnp.zeros(DUMMY_SIZE)])

    assert not target.exists()


def test_zero_activities_default_raises_for_unsupported_model():
    """Direct check of the documented fallback behaviour on ModelBase
    itself (independent of save_checkpoint's fail-fast wrapping above)."""
    with pytest.raises(NotImplementedError):
        DummyModelNoActivities.zero_activities(DummyConfig(size=DUMMY_SIZE))


# =============================================================================
# F. Model registry sanity (relied on by load_checkpoint's dispatch)
# =============================================================================

def test_model_registry_contains_registered_types():
    assert MODEL_REGISTRY["dummy_model_base_test"] is DummyModel
    assert MODEL_REGISTRY["dummy_model_base_test_with_self_config"] is DummyModelWithSelfConfig
    assert MODEL_REGISTRY["tpch"] is TpchModel


def test_duplicate_model_type_with_different_class_raises():
    """__init_subclass__ should refuse to silently let a second, distinct
    class steal an already-registered model_type."""
    with pytest.raises(ValueError):
        class ClashingModel(eqx.Module, ModelBase):
            model_type = "dummy_model_base_test"  # already taken by DummyModel
            config_cls = DummyConfig

            def predict(self, x):
                ...

            @classmethod
            def from_config(cls, config, *, key):
                ...


def test_subclass_missing_required_classvar_raises():
    with pytest.raises(TypeError):
        class MissingConfigCls(eqx.Module, ModelBase):
            model_type = "dummy_missing_config_cls"

            def predict(self, x):
                ...

            @classmethod
            def from_config(cls, config, *, key):
                ...


