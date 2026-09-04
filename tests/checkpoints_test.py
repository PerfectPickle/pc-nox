"""checkpoints_test.py

Tests for utils/checkpoints.py: find_latest_checkpoint and load_metadata.

We don't need real models here -- neither function touches anything but
checkpoint.json (find_latest_checkpoint reads "model_type" / "created_at",
load_metadata reads "metadata"), so the checkpoint dirs below are
hand-written fixtures rather than real save_checkpoint() output. Keeps
these tests fast and independent of model_base.py.
"""

import json

import pytest

from utils.checkpoints import find_latest_checkpoint, load_metadata


def make_checkpoint_dir(root, name, *, model_type="tpch", created_at="2026-01-01T00:00:00", metadata=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "checkpoint.json").write_text(json.dumps({
        "model_type": model_type,
        "created_at": created_at,
        "metadata": metadata if metadata is not None else {},
    }))
    return d


def test_returns_most_recent_by_created_at(tmp_path):
    make_checkpoint_dir(tmp_path, "a", created_at="2026-01-01T00:00:00")
    newest = make_checkpoint_dir(tmp_path, "b", created_at="2026-06-15T12:30:00")
    make_checkpoint_dir(tmp_path, "c", created_at="2026-03-01T00:00:00")

    result = find_latest_checkpoint(tmp_path)
    assert result == newest


def test_directory_name_is_irrelevant_only_created_at_matters(tmp_path):
    """A lexically 'earlier' directory name with a later created_at should
    still win -- guards against accidentally sorting by path instead of
    the timestamp inside checkpoint.json."""
    newest = make_checkpoint_dir(tmp_path, "aaa_first_alphabetically", created_at="2026-12-31T00:00:00")
    make_checkpoint_dir(tmp_path, "zzz_last_alphabetically", created_at="2026-01-01T00:00:00")

    result = find_latest_checkpoint(tmp_path)
    assert result == newest


def test_filters_by_model_type(tmp_path):
    make_checkpoint_dir(tmp_path, "tpch_old", model_type="tpch", created_at="2026-01-01T00:00:00")
    tpch_newest = make_checkpoint_dir(tmp_path, "tpch_new", model_type="tpch", created_at="2026-06-01T00:00:00")
    make_checkpoint_dir(tmp_path, "other_newest", model_type="some_other_model", created_at="2026-12-01T00:00:00")

    result = find_latest_checkpoint(tmp_path, model_type="tpch")
    assert result == tpch_newest


def test_model_type_none_considers_all_checkpoints(tmp_path):
    make_checkpoint_dir(tmp_path, "tpch_ckpt", model_type="tpch", created_at="2026-01-01T00:00:00")
    newest = make_checkpoint_dir(tmp_path, "other_ckpt", model_type="some_other_model", created_at="2026-12-01T00:00:00")

    result = find_latest_checkpoint(tmp_path, model_type=None)
    assert result == newest


def test_raises_file_not_found_when_root_has_no_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_checkpoint(tmp_path)


def test_raises_file_not_found_when_model_type_matches_nothing(tmp_path):
    make_checkpoint_dir(tmp_path, "tpch_ckpt", model_type="tpch", created_at="2026-01-01T00:00:00")
    with pytest.raises(FileNotFoundError):
        find_latest_checkpoint(tmp_path, model_type="nonexistent_model_type")


def test_ignores_non_checkpoint_directories_and_stray_files(tmp_path):
    """Directories without a checkpoint.json, and plain files sitting
    directly under root, should both be silently skipped rather than
    crashing the scan."""
    valid = make_checkpoint_dir(tmp_path, "valid_ckpt", created_at="2026-01-01T00:00:00")
    (tmp_path / "not_a_checkpoint_dir").mkdir()
    (tmp_path / "readme.txt").write_text("not a checkpoint")

    result = find_latest_checkpoint(tmp_path)
    assert result == valid


def test_accepts_str_root(tmp_path):
    valid = make_checkpoint_dir(tmp_path, "only_ckpt", created_at="2026-01-01T00:00:00")
    result = find_latest_checkpoint(str(tmp_path))
    assert result == valid


def test_default_root_is_checkpoints_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    valid = make_checkpoint_dir(tmp_path / "checkpoints", "only_ckpt", created_at="2026-01-01T00:00:00")

    result = find_latest_checkpoint()
    assert result.resolve() == valid.resolve()


# =============================================================================
# load_metadata
# =============================================================================

def test_load_metadata_returns_saved_metadata_dict(tmp_path):
    metadata = {"epoch": 12, "optim_name": "adamw", "optim_kwargs": {"learning_rate": 1e-3}}
    d = make_checkpoint_dir(tmp_path, "ckpt", metadata=metadata)

    assert load_metadata(d) == metadata


def test_load_metadata_defaults_to_empty_dict(tmp_path):
    """Mirrors save_checkpoint's own default (metadata or {})."""
    d = make_checkpoint_dir(tmp_path, "ckpt")  # no metadata= passed
    assert load_metadata(d) == {}


def test_load_metadata_does_not_require_weights_or_opt_state_files(tmp_path):
    """The whole point of load_metadata existing separately from
    load_checkpoint: it should work against a checkpoint dir that has
    only checkpoint.json -- no weights.eqx, opt_state.eqx, or
    activities.eqx need to exist."""
    metadata = {"note": "metadata-only checkpoint dir"}
    d = make_checkpoint_dir(tmp_path, "ckpt", metadata=metadata)
    assert not (d / "weights.eqx").exists()

    assert load_metadata(d) == metadata


def test_load_metadata_accepts_str_path(tmp_path):
    metadata = {"epoch": 3}
    d = make_checkpoint_dir(tmp_path, "ckpt", metadata=metadata)
    assert load_metadata(str(d)) == metadata


def test_load_metadata_raises_when_checkpoint_json_missing(tmp_path):
    empty_dir = tmp_path / "not_a_checkpoint"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_metadata(empty_dir)
