"""optim_registry_test.py

Tests for utils/optim_registry.py.
"""

import optax
import pytest

from utils.optim_registry import OPTIM_REGISTRY, build_optim


# A representative subset gets exercised individually below; this list
# instead sweeps ALL registered entries to catch typos/renames in the
# registry itself (e.g. a key that doesn't match any real optax function).
@pytest.mark.parametrize("name", sorted(OPTIM_REGISTRY))
def test_every_registered_optim_is_constructible(name):
    """Every entry should build without error using whatever its own
    required args are. Most optax optimisers accept `learning_rate=`
    positionally-or-by-keyword; a couple (e.g. polyak_sgd, lbfgs) don't
    require it at all, so we fall back to a no-arg build for those."""
    optim_fn = OPTIM_REGISTRY[name]
    try:
        optim = optim_fn(learning_rate=1e-3)
    except TypeError:
        optim = optim_fn()
    assert hasattr(optim, "init") and hasattr(optim, "update")


def test_build_optim_happy_path_matches_direct_call():
    """build_optim('adamw', **kwargs) should produce the same transform
    optax.adamw(**kwargs) would -- same pytree structure when init'd on
    a trivial param tree."""
    kwargs = {"learning_rate": 1e-3, "weight_decay": 1e-4}
    via_registry = build_optim("adamw", **kwargs)
    direct = optax.adamw(**kwargs)

    params = {"w": 0.0}
    assert (
        __import__("jax").tree_util.tree_structure(via_registry.init(params))
        == __import__("jax").tree_util.tree_structure(direct.init(params))
    )


def test_build_optim_forwards_kwargs():
    """A non-default kwarg should actually reach the optimiser, not get
    silently dropped -- checked via the resulting opt_state, since
    optax.GradientTransformation itself doesn't expose its config back."""
    default_state = optax.sgd(learning_rate=1e-3).init({"w": 0.0})
    momentum_state = build_optim("sgd", learning_rate=1e-3, momentum=0.9).init({"w": 0.0})
    # momentum=0.9 adds a trace/momentum term the plain SGD init doesn't have
    assert __import__("jax").tree_util.tree_structure(default_state) != (
        __import__("jax").tree_util.tree_structure(momentum_state)
    )


def test_build_optim_raises_key_error_for_unregistered_name():
    with pytest.raises(KeyError):
        build_optim("not_a_real_optim", learning_rate=1e-3)


def test_build_optim_error_message_hints_at_manual_registration():
    with pytest.raises(KeyError, match="OPTIM_REGISTRY\\['not_a_real_optim'\\] = "):
        build_optim("not_a_real_optim", learning_rate=1e-3)


def test_custom_optim_can_be_registered_and_used():
    """Documents/guards the extensibility path mentioned in the module
    docstring and error message: callers can register their own."""

    def my_optim(learning_rate):
        return optax.sgd(learning_rate)

    OPTIM_REGISTRY["my_custom_optim_for_test"] = my_optim
    try:
        optim = build_optim("my_custom_optim_for_test", learning_rate=1e-3)
        assert hasattr(optim, "init") and hasattr(optim, "update")
    finally:
        # don't leak this into other tests
        del OPTIM_REGISTRY["my_custom_optim_for_test"]


def test_registry_keys_are_unique_and_expected_count():
    """Loose sanity check -- catches an accidental duplicate key (which
    dict literals silently allow, just overwriting the earlier one) more
    than it enforces a specific count."""
    assert len(OPTIM_REGISTRY) == 28
    assert len(set(OPTIM_REGISTRY)) == len(OPTIM_REGISTRY)
