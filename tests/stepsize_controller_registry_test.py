"""stepsize_controller_registry_test.py

Tests for utils/stepsize_controller_registry.py.
"""

import diffrax
import jax.numpy as jnp
import pytest

from utils.stepsize_controller_registry import (
    STEPSIZE_CONTROLLER_REGISTRY,
    build_stepsize_controller,
)


# The registry intentionally provides a couple of metadata-friendly aliases
# for the same Diffrax class, so the expected mapping documents both the keys
# and the exact constructors they are meant to resolve to.
EXPECTED_CONTROLLERS = {
    "constant": diffrax.ConstantStepSize,
    "constantstepsize": diffrax.ConstantStepSize,
    "stepto": diffrax.StepTo,
    "pid": diffrax.PIDController,
    "pidcontroller": diffrax.PIDController,
    "clip": diffrax.ClipStepSizeController,
    "clipstepsizecontroller": diffrax.ClipStepSizeController,
}


def _construct_registered_controller(name):
    """Construct a registry entry with minimal valid constructor arguments."""
    controller_fn = STEPSIZE_CONTROLLER_REGISTRY[name]

    if controller_fn is diffrax.ConstantStepSize:
        return controller_fn()
    if controller_fn is diffrax.StepTo:
        return controller_fn(ts=jnp.array([0.0, 1.0]))
    if controller_fn is diffrax.PIDController:
        return controller_fn(rtol=1e-3, atol=1e-6)
    if controller_fn is diffrax.ClipStepSizeController:
        inner = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        return controller_fn(controller=inner)

    raise AssertionError(f"Unhandled controller in test helper: {controller_fn!r}")


@pytest.mark.parametrize("name", sorted(STEPSIZE_CONTROLLER_REGISTRY))
def test_every_registered_stepsize_controller_is_constructible(name):
    """Every registered entry should build with a minimal valid constructor.

    Unlike solvers, some controller classes require constructor arguments
    (e.g. StepTo needs ``ts`` and PIDController needs tolerances), so the test
    supplies the smallest sensible arguments for each constructor type.
    """
    controller = _construct_registered_controller(name)
    assert isinstance(controller, diffrax.AbstractStepSizeController)


def test_build_stepsize_controller_happy_path_matches_direct_call():
    """build_stepsize_controller('pid', **kwargs) should match direct use of
    diffrax.PIDController(**kwargs) for the same configuration."""
    kwargs = {"rtol": 1e-5, "atol": 1e-7}
    via_registry = build_stepsize_controller("pid", **kwargs)
    direct = diffrax.PIDController(**kwargs)

    assert type(via_registry) is type(direct)
    assert via_registry.rtol == direct.rtol == kwargs["rtol"]
    assert via_registry.atol == direct.atol == kwargs["atol"]


def test_build_stepsize_controller_forwards_kwargs():
    """A non-default StepTo argument should reach the underlying constructor."""
    ts = jnp.array([0.0, 0.25, 0.75, 1.0])

    via_registry = build_stepsize_controller("stepto", ts=ts)
    direct = diffrax.StepTo(ts=ts)

    assert type(via_registry) is type(direct)
    assert jnp.array_equal(via_registry.ts, direct.ts)


def test_build_stepsize_controller_can_construct_clip_wrapper():
    """Wrapper-controller kwargs, including the wrapped controller, should
    pass through unchanged."""
    inner = diffrax.PIDController(rtol=1e-4, atol=1e-8)
    via_registry = build_stepsize_controller(
        "clip",
        controller=inner,
    )

    assert isinstance(via_registry, diffrax.ClipStepSizeController)
    assert via_registry.controller is inner


def test_build_stepsize_controller_raises_key_error_for_unregistered_name():
    with pytest.raises(KeyError):
        build_stepsize_controller("not_a_real_stepsize_controller")


def test_build_stepsize_controller_error_message_hints_at_manual_registration():
    with pytest.raises(
        KeyError,
        match="STEPSIZE_CONTROLLER_REGISTRY\\['not_a_real_stepsize_controller'\\] = ",
    ):
        build_stepsize_controller("not_a_real_stepsize_controller")


def test_custom_stepsize_controller_can_be_registered_and_used():
    """Documents/guards the extensibility path mentioned in the module
    docstring and error message: callers can register their own controller."""

    def my_controller():
        return diffrax.ConstantStepSize()

    STEPSIZE_CONTROLLER_REGISTRY["my_custom_controller_for_test"] = my_controller
    try:
        controller = build_stepsize_controller("my_custom_controller_for_test")
        assert isinstance(controller, diffrax.ConstantStepSize)
    finally:
        # Don't leak this into other tests.
        del STEPSIZE_CONTROLLER_REGISTRY["my_custom_controller_for_test"]


def test_registry_keys_are_unique_and_expected_count():
    """Loose sanity check -- catches accidental duplicate keys in the dict
    literal and keeps the intended registry size visible in the tests."""
    assert len(STEPSIZE_CONTROLLER_REGISTRY) == 7
    assert len(set(STEPSIZE_CONTROLLER_REGISTRY)) == len(STEPSIZE_CONTROLLER_REGISTRY)


def test_registry_entries_match_the_expected_diffrax_classes():
    """Check the registry against an independent expected mapping.

    The constructibility tests alone cannot detect a mislabeled entry such as
    ``"pid": diffrax.ConstantStepSize`` because the wrong class would still
    instantiate successfully.
    """
    assert STEPSIZE_CONTROLLER_REGISTRY == EXPECTED_CONTROLLERS
