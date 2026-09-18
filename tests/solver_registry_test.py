"""solver_registry_test.py

Tests for utils/solver_registry.py.
"""

import diffrax
import pytest

from utils.solver_registry import SOLVER_REGISTRY, build_solver


# Every registered constructor is exercised below. Keeping this mapping
# explicit catches both stale names and mislabeled entries (e.g. a key that
# accidentally points at a different solver class).
EXPECTED_SOLVERS = {
    "euler": diffrax.Euler,
    "heun": diffrax.Heun,
    "midpoint": diffrax.Midpoint,
    "ralston": diffrax.Ralston,
    "bosh3": diffrax.Bosh3,
    "tsit5": diffrax.Tsit5,
    "dopri5": diffrax.Dopri5,
    "dopri8": diffrax.Dopri8,
    "impliciteuler": diffrax.ImplicitEuler,
    "kvaerno3": diffrax.Kvaerno3,
    "kvaerno4": diffrax.Kvaerno4,
    "kvaerno5": diffrax.Kvaerno5,
    "sil3": diffrax.Sil3,
    "kencarp3": diffrax.KenCarp3,
    "kencarp4": diffrax.KenCarp4,
    "kencarp5": diffrax.KenCarp5,
    "semiimpliciteuler": diffrax.SemiImplicitEuler,
    "reversibleheun": diffrax.ReversibleHeun,
    "leapfrogmidpoint": diffrax.LeapfrogMidpoint,
}


@pytest.mark.parametrize("name", sorted(SOLVER_REGISTRY))
def test_every_registered_solver_is_constructible(name):
    """Every registered solver should instantiate without error.

    All constructors currently registered here have defaults sufficient for
    a no-argument construction. This deliberately exercises the registry
    itself rather than a full ODE solve, so a typo/stale Diffrax class name is
    caught immediately and cheaply.
    """
    solver = SOLVER_REGISTRY[name]()
    assert isinstance(solver, diffrax.AbstractSolver)


def test_build_solver_happy_path_matches_direct_call():
    """build_solver('heun') should construct the same solver class as the
    corresponding direct Diffrax constructor."""
    via_registry = build_solver("heun")
    direct = diffrax.Heun()

    assert type(via_registry) is type(direct)
    assert isinstance(via_registry, diffrax.AbstractSolver)


def test_build_solver_forwards_kwargs():
    """A non-default solver kwarg should actually reach the constructor.

    ``scan_kind`` is a constructor argument on Tsit5; comparing the resulting
    field makes this test stronger than merely checking that construction did
    not raise.
    """
    kwargs = {"scan_kind": "bounded"}
    via_registry = build_solver("tsit5", **kwargs)
    direct = diffrax.Tsit5(**kwargs)

    assert type(via_registry) is type(direct)
    assert via_registry.scan_kind == direct.scan_kind == "bounded"


def test_build_solver_raises_key_error_for_unregistered_name():
    with pytest.raises(KeyError):
        build_solver("not_a_real_solver")


def test_build_solver_error_message_hints_at_manual_registration():
    with pytest.raises(KeyError, match="SOLVER_REGISTRY\\['not_a_real_solver'\\] = "):
        build_solver("not_a_real_solver")


def test_custom_solver_can_be_registered_and_used():
    """Documents/guards the extensibility path mentioned in the module
    docstring and error message: callers can register their own solver."""

    def my_solver():
        return diffrax.Heun()

    SOLVER_REGISTRY["my_custom_solver_for_test"] = my_solver
    try:
        solver = build_solver("my_custom_solver_for_test")
        assert isinstance(solver, diffrax.Heun)
    finally:
        # Don't leak this into other tests.
        del SOLVER_REGISTRY["my_custom_solver_for_test"]


def test_registry_keys_are_unique_and_expected_count():
    """Loose sanity check -- catches accidental duplicate keys in the dict
    literal and keeps the intended registry size visible in the tests."""
    assert len(SOLVER_REGISTRY) == 19
    assert len(set(SOLVER_REGISTRY)) == len(SOLVER_REGISTRY)


def test_registry_entries_match_the_identically_documented_solver_class():
    """Every registry key should point to the solver class documented for it.

    This catches a mislabeled entry that the constructibility test alone would
    miss: e.g. ``"heun": diffrax.Euler`` would still instantiate perfectly.
    """
    assert SOLVER_REGISTRY == EXPECTED_SOLVERS
