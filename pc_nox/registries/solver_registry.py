"""solver_registry.py

Name -> Diffrax solver-constructor registry, so a solver can be rebuilt
purely from a string name (e.g. one stashed in checkpoint metadata) plus
whatever kwargs it was originally built with.

The registry contains concrete ODE solver classes that implement
``diffrax.AbstractSolver``. Diffrax's abstract base classes themselves are
not registered because they cannot be instantiated directly.

Typical usage
-------------
Stash the solver name and kwargs in metadata at save time:

    solver = diffrax.Heun()

    metadata = {
        "solver_name": "heun",
        "solver_kwargs": {},
    }

and rebuild it later via:

    loaded_solver = build_solver(
        loaded.metadata["solver_name"],
        **loaded.metadata["solver_kwargs"],
    )

For solvers with constructor arguments, the kwargs are forwarded verbatim,
just like the optimiser registry. For example, implicit solvers may need
``root_finder=...`` and ``root_find_max_steps=...``.

The registry is intentionally a plain dict populated at import time and is
extensible by callers, mirroring ``OPTIM_REGISTRY``.
"""

from typing import Callable, Dict
import diffrax

SolverBuilder = Callable[..., diffrax.AbstractSolver]

# name : Diffrax solver constructor.
# Keys use the lowercase constructor name so metadata can use stable,
# human-readable names such as "heun", "tsit5", or "kvaerno5".
SOLVER_REGISTRY: Dict[str, SolverBuilder] = {
    # Explicit Runge--Kutta methods.
    "euler": diffrax.Euler,
    "heun": diffrax.Heun,
    "midpoint": diffrax.Midpoint,
    "ralston": diffrax.Ralston,
    "bosh3": diffrax.Bosh3,
    "tsit5": diffrax.Tsit5,
    "dopri5": diffrax.Dopri5,
    "dopri8": diffrax.Dopri8,

    # Implicit Runge--Kutta methods.
    "impliciteuler": diffrax.ImplicitEuler,
    "kvaerno3": diffrax.Kvaerno3,
    "kvaerno4": diffrax.Kvaerno4,
    "kvaerno5": diffrax.Kvaerno5,

    # IMEX methods.
    "sil3": diffrax.Sil3,
    "kencarp3": diffrax.KenCarp3,
    "kencarp4": diffrax.KenCarp4,
    "kencarp5": diffrax.KenCarp5,

    # Symplectic methods.
    "semiimpliciteuler": diffrax.SemiImplicitEuler,

    # Reversible methods.
    "reversibleheun": diffrax.ReversibleHeun,

    # Linear multistep methods.
    "leapfrogmidpoint": diffrax.LeapfrogMidpoint,
}


def build_solver(name: str, /, **kwargs) -> diffrax.AbstractSolver:
    """
    Build a Diffrax solver by registry name.

    Args:
        name: Key into SOLVER_REGISTRY, e.g. "heun" or "dopri5".
            Positional-only so it cannot collide with a constructor kwarg.
        **kwargs: Forwarded verbatim to the solver constructor. Most explicit
            ODE solvers take no kwargs; implicit solvers accept arguments such
            as ``root_finder`` and ``root_find_max_steps``.

    Returns:
        The constructed Diffrax solver, ready to pass to
        ``diffrax.diffeqsolve(..., solver=...)``.

    Raises:
        KeyError: if ``name`` is not registered. If it is a custom solver,
            register it before loading, for example::

                SOLVER_REGISTRY["my_solver"] = MySolver
    """
    try:
        solver_fn = SOLVER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"solver name={name!r} not in SOLVER_REGISTRY. If this is a custom "
            f"solver, register it before loading: "
            f"SOLVER_REGISTRY[{name!r}] = ..."
        ) from None
    return solver_fn(**kwargs)
