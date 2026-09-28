"""stepsize_controller_registry.py

Name -> Diffrax step-size-controller constructor registry, so a controller
can be rebuilt purely from a string name plus whatever kwargs it was
originally built with.

The registry contains Diffrax's concrete step-size-controller classes:
``ConstantStepSize``, ``StepTo``, ``PIDController``, and
``ClipStepSizeController``.

Typical usage
-------------
Stash the controller name and kwargs in metadata at save time:

    metadata = {
        "stepsize_controller_name": "pid",
        "stepsize_controller_kwargs": {
            "rtol": 1e-5,
            "atol": 1e-6,
        },
    }

and rebuild it later via:

    controller = build_stepsize_controller(
        loaded.metadata["stepsize_controller_name"],
        **loaded.metadata["stepsize_controller_kwargs"],
    )

As with the optimiser and solver registries, kwargs are forwarded verbatim
to the underlying Diffrax constructor. This means wrapper controllers such
as ``ClipStepSizeController`` can also be reconstructed, provided their
``controller`` kwarg is itself supplied as a constructed controller.

The registry is intentionally a plain dict populated at import time and is
extensible by callers.
"""

from typing import Callable, Dict
import diffrax

StepSizeControllerBuilder = Callable[
    ..., diffrax.AbstractStepSizeController
]

# name : Diffrax step-size-controller constructor.
# Keys are lowercase, stable metadata-friendly names rather than the exact
# class spelling used by Python.
STEPSIZE_CONTROLLER_REGISTRY: Dict[str, StepSizeControllerBuilder] = {
    "constant": diffrax.ConstantStepSize,
    "constantstepsize": diffrax.ConstantStepSize,
    "stepto": diffrax.StepTo,
    "pid": diffrax.PIDController,
    "pidcontroller": diffrax.PIDController,
    "clip": diffrax.ClipStepSizeController,
    "clipstepsizecontroller": diffrax.ClipStepSizeController,
}


def build_stepsize_controller(
    name: str, /, **kwargs
) -> diffrax.AbstractStepSizeController:
    """
    Build a Diffrax step-size controller by registry name.

    Args:
        name: Key into STEPSIZE_CONTROLLER_REGISTRY, e.g. "constant", "pid",
            "stepto", or "clip". Positional-only so it cannot collide with
            any constructor kwarg named ``name``.
        **kwargs: Forwarded verbatim to the controller constructor. For
            example, ``PIDController`` accepts ``rtol``/``atol`` and
            ``StepTo`` accepts ``ts``.

    Returns:
        The constructed Diffrax step-size controller, ready to pass to
        ``diffrax.diffeqsolve(..., stepsize_controller=...)``.

    Raises:
        KeyError: if ``name`` is not registered. If it is a custom controller,
            register it before loading, for example::

                STEPSIZE_CONTROLLER_REGISTRY["my_controller"] = MyController
    """
    try:
        controller_fn = STEPSIZE_CONTROLLER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"step-size controller name={name!r} not in "
            f"STEPSIZE_CONTROLLER_REGISTRY. If this is a custom controller, "
            f"register it before loading: "
            f"STEPSIZE_CONTROLLER_REGISTRY[{name!r}] = ..."
        ) from None
    return controller_fn(**kwargs)
