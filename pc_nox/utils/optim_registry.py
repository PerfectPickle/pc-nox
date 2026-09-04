"""optim_registry.py

Name -> optax optimiser-constructor registry, so an optimiser can be
rebuilt purely from a string name (e.g. one stashed in a checkpoint's
metadata) plus whatever kwargs it was originally built with.

Mirrors the ACT_FN_REGISTRY pattern in models/model_base.py (plain dict,
populated once at import time, extensible by callers) -- but lives here
in utils/, not model_base.py, because optimiser choice is a training-loop
concern, orthogonal to model architecture. ACT_FN_REGISTRY earns its spot
in model_base.py because activation functions are baked into a model's
config/layers at construction time; optimisers never are.

Typical usage
-------------
Nothing in ModelBase.save_checkpoint knows about optimiser *names* -- it
only ever stores an already-built opt_state, keyed on `has_opt_state`.
Reconstructing the actual optax.GradientTransformation used to produce
that opt_state is left to the caller (load_checkpoint's `optim=` arg), so
if you want that to happen "automatically" from metadata, stash the name
and kwargs at save time:

    model.save_checkpoint(
        path=...,
        opt_state=opt_state,
        metadata={"optim_name": "adamw", "optim_kwargs": {"learning_rate": 1e-3, "weight_decay": 1e-4}},
    )

and rebuild it at load time via this registry:

    loaded = TpchModel.load_checkpoint(
        path,
        optim=build_optim(loaded.metadata["optim_name"], **loaded.metadata["optim_kwargs"]),
    )
"""

from typing import Callable, Dict
import optax

OptimBuilder = Callable[..., optax.GradientTransformation]

# name : optax constructor. Keys match the optax function names 1:1 so
# metadata["optim_name"] can just be str(optim_fn.__name__) if you'd rather
# not hand-write it at save time.
OPTIM_REGISTRY: Dict[str, OptimBuilder] = {
    "adabelief": optax.adabelief,
    "adadelta": optax.adadelta,
    "adan": optax.adan,
    "adafactor": optax.adafactor,
    "adagrad": optax.adagrad,
    "adam": optax.adam,
    "adamw": optax.adamw,
    "adamax": optax.adamax,
    "adamaxw": optax.adamaxw,
    "amsgrad": optax.amsgrad,
    "fromage": optax.fromage,
    "lamb": optax.lamb,
    "lars": optax.lars,
    "lbfgs": optax.lbfgs,
    "lion": optax.lion,
    "nadam": optax.nadam,
    "nadamw": optax.nadamw,
    "noisy_sgd": optax.noisy_sgd,
    "novograd": optax.novograd,
    "optimistic_gradient_descent": optax.optimistic_gradient_descent,
    "optimistic_adam": optax.optimistic_adam,
    "polyak_sgd": optax.polyak_sgd,
    "radam": optax.radam,
    "rmsprop": optax.rmsprop,
    "sgd": optax.sgd,
    "sign_sgd": optax.sign_sgd,
    "sm3": optax.sm3,
    "yogi": optax.yogi,
}


def build_optim(name: str, /, **kwargs) -> optax.GradientTransformation:
    """
    Build an optax optimiser by registry name.

    Args:
        name: key into OPTIM_REGISTRY, e.g. "adamw". Positional-only so
            it can never collide with a kwarg an optimiser happens to take
            (several accept a `name=` kwarg, so `name` can't be a normal
            keyword parameter here without ambiguity).
        **kwargs: forwarded verbatim to the constructor, e.g.
            learning_rate=1e-3, weight_decay=1e-4.

    Returns:
        The constructed optax.GradientTransformation, e.g. ready to pass
        as ModelBase.load_checkpoint's `optim=`.

    Raises:
        KeyError: if `name` isn't registered. If it's a custom optimiser,
            register it before loading:
                OPTIM_REGISTRY["my_optim"] = my_optim_fn
    """
    try:
        optim_fn = OPTIM_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"optim name={name!r} not in OPTIM_REGISTRY. If this is a custom "
            f"optimiser, register it before loading: OPTIM_REGISTRY[{name!r}] = ..."
        ) from None
    return optim_fn(**kwargs)
