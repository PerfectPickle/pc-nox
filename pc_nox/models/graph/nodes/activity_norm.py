from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

ACTIVITY_NORM_REGISTRY: Dict[str, Callable[[Tuple[int, ...]], eqx.Module]] = {}


def register_activity_norm(name: str):
    """Decorator: register a factory `shape -> norm module` under `name`."""
    def decorator(factory):
        if name in ACTIVITY_NORM_REGISTRY:
            raise ValueError(f"activity norm {name!r} already registered")
        ACTIVITY_NORM_REGISTRY[name] = factory
        return factory
    return decorator


def make_activity_norm(kind: Optional[str], shape: Tuple[int, ...]):
    if kind is None:
        return None
    try:
        return ACTIVITY_NORM_REGISTRY[kind](shape)
    except KeyError:
        raise ValueError(
            f"unknown activity_norm {kind!r}; registered: {sorted(ACTIVITY_NORM_REGISTRY)}. "
            "A custom norm must be registered (its module imported) before building or loading."
        ) from None


# applies activity normalisation to x, if the corresponding key is in norms (i.e. it is intended to be normalised), else returns x as is.
def apply_norm(norms: Mapping[str, eqx.Module], field: str, x: Array) -> Array:
    n = norms.get(field)
    return x if n is None else n(x)


class ActivityRMSNorm(eqx.Module):
    """Divide out the RMS over ALL units, then apply a learnable gain along axis 0.

    gain: shape (D,) for a dense input, (C,) for a (C, H, W) input
          (one gain per channel, shared across pixels). No bias, as in Eq. 10.
    """
    gain: Array
    eps: float = eqx.field(static=True)      # inside the sqrt, so an all-zero input is safe

    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-5):
        self.gain = jnp.ones((shape[0],))
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        inv_rms = jax.lax.rsqrt(jnp.mean(x ** 2) + self.eps)    # mean over every unit = ||x||^2 / N
        # (C,) -> (C, 1, 1) for conv inputs; (D,) stays (D,) for dense inputs
        gain = jnp.reshape(self.gain, (-1,) + (1,) * (x.ndim - 1))
        return gain * x * inv_rms


register_activity_norm("rms")(ActivityRMSNorm)          # the class itself is the factory