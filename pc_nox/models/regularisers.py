"""regularisers.py

Pure penalty math shared by every model variant: given a list of weight
arrays (or states), compute an L2, orthogonality, or L1/L2 activity
penalty. Deliberately knows nothing about which weights belong to which
scope/layer -- that selection is genuinely variant-specific (it depends on
which fields exist on which layer classes) and stays in each model's own
`_weights_for_scope`/`_weight_entries`. Only the arithmetic moves here.
"""
from typing import List, Sequence
import jax.numpy as jnp
from jaxtyping import Array


def l2_reg(weights: Sequence[Array], coeff: float) -> Array:
    """0.5 * coeff * sum ||W||_F^2 over the given weights."""
    if coeff <= 0.0 or not weights:
        return jnp.asarray(0.0)
    sq_norm = sum(jnp.sum(W ** 2) for W in weights)
    return 0.5 * coeff * sq_norm


def orthogonal_reg(weights: Sequence[Array], coeff: float) -> Array:
    """0.5 * coeff * sum ||I - W^T W||_F^2 (or W W^T for a 'wide' W),
    using whichever Gram matrix is smaller per weight.
    """
    if coeff <= 0.0 or not weights:
        return jnp.asarray(0.0)
    reg = jnp.asarray(0.0)
    for W in weights:
        out_dim, in_dim = W.shape
        dim = min(out_dim, in_dim)
        gram = (W.T @ W) if in_dim <= out_dim else (W @ W.T)
        reg = reg + jnp.sum((jnp.eye(dim) - gram) ** 2)
    return 0.5 * coeff * reg


def activity_reg(states: Sequence[Array], coeff: float, kind: str = "l1") -> Array:
    """0.5 * coeff * sum ||s||_p^p over the given states, p=1 or p=2."""
    if coeff <= 0.0 or not states:
        return jnp.asarray(0.0)
    if kind == "l1":
        reg = sum(jnp.sum(jnp.abs(s)) for s in states)
    else:
        reg = sum(jnp.sum(s ** 2) for s in states)
    return 0.5 * coeff * reg


def l2_reg_by_group(groups: List[List[Array]], coeff: float) -> Array:
    """Per-group breakdown of `l2_reg`: one entry per group (possibly
    empty). `jnp.sum(...)` of the result equals `l2_reg` on the flattened
    weights.
    """
    if coeff <= 0.0:
        return jnp.zeros(len(groups))
    return jnp.stack([l2_reg(group, coeff) for group in groups])


def orthogonal_reg_by_group(groups: List[List[Array]], coeff: float) -> Array:
    """Per-group breakdown of `orthogonal_reg`."""
    if coeff <= 0.0:
        return jnp.zeros(len(groups))
    return jnp.stack([orthogonal_reg(group, coeff) for group in groups])


def activity_reg_by_layer(states: Sequence[Array], coeff: float, kind: str = "l1") -> Array:
    """Per-state breakdown of `activity_reg`: shape (len(states),)."""
    if coeff <= 0.0:
        return jnp.zeros(len(states))
    if kind == "l1":
        return jnp.stack([activity_reg([s], coeff, "l1") for s in states])
    return jnp.stack([activity_reg([s], coeff, "l2") for s in states])
