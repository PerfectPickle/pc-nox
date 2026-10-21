"""graph/nodes/norm.py

Activity normalisation: rescaling the activities (states) that feed into a node's
projection, so their size stays under control as networks get deep (the paper's
Eq. 10: mu = phi(W @ RMSNorm(z))). Not to be confused with `weight_norm.py`, which
rescales the WEIGHTS after each optimiser step. This rescales the ACTIVITIES, on
every forward computation.

THE KEY IDEA: every normalisation splits into two parts, handled in different places.

  1. The STATISTIC: a pure function with no parameters that rescales an activity
     vector using a REDUCTION (a quantity computed over many units at once, such as
     the RMS, mean or variance). Example, RMS norm:  x -> x / RMS(x).
     It is applied by the MODEL, in `GraphModel._inputs`, because only the model
     knows where one parent's activities end and the next begin. That matters for a
     node with several parents: the model can normalise each parent separately
     ("per_source" mode) before concatenating them, or the whole concatenation at
     once ("joint" mode). A node only ever sees the concatenated vector, so it
     cannot do the per-parent version itself.

  2. The AFFINE part: a learnable per-unit gain (and, for some norms, a bias).
     "Affine" just means "multiply, then add": y = gain * x + bias. It is applied
     by the NODE, which owns all learnable parameters. Because it works unit by unit
     (no reduction), applying it to a concatenation of parents is mathematically
     identical to applying separate gains to each parent. That is why the node can
     own it safely without knowing where the parents' boundaries are.

Together the two parts give exactly the standard formula, e.g. for RMS norm:
    gamma * x / RMS(x)    =   gain(rms_normalise(x))

Adding a new norm = write its statistic function (and, if it has learnable
parameters, an affine module), then register both below.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


# --------------------------------------------------------------------------- #
# Part 1: statistics (pure functions, no parameters; applied by the MODEL)
# --------------------------------------------------------------------------- #

def rms_normalise(x: Array, eps: float = 1e-5) -> Array:
    """Divide `x` by its root-mean-square: x / sqrt(mean(x^2) + eps).

    The mean runs over ALL units of `x` (for a (C, H, W) array, over every
    channel and pixel together), so only the direction of `x` survives and its
    overall size is divided out. `eps` sits INSIDE the square root so that an
    all-zero input gives zeros instead of a division by zero. Works on one
    example (no batch statistics), so it is safe under vmap/scan/grad.
    """
    return x * jax.lax.rsqrt(jnp.mean(x ** 2) + eps)    # rsqrt(v) = 1 / sqrt(v)


# --------------------------------------------------------------------------- #
# Part 2: affine modules (learnable, elementwise; owned by the NODE)
# --------------------------------------------------------------------------- #

class Gain(eqx.Module):
    """A learnable per-unit (or per-channel) scale: y = gain * x.

    This is the "affine" part of RMS norm (the paper's gamma). It has no bias, as
    in Eq. 10. It starts at 1, so at initialisation it changes nothing.

    Shape of `gain`:
      * dense input (D,)        -> one gain per unit, shape (D,)
      * conv input (C, H, W)    -> one gain per CHANNEL, shape (C,), shared across
        all pixels (the usual convention for conv layers; it keeps the parameter
        count independent of image size).

    It is elementwise (each output depends only on the matching input), so applying
    it to a concatenation of several parents equals applying separate gains to each.
    """
    gain: Array

    def __init__(self, shape: Tuple[int, ...]):
        self.gain = jnp.ones((shape[0],))

    def __call__(self, x: Array) -> Array:
        # Reshape (C,) -> (C, 1, 1) for conv inputs so it broadcasts across H and W;
        # for dense inputs (D,) the shape is unchanged.
        return jnp.reshape(self.gain, (-1,) + (1,) * (x.ndim - 1)) * x


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ActivityNorm:
    """One registered normalisation: its two halves.

    Attributes:
        stat: The parameter-free statistic, `array -> array` of the same shape.
            Called by the model in `GraphModel._inputs`.
        affine: Factory `shape -> module` building the node's learnable part for an
            input of that (combined) shape, e.g. `Gain`. None means the norm has no
            learnable parameters (e.g. a plain L2 unit-norm): only `stat` applies.
    """
    stat: Callable[[Array], Array]
    affine: Optional[Callable[[Tuple[int, ...]], eqx.Module]] = None


ACTIVITY_NORM_REGISTRY: Dict[str, ActivityNorm] = {
    "rms": ActivityNorm(stat=rms_normalise, affine=Gain),
}


def register_activity_norm(name: str, entry: ActivityNorm) -> None:
    """Add a custom norm. Must run (i.e. its module must be imported) before a model
    that uses it is built or a checkpoint that uses it is loaded."""
    if name in ACTIVITY_NORM_REGISTRY or name == "none":
        raise ValueError(f"activity norm {name!r} already registered (or reserved)")
    ACTIVITY_NORM_REGISTRY[name] = entry


def resolve_activity_norm(name: str) -> Optional[ActivityNorm]:
    """Look up a norm by its config name. "none" means off and returns None."""
    if name == "none":
        return None
    try:
        return ACTIVITY_NORM_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"activity_norm={name!r} not registered; registered: {sorted(ACTIVITY_NORM_REGISTRY)}. "
            "If this is a custom norm, register it (see register_activity_norm) before building or loading."
        ) from None


# --------------------------------------------------------------------------- #
# Helpers used by the nodes
# --------------------------------------------------------------------------- #

def make_activity_norms(
    activity_norm: str,
    norm_inputs: Tuple[str, ...],
    sizes: Mapping[str, Optional[Tuple[int, ...]]],
) -> Dict[str, eqx.Module]:
    """Build a node's affine modules, one per normalised input.

    Called in a node's `__init__`. The returned dict is stored on the node as
    `self.norms`; its KEYS are exactly the `NodeInputs` fields this node normalises,
    which is how the model knows to apply the statistic to them (see
    `NodeBase.normed_inputs`).

    Args:
        activity_norm: Registered norm name, or "none" (returns an empty dict).
        norm_inputs: The config's scope: which `NodeInputs` field names to normalise
            (e.g. ("parents_curr", "children_curr")).
        sizes: For each `NodeInputs` field this node can have, the shape of that
            input AFTER the model has combined its parents/children, or None if
            the node has no such input (e.g. a root node has no "parents_curr").
            Fields that are None or outside `norm_inputs` get no entry.
    """
    entry = resolve_activity_norm(activity_norm)
    if entry is None:
        return {}
    norms = {}
    for field, shape in sizes.items():
        if shape is None or field not in norm_inputs:
            continue
        # A norm with no learnable part still needs a key in the dict, because the key
        # is what tells the model to apply the statistic. Identity is a do-nothing,
        # parameter-free placeholder module (no leaves, so nothing to train or save).
        norms[field] = entry.affine(shape) if entry.affine is not None else eqx.nn.Identity()
    return norms


def apply_affine(norms: Mapping[str, eqx.Module], field: str, x: Array) -> Array:
    """Apply the node's learnable part to input `x` named `field`, if it has one.

    The node calls this inside `predict`, just before the projection:
        x = apply_affine(self.norms, "parents_curr", inp.parents_curr)
        pre = self.op_td(x)
    By then the model has already applied the statistic, so `x` is the
    normalised input and this only rescales it by the learned gain.
    """
    module = norms.get(field)
    return x if module is None else module(x)