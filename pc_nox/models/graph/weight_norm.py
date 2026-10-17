r"""graph/weight_norm.py

Variance-based weight normalisation from Meta-PCN (Ha et al., ICLR 2026,
Sec. 4.2 and Alg. 1 lines 21-24). Applied AFTER every optimiser step through
the `postprocess_params` hook, so it is a transform on the weights
themselves, not a term in the energy.

WHY. During inference, prediction errors are passed between layers through
$W^\top D(h)$, so their size is multiplied by roughly the spectral norm
$\lVert W\rVert_2$ per layer. If that is above 1 the errors explode with
depth; below 1 they vanish (the paper's "EVPE" pathology). Computing a
spectral norm every step is costly, so the paper uses a random-matrix
shortcut: for an $m\times n$ matrix with i.i.d. zero-mean entries of
variance $\sigma_w^2$,

    $\lVert W\rVert_2 \approx (\sqrt m + \sqrt n)\,\sigma_w$ .

Dividing by that estimate pins the spectral norm near 1:

    $W \leftarrow \dfrac{W}{(\sqrt m + \sqrt n)\,\sigma_w},\qquad
      \sigma_w = \sqrt{\mathrm{Var}(W)}$ .

It is one scalar per weight matrix, so it costs one reduction (a standard
deviation) and one multiply.

WHAT IS NORMALISED. Every tensor a node lists in `weight_entries()`:
  * dense weights as `(m, n) = (out, in)`;
  * conv kernels through the 2-D view `(C_out, C_in * kH * kW)` the node
    already provides -- the rescale is a SCALAR, so it is applied to the real
    (4-D) kernel and nothing needs reshaping back;
  * biases and the root `prior` are NOT in `weight_entries()`, so they are
    left alone (the paper normalises weight matrices only).
Which entries are touched is further narrowed by `weight_norm_scope`
("all" | "rec" | "ff", same tags as the weight-decay/orthogonal scopes).

Configured with `GraphModelConfig.weight_norm` ("none" | "variance"), with a
per-node override `NodeSpec.weight_norm`.
"""

import math
from typing import Callable, List

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .nodes.base import WeightEntry

WEIGHT_NORMS = ("none", "variance")


def variance_norm_scale(matrix: Array, eps: float = 1e-12) -> Array:
    r"""The factor $1 / \big((\sqrt m + \sqrt n)\,\sigma_w\big)$ for one weight matrix.

    Args:
        matrix: A 2-D `(m, n)` view of the weight (for a conv kernel,
            `(C_out, C_in * kH * kW)`).
        eps: Floor on $\sigma_w$ so an all-constant matrix cannot divide by zero.

    Returns:
        A scalar; multiply the real weight by it.
    """
    m, n = matrix.shape
    sigma_w = jnp.maximum(jnp.std(matrix), eps)      # sqrt(Var(W)) over all entries
    return 1.0 / ((math.sqrt(m) + math.sqrt(n)) * sigma_w)


def tag_in_scope(tag: str, scope: str) -> bool:
    """Whether a `WeightEntry.tag` ("rec" or "ff") falls in `scope` ("all" | "rec" | "ff")."""
    return scope == "all" or (scope == "rec") == (tag == "rec")


def weight_getter(node_name: str, path: str) -> Callable:
    """`f(model) -> the live array at model.nodes[node_name].<path>`, for a
    dotted `WeightEntry.name` such as "op_td.weight". Used both to read the
    weight and to tell `eqx.tree_at` where to write the rescaled one."""
    attrs = path.split(".")

    def get(model):
        obj = model.nodes[node_name]
        for a in attrs:
            obj = getattr(obj, a)
        return obj

    return get


def normalise_weights(model):
    """Return `model` with every in-scope weight of every opted-in node rescaled.

    Plain English: for each node whose `weight_norm` is "variance", find each
    of its weight tensors, compute its scalar correction from the 2-D view,
    and multiply the real tensor by it. Nodes that opt out, and any model with
    nothing opted in, come back unchanged (the same object).

    `model` needs only `.config` (a `GraphModelConfig`) and `.nodes`.
    """
    cfg = model.config
    getters: List[Callable] = []
    rescaled: List[Array] = []
    for spec in cfg.node_specs:
        if cfg.get(spec, "weight_norm") != "variance":
            continue
        entry: WeightEntry
        for entry in model.nodes[spec.name].weight_entries():
            if not tag_in_scope(entry.tag, cfg.weight_norm_scope):
                continue
            get = weight_getter(spec.name, entry.name)
            getters.append(get)
            rescaled.append(get(model) * variance_norm_scale(entry.matrix))
    if not getters:
        return model
    return eqx.tree_at(lambda m: tuple(g(m) for g in getters), model, tuple(rescaled))
