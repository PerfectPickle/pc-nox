r"""graph/energies.py

Energy strategies: how a node's state is scored against its
predictions. Chosen per node by registry name on the `NodeSpec`
(`energy="pc"`), independent of the node type, so any node can be paired
with any energy it is compatible with (compatibility is checked once, at
construction, by `validation.validate_pairings`).

A strategy is a pure function of the state, the node's `NodePrediction`,
and a `frozen` context built once per settle (empty for standard PC; see the
Meta-PCN section below for a strategy that uses it).

Registered strategies:
    "pc"   standard predictive coding, squared prediction error.
    "ce"   cross-entropy of a clamped target under the node's logits.
    "meta" Meta-PCN (Ha et al., ICLR 2026): inference on "errors of errors".
"""

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .nodes.base import NodeBase, NodePrediction


@dataclass(frozen=True)
class EnergyStrategy:
    """A registry entry: the energy function plus what it needs from a node.

    Attributes:
        fn: `energy(state, prediction: NodePrediction, frozen, path_weights) -> scalar`.
        required_paths: prediction paths the node MUST return (e.g. "td").
        allowed_paths: if not None, the ONLY paths this strategy knows how
            to score; a node returning any other path is rejected rather
            than silently mis-scored. None means "any paths".
        needs_pre_activations: whether the strategy reads
            `NodePrediction.pre_activations` (e.g. Meta-PCN's D(h)).
        valid_for_free_state: whether the energy is a PREDICTION ERROR on the
            node's own state, so it makes sense when the node is released
            (free). False for losses against data, like cross-entropy.
        init_frozen: optional `(node, prediction, learning) -> context`. If
            set, the model calls it once per settle (see
            `GraphModel.frozen_context`) and hands the result to `fn` as its
            `frozen` argument. This is how a strategy gets quantities that
            must stay CONSTANT while the states settle (Meta-PCN's frozen
            predictions). None means the strategy needs no such context.
    """
    fn: Callable
    required_paths: FrozenSet[str] = frozenset({"td"})
    allowed_paths: Optional[FrozenSet[str]] = None
    needs_pre_activations: bool = False
    valid_for_free_state: bool = True
    init_frozen: Optional[Callable] = None


# --------------------------------------------------------------------------- #
# Standard strategies
# --------------------------------------------------------------------------- #

def pc_energy(state: Array, prediction: NodePrediction, frozen=None, path_weights=None) -> Array:
    """Standard PC: 0.5 * ||state - prediction||^2, summed over every
    prediction path the node returns, each scaled by `path_weights[path]`
    (default 1; the model passes the resolved td/bu precisions) (one term for a plain node, two for a
    bidirectional one, and zero for a pure-data node such as `InputNode`).
    """
    w = path_weights or {} # i.e. 'precision' weighting on td, bu etc
    return sum((w.get(k, 1.0) * 0.5 * jnp.sum((state - p) ** 2) for k, p in prediction.predictions.items()), jnp.asarray(0.0))


def ce_energy(state: Array, prediction: NodePrediction, frozen=None, path_weights=None) -> Array:
    """Cross-entropy of `state` (a target distribution, e.g. one-hot) under
    the node's "td" prediction treated as LOGITS. Intended for clamped
    observation nodes; identical to the `loss="ce"` observation term.
    """
    return (path_weights or {}).get("td", 1.0) * -jnp.sum(state * jax.nn.log_softmax(prediction.predictions["td"]))


# --------------------------------------------------------------------------- #
# Meta-PCN
# --------------------------------------------------------------------------- #
#
# Reference: Ha et al., "Stable and Scalable Deep Predictive Coding Networks
# with Meta-Prediction Errors", ICLR 2026 (Sec. 4.1, Alg. 1, App. D and F).
#
# NOTATION (the paper's, for a chain of layers l = 1..L; in this package a
# "layer" is a node, and a node may have several parents/children and
# several prediction paths -- everything below applies per path):
#
#     $\hat z_{l+1} = f_l(z_l) = \phi(h_{l+1}),\qquad h_{l+1} = W_l z_l + b_l$
#     $\delta_l = z_l - \hat z_l$                       (prediction error, PE)
#     $D(h) = \mathrm{diag}(\phi'(h))$                  (activation derivative)
#
# WHAT STANDARD PC DOES. It minimises the free energy
#
#     $\mathcal F = \tfrac12 \sum_l \lVert \delta_l \rVert^2$,
#
# whose gradient w.r.t. a hidden state is
#
#     $\nabla_{z_l}\mathcal F = \delta_l - g_l(\delta_{l+1}, h_{l+1}),
#       \qquad g_l(\delta_{l+1}, h_{l+1}) = W_l^\top D(h_{l+1})\,\delta_{l+1}$.
#
# In words: "my own error, minus the error my child sends back to me through
# the transpose of the weights". Deep nets suffer because that feedback term
# is recomputed from a moving state at every inference step, which makes the
# dynamics non-linear and (for the wrong weight scale) exploding/vanishing.
#
# WHAT META-PCN CHANGES. At the feed-forward initialisation $z^{(0)}$ it
# FREEZES the predictions and the activation derivatives:
#
#     $c_l := \phi(h^{(0)}_l)$        (frozen prediction)
#     $\tilde\delta_l := z_l - c_l$    (error against the FROZEN prediction)
#
# so only the states $z_l$ keep moving. The stationarity condition becomes the
# LINEAR system (in $z$)
#
#     $\tilde F_l(z) = \tilde\delta_l - g_l(\tilde\delta_{l+1}, h^{(0)}_{l+1})
#         = (z_l - c_l) - W_l^\top D(h^{(0)}_{l+1})\,(z_{l+1} - c_{l+1})$,
#
# and inference minimises the "error of errors" (meta-PE) objective
#
#     $\mathcal J(\tilde\delta) = \tfrac12 \sum_l
#         \lVert \tilde\delta_l - g_l(\tilde\delta^*_{l+1}, h^{(0)}_{l+1}) \rVert^2$,
#
# where $\tilde\delta^*_{l+1}$ is the child's error treated as a CONSTANT
# (stop-gradient; Alg. 1 line 14). Because $\partial\tilde\delta_l/\partial z_l = I$,
#
#     $\nabla_{z_l}\mathcal J = \tilde F_l(z)$   (the linearised stationarity map).
#
# HOW IT IS IMPLEMENTED HERE. The settling loop only ever needs
# $\nabla_z E$, and it differentiates a SUM OF PER-NODE energies. So instead of
# coding $\mathcal J$ literally (which would need every node to see its
# children's errors), we give each node k the local energy
#
#     $E_k = \tfrac12 \lVert z_k - c_k \rVert^2
#            \;-\; \big\langle\, \mathrm{sg}\!\big(D^{(0)}_k\,\tilde\delta_k\big),\; h_k \,\big\rangle$,
#
# where $h_k$ is the node's LIVE pre-activation (linear in its parents'
# states), $c_k, D^{(0)}_k$ are the frozen quantities, and sg = stop-gradient.
# Differentiating:
#
#     $\partial E_k/\partial z_k = \tilde\delta_k$                          (own error)
#     $\partial E_k/\partial z_{\text{parent}} = -W^\top D^{(0)}_k\,\tilde\delta_k$   (error sent to parent)
#
# and summing over nodes gives, for every inferred node $l$,
# $\nabla_{z_l}\sum_k E_k = \tilde\delta_l - W_l^\top D^{(0)}_{l+1}\tilde\delta_{l+1} = \nabla_{z_l}\mathcal J$.
# The GRADIENT is therefore exactly the paper's; the scalar VALUE of $E$ is not
# $\mathcal J$ (it can be negative). If you want $\mathcal J$ as a convergence
# diagnostic: $\mathcal J = \tfrac12\sum_{l\ \mathrm{inferred}}\lVert\nabla_{z_l}E\rVert^2$.
#
# This works for any node that reports pre-activations (dense, conv, recurrent
# tPC-H nodes), with no node-specific code, and extends path-by-path to
# bidirectional nodes ("td" and "bu" each get their own frozen $c$ and $D^{(0)}$).
#
# LEARNING (App. F). Parameters do NOT follow $\mathcal J$ (that would need
# second derivatives through $g_l$). They follow ordinary PC against the
# frozen forward pass:
#
#     $\mathcal L(\theta) = \tfrac12 \sum_l
#         \lVert z^{(T)}_l - f_{l-1}(z^{(0)}_{l-1};\theta) \rVert^2$.
#
# In "learning mode" the same strategy therefore drops the coupling term and
# lets $c$ depend on the weights: $E_k = \tfrac12\lVert z^{(T)}_k - c_k(\theta)\rVert^2$.

class MetaFrozen(NamedTuple):
    """Per-path quantities Meta-PCN freezes at the initialisation point $z^{(0)}$.

    Attributes:
        c: The frozen prediction $c = \\phi(h^{(0)})$ for this path.
        d0: The frozen activation derivative $\\phi'(h^{(0)})$ (the diagonal of
            $D(h^{(0)})$). `None` means LEARNING mode: the coupling term is
            dropped and `c` is expected to depend on the weights (see
            `GraphModel.param_grad`).
    """
    c: Array
    d0: Optional[Array] = None
    learning: bool = False          # static Python bool, resolved at trace time


def meta_init_frozen(node: NodeBase, prediction: NodePrediction, learning: bool = False) -> Dict[str, MetaFrozen]:
    """Build one node's Meta-PCN frozen context from its predictions at the
    initialisation point (called once per settle by `GraphModel.frozen_context`).

    Plain English: record what the node predicts right now ($c$) and how
    sensitive that prediction is to its pre-activation ($\\phi'(h^{(0)})$).
    Both stay fixed while the states settle.

    Args:
        node: The built node. Its `td_act_fn` / `bu_act_fn` give $\\phi$ for
            each path; a node with none (e.g. the linear readout) is treated
            as $\\phi = \\mathrm{id}$, so $\\phi' = 1$.
        prediction: The node's `NodePrediction` evaluated at the init states.
        learning: If True, skip $\\phi'$ (not needed) and keep `c` live so the
            weight gradient can flow through it.

    Returns:
        `{path: MetaFrozen}` -- empty for a pure-data node with no predictions.
    """
    frozen = {}
    for path, c in prediction.predictions.items():
        if learning:
            frozen[path] = MetaFrozen(c)
            continue
        h = prediction.pre_activations[path]
        act = getattr(node, f"{path}_act_fn", None) or (lambda x: x)
        # phi'(h), elementwise, via a JVP with a vector of ones (valid for elementwise activations).
        dphi = jax.jvp(act, (h,), (jnp.ones_like(h),))[1]
        # A leaky td path predicts (1-alpha)*s_prev + alpha*phi(h), so its derivative is alpha*phi'(h).
        alpha = getattr(node, "alpha", None) if path == "td" else None
        frozen[path] = MetaFrozen(c, (alpha if alpha else 1.0) * dphi)
    return frozen


def meta_energy(state: Array, prediction: NodePrediction, frozen: Dict[str, MetaFrozen], path_weights=None) -> Array:
    r"""Meta-PCN energy of one node (see the section comment above for the full story).

    For each prediction path (td, and bu if bidirectional), with frozen
    $c$ and $D^{(0)}$ and the live pre-activation $h$:

    $$E = \tfrac12\lVert z - c\rVert^2 \;-\; \big\langle \mathrm{sg}(D^{(0)}(z - c)),\, h\big\rangle .$$

    Plain English: the first term pulls the node's state toward its frozen
    prediction (its own meta-PE). The second term does not move this node's own
    state (its error is stop-gradiented); its job is that differentiating it
    w.r.t. the node's PARENTS (through the live $h$) delivers the error signal
    $-W^\top D^{(0)}\tilde\delta$ back up the graph -- the paper's $g_l$ -- with
    the child's error held constant (stop-gradient).

    In learning mode (`frozen[path].d0 is None`) only the first term is kept,
    giving the paper's $\mathcal L(\theta)$ (Eq. 27).

    Args:
        state: This node's state $z$ (settling) or settled state $z^{(T)}$ (learning).
        prediction: The node's current `NodePrediction` (needs `pre_activations`).
        frozen: `{path: MetaFrozen}` from `meta_init_frozen`.
        path_weights: Per-path precision weights, as for `pc_energy`.
    """
    w = path_weights or {}
    total = jnp.asarray(0.0)
    for path, fz in frozen.items():
        err = state - fz.c                                  # meta-PE  \tilde\delta = z - c
        term = 0.5 * jnp.sum(err ** 2)
        if fz.d0 is not None:                               # inference mode: add the coupling term
            signal = jax.lax.stop_gradient(fz.d0 * err)     # D^(0) \tilde\delta, a constant
            term = term - jnp.sum(signal * prediction.pre_activations[path])
        total = total + w.get(path, 1.0) * term
    return total


def meta_ce_energy(state, prediction, frozen, path_weights=None):
    fz, w = frozen["td"], (path_weights or {}).get("td", 1.0)
    if fz.learning:                 # CE of the frozen-input logits, live in theta
        return -w * jnp.sum(state * jax.nn.log_softmax(fz.c))
    err = jax.lax.stop_gradient(jax.nn.softmax(fz.c) - state)
    return w * jnp.sum(err * prediction.pre_activations["td"])   # grad wrt parents = W^T (softmax(c) - y)


ENERGY_REGISTRY: Dict[str, EnergyStrategy] = {
    # any paths, including none (a pure-data node). That an INFERRED node returns a "td" prediction is
    # enforced separately, in validation, because initialisation needs it whatever the energy.
    "pc": EnergyStrategy(fn=pc_energy, required_paths=frozenset()),
    "ce": EnergyStrategy(fn=ce_energy, allowed_paths=frozenset({"td"}), valid_for_free_state=False),
    # Meta-PCN: any paths (td and/or bu, each with its own frozen context); needs the nodes' pre-activations.
    "meta": EnergyStrategy(fn=meta_energy, required_paths=frozenset(), needs_pre_activations=True,
                           init_frozen=meta_init_frozen),
    "meta_ce": EnergyStrategy(fn=meta_ce_energy, allowed_paths={"td"}, needs_pre_activations=True,
                           init_frozen=meta_init_frozen, valid_for_free_state=False),
}
