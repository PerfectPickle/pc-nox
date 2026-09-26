"""tpch/layers.py

The three layer roles that compose a `TpchModel` -- see `model.py`'s
module docstring for the shapes/equations. Split out on their own so a
sibling variant that reuses one or two of these unchanged (e.g. a variant
that only changes the observation layer) can import just those, and so
`model.py` isn't 250 lines longer than it needs to be.
"""

from typing import Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class TpchControlLayer(eqx.Module):
    """Top layer of a tPC-H network ("s" in the paper).

    This is the only layer driven by the raw external input x_t. It has two
    sets of weights:

        W_rec  ("A" in the paper): applied to its own previous state s_{t-1}
        W_in   ("B" in the paper): applied to the current control input x_t

    Prediction (eq. 19's first term, generalised):
        s_hat_t = f(W_rec @ s_{t-1} + W_in @ x_t)

    Optional leaky integration (tPC-E, eq. 29 of the tPC-E paper, linear
    case -- generalised here to the nonlinear/post-activation prediction,
    not verified against that paper's nonlinear appendix): when `alpha` is
    set, the returned prediction is instead

        s_hat_t = (1 - alpha) * s_{t-1} + alpha * f(W_rec @ s_{t-1} + W_in @ x_t)

    i.e. a leaky blend of the layer's own previous state with its usual
    instantaneous prediction, giving the state persistent memory across
    time. `alpha=None` (the default) recovers the original tPC-H
    prediction exactly -- opt-in extension, not a behaviour change for
    existing models. See `..eligibility` for the paired trace machinery
    this is meant to be combined with: the state now has lingering memory
    of old inputs, but plain Hebbian updates only ever correlate the
    current error with the CURRENT input -- eligibility traces are what
    let that lingering memory reach the weight update too.
    """

    W_rec: eqx.nn.Linear  # recurrent weight (from self at t-1)
    W_in: eqx.nn.Linear = None  # optional control input

    # static=True excludes this variable from the pytree, i.e. it's python metadata, not a leaf, and will be ignored during JAX autodifferentiation
    has_input: bool = eqx.field(static=True)
    act_fn: Callable = eqx.field(static=True)
    alpha: Optional[float] = eqx.field(static=True, default=None)  # tPC-E leak rate; None = disabled

    def __init__(
        self,
        state_size: int,  # nodes / width dimension
        input_size: Optional[int] = 0,
        act_fn: Callable = jnp.tanh,
        alpha: Optional[float] = None,
        *,
        key: PRNGKeyArray,
    ):
        key_rec, key_in = jr.split(key)
        # use_bias=False to match the paper exactly (eq. 19 has no bias terms)
        self.W_rec = eqx.nn.Linear(state_size, state_size, use_bias=False, key=key_rec)
        self.act_fn = act_fn
        self.alpha = alpha
        self.has_input = (input_size > 0)
        if self.has_input:  # create input 'control' weights if there is any input dim
            self.W_in = eqx.nn.Linear(input_size, state_size, use_bias=False, key=key_in)

    def pre_activation(self, state_prev: Array, control_input: Optional[Array] = None) -> Array:
        """A @ s_{t-1} + B @ x_t, i.e. the argument to `act_fn` -- exposed
        separately from `predict` so `f'(pre)` can be computed (via
        `..eligibility.elementwise_deriv`) for the traced weight-update
        rule; see `TpchModel.param_grad_traced`.
        """
        pre_acts = self.W_rec(state_prev)
        if self.has_input and control_input is not None:
            pre_acts = pre_acts + self.W_in(control_input)
        return pre_acts

    def predict(self, state_prev: Array, control_input: Optional[Array] = None) -> Array:
        """s_hat_t = f(A @ s_{t-1} + B @ x_t), or its leaky-integrated
        variant (see class docstring) when `alpha` is set."""
        instantaneous = self.act_fn(self.pre_activation(state_prev, control_input))
        if self.alpha is None:
            return instantaneous
        return (1.0 - self.alpha) * state_prev + self.alpha * instantaneous


class TpchHiddenLayer(eqx.Module):
    """Middle layer of a tPC-H network ("z" in the paper). Any number of
    these can be stacked between the control layer and the observation
    layer.

        W_rec ("P"): applied to its own previous state, z_{t-1}
        W_parent_prev ("Q"): applied to its parent's previous state, s_{t-1}
        W_parent_curr ("R"): applied to its parent's CURRENT state, s_t

    Prediction (eq. 19's second term, generalised):
        z_hat_t = f(W_rec @ z_{t-1} + W_parent_prev @ s_{t-1} + W_parent_curr @ s_t)

    The W_parent_curr / "R" pathway is what lets information flow down the
    hierarchy within a single time step, instead of only across time steps.

    Optional leaky integration (tPC-E, see `TpchControlLayer.alpha` for the
    same mechanism worked out on the simpler top layer): when `alpha` is
    set,
        z_hat_t = (1-alpha)*z_{t-1} + alpha*f(P@z_{t-1} + Q@s_{t-1} + R@s_t)
    All THREE weights get an eligibility trace under this layer's own
    alpha when leaky (see `..eligibility`), including W_parent_curr --
    even though R multiplies a same-timestep value, R's contribution still
    persists forward through this layer's own leaky memory of z_t, exactly
    the way W_rec's does; only the observation layer (no memory at all) is
    exempt from tracing. `alpha=None` (default) is the original tPC-H
    prediction, unchanged.
    """

    W_rec: eqx.nn.Linear
    W_parent_prev: eqx.nn.Linear
    W_parent_curr: eqx.nn.Linear
    act_fn: Callable = eqx.field(static=True)
    alpha: Optional[float] = eqx.field(static=True, default=None)  # tPC-E leak rate; None = disabled

    def __init__(
        self,
        state_size: int,
        parent_size: int,
        act_fn: Callable = jnp.tanh,
        alpha: Optional[float] = None,
        *,
        key: PRNGKeyArray,
    ):
        key_rec, key_parent_prev, key_parent_curr = jr.split(key, 3)
        self.W_rec = eqx.nn.Linear(state_size, state_size, use_bias=False, key=key_rec)
        self.W_parent_prev = eqx.nn.Linear(parent_size, state_size, use_bias=False, key=key_parent_prev)
        self.W_parent_curr = eqx.nn.Linear(parent_size, state_size, use_bias=False, key=key_parent_curr)
        self.act_fn = act_fn
        self.alpha = alpha

    def pre_activation(self, state_prev: Array, parent_prev: Array, parent_curr: Array) -> Array:
        """P @ z_{t-1} + Q @ s_{t-1} + R @ s_t -- see
        `TpchControlLayer.pre_activation` for why this is exposed."""
        return (
            self.W_rec(state_prev)
            + self.W_parent_prev(parent_prev)
            + self.W_parent_curr(parent_curr)
        )

    def predict(self, state_prev: Array, parent_prev: Array, parent_curr: Array) -> Array:
        """z_hat_t = f(P @ z_{t-1} + Q @ s_{t-1} + R @ s_t), or its
        leaky-integrated variant (see class docstring) when `alpha` is set."""
        instantaneous = self.act_fn(self.pre_activation(state_prev, parent_prev, parent_curr))
        if self.alpha is None:
            return instantaneous
        return (1.0 - self.alpha) * state_prev + self.alpha * instantaneous


class TpchObservationLayer(eqx.Module):
    """Bottom layer of a tPC-H network ("y" in the paper).

    Unlike the other two layer types above, the observation layer has no
    memory of its own and no nonlinearity -- it is a pure linear emission
    of whatever the lowest hidden layer's CURRENT state is. One set of
    weights:

        W_parent ("C"): applied to the parent layer's current state, z_t

    Prediction (eq. 19's third term):
        y_hat_t = C @ z_t

    Implements the minimal interface every observation-layer variant
    needs (`predict`, `energy`, `weights`) so `TpchModel` can treat any
    conforming layer -- homogeneous or node-heterogeneous -- the same
    way. See `TpchHeterogeneousObservationLayer` for an example of a
    variant with per-node-group losses/weights.
    """

    W_parent: eqx.nn.Linear
    loss: str = eqx.field(static=True, default="mse")  # 'mse' | 'ce'

    def __init__(self, obs_size: int, parent_size: int, loss: str = "mse", *, key: PRNGKeyArray):
        self.W_parent = eqx.nn.Linear(parent_size, obs_size, use_bias=False, key=key)
        self.loss = loss

    def predict(self, parent_curr: Array) -> Array:
        """y_hat_t = C @ z_t  (no activation function -- pure linear readout)"""
        return self.W_parent(parent_curr)

    def energy(self, y_hat: Array, observation: Array) -> Array:
        """Observation-term energy: 0.5 * ||y - y_hat||^2 for `loss='mse'`,
        or cross-entropy of `observation` under logits `y_hat` for
        `loss='ce'`. Moved here (out of `tpch_energy_fn`) so it can be
        overridden per observation-layer variant -- see
        `TpchHeterogeneousObservationLayer.energy` for a variant that
        applies a different loss per node group.
        """
        if self.loss == "mse":
            return 0.5 * jnp.sum((observation - y_hat) ** 2)
        else:  # "ce"
            return -jnp.sum(observation * jax.nn.log_softmax(y_hat))

    def weights(self) -> List[Array]:
        """Every weight matrix owned by this layer -- used by
        `TpchModel._all_weights`/`_ff_weights`/`_weight_entries` so weight
        enumeration doesn't need to know whether the model's observation
        layer is homogeneous or not.
        """
        return [self.W_parent.weight]


class TpchHeterogeneousObservationLayer(eqx.Module):
    """Example node-level-heterogeneous observation layer: partitions the
    observation vector into named groups, each with ITS OWN weight
    submatrix and ITS OWN loss kind -- e.g. an 'mse' group of ordinary
    sensory-reconstruction nodes alongside a 'ce' group of categorical
    action-readout nodes, sharing one parent (the lowest hidden layer)
    but nothing else.

    Implements exactly the same interface as `TpchObservationLayer`
    (`predict`, `energy`, `weights`), so it's a drop-in replacement:
    pass a pre-built instance as `TpchModel(..., observation_layer=...)`.
    Nothing in `TpchModel` needs to know or care that it isn't the plain
    layer -- that's the whole point of the interface split.

    `predict()`'s output is the concatenation of every group's prediction,
    in group order, so `observation`/`y` passed to `tpch_energy_fn` must
    be laid out the same way (group 0's slice first, then group 1's, ...).
    """

    groups: List[eqx.nn.Linear]
    group_names: Tuple[str, ...] = eqx.field(static=True)
    group_sizes: Tuple[int, ...] = eqx.field(static=True)
    group_losses: Tuple[str, ...] = eqx.field(static=True)  # 'mse' | 'ce', one per group

    def __init__(
        self,
        parent_size: int,
        group_specs,  # Sequence[Tuple[name: str, size: int, loss: str]]
        *,
        key: PRNGKeyArray,
    ):
        keys = jr.split(key, len(group_specs))
        self.groups = [
            eqx.nn.Linear(parent_size, size, use_bias=False, key=k)
            for (_, size, _), k in zip(group_specs, keys)
        ]
        self.group_names = tuple(name for name, _, _ in group_specs)
        self.group_sizes = tuple(size for _, size, _ in group_specs)
        self.group_losses = tuple(loss for _, _, loss in group_specs)

    def predict(self, parent_curr: Array) -> Array:
        """Concatenation of every group's own linear readout, in group order."""
        return jnp.concatenate([g(parent_curr) for g in self.groups])

    def energy(self, y_hat: Array, observation: Array) -> Array:
        """Sum of each group's own loss, applied to its own slice of
        `y_hat`/`observation` (slice boundaries determined by
        `group_sizes`, in the same order `predict()` concatenated them).
        """
        total = jnp.asarray(0.0)
        idx = 0
        for size, loss in zip(self.group_sizes, self.group_losses):
            y_hat_g = y_hat[idx: idx + size]
            obs_g = observation[idx: idx + size]
            if loss == "mse":
                total = total + 0.5 * jnp.sum((obs_g - y_hat_g) ** 2)
            else:  # "ce"
                total = total + -jnp.sum(obs_g * jax.nn.log_softmax(y_hat_g))
            idx += size
        return total

    def weights(self) -> List[Array]:
        """Every group's weight matrix, in group order."""
        return [g.weight for g in self.groups]
