"""tpch.py

Equinox implementation of temporal Hierarchical Predictive Coding (tPC-H),
following eqs. (19)-(27) of the tPC-H paper:

Ng-Kee-Kwong, J., Tang, M., Akam, T., & Bogacz, R. (2026). 
Learning complex temporal dependencies via local synaptic plasticity. 
bioRxiv. https://doi.org/10.64898/2026.07.09.737423

--------------------------------------------------------------------------
What tPC-H is, in one paragraph
--------------------------------------------------------------------------
A standard (non-hierarchical) temporal PC network has a single chain of
latent states s_{t-1} -> s_t -> ... driven by an optional top-down external input x_t, 
plus an output y_t = C @ s_t. tPC-H stacks several such chains on top of each
other. Every layer keeps its own latent state through time (its own
recurrent weight), but a layer's state is *also* shaped by the layer above
it (its "parent") -- both the parent's state one step ago AND the parent's
state at the current step. That second, same-time-step connection is what
makes the model "hierarchical" rather than just a stack of independent
RNNs: information from a higher layer can reach a lower layer within the
very same time step, and predictions flow more timesteps into the future with 
increased depth without violating locality.

--------------------------------------------------------------------------
The three layer roles (this is exactly the 2-hidden-layer case of eq. 19,
generalised to an arbitrary number of hidden layers)
--------------------------------------------------------------------------
    TpchControlLayer      (top)     e.g. "s" in the paper
        driven by: its own previous state (weight A) 
                 + optional external input x_t (weight B)
        s_hat_t = f(A @ s_{t-1} + B @ x_t)

    TpchHiddenLayer        (middle, any number of these stacked)  e.g. "z"
        driven by: its own previous state (weight P)
                 + its parent's previous state (weight Q)
                 + its parent's CURRENT state (weight R)
        z_hat_t = f(P @ z_{t-1} + Q @ s_{t-1} + R @ s_t)

    TpchObservationLayer   (bottom)  "y"
        driven by: its parent's current state only (weight C), no
        nonlinearity and no memory of its own -- it is a pure emission model
        y_hat_t = C @ z_t

Stacking N TpchHiddenLayers between one TpchControlLayer and one
TpchObservationLayer reproduces the general N-layer hierarchy that eqs.
(20)-(21) describe (they are written with a generic layer index k for
exactly this reason).

--------------------------------------------------------------------------
How inference & learning are implemented
--------------------------------------------------------------------------
Rather than hand-coding the closed-form gradients of eqs. (20)-(27), we
write down the scalar free energy F_t (eq. 19, generalised) as a plain
JAX-differentiable function of the states and weights, and let
`jax.grad` do the differentiation:

    * grad of F_t w.r.t. the current states  ==  eqs. (20)-(21) (inference)
    * grad of F_t w.r.t. the weights         ==  eqs. (22)-(27) (learning)

This is not a hand-wavy approximation: F_t is a SUM of independent
per-layer squared-error terms, and each state/weight only appears in one
or two of those terms, so autodiff reconstructs the exact local, Hebbian
("pre-synaptic activity x post-synaptic error") update rules the paper
derives by hand -- it just saves us from re-deriving and re-typing eqs.
(20)-(27) by hand, and it generalises for free to any number of hidden
layers.

Every function here operates on a *single, unbatched* time step (all arrays are 1-D). 
Use `jax.vmap` over a leading batch axis, and `jax.lax.scan` over a leading time axis, 
in the calling code that loops over a batch of sequences.
"""


import equinox as eqx
from .model_base import ModelBase, ACT_FN_REGISTRY, Activities, Predictions
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import warnings
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Callable, ClassVar, List, Sequence, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import diffrax


# =============================================================================
# 0. Config definition
# =============================================================================

# frozen to ensure hashable for JAX Jax compilation
@dataclass(frozen=True)
class TpchConfig:
    control_layer_size: int  # Control layer width
    hidden_sizes: Tuple[int]  # Ordered from top (just below control) to bottom (just above obs). Tuple is hashable, unlike List
    obs_size: int  # Observation / sensory layer width
    input_size: int = 0  # Control input (optional)
    act_fn: str = "tanh"  # Activation function name, used as lookup key in ACT_FN_REGISTRY
    loss: str = "mse"  # Observation-layer loss ('mse' | 'ce'); only changes the y_hat term, not the state-prediction terms
    weight_decay: float = 0.0  # Coefficient for L2 penalty on weights (0 = disabled)
    weight_decay_scope: str = "all"  # Which weights weight_decay applies to: 'all', 'rec' (recurrent, i.e. W_rec only), or 'ff' (everything except W_rec)
    orthogonal_penalty: float = 0.0  # Coefficient for orthogonality penalty ||I - W^T W||^2 on weights (0 = disabled)
    orthogonal_scope: str = "rec"  # Which weights orthogonal_penalty applies to: 'rec' (default, only weights guaranteed square), 'all', or 'ff'
    activity_decay: float = 0.0  # Coefficient for L1/L2 penalty on current states/activities (0 = disabled)
    activity_reg_type: str = "l1"  # Norm used by activity_decay ('l1' | 'l2')


def _scopes_overlap(scope_a: str, scope_b: str) -> bool:
    """Whether two weight scopes ('all' | 'rec' | 'ff') can refer to any of
    the same weight matrices. Purely symbolic (doesn't need actual weight
    arrays): 'all' is a superset of both 'rec' and 'ff', which are disjoint
    from each other. Used to decide whether weight_decay and
    orthogonal_penalty could compete for the same weights (see the warning
    in TpchModel.__init__).
    """
    if scope_a == "all" or scope_b == "all":
        return True
    return scope_a == scope_b



# =============================================================================
# 1. Layer definitions
# =============================================================================

class TpchControlLayer(eqx.Module):
    """Top layer of a tPC-H network ("s" in the paper).

    This is the only layer driven by the raw external input x_t. It has two
    sets of weights:

        W_rec  ("A" in the paper): applied to its own previous state s_{t-1}
        W_in   ("B" in the paper): applied to the current control input x_t

    Prediction (eq. 19's first term, generalised):
        s_hat_t = f(W_rec @ s_{t-1} + W_in @ x_t)
    """

    W_rec: eqx.nn.Linear # recurrent weight (from self at t-1)
    W_in: eqx.nn.Linear = None # optional control input

    # static=True excludes this variable from the pytree, i.e. it's python metadata, not a leaf, and will be ignored during JAX autodifferentiation
    has_input: bool = eqx.field(static=True)
    act_fn: Callable = eqx.field(static=True) 

    def __init__(
        self,
        state_size: int, # nodes / width dimension
        input_size: Optional[int] = 0,
        act_fn: Callable = jnp.tanh,
        *,
        key: PRNGKeyArray,
    ):
        key_rec, key_in = jr.split(key)
        # use_bias=False to match the paper exactly (eq. 19 has no bias terms)
        self.W_rec = eqx.nn.Linear(state_size, state_size, use_bias=False, key=key_rec)
        self.act_fn = act_fn
        self.has_input = (input_size > 0)
        if self.has_input: # create input 'control' weights if there is any input dim
            self.W_in = eqx.nn.Linear(input_size, state_size, use_bias=False, key=key_in)


    def predict(self, state_prev: Array, control_input: Optional[Array] = None) -> Array:
        """s_hat_t = f(A @ s_{t-1} + B @ x_t)"""
        rec_term = self.W_rec(state_prev)
        if not self.has_input or control_input is None:
            return self.act_fn(rec_term)
        return self.act_fn(rec_term + self.W_in(control_input))


class TpchHiddenLayer(eqx.Module):
    """A middle layer of a tPC-H network ("z" in the paper's 2-layer example).

    Any number of these can be stacked between the control layer and the
    observation layer. Each one has three sets of weights:

        W_rec         ("P"): applied to its own previous state, z_{t-1}
        W_parent_prev ("Q"): applied to its parent's previous state, s_{t-1}
        W_parent_curr ("R"): applied to its parent's CURRENT state, s_t

    Prediction (eq. 19's second term, generalised):
        z_hat_t = f(W_rec @ z_{t-1} + W_parent_prev @ s_{t-1} + W_parent_curr @ s_t)

    The W_parent_curr / "R" pathway is what lets information flow down the
    hierarchy within a single time step, instead of only across time steps.
    """

    W_rec: eqx.nn.Linear
    W_parent_prev: eqx.nn.Linear
    W_parent_curr: eqx.nn.Linear
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        state_size: int,
        parent_size: int,
        act_fn: Callable = jnp.tanh,
        *,
        key: PRNGKeyArray,
    ):
        key_rec, key_parent_prev, key_parent_curr = jr.split(key, 3)
        self.W_rec = eqx.nn.Linear(state_size, state_size, use_bias=False, key=key_rec)
        self.W_parent_prev = eqx.nn.Linear(parent_size, state_size, use_bias=False, key=key_parent_prev)
        self.W_parent_curr = eqx.nn.Linear(parent_size, state_size, use_bias=False, key=key_parent_curr)
        self.act_fn = act_fn

    def predict(self, state_prev: Array, parent_prev: Array, parent_curr: Array) -> Array:
        """z_hat_t = f(P @ z_{t-1} + Q @ s_{t-1} + R @ s_t)"""
        return self.act_fn(
            self.W_rec(state_prev)
            + self.W_parent_prev(parent_prev)
            + self.W_parent_curr(parent_curr)
        )


class TpchObservationLayer(eqx.Module):
    """Bottom layer of a tPC-H network ("y" in the paper).

    Unlike the other two layers types above, the observation layer has no memory of its
    own and no nonlinearity -- it is a pure linear emission of whatever the
    lowest hidden layer's CURRENT state is. One set of weights:

        W_parent ("C"): applied to the parent layer's current state, z_t

    Prediction (eq. 19's third term):
        y_hat_t = C @ z_t
    """

    W_parent: eqx.nn.Linear

    def __init__(self, obs_size: int, parent_size: int, *, key: PRNGKeyArray):
        self.W_parent = eqx.nn.Linear(parent_size, obs_size, use_bias=False, key=key)

    def predict(self, parent_curr: Array) -> Array:
        """y_hat_t = C @ z_t  (no activation function -- pure linear readout)"""
        return self.W_parent(parent_curr)



# =============================================================================
# 2. TpchModel: composes the three layer types into a full tPC-H hierarchy
# =============================================================================

class TpchModel(eqx.Module, ModelBase):
    """A full tPC-H hierarchy chained top to bottom.
    
    Consists of: one control layer, N >= 0 hidden layers, and one observation layer.

    Attributes:
        control_layer: The control layer module at the top of the network.
        hidden_layers: List of hidden layer modules.
        observation_layer: The observation layer module at the bottom of the network. 

    Args:
        control_layer_size: Width of the top/control layer.
        hidden_sizes: Sequence of hidden layer widths, number of elements determines the number of hidden layers.
        obs_size: Width of the observation / output.
        act_fn: Name of activation function used by the control and hidden layers ('tanh').
        key: Jax pseudo random number generator key used for layer initilizations.
        input_size: Width of input provided to control layer, defaults to 0.
        loss: Loss used for the observation term of the free energy. Either
            `"mse"` (default) or `"ce"` (cross-entropy, treating the
            observation layer's linear output as logits). Only affects the
            observation term -- state-prediction terms are always squared error.
        weight_decay: Coefficient for an L2 penalty on the weights (0 =
            disabled, the default). Applied to `weight_decay_scope` weights.
        weight_decay_scope: Which weights `weight_decay` applies to: `"all"`
            (default), `"rec"` (only the recurrent, self-to-self weights,
            i.e. W_rec on the control and hidden layers), or `"ff"` (every
            weight except W_rec -- use this to keep weight_decay from
            competing with an `orthogonal_penalty` also targeting W_rec).
        orthogonal_penalty: Coefficient for an orthogonality-promoting penalty
            of the form ||I - W^T @ W||^2 on the weights (0 = disabled, the
            default). Applied to `orthogonal_scope` weights.
        orthogonal_scope: Which weights `orthogonal_penalty` applies to: `"rec"`
            (default, the only weights guaranteed square), `"all"`, or `"ff"`.
        activity_decay: Coefficient for an L1 or L2 penalty on the current
            states/activities (0 = disabled, the default).
        activity_reg_type: Either `"l1"` (default) or `"l2"`, selecting the
            form of the `activity_decay` penalty.
    """
    model_type: ClassVar[str] = "tpch"
    config_cls: ClassVar[type] = TpchConfig
    config: TpchConfig = eqx.field(static=True)

    control_layer: TpchControlLayer
    hidden_layers: List[TpchHiddenLayer]
    observation_layer: TpchObservationLayer

    def __init__(
        self,
        control_layer_size: int,
        hidden_sizes: Sequence[int],
        obs_size: int,
        key: PRNGKeyArray,
        act_fn: str = "tanh",
        input_size: Optional[int] = 0, # optional control input
        loss: str = "mse",
        weight_decay: float = 0.0,
        weight_decay_scope: str = "all",
        orthogonal_penalty: float = 0.0,
        orthogonal_scope: str = "rec",
        activity_decay: float = 0.0,
        activity_reg_type: str = "l1",
    ):
        if loss not in ("mse", "ce"):
            raise ValueError(f"loss must be 'mse' or 'ce', got {loss!r}")
        if weight_decay_scope not in ("all", "rec", "ff"):
            raise ValueError(f"weight_decay_scope must be 'all', 'rec' or 'ff', got {weight_decay_scope!r}")
        if orthogonal_scope not in ("all", "rec", "ff"):
            raise ValueError(f"orthogonal_scope must be 'all', 'rec' or 'ff', got {orthogonal_scope!r}")
        if activity_reg_type not in ("l1", "l2"):
            raise ValueError(f"activity_reg_type must be 'l1' or 'l2', got {activity_reg_type!r}")

        # If weight_decay and orthogonal_penalty can touch the same weights,
        # they pull those weights' singular values in opposite directions
        # (decay -> 0, orthogonal -> 1). Solving for the gradient-descent
        # fixed point of the two regularisation terms alone gives an
        # equilibrium singular value of sigma* = sqrt(1 - weight_decay / (2 *
        # orthogonal_penalty)), which only exists (is real and nonzero) when
        # weight_decay < 2 * orthogonal_penalty; otherwise the only stable
        # point is sigma=0, i.e. the affected weights collapse to zero
        # instead of settling anywhere near orthogonal. This is a threshold
        # on the *regularisation* landscape only -- the task gradient during
        # real training can still push weights toward collapse for other
        # reasons even when this check passes, so it's a necessary sanity
        # check, not a guarantee about the full training dynamics.
        if (
            weight_decay > 0. and orthogonal_penalty > 0.
            and _scopes_overlap(weight_decay_scope, orthogonal_scope)
            and weight_decay >= 2 * orthogonal_penalty
        ):
            warnings.warn(
                f"weight_decay ({weight_decay}) >= 2 * orthogonal_penalty ({orthogonal_penalty}) "
                f"on overlapping scopes (weight_decay_scope={weight_decay_scope!r}, "
                f"orthogonal_scope={orthogonal_scope!r}): the two regularisers have no stable "
                "nonzero equilibrium on the shared weights and will tend to collapse them toward "
                "zero. Lower weight_decay, raise orthogonal_penalty, or use non-overlapping "
                "scopes (e.g. weight_decay_scope='ff' with orthogonal_scope='rec').",
                stacklevel=2,
            )

        self.config = TpchConfig(
            control_layer_size=control_layer_size,
            hidden_sizes=tuple(hidden_sizes),  # cast hidden_sizes to tuple to ensure config hashability
            obs_size=obs_size,
            input_size=input_size,
            act_fn=act_fn,
            loss=loss,
            weight_decay=weight_decay,
            weight_decay_scope=weight_decay_scope,
            orthogonal_penalty=orthogonal_penalty,
            orthogonal_scope=orthogonal_scope,
            activity_decay=activity_decay,
            activity_reg_type=activity_reg_type,
        )

        try:
            act_fn_callable = ACT_FN_REGISTRY[act_fn] # Get Callable act_fn. This is the ONE place this str -> callable lookup happens.
        except KeyError:
            raise KeyError(
                f"act_fn={act_fn!r} not in ACT_FN_REGISTRY. If this is a custom "
                f"activation, register it before loading: ACT_FN_REGISTRY[{act_fn!r}] = ..."
            ) from None

        n_hidden = len(hidden_sizes)
        key_control, *hidden_keys, key_obs = jr.split(key, 2 + n_hidden)

        self.control_layer = TpchControlLayer(
            state_size=control_layer_size, input_size=input_size, act_fn=act_fn_callable, key=key_control
        )

        hidden_layers = []
        parent_size = control_layer_size  # the first hidden layer's parent is the control layer
        for size, hkey in zip(hidden_sizes, hidden_keys):
            hidden_layers.append(
                TpchHiddenLayer(state_size=size, parent_size=parent_size, act_fn=act_fn_callable, key=hkey)
            )
            parent_size = size  # each subsequent hidden layer's parent is the one above it
        self.hidden_layers = hidden_layers

        # the observation layer's parent is the lowest hidden layer (or the
        # control layer itself, if there are no hidden layers at all)
        self.observation_layer = TpchObservationLayer(obs_size=obs_size, parent_size=parent_size, key=key_obs)


    def predict(
        self, states_prev: Activities, states_curr: Activities, control_input: Optional[Array] = None
    ) -> Tuple[Predictions, Array]:
        """Run every layer's `predict` once, given:
          - states_prev: every layer's state at t-1 (fixed, "memory")
          - states_curr: every layer's CURRENT guess for its state at t (this
            is what inference iteratively refines -- see `settle` below)
          - control_input: x_t

        Returns (predictions, y_hat) where `predictions` is a list aligned
        with `states_curr`, i.e. predictions[i] is what layer i "expected"
        its own states_curr[i] to be.
        """
        predictions = [self.control_layer.predict(states_prev[0], control_input)]

        for i, layer in enumerate(self.hidden_layers):
            own_prev = states_prev[i + 1] # to account for control layer state at index 0
            parent_prev = states_prev[i]       # layer above, one step ago
            parent_curr = states_curr[i]       # layer above, right now
            predictions.append(layer.predict(own_prev, parent_prev, parent_curr))

        y_hat = self.observation_layer.predict(states_curr[-1])
        return predictions, y_hat


    def init_activities(self, states_prev: Activities, control_input: Optional[Array] = None) -> Activities:
        """Feedforward "kick-start" pass: produces an initial prediction for every
        layer's current-time-step state using only states_prev and
        control_input, sweeping top-to-bottom as every non-control layer depends on its parent. 
        `settle` (below) then refines this prediction by descending the free energy.
        """
        control_pred = self.control_layer.predict(states_prev[0], control_input)
        states_curr = [control_pred]

        parent_pred = control_pred
        for i, layer in enumerate(self.hidden_layers):
            own_prev = states_prev[i + 1] # to account for control layer state at index 0
            parent_prev = states_prev[i]
            prediction = layer.predict(own_prev, parent_prev, parent_pred)
            states_curr.append(prediction)
            parent_pred = prediction

        return states_curr


    # =============================================================================
    # 3. Free energy -- eq. (19), generalised to an arbitrary number of layers
    # =============================================================================
    #
    # `self.config.loss` switches the observation term between MSE and
    # cross-entropy (see `tpch_energy_fn` below) -- it doesn't need any of
    # the plumbing discussed next, since it's just a branch inside the one
    # energy function every other method already calls.
    #
    # Regularisation design note (why this lives entirely inside
    # `tpch_energy_fn` and nowhere else):
    #
    # Every other method in this file -- `neg_activity_grad`, `param_grad`,
    # `make_activity_step`, etc. -- only ever differentiates
    # `tpch_energy_fn`, either w.r.t. `states_curr` (inference) or w.r.t.
    # `self` (learning). Autodiff already routes each regulariser to the
    # right place with zero extra plumbing:
    #   - `_activity_reg` depends only on `states_curr`, so it contributes a
    #     real term to the *inference* gradient and an exact-zero term to
    #     the *weight* gradient.
    #   - `_weight_l2_reg` / `_weight_orthogonal_reg` depend only on `self`'s weights, so
    #     they contribute a real term to the *learning* gradient and an
    #     exact-zero term to the *inference* gradient.
    # So it's correct -- not just convenient -- to add all three straight
    # into the one energy function below


    def _rec_weights(self) -> List[Array]:
        """The recurrent, self-to-self weights: control_layer.W_rec and every
        hidden layer's W_rec. These are the only weights guaranteed to be
        square (state_size x state_size), which makes them the natural
        default target for the orthogonality/spectral penalty below.
        """
        weights = [self.control_layer.W_rec.weight]
        weights += [layer.W_rec.weight for layer in self.hidden_layers]
        return weights


    # feedforward weights only
    def _ff_weights(self) -> List[Array]:
        """Every weight EXCEPT the recurrent ones: W_in (if present), each
        hidden layer's W_parent_prev/W_parent_curr, and the observation
        layer's W_parent. Useful when you want weight_decay to leave the
        recurrent weights alone entirely -- e.g. so it can't fight an
        orthogonal_penalty also targeting W_rec (see the note above
        `_weight_orthogonal_reg`) regardless of how the two coefficients are tuned.
        """
        weights = []
        if self.control_layer.has_input:
            weights.append(self.control_layer.W_in.weight)
        for layer in self.hidden_layers:
            weights += [layer.W_parent_prev.weight, layer.W_parent_curr.weight]
        weights.append(self.observation_layer.W_parent.weight)
        return weights


    def _all_weights(self) -> List[Array]:
        """Every weight matrix in the model (recurrent + feedforward/emission)."""
        weights = [self.control_layer.W_rec.weight]
        if self.control_layer.has_input:
            weights.append(self.control_layer.W_in.weight)
        for layer in self.hidden_layers:
            weights += [
                layer.W_rec.weight,
                layer.W_parent_prev.weight,
                layer.W_parent_curr.weight,
            ]
        weights.append(self.observation_layer.W_parent.weight)
        return weights


    def _weights_for_scope(self, scope: str) -> List[Array]:
        """Resolves a scope string ('all' | 'rec' | 'ff') to the weight
        arrays it refers to. Shared by `_weight_l2_reg` and `_weight_orthogonal_reg`
        so both regularisers pick their targets the same way.
        """
        if scope == "rec":
            return self._rec_weights()
        elif scope == "ff":
            return self._ff_weights()
        else:  # "all", validated in __init__
            return self._all_weights()


    # extra energy (penalty) from L2 weight decay / regularisation
    def _weight_l2_reg(self) -> Array:
        """0.5 * weight_decay * sum ||W||_F^2 (squared Frobenius norm) over
        `weight_decay_scope` weights."""
        if self.config.weight_decay <= 0.0:
            return jnp.asarray(0.0)
        weights = self._weights_for_scope(self.config.weight_decay_scope)
        sq_norm = sum(jnp.sum(W ** 2) for W in weights)
        return 0.5 * self.config.weight_decay * sq_norm


    def _weight_orthogonal_reg(self) -> Array:
        """0.5 * orthogonal_penalty * sum ||I - W^T W||_F^2 (or W W^T for a
        'wide' W) over `orthogonal_scope` weights -- an orthogonality penalty
        that discourages the corresponding linear map from expanding or
        contracting its input, which is particularly useful on recurrent
        weights to keep the state dynamics well-conditioned over time.
        """
        if self.config.orthogonal_penalty <= 0.0:
            return jnp.asarray(0.0)
        weights = self._weights_for_scope(self.config.orthogonal_scope)
        reg = jnp.asarray(0.0)
        for W in weights:
            # eqx.nn.Linear weights have shape (out_features, in_features).
            # Use whichever of W^T @ W or W @ W^T is smaller, both to save
            # compute and because only the smaller one can equal identity.
            out_dim, in_dim = W.shape
            dim = min(out_dim, in_dim)
            gram = (W.T @ W) if in_dim <= out_dim else (W @ W.T)
            reg = reg + jnp.sum((jnp.eye(dim) - gram) ** 2)
        return 0.5 * self.config.orthogonal_penalty * reg

    
    def _activity_reg(self, states_curr: Activities) -> Array:
        """0.5 * activity_decay * sum ||z||_p^p over every current state,
        with p=1 (sparsity-promoting) or p=2 (soft bound on activity norm)
        depending on `activity_reg_type`.
        """
        if self.config.activity_decay <= 0.:
            return jnp.asarray(0.0)
        if self.config.activity_reg_type == "l1":
            reg = sum(jnp.sum(jnp.abs(state)) for state in states_curr)
        else: # l2
            reg = sum(jnp.sum(state ** 2) for state in states_curr)
        return 0.5 * self.config.activity_decay * reg


    # -------------------------------------------------------------------
    # Layerwise regularisation (for `return_layerwise=True`) -- additive to
    # everything above; `_weight_l2_reg`/`_weight_orthogonal_reg`/
    # `_activity_reg` (the scalar totals used by the default, cached
    # `weight_reg_total` path) are untouched, deliberately, since they're
    # covered by strict atol=0.0 tests and float summation order matters
    # for those.
    #
    # Why this is exact, not an approximation (see the earlier "divide by
    # number of layers" discussion): weight_decay/orthogonal_penalty are
    # each a SUM over a set of weight MATRICES, and every matrix has one
    # unambiguous owning layer. Grouping matrices by owning layer before
    # summing, instead of after, is exactly the same total (sum of sums ==
    # sum of everything) -- just partitioned. activity_decay is already a
    # sum over one term per state, i.e. already one-per-layer; no grouping
    # needed at all.
    # -------------------------------------------------------------------

    def _weight_entries(self) -> List[Tuple[int, bool, Array]]:
        """(owning_layer_idx, is_recurrent, weight) for every weight matrix
        in the model. owning_layer_idx matches `layer_labels()`'s order:
        0 = control, 1..num_hidden = hidden layers top-to-bottom,
        num_hidden+1 = observation. Single source of truth for the
        layerwise regularisers below, so "which weights count as
        rec/ff" can't drift from `_rec_weights`/`_ff_weights`/`_all_weights`
        (which remain independent and untouched -- this doesn't replace them).
        """
        entries = [(0, True, self.control_layer.W_rec.weight)]
        if self.control_layer.has_input:
            entries.append((0, False, self.control_layer.W_in.weight))
        for i, layer in enumerate(self.hidden_layers, start=1):
            entries += [
                (i, True, layer.W_rec.weight),
                (i, False, layer.W_parent_prev.weight),
                (i, False, layer.W_parent_curr.weight),
            ]
        entries.append((len(self.hidden_layers) + 1, False, self.observation_layer.W_parent.weight))
        return entries

    def _weight_groups_for_scope(self, scope: str) -> List[List[Array]]:
        """Weights selected by `scope` ('all' | 'rec' | 'ff'), grouped by
        owning layer -- one sub-list per `layer_labels()` entry, possibly
        empty (e.g. a layer with no weight selected under 'ff' if it has
        no parent/input matrices). Same selection semantics as
        `_weights_for_scope`, just partitioned instead of flattened.
        """
        num_groups = len(self.hidden_layers) + 2
        groups = [[] for _ in range(num_groups)]
        for layer_idx, is_rec, weight in self._weight_entries():
            selected = scope == "all" or (scope == "rec" and is_rec) or (scope == "ff" and not is_rec)
            if selected:
                groups[layer_idx].append(weight)
        return groups

    def _weight_l2_reg_by_layer(self) -> Array:
        """Per-layer breakdown of `_weight_l2_reg()`: shape (num_hidden+2,),
        same order as `layer_labels()`. `jnp.sum(...)` of this exactly
        equals `_weight_l2_reg()`'s scalar (same terms, just partitioned).
        """
        num_groups = len(self.hidden_layers) + 2
        if self.config.weight_decay <= 0.:
            return jnp.zeros(num_groups)
        groups = self._weight_groups_for_scope(self.config.weight_decay_scope)
        per_layer = [
            0.5 * self.config.weight_decay * sum(jnp.sum(W ** 2) for W in group)
            if group else jnp.asarray(0.0)
            for group in groups
        ]
        return jnp.stack(per_layer)

    def _weight_orthogonal_reg_by_layer(self) -> Array:
        """Per-layer breakdown of `_weight_orthogonal_reg()`, same Gram-matrix
        convention (narrow-side identity) applied per matrix, grouped by
        owning layer instead of summed across the whole scope.
        """
        num_groups = len(self.hidden_layers) + 2
        if self.config.orthogonal_penalty <= 0.:
            return jnp.zeros(num_groups)
        groups = self._weight_groups_for_scope(self.config.orthogonal_scope)
        per_layer = []
        for group in groups:
            if not group:
                per_layer.append(jnp.asarray(0.0))
                continue
            reg = jnp.asarray(0.0)
            for W in group:
                out_dim, in_dim = W.shape
                dim = min(out_dim, in_dim)
                gram = (W.T @ W) if in_dim <= out_dim else (W @ W.T)
                reg = reg + jnp.sum((jnp.eye(dim) - gram) ** 2)
            per_layer.append(0.5 * self.config.orthogonal_penalty * reg)
        return jnp.stack(per_layer)

    def _activity_reg_by_layer(self, states_curr: Activities) -> Array:
        """Per-layer breakdown of the activity-regularisation term: shape
        (len(states_curr),) -- NOT padded to num_hidden+2, since activity
        regularisation has no observation-layer term (states_curr never
        includes one). Callers combining this with the weight-reg
        breakdowns above must pad with one trailing zero themselves (see
        `tpch_energy_fn`).
        """
        if self.config.activity_decay <= 0.:
            return jnp.zeros(len(states_curr))
        if self.config.activity_reg_type == "l1":
            per_layer = [0.5 * self.config.activity_decay * jnp.sum(jnp.abs(s)) for s in states_curr]
        else:
            per_layer = [0.5 * self.config.activity_decay * jnp.sum(s ** 2) for s in states_curr]
        return jnp.stack(per_layer)


    def tpch_energy_fn(
        self,
        states_prev: Activities,
        states_curr: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        weight_reg_total: Optional[Array] = None,
        return_layerwise: bool = False,
    ) -> Array:
        """
        Sum over every layer's energy, calculated with prediction errors.

        F_t = sum over every layer of 1/2 ||actual state - predicted state||^2,
        plus the observation term 1/2 ||y_t - y_hat_t||^2 for `loss="mse"`,
        or the cross-entropy of `y_t` under logits `y_hat_t` for
        `loss="ce"`), plus any weight/orthogonal/activity regularisation
        configured on `self.config`.

        This is exactly eq. (19), just written for however many layers `model`
        happens to have instead of being hard-coded to the 2-layer (s, z) case.

        `weight_reg_total`: internal-use only. `_weight_l2_reg() + _weight_orthogonal_reg()`
        is a pure function of `self`'s weights -- it cannot change while
        `states_curr` is being relaxed towards a fixed point during
        inference, since weights are frozen until `update_params` runs
        between time steps. Callers doing repeated inference steps (e.g.
        `settle`, `make_activity_step`) compute it ONCE and pass it in here
        to avoid recomputing the same matrix products on every one of the
        `n_steps` relaxation iterations. Leave as `None` (the default) to
        recompute fresh from `self` -- this is what `param_grad` and any
        one-off energy readout should do.

        `return_layerwise`: if True, returns an unsummed Array of per-layer
        energies instead of the total scalar -- one entry per element of
        `states_curr` (top-to-bottom: control, then each hidden layer),
        followed by one final entry for the observation term. Order matches
        `TpchModel.layer_labels()`. Unlike an earlier version of this
        method, regularisation IS included here, attributed exactly (not
        approximated) to whichever layer's weights/activities produced it
        -- see `_weight_l2_reg_by_layer`/`_weight_orthogonal_reg_by_layer`/
        `_activity_reg_by_layer`. `jnp.sum(...)` of this array exactly
        equals the summed-scalar output (`return_layerwise=False`) on the
        same inputs. Mainly for diagnostics/plotting (e.g.
        `plot_train_energies`), not for the inference/learning gradients,
        which always use the summed scalar.

        Args:
            states_prev: Activities from previous state.
            states_curr: Activities from current state.
            observation: Current observation.
            control_input: Optional control layer input.
            weight_reg_total: Precalculated regularisation penalty to be added, else calculates weight reg automatically.
            layerwise: Whether or not to return jnp stack of per layer energies, instead of the default total energy sum. Useful for diagnostics and visualisation.

        """
        predictions, y_hat = self.predict(states_prev, states_curr, control_input)

        layer_energies = []
        for state, prediction in zip(states_curr, predictions):
            error = state - prediction
            layer_energies.append(0.5 * jnp.sum(error ** 2))

        if self.config.loss == "mse":
            y_error = observation - y_hat
            obs_energy = 0.5 * jnp.sum(y_error ** 2)
        else:  # "ce", validated in __init__
            obs_energy = -jnp.sum(observation * jax.nn.log_softmax(y_hat))
        layer_energies.append(obs_energy)

        if return_layerwise:
            weight_l2_by_layer = self._weight_l2_reg_by_layer()
            weight_orth_by_layer = self._weight_orthogonal_reg_by_layer()
            activity_by_layer = self._activity_reg_by_layer(states_curr)
            # activity has no observation-layer term (states_curr never
            # includes one) -- pad with one trailing zero to line up with
            # the weight-reg breakdowns and layer_energies, which both do.
            activity_by_layer = jnp.concatenate([activity_by_layer, jnp.zeros((1,), dtype=activity_by_layer.dtype)])
            return jnp.stack(layer_energies) + weight_l2_by_layer + weight_orth_by_layer + activity_by_layer

        if weight_reg_total is None:
            weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        
        return sum(layer_energies)+ weight_reg_total + self._activity_reg(states_curr)



    # =============================================================================
    # 4. Inference -- discretised eqs. (20)-(21): settle the current states by
    #    gradient-descending the free energy while holding weights fixed
    # =============================================================================

    def neg_activity_grad(
        self,
        states_curr: Activities,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        weight_reg_total: Optional[Array] = None,
    ) -> Activities:
        """-dF_t/d(states_curr), i.e. the direction each state should move in to
        reduce the free energy. This is the generalised form of eqs. (20)-(21):
        for a middle layer, autodiff automatically combines "how wrong was my
        own prediction" (the -eps^z term) with "how did I mess up the layer
        below me" (the +R^T(eps^z_child ... ) / C^T eps^y term) -- exactly the
        two terms the paper derives by hand, but for however many layers you
        have.

        `weight_reg_total`: see the docstring on `tpch_energy_fn` -- passed
        straight through so repeated calls (from `settle`) don't recompute
        the weight/orthogonal regularisation on every inference step.
        """
        # Get the energy function as a function of only 's' (states_current), freezing the other params as constants
        # This enables taking the derivative with respect to 's' via jax autograd, which does so wrt first positional arg by default
        energy_of_states = lambda s: self.tpch_energy_fn(
            states_prev, s, observation, control_input, weight_reg_total=weight_reg_total
        )

        # Traverses the gradient pytree and applies the jnp.negative function to every layer
        # This effectively returns the negative gradient wrt states_curr of every layer, preserving the PyTree hierarchy
        return jax.tree_util.tree_map(jnp.negative, jax.grad(energy_of_states)(states_curr))


    def infer_step(
        self,
        states_curr: Activities,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        state_lr: float = 0.1,
        weight_reg_total: Optional[Array] = None,
    ) -> Activities:
        """One Euler step of the continuous-time inference dynamics in eqs.
        (20)-(21): states_curr <- states_curr + state_lr * (-dF_t/d(states_curr)).
        `state_lr` plays the role of the (dt / tau) discretisation step.

        `weight_reg_total` is the weight regularisation penalty held static during state settling.
        """
        grad_step = self.neg_activity_grad(
            states_curr, states_prev, observation, control_input, weight_reg_total=weight_reg_total
        )
        return jax.tree_util.tree_map(lambda s, g: s + state_lr * g, states_curr, grad_step)


    def settle(
        self,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        n_steps: int = 20,
        state_lr: float = 0.1,
    ) -> Activities:
        """Full per-time-step inference: start from the feedforward guess
        (`init_activities`) and take `n_steps` of gradient-descent inference to
        let the states relax towards a local minimum of F_t before learning.
        """
        # Weights are frozen for the whole relaxation below (they only change
        # in `update_params`, between time steps), so this is computed once
        # and reused for all `n_steps` rather than recomputed from `self` on
        # every iteration -- exact, not an approximation, since the value
        # cannot change during this loop.
        weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()

        states_curr = self.init_activities(states_prev, control_input)
        for _ in range(n_steps):
            states_curr = self.infer_step(
                states_curr, states_prev, observation, control_input, state_lr, weight_reg_total=weight_reg_total
            )
        return states_curr



    # =============================================================================
    # 5. Learning -- eqs. (22)-(27): local, Hebbian weight updates at the
    #    settled states
    # =============================================================================

    def param_grad(
        self,
        states_prev: Activities,
        states_curr: Activities,
        observation: Array,
        control_input: Optional[Array] = None
    ) -> PyTree:
        """dF_t/d(weights), evaluated at the settled states. Because F_t is a sum
        of per-layer squared-error terms and each weight matrix appears in
        exactly one of those terms, this reproduces eqs. (22)-(27) --
        "post-synaptic error x f'(pre-activation) x pre-synaptic activity" --
        exactly, without us ever writing f' by hand.

        Uses `eqx.filter_grad` (rather than plain `jax.grad`) so that the
        non-array fields on the layers, like `act_fn`, are safely ignored.
        """
        # Define energy funciton as a function of 'm', where 'm' will stand in for a TpchModel when the function is called below
        energy_of_weights = lambda m: m.tpch_energy_fn(states_prev, states_curr, observation, control_input)

        # Evaluate gradient of the above function with respect to 'm' (i.e. self passed to the function as 'm')
        return eqx.filter_grad(energy_of_weights)(self)


    def update_params(
        self,
        grads: PyTree,
        optim: optax.GradientTransformation,
        opt_state: optax.OptState,
    ) -> Tuple[eqx.Module, optax.OptState]:
        """Optax-driven weight update. `optim` can be optax.sgd(lr) to recover
        plain gradient descent, or optax.adam(lr), etc. `opt_state` must be
        carried forward by the caller between steps.
        """
        updates, opt_state = optim.update(grads, opt_state, self)
        updated_model = eqx.apply_updates(self, updates)
        return updated_model, opt_state


    # =============================================================================
    # 6. Scan-fused inference, for performance
    # =============================================================================
    #
    # `settle` above uses an ordinary Python `for` loop. Once wrapped in
    # `jax.jit`, that loop gets *unrolled* at trace time: `n_steps` iterations
    # become `n_steps` literal copies of the same graph baked into the
    # compiled program. Fine for small `n_steps`, but it means (a) compile
    # time and program size both grow with `n_steps`, and (b) since tPC-H also
    # has an outer loop over TIME (states_prev at step t+1 depends on the
    # settled states_curr at step t, a genuine sequential dependency), a plain
    # Python loop over time steps means re-dispatching a whole computation
    # once per time step -- T separate calls into XLA, each paying its own
    # dispatch overhead.
    #
    # `jax.lax.scan` fixes both: the loop body compiles once and runs `length`
    # times via a native XLA loop, so compile time / program size stay flat as
    # `n_steps` or the sequence length grow, and (once jitted) the WHOLE
    # sequence -- outer time loop and inner inference-relaxation loop both --
    # becomes a single dispatched computation. That fusion, not any per-step
    # arithmetic speedup, is the main thing being "unlocked": far fewer
    # host<->device round trips, which matters most on GPU/TPU.
    #
    # These use an `optax` optimiser for the activities (rather than the fixed
    # learning rate `infer_step` above uses), matching how JPC itself drives
    # inference (`jpc.update_pc_activities`) and matching your static_pcn code:
    # positive energy gradient in, `optim.update`, `optax.apply_updates` --
    # `optax.sgd(lr)` recovers plain gradient descent as a special case.

    def make_activity_step(
        self,
        activity_optim: optax.GradientTransformation,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None
    ):
        """Builds a scan body that runs ONE inference-relaxation update at a
        single, fixed time step -- the tPC-H analogue of `make_static_inference_step`,
        using `tpch_energy_fn` (eq. 19) in place of jpc's generic `pc_energy_fn`.

        carry: (states_curr, activity_opt_state)
        scan output: states_curr at every step, so you can inspect the full
            relaxation trajectory (e.g. to check/plot convergence) if you want.
        """
        # Same reasoning as in `settle`: weights are frozen for this whole
        # scan, so the weight/orthogonal regularisation is computed once here
        # and closed over, instead of being recomputed inside `energy_fn` on
        # every one of the `n_steps` scanned iterations.
        weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        energy_fn = lambda s: self.tpch_energy_fn(
            states_prev, s, observation, control_input, weight_reg_total=weight_reg_total
        )

        def activity_step(carry, _):
            states_curr, opt_state = carry
            grads = jax.grad(energy_fn)(states_curr)  # positive grad -- see eqs. (20)-(21)
            updates, opt_state = activity_optim.update(grads, opt_state, states_curr)
            states_curr = optax.apply_updates(states_curr, updates)
            return (states_curr, opt_state), states_curr

        return activity_step


    def settle_scan(
        self,
        activity_optim: optax.GradientTransformation,
        states_prev: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
        n_steps: int = 20,
        return_layerwise: bool = False,
    ) -> Activities:
        """Scan-fused equivalent of `settle`: same feedforward init, but the
        relaxation loop runs as a single `jax.lax.scan` instead of a Python
        `for` loop.

        `return_layerwise`: `make_activity_step`'s scan already records
        `states_curr` at every step (its `ys` output). By default that
        trajectory is thrown away and only the final `states_curr` is
        returned, same signature as always. If `return_layerwise=True`,
        it's additionally run back through `tpch_energy_fn(...,
        return_layerwise=True)` via `vmap`, and this returns `(states_curr,
        energy_trace)` instead, with `energy_trace` shape (n_steps,
        num_layers+1) -- see `tpch_energy_fn`'s `return_layerwise` docstring
        for the layer order -- for diagnostics/plotting (e.g.
        `plot_train_energies`). Same pattern as `return_layerwise` on
        `tpch_energy_fn` itself: one function, a bool flag decides what
        comes back out, rather than a second near-duplicate method to
        maintain. `return_layerwise` is a plain Python bool (resolved at
        trace time under jit, like any other static flag), not a traced
        value, so it's not something you'd toggle per-call inside a scan
        or vmap -- decide it once when you call `settle_scan`.

        Costs one extra `tpch_energy_fn(..., return_layerwise=True)`
        evaluation per step when `return_layerwise=True` -- prefer the
        default (`False`) as your per-step training call, and only pass
        `return_layerwise=True` where you actually want the trace (e.g.
        every `record_every`-th training iteration for logging).
        """
        states_curr0 = self.init_activities(states_prev, control_input)
        opt_state0 = activity_optim.init(states_curr0)

        activity_step = self.make_activity_step(activity_optim, states_prev, observation, control_input)
        (states_curr, _), states_hist = jax.lax.scan(activity_step, (states_curr0, opt_state0), xs=None, length=n_steps)
        
        if not return_layerwise:
            return states_curr

        energy_trace_fn = lambda s: self.tpch_energy_fn(
            states_prev, s, observation, control_input, return_layerwise=True
        )
        # get energy trace from already processed states history
        energy_trace = jax.vmap(energy_trace_fn)(states_hist)  # (n_steps, num_layers+1)
        return states_curr, energy_trace


    def make_tpch_sequence_step(
        self,
        activity_optim: optax.GradientTransformation,
        n_infer_steps: int = 20,
    ):
        """Builds a scan body that processes ONE time step of a sequence: settles
        that step's activities (itself a fused inner scan, via `settle_scan`)
        and hands the result on as `states_prev` for the next time step.

        Meant to be used as:

            final_states, (states_history, energies) = jax.lax.scan(
                make_tpch_sequence_step(model, activity_optim, n_infer_steps),
                states_prev_0,
                xs=(x_seq, y_seq),   # each leading axis = seq_len
            )

        which fuses the ENTIRE sequence -- every time step's inference-settling
        loop included -- into a single compiled computation. Note this only
        handles inference (weights fixed); a training step would additionally
        call `param_grad` + `update_params` (or an optax param optimiser)
        on `states_curr` after each time step, or accumulate gradients across
        the whole sequence before a single weight update -- whichever fits
        your training regime.
        """
        @eqx.filter_jit
        def sequence_step(states_prev, xy_t):
            control_input_t, observation_t = xy_t
            states_curr = self.settle_scan(
                activity_optim, states_prev, observation_t, control_input_t, n_steps=n_infer_steps
            )
            energy_t = self.tpch_energy_fn(states_prev, states_curr, observation_t, control_input_t)
            # this step's settled states become next step's states_prev
            return states_curr, (states_curr, energy_t)

        return sequence_step




# ---------------------------------------------------------------------------
# Diffrax equivalent of `make_activity_step` / `settle_scan`.
#
# The optax version does discrete gradient descent on the activities:
#     s_{k+1} = s_k - lr * grad E(s_k)
# which is exactly the forward-Euler discretisation (step size `lr`) of the
# continuous-time gradient flow
#     ds/dt = -dE/ds
# Diffrax integrates ODEs like this natively: instead of a `jax.lax.scan`
# over hand-rolled optax steps, you hand it a vector field and it produces
# its own fused, jit-friendly stepping loop (`diffeqsolve`) with the same
# "compile once, run to completion in one dispatched computation" property
# `jax.lax.scan` gave `settle_scan`.
# ---------------------------------------------------------------------------
 
    def make_vector_field(
        self,
        states_prev: "Activities",
        observation: "Array",
        control_input: Optional["Array"] = None,
    ):
        """Builds the ODE vector field for one inference-relaxation trajectory.
 
        Diffrax analogue of `make_activity_step`: instead of returning a
        function that performs ONE discrete gradient-descent update, this
        returns the continuous-time vector field `ds/dt = -dE/ds` itself,
        which `diffrax.diffeqsolve` then integrates. Uses `tpch_energy_fn`
        (eq. 19) exactly as `make_activity_step` does, including freezing and
        closing over the weight/orthogonal regularisation once per trajectory
        rather than recomputing it at every solver step/stage.
 
        Args:
            states_prev: Activities from the previous time step, held fixed
                for this whole trajectory (the state the relaxation is
                settling away from), same role as in `make_activity_step`.
            observation: Observation/target this trajectory's energy term is
                being fit to.
            control_input: Optional control-layer input for this time step.
                Defaults to None.
 
        Returns:
            A function `vector_field(t, states_curr, args) -> Activities`,
            in the `(t, y, args) -> dy/dt` signature `diffrax.ODETerm`
            expects. `t` and `args` are accepted but unused: the field is
            autonomous (no explicit time-dependence) and has no extra
            per-call data beyond what's already closed over above, but
            diffrax always calls vector fields with `(t, y, args)`, so both
            stay in the signature.
        """
        weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        energy_fn = lambda s: self.tpch_energy_fn(
            states_prev, s, observation, control_input, weight_reg_total=weight_reg_total
        )
 
        def vector_field(t, states_curr, args):
            grads = jax.grad(energy_fn)(states_curr)  # positive grad -- see eqs. (20)-(21)
            # ds/dt = -dE/ds: activities flow downhill on the energy landscape
            return jax.tree_util.tree_map(lambda g: -g, grads)
 
        return vector_field
 
 
    def make_steady_state_event(
        self,
        states_prev: "Activities",
        observation: "Array",
        control_input: Optional["Array"] = None,
        tol: Optional[float] = 1e-3,
        criterion: str = "rms",
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> diffrax.Event:
        """Builds an early-termination condition for adaptive-compute relaxation.

        Returns a `diffrax.Event` that fires once the activity dynamics have
        (approximately) reached steady state, by one of three criteria (see
        `criterion` below). This is what lets `settle_diffrax`'s Mode 1 (see
        its docstring) stop integrating as soon as a given input has actually
        converged, instead of always running to `max_t1`: an "easy" input that
        settles quickly then costs far fewer solver steps than a hard one,
        without you having to guess a horizon that's long enough for the
        hardest input but wasteful for the rest.

        Note (performance): every criterion recomputes the energy gradient
        independently of the solver's own internal stage evaluations, so
        enabling the event means doing extra `grad(energy_fn)` calls beyond
        what plain integration alone would need (roughly one extra per
        accepted step, on top of however many the solver's stages already
        use; `"energy_rate"` additionally calls `tpch_energy_fn` once more
        per check). This is a correctness-first implementation, not a tuned
        one -- worth knowing about, but not worth optimising before you've
        actually profiled whether it matters for your model size and step
        counts.

        Args:
            states_prev: Activities from the previous time step, held fixed
                for this whole trajectory. Same value you'll pass to
                `settle_diffrax` for the trajectory this event is guarding.
            observation: Observation/target for this trajectory.
            control_input: Optional control-layer input for this time step.
                Defaults to None.
            criterion: Which convergence check to use:
                * `"rms"` (default) -- the original criterion. RMS of the
                  raw vector field `ds/dt`, pooled unweighted across every
                  layer's elements, compared against `tol`. Simple and
                  cheap, but the pooled sum is dominated by whichever layer
                  has the most elements, so a fixed absolute `tol`
                  implicitly means different things for different layer
                  widths -- see `"relative_rms"` for a scale-aware
                  alternative.
                * `"relative_rms"` -- per-element `ds/dt` normalised by
                  `atol + rtol * |s|` (the same convention diffrax's own
                  `PIDController` uses for step-size error control), then
                  RMS-pooled and compared against `1.0`. Requires `rtol`/
                  `atol` instead of `tol`. Scale-invariant per element, so
                  a big layer no longer swamps a small one's contribution
                  the way it can under `"rms"`.
                * `"energy_rate"` -- relative rate of VFE decrease,
                  `||ds/dt||^2 / (|E| + eps)`, compared against `tol`.
                  Note: for gradient flow (`ds/dt = -dE/ds`), the *raw*
                  (non-relative) energy rate `dE/dt` is exactly
                  `-||ds/dt||^2` -- i.e. mathematically identical to
                  `"rms"` up to a fixed rescaling by element count, not a
                  genuinely different stopping rule. This criterion is
                  therefore deliberately the *relative* rate, normalised
                  by the trajectory's own current energy magnitude, which
                  is a real difference: it adapts the threshold to how
                  large VFE currently is rather than applying one fixed
                  absolute number regardless of scale -- closer to "has
                  the objective stopped improving" than either RMS variant.
            tol: Threshold for `"rms"` and `"energy_rate"`. Unused when
                `criterion="relative_rms"` (use `rtol`/`atol` instead).
                Defaults to 1e-3. Smaller values demand tighter
                convergence (more solver steps, closer to the true fixed
                point); larger values stop earlier and cheaper, at the
                cost of a less-settled state.
            rtol, atol: Relative/absolute scale for `"relative_rms"`, in
                the same `atol + rtol * |s|` convention as diffrax's
                step-size controllers. Required (both) when
                `criterion="relative_rms"`, otherwise unused.

        Returns:
            A `diffrax.Event` wrapping a `cond_fn(t, states_curr, args,
            **kwargs) -> bool`, suitable for passing as `diffeqsolve`'s
            `event=` argument. The extra keyword arguments diffrax passes to
            `cond_fn` (`terms`, `solver`, `dt0`, etc.) aren't needed for a
            steady-state check, so they're absorbed by `**_` internally.
        """
        vector_field = self.make_vector_field(states_prev, observation, control_input)

        if criterion == "rms":
            if tol is None:
                raise ValueError("criterion='rms' requires a non-None tol")

            def steady_state_cond(t, states_curr, args, **_):
                dstates = vector_field(t, states_curr, args)
                leaves = jax.tree_util.tree_leaves(dstates)
                sq_sum = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
                n_elements = sum(leaf.size for leaf in leaves)
                rms = jnp.sqrt(sq_sum / n_elements)
                return rms < tol

        elif criterion == "relative_rms":
            if rtol is None or atol is None:
                raise ValueError("criterion='relative_rms' requires both rtol and atol")

            def steady_state_cond(t, states_curr, args, **_):
                dstates = vector_field(t, states_curr, args)
                d_leaves = jax.tree_util.tree_leaves(dstates)
                s_leaves = jax.tree_util.tree_leaves(states_curr)
                sq_sum = 0.0
                n_elements = 0
                for d_leaf, s_leaf in zip(d_leaves, s_leaves):
                    scale = atol + rtol * jnp.abs(s_leaf)
                    sq_sum = sq_sum + jnp.sum(jnp.square(d_leaf / scale))
                    n_elements = n_elements + d_leaf.size
                rms = jnp.sqrt(sq_sum / n_elements)
                return rms < 1.0

        elif criterion == "energy_rate":
            if tol is None:
                raise ValueError("criterion='energy_rate' requires a non-None tol")

            def steady_state_cond(t, states_curr, args, **_):
                dstates = vector_field(t, states_curr, args)
                leaves = jax.tree_util.tree_leaves(dstates)
                sq_norm = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)  # = |dE/dt|
                energy = self.tpch_energy_fn(states_prev, states_curr, observation, control_input)
                return sq_norm / (jnp.abs(energy) + 1e-8) < tol

        else:
            raise ValueError(
                f"unknown criterion {criterion!r}; expected 'rms', 'relative_rms', or 'energy_rate'"
            )

        return diffrax.Event(steady_state_cond)
 
 
    def settle_diffrax(
        self,
        states_prev: "Activities",
        observation: "Array",
        control_input: Optional["Array"] = None,
        max_t1: float = 20.0,
        dt0: Optional[float] = None,
        n_save: int = 20,
        solver: Optional[diffrax.AbstractSolver] = None,
        stepsize_controller: Optional[diffrax.AbstractStepSizeController] = None,
        steady_state_tol: Optional[float] = 1e-3,
        steady_state_criterion: str = "rms",
        steady_state_rtol: Optional[float] = None,
        steady_state_atol: Optional[float] = None,
        return_layerwise: bool = False,
    ) -> Union["Activities", Tuple["Activities", "Array", "Array"]]:
        """Diffrax-fused equivalent of `settle_scan`.
 
        Same feedforward init as `settle_scan`, but the relaxation "loop" is
        now continuous-time gradient flow `ds/dt = -dE/ds`
        (`make_vector_field`), integrated by a single `diffrax.diffeqsolve`
        call instead of a `jax.lax.scan` over discrete optax steps.
 
        Supports two modes, both driven by `steady_state_tol`:
 
        * **Mode 1 -- event-based adaptive compute (default).**
          `steady_state_tol` is a float. A `make_steady_state_event(...,
          tol=steady_state_tol, criterion=steady_state_criterion, ...)` is
          built and passed to `diffeqsolve` as `event=...`, so integration
          stops as soon as the network has actually converged (by whichever
          check `steady_state_criterion` selects -- `"rms"`,
          `"relative_rms"`, or `"energy_rate"`; see
          `make_steady_state_event`'s docstring), rather than always
          running to `max_t1`. Total compute then varies input-by-input:
          an easy input that settles quickly costs fewer solver steps than
          a hard one.
        * **Mode 2 -- fixed-horizon integration.** `steady_state_tol=None`.
          No event is used; `diffeqsolve` always integrates the full `[0,
          max_t1]` window. The *step size itself* can still be adaptive as
          usual (the default `stepsize_controller` is a `PIDController`,
          taking bigger or smaller internal steps depending on local error),
          it's only the early-exit behaviour that's disabled -- useful when
          you want a predictable, fixed compute budget per call, or need
          `states_hist`/`sol.ts` densely and uniformly populated for
          downstream code that isn't written to expect early termination.
 
        To recover the exact `optax.sgd`-equivalent, fixed-step behaviour of
        the very first version of this function, combine Mode 2 with a fixed
        step size: pass `solver=diffrax.Euler()`,
        `stepsize_controller=diffrax.ConstantStepSize()`, `dt0=<lr>`,
        `max_t1=n_steps * lr`, `steady_state_tol=None`. That combination was
        verified to reproduce `settle_scan(activity_optim=optax.sgd(lr),
        n_steps=n_steps)` bit-for-bit.
 
        Args:
            states_prev: Activities from the previous time step -- the prior
                this call is relaxing away from. Same argument as
                `settle_scan`.
            observation: Observation/target to fit the settled activities to.
            control_input: Optional control-layer input. Defaults to None.
            max_t1: Upper bound on the integration horizon -- "how long the
                network is allowed to relax for", in continuous time rather
                than a discrete step count. Plays the role `n_steps` played
                in `settle_scan`, but as a time budget rather than a literal
                iteration count: in Mode 1 this is a ceiling that's rarely
                actually reached (the event usually fires first); in Mode 2
                it's always reached exactly. Defaults to 20.0.
            dt0: Initial step size. `None` (the default) lets the adaptive
                `stepsize_controller` pick one itself, the usual diffrax
                default; only worth setting explicitly for a
                `ConstantStepSize` controller, where it's the (fixed) step
                size for the whole solve.
            n_save: Number of evenly spaced diagnostic checkpoints between
                `t0=0` and `max_t1` at which `states_curr` is additionally
                recorded (via `SaveAt(ts=...)`), for building the
                `return_layerwise` trace. This is *not* a solver step count
                -- the solver itself may take anywhere from a handful to
                hundreds of internal steps between two consecutive save
                points, depending on `solver`/`stepsize_controller`; `n_save`
                only controls how finely you get to inspect the trajectory
                afterwards, independent of how the solver actually got there.
                Larger values give a smoother/finer trace for plotting at the
                cost of `n_save` extra `tpch_energy_fn(...,
                return_layerwise=True)` evaluations when `return_layerwise=
                True` (see below); has no effect at all on the returned
                `states_curr` or on solver accuracy. Defaults to 20.
            solver: Diffrax ODE solver. `None` (the default) resolves to
                `diffrax.Heun()`, a 2nd-order explicit Runge-Kutta method
                with an embedded error estimate suitable for adaptive
                stepping (matching the `jpc` library's
                `solve_pc_inference` default). Explicit `diffrax.Euler()` +
                `diffrax.ConstantStepSize()` recovers plain, literal gradient
                descent (see module docstring above).
            stepsize_controller: Diffrax step-size controller. `None` (the
                default) resolves to `diffrax.PIDController(rtol=1e-3,
                atol=1e-3)`. Pass `diffrax.ConstantStepSize()` for fixed-step
                integration (required to exactly reproduce `optax.sgd`, see
                above); note that its tolerances also implicitly determine
                how "settled" a state needs to be before Mode 1's steady-state
                event can fire, since the local error the controller is
                trying to keep within tolerance and the vector-field norm the
                event checks both shrink together as the trajectory
                approaches its fixed point.
            steady_state_tol: RMS (or, per `steady_state_criterion`,
                relative-RMS/energy-rate) threshold for early termination
                -- see "Mode 1" / "Mode 2" above. A float (e.g. the
                default `1e-3`) selects Mode 1; `None` selects Mode 2.
                Still the sole Mode 1/Mode 2 switch regardless of
                `steady_state_criterion` -- when `steady_state_criterion=
                "relative_rms"`, this value's numeric threshold is unused
                (see below), but it must still be non-`None` to select
                Mode 1.
            steady_state_criterion: Which convergence check Mode 1 uses --
                `"rms"` (default), `"relative_rms"`, or `"energy_rate"`;
                see `make_steady_state_event`'s docstring for what each
                one measures and why they can disagree on when an input
                has "actually" converged. No effect in Mode 2.
            steady_state_rtol, steady_state_atol: Relative/absolute scale
                for `steady_state_criterion="relative_rms"`, passed
                straight through to `make_steady_state_event`'s `rtol`/
                `atol`. Required (both) for that criterion, otherwise
                unused.
            return_layerwise: If True, additionally returns a per-layer
                energy trace and the times it was recorded at (see Returns
                below). Costs one extra `tpch_energy_fn(...,
                return_layerwise=True)` evaluation per save point (`n_save +
                2` of them; see `energy_trace` below) -- prefer the default
                (False) as your per-step training call, and only pass
                `return_layerwise=True` where you actually want the trace
                (e.g. every `record_every`-th training iteration for
                logging, mirroring `settle_scan`'s own usage pattern).
 
        Returns:
            If `return_layerwise` is False: `states_curr`, the settled
            Activities -- the true converged (Mode 1) or `max_t1`-truncated
            (Mode 2) state, exactly analogous to `settle_scan`'s return
            value.
 
            If `return_layerwise` is True: a 3-tuple `(states_curr,
            energy_trace, ts)`:
 
            * `states_curr`: as above.
            * `energy_trace`: per-layer energy breakdown at each of the
              `n_save + 2` save points (the `n_save + 1`-point `ts` grid,
              plus one extra entry for the exact moment of termination --
              see `ts` below), shape `(n_save + 2, num_layers + 1)`. See
              `tpch_energy_fn`'s `return_layerwise` docstring for the layer
              order.
            * `ts`: the times each row of `energy_trace` (and, internally,
              `states_hist`) was actually recorded at, shape `(n_save + 2,)`.
              In Mode 2, or in Mode 1 when the trajectory never converges
              before `max_t1`, every entry is finite. In Mode 1 with early
              convergence, entries at or after the stopping time are `inf`
              (diffrax leaves the unused portion of its fixed-size output
              buffer unfilled rather than back-filling it) -- this is *not*
              a bug, and it's exactly the signal you need: mask both `ts`
              and `energy_trace` with `jnp.isfinite(ts)` before
              plotting/aggregating, the same way you'd trim a variable-
              length trajectory from any other early-terminating solve. This
              is deliberately returned explicitly (rather than leaving the
              caller to reconstruct it) so downstream code -- e.g.
              `plot_train_energies` -- can tell which points in the trace
              actually occurred without re-deriving that from `energy_trace`
              itself.
        """
        if solver is None:
            solver = diffrax.Heun()
        if stepsize_controller is None:
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)
 
        states_curr0 = self.init_activities(states_prev, control_input)
        vector_field = self.make_vector_field(states_prev, observation, control_input)
 
        event = None
        if steady_state_tol is not None:  # Mode 1: event-based adaptive compute
            event = self.make_steady_state_event(
                states_prev, observation, control_input,
                tol=steady_state_tol, criterion=steady_state_criterion,
                rtol=steady_state_rtol, atol=steady_state_atol,
            )
        # else: Mode 2 -- fixed-horizon integration, `event=None` below
 
        t0 = 0.0
        ts = jnp.linspace(t0, max_t1, n_save + 1)
 
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            solver,
            t0=t0,
            t1=max_t1,
            dt0=dt0,
            y0=states_curr0,
            saveat=diffrax.SaveAt(t1=True, ts=ts),
            stepsize_controller=stepsize_controller,
            event=event,
        )
        states_hist = sol.ys  # (n_save + 2, ...) -- the ts grid, plus the t1/event entry
 
        # The true final state is whichever save point has the largest
        # *finite* recorded time -- this also just picks index -1 when every
        # grid point is finite (Mode 2, or an unconverged Mode 1 run), so it
        # subsumes plain "last entry" as a special case rather than needing a
        # separate code path per mode.
        finite_mask = jnp.isfinite(sol.ts)
        final_idx = jnp.argmax(jnp.where(finite_mask, sol.ts, -jnp.inf))
        states_curr = jax.tree_util.tree_map(lambda x: x[final_idx], states_hist)
 
        if not return_layerwise:
            return states_curr
 
        energy_trace_fn = lambda s: self.tpch_energy_fn(
            states_prev, s, observation, control_input, return_layerwise=True
        )
        # get energy trace from already processed states history -- entries
        # past an early steady-state stop will be inf/nan; `sol.ts` is
        # returned alongside so callers can mask them (see docstring above)
        # rather than having to reconstruct which points are valid themselves.
        energy_trace = jax.vmap(energy_trace_fn)(states_hist)  # (n_save + 2, num_layers + 1)
        return states_curr, energy_trace, sol.ts

    # =============================================================================
    # 7. Saving and Loading 
    # =============================================================================
    @classmethod
    def from_config(cls, config, *, key) -> "TpchModel":
        """
        (Re)build model from config
        """
        return cls(
            control_layer_size=config.control_layer_size,
            hidden_sizes=config.hidden_sizes,
            obs_size=config.obs_size,
            key=key,
            act_fn=config.act_fn, # plaintext name of act_fn, used with registry
            input_size=config.input_size,
            loss=config.loss,
            weight_decay=config.weight_decay,
            weight_decay_scope=config.weight_decay_scope,
            orthogonal_penalty=config.orthogonal_penalty,
            orthogonal_scope=config.orthogonal_scope,
            activity_decay=config.activity_decay,
            activity_reg_type=config.activity_reg_type,
        )

    
    @classmethod
    def layer_labels(cls, config) -> List[str]:
        """
        Labels matching `tpch_energy_fn(..., return_layerwise=True)`'s output
        order: control layer, then each hidden layer top-to-bottom, then
        the observation/output term.
        """
        return (
            ["Control"]
            + [f"Hidden {i + 1}" for i in range(len(config.hidden_sizes))]
            + ["Observation"]
        )


    @classmethod
    def zero_activities(cls, config: TpchConfig) -> Activities:
        """
        Builds activities skeleton for TpchModel loading.
        """
        sizes = [config.control_layer_size, *config.hidden_sizes]
        return [jnp.zeros(s) for s in sizes]



# =============================================================================
# 8. lax.scan training helpers
# =============================================================================

def make_train_step(param_optim: optax.GradientTransformation, activity_optim: optax.GradientTransformation, n_infer_steps: int, control_input: Optional[Array] = None):
    """Builds one fully-jitted training step: settle -> log-quantities -> weight update.

    The returned `train_step` is traced once per distinct `return_layerwise`
    value on first use, then reused for every subsequent call with that same
    value -- not re-traced per training iteration. Pass `return_layerwise=True`
    on the iterations where you want a per-layer energy trace for
    `plot_train_energies` (e.g. every `record_every`-th frame); the default
    `False` path stays on its own, cheaper compiled trace the rest of the
    time. This costs exactly two compiles total across a whole run (one per
    value ever passed), not one per iteration.

    Args:
        param_optim: Optax transform used for the weight update (`param_grad`
            -> `param_optim.update` -> `eqx.apply_updates`).
        activity_optim: Optax transform used for the inference/settling loop,
            passed straight through to `model.settle_scan`.
        n_infer_steps: Number of relaxation steps per call, i.e. `settle_scan`'s
            `n_steps`. Fixed at build time (not a `train_step` argument)
            because it becomes `jax.lax.scan`'s `length=` internally, which
            must be a concrete Python int known at trace time.
        control_input: Optional control-layer input, constant for the whole
            training run and closed over here rather than passed to
            `train_step` each call. Pass an actual array instead of `None`
            if it needs to vary per frame in your setup.

    Returns:
        train_step: A function with signature
            `train_step(model, param_opt_state, states_prev, y, return_layerwise=False)`
            -> `(model, param_opt_state, states_curr, y_hat_before, y_hat_after,
            energy_before, energy_after, energy_trace)`, where:
            - `model`: the model with one weight update applied.
            - `param_opt_state`: updated optimizer state for `param_optim`.
            - `states_curr`: the settled states, to pass back in as
                `states_prev` for the next call.
            - `y_hat_before`, `y_hat_after`: observation-layer predictions
                from the pre- and post-inference states, using the
                pre-update weights.
            - `energy_before`, `energy_after`: scalar total energies at
                those same two points.
            - `energy_trace`: per-layer energy breakdown across all
                `n_infer_steps` relaxation steps (shape `(n_infer_steps,
                num_layers + 1)`, see `settle_scan`'s docstring for row/column
                order) if `return_layerwise=True`, else `None`.
    """
    @eqx.filter_jit
    def train_step(model: TpchModel, param_opt_state, states_prev, y, return_layerwise: bool = False):
        states_curr_init = model.init_activities(states_prev, control_input)
        _, y_hat_before = model.predict(states_prev, states_curr_init, control_input)
        energy_before = model.tpch_energy_fn(states_prev, states_curr_init, y, control_input)

        settle_result = model.settle_scan(
            activity_optim, states_prev, y, control_input, n_steps=n_infer_steps, return_layerwise=return_layerwise
        )
        states_curr, energy_trace = settle_result if return_layerwise else (settle_result, None)

        _, y_hat_after = model.predict(states_prev, states_curr, control_input)
        energy_after = model.tpch_energy_fn(states_prev, states_curr, y, control_input)

        grads = model.param_grad(states_prev, states_curr, y, control_input)
        updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, param_opt_state, states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace

    return train_step


def make_train_run(param_optim, activity_optim, n_infer_steps, run_length, control_input=None):
    """Builds one fully-jitted, multi-frame training run: settle -> learn,
    repeated for `run_length` consecutive frames, fused into a single
    `jax.lax.scan` (and hence one JIT compile for the whole block) instead
    of one `eqx.filter_jit` call per frame the way `make_train_step` works.

    This is the same underlying computation as calling `make_train_step`'s
    `train_step` in a Python loop `run_length` times, just fused so XLA
    compiles and executes the whole block as one program -- verified to
    produce identical energies, settled states, and updated weights.

    Use this for two related patterns:
      - A fully jitted whole-training-run: pass `run_length=len(frames)`
        and call it once. Fastest option, at the cost of no side effects
        (plotting, checkpointing) until the whole run finishes.
        `y_hat_before`/`y_hat_after`/`energies_before`/`energies_after`/
        `energy_traces` are all materialized for every frame
        simultaneously, but each is cheap per frame (a prediction, a
        scalar, and a handful of per-layer scalars respectively) --
        device memory isn't the practical constraint here, the lack of
        any way to checkpoint or inspect progress mid-run is.
      - The "goldilocks" pattern: pass `run_length=record_every` and call
        this repeatedly from an outer Python loop, doing plotting/
        checkpointing in the gaps between calls (ordinary Python there --
        `model` is a concrete value at that point, not a tracer). One
        compile total (traced once, reused every block, same static-
        argument caching as `make_train_step`), and memory stays bounded
        by `run_length` rather than total training length.

    Args:
        param_optim: Optax transform used for the weight update at every
            frame in the run.
        activity_optim: Optax transform used for the inference/settling
            loop at every frame, passed straight through to
            `model.settle_scan`.
        n_infer_steps: Number of relaxation steps per frame, i.e.
            `settle_scan`'s `n_steps`. Fixed at build time, same reasoning
            as `make_train_step`: it becomes part of a `jax.lax.scan`
            `length=` internally (inside `settle_scan` itself), which must
            be a concrete Python int known at trace time.
        run_length: Number of frames processed per call to the returned
            `train_run` -- the length of this function's own outer
            `jax.lax.scan`. Also fixed at build time and for the same
            reason. Set to the length of one `ys` block you'll pass in.
        control_input: Optional control-layer input, constant for every
            frame in the run and closed over here rather than passed to
            `train_run` each call. Pass an actual array instead of `None`
            if it needs to vary per frame in your setup.

    Returns:
        train_run: A function with signature
            `train_run(model, param_opt_state, states_prev, ys, return_layerwise=False)`
            -> `(model, param_opt_state, states_curr, y_hat_before,
            y_hat_after, energies_before, energies_after, energy_traces)`,
            where `ys` is an array of `run_length` observations (leading
            axis = `run_length`), and:
              - `model`: the model after `run_length` weight updates, one
                per frame.
              - `param_opt_state`: `param_optim`'s state after those updates.
              - `states_curr`: the last frame's settled states, to pass
                back in as `states_prev` for the next call.
              - `y_hat_before`, `y_hat_after`: stacked per-frame
                observation-layer predictions from the pre- and
                post-inference states, shape `(run_length, obs_size)`,
                using each frame's pre-update weights.
              - `energies_before`, `energies_after`: stacked per-frame
                scalar total energies at those same two points (pre-update
                weights throughout), shape `(run_length,)` each -- exact
                per-frame analogue of `make_train_step`'s `energy_before`/
                `energy_after`, not to be confused with `energy_traces`
                (below), which measures something related but distinct:
                `energy_traces[i, 0]` is the energy after the FIRST
                relaxation step, whereas `energies_before[i]` is measured
                at zero relaxation steps (the raw feedforward guess) --
                close but not the same quantity.
              - `energy_traces`: per-frame, per-relaxation-step layerwise
                energy breakdown if `return_layerwise=True` (shape
                `(run_length, n_infer_steps, num_layers + 1)` -- one
                scalar per layer per step per frame, see
                `tpch_energy_fn`'s `return_layerwise` docstring for the
                layer order -- cheap even for a whole training run, since
                each entry is a single float, not a full state vector),
                else `None`. Same call-time-bool pattern as
                `make_train_step`: traced once per distinct value passed,
                cached thereafter, so toggling it doesn't cost a retrace
                per call. Realistically only useful at goldilocks-sized
                `run_length` regardless of its own (small) cost, since
                that's what periodic checkpointing/plotting already
                requires -- there's no way to interrupt a `scan` mid-run
                to look at it anyway.
    """
    @eqx.filter_jit
    def train_run(model: TpchModel, param_opt_state, states_prev: Activities, ys: Array, return_layerwise: bool = False):
        def step(carry, y_t):
            model, param_opt_state, states_prev = carry

            states_curr_init = model.init_activities(states_prev, control_input)
            _, y_hat_before = model.predict(states_prev, states_curr_init, control_input)
            energy_before_t = model.tpch_energy_fn(states_prev, states_curr_init, y_t, control_input)

            settle_result = model.settle_scan(
                activity_optim, states_prev, y_t, control_input, n_steps=n_infer_steps, return_layerwise=return_layerwise
            )
            states_curr, energy_trace_t = settle_result if return_layerwise else (settle_result, None)

            _, y_hat_after = model.predict(states_prev, states_curr, control_input)
            energy_after_t = model.tpch_energy_fn(states_prev, states_curr, y_t, control_input)

            grads = model.param_grad(states_prev, states_curr, y_t, control_input)
            updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
            model = eqx.apply_updates(model, updates)

            return (model, param_opt_state, states_curr), (
                y_hat_before, y_hat_after, energy_before_t, energy_after_t, energy_trace_t
            )

        (model, param_opt_state, states_curr), (y_hat_before, y_hat_after, energies_before, energies_after, energy_traces) = jax.lax.scan(
            step, (model, param_opt_state, states_prev), xs=ys, length=run_length
        )
        return model, param_opt_state, states_curr, y_hat_before, y_hat_after, energies_before, energies_after, energy_traces

    return train_run


# =============================================================================
# 9. Diffrax training helpers
# ============================================================================= 
# `_train_frame_pre` / `_train_frame_post` factor out the parts of a
# training frame that are IDENTICAL regardless of which settling mechanism
# produced `states_curr`: the pre-inference prediction/energy, and the
# post-inference prediction/energy/weight-update. Confirmed via
# `param_grad` (eqx.filter_grad over `tpch_energy_fn` evaluated AT the
# already-settled `states_curr`, not backprop through however it was
# found) that neither half differentiates through the settling process --
# so `settle_scan` and `settle_diffrax` really are drop-in-different only
# in the middle, and the shared helpers below are the actual reason
# `make_train_step`/`make_train_step_diffrax` (and their `_run` cousins)
# can't silently drift apart on the boilerplate they share.
# ---------------------------------------------------------------------------

def _train_frame_pre(model, states_prev, y, control_input):
    """Pre-inference half of one training frame: feedforward init, then
    the prediction/energy of that raw (pre-settling) guess. Identical for
    every settling mechanism, since it runs entirely before `settle_scan`
    / `settle_diffrax` is even called.
    """
    states_curr_init = model.init_activities(states_prev, control_input)
    _, y_hat_before = model.predict(states_prev, states_curr_init, control_input)
    energy_before = model.tpch_energy_fn(states_prev, states_curr_init, y, control_input)
    return y_hat_before, energy_before


def _train_frame_post(model, param_optim, param_opt_state, states_prev, states_curr, y, control_input):
    """Post-inference half of one training frame: prediction/energy of the
    settled state, weight gradient at that state, and the optax weight
    update. Identical for every settling mechanism, since `param_grad` is
    evaluated at `states_curr` as a plain value -- it doesn't matter
    whether `states_curr` came from a `jax.lax.scan` of optax activity
    steps or a `diffrax.diffeqsolve` gradient-flow integration, only that
    it's a settled `Activities` pytree.
    """
    _, y_hat_after = model.predict(states_prev, states_curr, control_input)
    energy_after = model.tpch_energy_fn(states_prev, states_curr, y, control_input)

    grads = model.param_grad(states_prev, states_curr, y, control_input)
    updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, param_opt_state, y_hat_after, energy_after


def make_train_step_diffrax(
    param_optim: optax.GradientTransformation,
    max_t1: float = 20.0,
    dt0: Optional[float] = None,
    n_save: int = 20,
    solver: Optional[diffrax.AbstractSolver] = None,
    stepsize_controller: Optional[diffrax.AbstractStepSizeController] = None,
    steady_state_tol: Optional[float] = 1e-3,
    steady_state_criterion: str = "rms",
    steady_state_rtol: Optional[float] = None,
    steady_state_atol: Optional[float] = None,
    control_input: Optional["Array"] = None,
):
    """Diffrax analogue of `make_train_step`: builds one fully-jitted
    training step using `model.settle_diffrax` in place of
    `model.settle_scan` for the inference/settling half.

    There's no `activity_optim` here (unlike `make_train_step`) -- with
    `settle_diffrax`, activity relaxation is governed by `solver` /
    `stepsize_controller` (how the ODE is integrated) and `steady_state_tol`
    (whether/when it stops early), not by an optax transform. Everything
    else about this function -- the pre-inference prediction/energy, the
    weight gradient at the settled state, the optax weight update -- is
    identical to `make_train_step`, factored into `_train_frame_pre` /
    `_train_frame_post` so the two variants share that logic outright
    rather than maintaining two copies of it.

    Same call-time-bool compile-caching behaviour as `make_train_step`:
    the returned `train_step` is traced once per distinct `return_layerwise`
    value on first use, then reused thereafter -- two compiles total across
    a run, not one per iteration.

    Args:
        param_optim: Optax transform used for the weight update
            (`param_grad` -> `param_optim.update` -> `eqx.apply_updates`).
        max_t1: Integration horizon passed straight through to
            `model.settle_diffrax`. Fixed at build time (not a `train_step`
            argument), matching how `make_train_step` fixes `n_infer_steps`
            -- kept consistent with the rest of `settle_diffrax`'s knobs
            below rather than because it's strictly required to be static
            (unlike `n_infer_steps`, which must be a concrete Python int
            because it becomes `jax.lax.scan`'s `length=`; `max_t1` has no
            such constraint on its own).
        dt0: Initial step size, passed straight through to
            `model.settle_diffrax`. Defaults to None (adaptive controller
            picks one).
        n_save: Number of diagnostic checkpoints for the `return_layerwise`
            trace, passed straight through to `model.settle_diffrax`. This
            one DOES need to be fixed at build time: it determines the
            shape of `settle_diffrax`'s `SaveAt(ts=...)` grid, and hence
            the shape of everything this function returns that's derived
            from it (`energy_trace`, `ts` below).
        solver: Diffrax ODE solver, passed straight through to
            `model.settle_diffrax`. `None` resolves to `diffrax.Heun()`
            there. Fixed at build time since it's a solver object, not an
            array.
        stepsize_controller: Diffrax step-size controller, passed straight
            through to `model.settle_diffrax`. `None` resolves to
            `diffrax.PIDController(rtol=1e-3, atol=1e-3)` there.
        steady_state_tol: Early-termination threshold passed straight
            through to `model.settle_diffrax` -- selects Mode 1 (float) or
            Mode 2 (`None`); see that function's docstring.
        steady_state_criterion, steady_state_rtol, steady_state_atol:
            Passed straight through to `model.settle_diffrax` -- pick and
            parameterise which convergence check Mode 1 uses; see
            `model.make_steady_state_event`'s docstring.
        control_input: Optional control-layer input, constant for the
            whole training run and closed over here rather than passed to
            `train_step` each call, same as in `make_train_step`.

    Returns:
        train_step: A function with signature
            `train_step(model, param_opt_state, states_prev, y, return_layerwise=False)`
            -> `(model, param_opt_state, states_curr, y_hat_before,
            y_hat_after, energy_before, energy_after, energy_trace, ts)`,
            where the first seven entries match `make_train_step`'s
            `train_step` exactly (same meaning, same shapes), and:
            - `energy_trace`: per-layer energy breakdown at each of the
                `n_save + 2` save points if `return_layerwise=True`, else
                `None` -- see `settle_diffrax`'s `Returns` for shape and
                the inf/nan-padding caveat on early-converged frames.
            - `ts`: the times each row of `energy_trace` was actually
                recorded at, shape `(n_save + 2,)`, or `None` if
                `return_layerwise=False`. Mask both with
                `jnp.isfinite(ts)` before plotting/aggregating -- this is
                exactly `settle_diffrax`'s own `ts` return, threaded
                through unchanged, for the same reason it's returned
                there: so the caller doesn't have to reconstruct validity
                from `energy_trace`'s own inf pattern.
    """
    if solver is None:
        solver = diffrax.Heun()
    if stepsize_controller is None:
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)

    @eqx.filter_jit
    def train_step(model: "TpchModel", param_opt_state, states_prev, y, return_layerwise: bool = False):
        y_hat_before, energy_before = _train_frame_pre(model, states_prev, y, control_input)

        settle_result = model.settle_diffrax(
            states_prev, y, control_input,
            max_t1=max_t1, dt0=dt0, n_save=n_save,
            solver=solver, stepsize_controller=stepsize_controller,
            steady_state_tol=steady_state_tol, steady_state_criterion=steady_state_criterion,
            steady_state_rtol=steady_state_rtol, steady_state_atol=steady_state_atol,
            return_layerwise=return_layerwise,
        )
        if return_layerwise:
            states_curr, energy_trace, ts = settle_result
        else:
            states_curr, energy_trace, ts = settle_result, None, None

        model, param_opt_state, y_hat_after, energy_after = _train_frame_post(
            model, param_optim, param_opt_state, states_prev, states_curr, y, control_input
        )

        return (
            model, param_opt_state, states_curr, y_hat_before, y_hat_after,
            energy_before, energy_after, energy_trace, ts,
        )

    return train_step


def make_train_run_diffrax(
    param_optim: optax.GradientTransformation,
    run_length: int,
    max_t1: float = 20.0,
    dt0: Optional[float] = None,
    n_save: int = 20,
    solver: Optional[diffrax.AbstractSolver] = None,
    stepsize_controller: Optional[diffrax.AbstractStepSizeController] = None,
    steady_state_tol: Optional[float] = 1e-3,
    steady_state_criterion: str = "rms",
    steady_state_rtol: Optional[float] = None,
    steady_state_atol: Optional[float] = None,
    control_input: Optional["Array"] = None,
):
    """Diffrax analogue of `make_train_run`: builds one fully-jitted,
    multi-frame training run using `model.settle_diffrax` in place of
    `model.settle_scan`, `run_length` frames fused into a single outer
    `jax.lax.scan` exactly as `make_train_run` does.

    This outer fusion is safe with diffrax settling underneath for two
    independent reasons, both confirmed rather than assumed:

    1. No adjoint machinery is involved. `param_grad` calls
       `eqx.filter_grad` on `tpch_energy_fn` evaluated AT `states_curr` as
       a plain value -- the settling process (`settle_diffrax`'s
       `diffeqsolve` call included) never sits inside anything
       `jax.grad`/`eqx.filter_grad` differentiates through, in either this
       function or `make_train_step_diffrax`. So there's no
       `RecursiveCheckpointAdjoint`-vs-`BacksolveAdjoint` decision to make
       here at all -- `diffeqsolve` runs in pure forward/inference mode,
       like any other array-producing op inside a `scan` body.
    2. Fusing frames into a scan doesn't blunt Mode 1's early-exit benefit.
       `diffeqsolve` is built on a shape-static `lax.while_loop` with a
       fixed `max_steps` ceiling -- the *compiled* program has one fixed
       shape (which is what makes it composable with `scan` at all), but
       the loop's runtime exit is still per-call: each frame in the scan
       gets its own independent early termination when its steady-state
       event fires, the same as it would calling `settle_diffrax` frame-
       by-frame outside a scan. (This is specifically unlike `vmap`-ing a
       batch of settles, where every element would be forced to run to
       whichever one converges last.)

    Args:
        param_optim: Optax transform used for the weight update at every
            frame in the run.
        run_length: Number of frames processed per call to the returned
            `train_run` -- the length of this function's own outer
            `jax.lax.scan`. Fixed at build time, same reasoning as
            `make_train_run`: it's a concrete Python int needed for
            `scan`'s `length=`. Set to the length of one `ys` block you'll
            pass in.
        max_t1, dt0, n_save, solver, stepsize_controller, steady_state_tol,
        steady_state_criterion, steady_state_rtol, steady_state_atol:
            Passed straight through to `model.settle_diffrax` at every
            frame -- see `make_train_step_diffrax`'s Args for what each
            one does and why it's fixed at build time here too.
        control_input: Optional control-layer input, constant for every
            frame in the run and closed over here, same as
            `make_train_run`.

    Returns:
        train_run: A function with signature
            `train_run(model, param_opt_state, states_prev, ys, return_layerwise=False)`
            -> `(model, param_opt_state, states_curr, y_hat_before,
            y_hat_after, energies_before, energies_after, energy_traces,
            ts_traces)`, where `ys` is an array of `run_length`
            observations (leading axis = `run_length`), and the first
            seven entries match `make_train_run`'s `train_run` exactly
            (same meaning, same shapes) except:
              - `energy_traces`: shape `(run_length, n_save + 2,
                num_layers + 1)` if `return_layerwise=True`, else `None`
                -- the `settle_scan`-based version's analogous output has
                shape `(run_length, n_infer_steps, num_layers + 1)`; the
                middle dimension differs in both size (`n_save + 2` vs.
                `n_infer_steps`) and meaning (fixed diagnostic-checkpoint
                count vs. literal relaxation-step count), per
                `settle_diffrax`'s docstring.
              - `ts_traces`: shape `(run_length, n_save + 2)` if
                `return_layerwise=True`, else `None` -- per-frame version
                of `settle_diffrax`'s `ts`, stacked the same way
                `energy_traces` is. Frame `i`'s `ts_traces[i]` may contain
                `inf` entries from that frame's own early termination,
                independent of every other frame's; mask each frame with
                `jnp.isfinite(ts_traces[i])` before plotting/aggregating,
                same caveat as `make_train_step_diffrax`.
    """
    if solver is None:
        solver = diffrax.Heun()
    if stepsize_controller is None:
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)

    @eqx.filter_jit
    def train_run(model: "TpchModel", param_opt_state, states_prev: "Activities", ys: "Array", return_layerwise: bool = False):
        def step(carry, y_t):
            model, param_opt_state, states_prev = carry

            y_hat_before, energy_before_t = _train_frame_pre(model, states_prev, y_t, control_input)

            settle_result = model.settle_diffrax(
                states_prev, y_t, control_input,
                max_t1=max_t1, dt0=dt0, n_save=n_save,
                solver=solver, stepsize_controller=stepsize_controller,
                steady_state_tol=steady_state_tol, steady_state_criterion=steady_state_criterion,
                steady_state_rtol=steady_state_rtol, steady_state_atol=steady_state_atol,
                return_layerwise=return_layerwise,
            )
            if return_layerwise:
                states_curr, energy_trace_t, ts_t = settle_result
            else:
                states_curr, energy_trace_t, ts_t = settle_result, None, None

            model, param_opt_state, y_hat_after, energy_after_t = _train_frame_post(
                model, param_optim, param_opt_state, states_prev, states_curr, y_t, control_input
            )

            return (model, param_opt_state, states_curr), (
                y_hat_before, y_hat_after, energy_before_t, energy_after_t, energy_trace_t, ts_t
            )

        (model, param_opt_state, states_curr), (
            y_hat_before, y_hat_after, energies_before, energies_after, energy_traces, ts_traces
        ) = jax.lax.scan(step, (model, param_opt_state, states_prev), xs=ys, length=run_length)

        return (
            model, param_opt_state, states_curr, y_hat_before, y_hat_after,
            energies_before, energies_after, energy_traces, ts_traces,
        )

    return train_run