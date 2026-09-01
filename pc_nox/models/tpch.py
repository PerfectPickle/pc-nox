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
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Callable, List, Sequence, Tuple, Optional
from pathlib import Path


# =============================================================================
# 0. Config definition
# =============================================================================

@dataclass
class TpchConfig:
    control_layer_size: int
    hidden_sizes: List[int]
    obs_size: int  # observation / sensory dim
    input_size: int = 0  # control input (optional)
    act_fn: str = "tanh"


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
        hidden_sizes: List of hidden layer widths, number of elements determines the number of hidden layers.
        obs_size: Width of the observation / output.
        act_fn: Activation function used by the control and hidden layers.
        key: Jax pseudo random number generator key used for layer initilizations.
        input_size: Width of input provided to control layer, defaults to 0.
    """
    model_type: ClassVar[str] = "tpch"
    config_cls: ClassVar[type] = TpchConfig

    control_layer: TpchControlLayer
    hidden_layers: List[TpchHiddenLayer]
    observation_layer: TpchObservationLayer

    def __init__(
        self,
        control_layer_size: int,
        hidden_sizes: Sequence[int],
        obs_size: int,
        key: PRNGKeyArray,
        act_fn: Callable = jnp.tanh,
        input_size: Optional[int] = 0, # optional control input
    ):
        n_hidden = len(hidden_sizes)
        key_control, *hidden_keys, key_obs = jr.split(key, 2 + n_hidden)

        self.control_layer = TpchControlLayer(
            state_size=control_layer_size, input_size=input_size, act_fn=act_fn, key=key_control
        )

        hidden_layers = []
        parent_size = control_layer_size  # the first hidden layer's parent is the control layer
        for size, hkey in zip(hidden_sizes, hidden_keys):
            hidden_layers.append(
                TpchHiddenLayer(state_size=size, parent_size=parent_size, act_fn=act_fn, key=hkey)
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

    def tpch_energy_fn(
        self,
        states_prev: Activities,
        states_curr: Activities,
        observation: Array,
        control_input: Optional[Array] = None,
    ) -> Array:
        """F_t = sum over every layer of 1/2 ||actual state - predicted state||^2,
        plus the observation term 1/2 ||y_t - y_hat_t||^2.

        This is exactly eq. (19), just written for however many layers `model`
        happens to have instead of being hard-coded to the 2-layer (s, z) case.
        """
        predictions, y_hat = self.predict(states_prev, states_curr, control_input)

        energy = jnp.asarray(0.0)
        for state, prediction in zip(states_curr, predictions):
            error = state - prediction
            energy = energy + 0.5 * jnp.sum(error ** 2)

        y_error = observation - y_hat
        energy = energy + 0.5 * jnp.sum(y_error ** 2)
        return energy



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
    ) -> Activities:
        """-dF_t/d(states_curr), i.e. the direction each state should move in to
        reduce the free energy. This is the generalised form of eqs. (20)-(21):
        for a middle layer, autodiff automatically combines "how wrong was my
        own prediction" (the -eps^z term) with "how did I mess up the layer
        below me" (the +R^T(eps^z_child ... ) / C^T eps^y term) -- exactly the
        two terms the paper derives by hand, but for however many layers you
        have.
        """
        # Get the energy function as a function of only 's' (states_current), freezing the other params as constants
        # This enables taking the derivative with respect to 's' via jax autograd, which does so wrt first positional arg by default
        energy_of_states = lambda s: self.tpch_energy_fn(states_prev, s, observation, control_input)

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
    ) -> Activities:
        """One Euler step of the continuous-time inference dynamics in eqs.
        (20)-(21): states_curr <- states_curr + state_lr * (-dF_t/d(states_curr)).
        `state_lr` plays the role of the (dt / tau) discretisation step.
        """
        grad_step = self.neg_activity_grad(states_curr, states_prev, observation, control_input)
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
        states_curr = self.init_activities(states_prev, control_input)
        for _ in range(n_steps):
            states_curr = self.infer_step(states_curr, states_prev, observation, control_input, state_lr)
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
        energy_fn = lambda s: self.tpch_energy_fn(states_prev, s, observation, control_input)

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
    ) -> Activities:
        """Scan-fused equivalent of `settle`: same feedforward init, but the
        relaxation loop runs as a single `jax.lax.scan` instead of a Python
        `for` loop.
        """
        states_curr0 = self.init_activities(states_prev, control_input)
        opt_state0 = activity_optim.init(states_curr0)

        activity_step = self.make_activity_step(activity_optim, states_prev, observation, control_input)
        (states_curr, _), _ = jax.lax.scan(activity_step, (states_curr0, opt_state0), xs=None, length=n_steps)
        return states_curr


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
        def sequence_step(states_prev, xy_t):
            control_input_t, observation_t = xy_t
            states_curr = self.settle_scan(
                activity_optim, states_prev, observation_t, control_input_t, n_steps=n_infer_steps
            )
            energy_t = self.tpch_energy_fn(states_prev, states_curr, observation_t, control_input_t)
            # this step's settled states become next step's states_prev
            return states_curr, (states_curr, energy_t)

        return sequence_step


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
            act_fn=ACT_FN_REGISTRY[config.act_fn],
            input_size=config.input_size,
        )

    
    @classmethod
    def zero_activities(cls, config: TpchConfig) -> Activities:
        """
        Builds activities skeleton for TpchModel loading.
        """
        sizes = [config.control_layer_size, *config.hidden_sizes]
        return [jnp.zeros(s) for s in sizes]


