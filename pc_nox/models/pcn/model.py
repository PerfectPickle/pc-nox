"""pcn/model.py

Static (non-temporal) hierarchical PCN. `layers` can be built two ways
(exactly one of `layer_sizes`/`layer_specs` must be passed to `__init__`):

  - `layer_sizes` (original, dense-only): a flat chain of widths,
    `(input_size, hidden_1, ..., hidden_k, output_size)` -> all `PcnLayer`.
  - `layer_specs` (conv/flatten/dense mix, e.g. for image input): see
    `config.py`'s module docstring for the spec format, and
    `_build_layers_from_specs` below for how it's turned into layer
    objects. `x` is then expected shaped (input_channels, *input_hw),
    NOT flattened.

Both paths produce the same thing either way: a flat `List` of layer
objects, each implementing `predict(parent_curr) -> own_prediction` and
`weights() -> List[Array]` (see `layers.py`). Everything below this point
-- `predict`, `init_activities`, `pcn_energy_fn`, `settle*`, `param_grad`
-- is unchanged by which path built `self.layers`; it only ever calls
that shared interface, never anything conv/dense-specific. That's the
main design point of this file: heterogeneity lives entirely in how
`self.layers` gets built and in the layer classes themselves, not here.

One nuance worth flagging: a `PcnFlattenLayer`'s output is still a free
interior latent like any other (it has an entry in `states_curr`, same as
a `PcnLayer`'s output does), even though its "prediction" is a
parameter-free reshape of its parent. Settling can in principle nudge it
away from that reshape (pulled by the energy of the layer above it), even
though nothing produced it via a weight. This isn't a correctness bug --
the energy term `0.5 * sum((target - prediction)**2)` is well-defined
either way -- just a small inefficiency (an extra free variable with a
fixed, cheap-to-compute optimum) worth knowing about if you're chasing
settling speed. Special-casing `predict`/`init_activities` to skip
optimisation over flatten outputs would remove it, at the cost of the
uniform-interface simplicity above; not done here.

--------------------------------------------------------------------------
How a Meta-PCN variant would extend this
--------------------------------------------------------------------------
Same relationship as `tpch_bi` to `tpch`: copy this package as a sibling
(`pcn_meta/`), not a subclass -- concrete variants that change energy/weights/settling wholesale get
little from inheriting a concrete `eqx.Module`, and real equinox/MRO
pitfalls). What would change, concretely:

  - `init_activities` (below) already computes exactly `c_l` (Algorithm 1
    lines 5-9) as its feedforward pass -- Meta-PCN's frozen predictions
    are these values, held fixed for the whole inference loop rather than
    recomputed each iteration.
  - `pcn_energy_fn` here is used for BOTH inference (`settle`/`settle_scan`)
    and learning (`param_grad`) -- one scalar, same as tPC-H. Meta-PCN
    needs two: a meta-PE objective J (with `jax.lax.stop_gradient` on the
    incoming delta_{l+1} term) for inference, and a plain sum-of-squared-
    settled-errors L(theta) for learning. Two internal methods instead of
    one `energy_fn`, wired the same way into `settle_scan`/`param_grad`.
  - Weight normalisation (Algorithm 1 lines 21-24) is a post-update
    transform, not a gradient penalty -- use `ModelBase.postprocess_params`
    (see `model_base.py`), not a new energy term.

--------------------------------------------------------------------------
What's shared with the temporal side, and what isn't
--------------------------------------------------------------------------
`penalties.py`'s L2/orthogonal/activity math is reused as-is (weight
selection is trivial here -- there's no rec/ff distinction, so `_all_weights()`
is the only scope). NOTE: `orthogonal_reg` likely assumes 2D weight
matrices -- a conv kernel from `PcnConvLayer.weights()` is 4D
(out_channels, in_channels, kH, kW), and hasn't been checked against that
function yet; see this file's TODO near `_weight_orthogonal_reg`.
`runners_static.py` (batch via `jax.vmap`, not `jax.lax.scan` across time
-- see that module's docstring) is this model's counterpart to
`runners_temporal.py`, and is NOT shared with it: there's no
`states_prev` for a batch of independent examples to carry, so there's
nothing for a `lax.scan`-based runner to do here that `vmap` doesn't do
more naturally. Nothing about `layer_specs`/conv support required any
change to `runners_static.py` -- it only calls the shared model
interface, and `jax.vmap` doesn't care that a leaf is a feature map
instead of a flat vector.
"""

from typing import Callable, ClassVar, List, Optional, Sequence, Tuple, Union
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from pc_nox.core import penalties
from pc_nox.core.activations import ACT_FN_REGISTRY
from pc_nox.core.model_base import Activities, ModelBase, Predictions
from .config import PcnConfig
from .layers import PcnConvLayer, PcnFlattenLayer, PcnLayer

# Activation names whose range is bounded -- used only for the loss='ce'
# readout-activation warning below. Adjust/extend to match your actual
# ACT_FN_REGISTRY; this is a best guess, not derived from that registry.
_BOUNDED_ACT_NAMES = {"tanh", "sigmoid", "hard_tanh"}


def _build_layers_from_specs(
    layer_specs: Sequence[Tuple],
    input_channels: int,
    input_hw: Tuple[int, int],
    act_fns: Sequence[Callable],
    keys: Sequence[PRNGKeyArray],
) -> List:
    """Turns a `layer_specs` tuple (see `config.py`'s module docstring for
    the format) into a flat list of layer objects. Each "dense" spec's
    input size is worked out with `jax.eval_shape` by tracing a
    zero-cost, weightless shape probe through the layers already built --
    this sidesteps reimplementing conv output-size arithmetic (stride/
    padding interactions, "SAME" semantics, etc.) here; `jax.eval_shape`
    asks JAX itself, so it's exact regardless of padding mode.
    """
    layers = []
    in_channels = input_channels
    probe_shape = (input_channels, *input_hw)
    key_iter = iter(keys)

    for spec, act_fn in zip(layer_specs, act_fns):
        kind = spec[0]
        if kind == "conv":
            _, out_channels, kernel_size, stride, padding = spec
            layer = PcnConvLayer(
                in_channels, out_channels, kernel_size, stride, padding, act_fn, key=next(key_iter)
            )
            probe_shape = jax.eval_shape(layer.predict, jax.ShapeDtypeStruct(probe_shape, jnp.float32)).shape
            in_channels = out_channels
            layers.append(layer)
        elif kind == "flatten":
            layer = PcnFlattenLayer()
            probe_shape = jax.eval_shape(layer.predict, jax.ShapeDtypeStruct(probe_shape, jnp.float32)).shape
            layers.append(layer)
        elif kind == "dense":
            _, out_size = spec
            if len(probe_shape) != 1:
                raise ValueError(
                    f"'dense' spec {spec!r} follows a non-flat shape {probe_shape!r}; "
                    "insert a ('flatten',) spec before the first 'dense' spec"
                )
            in_size = probe_shape[0]
            layer = PcnLayer(in_size, out_size, act_fn, key=next(key_iter))
            probe_shape = (out_size,)
            layers.append(layer)
        else:
            raise ValueError(f"unknown layer_spec kind {kind!r} in {spec!r} (expected 'conv', 'flatten', or 'dense')")

    return layers


def _zero_activities_from_specs(
    layer_specs: Sequence[Tuple], input_channels: int, input_hw: Tuple[int, int]
) -> Activities:
    """Shape-only counterpart to `_build_layers_from_specs`, for
    `PcnModel.zero_activities` -- same `jax.eval_shape` probe, but no
    weights/keys needed since we only want the shape each interior layer's
    output would have, not real layer objects.
    """
    probe_shape = (input_channels, *input_hw)
    zeros = []
    for spec in layer_specs[:-1]:  # last spec is the readout, not a free latent
        kind = spec[0]
        if kind == "conv":
            _, out_channels, kernel_size, stride, padding = spec
            dummy = PcnConvLayer(probe_shape[0], out_channels, kernel_size, stride, padding, jnp.tanh, key=jr.PRNGKey(0))
            probe_shape = jax.eval_shape(dummy.predict, jax.ShapeDtypeStruct(probe_shape, jnp.float32)).shape
        elif kind == "flatten":
            probe_shape = (int(jnp.prod(jnp.array(probe_shape))),)
        elif kind == "dense":
            _, out_size = spec
            probe_shape = (out_size,)
        else:
            raise ValueError(f"unknown layer_spec kind {kind!r} in {spec!r}")
        zeros.append(jnp.zeros(probe_shape))
    return zeros


class PcnModel(eqx.Module, ModelBase):
    """A static hierarchical PCN: z_1 (=x, clamped) -> z_2 -> ... ->
    z_{L-1} (free latents) -> z_L (=y, clamped).

    Attributes:
        layers: L-1 layer objects (`PcnLayer`/`PcnConvLayer`/
            `PcnFlattenLayer`), layers[i] predicting z_{i+2} from z_{i+1}
            (layers[0] predicts z_2 from z_1=x; layers[-1] predicts z_L
            from z_{L-1}, compared against y).

    Args:
        layer_sizes: dense-only path, (input_size, hidden_1, ...,
            output_size). Exactly one of layer_sizes/layer_specs required.
        layer_specs: conv/flatten/dense path -- see `config.py`'s module
            docstring for the format. Requires input_channels/input_hw.
        input_channels, input_hw: required with layer_specs; x is then
            expected shaped (input_channels, *input_hw), not flattened.
        act_fn: one name (broadcast to every layer), or one name per
            layer (see `config.py`'s module docstring re: 'flatten'
            entries). A warning is raised if loss='ce' and the readout's
            activation is bounded (e.g. 'tanh') -- see _BOUNDED_ACT_NAMES.
        loss: Output-layer loss, `"mse"` (default) or `"ce"`.
        weight_decay, orthogonal_penalty, activity_decay, activity_reg_type:
            same meaning as the like-named `TpchConfig` fields -- see
            `penalties.py`. No `_scope` variants: there's no rec/ff
            distinction in a static PCN, every penalty applies to every
            weight/state uniformly.
    """
    model_type: ClassVar[str] = "pcn"
    config_cls: ClassVar[type] = PcnConfig
    config: PcnConfig = eqx.field(static=True)

    layers: List

    def __init__(
        self,
        layer_sizes: Optional[Sequence[int]] = None,
        *,
        key: PRNGKeyArray,
        layer_specs: Optional[Sequence[Tuple]] = None,
        input_channels: Optional[int] = None,
        input_hw: Optional[Tuple[int, int]] = None,
        act_fn: Union[str, Sequence[str]] = "tanh",
        loss: str = "mse",
        weight_decay: float = 0.0,
        orthogonal_penalty: float = 0.0,
        activity_decay: float = 0.0,
        activity_reg_type: str = "l1",
    ):
        if (layer_sizes is None) == (layer_specs is None):
            raise ValueError("pass exactly one of layer_sizes= (dense-only) or layer_specs= (conv/flatten/dense)")
        if loss not in ("mse", "ce"):
            raise ValueError(f"loss must be 'mse' or 'ce', got {loss!r}")
        if activity_reg_type not in ("l1", "l2"):
            raise ValueError(f"activity_reg_type must be 'l1' or 'l2', got {activity_reg_type!r}")

        if layer_specs is not None:
            if input_channels is None or input_hw is None:
                raise ValueError("layer_specs requires input_channels= and input_hw=")
            layer_specs = tuple(tuple(spec) for spec in layer_specs)
            n_layers = len(layer_specs)
        else:
            if len(layer_sizes) < 2:
                raise ValueError(f"layer_sizes needs at least 2 entries (input, output), got {layer_sizes!r}")
            layer_sizes = tuple(layer_sizes)
            n_layers = len(layer_sizes) - 1

        names = (act_fn,) * n_layers if isinstance(act_fn, str) else tuple(act_fn)
        if len(names) != n_layers:
            raise ValueError(f"act_fn needs {n_layers} entries (one per layer), got {len(names)}")

        try:
            act_fns = [ACT_FN_REGISTRY[n] for n in names]
        except KeyError as e:
            raise KeyError(
                f"act_fn={e.args[0]!r} not in ACT_FN_REGISTRY. If this is a custom "
                f"activation, register it before loading: ACT_FN_REGISTRY[{e.args[0]!r}] = ..."
            ) from None

        if loss == "ce" and names[-1] in _BOUNDED_ACT_NAMES:
            warnings.warn(
                f"loss='ce' with a bounded readout activation ({names[-1]!r}) caps the "
                "logits and can hurt/collapse training -- consider an unbounded readout "
                "activation (e.g. 'identity') for the last layer.",
                stacklevel=2,
            )

        self.config = PcnConfig(
            layer_sizes=layer_sizes,
            layer_specs=layer_specs,
            input_channels=input_channels,
            input_hw=tuple(input_hw) if input_hw is not None else None,
            act_fn=act_fn if isinstance(act_fn, str) else names,
            loss=loss,
            weight_decay=weight_decay,
            orthogonal_penalty=orthogonal_penalty,
            activity_decay=activity_decay,
            activity_reg_type=activity_reg_type,
        )

        keys = jr.split(key, n_layers)
        if layer_specs is not None:
            self.layers = _build_layers_from_specs(layer_specs, input_channels, tuple(input_hw), act_fns, keys)
        else:
            self.layers = [
                PcnLayer(layer_sizes[i], layer_sizes[i + 1], act_fns[i], key=k) for i, k in enumerate(keys)
            ]

    # =========================================================================
    # Predict / init -- states_curr here means the INTERIOR latents only,
    # z_2..z_{L-1}; x (=z_1) and y (=z_L) are passed separately, exactly
    # like tPC-H passes control_input/observation alongside states_prev.
    # =========================================================================

    def predict(self, states_curr: Activities, x: Array) -> Tuple[Predictions, Array]:
        """Run every layer's `predict` once. `states_curr` is the current
        guess for z_2..z_{L-1} (len = len(self.layers) - 1); `x` is z_1
        (a flat vector for the layer_sizes path, or a (C, H, W) feature
        map for the layer_specs path).

        Returns (predictions, y_hat): predictions[i] is what layer i
        expected z_{i+2} to be, aligned with `states_curr` (so
        predictions[:-1] compares to states_curr, mirroring tPC-H's
        `predict`); y_hat = predictions[-1] is the final layer's
        prediction of y.
        """
        chain_in = [x] + list(states_curr)  # z_1..z_{L-1}
        predictions = [layer.predict(z) for layer, z in zip(self.layers, chain_in)]  # -> z_2..z_L
        y_hat = predictions[-1]
        return predictions, y_hat

    def init_activities(self, x: Array) -> Activities:
        """Feedforward pass: z_2..z_{L-1} computed by sweeping the input
        through every layer except the last (the last layer's prediction
        is y_hat, not a free latent). Exactly Algorithm 1 lines 4-8's `c_l`
        for l=2..L-1 -- see this module's docstring for how a Meta-PCN
        variant would freeze exactly these values.
        """
        states_curr = []
        z = x
        for layer in self.layers[:-1]:
            z = layer.predict(z)
            states_curr.append(z)
        return states_curr

    # =========================================================================
    # Weight enumeration / regularisation -- no rec/ff distinction (no
    # recurrent weights at all), so `penalties.py`'s math is applied to
    # every weight uniformly. `PcnFlattenLayer.weights()` contributes `[]`,
    # so it's automatically excluded here without any special-casing.
    # =========================================================================

    def _all_weights(self) -> List[Array]:
        return [w for layer in self.layers for w in layer.weights()]

    def _weight_l2_reg(self) -> Array:
        return penalties.l2_reg(self._all_weights(), self.config.weight_decay)

    def _weight_orthogonal_reg(self) -> Array:
        # TODO(conv): PcnConvLayer.weights() returns a 4D kernel
        # (out_channels, in_channels, kH, kW). Confirm penalties.orthogonal_reg
        # handles that shape sensibly (e.g. by reshaping to 2D internally) --
        # not yet checked. If not, this will need a conv-aware variant, or
        # conv kernels excluded from orthogonal_penalty for now.
        return penalties.orthogonal_reg(self._all_weights(), self.config.orthogonal_penalty)

    def _activity_reg(self, states_curr: Activities) -> Array:
        # TODO(conv): confirm penalties.activity_reg's l1/l2 sum works
        # element-wise regardless of an activity's rank (a (C,H,W) feature
        # map vs a flat vector) -- should be fine if it's implemented as
        # e.g. jnp.sum(jnp.abs(s)) per activity, but not yet checked here.
        return penalties.activity_reg(states_curr, self.config.activity_decay, self.config.activity_reg_type)

    # =========================================================================
    # Free energy
    # =========================================================================

    def pcn_energy_fn(
        self,
        states_curr: Activities,
        x: Array,
        y: Optional[Array],
        weight_reg_total: Optional[Array] = None,
        return_layerwise: bool = False,
    ) -> Array:
        """If `y` is None the output is left free (unclamped): the output
        error term is dropped, so settling never sees the label. Use this
        for test-time inference.

        Sum over every interior layer's squared prediction error, plus
        the output term (mse or ce against y), plus configured
        regularisation. Same shape as `tpch_energy_fn`, minus the temporal
        terms.
        """
        predictions, y_hat = self.predict(states_curr, x)
        targets = list(states_curr) + [y]  # z_2..z_{L-1}, y  (aligned with `predictions`)

        layer_energies = []
        for target, pred in zip(targets[:-1], predictions[:-1]):
            layer_energies.append(0.5 * jnp.sum((target - pred) ** 2))

        if y is None:
            # Free (unclamped) output: no target, so no output error term.
            obs_energy = jnp.zeros(())
        elif self.config.loss == "mse":
            obs_energy = 0.5 * jnp.sum((y - y_hat) ** 2)
        else:  # "ce"
            obs_energy = -jnp.sum(y * jax.nn.log_softmax(y_hat))
        layer_energies.append(obs_energy)

        if return_layerwise:
            return jnp.stack(layer_energies)  # regularisation NOT broken out per-layer here (see TpchModel for that pattern if needed)

        if weight_reg_total is None:
            weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        return sum(layer_energies) + weight_reg_total + self._activity_reg(states_curr)

    def energy_fn(self, *args, **kwargs):
        """Alias for `pcn_energy_fn` -- see `TpchModel.energy_fn`'s
        docstring for why this name exists (shared-runner convention)."""
        return self.pcn_energy_fn(*args, **kwargs)

    # =========================================================================
    # Inference
    # =========================================================================

    def neg_activity_grad(self, states_curr, x, y, weight_reg_total=None) -> Activities:
        energy_of_states = lambda s: self.pcn_energy_fn(s, x, y, weight_reg_total=weight_reg_total)
        return jax.tree_util.tree_map(jnp.negative, jax.grad(energy_of_states)(states_curr))

    def infer_step(self, states_curr, x, y, state_lr: float = 0.1, weight_reg_total=None) -> Activities:
        grad_step = self.neg_activity_grad(states_curr, x, y, weight_reg_total=weight_reg_total)
        return jax.tree_util.tree_map(lambda s, g: s + state_lr * g, states_curr, grad_step)

    def settle(self, x: Array, y: Array, n_steps: int = 20, state_lr: float = 0.1) -> Activities:
        weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        states_curr = self.init_activities(x)
        for _ in range(n_steps):
            states_curr = self.infer_step(states_curr, x, y, state_lr, weight_reg_total=weight_reg_total)
        return states_curr

    def make_activity_step(self, activity_optim: optax.GradientTransformation, x: Array, y: Array):
        weight_reg_total = self._weight_l2_reg() + self._weight_orthogonal_reg()
        energy_fn = lambda s: self.pcn_energy_fn(s, x, y, weight_reg_total=weight_reg_total)

        def activity_step(carry, _):
            states_curr, opt_state = carry
            grads = jax.grad(energy_fn)(states_curr)
            updates, opt_state = activity_optim.update(grads, opt_state, states_curr)
            states_curr = optax.apply_updates(states_curr, updates)
            return (states_curr, opt_state), states_curr

        return activity_step

    def settle_scan(
        self,
        activity_optim: optax.GradientTransformation,
        x: Array,
        y: Array,
        n_steps: int = 20,
        return_layerwise: bool = False,
    ) -> Activities:
        states_curr0 = self.init_activities(x)
        opt_state0 = activity_optim.init(states_curr0)
        activity_step = self.make_activity_step(activity_optim, x, y)
        (states_curr, _), states_hist = jax.lax.scan(activity_step, (states_curr0, opt_state0), xs=None, length=n_steps)

        if not return_layerwise:
            return states_curr
        energy_trace_fn = lambda s: self.pcn_energy_fn(s, x, y, return_layerwise=True)
        energy_trace = jax.vmap(energy_trace_fn)(states_hist)
        return states_curr, energy_trace

    # =========================================================================
    # Learning
    # =========================================================================

    def param_grad(self, states_curr: Activities, x: Array, y: Array) -> PyTree:
        energy_of_weights = lambda m: m.pcn_energy_fn(states_curr, x, y)
        return eqx.filter_grad(energy_of_weights)(self)

    def update_params(self, grads: PyTree, optim: optax.GradientTransformation, opt_state: optax.OptState):
        updates, opt_state = optim.update(grads, opt_state, self)
        updated_model = eqx.apply_updates(self, updates)
        return updated_model, opt_state

    # =========================================================================
    # Saving and loading
    # =========================================================================

    @classmethod
    def from_config(cls, config, *, key) -> "PcnModel":
        return cls(
            layer_sizes=config.layer_sizes,
            key=key,
            layer_specs=config.layer_specs,
            input_channels=config.input_channels,
            input_hw=config.input_hw,
            act_fn=config.act_fn,
            loss=config.loss,
            weight_decay=config.weight_decay,
            orthogonal_penalty=config.orthogonal_penalty,
            activity_decay=config.activity_decay,
            activity_reg_type=config.activity_reg_type,
        )

    @classmethod
    def layer_labels(cls, config: PcnConfig) -> List[str]:
        n_interior = (len(config.layer_specs) if config.layer_specs is not None else len(config.layer_sizes) - 1) - 1
        return [f"Hidden {i + 1}" for i in range(n_interior)] + ["Output"]

    @classmethod
    def zero_activities(cls, config: PcnConfig) -> Activities:
        if config.layer_specs is not None:
            return _zero_activities_from_specs(config.layer_specs, config.input_channels, config.input_hw)
        return [jnp.zeros(s) for s in config.layer_sizes[1:-1]]
