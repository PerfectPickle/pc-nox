"""inference.py

Model-agnostic activity-settling machinery, shared by every temporal PC
model variant (tPC-H, bidirectional tPC-H, and whatever comes after).
Nothing in this file knows about control layers, hidden layers,
observation layers, or any variant-specific config or weights: every
function here operates purely on plain callables --

    vector_field(t, states, args) -> d(states)/dt
    energy_fn(states) -> scalar

-- and an Activities pytree of initial states. Each model builds these
closures itself (closing over its own states_prev, observation,
control_input, frozen weight-regularisation total, etc.) and hands them to
the functions below. That's what lets every temporal variant share the
diffrax integration / early-termination / save-grid logic without sharing
anything about what its energy function actually computes.

This is a straight extraction of tPC-H's original `make_steady_state_event`
and `settle_diffrax` methods -- the ~350 lines that never referenced
`self.control_layer`/`self.hidden_layers`/`self.config` to begin with, just
`self.tpch_energy_fn` via a closure. Behaviour is unchanged; see
`tpch/model.py`'s `make_steady_state_event`/`settle_diffrax` for the thin
per-model adapters that build the closures and call through to here.

`jax.lax.scan`-based settling (the optax-activity-optimiser equivalent of
this file) is deliberately NOT here: it's ~50 lines per model
(`make_activity_step` + `settle_scan`), cheap enough to duplicate, and
different models may reasonably want different scan bodies (e.g. a model
that fixes some activities during inference, like Meta-PCN). Revisit that
choice only if it turns out to need shared bug fixes in practice.
"""

from typing import Callable, Optional, Tuple, Union

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

VectorField = Callable[[float, PyTree, None], PyTree]
EnergyFn = Callable[[PyTree], Array]
LayerwiseEnergyFn = Callable[[PyTree], Array]


def make_steady_state_event(
    vector_field: VectorField,
    tol: Optional[float] = 1e-3,
    criterion: str = "rms",
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    energy_fn: Optional[EnergyFn] = None,
) -> diffrax.Event:
    """Builds an early-termination condition for adaptive-compute relaxation.

    Returns a `diffrax.Event` that fires once the activity dynamics have
    (approximately) reached steady state, by one of three criteria (see
    `criterion` below). This is what lets `settle_diffrax`'s Mode 1 (see
    its docstring) stop integrating as soon as a given input has actually
    converged, instead of always running to `max_t1`.

    Note (performance): every criterion recomputes the energy gradient
    independently of the solver's own internal stage evaluations, so
    enabling the event means doing extra `grad`-equivalent `vector_field`
    calls beyond what plain integration alone would need (roughly one
    extra per accepted step, on top of however many the solver's stages
    already use; `"energy_rate"` additionally calls `energy_fn` once more
    per check).

    Args:
        vector_field: `ds/dt = -dE/ds`, as built by the calling model
            (e.g. `TpchModel.make_vector_field`). Same trajectory this
            event is guarding, so it must already close over the
            trajectory's `states_prev`/`observation`/`control_input`.
        criterion: Which convergence check to use:
            * `"rms"` (default) -- RMS of the raw vector field `ds/dt`,
              pooled unweighted across every layer's elements, compared
              against `tol`.
            * `"relative_rms"` -- per-element `ds/dt` normalised by
              `atol + rtol * |s|` (diffrax's own `PIDController`
              convention), RMS-pooled, compared against `1.0`. Requires
              `rtol`/`atol` instead of `tol`.
            * `"energy_rate"` -- relative rate of energy decrease,
              `||ds/dt||^2 / (|E| + eps)`, compared against `tol`.
              Requires `energy_fn`.
        tol: Threshold for `"rms"` and `"energy_rate"`. Unused for
            `"relative_rms"`. Defaults to 1e-3.
        rtol, atol: Relative/absolute scale for `"relative_rms"`, in the
            `atol + rtol * |s|` convention. Required (both) for that
            criterion, otherwise unused.
        energy_fn: Required (and only used) for `"energy_rate"`: the
            model's total scalar energy as a function of `states_curr`,
            evaluated at the fixed `states_prev`/`observation`/
            `control_input` of this trajectory.

    Returns:
        A `diffrax.Event` wrapping a `cond_fn(t, states_curr, args,
        **kwargs) -> bool`, suitable for passing as `diffeqsolve`'s
        `event=` argument.
    """
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
        if energy_fn is None:
            raise ValueError("criterion='energy_rate' requires energy_fn")

        def steady_state_cond(t, states_curr, args, **_):
            dstates = vector_field(t, states_curr, args)
            leaves = jax.tree_util.tree_leaves(dstates)
            sq_norm = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)  # = |dE/dt|
            energy = energy_fn(states_curr)
            return sq_norm / (jnp.abs(energy) + 1e-8) < tol

    else:
        raise ValueError(
            f"unknown criterion {criterion!r}; expected 'rms', 'relative_rms', or 'energy_rate'"
        )

    return diffrax.Event(steady_state_cond)


def settle_diffrax(
    vector_field: VectorField,
    init_states: PyTree,
    *,
    energy_fn: Optional[EnergyFn] = None,
    layerwise_energy_fn: Optional[LayerwiseEnergyFn] = None,
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
) -> Union[PyTree, Tuple[PyTree, Array, Array]]:
    """Continuous-time relaxation via a single `diffrax.diffeqsolve` call.

    Model-agnostic core of what used to be `TpchModel.settle_diffrax`.
    Integrates `ds/dt = vector_field(t, s, args)` from `init_states`,
    optionally stopping early once steady state is reached.

    Supports two modes, both driven by `steady_state_tol`:

    * **Mode 1 -- event-based adaptive compute (default).**
      `steady_state_tol` is a float. A `make_steady_state_event(...,
      tol=steady_state_tol, criterion=steady_state_criterion, ...)` is
      built and passed to `diffeqsolve` as `event=...`, so integration
      stops as soon as the trajectory has actually converged, rather than
      always running to `max_t1`.
    * **Mode 2 -- fixed-horizon integration.** `steady_state_tol=None`.
      No event is used; `diffeqsolve` always integrates the full
      `[0, max_t1]` window.

    Args:
        vector_field: `ds/dt = -dE/ds`, built by the calling model (e.g.
            `TpchModel.make_vector_field`), already closing over
            `states_prev`/`observation`/`control_input` and any frozen
            weight regularisation for this trajectory.
        init_states: The trajectory's `y0`, e.g. the calling model's
            feedforward `init_activities(...)` guess.
        energy_fn: Total scalar energy as a function of `states_curr`,
            same closure convention as `vector_field`. Required only for
            `steady_state_criterion="energy_rate"`.
        layerwise_energy_fn: Per-layer energy breakdown as a function of
            `states_curr`, returning an Array (one entry per layer). Used
            only when `return_layerwise=True`; the calling model decides
            what "per layer" means and what order the entries come in.
        max_t1, dt0, n_save, solver, stepsize_controller, steady_state_tol,
        steady_state_criterion, steady_state_rtol, steady_state_atol,
        return_layerwise: Same meaning as on the original
            `TpchModel.settle_diffrax` -- see that method's docstring
            (preserved verbatim there) for the full description of each.

    Returns:
        If `return_layerwise` is False: `states_curr`, the settled
        Activities.

        If `return_layerwise` is True: a 3-tuple `(states_curr,
        energy_trace, ts)` -- see the original `settle_diffrax`
        docstring for the inf-padding caveat on `ts`/`energy_trace`
        after an early Mode-1 stop.
    """
    if solver is None:
        solver = diffrax.Heun()
    if stepsize_controller is None:
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)

    event = None
    if steady_state_tol is not None:  # Mode 1: event-based adaptive compute
        event = make_steady_state_event(
            vector_field,
            tol=steady_state_tol, criterion=steady_state_criterion,
            rtol=steady_state_rtol, atol=steady_state_atol,
            energy_fn=energy_fn,
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
        y0=init_states,
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

    if layerwise_energy_fn is None:
        raise ValueError("return_layerwise=True requires layerwise_energy_fn")

    # get energy trace from already processed states history -- entries
    # past an early steady-state stop will be inf/nan; `sol.ts` is
    # returned alongside so callers can mask them, rather than having to
    # reconstruct which points are valid themselves.
    energy_trace = jax.vmap(layerwise_energy_fn)(states_hist)  # (n_save + 2, num_layers + 1)
    return states_curr, energy_trace, sol.ts
