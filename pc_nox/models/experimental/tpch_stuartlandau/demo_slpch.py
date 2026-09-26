"""demo_slpch.py

Standalone smoke-test / demo for the Stuart-Landau tPC-H (SL-tPC-H)
variant in `slpch/`.

`slpch/model.py` imports `from .. import inference` and `from
..model_base import ...`, exactly like `tpch/model.py` already does --
`slpch` needs to sit next to `inference.py`/`model_base.py` as a
subpackage of your project's existing root package, not next to them as
a bare top-level directory (a plain `python demo_slpch.py` with `slpch`
imported by bare name will hit "attempted relative import beyond
top-level package", for the same reason it would for `tpch`). Run this
file exactly the way you already run your `tpch` experiments/runners --
whatever that invocation is (e.g. `python -m your_package.demo_slpch`
from one level above your project root) already gives `slpch` the same
package context `tpch` has.

Shows three things, each a real correctness check, not just a "does it
run":

  1. Relaxation actually decreases the free energy (settle_scan).
  2. A single unit's trajectory in the complex plane visibly spirals from
     its feedforward init onto a bounded orbit under the combined
     Stuart-Landau + predictive-coding dynamics (settle_diffrax).
  3. A toy training loop (settle -> param_grad -> update_params) drives
     the free energy down over weight-update steps, exactly like
     TpchModel's training loop would.

Saves `slpch_demo.png` (energy trace + phase portrait) next to this file.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from slpch import SLPCHModel


def main():
    key = jr.PRNGKey(0)
    mkey, dkey = jr.split(key)

    model = SLPCHModel(
        control_layer_size=4,
        hidden_sizes=(6, 5),
        obs_size=3,
        input_size=2,
        key=mkey,
        gamma_init=1.0,
        omega_init_scale=3.0,
    )
    states_prev = SLPCHModel.zero_activities(model.config)
    k1, k2 = jr.split(dkey)
    control_input = jr.normal(k1, (2,))
    observation = jr.normal(k2, (3,))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ------------------------------------------------------------------
    # 1. Relaxation decreases the free energy (scan-fused settling)
    # ------------------------------------------------------------------
    _, energy_trace = model.settle_scan(
        optax.sgd(0.05), states_prev, observation, control_input, n_steps=300, return_layerwise=True,
    )
    total_energy = jnp.sum(energy_trace, axis=-1)
    axes[0].plot(total_energy)
    axes[0].set_title("settle_scan: free energy vs. relaxation step")
    axes[0].set_xlabel("relaxation step")
    axes[0].set_ylabel("F_t (total)")
    print(f"[1] settle_scan energy: {float(total_energy[0]):.4f} -> {float(total_energy[-1]):.4f}")

    # ------------------------------------------------------------------
    # 2. Phase portrait: one control-layer unit's trajectory in the
    #    complex plane, integrated with settle_diffrax over several
    #    intrinsic periods, should visibly settle onto a bounded orbit.
    # ------------------------------------------------------------------
    unit = 0
    period = 2 * jnp.pi / jnp.abs(model.control_layer.omega[unit])
    max_t1 = float(8 * period)
    # settle_diffrax itself only returns the final state (+ optional
    # energy trace) -- for the demo we want the FULL trajectory of one
    # unit, so we integrate the same vector field ourselves with a plain
    # Euler loop, purely for plotting (this is not how you'd normally
    # call the model -- see settle_diffrax/settle_scan for that).
    vf = model.make_vector_field(states_prev, observation, control_input)
    states_curr0 = model.init_activities(states_prev, control_input, observation)
    dt = max_t1 / 2000
    traj = [states_curr0[0][unit]]
    s = states_curr0
    for i in range(2000):
        ds = vf(i * dt, s, None)
        s = [si + dt * dsi for si, dsi in zip(s, ds)]
        traj.append(s[0][unit])
    traj = jnp.array(traj)
    gamma_unit = float(jax.nn.softplus(model.control_layer.gamma_raw[unit]))
    theta = jnp.linspace(0, 2 * jnp.pi, 200)
    axes[1].plot(jnp.sqrt(gamma_unit) * jnp.cos(theta), jnp.sqrt(gamma_unit) * jnp.sin(theta),
                 "k--", alpha=0.4, label=f"free-running limit cycle (r=sqrt(gamma)={gamma_unit**0.5:.2f})")
    axes[1].plot(jnp.real(traj), jnp.imag(traj), lw=0.8)
    axes[1].scatter([jnp.real(traj[0])], [jnp.imag(traj[0])], color="green", zorder=5, label="start")
    axes[1].scatter([jnp.real(traj[-1])], [jnp.imag(traj[-1])], color="red", zorder=5, label="end")
    axes[1].set_title(f"control unit {unit}: trajectory in the complex plane")
    axes[1].set_xlabel("Re(z)")
    axes[1].set_ylabel("Im(z)")
    axes[1].legend(fontsize=7)
    axes[1].set_aspect("equal")

    # ------------------------------------------------------------------
    # 3. Toy training loop: settle -> param_grad -> update_params should
    #    drive the free energy down over weight updates (separate from
    #    (1), which only shows within-step relaxation).
    # ------------------------------------------------------------------
    optim = optax.sgd(1e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    train_energies = []
    m = model
    for _ in range(80):
        states_curr = m.settle_scan(optax.sgd(0.05), states_prev, observation, control_input, n_steps=20)
        e = m.slpch_energy_fn(states_prev, states_curr, observation, control_input)
        train_energies.append(float(e))
        grads = m.param_grad(states_prev, states_curr, observation, control_input)
        m, opt_state = m.update_params(grads, optim, opt_state)
    axes[2].plot(train_energies)
    axes[2].set_title("toy training loop: free energy vs. weight update")
    axes[2].set_xlabel("weight-update step")
    axes[2].set_ylabel("F_t at settled state")
    print(f"[3] training-loop energy: {train_energies[0]:.4f} -> {train_energies[-1]:.4f}")

    fig.tight_layout()
    fig.savefig("slpch_demo.png", dpi=130)
    print("Saved slpch_demo.png")


if __name__ == "__main__":
    main()
