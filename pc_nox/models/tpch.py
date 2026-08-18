"""tpch.py

Equinox implementation of temporal Hierarchical Predictive Coding (tPC-H),
following eqs. (19)-(27) of the tPC-H paper.

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