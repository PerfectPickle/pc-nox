"""Pedagogical demonstration of explicit, step-by-step Hierarchical Temporal Predictive Coding using the TpchModel class.

⚠️ DEMONSTRATION ONLY — NOT RECOMMENDED FOR PRODUCTION OR EXPERIMENTS ⚠️
-------------------------------------------------------------------------
This script provides an uncompiled, explicit breakdown of the underlying steps in 
a Hierarchical Temporal Predictive Coding training loop (feedforward initialization, prior prediction, 
activity settling, posterior prediction, and weight updates). 

Because each step is called individually from Python without enclosing JIT compilation, 
this approach incurs severe Python overhead and is >10x slower than the compiled alternative. 
Use `make_train_step` (JIT-compiled per-step) or `make_train_run` (fused block scan) 
for actual training runs.

Educational Value & Internal Mechanics
--------------------------------------
This script explicitly reveals the internal mechanics of one frame iteration:

1. Data Ingestion & Preprocessing
   - Loads and normalizes video frames (`example_env.mp4`) into JAX arrays.

2. Model & Optimizer Setup
   - Instantiates `TpchModel` and Optax optimizers for activities and weights.

3. Uncompiled Step-by-Step Training Breakdown
   - Step 1 (Feedforward Init): Generates initial activity guess (`init_activities`).
   - Step 2 (Prior Prediction): Predicts sensory output before inference (`predict`).
   - Step 3 (Prior Energy): Computes initial Variational Free Energy (`tpch_energy_fn`).
   - Step 4 (Activity Settling): Relaxes states to minimize energy (`settle_scan`).
   - Step 5 (Posterior Prediction): Predicts sensory output after inference (`predict`).
   - Step 6 (Posterior Energy): Computes settled Variational Free Energy (`tpch_energy_fn`).
   - Step 7 (Weight Gradients): Computes parameter gradients (`param_grad`).
   - Step 8 (Weight Update): Applies Optax updates to the model (`eqx.apply_updates`).

4. Introspection & Visualization
   - Logs VFE before and after inference.
   - Plots prior vs. posterior reconstructions per frame.
   - Compiles output prediction frames into a video.

Performance Summary
-------------------
- Best Used For: Step-by-step debugging, learning.
- Avoid For: Benchmarking, full dataset training, or scaling up models.
"""

from pc_nox.utils.visualisation import compile_videos_from_frames, VisualPredictionPlotter
from pc_nox.models.tpch import TpchModel
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import imageio.v2 as imageio
import numpy as np

###
### It is NOT recommended to use this as a template!
###
### The make_train_step (JIT) approach in tpch_save-viz_example.py is over 10x faster.
### This example is more for demonstration purposes.
### Custom behaviour with JIT performance, requires a custom make_train_step-like function.
###

# Matching example_env.mp4
ENV_WIDTH = 16 # pixels
ENV_HEIGHT = 8 # pixels

# Number of settling iterations
NUM_INFERENCE_STEPS = 50

# example_env.mp4 has 2000 frames
N_TRAIN_ITERS = 1000

x = None # no control input

ENV_WIDTH = 16
ENV_HEIGHT = 8
CONTROL_WIDTH = 8
HIDDEN_SHAPE = [8, 16, 32, 64, 128] # # width, from highest layer to lowest, inlcuding output / sensory layer
OBS_WIDTH = ENV_WIDTH * ENV_HEIGHT

raw_frames = imageio.mimread("example_env.mp4", memtest=False)
frames = np.stack(raw_frames)
if frames.ndim == 4 and frames.shape[-1] in (3, 4):  # Convert RGB(A) to grayscale
    frames = np.mean(frames[..., :3], axis=-1)
# Normalize pixel values if they are in range 0-255
frames = jnp.array(frames, dtype=jnp.float32) / 255.0

key = jr.PRNGKey(0)
model_key, data_key = jr.split(key)

control_input = None

model = TpchModel(
        control_layer_size=CONTROL_WIDTH,
        hidden_sizes=HIDDEN_SHAPE,
        obs_size=OBS_WIDTH,
        key=model_key,
    )

param_optim = optax.adam(learning_rate=1e-3)
param_opt_state = param_optim.init(eqx.filter(model, eqx.is_array))
activity_optim = optax.adam(learning_rate=0.01)


# one random "previous states" tuple and one time step of data
prev_key, x_key, y_key = jr.split(data_key, 3)
states_prev = [
    jr.normal(k, (size,))
    for k, size in zip(jr.split(prev_key, 1 + len(HIDDEN_SHAPE)), [CONTROL_WIDTH] + HIDDEN_SHAPE)
]

energies = []

for i, y in enumerate(frames):
    y = y.reshape(-1)  # Ensure flattened shape matches OUTPUT_DIM (128)

    # 1. Get the initial feedforward activities guess
    states_curr_init = model.init_activities(states_prev, control_input)

    # 2. Get the prior sensory prediction (y_hat) BEFORE inference
    _, y_hat_before = model.predict(states_prev, states_curr_init, control_input)

    energy_before = model.tpch_energy_fn(states_prev, model.init_activities(states_prev), y)
    states_curr = model.settle_scan(activity_optim, states_prev, y, n_steps=20)
    # or alternatively,
    # states_curr = model.settle(states_prev, y, control_input, n_steps=20, state_lr=0.1)

    # Posterior sensory prediction (Reconstruction after settling)
    _, y_hat_after = model.predict(states_prev, states_curr, control_input)
    energy_after_inference = model.tpch_energy_fn(states_prev, states_curr, y)

    # --- C. PARAMETER WEIGHT UPDATE (Adam) ---
    grads = model.param_grad(states_prev, states_curr, y)
    updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
    model = eqx.apply_updates(model, updates)


    print(f"{i}. VFE before inference: {energy_before}")
    print(f"{i}. VFE after inference: {energy_after_inference}")

    # Plot predictions before and after settling. Significant performance hit, better to use PredictionRecorder and plot frames predictions after training / inference.
    prediction_plotter.update(
        y=y, 
        prior_pred=y_hat_before, 
        posterior_pred=y_hat_after, 
        inference_steps_made=NUM_INFERENCE_STEPS, 
        frame_number=i, 
        show_combined=False,
        save_combined=True,
        save_separate=False,
        output_dir="visual_predictions",
        total_frames=N_TRAIN_ITERS, # for 0 padding in file names
    )

    # Pass settled states as previous states for step t + 1
    states_prev = states_curr

compile_videos_from_frames()