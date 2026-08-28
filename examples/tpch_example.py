from pc_nox.utils.visualisation import plot_visual_prediction, compile_videos_from_frames
from pc_nox.models.tpch import TpchModel
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import imageio.v2 as imageio
import numpy as np

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


for i, y in enumerate(frames):
    # Normalize pixel values if they are in range 0-255
    y = jnp.array(y, dtype=jnp.float32) / 255.0
    y = y.reshape(-1)  # Ensure flattened shape matches OUTPUT_DIM (128)

    # 1. Get the initial feedforward activities guess
    states_curr_init = model.init_activities(states_prev, control_input)

    # 2. Get the prior sensory prediction (y_hat) BEFORE inference
    _, y_hat_before = model.predict(states_prev, states_curr_init, control_input)

    energy_before = model.tpch_energy_fn(states_prev, model.init_activities(states_prev), y)
    states_curr = model.settle_scan(activity_optim, states_prev, y, n_steps=20)

    # Posterior sensory prediction (Reconstruction after settling)
    _, y_hat_after = model.predict(states_prev, states_curr, control_input)
    energy_after_inference = model.tpch_energy_fn(states_prev, states_curr, y)

    # --- C. PARAMETER WEIGHT UPDATE (Adam) ---
    grads = model.param_grad(states_prev, states_curr, y)
    updates, param_opt_state = param_optim.update(grads, param_opt_state, model)
    model = eqx.apply_updates(model, updates)


    print(f"{i}. VFE before inference: {energy_before}")
    print(f"{i}. VFE after inference: {energy_after_inference}")

    # plot predictions before and after settling
    plot_visual_prediction(y, prior_pred=y_hat_before, posterior_pred=y_hat_after, inference_steps_made=NUM_INFERENCE_STEPS, output_shape=(ENV_HEIGHT, ENV_WIDTH), frame_number=i, save_combined=True, save_separate=True, show_combined=False)

    # Pass settled states as previous states for step t + 1
    states_prev = states_curr