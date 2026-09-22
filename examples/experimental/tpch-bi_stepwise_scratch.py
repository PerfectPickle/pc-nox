"""Frame-by-frame training of a Temporal Predictive Coding hierarchy using `make_train_step`.
 
Demonstrates an interactive, per-step execution workflow for training a 
temporal predictive coding model (`TpchModel`) on sequential video data. 
Unlike fused block-scan implementations, this script executes a single JIT-compiled 
training step within an outer Python loop. This approach prioritizes fine-grained 
introspection, real-time logging, interactive visualization, and frequent 
checkpointing over maximum XLA execution speed.
 
Key Workflow Phases
-------------------
1. Data Ingestion & Preprocessing
   - Loads the complete input video (`example_env.mp4`) into memory (before JAX is
     imported, to avoid os.fork() issues).
   - Converts RGB frames to grayscale and normalizes pixel values to [0.0, 1.0].
   - Flattens each frame's spatial dimensions into a vector matching the observation
     layer of the predictive coding hierarchy.
 
2. Model & Optimizer Initialization
   - Instantiates a fresh `TpchModel` hierarchy matched to environmental dimensions.
   - Configures separate Optax Adam optimizers for structural weight parameters 
     (`param_optim`) and latent activity relaxation (`activity_optim`).
   - Draws a random set of initial "previous" activities for the first time step.
   - Constructs the JIT-compiled step function via `make_train_step`.
 
3. Per-Step Iterative Training Loop
   - Loops over frame sequences in Python, executing `train_step` on each iteration.
   - Settles internal activities over `NUM_INFERENCE_STEPS` relaxation steps.
   - Updates model parameters based on settled state errors.
   - Carries settled states (`states_curr`) forward to initialize activities for 
     the subsequent time step (`states_prev`).
 
4. In-Loop Introspection & Artifact Generation
   - Logs Variational Free Energy (VFE) before and after activity settling at each
     checkpoint interval and on the final frame.
   - Optionally records per-layer energy traces per relaxation step.
   - Records the target observation together with the prior and posterior
     predictions for every frame via `PredictionRecorder`.
   - At each checkpoint interval (and on the final frame), saves the model,
     optimizer state, settled activities and metadata as a checkpoint, and flushes
     the recorded predictions to disk.
 
5. Post-Training Diagnostics
   - Plots layerwise energy traces across the entire training trajectory.
   - Replays the raw prediction recordings to reconstruct image-space prediction frames.
   - Compiles the reconstructed prediction frames into output video files.
 
Trade-off Summary
-----------------
- Pros: Direct access to intermediate states per frame; effortless integration with 
  Python-side visualizers, loggers, and conditional stopping logic.
- Cons: Incurs Python loop overhead between JIT step dispatches compared to 
  fused `jax.lax.scan` execution (`make_train_run`).
"""

from pathlib import Path
import imageio.v2 as imageio
video_path = Path(__file__).resolve().parent.parent / "example_env.mp4"
raw_frames = imageio.mimread(str(video_path), memtest=False) # read this before importing JAX, to avoid os.fork() issues

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*os.fork.*") # to ignore compile_videos() warning

from pc_nox.utils.visualisation import compile_videos_from_frames, plot_energies, PredictionRecorder, replay_recordings
from pc_nox.models.experimental.tpch_bi import TpchModel, make_train_step
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time

# Number of settling iterations
NUM_INFERENCE_STEPS = 20

# example_env.mp4 has 2000 frames
N_TRAIN_ITERS = 2000

# training step interval to save checkpoint at
CHECKPOINT_INTERVAL = 1000

# for plotting
RECORD_ENERGIES = True

# where the raw jax arrays get stored during inference/training
PREDICTIONS_RECORDING_DIR = "visual_predictions_raw"
# where the reconstructed visual predictions get saved to
PREDICTIONS_DIR = "visual_predictions"

CHECKPOINT_ROOT = "checkpoints/scan"

# Matching example_env.mp4
ENV_WIDTH = 16 # pixels
ENV_HEIGHT = 8 # pixels
ENV_COLOUR_CHANNELS = 1
CONTROL_WIDTH = 8
HIDDEN_SHAPE = [8, 16, 32, 64, 128] # # width, from highest layer to lowest, including output / sensory layer
OBS_WIDTH = ENV_WIDTH * ENV_HEIGHT * ENV_COLOUR_CHANNELS

control_input = None
# Stored on model.config if model is initialised with them
weight_decay = 0.00001 
weight_decay_scope: str = "all"
orthogonal_penalty: float = 0.001  # Coefficient for orthogonality penalty ||I - W^T W||^2 on weights (0 = disabled)
orthogonal_scope: str = "rec"
activity_decay = 0.00001
activity_reg_type: str = "l1"

# Not stored on model.config, so must be added to metadata to be saved
PARAM_OPTIM_NAME = "adam"
PARAM_LR = 0.001
ACTIVITY_OPTIM_NAME = "adam"
ACTIVITY_LR = 0.01

metadata = {
    "param_optim": {"name": PARAM_OPTIM_NAME, "learning_rate": PARAM_LR},
    "activity_optim": {"name": ACTIVITY_OPTIM_NAME, "learning_rate": ACTIVITY_LR},
    "num_inference_steps": NUM_INFERENCE_STEPS,
    "ENV_WIDTH": ENV_WIDTH,
    "ENV_HEIGHT": ENV_HEIGHT
}

frames = np.stack(raw_frames)
if frames.ndim == 4 and frames.shape[-1] in (3, 4):  # Convert RGB(A) to grayscale
    frames = np.mean(frames[..., :3], axis=-1)
# Normalize pixel values if they are in range 0-255
frames = jnp.array(frames, dtype=jnp.float32) / 255.0
# Flatten spatial dimensions into vectors for network observation layer: 
# (num_frames, H, W, C) -> (num_frames, H * W * C)
frames = frames.reshape(frames.shape[0], -1)

key = jr.PRNGKey(0)
model_key, data_key = jr.split(key)

model = TpchModel(
        control_layer_size=CONTROL_WIDTH,
        hidden_sizes=HIDDEN_SHAPE,
        obs_size=OBS_WIDTH,
        key=model_key,
        # weight_decay=weight_decay, this is where regularisation can be enabled, disabled by default.
    )

param_optim = optax.adam(learning_rate=PARAM_LR)
param_opt_state = param_optim.init(eqx.filter(model, eqx.is_array))
activity_optim = optax.adam(learning_rate=ACTIVITY_LR)


# one random "previous states" tuple and one time step of data
prev_key, x_key, y_key = jr.split(data_key, 3)
states_prev = [
    jr.normal(k, (size,))
    for k, size in zip(jr.split(prev_key, 1 + len(HIDDEN_SHAPE)), [CONTROL_WIDTH] + HIDDEN_SHAPE)
]

recorder = PredictionRecorder(output_dir=PREDICTIONS_RECORDING_DIR)
train_step = make_train_step(param_optim, activity_optim, NUM_INFERENCE_STEPS, control_input)
energies = []

# Tracking wall clock performance
start_time = time.perf_counter()
total_frames_processed = 0

for i, y in enumerate(frames[0:N_TRAIN_ITERS]):
    # full JIT inference and weight update for the current frame
    model, param_opt_state, states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace = train_step(
        model, param_opt_state, states_prev, y, return_layerwise=RECORD_ENERGIES
    )
    total_frames_processed += 1
    if RECORD_ENERGIES:
        energies.append(energy_trace.T)

    recorder.append(
    y=y, prior_pred=y_hat_before, posterior_pred=y_hat_after,
    inference_steps_made=NUM_INFERENCE_STEPS, frame_number=i,
    )
    if i % CHECKPOINT_INTERVAL == 0 or i == (N_TRAIN_ITERS - 1):
        print(f"{i}. VFE before inference: {energy_before}")
        print(f"{i}. VFE after inference: {energy_after}")
        metadata["last_frame_processed"] = i
        model.save_checkpoint(root=CHECKPOINT_ROOT, opt_state=param_opt_state, activities=states_curr, metadata=metadata)
        # Flush the prediction recorder to disk for later replaying / reconstruction
        recorder.flush()

    # Pass settled states as previous states for step t + 1
    states_prev = states_curr

# Stop the stopwatch
total_elapsed = time.perf_counter() - start_time
if total_frames_processed > 0:
    avg_ms_per_frame = (total_elapsed / total_frames_processed) * 1000
    fps = total_frames_processed / total_elapsed

    mins, secs = divmod(total_elapsed, 60)
    print(f"\nProcessed {total_frames_processed} frames in {int(mins)}m {secs:.2f}s")
    print(f"Average speed: {avg_ms_per_frame:.2f} ms/frame ({fps:.2f} FPS)\n")

print("Plotting energies...")
plot_energies(
    energies, 
    model=model, 
    save_plot=True, 
    separate_layers=True, 
    output_dir="figures",
    save_individual=True,
    save_overlay=True,
    display=False,
    grid_threshold=8
    )

print("Reconstructing and saving prediction plots...")
# replay and save frames as pngs, needed for compile_videos_from_frames() call below
replay_recordings(
    recordings_dir=PREDICTIONS_RECORDING_DIR, 
    output_shape=(ENV_HEIGHT, ENV_WIDTH),
    total_frames=len(frames),
    output_dir=PREDICTIONS_DIR,
    save_separate=True
    )
compile_videos_from_frames(output_dir=PREDICTIONS_DIR)