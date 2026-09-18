"""Frame-by-frame training of a Temporal Predictive Coding hierarchy using `make_train_step` and diffrax ODE solver for inference.
# TODO Update docstirng
Demonstrates an interactive, per-step execution workflow for training a 
temporal predictive coding model (`TpchModel`) on sequential video data. 
Unlike fused block-scan implementations, this script executes a single JIT-compiled 
training step within an outer Python loop. This approach prioritizes fine-grained 
introspection, real-time logging, interactive visualization, and frequent 
checkpointing over maximum XLA execution speed.

Key Workflow Phases
-------------------
1. Data Ingestion & Preprocessing
   - Reads input video (`example_env.mp4`) frame-by-frame.
   - Converts RGB frames to grayscale and normalizes pixel values to [0.0, 1.0].

2. Model & Optimizer Initialization
   - Instantiates a `TpchModel` hierarchy matched to environmental dimensions.
   - Configures separate Optax Adam optimizers for structural weight parameters 
     (`param_optim`) and latent activity relaxation (`activity_optim`).
   - Constructs the JIT-compiled step function via `make_train_step`.

3. Per-Step Iterative Training Loop
   - Loops over frame sequences in Python, executing `train_step` on each iteration.
   - Settles internal activities over `NUM_INFERENCE_STEPS` relaxation steps.
   - Updates model parameters based on settled state errors.
   - Carries settled states (`states_curr`) forward to initialize activities for 
     the subsequent time step (`states_prev`).

4. In-Loop Introspection & Artifact Generation
   - Logs Variational Free Energy (VFE) before and after activity settling.
   - Optionally records per-layer energy traces per relaxation step.
   - Renders visual reconstructions (prior vs. posterior predictions) per frame.
   - Saves model checkpoints and optimizer states at defined intervals.

5. Post-Training Diagnostics
   - Plots layerwise energy traces across the entire training trajectory.
   - Compiles generated visual prediction frames into output video files.

Trade-off Summary
-----------------
- Pros: Direct access to intermediate states per frame; effortless integration with 
  Python-side visualizers, loggers, and conditional stopping logic.
- Cons: Incurs Python loop overhead between JIT step dispatches compared to 
  fused `jax.lax.scan` execution (`make_train_run`).
"""

import imageio.v2 as imageio
raw_frames = imageio.mimread("example_env.mp4", memtest=False) # read this before importing JAX, to avoid os.fork() issues
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*os.fork.*") # to ignore compile_videos() warning

from pc_nox.utils.visualisation import compile_videos_from_frames, plot_train_energies, PredictionRecorder, replay_recordings
from pc_nox.utils.checkpoints import find_latest_checkpoint, load_metadata
from pc_nox.models.tpch import TpchModel, make_train_step_diffrax
from pc_nox.utils.optim_registry import build_optim
from pc_nox.utils.solver_registry import build_solver
from pc_nox.utils.stepsize_controller_registry import build_stepsize_controller
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time

# example_env.mp4 has 2000 frames
N_TRAIN_ITERS = 1000

# training step interval to save checkpoint at
CHECKPOINT_INTERVAL = 500

# for plotting
RECORD_ENERGIES = True

STEADY_STATE_CRITERION = "energy_rate" # How steady state is determined: "rms" (default), "relative_rms", or "energy_rate"

# where the raw jax arrays get stored during inference/training
PREDICTIONS_RECORDING_DIR = "visual_predictions_diff-load_raw"
# where the reconstructed visual predictions get saved to
PREDICTIONS_DIR = "visual_predictions_dif-load"

# Loading model, activities, and optimisers from saved checkpoint
latest_checkpoint = find_latest_checkpoint(root="checkpoints/diffrax-save-relative", model_type="tpch")
metadata = load_metadata(latest_checkpoint)

param_optim = build_optim(metadata["param_optim"]["name"], learning_rate=metadata["param_optim"]["learning_rate"])
solver = build_solver(metadata["solver_name"])
stepsize_controller = build_stepsize_controller(metadata["stepsize_controller"]["name"], **metadata["stepsize_controller"]["kwargs"])

loaded_checkpoint = TpchModel.load_checkpoint(latest_checkpoint, optim=param_optim)

model: TpchModel = loaded_checkpoint.model
states_prev = loaded_checkpoint.activities
param_opt_state = loaded_checkpoint.opt_state

control_input = None


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

recorder = PredictionRecorder(output_dir="visual_predictions_raw")
train_step = make_train_step_diffrax(
    param_optim, solver=solver, 
    stepsize_controller=stepsize_controller,
    steady_state_tol=metadata["steady_state_tol"],
    steady_state_criterion=STEADY_STATE_CRITERION,
    control_input=control_input
    )
all_energy_traces = []
all_energies_before = []
all_energies_after = []


START_FRAME_IDX = metadata["last_frame_processed"] + 1
END_FRAME_IDX = min(START_FRAME_IDX + N_TRAIN_ITERS, len(frames))

# Tracking wall clock performance
start_time = time.perf_counter()
total_frames_processed = 0

for i, y in enumerate(frames[START_FRAME_IDX:END_FRAME_IDX], start=START_FRAME_IDX):
    # full JIT inference and weight update for the current frame
    model, param_opt_state, states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace, ts = train_step(
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
        model.save_checkpoint(opt_state=param_opt_state, activities=states_curr, metadata=metadata)
        # Flush the prediction recorder to disk for later replaying / reconstruction
        recorder.flush()
        print(f"Save raw y predictions to {PREDICTIONS_RECORDING_DIR}")

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

print("Plotting train energies...")
plot_train_energies(
    energies, 
    model=model, 
    save_plot=True, 
    separate_layers=True, 
    output_dir="figures-diffrax-load",
    save_individual=True,
    save_overlay=True,
    display=False,
    grid_threshold=8,
    x_axis_label="Inference time (t)"
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