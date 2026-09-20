"""Block-scanned training of a Temporal Predictive Coding hierarchy using
`make_train_run_diffrax`, training from scratch (no checkpoint to resume from).

Demonstrates a block-based execution workflow analogous to the settle_scan
block-scan example, but with activity relaxation fused via `diffrax.diffeqsolve`
(`model.settle_diffrax`) instead of a fixed-iteration `jax.lax.scan` over discrete
optax steps. Consecutive frames are grouped into fixed-length scan blocks and
executed through a JIT-compiled `make_train_run_diffrax` runner, retaining
block-level checkpointing, prediction recording, and energy tracking.

## Key Workflow Phases

1. Data Ingestion & Preprocessing

   * Loads the complete input video (`example_env.mp4`) into memory.
   * Converts RGB/RGBA frames to grayscale and normalizes pixel values to [0.0, 1.0].
   * Flattens each frame's spatial dimensions into a vector matching the observation
     layer of the predictive coding hierarchy.

2. Model & Optimizer Initialization

   * Instantiates a fresh `TpchModel` hierarchy matched to environmental dimensions.
   * Configures an Optax Adam optimizer for the structural weight parameters
     (`param_optim`). There's no separate activity optimizer here (unlike the
     settle_scan example): with `settle_diffrax`, activity relaxation is governed
     by the diffrax `solver` / `stepsize_controller` / `steady_state_tol` instead.
   * Constructs the JIT-compiled block runner via `make_train_run_diffrax`
   * Draws a random set of initial "previous" activities for the first block..

3. Block-Scanned Training

   * Divides the training sequence into fixed-length blocks defined by
     `SCAN_BLOCK_LENGTH`.
   * Executes inference and parameter learning for each block through
     `make_train_run_diffrax`, settling each frame's activities via
     `diffeqsolve` rather than a fixed relaxation-step count.
   * Carries the final settled activities from each block forward as the initial
     state for the next block.
   * Creates a temporary scan runner for the final partial block when fewer than
     `SCAN_BLOCK_LENGTH` frames remain.

4. Block-Level Recording & Checkpointing

   * Records target observations together with prior and posterior predictions for
     every frame in each processed block using `PredictionRecorder.append_block`.
   * Saves a checkpoint after every block containing the updated model, optimizer
     state, activities, and latest processed-frame metadata.
   * Flushes recorded prediction data to disk after each block so that recordings
     remain available for later reconstruction and replay.

5. Energy Tracking & Performance Diagnostics

   * Collects per-frame, per-layer inference energy traces from each scanned block.
     Frames that reach steady state before `max_t1` leave an inf-padded tail in
     their trace, which `plot_energies` trims automatically.
   * Computes and reports mean VFE before settling, mean VFE after settling, and
     the corresponding energy reduction for each block, and overall.
   * Measures total wall-clock training time, average milliseconds per processed
     frame, and effective frames per second.
   * Aggregates recorded energy traces for subsequent layerwise visualization.

6. Post-Training Visualization

   * Generates layerwise training-energy plots from the accumulated inference
     traces, labelling the relaxation axis as continuous integration time.
   * Replays raw prediction recordings to reconstruct image-space prediction frames.
   * Compiles the reconstructed prediction frames into output video files.

## Trade-off Summary

* Pros: Lower Python dispatch overhead than a per-frame `make_train_step_diffrax`
  loop; efficient vectorized block processing; regular checkpointing; prediction
  recording remains recoverable between blocks; adaptive/event-based relaxation
  per frame instead of a fixed iteration count.
* Cons: Less fine-grained Python-side control during execution; individual frames
  inside a scan block cannot be inspected or acted upon until the block completes;
  prediction and energy data are retained at block granularity during the
  training loop.
"""

from pathlib import Path
import imageio.v2 as imageio
video_path = Path(__file__).resolve().parent.parent / "example_env.mp4"
raw_frames = imageio.mimread(str(video_path), memtest=False) # read this before importing JAX, to avoid os.fork() issues

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*os.fork.*") # to ignore compile_videos() warning

from pc_nox.utils.visualisation import compile_videos_from_frames, plot_energies, PredictionRecorder, replay_recordings
from pc_nox.models.tpch import TpchModel, make_train_run_diffrax
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

# how many training iterations to perform per scan block
SCAN_BLOCK_LENGTH = 500

# for plotting
RECORD_ENERGIES = True

# where the raw jax arrays get stored during inference/training
PREDICTIONS_RECORDING_DIR = "visual_predictions_raw"
# where the reconstructed visual predictions get saved to
PREDICTIONS_DIR = "visual_predictions"

CHECKPOINT_ROOT = "checkpoints/diffrax"

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

solver_name = "heun" # if none, defaults to Heun
stepsize_controller_name = "pid" # if none, defaults to standard PID with rtol=1e-3, atol=1e-3
controller_atol = 1e-3
controller_rtol = 1e-3

STEADY_STATE_CRITERION = "relative_rms" # How steady state is determined: "rms" (default), "relative_rms", or "energy_rate"
STEADY_STATE_ATOL = 1e-3
STEADY_STATE_RTOL = 1e-3

# There's no fixed relaxation-step count with settle_diffrax (unlike settle_scan's
# NUM_INFERENCE_STEPS): the solver takes an adaptive/event-terminated number of
# internal steps per frame. n_save is just how many snapshots settle_diffrax saves
# along the way -- used below purely as a display label, not a literal step count.
DIFFRAX_N_SAVE = 20
STEADY_STATE_TOL = 1e-2


metadata = {
    "param_optim": {"name": PARAM_OPTIM_NAME, "learning_rate": PARAM_LR},
    "solver_name": solver_name,
    "stepsize_controller": {
      "name": stepsize_controller_name, 
      "kwargs": {
            "rtol": controller_rtol,
            "atol": controller_atol,
        },
    },
    "steady_state_tol": STEADY_STATE_TOL, # numeric value unused here, just enabling adaptive compute
    "steady_state_criterion": STEADY_STATE_CRITERION,
    "steady_state_atol": STEADY_STATE_ATOL,
    "steady_state_rtol": STEADY_STATE_RTOL,
    "env_width": ENV_WIDTH,
    "env_height": ENV_HEIGHT
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

param_optim = build_optim(PARAM_OPTIM_NAME, learning_rate=PARAM_LR)
param_opt_state = param_optim.init(eqx.filter(model, eqx.is_array))
solver = build_solver(solver_name)
stepsize_controller = build_stepsize_controller(stepsize_controller_name, atol=controller_atol, rtol=controller_rtol)

# one random "previous states" tuple and one time step of data
prev_key, x_key, y_key = jr.split(data_key, 3)
states_prev = [
    jr.normal(k, (size,))
    for k, size in zip(jr.split(prev_key, 1 + len(HIDDEN_SHAPE)), [CONTROL_WIDTH] + HIDDEN_SHAPE)
]

recorder = PredictionRecorder(output_dir=PREDICTIONS_RECORDING_DIR)
train_run_block = make_train_run_diffrax(
    param_optim, run_length=SCAN_BLOCK_LENGTH, control_input=control_input,
    solver=solver, stepsize_controller=stepsize_controller, n_save=DIFFRAX_N_SAVE,
    steady_state_tol=STEADY_STATE_TOL, steady_state_criterion=STEADY_STATE_CRITERION,
    steady_state_atol=STEADY_STATE_ATOL, steady_state_rtol=STEADY_STATE_RTOL
)
all_energy_traces = []
all_energies_before = []
all_energies_after = []

START_FRAME_IDX = 0
END_FRAME_IDX = min(START_FRAME_IDX + N_TRAIN_ITERS, len(frames))

# Tracking wall clock performance
start_time = time.perf_counter()
total_frames_processed = 0

for block_start in range(START_FRAME_IDX, END_FRAME_IDX, SCAN_BLOCK_LENGTH):
    print("-----")
    print(f"Now training block starting at index {block_start}...")
    ys_block = frames[block_start : block_start + SCAN_BLOCK_LENGTH]
    current_block_len = len(ys_block)

    # If remaining frames is less than SCAN_BLOCK_LENGTH, create a temporary tail runner accordingly
    if current_block_len < SCAN_BLOCK_LENGTH:
        active_train_run = make_train_run_diffrax(
            param_optim, run_length=current_block_len, control_input=control_input,
            solver=solver, stepsize_controller=stepsize_controller, n_save=DIFFRAX_N_SAVE,
            steady_state_tol=STEADY_STATE_TOL, steady_state_criterion=STEADY_STATE_CRITERION,
            steady_state_atol=STEADY_STATE_ATOL, steady_state_rtol=STEADY_STATE_RTOL
        )
    else:
        active_train_run = train_run_block

    # JIT compiled inference + learning for current block
    # make_train_run_diffrax returns 9 values (make_train_run's 8, plus ts_traces --
    # the per-frame integration-time grid corresponding to energy_traces). ts_traces
    # isn't needed here: plot_energies trims each frame's inf-padded tail on
    # its own, from the energy values alone.
    model, param_opt_state, states_prev, y_before, y_after, energies_before, energies_after, energy_traces, ts_traces = active_train_run(
        model, param_opt_state, states_prev, ys_block, return_layerwise=True
    )

    total_frames_processed += len(ys_block)

    # Record every frame from this block.
    frame_numbers = np.arange(
        block_start,
        block_start + current_block_len,
    )
    # Record raw y, y_hat_before,y_hat_after
    recorder.append_block(
        y=ys_block,
        prior_pred=y_before,
        posterior_pred=y_after,
        inference_steps_made=DIFFRAX_N_SAVE,
        frame_numbers=frame_numbers,
    )
    # save checkpoint
    last_frame_processed = block_start + len(ys_block) - 1
    metadata["last_frame_processed"] = last_frame_processed
    model.save_checkpoint(root=CHECKPOINT_ROOT, metadata=metadata, opt_state=param_opt_state, activities=states_prev)

    # Flush the prediction recorder to disk for later replaying / reconstruction
    recorder.flush()

    # Energy plotting
    if RECORD_ENERGIES:
        # energy_traces is (SCAN_BLOCK_LENGTH, NUM_INFERENCE_STEPS, num_layers+1) --
        # one entry per FRAME in this block, not one entry for the whole block.
        # plot_energies wants one (num_layers, time_steps) array per
        # recorded iteration, so unpack the block and transpose each frame.
        for frame_trace in np.asarray(energy_traces):
            all_energy_traces.append(frame_trace.T)

    # All calculations happen vectorized
    eb = np.asarray(energies_before)
    ea = np.asarray(energies_after)
    deltas = eb - ea

    all_energies_before.append(eb)
    all_energies_after.append(ea)

    print(f"Mean VFE Before: {eb.mean():.4f}")
    print(f"Mean VFE After:  {ea.mean():.4f}")
    print(f"Mean Energy Drop (Δ): {deltas.mean():.4f}")


# Stop the stopwatch
total_elapsed = time.perf_counter() - start_time
if total_frames_processed > 0:
    avg_ms_per_frame = (total_elapsed / total_frames_processed) * 1000
    fps = total_frames_processed / total_elapsed

    mins, secs = divmod(total_elapsed, 60)
    print(f"\nProcessed {total_frames_processed} frames in {int(mins)}m {secs:.2f}s")
    print(f"Average speed: {avg_ms_per_frame:.2f} ms/frame ({fps:.2f} FPS)\n")


# Concatenate into single 1D arrays of shape (num_blocks * run_length,)
all_energies_before = np.concatenate(all_energies_before)
all_energies_after = np.concatenate(all_energies_after)

eb = all_energies_before
ea = all_energies_after
deltas = eb - ea

print(f"\n--- Overall VFE stats ---")
print(f"Mean VFE Before: {eb.mean():.4f}")
print(f"Mean VFE After:  {ea.mean():.4f}")
print(f"Mean Energy Drop (Δ): {deltas.mean():.4f}\n")

print("Plotting energies...")
plot_energies(
    all_energy_traces, 
    model=model, 
    save_plot=True, 
    separate_layers=True, 
    output_dir="figures",
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