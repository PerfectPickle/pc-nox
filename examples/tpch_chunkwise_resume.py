"""Block-scanned training of a Temporal Predictive Coding hierarchy using `make_train_run`.

Demonstrates a checkpoint-resuming, block-based execution workflow for continuing
training of a temporal predictive coding model (`TpchModel`) on sequential video
data. Unlike the interactive per-frame implementation based on `make_train_step`,
this script groups consecutive frames into fixed-length scan blocks and executes
each block through a JIT-compiled `jax.lax.scan`-style training runner. This approach
reduces Python-side dispatch overhead and improves throughput while retaining
block-level checkpointing, prediction recording, energy tracking, and post-training
visualization.

## Key Workflow Phases

1. Checkpoint & Model Restoration

   * Locates the latest `tpch` checkpoint and loads its associated metadata.
   * Reconstructs the parameter and activity optimizers from the saved configuration.
   * Restores the trained `TpchModel`, latent activities, and parameter optimizer state.
   * Resumes processing from the frame immediately following the last frame recorded
     in the checkpoint metadata.

2. Data Ingestion & Preprocessing

   * Loads the complete input video (`example_env.mp4`) into memory.
   * Converts RGB/RGBA frames to grayscale and normalizes pixel values to [0.0, 1.0].
   * Flattens each frame's spatial dimensions into a vector matching the observation
     layer of the predictive coding hierarchy.

3. Block-Scanned Training

   * Divides the remaining training sequence into fixed-length blocks defined by
     `SCAN_BLOCK_LENGTH`.
   * Executes inference and parameter learning for each block through `make_train_run`.
   * Performs `NUM_INFERENCE_STEPS` activity-settling iterations for every frame
     within the scanned block.
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
   * Computes and reports mean VFE before settling, mean VFE after settling, and
     the corresponding energy reduction for each block.
   * Measures total wall-clock training time, average milliseconds per processed
     frame, and effective frames per second.
   * Aggregates recorded energy traces for subsequent layerwise visualization.

6. Post-Training Visualization

   * Generates layerwise training-energy plots from the accumulated inference traces.
   * Replays raw prediction recordings to reconstruct image-space prediction frames.
   * Compiles the reconstructed prediction frames into output video files.

## Trade-off Summary

* Pros: Lower Python dispatch overhead than per-frame execution; efficient vectorized
  block processing; regular checkpointing; prediction recording remains recoverable
  between blocks; preserves detailed per-frame energy traces within each scanned block.
* Cons: Less fine-grained Python-side control during execution than `make_train_step`;
  individual frames inside a scan block cannot be inspected or acted upon until the
  block completes; prediction and energy data are retained at block granularity
  during the training loop.
  """

import imageio.v2 as imageio
raw_frames = imageio.mimread("example_env.mp4", memtest=False) # read this before importing JAX, to avoid os.fork() issues
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*os.fork.*") # to ignore compile_videos() warning

from pc_nox.utils.visualisation import VisualPredictionPlotter, compile_videos_from_frames, plot_train_energies, PredictionRecorder, replay_recordings
from pc_nox.models.tpch import TpchModel, make_train_step, make_train_run
from pc_nox.utils.optim_registry import build_optim
from pc_nox.utils.checkpoints import find_latest_checkpoint, load_metadata
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time

# Number of settling iterations
NUM_INFERENCE_STEPS = 20

# example_env.mp4 has 2000 frames
N_TRAIN_ITERS = 1000

# how many training iterations to perform per scan block
SCAN_BLOCK_LENGTH = 500

# for plotting
RECORD_ENERGIES = True

# where the raw jax arrays get stored during inference/training
PREDICTIONS_RECORDING_DIR = "visual_predictions_raw-load"
# where the reconstructed visual predictions get saved to
PREDICTIONS_DIR = "visual_predictions-load"

control_input = None

# Loading model, activities, and optimisers from saved checkpoint
latest_checkpoint = find_latest_checkpoint(model_type="tpch")
metadata = load_metadata(latest_checkpoint)

param_optim = build_optim(metadata["param_optim"]["name"], learning_rate=metadata["param_optim"]["learning_rate"])
activity_optim = build_optim(metadata["activity_optim"]["name"], learning_rate=metadata["activity_optim"]["learning_rate"])
loaded_checkpoint = TpchModel.load_checkpoint(latest_checkpoint, optim=param_optim)

model: TpchModel = loaded_checkpoint.model
states_prev = loaded_checkpoint.activities
param_opt_state = loaded_checkpoint.opt_state

ENV_WIDTH = metadata["ENV_WIDTH"]
ENV_HEIGHT = metadata["ENV_HEIGHT"]

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



recorder = PredictionRecorder(output_dir=PREDICTIONS_RECORDING_DIR)
train_run_block = make_train_run(param_optim, activity_optim, NUM_INFERENCE_STEPS, run_length=SCAN_BLOCK_LENGTH, control_input=control_input)
all_energy_traces = []
all_energies_before = []
all_energies_after = []

START_FRAME_IDX = metadata["last_frame_processed"] + 1
END_FRAME_IDX = min(START_FRAME_IDX + N_TRAIN_ITERS, len(frames))

# Tracking wall clock performance
start_time = time.perf_counter()
total_frames_processed = 0


for block_start in range(START_FRAME_IDX, END_FRAME_IDX, SCAN_BLOCK_LENGTH):
    print("-----")
    print(f"Now training block starting at index {block_start}...")
    ys_block = frames[block_start : block_start + SCAN_BLOCK_LENGTH]
    current_block_len = len(ys_block)

    # If the remaining frames is less than SCAN_BLOCK_LENGTH, create a temporary tail runner accordingly
    if current_block_len < SCAN_BLOCK_LENGTH:
        active_train_run = make_train_run(
            param_optim, activity_optim, NUM_INFERENCE_STEPS, 
            run_length=current_block_len, control_input=control_input
        )
    else:
        active_train_run = train_run_block

    # JIT compiled inference + learning for current block
    model, param_opt_state, states_prev, y_before, y_after, energies_before, energies_after, energy_traces = active_train_run(
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
        inference_steps_made=NUM_INFERENCE_STEPS,
        frame_numbers=frame_numbers,
    )
    # save checkpoint
    metadata["last_frame_processed"] = block_start + len(ys_block) - 1
    model.save_checkpoint(path=f"checkpoints/step_{block_start}", metadata=metadata, opt_state=param_opt_state, activities=states_prev)

    # Flush the prediction recorder to disk for later replaying / reconstruction
    recorder.flush()

    # Energy plotting
    if RECORD_ENERGIES:
        # energy_traces is (SCAN_BLOCK_LENGTH, NUM_INFERENCE_STEPS, num_layers+1) --
        # one entry per FRAME in this block, not one entry for the whole block.
        # plot_train_energies wants one (num_layers, time_steps) array per
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

# Concatenate into single 1D arrays of shape (num_blocks * run_length,)
all_energies_before = np.concatenate(all_energies_before)
all_energies_after = np.concatenate(all_energies_after)

eb = np.asarray(energies_before)
ea = np.asarray(energies_after)
deltas = eb - ea

print(f"\n--- Overall VFE stats ---")
print(f"Mean VFE Before: {eb.mean():.4f}")
print(f"Mean VFE After:  {ea.mean():.4f}")
print(f"Mean Energy Drop (Δ): {deltas.mean():.4f}")
if total_frames_processed > 0:
    avg_ms_per_frame = (total_elapsed / total_frames_processed) * 1000
    fps = total_frames_processed / total_elapsed

    mins, secs = divmod(total_elapsed, 60)
    print(f"\nProcessed {total_frames_processed} frames in {int(mins)}m {secs:.2f}s")
    print(f"Average speed: {avg_ms_per_frame:.2f} ms/frame ({fps:.2f} FPS)\n")

print("Plotting train energies...")
plot_train_energies(
    all_energy_traces, 
    model=model, 
    save_plot=True, 
    separate_layers=True, 
    output_dir="figures-load",
    save_individual=True,
    save_overlay=True,
    display=False,
    grid_threshold=8,
    iteration_offset=START_FRAME_IDX
    )

print("Reconstructing and saving prediction plots...")
# replay and save frames as pngs, needed for compile_videos_from_frames() call below
replay_recordings(
    recordings_dir=PREDICTIONS_RECORDING_DIR, 
    output_shape=(ENV_HEIGHT, ENV_WIDTH),
    total_frames=len(frames),
    output_dir=PREDICTIONS_DIR
    )
compile_videos_from_frames(output_dir=PREDICTIONS_DIR)