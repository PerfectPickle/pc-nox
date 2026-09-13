from pc_nox.utils.visualisation import VisualPredictionPlotter, compile_videos_from_frames, plot_train_energies
from pc_nox.models.tpch import TpchModel, make_train_step, make_train_run
from pc_nox.utils.optim_registry import build_optim
from pc_nox.utils.checkpoints import find_latest_checkpoint, load_metadata
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import imageio.v2 as imageio
import numpy as np
import time


# Number of settling iterations
NUM_INFERENCE_STEPS = 50

# example_env.mp4 has 2000 frames
N_TRAIN_ITERS = 500

# how many training iterations to perform per scan block
SCAN_BLOCK_LENGTH = 250

# for plotting
RECORD_ENERGIES = True

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

raw_frames = imageio.mimread("example_env.mp4", memtest=False)
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


prediction_plotter = VisualPredictionPlotter(output_shape=(ENV_HEIGHT, ENV_WIDTH))
train_run_block = make_train_run(param_optim, activity_optim, NUM_INFERENCE_STEPS, run_length=SCAN_BLOCK_LENGTH, control_input=control_input)
all_energy_traces = []
all_energies_before = []
all_energies_after = []

START_FRAME_IDX = metadata["last_frame_processed"] + 1
END_FRAME_IDX = min(START_FRAME_IDX + N_TRAIN_ITERS, len(frames))

# Tracking wall clock performance
start_time = time.perf_counter()
total_frames_processed = 0

for block_start in range(START_FRAME_IDX, len(frames), SCAN_BLOCK_LENGTH):
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
    # save checkpoint
    metadata["last_frame_processed"] = block_start + len(ys_block) - 1
    model.save_checkpoint(path=f"checkpoints/step_{block_start}", metadata=metadata, opt_state=param_opt_state, activities=states_prev)

    # Energy plotting
    if RECORD_ENERGIES:
        # energy_traces is (SCAN_BLOCK_LENGTH, NUM_INFERENCE_STEPS, num_layers+1) --
        # one entry per FRAME in this block, not one entry for the whole block.
        # plot_train_energies wants one (num_layers, time_steps) array per
        # recorded iteration, so unpack the block and transpose each frame.
        for frame_trace in np.asarray(energy_traces):
            all_energy_traces.append(frame_trace.T)
            
    n_frames = len(frames)
    for j, i in enumerate(range(block_start, block_start + SCAN_BLOCK_LENGTH)):
        prediction_plotter.update(
            ys_block[j], 
            y_before[j], 
            y_after[j], 
            NUM_INFERENCE_STEPS, 
            frame_number=i, 
            show_combined=False,
            save_combined=True,
            save_separate=False,
            total_frames=n_frames,
            output_dir="visual_predictions4",
            )

    # All calculations happen vectorized
    eb = np.asarray(energies_before)
    ea = np.asarray(energies_after)
    deltas = eb - ea

    all_energies_before.append(eb)
    all_energies_after.append(ea)

    print(f"Mean VFE Before: {eb.mean():.4f}")
    print(f"Mean VFE After:  {ea.mean():.4f}")
    print(f"Mean Energy Drop (Δ): {deltas.mean():.4f}")
    print("-----")


# Stop the stopwatch
total_elapsed = time.perf_counter() - start_time

prediction_plotter.close()

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


plot_train_energies(
    all_energy_traces, 
    model=model, 
    save_plot=True, 
    separate_layers=True, 
    output_dir="figures-load",
    save_individual=True,
    save_overlay=True,
    display=False,
    grid_threshold=8
    )
compile_videos_from_frames(output_dir="visual_predictions4")