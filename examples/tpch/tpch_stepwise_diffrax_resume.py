"""Frame-by-frame training of a Temporal Predictive Coding hierarchy using `make_train_step_diffrax` and a diffrax ODE solver for inference, resuming from a checkpoint.
 
Demonstrates an interactive, per-step, checkpoint-resuming execution workflow for
continuing training of a temporal predictive coding model (`TpchModel`) on
sequential video data. For each frame, activity relaxation is performed by
`diffrax.diffeqsolve` (via `model.settle_diffrax`) rather than by a fixed number of
discrete optimizer steps, so no activity optimizer is involved. Unlike fused
block-scan implementations, this script executes a single JIT-compiled training
step within an outer Python loop. This approach prioritizes fine-grained
introspection, real-time logging, interactive visualization, and frequent
checkpointing over maximum XLA execution speed.
 
Key Workflow Phases
-------------------
1. Checkpoint & Model Restoration
   - Locates the latest `tpch` checkpoint and loads its associated metadata.
   - Reads the environment dimensions (`env_width`, `env_height`) from the metadata.
   - Reconstructs the parameter optimizer, diffrax solver, and stepsize controller
     from the saved configuration.
   - Restores the trained `TpchModel`, latent activities, and parameter optimizer state.
   - Resumes processing from the frame immediately following the last frame recorded
     in the checkpoint metadata.
 
2. Data Ingestion & Preprocessing
   - Loads the complete input video (`example_env.mp4`) into memory (before JAX is
     imported, to avoid os.fork() issues).
   - Converts RGB frames to grayscale and normalizes pixel values to [0.0, 1.0].
   - Flattens each frame's spatial dimensions into a vector matching the observation
     layer of the predictive coding hierarchy.
 
3. Training Step Construction
   - Constructs the JIT-compiled step function via `make_train_step_diffrax`.
   - The steady-state criterion and tolerance that terminate relaxation
     (`STEADY_STATE_CRITERION`, `STEADY_STATE_TOL`) are set at the top of this script
     rather than restored from the checkpoint.
   - There is no fixed relaxation-step count: the solver takes an adaptive,
     event-terminated number of internal steps per frame. `DIFFRAX_N_SAVE` is only
     used as a display label for the number of saved snapshots.
 
4. Per-Step Iterative Training Loop
   - Loops over the next `N_TRAIN_ITERS` frames (capped at the end of the video) in
     Python, executing `train_step` on each iteration.
   - Settles internal activities for each frame by integrating the relaxation ODE
     until steady state (or the integration horizon).
   - Updates model parameters based on settled state errors.
   - Carries settled states (`states_curr`) forward to initialize activities for
     the subsequent time step (`states_prev`).
 
5. In-Loop Introspection & Artifact Generation
   - Logs Variational Free Energy (VFE) before and after activity settling at each
     checkpoint interval and on the final frame.
   - Optionally records per-layer energy traces per frame. Frames that reach steady
     state early leave an inf-padded tail in their trace, which `plot_energies`
     trims automatically.
   - Records the target observation together with the prior and posterior
     predictions for every frame via `PredictionRecorder`.
   - At each checkpoint interval (and on the final frame), saves the model,
     optimizer state, settled activities and updated last-processed-frame metadata
     as a checkpoint, and flushes the recorded predictions to disk.
 
6. Post-Training Diagnostics
   - Plots layerwise energy traces across the entire training trajectory,
     labelling the relaxation axis as continuous integration time.
   - Replays the raw prediction recordings to reconstruct image-space prediction frames.
   - Compiles the reconstructed prediction frames into output video files.
 
Trade-off Summary
-----------------
- Pros: Direct access to intermediate states per frame; effortless integration with
  Python-side visualizers, loggers, and conditional stopping logic; adaptive
  per-frame relaxation instead of a fixed iteration count.
- Cons: Incurs Python loop overhead between JIT step dispatches compared to fused
  block execution (`make_train_run_diffrax`).
"""

from pathlib import Path
import imageio.v2 as imageio
video_path = Path(__file__).resolve().parent.parent / "example_env.mp4"
raw_frames = imageio.mimread(str(video_path), memtest=False) # read this before importing JAX, to avoid os.fork() issues

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*os.fork.*") # to ignore compile_videos() warning

from pc_nox.utils.visualisation import compile_videos_from_frames, plot_energies, PredictionRecorder, replay_recordings
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

# There's no fixed relaxation-step count with settle_diffrax (unlike settle_scan's
# NUM_INFERENCE_STEPS): the solver takes an adaptive/event-terminated number of
# internal steps per frame. n_save is just how many snapshots settle_diffrax saves
# along the way -- used below purely as a display label, not a literal step count.
DIFFRAX_N_SAVE = 20

STEADY_STATE_CRITERION = "energy_rate" # How steady state is determined: "rms" (default), "relative_rms", or "energy_rate"
STEADY_STATE_TOL = 1e-1

# where the raw jax arrays get stored during inference/training
PREDICTIONS_RECORDING_DIR = "visual_predictions_raw"
# where the reconstructed visual predictions get saved to
PREDICTIONS_DIR = "visual_predictions"

CHECKPOINT_ROOT = "checkpoints/diffrax"

# Loading model, activities, and optimisers from saved checkpoint
latest_checkpoint = find_latest_checkpoint(root=CHECKPOINT_ROOT, model_type="tpch")
metadata = load_metadata(latest_checkpoint)

ENV_WIDTH = metadata["env_width"]
ENV_HEIGHT = metadata["env_height"]

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

recorder = PredictionRecorder(output_dir=PREDICTIONS_RECORDING_DIR)
train_step = make_train_step_diffrax(
    param_optim, solver=solver, 
    stepsize_controller=stepsize_controller,
    steady_state_tol=STEADY_STATE_TOL,
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
        all_energy_traces.append(energy_trace.T)

    recorder.append(
    y=y, prior_pred=y_hat_before, posterior_pred=y_hat_after,
    inference_steps_made=DIFFRAX_N_SAVE, frame_number=i,
    )
    if i % CHECKPOINT_INTERVAL == 0 or i == (END_FRAME_IDX - 1):
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
    save_separate=True,
    show_steps_made=False
    )
compile_videos_from_frames(output_dir=PREDICTIONS_DIR)