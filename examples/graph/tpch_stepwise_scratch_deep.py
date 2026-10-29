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

from pc_nox.utils.visualisation import compile_videos_from_frames, plot_energies, PredictionRecorder, VisualPredictionPlotter, replay_recordings
from pc_nox.models.graph import GraphModel, GraphModelConfig, NodeSpec
from pc_nox.models.graph.runners import make_train_step
from pc_nox.models.graph.optim_utils import grouped_optimiser
from pc_nox.models.graph.runners import zero_states

import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time

import torch
import torch.nn.functional as F

# --- DINOv2 Setup ---
DINO_REPO = "facebookresearch/dinov2"
DINO_WEIGHTS = "dinov2_vitb14_reg4_pretrain_86M.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

dino = torch.hub.load(DINO_REPO, "dinov2_vitb14_reg", source="github", pretrained=False)
dino.load_state_dict(torch.load(DINO_WEIGHTS, map_location="cpu"))
dino = dino.to(device).eval()

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

@torch.no_grad()
def dino_encode(images_uint8, batch_size=128):
    """(N, H, W, 3) uint8 -> (N, 768) DINOv2 latents as JAX array."""
    feats = []
    for i in range(0, len(images_uint8), batch_size):
        x = torch.from_numpy(images_uint8[i : i + batch_size]).to(device).float() / 255.0
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        x = (x - IMNET_MEAN) / IMNET_STD
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            f = dino(x)
        feats.append(f.float().cpu().numpy())
    return jnp.array(np.concatenate(feats))


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
CONTROL_ALPHA = None#0.5
HIDDEN_SHAPE = [128, 128, 128, 128, 128] # # width, from highest layer to lowest, including output / sensory layer
HIDDEN_ALPHAS = None#[0.5, 0.5, 0.5, 0.45, 0.25]
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
PARAM_OPTIM_NAME = "adamw"
PARAM_LR = 0.001
ACTIVITY_OPTIM_NAME = "adamw"
ACTIVITY_LR = 0.01

metadata = {
    "param_optim": {
      "name": PARAM_OPTIM_NAME, 
      "base_lr": PARAM_LR,
      "lr_scales": {"kind:pcn_conv": 0.1}
    },
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

# # Stack raw frames as RGB uint8 (N, H, W, 3)
# raw_frames_arr = np.stack(raw_frames)
# if raw_frames_arr.ndim == 4 and raw_frames_arr.shape[-1] >= 3:
#     raw_frames_arr = raw_frames_arr[..., :3]  # Drop alpha if present

# # Encode frames into DINOv2 latents (N, 768)
# X_train_latents = dino_encode(raw_frames_arr)

# # Standardize features using dataset statistics
# feat_mean = X_train_latents.mean(axis=0, keepdims=True)
# feat_std = X_train_latents.std(axis=0, keepdims=True) + 1e-6
# frames = (X_train_latents - feat_mean) / feat_std


key = jr.PRNGKey(0)
model_key, data_key = jr.split(key)

# -------------------------------------------------------------------------------------------------------------
# The graph. What changed from your version, and why:
#   * parents=("ctrl0",)  -- a one-element tuple needs the trailing comma, else Python hands over the STRING "ctrl0".
#   * o3: act_fn="identity" -- "linear" is not in ACT_FN_REGISTRY (tpch_obs is a purely linear readout regardless).
#   * c5, image6: options={"td_output_padding": 1} -- each ConvTranspose2d (k3, s2, p1) maps n -> 2n-1 by default;
#     the extra output padding makes it exactly 2n (2->4, 4->8, and 8->16).
#   * image6 is kind "pcn_conv", not "input": "input" is a parentless data ROOT (the image at the top of a
#     supervised chain). Here the image is the LEAF the decoder predicts, so it is a clamped conv node.
# -------------------------------------------------------------------------------------------------------------
def make_node_specs(obs_dim=128, num_hidden=64, min_width=4, highway_checkpoint_interval=5):
    obs_level = num_hidden + 1
    obs_name = f"o{obs_level}"

    # Calculate node widths bottom-up from observation node
    sizes = []
    curr = obs_dim
    for _ in range(num_hidden + 1):  # 1 control + num_hidden hidden nodes
        curr = max(min_width, curr // 2)
        sizes.append(curr)
    sizes.reverse()  # Index 0 is ctrl0; Index 1..num_hidden are h1..h64

    specs = []

    for idx in range(num_hidden + 1):
        name = "ctrl0" if idx == 0 else f"h{idx}"
        node_type = "tpch_control" if idx == 0 else "tpch_hidden"

        kwargs = {}

        # Set parent connections
        if idx == 1:
            kwargs["parents"] = ("ctrl0",)
        elif idx > 1:
            kwargs["parents"] = (f"h{idx - 1}",)

        # Attach highway_from_td to every n-th node (0, n, 2n, 3n...)
        if highway_checkpoint_interval > 0 and idx % highway_checkpoint_interval == 0:
            kwargs["highway_from_td"] = (obs_name,)

        specs.append(NodeSpec(name, node_type, (sizes[idx],), **kwargs))

    # Append observation node at the bottom
    specs.append(
        NodeSpec(
            obs_name,
            "tpch_obs",
            (obs_dim,),
            td_act_fn="identity",
            parents=(f"h{num_hidden}",),
            clamped=True,
        )
    )

    return tuple(specs)


#NodeSpec("h2", "tpch_hidden", (64,), parents=("h1",), skip_parents=("node m levels above")),

# Example usage with every 10th node getting a highway connection
node_specs = make_node_specs(obs_dim=128, num_hidden=64, min_width=4, highway_checkpoint_interval=5)


node_specs = make_node_specs(obs_dim=128, num_hidden=64)
config = GraphModelConfig(
    node_specs=node_specs,
    # node_specs=(
    #     NodeSpec("ctrl0", "tpch_control", (16,), highway_from_td=("o3",)),
    #     NodeSpec("h1", "tpch_hidden", (32,), parents=("ctrl0",)),
    #     NodeSpec("h2", "tpch_hidden", (64,), parents=("h1",)),
    #     NodeSpec("o3", "tpch_obs", (128,), td_act_fn="identity", parents=("h2",), clamped=True),
    #     # NodeSpec("c4", "pcn_conv", (8, 2, 4), act_fn="tanh", parents=("o3",)),
    #     # NodeSpec("c5", "pcn_conv", (4, 4, 8), act_fn="tanh", parents=("c4",), options={"td_output_padding": 1}),
    #     # NodeSpec("image6", "pcn_conv", (1, 8, 16), act_fn="identity", parents=("c5",), clamped=True,
    #     #          options={"td_output_padding": 1}),
    # ),
    activity_norm="none",
    highway_alpha=1.0,
    bidirectional=False,
    td_precision=1.0,
    bu_precision=1.0,
    energy="pc",
    td_act_fn="gelu", 
    bu_act_fn="gelu",
)
model = GraphModel(config, key=model_key)

#param_optim = grouped_optimiser(config, optax.adam, PARAM_LR, {"kind:pcn_conv": 0.1})
param_optim = optax.adamw(learning_rate=PARAM_LR)
param_opt_state = param_optim.init(eqx.filter(model, eqx.is_array))
activity_optim = optax.adam(learning_rate=ACTIVITY_LR)


# one random "previous states" tuple and one time step of data
prev_key, x_key, y_key = jr.split(data_key, 3)
states_prev = zero_states(model, batch_size=1)

recorder = PredictionRecorder(output_dir=PREDICTIONS_RECORDING_DIR)
train_step = make_train_step(param_optim, activity_optim, NUM_INFERENCE_STEPS)
energies = []

# Tracking wall clock performance
start_time = time.perf_counter()
total_frames_processed = 0

for i, frame in enumerate(frames[0:N_TRAIN_ITERS]):
    y = {"o65": jnp.reshape(frame, (1, -1))}
    #y = {"o3": jnp.reshape(frame, (1, 1, 8, 16))}    # (batch=1, C, H, W); accepts (8, 16) or flat (128,) frames alike
    # full JIT inference and weight update for the current frame
        # full JIT inference and weight update for the current frame. Outputs carry the batch axis, so [0] picks the sequence.
    model, param_opt_state, states_curr, y_hat_before, y_hat_after, energy_before, energy_after, energy_trace = train_step(
        model, param_opt_state, y, states_prev=states_prev, init_direction="td", before_init_dir="td", return_layerwise=False
    )
    states_prev = states_curr                            # thread time: this frame's settled states are the next frame's "previous"
 
    prediction = y_hat_after["o65"][0]                # (1, 8, 16): what the network predicted the frame to be
    energies.append(float(energy_after[0]))

    total_frames_processed += 1
    y_np = np.asarray(frame)  # or np.asarray(y["o3"][0])
    prior_pred_np = np.asarray(y_hat_before["o65"][0])
    posterior_pred_np = np.asarray(y_hat_after["o65"][0])
    recorder.append(
    y=y_np, prior_pred=prior_pred_np, posterior_pred=posterior_pred_np,
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

# print("Plotting energies...")
# plot_energies(
#     energies, 
#     model=model, 
#     save_plot=True, 
#     separate_layers=True, 
#     output_dir="figures",
#     save_individual=True,
#     save_overlay=True,
#     display=False,
#     grid_threshold=8
#     )

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
#     recorder.append(
#     y=y, prior_pred=y_hat_before, posterior_pred=y_hat_after,
#     inference_steps_made=NUM_INFERENCE_STEPS, frame_number=i,
#     )
#     if i % CHECKPOINT_INTERVAL == 0 or i == (N_TRAIN_ITERS - 1):
#         print(f"{i}. VFE before inference: {energy_before}")
#         print(f"{i}. VFE after inference: {energy_after}")
#         metadata["last_frame_processed"] = i
#         model.save_checkpoint(root=CHECKPOINT_ROOT, opt_state=param_opt_state, activities=states_curr, metadata=metadata)
#         # Flush the prediction recorder to disk for later replaying / reconstruction
#         recorder.flush()

#     # Pass settled states as previous states for step t + 1
#     states_prev = states_curr

# # Stop the stopwatch
# total_elapsed = time.perf_counter() - start_time
# if total_frames_processed > 0:
#     avg_ms_per_frame = (total_elapsed / total_frames_processed) * 1000
#     fps = total_frames_processed / total_elapsed

#     mins, secs = divmod(total_elapsed, 60)
#     print(f"\nProcessed {total_frames_processed} frames in {int(mins)}m {secs:.2f}s")
#     print(f"Average speed: {avg_ms_per_frame:.2f} ms/frame ({fps:.2f} FPS)\n")

# # print("Plotting energies...")
# # plot_energies(
# #     energies, 
# #     model=model, 
# #     save_plot=True, 
# #     separate_layers=True, 
# #     output_dir="figures",
# #     save_individual=True,
# #     save_overlay=True,
# #     display=False,
# #     grid_threshold=8
# #     )

# #print("Reconstructing and saving prediction plots...")
# # # replay and save frames as pngs, needed for compile_videos_from_frames() call below
# # replay_recordings(
# #     recordings_dir=PREDICTIONS_RECORDING_DIR, 
# #     output_shape=(24, 32), #(ENV_HEIGHT, ENV_WIDTH),
# #     total_frames=len(frames),
# #     output_dir=PREDICTIONS_DIR,
# #     save_separate=True
# #     )


# # 1. Prepare target pixel vectors (N, 128)
# pixel_targets = np.stack(raw_frames)
# if pixel_targets.ndim == 4 and pixel_targets.shape[-1] >= 3:
#     pixel_targets = np.mean(pixel_targets[..., :3], axis=-1)
# pixel_targets = pixel_targets.reshape(len(pixel_targets), ENV_HEIGHT * ENV_WIDTH) / 255.0

# # 2. Add bias column (1s) to 768-D features -> shape (N, 769)
# X_train_np = np.array(frames)
# X_tilde = np.hstack([X_train_np, np.ones((len(X_train_np), 1))])

# # 3. Solve for weight matrix W: (769, 128)
# print("Fitting linear latent-to-pixel decoder using NumPy...")
# W_decoder, _, _, _ = np.linalg.lstsq(X_tilde, pixel_targets, rcond=None)

# # Helper function to map latents (N, 768) -> decoded pixels (N, 8, 16)
# def decode_to_pixels(latents_array):
#     latents_np = np.asarray(latents_array)
#     # Add bias term for inference
#     latents_tilde = np.hstack([latents_np, np.ones((len(latents_np), 1))])
#     decoded_flat = latents_tilde @ W_decoder  # (N, 128)
#     decoded_clipped = np.clip(decoded_flat, 0.0, 1.0)
#     return decoded_clipped.reshape(-1, ENV_HEIGHT, ENV_WIDTH)


# import glob
# import os

# print("Reconstructing decoded pixel prediction plots...")

# chunk_paths = sorted(glob.glob(os.path.join(PREDICTIONS_RECORDING_DIR, "chunk_*.npz")))
# plotter = VisualPredictionPlotter(output_shape=(ENV_HEIGHT, ENV_WIDTH), cmap="gray")

# try:
#     for path in chunk_paths:
#         data = np.load(path)
        
#         # Decode 768-D latents back to (N, 8, 16) pixel frames
#         y_pix = decode_to_pixels(data["y"])
#         prior_pix = decode_to_pixels(data["prior_pred"])
#         post_pix = decode_to_pixels(data["posterior_pred"])
        
#         for i in range(len(data["frame_numbers"])):
#             plotter.update(
#                 y=y_pix[i],
#                 prior_pred=prior_pix[i],
#                 posterior_pred=post_pix[i],
#                 inference_steps_made=int(data["inference_steps"][i]),
#                 frame_number=int(data["frame_numbers"][i]),
#                 total_frames=len(frames),
#                 output_dir=PREDICTIONS_DIR,
#                 save_separate=True,
#                 save_combined=True,
#             )
# finally:
#     plotter.close()

# # Compile decoded image frames into MP4 video
# compile_videos_from_frames(output_dir=PREDICTIONS_DIR)

# #compile_videos_from_frames(output_dir=PREDICTIONS_DIR)