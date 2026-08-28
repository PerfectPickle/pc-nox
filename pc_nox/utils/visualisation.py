import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import re
import glob
import imageio.v2 as imageio
import os
from matplotlib.lines import Line2D


def plot_train_energies(energies, ts, layer_labels=None, colormaps=None, save_plot: bool = False, output_dir: str = "figures"):
    r"""
    Plots training energies over inference iterations for arbitrary network sizes.
    
    Parameters:
    -----------
    energies : list or np.ndarray
        Training energy arrays per iteration. Shape: (num_train_iters, num_layers, time_steps)
    ts : list or np.ndarray
        Time steps array where ts[0] defines t_max cutoff.
    layer_labels : list of str, optional
        Custom labels for each layer/component. Defaults to [r"$\ell_1$", r"$\ell_2$", ...].
    colormaps : list of str or Colormap objects, optional
        List of colormaps to distinguish layers. Defaults to standard sequential maps.
    """
    t_max = int(ts[0])
    num_iterations = len(energies)
    num_layers = energies[0].shape[0]  # Dynamically detect network layer count
    
    # 1. Dynamic Legend Labels
    if layer_labels is None:
        layer_labels = [rf"$\ell_{{{i+1}}}$" for i in range(num_layers)]
        
    # 2. Dynamic Colormap Selection
    default_cmap_names = ["Blues", "Reds", "Greens", "Oranges", "Purples", "YlOrBr", "PuBu", "RdPu"]
    if colormaps is None:
        if num_layers <= len(default_cmap_names):
            colormaps = [plt.get_cmap(name) for name in default_cmap_names[:num_layers]]
        else:
            # Fallback for large networks: generate distinct hue colormaps on the fly
            colormaps = []
            for i in range(num_layers):
                hue = i / num_layers
                colors = [mcolors.hsv_to_rgb((hue, 0.25, 0.9)), mcolors.hsv_to_rgb((hue, 1.0, 0.4))]
                colormaps.append(mcolors.LinearSegmentedColormap.from_list(f"dynamic_cmap_{i}", colors))
    else:
        colormaps = [plt.get_cmap(c) if isinstance(c, str) else c for c in colormaps]

    norm = mcolors.Normalize(vmin=0, vmax=max(1, num_iterations - 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # 3. Dynamic Plotting Loop
    for t, energies_iter in enumerate(energies):
        norm_t = norm(t)
        for i in range(num_layers):
            ax.plot(energies_iter[i, :t_max], color=colormaps[i](norm_t))

    legend_handles = [
        Line2D([0], [0], color=colormaps[i](0.85), linewidth=3)
        for i in range(num_layers)
    ]

    # Formatting & Legend
    ax.legend(legend_handles, layer_labels, loc="upper right", fontsize=16)
    
    # Modern colorbar setup (removed legacy `sm._A = []` syntax)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("Greys"), norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Training iteration", fontsize=16, labelpad=14)
    cbar.ax.tick_params(labelsize=14)
    
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("Inference iterations", fontsize=18, labelpad=14)
    ax.set_ylabel("Energy", fontsize=18, labelpad=14)
    ax.set_yscale("log")
    
    plt.tight_layout()

    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "train_energies.png"), bbox_inches='tight', dpi=150)
        

    plt.show()



##
##################### Save / plot / show prediction frame ###########################
##

def plot_visual_prediction(
        y: ArrayLike,
        prior_pred: ArrayLike,
        posterior_pred: ArrayLike,
        inference_steps_made: int,
        output_shape: tuple,
        frame_number: int,
        figsize: tuple = (10, 3),
        cmap: str = 'gray',
        show_combined: bool = True,
        save_combined: bool = True,
        save_separate: bool = False,
        output_dir: str = "visual_predictions",
        total_frames: int = 10000,
    ):
    """
    Plots ground-truth visual output (generative process frame) against the
    generative model's reconstructed (reshaped) predicted output, before and
    after inference.

    Parameters:
    -----------
    y : jax.Array or ArrayLike
        Ground truth, i.e. current frame
    prior_pred : jax.Array or ArrayLike
        Predicted sensory observation / output BEFORE inference
    posterior_pred : jax.Array or ArrayLike
        Predicted sensory observation / output AFTER inference
    inference_steps_made : int
        Number of inference steps taken to achieve the posterior
    output_shape : tuple
        Shape to reshape each flattened array into before plotting
        (e.g. (H, W) or (H, W, C))
    frame_number : int
        Index of the current frame, used for the plot title and
        for constructing output filenames
    figsize : tuple, optional
        Size of the combined (1x3) figure. Default (10, 3)
    cmap : str, optional
        Colormap used for all three images. Default 'gray'
    show_combined : bool, optional
        Whether to display the figure interactively (plt.show()).
        Default True
    save_combined : bool, optional
        Whether to save the combined 3-panel figure to disk.
        Saved under `{output_dir}/combined/frame_{frame_number:04d}.png`.
        Default True
    save_separate : bool, optional
        Whether to additionally save y, prior_pred, and posterior_pred
        as three separate, borderless images (useful for building an
        animation/video from a directory of frames later). Each is saved
        under its own subfolder:
        `{output_dir}/ground_truth/frame_{frame_number:04d}.png`,
        `{output_dir}/prior_pred/frame_{frame_number:04d}.png`,
        `{output_dir}/posterior_pred/frame_{frame_number:04d}.png`.
        Default False
    output_dir : str, optional
        Base directory under which the `combined/` and (if requested)
        per-array subfolders are created. Default "visual_predictions"
    total_frames : int, optional
        Total number of frames you expect to save, used to compute a
        zero-padding width wide enough to keep filenames sortable
        lexicographically (e.g. total_frames=1_000_000 -> 6-digit
        padding: frame_000000.png ... frame_999999.png).

    Returns:
    --------
    None
    """

    pad_width = len(str(total_frames - 1))
    frame_str = f"frame_{frame_number:0{pad_width}d}.png"

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(y.reshape(output_shape), cmap=cmap)
    axes[0].set_title(f"Frame {frame_number}: Ground Truth")
    axes[1].imshow(prior_pred.reshape(output_shape), cmap=cmap)
    axes[1].set_title("Prior (Step 0)")
    axes[2].imshow(posterior_pred.reshape(output_shape), cmap=cmap)
    axes[2].set_title(f"Posterior (Step {inference_steps_made})")
    for ax in axes:
        ax.axis('off')

    if save_combined:
        combined_dir = os.path.join(output_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)
        fig.savefig(os.path.join(combined_dir, frame_str), bbox_inches='tight', dpi=150)

    if save_separate:
        arrays_by_name = {
            "ground_truth": y,
            "prior_pred": prior_pred,
            "posterior_pred": posterior_pred,
        }
        for name, arr in arrays_by_name.items():
            sub_dir = os.path.join(output_dir, name)
            os.makedirs(sub_dir, exist_ok=True)
            img = np.asarray(arr).reshape(output_shape)
            plt.imsave(os.path.join(sub_dir, frame_str), img, cmap=cmap)

    if show_combined:
        plt.show()
    else:
        plt.close(fig)



##
##################### Save videos from saved frames ###########################
##

def _natural_key(path: str):
    """Sort key that orders 'frame_2' before 'frame_10' regardless of
    zero-padding width, so mixed-padding runs still sort correctly."""
    fname = os.path.basename(path)
    return [int(tok) if tok.isdigit() else tok
            for tok in re.split(r'(\d+)', fname)]


def compile_videos_from_frames(
        output_dir: str = "visual_predictions",
        fps: int = 15,
        subdirs: list[str] | None = None,
        quality: int = 6,
    ):
    """
    Compiles each subfolder of frame images inside `output_dir` into its
    own mp4, saved back into `output_dir`.

    e.g. visual_predictions/ground_truth/frame_*.png ->
         visual_predictions/ground_truth.mp4

    Parameters
    ----------
    output_dir : str
        Base directory containing per-array subfolders of frames
        (as produced by plot_visual_prediction).
    fps : int
        Frames per second for the output video. Default 15.
    subdirs : list of str, optional
        Which subfolders to compile (e.g. ["ground_truth", "prior_pred"]).
        If None, auto-detects all subfolders of `output_dir` that contain
        at least one frame_*.png file.
    quality : int
        imageio/ffmpeg quality setting, 0 (worst) to 10 (best).
        Default 6 is decent/reasonable, not maximal.

    Requires: pip install imageio[ffmpeg]
    """
    if subdirs is None:
        subdirs = [
            d for d in sorted(os.listdir(output_dir))
            if os.path.isdir(os.path.join(output_dir, d))
            and glob.glob(os.path.join(output_dir, d, "frame_*.png"))
        ]

    for sub in subdirs:
        frame_paths = sorted(
            glob.glob(os.path.join(output_dir, sub, "frame_*.png")),
            key=_natural_key,
        )
        if not frame_paths:
            print(f"Skipping '{sub}': no frames found.")
            continue

        video_path = os.path.join(output_dir, f"{sub}.mp4")
        with imageio.get_writer(
            video_path, fps=fps, codec='libx264', quality=quality, macro_block_size=1
        ) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))

        print(f"Saved {len(frame_paths)} frames -> {video_path}")