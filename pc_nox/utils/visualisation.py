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
from matplotlib.collections import LineCollection



##
##################### Plot Energies ###########################
##

import os
import re
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# A small set of hand-picked, maximally-distinguishable anchor hues
# (based on matplotlib's tab10/tab20 qualitative families, which are
# chosen specifically to stay separable rather than just evenly spaced
# around the hue wheel). Order matters: earlier entries are the most
# mutually distinct, so with few layers you get the clearest subset.
_QUALITATIVE_ANCHORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#000000",  # black
    "#f5b041",  # gold
]


def _qualitative_layer_colormaps(num_layers):
    """
    One colormap per layer, each running from a light tint of a
    hand-picked anchor color up to the anchor color itself. Layer
    identity is carried by hue (anchor color), training-iteration
    progress is carried by lightness/saturation within that hue.

    If num_layers exceeds the anchor list, colors are recycled (a repeat
    is unavoidable without picking a worse hue) -- with enough layers to
    hit that, you should be relying on the per-layer grid/individual
    exports rather than color alone to tell layers apart anyway.
    """
    colormaps = []
    for i in range(num_layers):
        anchor = _QUALITATIVE_ANCHORS[i % len(_QUALITATIVE_ANCHORS)]
        r, g, b = mcolors.to_rgb(anchor)
        light = tuple(c + (1 - c) * 0.85 for c in (r, g, b))
        colormaps.append(
            mcolors.LinearSegmentedColormap.from_list(f"qual_cmap_{i}", [light, anchor])
        )
    return colormaps


def _procedural_layer_colormaps(num_layers):
    """
    Legacy hue-spaced generation (as in the original function). Kept
    available via `color_scheme="procedural"`, but this is the scheme
    that gets visually confusing once you have more than a handful of
    layers, since evenly-spaced hue angles are not the same thing as
    perceptually distinct colors.
    """
    colormaps = []
    for i in range(num_layers):
        hue = i / num_layers
        colors = [mcolors.hsv_to_rgb((hue, 0.25, 0.9)), mcolors.hsv_to_rgb((hue, 1.0, 0.4))]
        colormaps.append(mcolors.LinearSegmentedColormap.from_list(f"proc_cmap_{i}", colors))
    return colormaps


def _sanitize_for_filename(label):
    """
    Turns a layer label (which may contain LaTeX like r"$\\ell_{3}$", or
    free text like "Hidden 2") into a safe, readable filename fragment.
    """
    s = re.sub(r"[\\${}]", "", label)          # strip LaTeX markup chars
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)        # collapse everything else to underscores
    s = s.strip("_").lower()
    return s or "layer"


def _grid_dims(num_layers, max_cols=None):
    """
    Picks (nrows, ncols) for a roughly-square grid of subplots. Capped by
    `max_cols` if given, otherwise sized to sqrt(num_layers) -- e.g. a
    10x10 grid for 100 layers rather than a 100-row column.
    """
    ncols = max(1, math.ceil(math.sqrt(num_layers)))
    if max_cols is not None:
        ncols = min(ncols, max_cols)
    nrows = math.ceil(num_layers / ncols)
    return nrows, ncols


def _draw_layer_traces(ax, layer_idx, energies, trace_lengths, cmap, norm, iter_positions,
                        linewidth=1.2, alpha=0.85):
    """
    Draws one layer's traces (across all recorded training iterations)
    onto `ax` as a single batched LineCollection. Shared by the facet/grid
    view and the standalone individual-layer exports, so there's exactly
    one place that knows how to draw a layer.
    """
    colors = [cmap(norm(t)) for t in iter_positions]
    segments = [
        np.column_stack([np.arange(length), energies_iter[layer_idx, :length]])
        for energies_iter, length in zip(energies, trace_lengths)
    ]
    lc = LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_yscale("log")


def _draw_overlay(ax, num_layers, energies, trace_lengths, colormaps, norm, iter_positions,
                   layer_labels, ylabel):
    """
    Draws all layers overlaid on a single axis (the original combined
    view). Used both for the main plot when separate_layers=False, and
    for the optional standalone overlay export when separate_layers=True.

    Legend sizing adapts to num_layers: a plain single-column legend at
    "upper right" works fine for a handful of layers, but with dozens+ it
    would either overflow the figure (crashing tight_layout) or become
    unreadable clutter. Past a threshold it switches to a smaller,
    multi-column legend placed outside the axes -- still not something
    you'd read layer-by-layer at 100 entries, but it renders correctly
    and remains legible for moderate counts. For genuinely large
    networks, the grid/individual exports are the actual tool for
    telling layers apart; this overlay is a "everything at a glance"
    supplement, not a replacement.
    """
    legend_handles = []
    for i in range(num_layers):
        _draw_layer_traces(ax, i, energies, trace_lengths, colormaps[i], norm, iter_positions,
                            linewidth=1.0, alpha=1.0)
        legend_handles.append(Line2D([0], [0], color=colormaps[i](0.85), linewidth=3))

    if num_layers <= 15:
        ax.legend(legend_handles, layer_labels, loc="upper right", fontsize=16)
    else:
        ncol = math.ceil(num_layers / 15)
        fontsize = max(6, 16 - 0.1 * num_layers)
        ax.legend(
            legend_handles, layer_labels,
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=fontsize, ncol=ncol, borderaxespad=0,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("Inference iterations", fontsize=18, labelpad=14)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=14)


def _add_overlay_colorbar(fig, ax, norm, num_layers):
    """
    Adds the "training iteration" colorbar for an overlay plot. When the
    legend is small enough to sit at "upper right" inside the axes (see
    _draw_overlay), a vertical colorbar to the right of the axes is fine.
    Once the legend grows and gets pushed outside the axes to the right
    (num_layers > 15), a vertical colorbar there would collide with it --
    so past that threshold the colorbar moves to a horizontal strip above
    the plot instead, leaving the whole right margin free for the legend.
    """
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("Greys"), norm=norm)
    if num_layers <= 15:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Training iteration", fontsize=16, labelpad=14)
        cbar.ax.tick_params(labelsize=14)
    else:
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", location="top",
                             fraction=0.05, pad=0.03)
        cbar.set_label("Training iteration", fontsize=12, labelpad=8)
        cbar.ax.tick_params(labelsize=10)
    return cbar


def plot_train_energies(
    energies,
    model=None,
    t_max=None,
    layer_labels=None,
    colormaps=None,
    color_scheme: str = "qualitative",
    ylabel: str = "Variational Free Energy (VFE)",
    separate_layers: bool = False,
    layout: str = "auto",
    grid_threshold: int = 6,
    max_cols: int = None,
    share_y: bool = False,
    save_plot: bool = False,
    save_overlay: bool = False,
    save_individual: bool = False,
    display: bool = True,
    dpi: int = 300,
    output_dir: str = "figures",
):
    r"""
    Plots training energies (e.g. variational free energy) over inference
    iterations for arbitrary network sizes.

    Args:
    energies: list or np.ndarray. Training energy arrays per iteration. Shape: (num_train_iters, num_layers, time_steps)
    model: ModelBase, optional. If given and `layer_labels` is not, labels are auto-derived via `type(model).layer_labels(model.config)`. Falls back to the generic $\ell_1, \ell_2, ...$ labels silently if the model doesn't implement it, same as `model=None`.
    t_max: int, optional. Caps how many inference-relaxation steps are plotted per recorded iteration, via `min(that iteration's own trace length, t_max)`. Traces are allowed to have different lengths between recorded iterations; each is handled independently.
    layer_labels: list of str, optional. Explicit labels for each layer/component. Takes precedence over `model`. Defaults to [r"$\ell_1$", r"$\ell_2$", ...].
    colormaps: list of str or Colormap objects, optional. Explicit colormaps, one per layer. Takes precedence over `color_scheme`, and is honored in every layout (single overlay, column, and grid) -- never silently collapsed to a single global colormap.
    color_scheme: {"qualitative", "procedural"}, optional. Only used when `colormaps` is None. "qualitative" (default) uses a small curated set of hand-picked, distinguishable anchor colors. "procedural" reproduces the original hue-spaced auto-generation.
    ylabel: str, optional. Label for the energy axis. Default "Variational Free Energy (VFE)". Applied to the overlay plot's y-axis and, in facet/grid mode, as a single figure-wide label (`fig.supylabel`) rather than repeated on every subplot.
    separate_layers: bool, optional. If True, gives each layer its own subplot instead of overlaying everything on one axis. Default False.
    layout: {"auto", "column", "grid"}, optional. Only relevant when `separate_layers=True`. "column" always stacks subplots in a single column (the original facet behavior) -- fine for a handful of layers but becomes an impractically tall image for large networks. "grid" always arranges subplots in a roughly-square grid (e.g. 10x10 for 100 layers), which scales much better. "auto" (default) picks "column" when `num_layers <= grid_threshold` and "grid" otherwise.
    grid_threshold: int, optional. Layer count above which `layout="auto"` switches from column to grid. Default 6.
    max_cols: int, optional. Caps the number of columns in grid layout. Default None (columns sized to sqrt(num_layers)).
    share_y: bool, optional. Only relevant when `separate_layers=True`. If True, all subplots share one y-axis range. Default False, because e.g. an observation/output layer's energy can sit at a genuinely different scale than hidden layers -- a shared axis can flatten smaller-magnitude layers into an unreadable line.
    save_plot: bool, optional. Whether to save the main displayed figure (whatever `separate_layers`/`layout` produces) to `{output_dir}/train_energies.png`. Default False.
    save_overlay: bool, optional. If True, additionally renders and saves the single-axis, all-layers overlay to `{output_dir}/train_energies_overlay.png`, even when `separate_layers=True` is used for the main/displayed plot. Lets you keep both the "many small panels" and "everything at a glance" views without calling this function twice. Default False.
    save_individual: bool, optional. If True, additionally saves each layer as its own standalone PNG under `{output_dir}/individual/`, named after the layer's (sanitized) label, e.g. `train_energies_03_hidden_2.png`. Useful for large networks where you want to inspect one layer at a time without regenerating the whole plot. Default False.
    display: bool, optional. If true, the plot is displayed using plt.show().
    dpi: int, optional. What dpi (resolution) to save the plots at.
    output_dir: str, optional. Directory under which plots are saved, if any of the `save_*` options are True. Default "figures".

    Returns:
    None
    """
    if layout not in ("auto", "column", "grid"):
        raise ValueError(f"layout must be 'auto', 'column', or 'grid', got {layout!r}")

    t_max = int(t_max) if t_max is not None else None
    num_iterations = len(energies)
    num_layers = energies[0].shape[0]  # Dynamically detect network layer count

    # 1. Dynamic Legend Labels (unchanged from the original)
    if layer_labels is None and model is not None:
        try:
            layer_labels = type(model).layer_labels(model.config)
        except NotImplementedError:
            layer_labels = None

    if layer_labels is None:
        layer_labels = [rf"$\ell_{{{i+1}}}$" for i in range(num_layers)]

    if len(layer_labels) != num_layers:
        raise ValueError(
            f"layer_labels has {len(layer_labels)} entries but energies has "
            f"{num_layers} layers per iteration -- these must match "
            "(e.g. re-derive labels if the model architecture changed since "
            "these energies were logged)."
        )

    # 2. Colormap selection -- explicit `colormaps` always wins, in every layout.
    if colormaps is None:
        if color_scheme == "qualitative":
            colormaps = _qualitative_layer_colormaps(num_layers)
        elif color_scheme == "procedural":
            colormaps = _procedural_layer_colormaps(num_layers)
        else:
            raise ValueError(
                f"color_scheme must be 'qualitative' or 'procedural', got {color_scheme!r}"
            )
    else:
        colormaps = [plt.get_cmap(c) if isinstance(c, str) else c for c in colormaps]
        if len(colormaps) != num_layers:
            raise ValueError(
                f"colormaps has {len(colormaps)} entries but there are "
                f"{num_layers} layers -- pass one colormap per layer."
            )

    norm = mcolors.Normalize(vmin=0, vmax=max(1, num_iterations - 1))
    trace_lengths = [
        min(energies_iter.shape[1], t_max) if t_max is not None else energies_iter.shape[1]
        for energies_iter in energies
    ]
    iter_positions = list(range(num_iterations))

    resolved_layout = layout
    if separate_layers and layout == "auto":
        resolved_layout = "grid" if num_layers > grid_threshold else "column"

    # 3. Build the main/displayed figure.
    if separate_layers:
        if resolved_layout == "grid":
            nrows, ncols = _grid_dims(num_layers, max_cols=max_cols)
        else:
            nrows, ncols = num_layers, 1

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(max(3.2 * ncols, 5), max(2.2 * nrows, 4)),
            sharex=True,
            sharey=share_y,
            layout="constrained",
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for i in range(num_layers):
            ax = flat_axes[i]
            _draw_layer_traces(ax, i, energies, trace_lengths, colormaps[i], norm, iter_positions)
            ax.set_title(layer_labels[i], fontsize=11)
            ax.tick_params(axis="both", which="major", labelsize=10)

        # Hide any unused cells in a non-exact grid (e.g. 10 layers in a 4x3 grid).
        for j in range(num_layers, len(flat_axes)):
            flat_axes[j].set_visible(False)

        used_axes = flat_axes[:num_layers]
        fig.supxlabel("Inference iterations", fontsize=14)
        fig.supylabel(ylabel, fontsize=14)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("Greys"), norm=norm)
        cbar = fig.colorbar(sm, ax=list(used_axes), fraction=0.02, pad=0.02)
        cbar.set_label("Training iteration", fontsize=12, labelpad=12)
        cbar.ax.tick_params(labelsize=10)

    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        _draw_overlay(ax, num_layers, energies, trace_lengths, colormaps, norm, iter_positions,
                      layer_labels, ylabel)
        ax.tick_params(axis="both", which="major", labelsize=16)
        _add_overlay_colorbar(fig, ax, norm, num_layers)

    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "train_energies.png"), bbox_inches="tight", dpi=dpi)

    # 4. Optional standalone overlay export, independent of the main layout above.
    if save_overlay and separate_layers:
        os.makedirs(output_dir, exist_ok=True)
        overlay_fig, overlay_ax = plt.subplots(figsize=(8, 4))
        _draw_overlay(overlay_ax, num_layers, energies, trace_lengths, colormaps, norm,
                      iter_positions, layer_labels, ylabel)
        overlay_ax.tick_params(axis="both", which="major", labelsize=16)
        _add_overlay_colorbar(overlay_fig, overlay_ax, norm, num_layers)
        overlay_fig.savefig(os.path.join(output_dir, "train_energies_overlay.png"),
                             bbox_inches="tight", dpi=dpi)
        plt.close(overlay_fig)

    # 5. Optional per-layer standalone exports.
    if save_individual:
        indiv_dir = os.path.join(output_dir, "vfe_layerwise")
        os.makedirs(indiv_dir, exist_ok=True)
        for i in range(num_layers):
            ind_fig, ind_ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
            _draw_layer_traces(ind_ax, i, energies, trace_lengths, colormaps[i], norm, iter_positions)
            ind_ax.set_title(layer_labels[i], fontsize=13)
            ind_ax.set_xlabel("Inference iterations", fontsize=12)
            ind_ax.set_ylabel(ylabel, fontsize=12)
            ind_ax.tick_params(axis="both", which="major", labelsize=11)
            fname = f"train_energies_{i+1:03d}_{_sanitize_for_filename(layer_labels[i])}.png"
            ind_fig.savefig(os.path.join(indiv_dir, fname), bbox_inches="tight", dpi=dpi)
            plt.close(ind_fig)  # close immediately -- important once num_layers is large

    if display:
        plt.show()
    plt.close(fig)


##
##################### Save / plot / show prediction frame ###########################
##

#TODO improve performance
class VisualPredictionPlotter:
    """
    Reusable plotter for visual predictions — avoids re-creating the
    Figure/Axes on every call (expensive when called once per training frame).

    Maintains a single Figure/Axes/image-artist set for the lifetime of the
    object; each call to `update()` mutates the existing artists' data in
    place rather than rebuilding the plot from scratch.
    """

    def __init__(self, output_shape, figsize=(10, 3), cmap='gray'):
        """
        Creates the figure, axes, and image artists once, up front.

        Parameters:
        -----------
        output_shape : tuple
            Shape to reshape each flattened array into before plotting
            (e.g. (H, W) or (H, W, C))
        figsize : tuple, optional
            Size of the combined (1x3) figure. Default (10, 3)
        cmap : str, optional
            Colormap used for all three images. Default 'gray'

        Returns:
        --------
        None
        """
        self.output_shape = output_shape
        self.fig, self.axes = plt.subplots(1, 3, figsize=figsize)
        self.cmap = cmap
        # Create the three image artists once, with dummy data
        blank = np.zeros(output_shape)
        self.im_y = self.axes[0].imshow(blank, cmap=cmap)
        self.im_prior = self.axes[1].imshow(blank, cmap=cmap)
        self.im_post = self.axes[2].imshow(blank, cmap=cmap)
        self.axes[1].set_title("Prior (Step 0)")
        for ax in self.axes:
            ax.axis('off')
        # Do layout once up front instead of on every savefig call
        self.fig.tight_layout()


    def update(
        self,
        y, prior_pred, posterior_pred,
        inference_steps_made, frame_number,
        show_combined=False, save_combined=True,
        save_separate=False, output_dir="visual_predictions",
        total_frames=10000,
    ):
        """
        Updates ground-truth visual output (generative process frame) against
        the generative model's reconstructed (reshaped) predicted output,
        before and after inference, reusing the existing figure/axes.

        Args:
        y : jax.Array or ArrayLike
            Ground truth, i.e. current frame
        prior_pred : jax.Array or ArrayLike
            Predicted sensory observation / output BEFORE inference
        posterior_pred : jax.Array or ArrayLike
            Predicted sensory observation / output AFTER inference
        inference_steps_made : int
            Number of inference steps taken to achieve the posterior
        frame_number : int
            Index of the current frame, used for the plot title and
            for constructing output filenames
        show_combined : bool, optional
            Whether to draw the figure to its canvas and pause briefly
            (interactive display). Default True
        save_combined : bool, optional
            Whether to save the combined 3-panel figure to disk.
            Saved under `{output_dir}/combined/frame_{frame_number:0N}.png`.
            Default True
        save_separate : bool, optional
            Whether to additionally save y, prior_pred, and posterior_pred
            as three separate, borderless images (useful for building an
            animation/video from a directory of frames later). Each is saved
            under its own subfolder:
            `{output_dir}/ground_truth/frame_{frame_number:0N}.png`,
            `{output_dir}/prior_pred/frame_{frame_number:0N}.png`,
            `{output_dir}/posterior_pred/frame_{frame_number:0N}.png`.
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
        y_img = np.asarray(y).reshape(self.output_shape)
        prior_img = np.asarray(prior_pred).reshape(self.output_shape)
        post_img = np.asarray(posterior_pred).reshape(self.output_shape)
        self.im_y.set_data(y_img)
        self.im_prior.set_data(prior_img)
        self.im_post.set_data(post_img)
        # Rescale color limits per-image (skip this if your data range
        # is already fixed/normalized — it's extra work per frame)
        self.im_y.autoscale()
        self.im_prior.autoscale()
        self.im_post.autoscale()
        self.axes[0].set_title(f"Frame {frame_number}: Ground Truth")
        self.axes[2].set_title(f"Posterior (Step {inference_steps_made})")
        if save_combined:
            combined_dir = os.path.join(output_dir, "combined")
            os.makedirs(combined_dir, exist_ok=True)
            self.fig.savefig(os.path.join(combined_dir, frame_str), dpi=150)
            # note: dropped bbox_inches='tight' — it forces a fresh layout
            # computation every save; tight_layout() once in __init__ instead
        if save_separate:
            arrays_by_name = {
                "ground_truth": y_img,
                "prior_pred": prior_img,
                "posterior_pred": post_img,
            }
            for name, arr in arrays_by_name.items():
                sub_dir = os.path.join(output_dir, name)
                os.makedirs(sub_dir, exist_ok=True)
                plt.imsave(os.path.join(sub_dir, frame_str), arr, cmap=self.cmap)
        if show_combined:
            self.fig.canvas.draw()
            plt.pause(0.001)

    def close(self):
        """
        Closes the underlying figure and releases its resources.

        Call this once, after training/plotting is complete, to free the
        figure that was kept alive for reuse across all `update()` calls.

        Returns:
        --------
        None
        """
        plt.close(self.fig)


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