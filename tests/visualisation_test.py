"""
Pytest suite for visualisation.py.

Covers: colormap generation helpers, filename sanitizing, grid-dimension
sizing, the low-level drawing helpers, plot_energies (validation,
layout selection, file outputs), VisualPredictionPlotter (state mutation
and file outputs), PredictionRecorder + replay_recordings (the buffered
recording / offline-replay pipeline), the natural sort key, and
compile_videos_from_frames (with imageio mocked out so no ffmpeg binary
is required to run the tests).

Run with:  pytest -v test_visualisation.py
"""
import os
import glob
from unittest.mock import MagicMock, call

import matplotlib
matplotlib.use("Agg")  # headless backend, must be set before pyplot is used

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pytest

import utils.visualisation as viz

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _legend_handle_count(legend):
    """matplotlib renamed Legend.legendHandles -> legend_handles in 3.7+."""
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = legend.legendHandles
    return len(handles)


def make_energies(num_iterations=3, num_layers=3, time_steps=5, varying_lengths=False):
    """Builds a list of (num_layers, time_steps) arrays, positive-valued so
    the log y-scale used by the energy traces doesn't choke on <= 0."""
    energies = []
    rng = np.random.default_rng(0)
    for it in range(num_iterations):
        t = time_steps - it if (varying_lengths and time_steps - it > 1) else time_steps
        energies.append(rng.uniform(0.1, 5.0, size=(num_layers, t)))
    return energies


def make_energies_with_valid_lengths(valid_lengths, num_layers=2, time_steps=6):
    """Builds a list of (num_layers, time_steps) arrays, each finite up to
    its own `valid_lengths[i]` and inf-padded from that point on -- the
    fixed-shape, inf-padded-tail pattern a real `settle_diffrax` trace has
    on early convergence. This is a different mechanism from
    `make_energies(varying_lengths=True)` above, which returns genuinely
    shorter arrays (real ragged data) rather than a fixed-size buffer with
    unfilled padding, so it doesn't exercise `_trace_valid_length` at all."""
    energies = []
    rng = np.random.default_rng(1)
    for length in valid_lengths:
        arr = rng.uniform(0.1, 5.0, size=(num_layers, time_steps))
        if length < time_steps:
            arr[:, length:] = np.inf
        energies.append(arr)
    return energies


@pytest.fixture(autouse=True)
def _close_all_figures_after_test():
    """Prevents figures leaking between tests from masking figure-count assertions."""
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# _qualitative_layer_colormaps / _procedural_layer_colormaps
# --------------------------------------------------------------------------

class TestLayerColormaps:

    @pytest.mark.parametrize("fn", [viz._qualitative_layer_colormaps, viz._procedural_layer_colormaps])
    def test_returns_one_colormap_per_layer(self, fn):
        cmaps = fn(5)
        assert len(cmaps) == 5
        assert all(isinstance(c, mcolors.Colormap) for c in cmaps)

    @pytest.mark.parametrize("fn", [viz._qualitative_layer_colormaps, viz._procedural_layer_colormaps])
    def test_zero_layers_returns_empty_list(self, fn):
        assert fn(0) == []

    def test_qualitative_recycles_anchors_past_list_length(self):
        n_anchors = len(viz._QUALITATIVE_ANCHORS)
        cmaps = viz._qualitative_layer_colormaps(n_anchors + 2)
        # index i and index i + n_anchors should be built from the same anchor hue
        assert np.allclose(cmaps[0](1.0), cmaps[n_anchors](1.0), atol=1e-6)
        assert np.allclose(cmaps[1](1.0), cmaps[n_anchors + 1](1.0), atol=1e-6)

    def test_qualitative_endpoint_is_the_anchor_color(self):
        anchor = viz._QUALITATIVE_ANCHORS[0]
        cmap = viz._qualitative_layer_colormaps(1)[0]
        assert np.allclose(cmap(1.0)[:3], mcolors.to_rgb(anchor), atol=1e-6)

    def test_procedural_distinct_hues_for_distinct_layers(self):
        cmaps = viz._procedural_layer_colormaps(4)
        colors_at_end = [c(1.0)[:3] for c in cmaps]
        # No two layers should land on an identical color for 4 evenly-spaced hues.
        for i in range(len(colors_at_end)):
            for j in range(i + 1, len(colors_at_end)):
                assert not np.allclose(colors_at_end[i], colors_at_end[j], atol=1e-3)


# --------------------------------------------------------------------------
# _sanitize_for_filename
# --------------------------------------------------------------------------

class TestSanitizeForFilename:

    def test_latex_markup_stripped(self):
        assert viz._sanitize_for_filename(r"$\ell_{3}$") == "ell_3"

    def test_free_text_with_spaces(self):
        assert viz._sanitize_for_filename("Hidden 2") == "hidden_2"

    def test_mixed_punctuation_collapsed_to_single_underscore(self):
        assert viz._sanitize_for_filename("Layer -- 1!!") == "layer_1"

    def test_empty_after_stripping_falls_back_to_layer(self):
        assert viz._sanitize_for_filename("$$$") == "layer"
        assert viz._sanitize_for_filename("") == "layer"

    def test_leading_trailing_underscores_stripped(self):
        assert viz._sanitize_for_filename("__hidden__") == "hidden"

    def test_already_clean_lowercase_unchanged(self):
        assert viz._sanitize_for_filename("layer3") == "layer3"


# --------------------------------------------------------------------------
# _grid_dims
# --------------------------------------------------------------------------

class TestGridDims:

    @pytest.mark.parametrize("num_layers,expected", [
        (1, (1, 1)),
        (4, (2, 2)),
        (5, (2, 3)),
        (10, (3, 4)),
    ])
    def test_default_square_ish_grid(self, num_layers, expected):
        assert viz._grid_dims(num_layers) == expected

    def test_max_cols_caps_columns(self):
        nrows, ncols = viz._grid_dims(10, max_cols=2)
        assert ncols == 2
        assert nrows == 5
        assert nrows * ncols >= 10

    def test_grid_always_fits_all_layers(self):
        for n in range(1, 30):
            nrows, ncols = viz._grid_dims(n)
            assert nrows * ncols >= n


# --------------------------------------------------------------------------
# _trace_valid_length
# --------------------------------------------------------------------------

class TestTraceValidLength:

    def test_fully_finite_trace_returns_full_length(self):
        energies_iter = np.random.default_rng(0).uniform(0.1, 5.0, size=(3, 6))
        assert viz._trace_valid_length(energies_iter) == 6

    def test_trailing_inf_suffix_returns_first_non_finite_index(self):
        energies_iter = np.random.default_rng(0).uniform(0.1, 5.0, size=(3, 6))
        energies_iter[:, 4:] = np.inf
        assert viz._trace_valid_length(energies_iter) == 4

    def test_trailing_nan_suffix_is_also_treated_as_invalid(self):
        """np.isfinite treats nan the same as inf -- settle_diffrax always
        pads with inf specifically, but the helper itself is agnostic."""
        energies_iter = np.random.default_rng(0).uniform(0.1, 5.0, size=(3, 6))
        energies_iter[:, 2:] = np.nan
        assert viz._trace_valid_length(energies_iter) == 2

    def test_a_single_non_finite_layer_invalidates_the_whole_column(self):
        """Only one layer needs to go non-finite at a given step for that
        step to be treated as invalid, matching settle_diffrax's own
        cross-layer inf padding (every layer's energy is undefined once the
        integrator has stopped, not just the layer that triggered it)."""
        energies_iter = np.random.default_rng(0).uniform(0.1, 5.0, size=(3, 6))
        energies_iter[1, 3:] = np.inf  # only layer index 1 goes non-finite
        assert viz._trace_valid_length(energies_iter) == 3

    def test_non_finite_from_the_first_step_clamps_to_one(self):
        """A pathological invalid-from-t=0 trace must still yield a
        plottable single point instead of an empty (zero-length) slice."""
        energies_iter = np.full((3, 6), np.inf)
        assert viz._trace_valid_length(energies_iter) == 1

    def test_single_time_step_trace_returns_one(self):
        energies_iter = np.random.default_rng(0).uniform(0.1, 5.0, size=(3, 1))
        assert viz._trace_valid_length(energies_iter) == 1


# --------------------------------------------------------------------------
# _draw_layer_traces
# --------------------------------------------------------------------------

class TestDrawLayerTraces:

    def test_draws_one_segment_per_iteration_and_sets_log_scale(self):
        energies = make_energies(num_iterations=4, num_layers=2, time_steps=6)
        trace_lengths = [e.shape[1] for e in energies]
        cmap = plt.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=0, vmax=3)
        fig, ax = plt.subplots()
        viz._draw_layer_traces(ax, layer_idx=0, energies=energies, trace_lengths=trace_lengths,
                                cmap=cmap, norm=norm, iter_positions=list(range(4)))
        assert len(ax.collections) == 1
        lc = ax.collections[0]
        assert len(lc.get_segments()) == 4
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_respects_varying_trace_lengths(self):
        energies = make_energies(num_iterations=3, num_layers=2, time_steps=6, varying_lengths=True)
        trace_lengths = [e.shape[1] for e in energies]
        fig, ax = plt.subplots()
        viz._draw_layer_traces(ax, 0, energies, trace_lengths, plt.get_cmap("viridis"),
                                mcolors.Normalize(0, 2), list(range(3)))
        lc = ax.collections[0]
        seg_lengths = [len(seg) for seg in lc.get_segments()]
        assert seg_lengths == trace_lengths
        plt.close(fig)


# --------------------------------------------------------------------------
# _draw_overlay / _add_overlay_colorbar
# --------------------------------------------------------------------------

class TestDrawOverlay:

    def test_small_layer_count_uses_inline_legend(self):
        num_layers = 3
        energies = make_energies(num_iterations=2, num_layers=num_layers)
        trace_lengths = [e.shape[1] for e in energies]
        colormaps = viz._qualitative_layer_colormaps(num_layers)
        fig, ax = plt.subplots()
        viz._draw_overlay(ax, num_layers, energies, trace_lengths, colormaps,
                           mcolors.Normalize(0, 1), [0, 1], ["a", "b", "c"], "VFE")
        legend = ax.get_legend()
        assert legend is not None
        assert _legend_handle_count(legend) == num_layers
        assert ax.get_xlabel() == "Inference iterations"
        assert ax.get_ylabel() == "VFE"
        plt.close(fig)

    def test_large_layer_count_uses_outside_multicolumn_legend(self):
        num_layers = 20
        energies = make_energies(num_iterations=2, num_layers=num_layers)
        trace_lengths = [e.shape[1] for e in energies]
        colormaps = viz._qualitative_layer_colormaps(num_layers)
        labels = [f"l{i}" for i in range(num_layers)]
        fig, ax = plt.subplots()
        viz._draw_overlay(ax, num_layers, energies, trace_lengths, colormaps,
                           mcolors.Normalize(0, 1), [0, 1], labels, "VFE")
        legend = ax.get_legend()
        assert legend is not None
        assert _legend_handle_count(legend) == num_layers
        plt.close(fig)

    def test_colorbar_vertical_for_small_layer_count(self):
        fig, ax = plt.subplots()
        cbar = viz._add_overlay_colorbar(fig, ax, mcolors.Normalize(0, 1), num_layers=5)
        assert cbar.orientation == "vertical"
        plt.close(fig)

    def test_colorbar_horizontal_for_large_layer_count(self):
        fig, ax = plt.subplots()
        cbar = viz._add_overlay_colorbar(fig, ax, mcolors.Normalize(0, 1), num_layers=20)
        assert cbar.orientation == "horizontal"
        plt.close(fig)


# --------------------------------------------------------------------------
# plot_energies
# --------------------------------------------------------------------------

class TestPlotTrainEnergies:

    def test_invalid_layout_raises(self):
        with pytest.raises(ValueError, match="layout"):
            viz.plot_energies(make_energies(), layout="diagonal", display=False)

    def test_invalid_color_scheme_raises(self):
        with pytest.raises(ValueError, match="color_scheme"):
            viz.plot_energies(make_energies(), color_scheme="rainbow", display=False)

    def test_layer_labels_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="layer_labels"):
            viz.plot_energies(make_energies(num_layers=3), layer_labels=["a", "b"], display=False)

    def test_colormaps_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="colormaps"):
            viz.plot_energies(make_energies(num_layers=3), colormaps=["viridis", "plasma"],
                                     display=False)

    def test_runs_without_error_default_args(self):
        viz.plot_energies(make_energies(), display=False)

    def test_default_labels_are_generic_ell(self, tmp_path):
        # Indirect check: individual export file names embed the sanitized label.
        energies = make_energies(num_layers=2)
        viz.plot_energies(energies, save_individual=True, output_dir=str(tmp_path), display=False)
        files = sorted(os.listdir(tmp_path / "vfe_layerwise"))
        assert any("ell_1" in f for f in files)
        assert any("ell_2" in f for f in files)

    def test_model_layer_labels_used_when_available(self, tmp_path):
        class FakeModel:
            config = object()

            @classmethod
            def layer_labels(cls, config):
                return ["alpha", "beta"]

        viz.plot_energies(make_energies(num_layers=2), model=FakeModel(),
                                 save_individual=True, output_dir=str(tmp_path), display=False)
        files = sorted(os.listdir(tmp_path / "vfe_layerwise"))
        assert any("alpha" in f for f in files)
        assert any("beta" in f for f in files)

    def test_model_layer_labels_not_implemented_falls_back(self, tmp_path):
        class FakeModel:
            config = object()

            @classmethod
            def layer_labels(cls, config):
                raise NotImplementedError

        viz.plot_energies(make_energies(num_layers=2), model=FakeModel(),
                                 save_individual=True, output_dir=str(tmp_path), display=False)
        files = sorted(os.listdir(tmp_path / "vfe_layerwise"))
        assert any("ell_1" in f for f in files)

    def test_explicit_layer_labels_take_precedence_over_model(self, tmp_path):
        class FakeModel:
            config = object()

            @classmethod
            def layer_labels(cls, config):
                return ["should_not_be_used_a", "should_not_be_used_b"]

        viz.plot_energies(make_energies(num_layers=2), model=FakeModel(),
                                 layer_labels=["explicit_a", "explicit_b"],
                                 save_individual=True, output_dir=str(tmp_path), display=False)
        files = sorted(os.listdir(tmp_path / "vfe_layerwise"))
        assert any("explicit_a" in f for f in files)
        assert not any("should_not_be_used" in f for f in files)

    def test_save_plot_writes_expected_file(self, tmp_path):
        viz.plot_energies(make_energies(), save_plot=True, output_dir=str(tmp_path), display=False)
        assert (tmp_path / "train_energies.png").exists()

    def test_save_overlay_writes_expected_file_only_with_separate_layers(self, tmp_path):
        # save_overlay is documented as only taking effect when separate_layers=True
        viz.plot_energies(make_energies(), save_overlay=True, separate_layers=True,
                                 output_dir=str(tmp_path), display=False)
        assert (tmp_path / "train_energies_overlay.png").exists()

    def test_save_overlay_ignored_without_separate_layers(self, tmp_path):
        viz.plot_energies(make_energies(), save_overlay=True, separate_layers=False,
                                 output_dir=str(tmp_path), display=False)
        assert not (tmp_path / "train_energies_overlay.png").exists()

    def test_save_individual_writes_one_file_per_layer(self, tmp_path):
        num_layers = 4
        viz.plot_energies(make_energies(num_layers=num_layers), save_individual=True,
                                 output_dir=str(tmp_path), display=False)
        files = os.listdir(tmp_path / "vfe_layerwise")
        assert len(files) == num_layers

    def test_no_files_written_when_no_save_flag_set(self, tmp_path):
        viz.plot_energies(make_energies(), output_dir=str(tmp_path), display=False)
        assert not tmp_path.exists() or os.listdir(tmp_path) == []

    def test_all_figures_closed_after_call(self, tmp_path):
        before = len(plt.get_fignums())
        viz.plot_energies(make_energies(num_layers=3), save_overlay=True,
                                 save_individual=True, separate_layers=True, display=False,
                                 output_dir=str(tmp_path))
        after = len(plt.get_fignums())
        assert after == before

    def test_layout_auto_uses_grid_dims_only_above_threshold(self, monkeypatch):
        real_grid_dims = viz._grid_dims
        spy = MagicMock(side_effect=real_grid_dims)
        monkeypatch.setattr(viz, "_grid_dims", spy)

        # below threshold -> column layout, _grid_dims not consulted
        viz.plot_energies(make_energies(num_layers=3), separate_layers=True,
                                 layout="auto", grid_threshold=6, display=False)
        assert spy.call_count == 0

        # above threshold -> grid layout, _grid_dims consulted
        viz.plot_energies(make_energies(num_layers=8), separate_layers=True,
                                 layout="auto", grid_threshold=6, display=False)
        assert spy.call_count == 1

    def test_explicit_grid_layout_used_even_for_few_layers(self, monkeypatch):
        spy = MagicMock(side_effect=viz._grid_dims)
        monkeypatch.setattr(viz, "_grid_dims", spy)
        viz.plot_energies(make_energies(num_layers=2), separate_layers=True,
                                 layout="grid", display=False)
        assert spy.call_count == 1

    def test_explicit_column_layout_used_even_for_many_layers(self, monkeypatch):
        spy = MagicMock(side_effect=viz._grid_dims)
        monkeypatch.setattr(viz, "_grid_dims", spy)
        viz.plot_energies(make_energies(num_layers=12), separate_layers=True,
                                 layout="column", display=False)
        assert spy.call_count == 0

    def test_t_max_does_not_crash_with_varying_trace_lengths(self):
        energies = make_energies(num_iterations=3, num_layers=2, time_steps=10, varying_lengths=True)
        viz.plot_energies(energies, t_max=3, display=False)

    def test_explicit_colormaps_accepted_as_strings_and_objects(self):
        cmaps = ["viridis", plt.get_cmap("plasma")]
        viz.plot_energies(make_energies(num_layers=2), colormaps=cmaps, display=False)

    # ---- Diffrax-style (inf-padded) traces -----------------------------------
    # settle_diffrax leaves the unused tail of its fixed-size save buffer as
    # inf rather than back-filling it (see settle_diffrax's docstring in
    # tpch.py), so a training loop that records its energy_trace straight
    # into `energies` hands plot_energies fixed-shape arrays with a
    # trailing inf run on any iteration that converged early. The tests
    # below exercise exactly that shape, via _trace_valid_length's
    # integration into plot_energies (rather than just the unit tests
    # on _trace_valid_length itself, above).

    def test_inf_padded_diffrax_style_traces_are_trimmed_before_drawing(self, monkeypatch):
        """End-to-end: plot_energies must run each iteration's trace
        through _trace_valid_length before handing trace_lengths to the
        drawing helpers, so an early-converged settle_diffrax iteration
        (fixed shape, inf-padded tail) is trimmed exactly like a genuinely
        shorter settle_scan trace would be."""
        spy = MagicMock(side_effect=viz._draw_layer_traces)
        monkeypatch.setattr(viz, "_draw_layer_traces", spy)

        time_steps = 6
        valid_lengths = [time_steps, 4, time_steps]
        energies = make_energies_with_valid_lengths(valid_lengths, num_layers=2, time_steps=time_steps)

        viz.plot_energies(energies, separate_layers=True, display=False)

        # _draw_layer_traces is called once per layer, but trace_lengths (arg
        # index 3) is the same list every time -- check the first call only.
        trace_lengths = spy.call_args_list[0].args[3]
        assert trace_lengths == valid_lengths

    def test_inf_padded_traces_compose_correctly_with_t_max(self, monkeypatch):
        """t_max must cap the already-trimmed valid length, not the raw
        (padded) buffer size -- min(valid_length, t_max) per iteration."""
        spy = MagicMock(side_effect=viz._draw_layer_traces)
        monkeypatch.setattr(viz, "_draw_layer_traces", spy)

        time_steps = 10
        valid_lengths = [time_steps, 4, time_steps]
        energies = make_energies_with_valid_lengths(valid_lengths, num_layers=2, time_steps=time_steps)

        viz.plot_energies(energies, t_max=6, separate_layers=True, display=False)

        trace_lengths = spy.call_args_list[0].args[3]
        assert trace_lengths == [6, 4, 6]

    def test_fully_finite_traces_are_unaffected_by_trimming(self, monkeypatch):
        """A settle_scan-style fully-finite trace's valid length is just its
        own full length -- confirms the new trimming logic is a strict
        generalisation, not a behaviour change, for existing callers."""
        spy = MagicMock(side_effect=viz._draw_layer_traces)
        monkeypatch.setattr(viz, "_draw_layer_traces", spy)

        energies = make_energies(num_iterations=3, num_layers=2, time_steps=7)
        viz.plot_energies(energies, separate_layers=True, display=False)

        trace_lengths = spy.call_args_list[0].args[3]
        assert trace_lengths == [7, 7, 7]

    def test_diffrax_style_traces_do_not_crash_in_overlay_or_grid_layout(self):
        time_steps = 8
        valid_lengths = [time_steps, 3, 5, time_steps]
        energies = make_energies_with_valid_lengths(valid_lengths, num_layers=4, time_steps=time_steps)
        viz.plot_energies(energies, display=False)  # overlay (default)
        viz.plot_energies(energies, separate_layers=True, layout="grid", display=False)

    def test_diffrax_style_traces_do_not_crash_when_saved(self, tmp_path):
        time_steps = 6
        valid_lengths = [time_steps, 2, time_steps]
        energies = make_energies_with_valid_lengths(valid_lengths, num_layers=2, time_steps=time_steps)
        viz.plot_energies(
            energies, save_plot=True, save_overlay=True, save_individual=True,
            separate_layers=True, display=False, output_dir=str(tmp_path),
        )
        assert os.path.isfile(os.path.join(str(tmp_path), "train_energies.png"))

    # ---- x_axis_label ----------------------------------------------------------

    def test_default_x_axis_label_is_inference_iterations_on_overlay(self, monkeypatch):
        spy = MagicMock(side_effect=viz._draw_overlay)
        monkeypatch.setattr(viz, "_draw_overlay", spy)

        viz.plot_energies(make_energies(), display=False)

        ax = spy.call_args_list[0].args[0]
        assert ax.get_xlabel() == "Inference iterations"

    def test_custom_x_axis_label_applied_to_overlay(self, monkeypatch):
        spy = MagicMock(side_effect=viz._draw_overlay)
        monkeypatch.setattr(viz, "_draw_overlay", spy)

        viz.plot_energies(make_energies(), x_axis_label="Inference time (t)", display=False)

        ax = spy.call_args_list[0].args[0]
        assert ax.get_xlabel() == "Inference time (t)"

    def test_custom_x_axis_label_applied_to_grid_supxlabel(self, monkeypatch):
        created = []
        real_subplots = plt.subplots

        def capturing_subplots(*args, **kwargs):
            fig, axes = real_subplots(*args, **kwargs)
            created.append(fig)
            return fig, axes

        monkeypatch.setattr(plt, "subplots", capturing_subplots)

        viz.plot_energies(
            make_energies(num_layers=2), separate_layers=True,
            x_axis_label="Inference time (t)", display=False,
        )

        assert created[0].get_supxlabel() == "Inference time (t)"

    def test_custom_x_axis_label_applied_to_saved_individual_plots(self, monkeypatch, tmp_path):
        spy = MagicMock(side_effect=viz._draw_layer_traces)
        monkeypatch.setattr(viz, "_draw_layer_traces", spy)

        viz.plot_energies(
            make_energies(num_layers=2), save_individual=True, x_axis_label="Inference time (t)",
            display=False, output_dir=str(tmp_path),
        )

        # separate_layers is left False (overlay main plot), so the only
        # calls to _draw_layer_traces at all come from the save_individual
        # export pass -- any of them will do.
        ind_ax = spy.call_args_list[-1].args[0]
        assert ind_ax.get_xlabel() == "Inference time (t)"


# --------------------------------------------------------------------------
# VisualPredictionPlotter
# --------------------------------------------------------------------------

class TestVisualPredictionPlotter:

    def test_init_creates_three_blank_axes(self):
        plotter = viz.VisualPredictionPlotter(output_shape=(4, 5))
        assert len(plotter.axes) == 3
        assert plotter.im_y.get_array().shape == (4, 5)
        plotter.close()

    def test_update_reshapes_and_sets_image_data(self):
        plotter = viz.VisualPredictionPlotter(output_shape=(2, 3))
        y = np.arange(6, dtype=float)
        prior = np.arange(6, 12, dtype=float)
        post = np.arange(12, 18, dtype=float)
        plotter.update(y=y, prior_pred=prior, posterior_pred=post,
                        inference_steps_made=7, frame_number=0, save_combined=False)
        assert np.array_equal(np.asarray(plotter.im_y.get_array()), y.reshape(2, 3))
        assert np.array_equal(np.asarray(plotter.im_prior.get_array()), prior.reshape(2, 3))
        assert np.array_equal(np.asarray(plotter.im_post.get_array()), post.reshape(2, 3))
        plotter.close()

    def test_update_accepts_jax_arrays(self):
        plotter = viz.VisualPredictionPlotter(output_shape=(2, 2))
        y = jnp.arange(4, dtype=jnp.float32)
        plotter.update(y=y, prior_pred=y, posterior_pred=y,
                        inference_steps_made=1, frame_number=0, save_combined=False)
        assert np.array_equal(np.asarray(plotter.im_y.get_array()), np.asarray(y).reshape(2, 2))
        plotter.close()

    def test_titles_updated(self):
        plotter = viz.VisualPredictionPlotter(output_shape=(2, 2))
        y = np.zeros(4)
        plotter.update(y=y, prior_pred=y, posterior_pred=y,
                        inference_steps_made=9, frame_number=42, save_combined=False)
        assert plotter.axes[0].get_title() == "Frame 42: Ground Truth"
        assert plotter.axes[2].get_title() == "Posterior (Step 9)"
        plotter.close()

    def test_save_combined_writes_expected_file(self, tmp_path):
        plotter = viz.VisualPredictionPlotter(output_shape=(2, 2))
        y = np.zeros(4)
        plotter.update(y=y, prior_pred=y, posterior_pred=y, inference_steps_made=0,
                        frame_number=3, save_combined=True, output_dir=str(tmp_path),
                        total_frames=1000)
        # total_frames=1000 -> pad width len("999") == 3
        assert (tmp_path / "combined" / "frame_003.png").exists()
        plotter.close()

    def test_save_combined_false_creates_no_directory(self, tmp_path):
        plotter = viz.VisualPredictionPlotter(output_shape=(2, 2))
        y = np.zeros(4)
        plotter.update(y=y, prior_pred=y, posterior_pred=y, inference_steps_made=0,
                        frame_number=0, save_combined=False, output_dir=str(tmp_path))
        assert not (tmp_path / "combined").exists()
        plotter.close()

    def test_save_separate_writes_three_files(self, tmp_path):
        plotter = viz.VisualPredictionPlotter(output_shape=(2, 2))
        y = np.zeros(4)
        plotter.update(y=y, prior_pred=y, posterior_pred=y, inference_steps_made=0,
                        frame_number=0, save_combined=False, save_separate=True,
                        output_dir=str(tmp_path), total_frames=10)
        assert (tmp_path / "ground_truth" / "frame_0.png").exists()
        assert (tmp_path / "prior_pred" / "frame_0.png").exists()
        assert (tmp_path / "posterior_pred" / "frame_0.png").exists()
        plotter.close()

    def test_frame_number_zero_padding_scales_with_total_frames(self, tmp_path):
        plotter = viz.VisualPredictionPlotter(output_shape=(2, 2))
        y = np.zeros(4)
        plotter.update(y=y, prior_pred=y, posterior_pred=y, inference_steps_made=0,
                        frame_number=7, save_combined=True, output_dir=str(tmp_path),
                        total_frames=1_000_000)
        assert (tmp_path / "combined" / "frame_000007.png").exists()
        plotter.close()

    def test_show_combined_does_not_raise(self):
        plotter = viz.VisualPredictionPlotter(output_shape=(2, 2))
        y = np.zeros(4)
        plotter.update(y=y, prior_pred=y, posterior_pred=y, inference_steps_made=0,
                        frame_number=0, show_combined=True, save_combined=False)
        plotter.close()

    def test_close_closes_figure(self):
        plotter = viz.VisualPredictionPlotter(output_shape=(2, 2))
        fignum = plotter.fig.number
        assert fignum in plt.get_fignums()
        plotter.close()
        assert fignum not in plt.get_fignums()


# --------------------------------------------------------------------------
# PredictionRecorder
# --------------------------------------------------------------------------

class TestPredictionRecorder:

    def test_append_does_not_touch_disk(self, tmp_path):
        recorder = viz.PredictionRecorder(output_dir=str(tmp_path))
        recorder.append(y=jnp.zeros(4), prior_pred=jnp.zeros(4), posterior_pred=jnp.zeros(4),
                         inference_steps_made=1, frame_number=0)
        assert glob.glob(str(tmp_path / "chunk_*.npz")) == []

    def test_flush_with_empty_buffer_is_noop(self, tmp_path):
        recorder = viz.PredictionRecorder(output_dir=str(tmp_path))
        recorder.flush()
        assert glob.glob(str(tmp_path / "chunk_*.npz")) == []

    def test_flush_writes_expected_chunk_contents(self, tmp_path):
        recorder = viz.PredictionRecorder(output_dir=str(tmp_path))
        for i in range(3):
            recorder.append(
                y=jnp.full((2, 2), i, dtype=jnp.float32),
                prior_pred=jnp.full((2, 2), i + 10, dtype=jnp.float32),
                posterior_pred=jnp.full((2, 2), i + 20, dtype=jnp.float32),
                inference_steps_made=5 + i,
                frame_number=i,
            )
        recorder.flush()

        chunk_files = glob.glob(str(tmp_path / "chunk_*.npz"))
        assert len(chunk_files) == 1
        assert os.path.basename(chunk_files[0]) == "chunk_0000000_0000002.npz"

        data = np.load(chunk_files[0])
        assert data["y"].shape == (3, 2, 2)
        assert np.array_equal(data["y"][1], np.full((2, 2), 1.0))
        assert np.array_equal(data["prior_pred"][2], np.full((2, 2), 12.0))
        assert np.array_equal(data["posterior_pred"][0], np.full((2, 2), 20.0))
        assert list(data["frame_numbers"]) == [0, 1, 2]
        assert list(data["inference_steps"]) == [5, 6, 7]

    def test_flush_resets_buffer_for_next_chunk(self, tmp_path):
        recorder = viz.PredictionRecorder(output_dir=str(tmp_path))
        recorder.append(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), 1, frame_number=0)
        recorder.flush()
        recorder.append(jnp.ones(2), jnp.ones(2), jnp.ones(2), 1, frame_number=1)
        recorder.flush()

        chunk_files = sorted(glob.glob(str(tmp_path / "chunk_*.npz")))
        assert len(chunk_files) == 2
        second = np.load(chunk_files[1])
        assert list(second["frame_numbers"]) == [1]

    def test_chunk_naming_uses_append_order_not_sorted_min_max(self, tmp_path):
        # documents actual behavior: lo/hi come from list[0] / list[-1], i.e.
        # append order, not sorted order -- if frame numbers are ever appended
        # out of order the filename will reflect that, not the true min/max.
        recorder = viz.PredictionRecorder(output_dir=str(tmp_path))
        recorder.append(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), 1, frame_number=5)
        recorder.append(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), 1, frame_number=2)
        recorder.flush()
        chunk_files = glob.glob(str(tmp_path / "chunk_*.npz"))
        assert os.path.basename(chunk_files[0]) == "chunk_0000005_0000002.npz"

    def test_output_dir_created_on_init(self, tmp_path):
        target = tmp_path / "nested" / "dir"
        assert not target.exists()
        viz.PredictionRecorder(output_dir=str(target))
        assert target.exists()

    def test_append_block_writes_all_frames_with_scalar_inference_steps(self, tmp_path):
        recorder = viz.PredictionRecorder(output_dir=str(tmp_path))

        y = jnp.stack([jnp.full((2, 2), i, dtype=jnp.float32) for i in range(3)])
        prior = y + 10
        posterior = y + 20

        recorder.append_block(
            y=y,
            prior_pred=prior,
            posterior_pred=posterior,
            inference_steps_made=7,
            frame_numbers=[10, 11, 12],
        )
        recorder.flush()

        chunk_files = glob.glob(str(tmp_path / "chunk_*.npz"))
        assert len(chunk_files) == 1
        assert os.path.basename(chunk_files[0]) == "chunk_0000010_0000012.npz"

        data = np.load(chunk_files[0])
        assert data["y"].shape == (3, 2, 2)
        assert data["prior_pred"].shape == (3, 2, 2)
        assert data["posterior_pred"].shape == (3, 2, 2)
        assert np.array_equal(data["y"][1], np.full((2, 2), 1.0))
        assert np.array_equal(data["prior_pred"][2], np.full((2, 2), 12.0))
        assert np.array_equal(data["posterior_pred"][0], np.full((2, 2), 20.0))
        assert list(data["frame_numbers"]) == [10, 11, 12]
        assert list(data["inference_steps"]) == [7, 7, 7]

    def test_multiple_blocks_are_concatenated_along_frame_axis(self, tmp_path):
        recorder = viz.PredictionRecorder(output_dir=str(tmp_path))

        block1 = jnp.stack([jnp.full((2,), i, dtype=jnp.float32) for i in (0, 1)])
        block2 = jnp.stack([jnp.full((2,), i, dtype=jnp.float32) for i in (2, 3, 4)])

        recorder.append_block(
            y=block1,
            prior_pred=block1 + 10,
            posterior_pred=block1 + 20,
            inference_steps_made=5,
            frame_numbers=[0, 1],
        )
        recorder.append_block(
            y=block2,
            prior_pred=block2 + 10,
            posterior_pred=block2 + 20,
            inference_steps_made=6,
            frame_numbers=[2, 3, 4],
        )
        recorder.flush()

        chunk_files = glob.glob(str(tmp_path / "chunk_*.npz"))
        assert len(chunk_files) == 1

        data = np.load(chunk_files[0])
        assert data["y"].shape == (5, 2)
        assert np.array_equal(data["y"][:, 0], np.arange(5, dtype=np.float32))
        assert np.array_equal(data["prior_pred"][:, 0], np.arange(10, 15, dtype=np.float32))
        assert np.array_equal(data["posterior_pred"][:, 0], np.arange(20, 25, dtype=np.float32))
        assert list(data["frame_numbers"]) == [0, 1, 2, 3, 4]
        assert list(data["inference_steps"]) == [5, 5, 6, 6, 6]

    def test_single_frame_append_matches_length_one_block(self, tmp_path):
        append_dir = tmp_path / "append"
        block_dir = tmp_path / "block"

        recorder_append = viz.PredictionRecorder(output_dir=str(append_dir))
        recorder_append.append(
            y=jnp.array([1.0, 2.0]),
            prior_pred=jnp.array([3.0, 4.0]),
            posterior_pred=jnp.array([5.0, 6.0]),
            inference_steps_made=9,
            frame_number=7,
        )
        recorder_append.flush()

        recorder_block = viz.PredictionRecorder(output_dir=str(block_dir))
        recorder_block.append_block(
            y=jnp.array([[1.0, 2.0]]),
            prior_pred=jnp.array([[3.0, 4.0]]),
            posterior_pred=jnp.array([[5.0, 6.0]]),
            inference_steps_made=9,
            frame_numbers=[7],
        )
        recorder_block.flush()

        append_data = np.load(glob.glob(str(append_dir / "chunk_*.npz"))[0])
        block_data = np.load(glob.glob(str(block_dir / "chunk_*.npz"))[0])

        for key in ["y", "prior_pred", "posterior_pred", "frame_numbers", "inference_steps"]:
            assert np.array_equal(append_data[key], block_data[key])

    def test_append_and_append_block_can_be_mixed(self, tmp_path):
        recorder = viz.PredictionRecorder(output_dir=str(tmp_path))

        recorder.append(
            y=jnp.array([0.0, 0.0]),
            prior_pred=jnp.array([10.0, 10.0]),
            posterior_pred=jnp.array([20.0, 20.0]),
            inference_steps_made=1,
            frame_number=0,
        )

        block = jnp.stack([
            jnp.full((2,), 1.0, dtype=jnp.float32),
            jnp.full((2,), 2.0, dtype=jnp.float32),
            jnp.full((2,), 3.0, dtype=jnp.float32),
        ])
        recorder.append_block(
            y=block,
            prior_pred=block + 10,
            posterior_pred=block + 20,
            inference_steps_made=2,
            frame_numbers=[1, 2, 3],
        )

        recorder.append(
            y=jnp.array([4.0, 4.0]),
            prior_pred=jnp.array([14.0, 14.0]),
            posterior_pred=jnp.array([24.0, 24.0]),
            inference_steps_made=3,
            frame_number=4,
        )

        recorder.flush()

        chunk_files = glob.glob(str(tmp_path / "chunk_*.npz"))
        assert len(chunk_files) == 1

        data = np.load(chunk_files[0])
        assert data["y"].shape == (5, 2)
        assert np.array_equal(data["y"][:, 0], np.arange(5, dtype=np.float32))
        assert list(data["frame_numbers"]) == [0, 1, 2, 3, 4]
        assert list(data["inference_steps"]) == [1, 2, 2, 2, 3]

    def test_append_block_rejects_mismatched_frame_counts(self, tmp_path):
        recorder = viz.PredictionRecorder(output_dir=str(tmp_path))

        y = jnp.zeros((3, 2))
        prior = jnp.zeros((3, 2))
        posterior = jnp.zeros((3, 2))

        with pytest.raises(ValueError, match="frame_numbers"):
            recorder.append_block(
                y=y,
                prior_pred=prior,
                posterior_pred=posterior,
                inference_steps_made=1,
                frame_numbers=[0, 1],
            )

        with pytest.raises(ValueError, match="frame_numbers"):
            recorder.append_block(
                y=y,
                prior_pred=prior[:2],
                posterior_pred=posterior,
                inference_steps_made=1,
                frame_numbers=[0, 1, 2],
            )

        with pytest.raises(ValueError, match="frame_numbers"):
            recorder.append_block(
                y=y,
                prior_pred=prior,
                posterior_pred=posterior[:2],
                inference_steps_made=1,
                frame_numbers=[0, 1, 2],
            )

    def test_append_block_rejects_mismatched_inference_steps(self, tmp_path):
        recorder = viz.PredictionRecorder(output_dir=str(tmp_path))
        y = jnp.zeros((3, 2))

        with pytest.raises(ValueError, match="inference_steps_made"):
            recorder.append_block(
                y=y,
                prior_pred=y,
                posterior_pred=y,
                inference_steps_made=[1, 2],
                frame_numbers=[0, 1, 2],
            )


# --------------------------------------------------------------------------
# replay_recordings
# --------------------------------------------------------------------------

class TestReplayRecordings:

    def _make_chunk(self, recordings_dir, frame_numbers, shape=(2, 2)):
        recorder = viz.PredictionRecorder(output_dir=recordings_dir)
        for fn in frame_numbers:
            recorder.append(
                y=jnp.full(shape, fn, dtype=jnp.float32),
                prior_pred=jnp.full(shape, fn, dtype=jnp.float32),
                posterior_pred=jnp.full(shape, fn, dtype=jnp.float32),
                inference_steps_made=1,
                frame_number=fn,
            )
        recorder.flush()

    def test_no_chunks_found_prints_message_and_returns(self, tmp_path, capsys):
        viz.replay_recordings(str(tmp_path), output_shape=(2, 2),
                               output_dir=str(tmp_path / "out"))
        captured = capsys.readouterr()
        assert "No chunks found" in captured.out
        assert not (tmp_path / "out").exists()

    def test_replays_all_frames_across_chunks_in_order(self, tmp_path):
        recordings_dir = str(tmp_path / "raw")
        out_dir = str(tmp_path / "out")
        self._make_chunk(recordings_dir, [0, 1])
        self._make_chunk(recordings_dir, [2, 3, 4])

        viz.replay_recordings(recordings_dir, output_shape=(2, 2), output_dir=out_dir,
                               total_frames=5)

        combined_files = sorted(os.listdir(os.path.join(out_dir, "combined")))
        assert combined_files == [f"frame_{i}.png" for i in range(5)]

    def test_total_frames_inferred_from_last_chunk_when_not_given(self, tmp_path):
        recordings_dir = str(tmp_path / "raw")
        out_dir = str(tmp_path / "out")
        self._make_chunk(recordings_dir, list(range(12)))  # -> total_frames inferred as 12

        viz.replay_recordings(recordings_dir, output_shape=(2, 2), output_dir=out_dir)

        # total_frames=12 -> pad width len("11") == 2
        assert os.path.exists(os.path.join(out_dir, "combined", "frame_00.png"))
        assert os.path.exists(os.path.join(out_dir, "combined", "frame_11.png"))

    def test_extra_plotter_kwargs_are_forwarded(self, tmp_path):
        recordings_dir = str(tmp_path / "raw")
        out_dir = str(tmp_path / "out")
        self._make_chunk(recordings_dir, [0])

        viz.replay_recordings(recordings_dir, output_shape=(2, 2), output_dir=out_dir,
                               total_frames=1, save_combined=False, save_separate=True)

        assert not os.path.exists(os.path.join(out_dir, "combined"))
        assert os.path.exists(os.path.join(out_dir, "ground_truth", "frame_0.png"))

    def test_plotter_closed_even_if_update_raises(self, tmp_path, monkeypatch):
        recordings_dir = str(tmp_path / "raw")
        self._make_chunk(recordings_dir, [0, 1])

        close_calls = []
        real_close = viz.VisualPredictionPlotter.close

        def spying_close(self):
            close_calls.append(True)
            return real_close(self)

        def raising_update(self, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(viz.VisualPredictionPlotter, "close", spying_close)
        monkeypatch.setattr(viz.VisualPredictionPlotter, "update", raising_update)

        with pytest.raises(RuntimeError, match="boom"):
            viz.replay_recordings(recordings_dir, output_shape=(2, 2),
                                   output_dir=str(tmp_path / "out"), total_frames=2)

        assert close_calls == [True]


# --------------------------------------------------------------------------
# _natural_key
# --------------------------------------------------------------------------

class TestNaturalKey:

    def test_orders_numerically_not_lexicographically(self):
        paths = ["frame_10.png", "frame_2.png", "frame_1.png"]
        assert sorted(paths, key=viz._natural_key) == ["frame_1.png", "frame_2.png", "frame_10.png"]

    def test_mixed_zero_padding_widths_sort_correctly(self):
        paths = ["frame_002.png", "frame_10.png", "frame_1.png"]
        assert sorted(paths, key=viz._natural_key) == ["frame_1.png", "frame_002.png", "frame_10.png"]

    def test_ignores_directory_component(self):
        paths = ["/a/b/frame_10.png", "/z/y/frame_2.png"]
        assert sorted(paths, key=viz._natural_key)[0].endswith("frame_2.png")


# --------------------------------------------------------------------------
# compile_videos_from_frames (imageio mocked out -- no ffmpeg needed)
# --------------------------------------------------------------------------

class FakeWriter:
    """Records append_data calls; usable as `with imageio.get_writer(...) as w`."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.appended = []
        FakeWriter.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def append_data(self, data):
        self.appended.append(data)


class TestCompileVideosFromFrames:

    def setup_method(self):
        FakeWriter.instances = []

    def _touch(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a").close()

    def test_autodetects_subdirs_with_frames_only(self, tmp_path, monkeypatch):
        base = str(tmp_path)
        self._touch(os.path.join(base, "ground_truth", "frame_0.png"))
        self._touch(os.path.join(base, "prior_pred", "frame_0.png"))
        os.makedirs(os.path.join(base, "empty_dir"))  # no frames -- should be skipped

        monkeypatch.setattr(viz.imageio, "get_writer", FakeWriter)
        monkeypatch.setattr(viz.imageio, "imread", lambda p: np.zeros((2, 2)))

        viz.compile_videos_from_frames(output_dir=base)

        written = sorted(w.args[0] for w in FakeWriter.instances)
        assert written == [
            os.path.join(base, "ground_truth.mp4"),
            os.path.join(base, "prior_pred.mp4"),
        ]

    def test_explicit_subdirs_with_no_frames_are_skipped_and_reported(self, tmp_path, monkeypatch, capsys):
        base = str(tmp_path)
        os.makedirs(os.path.join(base, "empty_dir"))

        monkeypatch.setattr(viz.imageio, "get_writer", FakeWriter)
        monkeypatch.setattr(viz.imageio, "imread", lambda p: np.zeros((2, 2)))

        viz.compile_videos_from_frames(output_dir=base, subdirs=["empty_dir"])

        assert FakeWriter.instances == []
        assert "Skipping 'empty_dir'" in capsys.readouterr().out

    def test_frames_appended_in_natural_sorted_order(self, tmp_path, monkeypatch):
        base = str(tmp_path)
        sub = os.path.join(base, "ground_truth")
        for n in [10, 2, 1]:
            self._touch(os.path.join(sub, f"frame_{n}.png"))

        monkeypatch.setattr(viz.imageio, "get_writer", FakeWriter)
        # tag each "frame" with the number parsed from its filename so we can check order
        monkeypatch.setattr(
            viz.imageio, "imread",
            lambda p: np.full((1, 1), int(os.path.basename(p).split("_")[1].split(".")[0])),
        )

        viz.compile_videos_from_frames(output_dir=base, subdirs=["ground_truth"])

        appended_values = [int(a[0, 0]) for a in FakeWriter.instances[0].appended]
        assert appended_values == [1, 2, 10]

    def test_fps_and_quality_forwarded_to_writer(self, tmp_path, monkeypatch):
        base = str(tmp_path)
        self._touch(os.path.join(base, "ground_truth", "frame_0.png"))

        monkeypatch.setattr(viz.imageio, "get_writer", FakeWriter)
        monkeypatch.setattr(viz.imageio, "imread", lambda p: np.zeros((2, 2)))

        viz.compile_videos_from_frames(output_dir=base, fps=30, quality=9,
                                        subdirs=["ground_truth"])

        writer = FakeWriter.instances[0]
        assert writer.kwargs["fps"] == 30
        assert writer.kwargs["quality"] == 9
