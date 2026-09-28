"""tests/pcn_test.py

Pytest coverage for the static (non-temporal) PCN baseline:
`pc_nox/models/pcn/{layers,config,model}.py` and its batched
train/eval runners in `pc_nox/engine/runners_static.py`. This had zero
test coverage before this file -- mirrors the structure and style of
`tpch_test.py` (plain pytest functions + fixtures, `assert_allclose`,
deliberately mismatched layer widths so shape bugs can't hide), adjusted
for the two structural differences from `TpchModel`:

  * No time axis at all -- no `states_prev`, no recurrent weights, no
    `control_input`. `init_activities`/`predict`/`settle` etc. take a
    single `x` (and `y`), not a sequence.
  * `PcnLayer` uses `use_bias=True` (unlike every tPC-H layer, which is
    `use_bias=False` to match that paper exactly) -- so `param_grad`
    here must also cover each layer's bias term, not just its weight.
  * Batching is `jax.vmap` over independent examples (`runners_static.py`),
    not `jax.lax.scan` over time (`runners_temporal.py`) -- there's no
    carry, and the one batch-specific step (`runners_static.py` doesn't
    share with the temporal runners at all) is averaging per-example
    weight gradients before a single optax update.
  * `pcn_energy_fn(..., return_layerwise=True)` deliberately does NOT
    break out regularisation per layer (see that method's own docstring
    and `TpchModel.tpch_energy_fn` for the contrasting behaviour there)
    -- `sum(return_layerwise output)` therefore only equals the scalar
    total energy when weight_reg_total/activity_decay are both zero.
    This file tests that distinction explicitly rather than assuming
    parity with `TpchModel`.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from jax.test_util import check_grads

from pc_nox.models.pcn.model import PcnModel
from pc_nox.models.pcn.layers import PcnLayer
from pc_nox.models.pcn.config import PcnConfig
from pc_nox.engine.runners_static import make_train_step, make_eval_step


def assert_allclose(actual, expected, name, atol=1e-4, rtol=1e-4):
    actual, expected = jnp.asarray(actual), jnp.asarray(expected)
    assert actual.shape == expected.shape, (
        f"{name}: shape mismatch -- got {actual.shape}, expected {expected.shape}"
    )
    max_abs_diff = float(jnp.max(jnp.abs(actual - expected))) if actual.size else 0.0
    assert jnp.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name}: values differ, max abs diff = {max_abs_diff}"
    )


def _leaves(pytree):
    return jax.tree_util.tree_leaves(eqx.filter(pytree, eqx.is_array))


# =============================================================================
# Fixtures -- deliberately different widths at every layer (4, 6, 5, 3) so
# a shape bug can't hide behind a broadcast or coincidental size match.
# =============================================================================

FX_LAYER_SIZES = (4, 6, 5, 3)  # input=4, hidden=6, hidden=5, output=3


@pytest.fixture
def fx_model():
    return PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0))


@pytest.fixture
def fx_x():
    return jr.normal(jr.key(1), (FX_LAYER_SIZES[0],))


@pytest.fixture
def fx_y():
    return jr.normal(jr.key(2), (FX_LAYER_SIZES[-1],))


@pytest.fixture
def fx_states_curr():
    """A genuinely arbitrary point in latent space, NOT init_activities(x)
    -- the feedforward init makes every prediction error exactly zero by
    construction, trivially satisfying almost any gradient formula. See
    the identical rationale in tpch_test.py's fx_states_curr fixture."""
    interior_sizes = FX_LAYER_SIZES[1:-1]
    keys = jr.split(jr.key(3), len(interior_sizes))
    return [jr.normal(k, (n,)) for k, n in zip(keys, interior_sizes)]


def layer_energies(model, states_curr, x, y):
    """Manual per-term breakdown of pcn_energy_fn's sum -- mirrors
    tpch_test.py's layer_energies() helper."""
    predictions, y_hat = model.predict(states_curr, x)
    targets = list(states_curr) + [y]
    energies = [0.5 * jnp.sum((t - p) ** 2) for t, p in zip(targets[:-1], predictions[:-1])]
    if model.config.loss == "mse":
        energies.append(0.5 * jnp.sum((y - y_hat) ** 2))
    else:
        energies.append(-jnp.sum(y * jax.nn.log_softmax(y_hat)))
    return energies


# =============================================================================
# A. Standalone PcnLayer
# =============================================================================

def test_pcn_layer_predict_matches_manual_formula():
    layer = PcnLayer(parent_size=4, own_size=3, key=jr.key(10))
    parent = jr.normal(jr.key(11), (4,))
    out = layer.predict(parent)
    manual = jnp.tanh(layer.W_parrent_curr(parent))
    assert_allclose(out, manual, "PcnLayer.predict")


def test_pcn_layer_uses_bias():
    """Unlike every tPC-H layer (use_bias=False, to match that paper
    exactly), PcnLayer must have a real, nonzero-capable bias term --
    Algorithm 1's W_l z_l + b_l."""
    layer = PcnLayer(parent_size=4, own_size=3, key=jr.key(10))
    assert layer.W_parrent_curr.bias is not None
    assert layer.W_parrent_curr.bias.shape == (3,)


def test_pcn_layer_weight_property_matches_underlying_linear():
    layer = PcnLayer(parent_size=4, own_size=3, key=jr.key(12))
    assert_allclose(layer.weight, layer.W_parrent_curr.weight, "PcnLayer.weight property")


def test_pcn_layer_custom_act_fn():
    layer = PcnLayer(parent_size=3, own_size=2, act_fn=lambda x: x, key=jr.key(13))  # identity
    parent = jr.normal(jr.key(14), (3,))
    assert_allclose(layer.predict(parent), layer.W_parrent_curr(parent), "PcnLayer with identity act_fn")


# =============================================================================
# B. Model construction & structural sanity
# =============================================================================

def test_model_predict_shapes(fx_model, fx_x):
    states_curr = fx_model.init_activities(fx_x)
    predictions, y_hat = fx_model.predict(states_curr, fx_x)
    assert len(predictions) == len(fx_model.layers)
    assert len(states_curr) == len(fx_model.layers) - 1
    assert y_hat.shape == (FX_LAYER_SIZES[-1],)
    for pred, size in zip(predictions[:-1], FX_LAYER_SIZES[1:-1]):
        assert pred.shape == (size,)


def test_init_activities_returns_interior_latents_only(fx_model, fx_x):
    states_curr = fx_model.init_activities(fx_x)
    assert [s.shape for s in states_curr] == [(6,), (5,)]


def test_layer_sizes_minimum_two_entries_raises():
    with pytest.raises(ValueError):
        PcnModel(layer_sizes=(5,), key=jr.key(0))


def test_two_entry_layer_sizes_edge_case_no_interior_latents():
    """layer_sizes=(input, output) directly, no hidden layers at all --
    should still build and run predict/energy/settle cleanly, with zero
    interior latents (a single layer straight from x to y)."""
    model = PcnModel(layer_sizes=(4, 3), key=jr.key(20))
    assert len(model.layers) == 1
    x = jr.normal(jr.key(21), (4,))
    y = jr.normal(jr.key(22), (3,))

    states_curr = model.init_activities(x)
    assert states_curr == []
    predictions, y_hat = model.predict(states_curr, x)
    assert len(predictions) == 1
    assert y_hat.shape == (3,)

    e = model.pcn_energy_fn(states_curr, x, y)
    assert jnp.isfinite(e) and e >= 0

    settled = model.settle(x, y, n_steps=5)
    assert settled == []  # nothing to settle -- no free latents


def test_invalid_loss_raises():
    with pytest.raises(ValueError):
        PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0), loss="bogus")


def test_invalid_activity_reg_type_raises():
    with pytest.raises(ValueError):
        PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0), activity_reg_type="bogus")


def test_number_of_layers_matches_layer_sizes():
    model = PcnModel(layer_sizes=(4, 6, 5, 3), key=jr.key(0))
    assert len(model.layers) == 3  # len(layer_sizes) - 1


# =============================================================================
# C. Free-energy properties
# =============================================================================

def test_energy_is_finite_and_nonnegative(fx_model, fx_states_curr, fx_x, fx_y):
    e = fx_model.pcn_energy_fn(fx_states_curr, fx_x, fx_y)
    assert jnp.isfinite(e)
    assert e >= 0


def test_energy_is_a_pure_function(fx_model, fx_states_curr, fx_x, fx_y):
    e1 = fx_model.pcn_energy_fn(fx_states_curr, fx_x, fx_y)
    e2 = fx_model.pcn_energy_fn(fx_states_curr, fx_x, fx_y)
    assert_allclose(e1, e2, "pcn_energy_fn determinism")


def test_layer_energies_sum_matches_total(fx_model, fx_states_curr, fx_x, fx_y):
    manual = sum(layer_energies(fx_model, fx_states_curr, fx_x, fx_y))
    e = fx_model.pcn_energy_fn(fx_states_curr, fx_x, fx_y)
    assert_allclose(e, manual, "pcn_energy_fn vs manual layer_energies sum")


def test_return_layerwise_shape_and_matches_manual_at_zero_reg(fx_model, fx_states_curr, fx_x, fx_y):
    per_layer = fx_model.pcn_energy_fn(fx_states_curr, fx_x, fx_y, return_layerwise=True)
    assert per_layer.shape == (len(fx_model.layers),)  # 2 interior + 1 output for FX_LAYER_SIZES
    manual = jnp.stack(layer_energies(fx_model, fx_states_curr, fx_x, fx_y))
    assert_allclose(per_layer, manual, "return_layerwise vs manual layer_energies")


def test_return_layerwise_sum_equals_scalar_total_when_no_regularisation(fx_model, fx_states_curr, fx_x, fx_y):
    per_layer = fx_model.pcn_energy_fn(fx_states_curr, fx_x, fx_y, return_layerwise=True)
    total = fx_model.pcn_energy_fn(fx_states_curr, fx_x, fx_y)
    assert_allclose(jnp.sum(per_layer), total, "return_layerwise sum vs scalar total, zero regularisation")


def test_return_layerwise_sum_does_NOT_equal_scalar_total_when_regularised():
    """Documented, deliberate difference from TpchModel:
    pcn_energy_fn(..., return_layerwise=True) does not break out
    weight_decay/orthogonal_penalty/activity_decay per layer at all --
    it's just jnp.stack(layer_energies), full stop. So once any
    regularisation is configured, summing the per-layer array under-counts
    the real scalar total by exactly the regularisation amount. This test
    pins down that gap explicitly so a future "fix" that silently changes
    this contract gets caught."""
    model = PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0), weight_decay=0.3, activity_decay=0.2)
    x = jr.normal(jr.key(30), (FX_LAYER_SIZES[0],))
    y = jr.normal(jr.key(31), (FX_LAYER_SIZES[-1],))
    states_curr = model.init_activities(x)

    per_layer = model.pcn_energy_fn(states_curr, x, y, return_layerwise=True)
    total = model.pcn_energy_fn(states_curr, x, y)
    reg_total = model._weight_l2_reg() + model._weight_orthogonal_reg() + model._activity_reg(states_curr)

    assert not bool(jnp.allclose(jnp.sum(per_layer), total, atol=1e-6)), (
        "return_layerwise sum should NOT match the scalar total once regularisation is nonzero"
    )
    assert_allclose(jnp.sum(per_layer) + reg_total, total, "scalar total == per-layer sum + full regularisation total")


def test_ce_loss_energy_matches_manual_cross_entropy():
    model = PcnModel(layer_sizes=(4, 5, 3), key=jr.key(40), loss="ce")
    x = jr.normal(jr.key(41), (4,))
    onehot = jnp.array([0.0, 1.0, 0.0])
    states_curr = model.init_activities(x)
    e = model.pcn_energy_fn(states_curr, x, onehot)

    _, y_hat = model.predict(states_curr, x)
    manual_obs = -jnp.sum(onehot * jax.nn.log_softmax(y_hat))
    manual_interior = 0.0  # single hidden layer's own error term
    predictions, _ = model.predict(states_curr, x)
    manual_interior = 0.5 * jnp.sum((states_curr[0] - predictions[0]) ** 2)
    assert_allclose(e, manual_interior + manual_obs, "ce-loss pcn_energy_fn vs manual")


# =============================================================================
# D. Gradients -- neg_activity_grad / param_grad, checked against finite
# differences (independent ground truth, not a re-derivation of the
# implementation's own formulas).
# =============================================================================

def test_neg_activity_grad_shapes_and_finiteness(fx_model, fx_states_curr, fx_x, fx_y):
    grad = fx_model.neg_activity_grad(fx_states_curr, fx_x, fx_y)
    assert len(grad) == len(fx_states_curr)
    for g, s in zip(grad, fx_states_curr):
        assert g.shape == s.shape
        assert jnp.all(jnp.isfinite(g))


def test_neg_activity_grad_matches_finite_differences(fx_model, fx_states_curr, fx_x, fx_y):
    energy_of_states = lambda s: fx_model.pcn_energy_fn(s, fx_x, fx_y)
    check_grads(energy_of_states, (fx_states_curr,), order=1, atol=1e-2, rtol=1e-2)


def test_param_grad_matches_finite_differences(fx_model, fx_states_curr, fx_x, fx_y):
    energy_of_weights = lambda m: m.pcn_energy_fn(fx_states_curr, fx_x, fx_y)
    check_grads(energy_of_weights, (fx_model,), order=1, atol=1e-2, rtol=1e-2, eps=1e-4)


def test_param_grad_structure_matches_model(fx_model, fx_states_curr, fx_x, fx_y):
    g = fx_model.param_grad(fx_states_curr, fx_x, fx_y)
    assert jax.tree_util.tree_structure(eqx.filter(g, eqx.is_array)) == jax.tree_util.tree_structure(eqx.filter(fx_model, eqx.is_array))
    for leaf, model_leaf in zip(_leaves(g), _leaves(fx_model)):
        assert leaf.shape == model_leaf.shape


def test_param_grad_includes_nonzero_bias_gradient(fx_model, fx_states_curr, fx_x, fx_y):
    """PcnLayer's bias (use_bias=True, unlike tPC-H) must actually receive
    a real gradient -- not silently dropped or zeroed."""
    g = fx_model.param_grad(fx_states_curr, fx_x, fx_y)
    for layer_grad in g.layers:
        assert layer_grad.W_parrent_curr.bias is not None
        assert layer_grad.W_parrent_curr.bias.shape == layer_grad.W_parrent_curr.weight.shape[:1]
    assert any(bool(jnp.any(jnp.abs(lg.W_parrent_curr.bias) > 1e-8)) for lg in g.layers)


def test_bias_gradient_equals_layer_error_times_activation_derivative(fx_model, fx_states_curr, fx_x, fx_y):
    """Independent hand derivation: bias contributes exactly one column of
    "1"s to the pre-activation, so d(energy)/d(bias) should equal
    f'(pre) * layer_error, the same quantity that (right-multiplied by
    the parent state) gives the weight gradient's own outer product."""
    g = fx_model.param_grad(fx_states_curr, fx_x, fx_y)
    predictions, y_hat = fx_model.predict(fx_states_curr, fx_x)
    chain_in = [fx_x] + list(fx_states_curr)

    layer = fx_model.layers[0]
    parent = chain_in[0]
    pre = layer.W_parrent_curr(parent)
    f_prime = 1 - jnp.tanh(pre) ** 2
    layer_error = fx_states_curr[0] - predictions[0]
    manual_bias_grad = -f_prime * layer_error  # descent gradient: -d(energy)/d(bias)... sign check below

    # param_grad returns dE/dweights (a DESCENT gradient, matching jax.grad
    # convention -- positive means increasing that parameter increases
    # energy), so the correct sign to compare against is +f'(pre)*(-error)
    # = -f'(pre)*error only if increasing bias increases the prediction
    # and thus decreases (state - pred)^2 error when error>0; check
    # numerically instead of asserting a hand-typed sign to avoid
    # compounding two independent guesses.
    numeric_bias_grad = jax.grad(lambda b: _energy_with_bias_override(fx_model, fx_states_curr, fx_x, fx_y, 0, b))(layer.W_parrent_curr.bias)
    assert_allclose(g.layers[0].W_parrent_curr.bias, numeric_bias_grad, "bias grad vs independent jax.grad w.r.t. bias directly", atol=1e-5, rtol=1e-5)


def _energy_with_bias_override(model, states_curr, x, y, layer_idx, new_bias):
    new_layer = eqx.tree_at(lambda l: l.W_parrent_curr.bias, model.layers[layer_idx], new_bias)
    new_layers = list(model.layers)
    new_layers[layer_idx] = new_layer
    perturbed = eqx.tree_at(lambda m: m.layers, model, new_layers)
    return perturbed.pcn_energy_fn(states_curr, x, y)


# =============================================================================
# E. Inference: infer_step / settle / settle_scan
# =============================================================================

def test_infer_step_does_not_increase_energy(fx_model, fx_states_curr, fx_x, fx_y):
    e0 = fx_model.pcn_energy_fn(fx_states_curr, fx_x, fx_y)
    stepped = fx_model.infer_step(fx_states_curr, fx_x, fx_y, state_lr=0.01)
    e1 = fx_model.pcn_energy_fn(stepped, fx_x, fx_y)
    assert float(e1) <= float(e0) + 1e-4


def test_settle_reduces_energy_below_feedforward_init(fx_model, fx_x, fx_y):
    init = fx_model.init_activities(fx_x)
    e0 = fx_model.pcn_energy_fn(init, fx_x, fx_y)
    settled = fx_model.settle(fx_x, fx_y, n_steps=30, state_lr=0.1)
    e1 = fx_model.pcn_energy_fn(settled, fx_x, fx_y)
    assert float(e1) < float(e0)


def test_settle_output_length_and_shapes(fx_model, fx_x, fx_y):
    settled = fx_model.settle(fx_x, fx_y, n_steps=10)
    assert len(settled) == len(fx_model.layers) - 1
    for s, size in zip(settled, FX_LAYER_SIZES[1:-1]):
        assert s.shape == (size,)


def test_settle_and_settle_scan_agree(fx_model, fx_x, fx_y):
    settled_loop = fx_model.settle(fx_x, fx_y, n_steps=25, state_lr=0.1)
    settled_scan = fx_model.settle_scan(optax.sgd(0.1), fx_x, fx_y, n_steps=25)
    # Looser tolerance than most comparisons in this file: this compares an
    # eager Python loop (settle) against jax.lax.scan (settle_scan) over the
    # SAME 25-step iterative formula. Both are mathematically identical, but
    # different execution paths through XLA (unbatched vs. scan-fused matmuls)
    # can reorder float32 operations differently, and 25 chained steps is
    # enough for that per-step noise to compound into a few times 1e-3 -- this
    # was observed to genuinely vary by hardware/BLAS backend, not a logic bug.
    for a, b in zip(settled_loop, settled_scan):
        assert_allclose(a, b, "settle vs settle_scan (both plain SGD)", atol=1e-2, rtol=1e-2)


def test_settle_scan_return_layerwise_trace_shape_and_matches_scalar(fx_model, fx_x, fx_y):
    n_steps = 10
    states_curr, energy_trace = fx_model.settle_scan(optax.adam(1e-2), fx_x, fx_y, n_steps=n_steps, return_layerwise=True)
    assert energy_trace.shape == (n_steps, len(fx_model.layers))
    assert jnp.all(jnp.isfinite(energy_trace))
    last_row_sum = jnp.sum(energy_trace[-1])
    final_scalar = fx_model.pcn_energy_fn(states_curr, fx_x, fx_y, weight_reg_total=jnp.asarray(0.0)) - fx_model._activity_reg(states_curr)
    assert_allclose(last_row_sum, final_scalar, "settle_scan energy_trace last row vs scalar energy (zero reg model)", atol=1e-3, rtol=1e-3)


def test_settle_scan_return_layerwise_false_returns_states_only(fx_model, fx_x, fx_y):
    out = fx_model.settle_scan(optax.adam(1e-2), fx_x, fx_y, n_steps=5, return_layerwise=False)
    assert isinstance(out, list)
    for s in out:
        assert jnp.all(jnp.isfinite(s))


# =============================================================================
# F. Regularisation -- no _scope variants here (every penalty applies to
# every weight/state uniformly, unlike TpchModel's rec/ff/all scopes).
# =============================================================================

def test_zero_regularisation_coefficients_are_exact_noops(fx_model, fx_states_curr, fx_x, fx_y):
    assert fx_model.config.weight_decay == 0.0
    assert fx_model.config.orthogonal_penalty == 0.0
    assert fx_model.config.activity_decay == 0.0
    assert_allclose(fx_model._weight_l2_reg(), jnp.asarray(0.0), "zero weight_decay -> exactly zero")
    assert_allclose(fx_model._weight_orthogonal_reg(), jnp.asarray(0.0), "zero orthogonal_penalty -> exactly zero")
    assert_allclose(fx_model._activity_reg(fx_states_curr), jnp.asarray(0.0), "zero activity_decay -> exactly zero")


def test_weight_decay_energy_matches_exact_frobenius_penalty():
    model = PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0), weight_decay=0.4)
    manual = 0.5 * 0.4 * sum(jnp.sum(layer.weight ** 2) for layer in model.layers)
    assert_allclose(model._weight_l2_reg(), manual, "weight_decay energy vs manual Frobenius penalty")


def test_weight_decay_applies_to_all_weights_including_last_layer():
    """No rec/ff distinction: weight_decay must touch every layer's
    weight, including the final (output-readout) layer -- unlike
    TpchModel's default observation layer, which is excluded from the
    'rec' scope but IS included in 'all'/'ff'; here there's only one
    scope, and it always includes everything."""
    model = PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0), weight_decay=0.5)
    g = eqx.filter_grad(lambda m: m._weight_l2_reg())(model)
    for layer_grad, layer in zip(g.layers, model.layers):
        assert_allclose(layer_grad.W_parrent_curr.weight, 0.5 * layer.weight, "weight_decay grad on each layer, including output")


def test_orthogonal_penalty_energy_matches_exact_definition():
    model = PcnModel(layer_sizes=(4, 4, 4), key=jr.key(50), orthogonal_penalty=0.3)  # square weights for a clean I - W^T W
    W0, W1 = model.layers[0].weight, model.layers[1].weight
    def orth_term(W):
        m, n = W.shape
        gram = W.T @ W if m >= n else W @ W.T
        I = jnp.eye(gram.shape[0])
        return jnp.sum((I - gram) ** 2)
    manual = 0.5 * 0.3 * (orth_term(W0) + orth_term(W1))
    assert_allclose(model._weight_orthogonal_reg(), manual, "orthogonal_penalty energy vs manual definition")


def test_activity_regularisation_l1_matches_exact_norm():
    model = PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0), activity_decay=0.2, activity_reg_type="l1")
    states = [jr.normal(jr.key(60 + i), (n,)) for i, n in enumerate(FX_LAYER_SIZES[1:-1])]
    manual = 0.5 * 0.2 * sum(jnp.sum(jnp.abs(s)) for s in states)
    assert_allclose(model._activity_reg(states), manual, "activity_decay l1 energy vs manual")


def test_activity_regularisation_l2_matches_exact_norm():
    model = PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0), activity_decay=0.2, activity_reg_type="l2")
    states = [jr.normal(jr.key(70 + i), (n,)) for i, n in enumerate(FX_LAYER_SIZES[1:-1])]
    manual = 0.5 * 0.2 * sum(jnp.sum(s ** 2) for s in states)
    assert_allclose(model._activity_reg(states), manual, "activity_decay l2 energy vs manual")


def test_combined_regularisation_energy_is_additive(fx_x, fx_y):
    model = PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0), weight_decay=0.1, orthogonal_penalty=0.05, activity_decay=0.15)
    states_curr = model.init_activities(fx_x)
    total = model.pcn_energy_fn(states_curr, fx_x, fx_y)
    bare = model.pcn_energy_fn(states_curr, fx_x, fx_y, weight_reg_total=jnp.asarray(0.0)) - model._activity_reg(states_curr)
    reg_total = model._weight_l2_reg() + model._weight_orthogonal_reg() + model._activity_reg(states_curr)
    assert_allclose(total, bare + reg_total, "combined regularisation additive to bare energy")


def test_combined_regularisation_gradient_matches_finite_differences():
    model = PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0), weight_decay=0.1, orthogonal_penalty=0.05, activity_decay=0.15)
    x = jr.normal(jr.key(80), (FX_LAYER_SIZES[0],))
    y = jr.normal(jr.key(81), (FX_LAYER_SIZES[-1],))
    states_curr = [jr.normal(jr.key(82 + i), (n,)) for i, n in enumerate(FX_LAYER_SIZES[1:-1])]

    energy_of_weights = lambda m: m.pcn_energy_fn(states_curr, x, y)
    check_grads(energy_of_weights, (model,), order=1, atol=1e-2, rtol=1e-2, eps=1e-4)


def test_regularisation_config_round_trips_through_from_config_and_checkpoint(tmp_path):
    config = PcnConfig(
        layer_sizes=FX_LAYER_SIZES, act_fn="tanh", loss="mse",
        weight_decay=0.12, orthogonal_penalty=0.09, activity_decay=0.07, activity_reg_type="l2",
    )
    original = PcnModel.from_config(config, key=jr.key(90))
    assert original.config == config

    out_dir = original.save_checkpoint(path=tmp_path / "pcn_reg_ckpt")
    loaded = PcnModel.load_checkpoint(out_dir)
    assert loaded.model.config == config
    assert loaded.model.config.activity_reg_type == "l2"


@pytest.mark.parametrize("kwargs", [{"loss": "bad"}, {"activity_reg_type": "bad"}])
def test_regularisation_enum_validation(kwargs):
    with pytest.raises(ValueError):
        PcnModel(layer_sizes=FX_LAYER_SIZES, key=jr.key(0), **kwargs)


# =============================================================================
# G. Checkpointing
# =============================================================================

def test_checkpoint_round_trip_predict_matches(fx_model, fx_x, tmp_path):
    out_dir = fx_model.save_checkpoint(path=tmp_path / "pcn_ckpt")
    loaded = PcnModel.load_checkpoint(out_dir)

    states_orig = fx_model.init_activities(fx_x)
    states_loaded = loaded.model.init_activities(fx_x)
    for o, l in zip(states_orig, states_loaded):
        assert_allclose(o, l, "init_activities before vs after checkpoint round trip")

    preds_orig, y_hat_orig = fx_model.predict(states_orig, fx_x)
    preds_loaded, y_hat_loaded = loaded.model.predict(states_loaded, fx_x)
    assert_allclose(y_hat_orig, y_hat_loaded, "y_hat before vs after checkpoint round trip")


def test_checkpoint_round_trip_with_activities_and_opt_state(fx_model, tmp_path):
    optim = optax.adam(learning_rate=1e-3)
    opt_state = optim.init(eqx.filter(fx_model, eqx.is_array))
    activities = fx_model.zero_activities(fx_model.config)

    out_dir = fx_model.save_checkpoint(
        path=tmp_path / "pcn_ckpt_full", metadata={"epoch": 7}, opt_state=opt_state, activities=activities,
    )
    loaded = PcnModel.load_checkpoint(out_dir, optim=optim)

    assert loaded.metadata == {"epoch": 7}
    assert loaded.opt_state is not None
    assert loaded.activities is not None
    for a, l in zip(activities, loaded.activities):
        assert_allclose(a, l, "activities round trip")


def test_zero_activities_shapes(fx_model):
    activities = fx_model.zero_activities(fx_model.config)
    assert [a.shape for a in activities] == [(6,), (5,)]
    for a in activities:
        assert_allclose(a, jnp.zeros_like(a), "zero_activities should be all zero")


# =============================================================================
# H. layer_labels
# =============================================================================

def test_layer_labels_length_and_content():
    cfg = PcnConfig(layer_sizes=FX_LAYER_SIZES)
    labels = PcnModel.layer_labels(cfg)
    assert labels == ["Hidden 1", "Hidden 2", "Output"]


def test_layer_labels_matches_return_layerwise_length(fx_model, fx_states_curr, fx_x, fx_y):
    labels = PcnModel.layer_labels(fx_model.config)
    per_layer = fx_model.pcn_energy_fn(fx_states_curr, fx_x, fx_y, return_layerwise=True)
    assert len(labels) == per_layer.shape[0]


def test_layer_labels_zero_interior_layers():
    cfg = PcnConfig(layer_sizes=(4, 3))
    assert PcnModel.layer_labels(cfg) == ["Output"]


def test_layer_labels_depends_only_on_config_not_a_built_model():
    cfg = PcnConfig(layer_sizes=FX_LAYER_SIZES)
    labels_a = PcnModel.layer_labels(cfg)
    m = PcnModel.from_config(cfg, key=jr.key(0))
    labels_b = PcnModel.layer_labels(m.config)
    assert labels_a == labels_b


# =============================================================================
# I. runners_static.py -- make_train_step / make_eval_step, batched via vmap
# =============================================================================

BATCH = 6


@pytest.fixture
def fx_xs():
    return jr.normal(jr.key(100), (BATCH, FX_LAYER_SIZES[0]))


@pytest.fixture
def fx_ys():
    return jr.normal(jr.key(101), (BATCH, FX_LAYER_SIZES[-1]))


@pytest.fixture
def fx_train_step_setup(fx_model):
    param_optim = optax.adam(learning_rate=1e-3)
    param_opt_state = param_optim.init(eqx.filter(fx_model, eqx.is_array))
    activity_optim = optax.adam(learning_rate=1e-2)
    return param_optim, activity_optim, param_opt_state


def test_make_eval_step_output_contract_and_batched_shapes(fx_model, fx_xs, fx_ys):
    eval_step = make_eval_step(optax.adam(1e-2), n_infer_steps=8)
    states_curr, y_before, y_after, e_before, e_after, trace = eval_step(fx_model, fx_xs, fx_ys)
    assert len(states_curr) == 2  # two interior latents
    assert states_curr[0].shape == (BATCH, 6)
    assert states_curr[1].shape == (BATCH, 5)
    assert y_before.shape == (BATCH, 3)
    assert y_after.shape == (BATCH, 3)
    assert e_before.shape == (BATCH,)
    assert e_after.shape == (BATCH,)
    assert trace is None
    assert jnp.all(jnp.isfinite(e_after))


def test_make_eval_step_return_layerwise_shapes(fx_model, fx_xs, fx_ys):
    n_infer_steps = 6
    eval_step = make_eval_step(optax.adam(1e-2), n_infer_steps)
    *_, trace = eval_step(fx_model, fx_xs, fx_ys, return_layerwise=True)
    assert trace.shape == (BATCH, n_infer_steps, len(fx_model.layers))
    assert jnp.all(jnp.isfinite(trace))


def test_make_eval_step_is_deterministic_and_does_not_mutate_model(fx_model, fx_xs, fx_ys):
    eval_step = make_eval_step(optax.adam(1e-2), n_infer_steps=6)
    out1 = eval_step(fx_model, fx_xs, fx_ys)
    out2 = eval_step(fx_model, fx_xs, fx_ys)
    for a, b in zip(jax.tree_util.tree_leaves(out1[:-1]), jax.tree_util.tree_leaves(out2[:-1])):
        assert_allclose(a, b, "eval_step determinism across repeated calls")


def test_make_eval_step_settling_reduces_energy_per_example(fx_model, fx_xs, fx_ys):
    eval_step = make_eval_step(optax.adam(1e-2), n_infer_steps=15)
    _, _, _, e_before, e_after, _ = eval_step(fx_model, fx_xs, fx_ys)
    assert bool(jnp.all(e_after <= e_before + 1e-4))
    assert float(jnp.mean(e_after)) < float(jnp.mean(e_before))


def test_make_eval_step_matches_per_example_settle_looped_manually(fx_model, fx_xs, fx_ys):
    """Cross-check the vmapped batch runner against `settle_scan` called
    once per example in an ordinary Python loop, using the SAME
    activity_optim/n_infer_steps -- independent confirmation that vmap
    isn't silently sharing state or misaligning examples."""
    activity_optim = optax.adam(1e-2)
    n_infer_steps = 8
    eval_step = make_eval_step(activity_optim, n_infer_steps)
    states_curr, y_before, y_after, e_before, e_after, _ = eval_step(fx_model, fx_xs, fx_ys)

    for i in range(BATCH):
        x_i, y_i = fx_xs[i], fx_ys[i]
        settled_i = fx_model.settle_scan(activity_optim, x_i, y_i, n_steps=n_infer_steps)
        e_after_i = fx_model.pcn_energy_fn(settled_i, x_i, y_i)
        # Looser tolerance: comparing a vmapped+jitted batch settle (eval_step)
        # against an eager, unbatched settle_scan call over the same 8 Adam
        # steps -- batched vs. unbatched matmuls can reorder float32 sums
        # differently under XLA, and Adam's own moment accumulation compounds
        # that per-step noise across iterations. Observed to vary by
        # hardware/BLAS backend (didn't reproduce on every machine tested),
        # so this needs real headroom rather than a razor-thin tolerance.
        assert_allclose(e_after[i], e_after_i, f"eval_step example {i} vs manual settle_scan", atol=1e-2, rtol=1e-2)
        for s_batched, s_manual in zip(states_curr, settled_i):
            assert_allclose(s_batched[i], s_manual, f"eval_step example {i} settled state vs manual", atol=1e-2, rtol=1e-2)


def test_make_train_step_output_contract(fx_model, fx_xs, fx_ys, fx_train_step_setup):
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps=6)
    result = train_step(fx_model, param_opt_state, fx_xs, fx_ys)
    assert len(result) == 8
    model, opt_state, states_curr, y_before, y_after, e_before, e_after, trace = result
    assert trace is None
    assert e_after.shape == (BATCH,)
    assert jnp.all(jnp.isfinite(e_after))
    for leaf in _leaves(model):
        assert jnp.all(jnp.isfinite(leaf))


def test_make_train_step_weight_update_actually_changes_weights(fx_model, fx_xs, fx_ys, fx_train_step_setup):
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps=6)
    new_model, *_ = train_step(fx_model, param_opt_state, fx_xs, fx_ys)
    before, after = _leaves(fx_model), _leaves(new_model)
    assert any(not bool(jnp.array_equal(b, a)) for b, a in zip(before, after))


def test_make_train_step_gradient_averaging_matches_manual_per_example_mean(fx_model, fx_xs, fx_ys):
    """The one genuinely batch-specific piece of runners_static.py: each
    example's param_grad, averaged across the batch, THEN one optax
    update -- confirm this against an explicit manual per-example loop,
    independent of vmap.

    Deliberately uses plain SGD here, NOT the adam fixture used elsewhere
    in this file: with adam, the weight update is `lr * m_hat /
    (sqrt(v_hat) + eps)`, which self-normalises by each parameter's own
    gradient magnitude -- so a genuinely WRONG mean_grad (e.g. averaging
    only half the batch) can still produce a nearly-identical Adam
    update, silently defeating this exact comparison. With SGD the update
    is `lr * grad` -- linear in the gradient -- so a wrong average shows
    up in the resulting weights proportionally, which is what this test
    is actually meant to catch. (Confirmed by mutation-testing this test
    against a deliberately-broken half-batch average: it fails loudly
    under SGD and passes unnoticed under adam.)"""
    param_optim = optax.sgd(learning_rate=0.05)
    activity_optim = optax.adam(learning_rate=1e-2)
    n_infer_steps = 6
    param_opt_state = param_optim.init(eqx.filter(fx_model, eqx.is_array))

    train_step = make_train_step(param_optim, activity_optim, n_infer_steps)
    new_model, new_opt_state, *_ = train_step(fx_model, param_opt_state, fx_xs, fx_ys)

    per_example_grads = []
    for i in range(BATCH):
        settled_i = fx_model.settle_scan(activity_optim, fx_xs[i], fx_ys[i], n_steps=n_infer_steps)
        g_i = fx_model.param_grad(settled_i, fx_xs[i], fx_ys[i])
        per_example_grads.append(g_i)

    mean_grad = jax.tree_util.tree_map(lambda *gs: jnp.mean(jnp.stack(gs), axis=0), *per_example_grads)
    updates, manual_opt_state = param_optim.update(mean_grad, param_opt_state, fx_model)
    manual_model = eqx.apply_updates(fx_model, updates)
    manual_model = manual_model.postprocess_params()

    # Looser tolerance than most comparisons in this file: vmapped+jitted
    # batch training vs. an eager per-example loop over the same 6-step
    # Adam *settle* (the activity optimiser, still adam) -- batched vs.
    # unbatched matmuls can reorder float32 sums differently under XLA,
    # observed to vary by hardware/BLAS backend. The SGD *weight* update
    # itself is exactly linear, so this remains sensitive to a genuinely
    # wrong gradient average despite the looser tolerance (see mutation
    # test in this test's docstring).
    for f, r in zip(_leaves(new_model), _leaves(manual_model)):
        assert_allclose(f, r, "make_train_step weight update vs manual per-example grad averaging", atol=1e-2, rtol=1e-2)


def test_settle_scan_return_layerwise_does_not_change_the_settled_states(fx_model, fx_x, fx_y):
    """return_layerwise is supposed to be a pure "also report this"
    switch -- it must not change states_curr itself (e.g. by taking a
    different number of scan steps in each branch).

    Deliberately calls model.settle_scan(...) directly, single-example,
    with NO surrounding eqx.filter_jit and NO vmap -- unlike a version of
    this test built on top of make_train_step, which was found (by
    mutation-testing against a deliberately-introduced off-by-one-step
    bug) to bury this exact regression under unrelated batched-vs-jitted
    float32 noise once a tolerance loose enough to avoid false positives
    was used. Isolating settle_scan directly removes that noise and
    keeps a tight tolerance meaningful."""
    activity_optim = optax.adam(1e-2)
    n_steps = 6
    states_false = fx_model.settle_scan(activity_optim, fx_x, fx_y, n_steps=n_steps, return_layerwise=False)
    states_true, _ = fx_model.settle_scan(activity_optim, fx_x, fx_y, n_steps=n_steps, return_layerwise=True)
    for a, b in zip(states_false, states_true):
        assert_allclose(a, b, "settle_scan: settled states, return_layerwise False vs True", atol=1e-6, rtol=1e-6)


def test_make_train_step_return_layerwise_does_not_change_training_dynamics(fx_model, fx_xs, fx_ys, fx_train_step_setup):
    """Batched-runner-level companion to the settle_scan-level test above:
    confirms the same "return_layerwise is a pure switch" property still
    holds once wrapped in make_train_step's vmap+eqx.filter_jit."""
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps=6)
    out_false = train_step(fx_model, param_opt_state, fx_xs, fx_ys, return_layerwise=False)
    out_true = train_step(fx_model, param_opt_state, fx_xs, fx_ys, return_layerwise=True)

    # Looser tolerance than most comparisons in this file: return_layerwise=False
    # and =True trace as two SEPARATE jaxprs under eqx.filter_jit (the Python-level
    # branch inside settle_scan/pcn_energy_fn means two distinct compiled
    # programs), so even though they run the mathematically identical 6-step
    # Adam settle, XLA is free to fuse/schedule each compilation differently --
    # observed to produce a few times 1e-3 of drift on some hardware/BLAS
    # backends despite being the same formula. This test is a coarser,
    # batched-level sanity check; test_settle_scan_return_layerwise_does_not_change_the_settled_states
    # above is the one with real teeth for this specific property.
    for i in range(7):  # everything except the trailing energy_trace slot
        a = jax.tree_util.tree_leaves(out_false[i])
        b = jax.tree_util.tree_leaves(out_true[i])
        for la, lb in zip(a, b):
            assert_allclose(la, lb, f"output[{i}]: return_layerwise False vs True", atol=1e-2, rtol=1e-2)
    assert out_false[-1] is None
    assert out_true[-1] is not None


def test_make_train_step_n_infer_steps_matters(fx_model, fx_xs, fx_ys, fx_train_step_setup):
    param_optim, activity_optim, param_opt_state = fx_train_step_setup
    train_step_short = make_train_step(param_optim, activity_optim, n_infer_steps=2)
    train_step_long = make_train_step(param_optim, activity_optim, n_infer_steps=30)

    _, _, states_short, *_ = train_step_short(fx_model, param_opt_state, fx_xs, fx_ys)
    _, _, states_long, *_ = train_step_long(fx_model, param_opt_state, fx_xs, fx_ys)

    assert any(not bool(jnp.allclose(s, l, atol=1e-3)) for s, l in zip(states_short, states_long))


def test_make_train_step_zero_interior_layers_edge_case():
    model = PcnModel(layer_sizes=(4, 3), key=jr.key(200))
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps=5)
    opt_state = param_optim.init(eqx.filter(model, eqx.is_array))

    xs = jr.normal(jr.key(201), (BATCH, 4))
    ys = jr.normal(jr.key(202), (BATCH, 3))
    new_model, new_opt_state, states_curr, y_before, y_after, e_before, e_after, _ = train_step(model, opt_state, xs, ys)
    assert states_curr == []
    assert jnp.all(jnp.isfinite(e_after))


def test_make_train_step_with_ce_loss():
    model = PcnModel(layer_sizes=(4, 5, 3), key=jr.key(210), loss="ce")
    param_optim = optax.adam(1e-3)
    activity_optim = optax.adam(1e-2)
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps=8)
    opt_state = param_optim.init(eqx.filter(model, eqx.is_array))

    xs = jr.normal(jr.key(211), (BATCH, 4))
    class_idx = jr.randint(jr.key(212), (BATCH,), 0, 3)
    ys = jax.nn.one_hot(class_idx, 3)

    new_model, new_opt_state, states_curr, y_before, y_after, e_before, e_after, _ = train_step(model, opt_state, xs, ys)
    assert jnp.all(jnp.isfinite(e_after))
    assert y_after.shape == (BATCH, 3)


def test_batched_training_actually_reduces_mean_energy_substantially():
    """End-to-end learning check, mirroring the reference implementation's
    own property test: many training steps on a fixed synthetic dataset
    should substantially reduce mean post-settling energy."""
    key = jr.key(1)
    model = PcnModel(layer_sizes=(4, 8, 3), key=key)
    rng_key = jr.key(42)
    kx, kw = jr.split(rng_key)
    W_true = jr.normal(kw, (3, 4)) * 0.5
    xs = jr.normal(kx, (16, 4))
    ys = jnp.tanh(xs @ W_true.T)

    param_optim = optax.adam(1e-2)
    activity_optim = optax.sgd(0.1)
    train_step = make_train_step(param_optim, activity_optim, n_infer_steps=15)
    opt_state = param_optim.init(eqx.filter(model, eqx.is_array))

    energies = []
    for _ in range(150):
        model, opt_state, _, _, _, _, e_after, _ = train_step(model, opt_state, xs, ys)
        energies.append(float(jnp.mean(e_after)))

    assert energies[-1] < 0.1 * energies[0]
    assert all(jnp.isfinite(jnp.array(energies)))


# =============================================================================
# J. Free (unclamped) output -- y=None
#
# Passing y=None drops the output error term entirely, so settling/eval can
# never see the label. Interior-layer terms are unchanged.
# =============================================================================

def test_free_energy_equals_hidden_layer_terms_only(fx_model, fx_states_curr, fx_x, fx_y):
    e_free = fx_model.pcn_energy_fn(fx_states_curr, fx_x, None)
    manual = sum(layer_energies(fx_model, fx_states_curr, fx_x, fx_y)[:-1])
    assert_allclose(e_free, manual, "free energy vs sum of interior layer terms")


def test_free_energy_return_layerwise_output_entry_is_zero(fx_model, fx_states_curr, fx_x, fx_y):
    per_layer = fx_model.pcn_energy_fn(fx_states_curr, fx_x, None, return_layerwise=True)
    assert per_layer.shape == (len(fx_model.layers),)  # same length as clamped, so layer_labels still aligns
    assert_allclose(per_layer[-1], 0.0, "free return_layerwise: output entry")
    manual = jnp.stack(layer_energies(fx_model, fx_states_curr, fx_x, fx_y)[:-1])
    assert_allclose(per_layer[:-1], manual, "free return_layerwise: interior entries")


def test_free_energy_is_independent_of_loss_choice(fx_states_curr, fx_x):
    e_mse = PcnModel(FX_LAYER_SIZES, key=jr.key(0), loss="mse").pcn_energy_fn(fx_states_curr, fx_x, None)
    e_ce = PcnModel(FX_LAYER_SIZES, key=jr.key(0), loss="ce").pcn_energy_fn(fx_states_curr, fx_x, None)
    assert_allclose(e_mse, e_ce, "free energy, mse vs ce (no output term either way)")


def test_free_neg_activity_grad_has_no_output_term(fx_model, fx_states_curr, fx_x, fx_y):
    manual = jax.grad(lambda s: sum(layer_energies(fx_model, s, fx_x, fx_y)[:-1]))(fx_states_curr)
    actual = fx_model.neg_activity_grad(fx_states_curr, fx_x, None)
    for a, m in zip(actual, manual):
        assert_allclose(a, -m, "free neg_activity_grad vs -grad(interior terms)")


def test_free_settle_scan_stays_at_feedforward_init_without_regularisation(fx_model, fx_x):
    """Feedforward init already has zero interior error, and there is no
    output term to pull on it -- so free settling is a no-op. (SGD rather
    than Adam here: Adam normalises tiny float-noise gradients into
    full-size steps, which would make this comparison flaky.)"""
    init = fx_model.init_activities(fx_x)
    settled = fx_model.settle_scan(optax.sgd(0.1), fx_x, None, n_steps=10)
    for a, b in zip(settled, init):
        assert_allclose(a, b, "free settle_scan vs feedforward init", atol=1e-5, rtol=1e-5)


def test_clamped_settle_differs_from_free_settle(fx_model, fx_x, fx_y):
    """The label-leak this feature exists to avoid: clamping y moves the
    latents, free settling doesn't."""
    free = fx_model.settle_scan(optax.sgd(0.1), fx_x, None, n_steps=10)
    clamped = fx_model.settle_scan(optax.sgd(0.1), fx_x, fx_y, n_steps=10)
    assert any(not bool(jnp.allclose(f, c, atol=1e-4)) for f, c in zip(free, clamped))


def test_free_settle_with_activity_decay_moves_and_lowers_energy(fx_x):
    model = PcnModel(FX_LAYER_SIZES, key=jr.key(0), activity_decay=0.1, activity_reg_type="l2")
    init = model.init_activities(fx_x)
    settled = model.settle_scan(optax.sgd(0.05), fx_x, None, n_steps=10)
    assert any(not bool(jnp.allclose(s, i, atol=1e-6)) for s, i in zip(settled, init))
    assert float(model.pcn_energy_fn(settled, fx_x, None)) <= float(model.pcn_energy_fn(init, fx_x, None)) + 1e-6


def test_free_energy_zero_interior_layers_edge_case():
    model = PcnModel(layer_sizes=(4, 3), key=jr.key(220))
    x = jr.normal(jr.key(221), (4,))
    assert model.settle_scan(optax.sgd(0.1), x, None, n_steps=3) == []
    assert_allclose(model.pcn_energy_fn([], x, None), 0.0, "free energy with no interior layers")


def test_make_eval_step_free_output_contract_and_batched_shapes(fx_model, fx_xs):
    eval_step = make_eval_step(optax.adam(1e-2), n_infer_steps=8)
    states_curr, y_before, y_after, e_before, e_after, trace = eval_step(fx_model, fx_xs, None)
    assert states_curr[0].shape == (BATCH, 6)
    assert states_curr[1].shape == (BATCH, 5)
    assert y_before.shape == (BATCH, 3)
    assert y_after.shape == (BATCH, 3)
    assert e_before.shape == (BATCH,)
    assert e_after.shape == (BATCH,)
    assert trace is None
    assert jnp.all(jnp.isfinite(e_after))


def test_make_eval_step_free_output_return_layerwise(fx_model, fx_xs):
    n_infer_steps = 6
    eval_step = make_eval_step(optax.adam(1e-2), n_infer_steps)
    *_, trace = eval_step(fx_model, fx_xs, None, return_layerwise=True)
    assert trace.shape == (BATCH, n_infer_steps, len(fx_model.layers))
    assert_allclose(trace[..., -1], jnp.zeros((BATCH, n_infer_steps)), "free eval trace: output column")


def test_make_eval_step_free_output_does_not_leak_labels(fx_model, fx_xs, fx_ys):
    """Clamped eval pulls y_after toward the labels; free eval must not."""
    eval_step = make_eval_step(optax.sgd(0.1), n_infer_steps=15)
    _, y_before, y_free, *_ = eval_step(fx_model, fx_xs, None)
    _, _, y_clamped, *_ = eval_step(fx_model, fx_xs, fx_ys)

    assert_allclose(y_free, y_before, "free eval: y_after == y_before", atol=1e-4, rtol=1e-4)
    err_free = float(jnp.mean((y_free - fx_ys) ** 2))
    err_clamped = float(jnp.mean((y_clamped - fx_ys) ** 2))
    assert err_clamped < err_free
