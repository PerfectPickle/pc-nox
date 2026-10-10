"""graph/optim_utils.py

Per-node learning rates via `optax.multi_transform`, for the parameter optimiser
and/or the activity optimiser.

Design notes
------------
* Rules live HERE (training-script level), not on `NodeSpec`. A learning rate does
  not change what the model computes, so by `ModelBase`'s own config-vs-metadata
  rule it is run metadata. Keeping it out of the config also leaves the config
  format (and old checkpoints) untouched and lets you sweep rates without editing
  the model definition. Record the rules you used in checkpoint `metadata`.
* `optax.multi_transform` returns ONE `GradientTransformation` with ONE state
  pytree, so the checkpoint format does not change: pass the same grouped optimiser
  as `optim=` to `load_checkpoint` and the state restores like any other. (The
  activity optimiser's state is created inside every settle and never checkpointed.)
* Rules are keyed `"kind:<NodeSpec.kind>"` or `"name:<NodeSpec.name>"` and map to a
  multiplier on `base_lr`. A node takes the most specific rule that matches it
  (name beats kind); unmatched nodes use scale 1.0. A rule that matches no node is
  an error (it is almost always a typo).

Example (the conv-nodes-need-a-smaller-rate hypothesis):

    param_optim = grouped_optimiser(config, optax.adam, 1e-3, {"kind:pcn_conv": 0.1})
    act_optim   = grouped_optimiser(config, optax.sgd,  0.5,  {"kind:pcn_conv": 0.5}, target="states")
"""

from typing import Callable, Dict, Mapping, Optional, Tuple

import jax
import optax

from .config import GraphModelConfig


def resolve_groups(config: GraphModelConfig, lr_scales: Optional[Mapping[str, float]]) -> Tuple[Dict[str, str], Dict[str, float]]:
    """node name -> group label, and group label -> lr multiplier."""
    lr_scales = dict(lr_scales or {})
    kinds, names = {s.kind for s in config.node_specs}, set(config.names)
    for key, scale in lr_scales.items():
        prefix, _, ident = key.partition(":")
        if prefix not in ("kind", "name") or not ident:
            raise ValueError(f"rule {key!r} must look like 'kind:<kind>' or 'name:<node name>'")
        valid = kinds if prefix == "kind" else names
        if ident not in valid:
            raise ValueError(f"rule {key!r} matches no node; available {prefix}s: {sorted(valid)}")
        if not scale > 0:
            raise ValueError(f"rule {key!r}: scale must be > 0, got {scale}")
    group_of, scale_of = {}, {"default": 1.0}
    for spec in config.node_specs:
        label = next((k for k in (f"name:{spec.name}", f"kind:{spec.kind}") if k in lr_scales), "default")
        group_of[spec.name] = label
        if label != "default":
            scale_of[label] = float(lr_scales[label])
    return group_of, scale_of


def grouped_optimiser(
    config: GraphModelConfig,
    make_optimiser: Callable[[float], optax.GradientTransformation],
    base_lr: float,
    lr_scales: Optional[Mapping[str, float]] = None,
    target: str = "params",
) -> optax.GradientTransformation:
    """An optimiser whose learning rate depends on the node.

    Args:
        make_optimiser: `lr -> GradientTransformation` (e.g. `optax.adam`, or
            `lambda lr: optax.sgd(lr, momentum=0.9)`).
        base_lr: the learning rate for unmatched nodes.
        lr_scales: rules, see the module docstring.
        target: "params" for a weight optimiser (`optim.init(eqx.filter(model,
            eqx.is_array))`), or "states" for an activity optimiser (applied to the
            `{node name: state}` dict inside settling).
    """
    if target not in ("params", "states"):
        raise ValueError(f"target must be 'params' or 'states', got {target!r}")
    group_of, scale_of = resolve_groups(config, lr_scales)
    transforms = {label: make_optimiser(base_lr * scale) for label, scale in scale_of.items()}

    def node_of(path):
        for key in path:
            if isinstance(key, jax.tree_util.DictKey):
                return key.key
        raise ValueError(f"no node name found in pytree path {path}")

    if target == "params":
        labels = lambda params: jax.tree_util.tree_map_with_path(lambda path, _: group_of[node_of(path)], params)
    else:
        labels = lambda states: {name: group_of[name] for name in states}
    return optax.multi_transform(transforms, labels)
