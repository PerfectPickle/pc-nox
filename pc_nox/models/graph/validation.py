"""graph/validation.py

Stage-2 validation for a `GraphModel`: checks that run AFTER the node
modules are built (so their static capability properties exist) but BEFORE
anything is traced or jitted. Stage 1 (topology, levels, names, shapes)
needs only the specs and lives in `GraphConfig.__post_init__`.

`EnergyStrategy` and the registry live in `energies.py`; this module only
imports the type.
"""

from typing import List, Mapping

from .energies import EnergyStrategy


def validate_pairings(
    config,                                       # GraphModelConfig
    nodes: Mapping[str, object],                  # name -> built NodeBase
    energy_registry: Mapping[str, EnergyStrategy],
) -> None:
    """Check every node spec in `config` against its built node and the
    energy strategy it resolves to (the spec's own `energy`, or the
    config-wide default when the spec leaves it None).

    Collects ALL problems and raises one `ValueError` listing them, each
    naming the node, so a bad config is fixed in one pass rather than one
    error at a time. Pure Python on static values; nothing here is traced.
    """
    errors: List[str] = []
    children = config.children()

    for spec in config.node_specs:
        node = nodes[spec.name]
        energy_name = config.get(spec, "energy")
        where = f"node {spec.name!r} ({type(node).__name__}, energy={energy_name!r})"
        keys = frozenset(node.prediction_keys)

        # --- spec setting vs node capability -------------------------------
        if spec.bidirectional and not node.supports_bottom_up:
            errors.append(f"{where}: bidirectional=True but this node type has no bottom-up path")
        if ("bu" in keys) != spec.bidirectional:
            errors.append(
                f"{where}: spec.bidirectional={spec.bidirectional} disagrees with "
                f"the node's prediction paths {sorted(keys)}"
            )

        # --- node needs vs graph structure ---------------------------------
        if node.needs_children and not children[spec.name]:
            errors.append(f"{where}: needs children (bottom-up path) but no node lists it as a parent")
        if node.state_shape != tuple(spec.shape):
            errors.append(f"{where}: node state_shape {node.state_shape} != spec shape {tuple(spec.shape)}")

        # --- node needs vs the spec's wiring -------------------------------
        if node.needs_parents_curr and not spec.parents:
            errors.append(f"{where}: needs parents but the spec lists none")
        if spec.parents and not node.needs_parents_curr:
            errors.append(f"{where}: spec lists parents {spec.parents} but this node type ignores them")
        if node.needs_parents_prev:
            for p in spec.parents:
                if config.spec(p).clamped:
                    errors.append(
                        f"{where}: needs its parents' previous states but parent {p!r} is clamped "
                        "(clamped nodes keep no previous state)"
                    )
        if spec.clamped and node.needs_own_prev:
            errors.append(f"{where}: is clamped but needs its own previous state (recurrent)")
        if node.needs_control_input and config.input_size == 0:
            errors.append(f"{where}: needs the control input but config.input_size is 0")

        if not spec.clamped and "td" not in keys:
            errors.append(
                f"{where}: an inferred (unclamped) node must return a 'td' prediction, "
                f"but this node returns {sorted(keys) or 'no predictions'}; "
                "a node that only carries data should be clamped=True"
            )

        if spec.release and not keys:
            errors.append(f"{where}: is released (free when unobserved) but predicts nothing in either direction, so it cannot be initialised")

        # --- node outputs vs energy requirements ---------------------------
        strategy = energy_registry.get(energy_name)
        if strategy is None:
            errors.append(f"{where}: unknown energy; registered: {sorted(energy_registry)}")
            continue

        missing = strategy.required_paths - keys
        if missing:
            errors.append(f"{where}: energy requires paths {sorted(missing)} that the node does not return")
        if strategy.allowed_paths is not None:
            unsupported = keys - strategy.allowed_paths
            if unsupported:
                errors.append(
                    f"{where}: node returns paths {sorted(unsupported)} that this energy "
                    f"cannot score (it handles {sorted(strategy.allowed_paths)})"
                )
        if spec.release and not strategy.valid_for_free_state:
            errors.append(
                f"{where}: is released (free when unobserved) but its energy {energy_name!r} is a loss against data, "
                "not a prediction error on a free state; use a squared-error energy such as 'pc'"
            )
        # (a pure-data node with no prediction paths, e.g. InputNode, has nothing to score, so cannot "need" them)
        if strategy.needs_pre_activations and keys and not node.provides_pre_activations:
            errors.append(f"{where}: energy needs pre-activations but the node does not provide them")

    if config.input_size > 0 and not any(n.needs_control_input for n in nodes.values()):
        errors.append(f"config.input_size={config.input_size} but no node consumes the control input")

    if errors:
        raise ValueError("invalid graph:\n  - " + "\n  - ".join(errors))


# GraphModel.__init__ calls this once, right after building the nodes:
#
#     validate_pairings(config, self.nodes, ENERGY_REGISTRY)
