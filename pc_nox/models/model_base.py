from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Type
from dataclasses import asdict
import json
from datetime import datetime
import equinox as eqx
import jax.random as jr

import jax.numpy as jnp
import  jax.nn as jnn

# simple name : class mapping, e.g. "tpch" : TpchModel. Registered automatically on init
MODEL_REGISTRY = {}
ACT_FN_REGISTRY = {"tanh": jnp.tanh, "relu": jnn.relu, "identity": lambda x: x}


class ModelBase(ABC):
    """
    Interface for all Model classses.
    """
    SCHEMA_VERSION: ClassVar[int] = 1  # envelope version, defined once here, for saving/loading

    model_type: ClassVar[str]
    config_cls: ClassVar[Type]

    # Check that required attributes config_cls and model_type have been defined in subclass
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # enforce that the child class explicitly defined these attributes
        required_vars = ["config_cls", "model_type"]
        for var in required_vars:
            # check cls.__dict__ to ensure the CHILD defined it 
            # not just inheriting a blank/placeholder value from the parent
            if var not in cls.__dict__:
                raise TypeError(
                    f"Class {cls.__name__} failed to define required ClassVar: '{var}'"
                )

        # prevent duplicate registration
        if cls.model_type in MODEL_REGISTRY and MODEL_REGISTRY[cls.model_type] is not cls:
            raise ValueError(f"model_type '{cls.model_type}' already registered")
        # register model_type
        MODEL_REGISTRY[cls.model_type] = cls

    # @abstract_method
    # def build_activities():
    #     """
    #     Builds or rebuilds activities (latent states).
    #     """
    #     pass

    @abstractmethod
    def predict():
        """
        Runs every layer's `predict` function once, given required args.
        Used by energy function.
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config, *, key):
        """
        (Re)build model from config, returns model.
        """
        pass

    
    # If more 'activities only' methods emerge, consider making an additional class
    @classmethod
    def zero_activities(cls, config) -> Activities:
        """
        Builds empty activies skeleton when implemented by subclass, else raises error.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support resumable activities "
            f"(no zero_activities implementation)."
        )


    def save_checkpoint(self, config, *, path: str | Path | None = None, metadata=None, opt_state=None, activities=None):
        """
        Save checkpoint: Deserialise and save all model config and metadata to dir
        """
        # ensure sure activities skeleton builder is implemented, if relevant, for loading
        if activities is not None:
            # fail fast if this model can't actually reconstruct a matching skeleton later
            type(self).zero_activities(config)  # raises NotImplementedError if unsupported

        created_at = datetime.now()

        # generate dynamic default path if no path is provided
        if path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = Path(f"checkpoints/{timestamp}")
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": self.SCHEMA_VERSION,
            "model_type": self.model_type,
            "created_at": created_at.isoformat(),
            "config": asdict(config),
            "metadata": metadata or {},
            "has_opt_state": opt_state is not None,
            "has_activities": activities is not None,
        }
        (path / "checkpoint.json").write_text(json.dumps(checkpoint, indent=4))
        eqx.tree_serialise_leaves(path / "weights.eqx", self)
        if opt_state is not None:
            eqx.tree_serialise_leaves(path / "opt_state.eqx", opt_state)
        if activities is not None:
            eqx.tree_serialise_leaves(path / "activities.eqx", activities)
        return path


    @classmethod
    def load_checkpoint(cls, path, *, key=None, optim=None, activities_skeleton=None):
        """
        Load checkpoint: Serialise and load all model config and metadata from dir
        """
        path = Path(path)
        checkpoint = json.loads((path / "checkpoint.json").read_text())
        cls = MODEL_REGISTRY[checkpoint["model_type"]]
        config = cls.config_cls(**checkpoint["config"])
        model = eqx.tree_deserialise_leaves(
            path / "weights.eqx", cls.from_config(config, key=key or jr.PRNGKey(0))
        )

        opt_state = None
        if checkpoint["has_opt_state"]:
            if optim is None:
                raise ValueError("checkpoint includes opt_state; pass `optim=` (same optax transform used to train) to load it")
            opt_state = eqx.tree_deserialise_leaves(
                path / "opt_state.eqx", optim.init(eqx.filter(model, eqx.is_array))
            )

        activities = None
        if checkpoint["has_activities"]:
            skeleton = activities_skeleton or cls.zero_activities(config)
            activities = eqx.tree_deserialise_leaves(path / "activities.eqx", skeleton)

        return model, checkpoint["metadata"], opt_state, activities

    
    @classmethod
    def find_latest_checkpoint(cls, root="checkpoints", model_type=None) -> Path:
        root = Path(root)
        candidates = []
        for d in root.iterdir():
            cp_file = d / "checkpoint.json"
            if not (d.is_dir() and cp_file.exists()):
                continue
            checkpoint = json.loads(cp_file.read_text())
            if model_type is not None and checkpoint["model_type"] != model_type:
                continue
            candidates.append((checkpoint["created_at"], d))

        if not candidates:
            raise FileNotFoundError(f"No checkpoints found under {root}" + (f" for model_type={model_type!r}" if model_type else ""))
        candidates.sort(key=lambda pair: pair[0])  # ISO strings sort chronologically as strings too
        return candidates[-1][1]