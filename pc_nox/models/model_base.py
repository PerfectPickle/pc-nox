from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, ClassVar, Type, Optional, Any, Self, NamedTuple
from jaxtyping import Array
from dataclasses import asdict
import json
from datetime import datetime
import equinox as eqx
import optax
import jax.random as jr
import jax.numpy as jnp
import  jax.nn as jnn

# type aliases
Activities = List[Array]
Predictions = List[Array]

# simple name : class mapping, e.g. "tpch" : TpchModel. Registered automatically on init
MODEL_REGISTRY = {}
ACT_FN_REGISTRY = {"tanh": jnp.tanh, "relu": jnn.relu, "identity": lambda x: x}


# Named tuple for readability
class LoadedCheckpoint(NamedTuple):
    """
    Result of ModelBase.load_checkpoint().

    Attributes:
        model: the reconstructed model, weights restored.
        metadata: whatever dict was passed to save_checkpoint's metadata=.
        opt_state: restored optax state, or None if the checkpoint had none.
            Pass the *same* optax transform used at save time as optim= to load it.
        activities: restored states (e.g. states_prev), or None.
    """
    model: "ModelBase"
    metadata: dict
    opt_state: optax.OptState | None  # param opt state, to be clear
    activities: Activities | None


class ModelBase(ABC):
    """
    Base class for all PCN model variants.

    Subclasses must define `model_type: ClassVar[str]`,
    `config_cls: ClassVar[Type]`, and `self.config`, and implement 
    `predict()` and `from_config()`. Registering a subclass 
    (via __init_subclass__) makes it resolvable by name through 
    MODEL_REGISTRY, which is what lets load_checkpoint() reconstruct 
    the correct subclass from a checkpoint's model_type field alone. 

    Attributes:
        SCHEMA_VERSION: Envelope version, defined once here, for saving/loading.
        model_type: Name of model type for registry usage.
        config_cls: Dataclass for model config, used to interpret saved config.
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


    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Runs every layer's `predict` function once, given required args.
        Used by energy function.
        """
        ...
    
    @classmethod
    @abstractmethod
    def from_config(cls, config, *, key) -> "ModelBase":
        """
        (Re)build model from config, returns model.

        Args:
            config: Instance of this class's config_cls.
            key: PRNGKey for parameter initialisation.
        """
        ...

    
    # If more 'activities only' methods emerge, consider making an additional class
    @classmethod
    def zero_activities(cls, config) -> Activities:
        """
        Builds empty activies skeleton when implemented by subclass, else raises error.

        Args:
            config: Instance of this class's config_cls.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support resumable activities "
            f"(no zero_activities implementation)."
        )


    # --- config vs. metadata ---
    # config:   anything that changes what the model COMPUTES. If reloading
    #           without this setting would give different energies/gradients/
    #           predictions on identical weights, it belongs here (layer sizes,
    #           act_fn, regularization coefficients, architectural switches).
    #           Lives on the model itself (static field) and is restored
    #           automatically on load -- the caller never has to know it existed.
    # metadata: anything that describes THIS RUN rather than the model itself
    #           (epoch, dataset name, which optimizer/lr produced this
    #           opt_state, git commit, free-text notes). The model has no way
    #           to know any of this on its own -- only the training loop does,
    #           at the moment save_checkpoint() is called. Freeform, caller-
    #           supplied, never affects what the model computes.
    def save_checkpoint(self, *, path: str | Path | None = None, config=None, metadata=None, opt_state=None, activities=None) -> Path:
        """
        Save model checkpoint.

        Args:
            config: Dataclass object containing all information necessary to reconstruct the model, such as variables describing network shape.
            path: Directory to save checkpoint files into.
            metadata: Dictionary containing any other relevant information, such as optim type, learning rates, last frame processed, env type, etc.
            opt_state: OptState of Optax parameter optimiser.
            activities: Latent states, for smooth resumption of inference in temporal models.
        """
        config = config if config is not None else getattr(self, "config", None)
        if config is None:
            raise ValueError(
                f"{type(self).__name__} has no self.config and none was passed explicitly. "
                "Store the config used in from_config() as self.config, or pass config= here."
            )

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
    def load_checkpoint(cls, path, *, key=None, optim=None, activities_skeleton=None)-> LoadedCheckpoint:
        """
        Load a checkpoint saved by save_checkpoint().

        Returns a LoadedCheckpoint with attributes:
            model:      the reconstructed model, weights restored, config
                        restored automatically.
            metadata:   whatever dict was passed to save_checkpoint's metadata=.
            opt_state:  restored optax state, or None if none was saved.
                        Pass the SAME optax transform used at save time as
                        optim= to restore it correctly.
            activities: restored state estimates (e.g. states_prev to resume
                        settling from), or None.
        """
        path = Path(path)
        checkpoint = json.loads((path / "checkpoint.json").read_text())
        cls = MODEL_REGISTRY[checkpoint["model_type"]]
        config = cls.config_cls(**checkpoint["config"])
        # from_config effectively restores self.config via model initialisation. deserialise only overwrites dynamic leaves on skeleton, self.config is static.
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

        return LoadedCheckpoint(model, checkpoint["metadata"], opt_state, activities)
