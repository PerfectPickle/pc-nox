from pathlib import Path
from typing import Optional
import json


def find_latest_checkpoint(root: str | Path = "checkpoints", model_type: Optional[str] | None = None) -> Path:
    """
    Returns Path of most recently saved checkpoint in root according to checkpoint.json

    Supported model_type values: 'tpch'

    Args:
        root: Directory to search for checkpoints (which are directories themselves)
        model_type: The plaintext name of the model, which every child of BaseModel must have specified.
    """
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


def load_metadata(path: str | Path) -> dict:
        """
        Return a checkpoint's metadata, without touching weights.eqx,
        opt_state.eqx, or activities.eqx.

        checkpoint.json is small and read on its own; this is here so you
        can inspect metadata (e.g. to look up which optimiser was used)
        before calling load_checkpoint, without paying for a full model
        load or having to load twice:

            metadata = load_metadata(path)
            optim = build_optim(metadata["optim_name"], **metadata["optim_kwargs"])
            checkpoint = TpchModel.load_checkpoint(path, optim=optim)
        """
        path = Path(path)
        checkpoint = json.loads((path / "checkpoint.json").read_text())
        return checkpoint["metadata"]