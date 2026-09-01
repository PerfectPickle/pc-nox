from pathlib import Path
from typing import Optional 

def find_latest_checkpoint(root: str | Path = "checkpoints", model_type: Optional[str] | None = None) -> Path:
    """
    Returns Path of most recently saved checkpoint in root according to checkpoint.json

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