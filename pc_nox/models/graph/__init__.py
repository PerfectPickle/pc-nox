"""Graph-structured temporal predictive coding: named nodes, partial
connectivity (tails), mixed dense/conv nodes. See `model.py`."""

from .config import GraphModelConfig, NodeSpec
from .model import GraphModel

__all__ = ["GraphModel", "GraphModelConfig", "NodeSpec"]
