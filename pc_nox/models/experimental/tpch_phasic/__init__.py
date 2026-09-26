"""kpch/__init__.py -- Kuramoto tPC-H (phase-only, no intrinsic dynamics).

Every layer's state is a real angle theta, constrained to the unit
circle by construction (no amplitude channel, no oscillator dynamics --
see model.py's module docstring for the full design and how it differs
from slpch).
"""

from .config import KPCHConfig
from .layers import ComplexLinear, PhaseControlLayer, PhaseHiddenLayer, PhaseObservationLayer
from .model import KPCHModel

__all__ = [
    "KPCHConfig",
    "ComplexLinear",
    "PhaseControlLayer",
    "PhaseHiddenLayer",
    "PhaseObservationLayer",
    "KPCHModel",
]
