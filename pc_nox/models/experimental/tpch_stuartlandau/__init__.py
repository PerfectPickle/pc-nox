"""slpch/__init__.py -- Stuart-Landau tPC-H (SL-tPC-H).

Complex-valued, coupled-oscillator sibling of `tpch`: every layer's state
is a Stuart-Landau (supercritical Hopf) oscillator instead of a
tanh-activated real vector. See `model.py`'s module docstring for the
full design (in particular why energy isn't required to decrease
monotonically during settling here, unlike in TpchModel, and the JAX
complex-gradient convention gotcha `_descent_direction` exists to fix).
"""

from .config import SLPCHConfig
from .layers import ComplexLinear, SLControlLayer, SLHiddenLayer, SLObservationLayer
from .model import SLPCHModel

__all__ = [
    "SLPCHConfig",
    "ComplexLinear",
    "SLControlLayer",
    "SLHiddenLayer",
    "SLObservationLayer",
    "SLtPCHModel",
]
