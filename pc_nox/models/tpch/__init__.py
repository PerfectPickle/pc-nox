"""tpch/__init__.py

Re-exports tPC-H's public surface at package level, so existing code that
did `from models.tpch import TpchModel, TpchConfig, make_train_step` against
the old single-file `tpch.py` keeps working unchanged against this package.

`make_eval_step`/`make_train_run`/etc. now live in `runners_temporal.py`
(model-agnostic, shared with other temporal variants) rather than in this
package -- re-exported here for the same backward-compatibility reason.
New code that also wants to drive a *different* temporal variant with the
same runners should import them from `models.runners_temporal` directly.
"""

from ..runners_temporal import (
    make_eval_run,
    make_eval_run_diffrax,
    make_eval_step,
    make_eval_step_diffrax,
    make_train_run,
    make_train_run_diffrax,
    make_train_step,
    make_train_step_diffrax,
)
from .config import TpchConfig
from .layers import TpchControlLayer, TpchHiddenLayer, TpchObservationLayer
from .model import TpchModel

__all__ = [
    "TpchConfig",
    "TpchControlLayer",
    "TpchHiddenLayer",
    "TpchObservationLayer",
    "TpchModel",
    "make_eval_step",
    "make_eval_run",
    "make_train_step",
    "make_train_run",
    "make_train_step_diffrax",
    "make_train_run_diffrax",
    "make_eval_step_diffrax",
    "make_eval_run_diffrax",
]
