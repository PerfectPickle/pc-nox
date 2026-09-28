from typing import Callable
import jax.numpy as jnp
import jax.nn as jnn


ACT_FN_REGISTRY: dict[str, Callable]= {
    "tanh": jnp.tanh, 
    "relu": jnn.relu, 
    "identity": lambda x: x}