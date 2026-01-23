import importlib.util as importlib_util

if importlib_util.find_spec("torch") is None:
    msg = 'torch features are not available. Install the required dependencies with: pip install "tinytools[torch]"'
    raise ImportError(msg)

from .modules import ConstantLayer, FFBlock, GatedMLP, LocationHead, VanillaMLP
from .utils import freeze_module, get_zero_safe_values

__all__ = [
    "ConstantLayer",
    "FFBlock",
    "GatedMLP",
    "LocationHead",
    "VanillaMLP",
    "freeze_module",
    "get_zero_safe_values",
]
