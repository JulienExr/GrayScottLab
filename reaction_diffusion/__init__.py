from .colormaps import (
    COLORMAP_NAMES,
    COLORMAPS,
    DEFAULT_COLORMAP,
    apply,
    next_colormap,
)
from .presets import PRESET_NAMES, PRESETS, next_preset
from .simulation import GrayScottSimulation

__all__ = [
    "GrayScottSimulation",
    "PRESETS",
    "PRESET_NAMES",
    "next_preset",
    "COLORMAPS",
    "COLORMAP_NAMES",
    "DEFAULT_COLORMAP",
    "apply",
    "next_colormap",
]
