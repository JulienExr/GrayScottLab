from .colormaps import (
    COLORMAP_NAMES,
    COLORMAPS,
    DEFAULT_COLORMAP,
    apply,
    next_colormap,
)
from .presets import PRESET_NAMES, PRESETS, next_preset
from .simulation import GrayScottSimulation


def create_simulation(
    size: int = 256,
    preset: str = "spots",
    seed: int | None = None,
    *,
    backend: str = "cpu",
):
    """Build a simulation on the requested backend.

    ``backend="cpu"`` returns the pure-NumPy :class:`GrayScottSimulation` (the default,
    no extra dependencies). ``backend="gpu"`` returns the Taichi-accelerated backend,
    which fuses the whole step into one GPU kernel; it requires the ``gpu`` extra
    (``pip install -e ".[gpu]"``). Both classes share the same public API.
    """
    if backend == "cpu":
        return GrayScottSimulation(size=size, preset=preset, seed=seed)
    if backend == "gpu":
        from .simulation_taichi import TaichiGrayScottSimulation

        return TaichiGrayScottSimulation(size=size, preset=preset, seed=seed)
    raise ValueError(f"Unknown backend {backend!r} (expected 'cpu' or 'gpu')")


__all__ = [
    "GrayScottSimulation",
    "create_simulation",
    "PRESETS",
    "PRESET_NAMES",
    "next_preset",
    "COLORMAPS",
    "COLORMAP_NAMES",
    "DEFAULT_COLORMAP",
    "apply",
    "next_colormap",
]
