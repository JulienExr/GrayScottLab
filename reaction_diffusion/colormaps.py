"""Colormaps mapping a concentration field to RGB.

Each colormap is a 256-entry lookup table (LUT) of shape (256, 3) uint8.
Mapping a normalised field is then a single fast array-index operation.

Vibrant perceptual colormaps are sourced from matplotlib when available;
a set of hand-coded analytic colormaps is always present as a fallback.
"""
from __future__ import annotations

import numpy as np

_RAMP = np.linspace(0.0, 1.0, 256)

# The V concentration sits in roughly this band for the stable Gray-Scott
# regimes. Mapping against a *fixed* range (rather than each frame's own
# min/max) keeps the colours steady instead of "breathing" as the field
# evolves. Pass vmin/vmax to apply() to override.
DISPLAY_VMIN = 0.0
DISPLAY_VMAX = 0.4


def normalize(
    V: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Scale V into [0, 1]. With no bounds, uses the field's own min/max."""
    if vmin is None:
        vmin = float(V.min())
    if vmax is None:
        vmax = float(V.max())
    n = (V - vmin) / (vmax - vmin + 1e-8)
    return np.clip(n, 0.0, 1.0)


# ----------------------------------------------------------------------
# Analytic colormaps defined on a 1D ramp t∈[0,1] → (N, 3) float in [0,1]
# ----------------------------------------------------------------------

def _grayscale(t: np.ndarray) -> np.ndarray:
    return np.stack([t, t, t], axis=-1)


def _fire(t: np.ndarray) -> np.ndarray:
    r = np.clip(t * 2, 0, 1)
    g = np.clip(t * 3 - 1, 0, 1)
    b = np.clip(t * 10 - 9, 0, 1)
    return np.stack([r, g, b], axis=-1)


def _ocean(t: np.ndarray) -> np.ndarray:
    r = np.clip(t * 2 - 1, 0, 1) * 0.25
    g = np.clip(t * 1.5, 0, 1) * 0.70
    b = 0.55 + t * 0.45
    return np.stack([r, g, b], axis=-1)


def _neon(t: np.ndarray) -> np.ndarray:
    r = np.sin(t * np.pi)
    g = np.clip(t * 2, 0, 1)
    b = (1.0 - np.clip(t * 2 - 1, 0, 1)) * 0.78 + 0.22
    return np.stack([r, g, b], axis=-1)


def _toxic(t: np.ndarray) -> np.ndarray:
    r = np.clip(t * 3 - 2, 0, 1) * 0.70
    g = np.clip(t * 1.2, 0, 1) * 0.90
    b = np.clip(t * 5 - 4, 0, 1) * 0.40
    return np.stack([r, g, b], axis=-1)


def _lut_from_fn(fn) -> np.ndarray:
    rgb = np.clip(fn(_RAMP), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# LUT registry
# ----------------------------------------------------------------------

LUTS: dict[str, np.ndarray] = {}


def _register_custom() -> None:
    LUTS["fire"] = _lut_from_fn(_fire)
    LUTS["ocean"] = _lut_from_fn(_ocean)
    LUTS["neon"] = _lut_from_fn(_neon)
    LUTS["toxic"] = _lut_from_fn(_toxic)
    LUTS["grayscale"] = _lut_from_fn(_grayscale)


def _register_matplotlib() -> None:
    try:
        import matplotlib
    except Exception:
        return
    pairs = [
        ("inferno", "inferno"),
        ("magma", "magma"),
        ("plasma", "plasma"),
        ("viridis", "viridis"),
        ("turbo", "turbo"),
        ("twilight", "twilight_shifted"),
    ]
    for key, mpl_name in pairs:
        try:
            cmap = matplotlib.colormaps[mpl_name]
            rgba = np.asarray(cmap(_RAMP))
            LUTS[key] = (rgba[:, :3] * 255).astype(np.uint8)
        except Exception:
            continue


_register_custom()
_register_matplotlib()

# Display order: colourful perceptual maps first, then the analytic ones.
_PREFERRED = [
    "inferno", "magma", "plasma", "viridis", "turbo", "twilight",
    "fire", "neon", "toxic", "ocean", "grayscale",
]
COLORMAP_NAMES: list[str] = [n for n in _PREFERRED if n in LUTS]

# Backwards-compatible alias (was a name -> function dict).
COLORMAPS = LUTS

DEFAULT_COLORMAP = "inferno" if "inferno" in LUTS else "fire"


def next_colormap(current: str) -> str:
    if current not in COLORMAP_NAMES:
        return COLORMAP_NAMES[0]
    idx = COLORMAP_NAMES.index(current)
    return COLORMAP_NAMES[(idx + 1) % len(COLORMAP_NAMES)]


def apply(
    V: np.ndarray,
    name: str,
    vmin: float = DISPLAY_VMIN,
    vmax: float = DISPLAY_VMAX,
) -> np.ndarray:
    """Normalise V and map it through the named colormap → (H, W, 3) uint8.

    By default V is mapped against the fixed [DISPLAY_VMIN, DISPLAY_VMAX]
    band so colours stay stable frame to frame.
    """
    lut = LUTS.get(name)
    if lut is None:
        lut = LUTS[COLORMAP_NAMES[0]]
    v = normalize(V, vmin, vmax)
    idx = (v * 255).astype(np.uint8)
    return lut[idx]
