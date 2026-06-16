import numpy as np
import pytest

from reaction_diffusion.colormaps import (
    COLORMAP_NAMES,
    DEFAULT_COLORMAP,
    apply,
    next_colormap,
    normalize,
)


def test_at_least_custom_colormaps_present():
    # The hand-coded colormaps must always exist, even without matplotlib.
    for name in ("fire", "ocean", "neon", "toxic", "grayscale"):
        assert name in COLORMAP_NAMES


def test_default_colormap_is_registered():
    assert DEFAULT_COLORMAP in COLORMAP_NAMES


def test_apply_returns_rgb_uint8():
    V = np.random.rand(32, 48)
    for name in COLORMAP_NAMES:
        rgb = apply(V, name)
        assert rgb.shape == (32, 48, 3)
        assert rgb.dtype == np.uint8


def test_apply_unknown_name_falls_back():
    V = np.random.rand(16, 16)
    rgb = apply(V, "does-not-exist")
    assert rgb.shape == (16, 16, 3)


def test_normalize_range():
    V = np.array([[1.0, 2.0], [3.0, 5.0]])
    n = normalize(V)
    assert n.min() >= 0.0
    assert n.max() <= 1.0


def test_normalize_uniform_field_is_safe():
    V = np.full((8, 8), 0.5)
    n = normalize(V)  # must not divide by zero
    assert np.all(np.isfinite(n))


def test_next_colormap_cycles():
    current = COLORMAP_NAMES[0]
    seen = set()
    for _ in range(len(COLORMAP_NAMES)):
        seen.add(current)
        current = next_colormap(current)
    assert current == COLORMAP_NAMES[0]
    assert len(seen) == len(COLORMAP_NAMES)
