"""Tests for the Taichi GPU backend.

Skipped entirely when Taichi is not installed (it is an optional 'gpu' extra).
Taichi auto-selects a backend, so these also pass on CPU-only CI machines.
"""
import numpy as np
import pytest

pytest.importorskip("taichi")

from reaction_diffusion import create_simulation  # noqa: E402
from reaction_diffusion.simulation import GrayScottSimulation  # noqa: E402
from reaction_diffusion.simulation_taichi import TaichiGrayScottSimulation  # noqa: E402


def _make(size=64, preset="coral", seed=1):
    # fp64 so it can be compared against the float64 NumPy core to tight tolerance.
    return TaichiGrayScottSimulation(size, preset=preset, seed=seed, fp64=True)


def test_init_shape_and_dtype():
    g = _make()
    assert g.U.shape == (64, 64)
    assert g.V.shape == (64, 64)


def test_values_in_range_after_steps():
    g = _make(seed=0)
    g.step(200)
    assert g.U.min() >= 0.0 and g.U.max() <= 1.0
    assert g.V.min() >= 0.0 and g.V.max() <= 1.0


def test_step_count_increments():
    g = _make()
    g.step(5)
    g.step(3)
    assert g.step_count == 8


def test_matches_numpy_core():
    """From identical initial state, GPU (fp64) must track the NumPy core closely."""
    g = _make(preset="coral", seed=1)
    c = GrayScottSimulation(64, preset="coral", seed=1)
    # Force identical initial conditions on both backends.
    c.U[:] = g.U
    c.V[:] = g.V
    g.step(25)
    c.step(25)
    assert np.allclose(g.U, c.U, atol=1e-9)
    assert np.allclose(g.V, c.V, atol=1e-9)


def test_inject_and_erase():
    g = _make()
    g.inject(32, 32, radius=2)
    assert g.V[32, 32] == 1.0
    assert g.U[32, 32] == 0.0
    g.erase(32, 32, radius=2)
    assert g.V[32, 32] == 0.0
    assert g.U[32, 32] == 1.0


def test_reset_clears_step_count():
    g = _make()
    g.step(10)
    g.reset()
    assert g.step_count == 0


def test_set_param_affects_trajectory():
    a = _make(preset="spots", seed=3)
    b = _make(preset="spots", seed=3)
    b.set_param("F", 0.02)
    b.set_param("k", 0.05)
    a.step(30)
    b.step(30)
    assert not np.array_equal(a.V, b.V)


def test_set_param_rejects_unknown_key():
    g = _make()
    with pytest.raises(KeyError):
        g.set_param("nonsense", 1.0)


def test_factory_returns_gpu_backend():
    sim = create_simulation(32, preset="spots", backend="gpu")
    assert isinstance(sim, TaichiGrayScottSimulation)


def test_factory_cpu_is_numpy():
    sim = create_simulation(32, preset="spots", backend="cpu")
    assert isinstance(sim, GrayScottSimulation)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        create_simulation(32, backend="quantum")
