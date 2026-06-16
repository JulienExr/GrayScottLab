import numpy as np
import pytest

from reaction_diffusion.simulation import GrayScottSimulation


def test_init_shape():
    sim = GrayScottSimulation(64)
    assert sim.U.shape == (64, 64)
    assert sim.V.shape == (64, 64)


def test_init_values_range():
    sim = GrayScottSimulation(64)
    assert sim.U.min() >= 0.0
    assert sim.U.max() <= 1.0
    assert sim.V.min() >= 0.0
    assert sim.V.max() <= 1.0


def test_step_preserves_shape():
    sim = GrayScottSimulation(64)
    sim.step(1)
    assert sim.U.shape == (64, 64)
    assert sim.V.shape == (64, 64)


def test_step_values_in_range():
    sim = GrayScottSimulation(64, seed=0)
    sim.step(200)
    assert sim.U.min() >= 0.0
    assert sim.U.max() <= 1.0
    assert sim.V.min() >= 0.0
    assert sim.V.max() <= 1.0


def test_step_count_increments():
    sim = GrayScottSimulation(64)
    sim.step(5)
    assert sim.step_count == 5
    sim.step(3)
    assert sim.step_count == 8


def test_inject_sets_v():
    sim = GrayScottSimulation(64)
    sim.inject(32, 32, radius=1)
    assert sim.V[32, 32] == 1.0
    assert sim.U[32, 32] == 0.0


def test_erase_restores():
    sim = GrayScottSimulation(64)
    sim.inject(32, 32, radius=5)
    sim.erase(32, 32, radius=5)
    assert sim.V[32, 32] == 0.0
    assert sim.U[32, 32] == 1.0


def test_reset_clears_step_count():
    sim = GrayScottSimulation(64)
    sim.step(10)
    sim.reset()
    assert sim.step_count == 0


def test_reset_seed_reproducible():
    sim = GrayScottSimulation(64)
    sim.reset(seed=42)
    U1 = sim.U.copy()
    V1 = sim.V.copy()
    sim.reset(seed=42)
    assert np.array_equal(sim.U, U1)
    assert np.array_equal(sim.V, V1)


def test_set_preset_changes_params():
    sim = GrayScottSimulation(64, preset="spots")
    original_F = sim.params["F"]
    sim.set_preset("maze")
    assert sim.preset_name == "maze"
    assert sim.params["F"] != original_F


def test_set_preset_preserves_fields():
    sim = GrayScottSimulation(64, seed=0)
    sim.step(10)
    U_before = sim.U.copy()
    V_before = sim.V.copy()
    sim.set_preset("coral")
    assert np.array_equal(sim.U, U_before)
    assert np.array_equal(sim.V, V_before)


def test_set_param_updates_value():
    sim = GrayScottSimulation(64, preset="spots")
    sim.set_param("F", 0.042)
    sim.set_param("k", 0.061)
    assert sim.params["F"] == 0.042
    assert sim.params["k"] == 0.061


def test_set_param_steps_is_int():
    sim = GrayScottSimulation(64)
    sim.set_param("steps_per_frame", 12.0)
    assert sim.params["steps_per_frame"] == 12
    assert isinstance(sim.params["steps_per_frame"], int)


def test_set_param_rejects_unknown_key():
    sim = GrayScottSimulation(64)
    with pytest.raises(KeyError):
        sim.set_param("nonsense", 1.0)


def test_set_param_affects_step():
    """Changing F/k live should change the trajectory."""
    a = GrayScottSimulation(48, preset="spots", seed=3)
    b = GrayScottSimulation(48, preset="spots", seed=3)
    b.set_param("F", 0.02)
    b.set_param("k", 0.05)
    a.step(30)
    b.step(30)
    assert not np.array_equal(a.V, b.V)


def test_all_presets_simulate_without_error():
    """Smoke test: every preset runs 50 steps on a small grid."""
    from reaction_diffusion.presets import PRESET_NAMES
    for name in PRESET_NAMES:
        sim = GrayScottSimulation(32, preset=name, seed=0)
        sim.step(50)
        assert 0.0 <= sim.U.min() and sim.U.max() <= 1.0
