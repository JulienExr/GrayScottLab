import pytest

from reaction_diffusion.presets import PRESETS, PRESET_NAMES, next_preset

REQUIRED_KEYS = {"Du", "Dv", "F", "k", "steps_per_frame", "description"}


def test_all_presets_have_required_keys():
    for name, params in PRESETS.items():
        for key in REQUIRED_KEYS:
            assert key in params, f"Preset {name!r} missing key {key!r}"


def test_preset_names_are_non_empty_strings():
    for name in PRESETS:
        assert isinstance(name, str) and len(name) > 0


def test_preset_names_constant_matches_dict_keys():
    assert PRESET_NAMES == list(PRESETS.keys())


def test_next_preset_advances():
    first = PRESET_NAMES[0]
    second = PRESET_NAMES[1]
    assert next_preset(first) == second


def test_next_preset_wraps_at_end():
    last = PRESET_NAMES[-1]
    first = PRESET_NAMES[0]
    assert next_preset(last) == first


def test_param_values_in_physical_range():
    for name, p in PRESETS.items():
        assert 0 < p["F"] < 0.1, f"{name}: F={p['F']} out of range (0, 0.1)"
        assert 0 < p["k"] < 0.1, f"{name}: k={p['k']} out of range (0, 0.1)"
        assert 0 < p["Du"] < 1.0, f"{name}: Du={p['Du']} out of range (0, 1)"
        assert 0 < p["Dv"] < 1.0, f"{name}: Dv={p['Dv']} out of range (0, 1)"
        assert p["Dv"] < p["Du"], f"{name}: Dv >= Du (autocatalyst must diffuse slower)"
        assert p["steps_per_frame"] >= 1, f"{name}: steps_per_frame must be >= 1"


def test_description_is_non_empty_string():
    for name, p in PRESETS.items():
        assert isinstance(p["description"], str) and len(p["description"]) > 0, \
            f"{name}: description must be a non-empty string"


def test_preset_count():
    assert len(PRESETS) == 7


def test_full_cycle_returns_to_start():
    current = PRESET_NAMES[0]
    for _ in range(len(PRESET_NAMES)):
        current = next_preset(current)
    assert current == PRESET_NAMES[0]
