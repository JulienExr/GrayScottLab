"""Tests for UI widgets that carry logic (PhaseMap coordinate mapping + Panel wiring).

Pure-geometry tests need no display; the Panel integration test runs under the dummy
SDL driver so it works headlessly in CI.
"""
import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from reaction_diffusion.ui import Panel, PhaseMap, Slider  # noqa: E402

POINTS = [
    ("spots", 0.0365, 0.0600),
    ("coral", 0.0580, 0.0650),
    ("cells", 0.0260, 0.0510),
]


def _phase_map(rect=(0, 0, 100, 200)):
    pm = PhaseMap(POINTS, f_range=(0.0, 0.10), k_range=(0.04, 0.07))
    pm.rect = pygame.Rect(*rect)
    return pm


def test_value_at_top_left_is_high_f_low_k():
    pm = _phase_map()
    f, k = pm.value_at((pm.rect.x, pm.rect.y))  # top-left
    assert f == pytest.approx(pm.f_max)
    assert k == pytest.approx(pm.k_min)


def test_value_at_bottom_right_is_low_f_high_k():
    pm = _phase_map()
    f, k = pm.value_at((pm.rect.right, pm.rect.bottom))
    assert f == pytest.approx(pm.f_min)
    assert k == pytest.approx(pm.k_max)


def test_value_at_clamps_outside():
    pm = _phase_map()
    f, k = pm.value_at((pm.rect.x - 500, pm.rect.y - 500))
    assert pm.f_min <= f <= pm.f_max
    assert pm.k_min <= k <= pm.k_max


def test_screen_value_roundtrip():
    pm = _phase_map()
    for f, k in [(0.05, 0.05), (0.02, 0.045), (0.09, 0.068)]:
        x, y = pm._screen(f, k)
        f2, k2 = pm.value_at((x, y))
        assert f2 == pytest.approx(f, abs=5e-4)
        assert k2 == pytest.approx(k, abs=5e-4)


def test_hit():
    pm = _phase_map((10, 10, 80, 80))
    assert pm.hit((50, 50))
    assert not pm.hit((200, 200))


@pytest.fixture(scope="module")
def _pygame():
    pygame.init()
    pygame.display.set_mode((640, 480))
    yield
    pygame.quit()


def test_phase_click_updates_fk_sliders(_pygame):
    font = pygame.font.Font(None, 16)
    panel = Panel(pygame.Rect(0, 0, 340, 480), font, font, font)
    panel.add_slider("F", Slider("F", 0.0, 0.10, 0.04))
    panel.add_slider("k", Slider("k", 0.0, 0.10, 0.06))
    panel.set_phase_map(PhaseMap(POINTS, f_range=(0.0, 0.10), k_range=(0.04, 0.07)))
    panel.layout()

    pm = panel.phase_map
    px, py = pm._screen(0.058, 0.065)  # over the coral dot
    consumed = panel.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(px, py))
    )
    assert consumed
    assert panel.sliders["F"].value == pytest.approx(0.058, abs=2e-3)
    assert panel.sliders["k"].value == pytest.approx(0.065, abs=2e-3)
