import re

from reaction_diffusion.utils import screen_to_grid, timestamp_filename


# ----------------------------------------------------------------------
# screen_to_grid
# ----------------------------------------------------------------------

def test_maps_top_left_corner():
    assert screen_to_grid((0, 0), (0, 0, 100, 100), 50) == (0, 0)


def test_maps_center():
    assert screen_to_grid((50, 50), (0, 0, 100, 100), 50) == (25, 25)


def test_respects_viewport_offset():
    # A viewport not anchored at the origin (sim is centered next to the panel).
    assert screen_to_grid((200, 200), (200, 200, 100, 100), 50) == (0, 0)


def test_returns_none_outside_viewport():
    vp = (200, 100, 100, 100)
    assert screen_to_grid((0, 0), vp, 50) is None        # left of viewport
    assert screen_to_grid((350, 150), vp, 50) is None     # right (e.g. panel)
    assert screen_to_grid((250, 50), vp, 50) is None      # above
    assert screen_to_grid((250, 250), vp, 50) is None     # below


def test_result_is_clamped_in_bounds():
    # The far edge maps to size-1 at most, never out of range.
    gx, gy = screen_to_grid((99, 99), (0, 0, 100, 100), 50)
    assert 0 <= gx < 50 and 0 <= gy < 50


# ----------------------------------------------------------------------
# timestamp_filename
# ----------------------------------------------------------------------

def test_timestamp_filename_format():
    path = timestamp_filename("screenshot", "png")
    assert path.parent.name == "outputs"
    assert re.fullmatch(r"screenshot_\d{8}_\d{6}\.png", path.name)
