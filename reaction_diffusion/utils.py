from __future__ import annotations

import pathlib
from datetime import datetime


def ensure_outputs_dir() -> pathlib.Path:
    p = pathlib.Path(__file__).parent.parent / "outputs"
    p.mkdir(exist_ok=True)
    return p


def timestamp_filename(prefix: str, ext: str) -> pathlib.Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_outputs_dir() / f"{prefix}_{ts}.{ext}"


def screen_to_grid(
    pos: tuple[int, int],
    viewport: tuple[int, int, int, int],
    size: int,
) -> tuple[int, int] | None:
    """Map a screen pixel to (col, row) grid coords inside the sim viewport.

    `viewport` is (x, y, w, h). Returns None if the position is outside it.
    """
    vx, vy, vw, vh = viewport
    if not (vx <= pos[0] < vx + vw and vy <= pos[1] < vy + vh):
        return None
    gx = int((pos[0] - vx) / vw * size)
    gy = int((pos[1] - vy) / vh * size)
    gx = max(0, min(size - 1, gx))
    gy = max(0, min(size - 1, gy))
    return (gx, gy)
