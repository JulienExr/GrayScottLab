"""Generate showcase artwork for the README.

Runs the simulation headless (pure NumPy, no display needed) and writes:

  outputs/hero.png        a high-res still of a developed pattern
  outputs/evolution.gif   an animated GIF of a pattern growing from the seed

Usage:
    python generate_pics.py

Requires Pillow (installed with the ``dev`` extra: ``pip install -e ".[dev]"``).
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from reaction_diffusion import GrayScottSimulation, apply, create_simulation
from reaction_diffusion.utils import ensure_outputs_dir

SEED = 7  # fixed so the artwork is reproducible


def _frame(sim: GrayScottSimulation, colormap: str, upscale: int) -> Image.Image:
    rgb = apply(sim.V, colormap)  # (H, W, 3) uint8, fixed-range colours
    img = Image.fromarray(rgb)
    if upscale > 1:
        img = img.resize(
            (img.width * upscale, img.height * upscale), Image.LANCZOS
        )
    return img


def make_still(
    preset: str,
    colormap: str,
    path,
    *,
    size: int = 256,
    steps: int = 8000,
    upscale: int = 2,
) -> None:
    """Run a preset to a mature state and save a single frame."""
    sim = GrayScottSimulation(size=size, preset=preset, seed=SEED)
    sim.step(steps)
    _frame(sim, colormap, upscale).save(path)
    print(f"wrote {path}")


def make_gif(
    preset: str,
    colormap: str,
    path,
    *,
    size: int = 192,
    warmup: int = 600,
    frames: int = 70,
    steps_per_frame: int = 24,
    upscale: int = 2,
    duration_ms: int = 80,
) -> None:
    """Capture the pattern spreading from the central seed into a GIF."""
    sim = GrayScottSimulation(size=size, preset=preset, seed=SEED)
    sim.step(warmup)
    imgs = []
    for _ in range(frames):
        sim.step(steps_per_frame)
        imgs.append(_frame(sim, colormap, upscale))
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"wrote {path} ({len(imgs)} frames)")


# Curated for the montage: presets that develop a rich, frame-filling texture from
# uniform noise. (coral and unstable sit on the extinction boundary -- they only
# sustain as a transient growing front, so they read as mostly-empty stills and are
# better seen live in the app. All seven presets are still listed in the README table.)
_GALLERY_PRESETS = ("spots", "maze", "worms", "cells", "fingerprints")


def make_gallery(
    path,
    *,
    colormap: str = "inferno",
    size: int = 220,
    steps: int = 6000,
    cols: int = 5,
    pad: int = 6,
    label_h: int = 22,
) -> None:
    """Render the showcase presets from uniform noise and tile them into a montage.

    Uses the NumPy backend so it can seed the full domain by writing U/V directly.
    Runs in a few seconds at this size; no GPU required.
    """
    import numpy as np

    tiles: list[Image.Image] = []
    for name in _GALLERY_PRESETS:
        sim = create_simulation(size=size, preset=name, seed=SEED, backend="cpu")
        # Seed ~5% of cells across the whole grid so the texture develops everywhere.
        rng = np.random.default_rng(SEED)
        mask = rng.random((size, size)) < 0.05
        sim.U[mask] = 0.5
        sim.V[mask] = 0.25
        sim.step(steps)
        tile = Image.fromarray(apply(sim.V, colormap)).convert("RGB")
        canvas = Image.new("RGB", (size, size + label_h), (12, 12, 14))
        canvas.paste(tile, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, size + 4), name, fill=(235, 235, 235))
        tiles.append(canvas)

    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    grid = Image.new(
        "RGB",
        (cols * tw + (cols + 1) * pad, rows * th + (rows + 1) * pad),
        (12, 12, 14),
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid.paste(tile, (pad + c * (tw + pad), pad + r * (th + pad)))
    grid.save(path)
    print(f"wrote {path} ({len(tiles)} presets)")


def main() -> None:
    out = ensure_outputs_dir()
    make_still("maze", "inferno", out / "hero.png", steps=9000)
    make_gif(
        "worms", "viridis", out / "evolution.gif",
        warmup=800, frames=64, steps_per_frame=80, upscale=1, duration_ms=70,
    )
    make_gallery(out / "gallery.png")


if __name__ == "__main__":
    main()
