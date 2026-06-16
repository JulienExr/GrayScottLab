"""Generate showcase artwork for the README.

Runs the simulation headless (pure NumPy, no display needed) and writes:

  outputs/hero.png        a high-res still of a developed pattern
  outputs/evolution.gif   an animated GIF of a pattern growing from the seed

Usage:
    python generate_pics.py

Requires Pillow (installed with the ``dev`` extra: ``pip install -e ".[dev]"``).
"""
from __future__ import annotations

from PIL import Image

from reaction_diffusion import GrayScottSimulation, apply
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


def main() -> None:
    out = ensure_outputs_dir()
    make_still("maze", "inferno", out / "hero.png", steps=9000)
    make_gif(
        "worms", "viridis", out / "evolution.gif",
        warmup=800, frames=64, steps_per_frame=80, upscale=1, duration_ms=70,
    )


if __name__ == "__main__":
    main()
