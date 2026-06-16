# Gray-Scott Lab

An interactive real-time simulation of **Gray-Scott reaction-diffusion patterns** — a class of Turing instabilities that generate the spots, stripes, and labyrinthine shapes found throughout nature.

![screenshot placeholder](outputs/)

---

## What is this?

Gray-Scott Lab is a Python application that simulates two chemicals (U and V) diffusing and reacting on a 2D grid in real time. By adjusting just two parameters (F and k), the system self-organises into strikingly different patterns: leopard spots, coral branches, fingerprint ridges, cellular structures, and more.

The simulation runs at 60 fps in a fullscreen window with an **interactive control panel** (drag sliders to tune the model live), mouse injection, vibrant colormaps, and live preset switching.

### Highlights

- **Live control panel** — drag sliders for Feed (F), Kill (k) and the two diffusion rates and watch the pattern morph instantly; buttons to cycle presets/colormaps, pause, reset, and save.
- **Fullscreen by default** — the simulation auto-fits the screen next to the panel (use `--windowed` for a resizable window).
- **Vibrant colormaps** — perceptual maps (inferno, magma, plasma, viridis, turbo, twilight) via matplotlib, plus hand-coded ones (fire, neon, toxic, ocean, grayscale).
- **Paint with the mouse** — left-drag to inject the V chemical, right-drag to erase.

---

## Mathematical Background

The Gray-Scott model is a system of two coupled reaction-diffusion PDEs:

```
∂U/∂t = Du · ∇²U  −  U·V²  +  F·(1 − U)
∂V/∂t = Dv · ∇²V  +  U·V²  −  (F + k)·V
```

**Intuition:**
- `U` is a "food" chemical, continuously supplied at rate `F`.
- `V` is an "activator" that consumes U autocatalytically (`U + 2V → 3V`) and decays at rate `F + k`.
- The two species diffuse at different rates: `Du > Dv`, meaning U spreads faster than V.

**The Laplacian `∇²`** is approximated on a discrete grid with a weighted 3×3 stencil:

```
[ 0.05  0.20  0.05 ]
[ 0.20  -1.0  0.20 ]
[ 0.05  0.20  0.05 ]
```

This stencil sums to zero (a valid Laplacian) and uses periodic boundary conditions (the grid wraps toroidally).

**Parameters F and k:**

| Parameter | Role | Effect of increasing |
|---|---|---|
| `F` (feed rate) | Rate at which U is replenished | More energy → larger, more active patterns |
| `k` (kill rate) | Rate at which V is removed | Higher k kills V faster → patterns shrink or disappear |

Small changes in (F, k) produce qualitatively different morphologies. The parameter space has been mapped empirically and the presets correspond to well-known stable regions.

**Connection to Turing patterns and morphogenesis:**

Alan Turing proposed in 1952 (*"The Chemical Basis of Morphogenesis"*) that pairs of diffusing chemicals — one activating itself and inhibiting the other, the other doing the reverse — could spontaneously break spatial symmetry and form stable periodic patterns. This is now called a **Turing instability**.

Gray-Scott is one of the cleanest examples: V activates itself (`U·V²` term), U is the inhibitor that gets consumed. The crucial ingredient is the **diffusion ratio** `Du/Dv > 1`: the inhibitor must diffuse faster than the activator, which prevents any local dominance from spreading too fast. The interplay between reaction and differential diffusion drives the system away from the uniform state and into the rich pattern space you see here.

These mechanisms appear in real biology: skin pigmentation (zebrafish stripes), hair follicle spacing, digit formation in embryos, and seashell patterns are all believed to involve Turing-like instabilities.

---

## Installation

```bash
git clone https://github.com/your-username/GrayScottLab.git
cd GrayScottLab
pip install -e ".[dev]"
```

Or without editable install:
```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, pygame 2.5+, numpy 1.24+

---

## Usage

```bash
python main.py
```

The app launches **fullscreen** by default. Press `ESC` to quit.

**CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--size N` | 256 | Grid size N×N |
| `--preset NAME` | spots | Starting preset |
| `--steps N` | preset default | Simulation steps per frame |
| `--seed N` | random | Random seed for reproducibility |
| `--windowed` | off | Run in a resizable window instead of fullscreen |

**Examples:**
```bash
python main.py --preset coral
python main.py --preset fingerprints --size 320
python main.py --preset spots --seed 42 --windowed
```

---

## Controls

Everything is reachable from the on-screen **control panel** (right side), but
keyboard and mouse shortcuts are available too:

| Control | Action |
|---|---|
| **Sliders** | Tune Feed (F), Kill (k), Diffuse U/V, and Speed live |
| **Buttons** | Cycle preset / colormap, Pause, Reset, Save PNG |
| `SPACE` | Pause / resume |
| `R` | Reset simulation |
| `S` | Save screenshot to `outputs/` |
| `P` | Cycle to next preset |
| `C` | Cycle to next colormap |
| `+` / `-` | Increase / decrease steps per frame |
| `H` | Toggle on-screen help overlay |
| `ESC` | Quit |
| **Left click** (hold/drag) | Inject chemical V (seed new pattern) |
| **Right click** (hold/drag) | Erase V, restore U (clear region) |

> Editing a slider changes the model immediately without resetting the grid, so
> you can watch one pattern continuously morph into another. Press **Reset** (or
> `R`) to re-seed the grid; your slider values are kept.

---

## Presets

| Name | F | k | Character |
|---|---|---|---|
| `spots` | 0.0365 | 0.0600 | Isolated circular spots |
| `maze` | 0.0290 | 0.0570 | Labyrinthine maze channels |
| `coral` | 0.0580 | 0.0650 | Branching coral structures |
| `worms` | 0.0390 | 0.0580 | Squiggly worm-like patterns |
| `cells` | 0.0260 | 0.0510 | Cell division dynamics |
| `unstable` | 0.0620 | 0.0609 | Chaotic, constantly shifting |
| `fingerprints` | 0.0600 | 0.0625 | Dense parallel ridges |

All presets use `Du=0.2097, Dv=0.1050` except `fingerprints` (`Du=0.19, Dv=0.05`).

---

## Colormaps

Cycle them live with `C` or the **Color** button.

Perceptual (via matplotlib, fall back gracefully if not installed):
- **inferno**, **magma**, **plasma**, **viridis** — smooth perceptual gradients
- **turbo** — high-contrast rainbow
- **twilight** — cyclic blue↔red

Hand-coded:
- **fire** — black → red → yellow → white
- **neon** — magenta → cyan → bright
- **toxic** — black → acid green
- **ocean** — deep blue → teal → white
- **grayscale** — clean, neutral, scientific

---

## Project Structure

```
GrayScottLab/
├── main.py                    Entry point + game loop
├── reaction_diffusion/
│   ├── simulation.py          GrayScottSimulation class (NumPy core)
│   ├── presets.py             Preset dictionary + helpers
│   ├── renderer.py            Pygame renderer, viewport + HUD
│   ├── ui.py                  Slider / Button / Panel widgets
│   ├── colormaps.py           LUT colormaps (V → RGB)
│   ├── controls.py            AppState + event handling
│   └── utils.py               File I/O + coordinate helpers
├── tests/
│   ├── test_simulation.py
│   ├── test_presets.py
│   └── test_colormaps.py
└── outputs/                   Screenshots saved here
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Ideas for Future Improvements

1. **GPU acceleration with Taichi or CuPy** — the Laplacian convolution is embarrassingly parallel; a GPU would allow 1024×1024 grids at 60 fps.

2. **GIF / video export** — accumulate frames in a ring buffer and write them to an animated GIF (imageio) or MP4 (ffmpeg) on demand, capturing the pattern evolution over time.

3. **Adjustable brush + seed shapes** — a brush-size slider and circle/line/image seed masks for more expressive initial conditions and painting.

4. **Parameter-space map** — overlay the current (F, k) on the empirical Gray-Scott phase diagram so you can navigate by clicking regions.

5. **Multispecies models** — extend to 3-species systems (e.g., Oregonator, Brusselator) to access richer pattern classes including spirals and travelling waves.

---

## License

MIT
