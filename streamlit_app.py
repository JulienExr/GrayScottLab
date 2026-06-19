"""Streamlit front-end for the Gray-Scott simulation.

A browser playground that reuses the headless NumPy core (no pygame): tweak
the parameters, pick a preset/colormap, and watch the pattern evolve live.

Run locally:
    pip install -e ".[web]"
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from reaction_diffusion import (
    DEFAULT_COLORMAP,
    COLORMAP_NAMES,
    GrayScottSimulation,
    apply,
)
from reaction_diffusion.presets import PRESETS, PRESET_NAMES

GRID_SIZES = [128, 160, 192, 224, 256]
PARAM_KEYS = ("F", "k", "Du", "Dv")

st.set_page_config(page_title="Gray-Scott Lab", page_icon="🧬", layout="wide")


def get_sim(size: int, preset: str) -> GrayScottSimulation:
    """Fetch the simulation from session state, rebuilding it on size change."""
    sim = st.session_state.get("sim")
    if sim is None or sim.size != size:
        sim = GrayScottSimulation(size=size, preset=preset)
        st.session_state.sim = sim
    return sim


def adopt_preset(preset: str) -> None:
    """Load a preset's parameters into the slider widgets (via their keys)."""
    p = PRESETS[preset]
    for key in PARAM_KEYS:
        st.session_state[key] = p[key]
    st.session_state.speed = p["steps_per_frame"]


# --- Sidebar controls -------------------------------------------------
with st.sidebar:
    st.title("🧬 Gray-Scott Lab")
    st.caption("Reaction-diffusion playground")

    preset = st.selectbox("Preset", PRESET_NAMES, index=0)
    # When the preset changes, refresh the sliders to match it.
    if st.session_state.get("_preset") != preset:
        st.session_state._preset = preset
        adopt_preset(preset)

    size = st.select_slider("Grid size", GRID_SIZES, value=192)
    cmap = st.selectbox(
        "Colormap", COLORMAP_NAMES, index=COLORMAP_NAMES.index(DEFAULT_COLORMAP)
    )

    st.divider()
    F = st.slider("Feed (F)", 0.0, 0.10, key="F", step=0.0005, format="%.4f")
    k = st.slider("Kill (k)", 0.0, 0.10, key="k", step=0.0005, format="%.4f")
    Du = st.slider("Diffuse U", 0.0, 0.30, key="Du", step=0.001, format="%.3f")
    Dv = st.slider("Diffuse V", 0.0, 0.20, key="Dv", step=0.001, format="%.3f")
    speed = st.slider("Steps / frame", 1, 40, key="speed")

    st.divider()
    running = st.toggle("▶ Run", value=True)
    c1, c2 = st.columns(2)
    reset = c1.button("↻ Reset", use_container_width=True)
    seed = c2.button("✚ Seed center", use_container_width=True)

# --- Apply control actions -------------------------------------------
sim = get_sim(size, preset)
for key in PARAM_KEYS:
    sim.set_param(key, st.session_state[key])

if reset:
    sim.reset()
if seed:
    sim.inject(size // 2, size // 2, radius=max(4, size // 12))

# --- Live render fragment --------------------------------------------
st.subheader(f"{preset} · {cmap}")
frame_slot = st.empty()
caption_slot = st.empty()


def _draw() -> None:
    rgb = apply(sim.V, cmap)
    factor = max(1, 512 // size)  # nearest-neighbour upscale for crisp pixels
    if factor > 1:
        rgb = np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)
    frame_slot.image(rgb)
    caption_slot.caption(f"step {sim.step_count}")


@st.fragment(run_every=0.05 if running else None)
def live() -> None:
    if running:
        sim.step(speed)
    _draw()


live()
