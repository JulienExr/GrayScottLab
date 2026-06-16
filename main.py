from __future__ import annotations

import argparse
import sys

import pygame

from reaction_diffusion import (
    DEFAULT_COLORMAP,
    GrayScottSimulation,
    PRESET_NAMES,
    PRESETS,
    apply,
    next_colormap,
    next_preset,
)
from reaction_diffusion.controls import AppState, handle_events
from reaction_diffusion.renderer import Renderer
from reaction_diffusion.ui import Button, Panel, Slider
from reaction_diffusion.utils import timestamp_filename

PARAM_SLIDERS = ("F", "k", "Du", "Dv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gray-Scott reaction-diffusion simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --preset coral\n"
            "  python main.py --preset spots --seed 42 --steps 12\n"
            "  python main.py --size 320 --windowed\n"
        ),
    )
    parser.add_argument("--size", type=int, default=256, metavar="N",
                        help="Grid size NxN (default: 256)")
    parser.add_argument("--preset", choices=PRESET_NAMES, default="spots",
                        help="Starting preset (default: spots)")
    parser.add_argument("--steps", type=int, default=None, metavar="N",
                        help="Steps per frame override (default: preset value)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible initialisation")
    parser.add_argument("--windowed", action="store_true",
                        help="Run in a resizable window instead of fullscreen")
    return parser.parse_args()


def build_panel(renderer: Renderer, sim: GrayScottSimulation, state: AppState) -> Panel:
    panel = Panel(
        renderer.panel_rect,
        renderer.title_font,
        renderer.font,
        renderer.small_font,
    )
    p = sim.params
    panel.add_slider("F", Slider("Feed  (F)", 0.0, 0.10, p["F"]))
    panel.add_slider("k", Slider("Kill  (k)", 0.0, 0.10, p["k"]))
    panel.add_slider("Du", Slider("Diffuse U", 0.0, 0.30, p["Du"]))
    panel.add_slider("Dv", Slider("Diffuse V", 0.0, 0.20, p["Dv"]))
    panel.add_slider(
        "spf", Slider("Speed", 1, 40, state.steps_per_frame, integer=True)
    )

    def do_reset() -> None:
        state.reset_requested = True

    def do_pause() -> None:
        state.paused = not state.paused

    def do_preset() -> None:
        state.current_preset = next_preset(state.current_preset)

    def do_colormap() -> None:
        state.current_colormap = next_colormap(state.current_colormap)

    def do_save() -> None:
        state.screenshot_requested = True

    panel.add_button(Button(lambda: f"Preset:  {state.current_preset}", do_preset))
    panel.add_button(Button(lambda: f"Color:   {state.current_colormap}", do_colormap))
    panel.add_button(Button(lambda: "Resume" if state.paused else "Pause", do_pause))
    panel.add_button(Button("Reset", do_reset))
    panel.add_button(Button("Save PNG", do_save))

    panel.info_lines = [
        lambda: PRESETS[state.current_preset]["description"],
        lambda: "LMB paint  ·  RMB erase",
        lambda: "Press  H  for full help",
    ]
    panel.layout()
    return panel


def sync_param_sliders(panel: Panel, sim: GrayScottSimulation) -> None:
    p = sim.params
    for key in PARAM_SLIDERS:
        panel.sliders[key].set_value(p[key])


def main() -> None:
    args = parse_args()
    if args.size < 16:
        print("Error: --size must be >= 16", file=sys.stderr)
        sys.exit(1)

    sim = GrayScottSimulation(size=args.size, preset=args.preset, seed=args.seed)
    renderer = Renderer(size=args.size, fullscreen=not args.windowed)

    state = AppState(
        current_preset=args.preset,
        steps_per_frame=args.steps if args.steps is not None else sim.steps_per_frame,
        current_colormap=DEFAULT_COLORMAP,
    )
    panel = build_panel(renderer, sim, state)

    clock = pygame.time.Clock()
    brush = max(4, args.size // 28)

    while state.running:
        # 1. Collect events; let the UI panel consume its own, pass the rest on.
        remaining: list[pygame.event.Event] = []
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                renderer.resize(event.w, event.h)
                panel.set_rect(renderer.panel_rect)
            elif not panel.handle_event(event):
                remaining.append(event)

        spf_before_slider = panel.sliders["spf"].value
        spf_before_state = state.steps_per_frame
        handle_events(remaining, state, renderer.viewport_tuple, sim.size)

        # 2. Reconcile speed between the slider and keyboard +/- shortcuts.
        if panel.sliders["spf"].value != spf_before_slider:
            state.steps_per_frame = int(panel.sliders["spf"].value)
        elif state.steps_per_frame != spf_before_state:
            panel.sliders["spf"].set_value(state.steps_per_frame)

        # 3. One-shot actions.
        if state.reset_requested:
            sim.reset()
            state.reset_requested = False
        if state.screenshot_requested:
            rgb = apply(sim.V, state.current_colormap)
            renderer.save_screenshot(rgb, str(timestamp_filename("screenshot", "png")))
            state.screenshot_requested = False

        # 4. Preset change → adopt params into sliders + speed.
        if state.current_preset != sim.preset_name:
            sim.set_preset(state.current_preset)
            sync_param_sliders(panel, sim)
            panel.sliders["spf"].set_value(sim.steps_per_frame)
            state.steps_per_frame = sim.steps_per_frame

        # 5. Push slider values into the live model.
        for key in PARAM_SLIDERS:
            sim.set_param(key, panel.sliders[key].value)

        # 6. Mouse painting (before stepping so it shows this frame).
        if state.inject_pos is not None:
            sim.inject(*state.inject_pos, radius=brush)
        if state.erase_pos is not None:
            sim.erase(*state.erase_pos, radius=brush)

        # 7. Advance and render.
        if not state.paused:
            sim.step(n=state.steps_per_frame)

        rgb = apply(sim.V, state.current_colormap)
        renderer.draw(
            rgb=rgb,
            panel=panel,
            fps=clock.get_fps(),
            paused=state.paused,
            step_count=sim.step_count,
            show_help=state.show_help,
        )
        clock.tick(60)

    renderer.close()


if __name__ == "__main__":
    main()
