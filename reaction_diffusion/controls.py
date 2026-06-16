from __future__ import annotations

import dataclasses

import pygame

from .colormaps import next_colormap
from .presets import next_preset
from .utils import screen_to_grid


@dataclasses.dataclass
class AppState:
    paused: bool = False
    show_help: bool = False
    current_preset: str = "spots"
    current_colormap: str = "grayscale"
    steps_per_frame: int = 8
    running: bool = True
    screenshot_requested: bool = False
    reset_requested: bool = False
    inject_pos: tuple[int, int] | None = None
    erase_pos: tuple[int, int] | None = None


def handle_events(
    events: list[pygame.event.Event],
    state: AppState,
    viewport: tuple[int, int, int, int],
    size: int,
) -> None:
    """Mutate state in-place from events not already consumed by the UI panel.

    Mouse positions are mapped into grid coordinates relative to the sim
    viewport; clicks outside it (e.g. over the panel) are ignored.
    """
    for event in events:
        if event.type == pygame.QUIT:
            state.running = False

        elif event.type == pygame.KEYDOWN:
            _handle_keydown(event, state)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            grid = screen_to_grid(event.pos, viewport, size)
            if grid is not None:
                if event.button == 1:
                    state.inject_pos = grid
                elif event.button == 3:
                    state.erase_pos = grid

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                state.inject_pos = None
            elif event.button == 3:
                state.erase_pos = None

        elif event.type == pygame.MOUSEMOTION:
            grid = screen_to_grid(event.pos, viewport, size)
            state.inject_pos = grid if (event.buttons[0] and grid) else None
            state.erase_pos = grid if (event.buttons[2] and grid) else None


def _handle_keydown(event: pygame.event.Event, state: AppState) -> None:
    key = event.key
    if key == pygame.K_ESCAPE:
        state.running = False
    elif key == pygame.K_SPACE:
        state.paused = not state.paused
    elif key == pygame.K_r:
        state.reset_requested = True
    elif key == pygame.K_s:
        state.screenshot_requested = True
    elif key == pygame.K_p:
        state.current_preset = next_preset(state.current_preset)
    elif key == pygame.K_h:
        state.show_help = not state.show_help
    elif key == pygame.K_c:
        state.current_colormap = next_colormap(state.current_colormap)
    elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        state.steps_per_frame = min(state.steps_per_frame + 1, 40)
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        state.steps_per_frame = max(state.steps_per_frame - 1, 1)
