"""Lightweight Pygame UI widgets: Slider, Button and a Panel container.

The panel hosts the interactive controls (parameter sliders + action
buttons) drawn in a sidebar next to the simulation. It is deliberately
dependency-free — just pygame primitives — to keep the project lean.
"""
from __future__ import annotations

from typing import Callable

import pygame

# --- Theme ------------------------------------------------------------
PANEL_BG = (30, 31, 40)
PANEL_EDGE = (52, 54, 68)
TRACK = (60, 62, 76)
ACCENT = (96, 176, 255)
HANDLE = (240, 240, 245)
TEXT = (224, 224, 232)
LABEL = (200, 202, 214)
MUTED = (150, 152, 166)
BTN = (52, 54, 68)
BTN_HOVER = (72, 76, 96)
BTN_EDGE = (92, 96, 116)

Label = str | Callable[[], str]


def _resolve(label: Label) -> str:
    return label() if callable(label) else label


class Slider:
    """A horizontal slider mapping a track rect to a value range."""

    def __init__(
        self,
        label: str,
        vmin: float,
        vmax: float,
        value: float,
        *,
        integer: bool = False,
        fmt: str = "{:.4f}",
    ) -> None:
        self.label = label
        self.vmin = vmin
        self.vmax = vmax
        self.value = float(value)
        self.integer = integer
        self.fmt = fmt
        self.rect = pygame.Rect(0, 0, 0, 0)  # track rect, assigned by layout
        self._drag = False

    # -- value helpers --
    def set_value(self, v: float) -> None:
        v = max(self.vmin, min(self.vmax, v))
        self.value = round(v) if self.integer else v

    @property
    def _t(self) -> float:
        if self.vmax == self.vmin:
            return 0.0
        return (self.value - self.vmin) / (self.vmax - self.vmin)

    def _value_from_x(self, x: int) -> float:
        t = (x - self.rect.x) / max(self.rect.w, 1)
        t = max(0.0, min(1.0, t))
        v = self.vmin + t * (self.vmax - self.vmin)
        return round(v) if self.integer else v

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.rect.inflate(16, 22).collidepoint(pos)

    # -- events --
    def handle_event(self, event: pygame.event.Event) -> bool:
        old = self.value
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hit(event.pos):
                self._drag = True
                self.value = self._value_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag = False
        elif event.type == pygame.MOUSEMOTION and self._drag:
            self.value = self._value_from_x(event.pos[0])
        return self.value != old

    # -- draw --
    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        lab = font.render(self.label, True, LABEL)
        surface.blit(lab, (self.rect.x, self.rect.y - 21))
        val_str = str(int(self.value)) if self.integer else self.fmt.format(self.value)
        val = font.render(val_str, True, ACCENT)
        surface.blit(val, (self.rect.right - val.get_width(), self.rect.y - 21))

        track = pygame.Rect(self.rect.x, self.rect.centery - 2, self.rect.w, 4)
        pygame.draw.rect(surface, TRACK, track, border_radius=2)
        fill_w = int(self.rect.w * self._t)
        fill = pygame.Rect(self.rect.x, self.rect.centery - 2, fill_w, 4)
        pygame.draw.rect(surface, ACCENT, fill, border_radius=2)

        hx = self.rect.x + fill_w
        pygame.draw.circle(surface, HANDLE, (hx, self.rect.centery), 8)
        pygame.draw.circle(surface, ACCENT, (hx, self.rect.centery), 8, 2)


class Button:
    """A clickable button; label may be static or a callable for live text."""

    def __init__(self, label: Label, callback: Callable[[], None]) -> None:
        self.label = label
        self.callback = callback
        self.rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(surface, BTN_HOVER if hover else BTN, self.rect, border_radius=6)
        pygame.draw.rect(surface, BTN_EDGE, self.rect, 1, border_radius=6)
        txt = font.render(_resolve(self.label), True, TEXT)
        surface.blit(
            txt,
            (self.rect.centerx - txt.get_width() // 2,
             self.rect.centery - txt.get_height() // 2),
        )


class Panel:
    """Sidebar container that lays out, dispatches events to, and draws widgets."""

    def __init__(
        self,
        rect: pygame.Rect,
        title_font: pygame.font.Font,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        pad: int = 18,
    ) -> None:
        self.rect = rect
        self.title_font = title_font
        self.font = font
        self.small_font = small_font
        self.pad = pad
        self.sliders: dict[str, Slider] = {}
        self.buttons: list[Button] = []
        self.info_lines: list[Callable[[], str]] = []
        self._active: Slider | None = None
        self._info_y = rect.y

    # -- construction --
    def add_slider(self, key: str, slider: Slider) -> Slider:
        self.sliders[key] = slider
        return slider

    def add_button(self, button: Button) -> Button:
        self.buttons.append(button)
        return button

    def set_rect(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self.layout()

    def layout(self) -> None:
        x = self.rect.x + self.pad
        w = self.rect.w - 2 * self.pad
        y = self.rect.y + 92
        for slider in self.sliders.values():
            slider.rect = pygame.Rect(x, y, w, 6)
            y += 46
        y += 10
        for button in self.buttons:
            button.rect = pygame.Rect(x, y, w, 36)
            y += 44
        self._info_y = y + 8

    # -- events --
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if the event was consumed by the panel."""
        if event.type == pygame.MOUSEMOTION:
            if self._active is not None:
                self._active.handle_event(event)
                return True
            return self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEBUTTONUP:
            if self._active is not None:
                self._active.handle_event(event)
                self._active = None
                return True
            return self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if not self.rect.collidepoint(event.pos):
                return False
            if event.button == 1:
                for slider in self.sliders.values():
                    if slider.hit(event.pos):
                        slider.handle_event(event)
                        self._active = slider
                        return True
                for button in self.buttons:
                    if button.rect.collidepoint(event.pos):
                        button.callback()
                        return True
            return True  # swallow any click landing on the panel

        return False

    # -- draw --
    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.rect)
        pygame.draw.line(surface, PANEL_EDGE, self.rect.topleft, self.rect.bottomleft, 1)

        title = self.title_font.render("Gray-Scott Lab", True, (242, 242, 250))
        surface.blit(title, (self.rect.x + self.pad, self.rect.y + 24))
        sub = self.small_font.render("reaction-diffusion playground", True, MUTED)
        surface.blit(sub, (self.rect.x + self.pad, self.rect.y + 54))

        for slider in self.sliders.values():
            slider.draw(surface, self.font)
        for button in self.buttons:
            button.draw(surface, self.font)

        y = self._info_y
        for fn in self.info_lines:
            txt = self.small_font.render(fn(), True, MUTED)
            surface.blit(txt, (self.rect.x + self.pad, y))
            y += 19
