"""Lightweight Pygame UI widgets: Slider, Button and a Panel container.

The panel hosts the interactive controls (parameter sliders + action
buttons) drawn in a sidebar next to the simulation. It is deliberately
dependency-free — just pygame primitives — to keep the project lean.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
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

# Per-preset accent colours, reused as region tints on the phase map.
PRESET_COLORS: dict[str, tuple[int, int, int]] = {
    "spots": (90, 170, 255),
    "maze": (120, 220, 150),
    "coral": (255, 150, 90),
    "worms": (235, 120, 200),
    "cells": (180, 140, 255),
    "unstable": (255, 95, 95),
    "fingerprints": (240, 200, 90),
}

Label = str | Callable[[], str]


def _resolve(label: Label) -> str:
    return label() if callable(label) else label


def _clamp01(t: float) -> float:
    return 0.0 if t < 0.0 else 1.0 if t > 1.0 else t


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


class PhaseMap:
    """A clickable (F, k) phase diagram of the Gray-Scott parameter space.

    The background is a Voronoi tiling: every point is tinted by the colour of the
    nearest preset, so the map reads as labelled "pattern regions". A crosshair marks
    the current (F, k); clicking or dragging anywhere sets new values. The map covers
    the interesting window of parameter space, not the full slider range.
    """

    def __init__(
        self,
        points: list[tuple[str, float, float]],
        f_range: tuple[float, float] = (0.0, 0.09),
        k_range: tuple[float, float] = (0.03, 0.07),
    ) -> None:
        self.points = points  # (preset_name, F, k)
        self.f_min, self.f_max = f_range
        self.k_min, self.k_max = k_range
        self.rect = pygame.Rect(0, 0, 0, 0)
        self._bg: pygame.Surface | None = None
        self._bg_size: tuple[int, int] | None = None

    # -- coordinate mapping (k → x across, F → y with high F at the top) --
    def value_at(self, pos: tuple[int, int]) -> tuple[float, float]:
        tk = _clamp01((pos[0] - self.rect.x) / max(self.rect.w, 1))
        tf = _clamp01((pos[1] - self.rect.y) / max(self.rect.h, 1))
        k = self.k_min + tk * (self.k_max - self.k_min)
        f = self.f_max - tf * (self.f_max - self.f_min)
        return f, k

    def _screen(self, f: float, k: float) -> tuple[int, int]:
        tk = _clamp01((k - self.k_min) / (self.k_max - self.k_min or 1))
        tf = _clamp01((self.f_max - f) / (self.f_max - self.f_min or 1))
        return self.rect.x + int(tk * self.rect.w), self.rect.y + int(tf * self.rect.h)

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)

    # -- background (Voronoi region tiles), built once per size --
    def _build_bg(self) -> pygame.Surface:
        w, h = self.rect.w, self.rect.h
        ks = np.linspace(self.k_min, self.k_max, w, dtype=np.float32)
        fs = np.linspace(self.f_max, self.f_min, h, dtype=np.float32)  # top = high F
        K, F = np.meshgrid(ks, fs)
        kspan = (self.k_max - self.k_min) or 1.0
        fspan = (self.f_max - self.f_min) or 1.0

        dmin = np.full((h, w), np.inf, dtype=np.float32)
        idx = np.zeros((h, w), dtype=np.int32)
        palette = []
        for i, (name, pf, pk) in enumerate(self.points):
            d = ((K - pk) / kspan) ** 2 + ((F - pf) / fspan) ** 2
            closer = d < dmin
            dmin = np.where(closer, d, dmin)
            idx = np.where(closer, i, idx)
            palette.append(PRESET_COLORS.get(name, (160, 160, 160)))

        rgb = np.array(palette, dtype=np.float32)[idx]          # (h, w, 3)
        glow = 0.30 + 0.50 * np.exp(-14.0 * dmin)               # brighten near centres
        rgb = np.clip(rgb * glow[..., None], 0, 255).astype(np.uint8)
        return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

    # -- draw --
    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        cur_f: float,
        cur_k: float,
    ) -> None:
        if self.rect.w <= 0 or self.rect.h <= 0:
            return
        if self._bg is None or self._bg_size != (self.rect.w, self.rect.h):
            self._bg = self._build_bg()
            self._bg_size = (self.rect.w, self.rect.h)

        cap = small_font.render("Phase map — click to set F, k", True, LABEL)
        surface.blit(cap, (self.rect.x, self.rect.y - 18))

        surface.blit(self._bg, self.rect.topleft)
        pygame.draw.rect(surface, PANEL_EDGE, self.rect, 1)

        # Preset dots.
        for name, pf, pk in self.points:
            px, py = self._screen(pf, pk)
            pygame.draw.circle(surface, (12, 12, 16), (px, py), 4)
            pygame.draw.circle(surface, PRESET_COLORS.get(name, HANDLE), (px, py), 3)

        # Current (F, k) crosshair + ring.
        mx, my = self._screen(cur_f, cur_k)
        pygame.draw.line(surface, (255, 255, 255), (self.rect.x, my), (self.rect.right, my), 1)
        pygame.draw.line(surface, (255, 255, 255), (mx, self.rect.y), (mx, self.rect.bottom), 1)
        pygame.draw.circle(surface, (255, 255, 255), (mx, my), 6, 2)
        pygame.draw.circle(surface, ACCENT, (mx, my), 3)

        # Axis labels.
        kx = small_font.render(f"k {self.k_min:.02f}–{self.k_max:.02f} →", True, MUTED)
        surface.blit(kx, (self.rect.x, self.rect.bottom + 3))
        fy = small_font.render(f"↑ F {self.f_min:.02f}–{self.f_max:.02f}", True, MUTED)
        surface.blit(fy, (self.rect.x, self.rect.y + 2))


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
        self.phase_map: PhaseMap | None = None
        self._active: Slider | None = None
        self._active_phase = False
        self._info_y = rect.y

    # -- construction --
    def add_slider(self, key: str, slider: Slider) -> Slider:
        self.sliders[key] = slider
        return slider

    def add_button(self, button: Button) -> Button:
        self.buttons.append(button)
        return button

    def set_phase_map(self, phase_map: PhaseMap) -> PhaseMap:
        self.phase_map = phase_map
        return phase_map

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
        y = self._info_y + len(self.info_lines) * 19 + 14

        # Phase map fills the remaining width as a square, capped by the space left
        # in the panel (it simply shrinks on short windows).
        if self.phase_map is not None:
            top = y + 20  # room for the caption line above the map
            avail_h = (self.rect.bottom - self.pad) - top - 18  # 18 for axis label
            side = max(0, min(w, avail_h))
            self.phase_map.rect = pygame.Rect(x, top, side, side)

    # -- events --
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if the event was consumed by the panel."""
        if event.type == pygame.MOUSEMOTION:
            if self._active is not None:
                self._active.handle_event(event)
                return True
            if self._active_phase:
                self._apply_phase(event.pos)
                return True
            return self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEBUTTONUP:
            if self._active is not None:
                self._active.handle_event(event)
                self._active = None
                return True
            if self._active_phase:
                self._active_phase = False
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
                if self.phase_map is not None and self.phase_map.hit(event.pos):
                    self._active_phase = True
                    self._apply_phase(event.pos)
                    return True
                for button in self.buttons:
                    if button.rect.collidepoint(event.pos):
                        button.callback()
                        return True
            return True  # swallow any click landing on the panel

        return False

    def _apply_phase(self, pos: tuple[int, int]) -> None:
        """Push a phase-map click/drag into the F and k sliders."""
        if self.phase_map is None:
            return
        f, k = self.phase_map.value_at(pos)
        if "F" in self.sliders:
            self.sliders["F"].set_value(f)
        if "k" in self.sliders:
            self.sliders["k"].set_value(k)

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

        if self.phase_map is not None and "F" in self.sliders and "k" in self.sliders:
            self.phase_map.draw(
                surface, self.font, self.small_font,
                self.sliders["F"].value, self.sliders["k"].value,
            )
