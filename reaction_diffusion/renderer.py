from __future__ import annotations

import pygame
import numpy as np

from .ui import Panel


class Renderer:
    """Pygame renderer: a large window with the simulation auto-fit on the
    left and a control panel docked on the right."""

    PANEL_WIDTH = 340
    BG = (18, 18, 24)
    VIEWPORT_EDGE = (60, 62, 76)

    _HELP_LINES = [
        "MOUSE",
        "  Left  drag   inject V (paint)",
        "  Right drag   erase  V (clear)",
        "",
        "KEYBOARD",
        "  SPACE  pause / resume",
        "  R      reset simulation",
        "  S      save screenshot",
        "  P      next preset",
        "  C      next colormap",
        "  +/-    sim speed",
        "  H      toggle this help",
        "  ESC    quit",
    ]

    def __init__(
        self,
        size: int,
        fullscreen: bool = True,
        window_size: tuple[int, int] | None = None,
    ) -> None:
        if not pygame.get_init():
            pygame.init()
        pygame.display.set_caption("Gray-Scott Lab")

        if fullscreen:
            desktop = pygame.display.get_desktop_sizes()[0]
            self.screen = pygame.display.set_mode(desktop, pygame.FULLSCREEN)
        else:
            if window_size is None:
                dw, dh = pygame.display.get_desktop_sizes()[0]
                window_size = (min(1366, int(dw * 0.9)), min(840, int(dh * 0.9)))
            self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)

        self.size = size
        self.sim_surface = pygame.Surface((size, size))
        self._init_fonts()
        self.recompute_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _init_fonts(self) -> None:
        def _font(size: int, bold: bool = False) -> pygame.font.Font:
            try:
                return pygame.font.SysFont("monospace", size, bold=bold)
            except Exception:
                return pygame.font.Font(None, size + 2)

        self.title_font = _font(22, bold=True)
        self.font = _font(15)
        self.small_font = _font(13)

    def recompute_layout(self) -> None:
        self.W, self.H = self.screen.get_size()
        pw = min(self.PANEL_WIDTH, self.W // 2)
        self.panel_rect = pygame.Rect(self.W - pw, 0, pw, self.H)

        avail_w = self.W - pw
        avail_h = self.H
        side = max(min(avail_w, avail_h) - 28, 64)
        vx = (avail_w - side) // 2
        vy = (avail_h - side) // 2
        self.viewport = pygame.Rect(vx, vy, side, side)

    def resize(self, w: int, h: int) -> None:
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.recompute_layout()

    @property
    def viewport_tuple(self) -> tuple[int, int, int, int]:
        return (self.viewport.x, self.viewport.y, self.viewport.w, self.viewport.h)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        rgb: np.ndarray,
        panel: Panel,
        fps: float,
        paused: bool,
        step_count: int,
        show_help: bool = False,
    ) -> None:
        self.screen.fill(self.BG)

        # pygame surfarray is column-major [x, y]; NumPy is [row, col].
        pygame.surfarray.blit_array(self.sim_surface, rgb.transpose(1, 0, 2))
        scaled = pygame.transform.smoothscale(
            self.sim_surface, (self.viewport.w, self.viewport.h)
        )
        self.screen.blit(scaled, self.viewport.topleft)
        pygame.draw.rect(self.screen, self.VIEWPORT_EDGE, self.viewport, 1)

        self._draw_hud(fps, paused, step_count)
        panel.draw(self.screen)
        if show_help:
            self._draw_help()

        pygame.display.flip()

    def save_screenshot(self, rgb: np.ndarray, filename: str) -> None:
        surf = pygame.Surface((self.size, self.size))
        pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
        out = max(1024, self.viewport.w)
        surf = pygame.transform.smoothscale(surf, (out, out))
        pygame.image.save(surf, filename)
        print(f"Screenshot saved: {filename}")

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _draw_hud(self, fps: float, paused: bool, step_count: int) -> None:
        lines = [
            (f"FPS {fps:4.0f}", (220, 220, 220)),
            ("PAUSED" if paused else "RUNNING",
             (255, 200, 80) if paused else (120, 230, 120)),
            (f"step {step_count}", (200, 200, 200)),
        ]
        pad, lh, w = 6, 18, 116
        box = pygame.Surface((w, pad * 2 + lh * len(lines)), pygame.SRCALPHA)
        box.fill((0, 0, 0, 140))
        bx, by = self.viewport.x + 8, self.viewport.y + 8
        self.screen.blit(box, (bx, by))
        for i, (text, color) in enumerate(lines):
            surf = self.small_font.render(text, True, color)
            self.screen.blit(surf, (bx + pad, by + pad + i * lh))

    def _draw_help(self) -> None:
        pad, lh = 12, 19
        w = 320
        h = pad * 2 + lh * len(self._HELP_LINES)
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 200))
        px = self.viewport.centerx - w // 2
        py = self.viewport.centery - h // 2
        self.screen.blit(panel, (px, py))
        for i, text in enumerate(self._HELP_LINES):
            color = (140, 180, 255) if text and not text.startswith(" ") else (225, 225, 230)
            surf = self.small_font.render(text, True, color)
            self.screen.blit(surf, (px + pad, py + pad + i * lh))
