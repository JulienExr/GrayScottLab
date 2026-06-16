from __future__ import annotations

import numpy as np

from .presets import PRESETS


class GrayScottSimulation:
    """Gray-Scott reaction-diffusion model on a periodic 2D grid.

    ∂U/∂t = Du·∇²U - U·V² + F·(1 - U)
    ∂V/∂t = Dv·∇²V + U·V² - (F + k)·V
    """

    def __init__(
        self,
        size: int = 256,
        preset: str = "spots",
        seed: int | None = None,
    ) -> None:
        self.size = size
        self.preset_name = preset
        self._params: dict = PRESETS[preset].copy()
        self.U: np.ndarray = np.empty((size, size), dtype=np.float64)
        self.V: np.ndarray = np.empty((size, size), dtype=np.float64)
        self.step_count: int = 0
        self._rng = np.random.default_rng(seed)
        self.reset(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Reinitialize the grid with a noisy seed square at the center."""
        self.U[:] = 1.0
        self.V[:] = 0.0
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        r = max(self.size // 10, 4)
        cx = cy = self.size // 2
        sl = (slice(cy - r, cy + r), slice(cx - r, cx + r))
        h, w = 2 * r, 2 * r
        self.U[sl] = 0.50 + self._rng.uniform(-0.05, 0.05, (h, w))
        self.V[sl] = 0.25 + self._rng.uniform(-0.05, 0.05, (h, w))
        self.step_count = 0

    def set_preset(self, preset: str) -> None:
        """Switch to a different preset without resetting U/V (allows live morphing)."""
        self.preset_name = preset
        self._params = PRESETS[preset].copy()

    def set_param(self, key: str, value: float) -> None:
        """Update a single model parameter live (used by the UI sliders)."""
        if key in ("Du", "Dv", "F", "k"):
            self._params[key] = float(value)
        elif key == "steps_per_frame":
            self._params[key] = int(value)
        else:
            raise KeyError(f"Unknown parameter: {key!r}")

    def step(self, n: int = 1) -> None:
        """Advance the simulation by n Euler steps."""
        Du = self._params["Du"]
        Dv = self._params["Dv"]
        F = self._params["F"]
        k = self._params["k"]
        for _ in range(n):
            Lu = self._laplacian(self.U)
            Lv = self._laplacian(self.V)
            uvv = self.U * self.V * self.V
            self.U += Du * Lu - uvv + F * (1.0 - self.U)
            self.V += Dv * Lv + uvv - (F + k) * self.V
            np.clip(self.U, 0.0, 1.0, out=self.U)
            np.clip(self.V, 0.0, 1.0, out=self.V)
        self.step_count += n

    def inject(self, cx: int, cy: int, radius: int = 10) -> None:
        """Set V=1, U=0 in a circle — seeds new pattern growth."""
        Y, X = np.ogrid[: self.size, : self.size]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
        self.V[mask] = 1.0
        self.U[mask] = 0.0

    def erase(self, cx: int, cy: int, radius: int = 10) -> None:
        """Set V=0, U=1 in a circle — restores the trivial fixed point."""
        Y, X = np.ogrid[: self.size, : self.size]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
        self.V[mask] = 0.0
        self.U[mask] = 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict:
        return self._params.copy()

    @property
    def steps_per_frame(self) -> int:
        return self._params["steps_per_frame"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _laplacian(self, Z: np.ndarray) -> np.ndarray:
        """Weighted 3×3 stencil via np.roll (periodic boundary).

        Weights: center=-1, cardinal=0.20, diagonal=0.05
        Sum = -1 + 4×0.20 + 4×0.05 = 0 ✓
        """
        N = np.roll(Z, -1, axis=0)
        S = np.roll(Z, 1, axis=0)
        E = np.roll(Z, 1, axis=1)
        W = np.roll(Z, -1, axis=1)
        return (
            -1.0 * Z
            + 0.20 * (N + S + E + W)
            + 0.05 * (
                np.roll(N, 1, axis=1)   # NE
                + np.roll(N, -1, axis=1)  # NW
                + np.roll(S, 1, axis=1)   # SE
                + np.roll(S, -1, axis=1)  # SW
            )
        )
