"""GPU-accelerated Gray-Scott backend built on Taichi.

This mirrors the public API of :class:`reaction_diffusion.simulation.GrayScottSimulation`
but runs the hot path -- the Laplacian stencil plus the reaction/clip update -- as a
single fused Taichi kernel. One kernel launch per step (instead of NumPy's ~30 array
ops) is what lets 1024x1024 grids run at 60 fps.

Taichi picks the fastest available backend at init: CUDA on an NVIDIA GPU, Vulkan on
an Intel/AMD iGPU, and CPU as a last resort -- so this runs on virtually any machine.
The import is deliberately isolated in this module: ``import reaction_diffusion`` never
requires Taichi; only constructing a GPU simulation does.

Interactive edits (reset/inject/erase) round-trip through NumPy for clarity -- they are
rare, mouse-driven events, not part of the steady-state loop, so the host<->device copy
is irrelevant to performance.

Note: this module deliberately does *not* use ``from __future__ import annotations``.
PEP 563 string annotations would turn the kernel's ``ti.template()`` parameter hints
into strings, which Taichi cannot parse. Runtime ``X | None`` hints are fine on the
Python 3.10+ this project targets.
"""
import numpy as np

from .presets import PRESETS

# Stencil weights (must match simulation.py's NumPy core exactly).
_CENTER = -1.0
_CARD = 0.20
_DIAG = 0.05

# Taichi's runtime is process-global: ``ti.init`` resets it and wipes every existing
# field. We therefore init at most once per (arch, precision) and let multiple
# simulations share that runtime, so constructing a second sim never clobbers the first.
_RUNTIME: tuple | None = None


def _ensure_runtime(ti, arch, default_fp) -> None:
    global _RUNTIME
    key = (arch, default_fp)
    if _RUNTIME != key:
        ti.init(arch=arch, default_fp=default_fp, random_seed=0)
        _RUNTIME = key


class TaichiGrayScottSimulation:
    """Gray-Scott reaction-diffusion on the GPU via Taichi (NumPy-compatible API)."""

    def __init__(
        self,
        size: int = 256,
        preset: str = "spots",
        seed: int | None = None,
        *,
        arch: str = "gpu",
        fp64: bool = False,
    ) -> None:
        import taichi as ti

        self._ti = ti
        self._dtype = ti.f64 if fp64 else ti.f32
        self._np_dtype = np.float64 if fp64 else np.float32

        arch_map = {
            "gpu": ti.gpu,
            "cuda": ti.cuda,
            "vulkan": ti.vulkan,
            "cpu": ti.cpu,
        }
        _ensure_runtime(ti, arch_map.get(arch, ti.gpu), self._dtype)
        # Record what we actually landed on (ti.init falls back gpu -> cpu silently).
        self.backend = str(ti.lang.impl.current_cfg().arch).rsplit(".", 1)[-1]

        self.size = size
        self.preset_name = preset
        self._params: dict = PRESETS[preset].copy()
        self.step_count = 0
        self._rng = np.random.default_rng(seed)

        # Double-buffered fields: read from index `_cur`, write to the other, then swap.
        self._U = [ti.field(self._dtype, shape=(size, size)) for _ in range(2)]
        self._V = [ti.field(self._dtype, shape=(size, size)) for _ in range(2)]
        self._cur = 0

        self._build_kernel()
        self.reset(seed)

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def _build_kernel(self) -> None:
        ti = self._ti
        n = self.size
        fp = self._dtype  # scalar args match the field dtype (avoids precision warnings)

        @ti.kernel
        def _step_kernel(
            U: ti.template(), V: ti.template(),
            Un: ti.template(), Vn: ti.template(),
            Du: fp, Dv: fp, F: fp, k: fp,
        ):
            for i, j in U:
                # Periodic neighbours (toroidal wrap).
                ip, im = (i + 1) % n, (i - 1 + n) % n
                jp, jm = (j + 1) % n, (j - 1 + n) % n

                u, v = U[i, j], V[i, j]
                lap_u = (
                    _CENTER * u
                    + _CARD * (U[im, j] + U[ip, j] + U[i, jm] + U[i, jp])
                    + _DIAG * (U[im, jm] + U[im, jp] + U[ip, jm] + U[ip, jp])
                )
                lap_v = (
                    _CENTER * v
                    + _CARD * (V[im, j] + V[ip, j] + V[i, jm] + V[i, jp])
                    + _DIAG * (V[im, jm] + V[im, jp] + V[ip, jm] + V[ip, jp])
                )
                uvv = u * v * v
                nu = u + Du * lap_u - uvv + F * (1.0 - u)
                nv = v + Dv * lap_v + uvv - (F + k) * v
                Un[i, j] = ti.math.clamp(nu, 0.0, 1.0)
                Vn[i, j] = ti.math.clamp(nv, 0.0, 1.0)

        self._step_kernel = _step_kernel

    # ------------------------------------------------------------------
    # Public API (mirrors GrayScottSimulation)
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Reinitialise the grid with a noisy seed square at the centre."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        U = np.ones((self.size, self.size), dtype=self._np_dtype)
        V = np.zeros((self.size, self.size), dtype=self._np_dtype)
        r = max(self.size // 10, 4)
        c = self.size // 2
        sl = (slice(c - r, c + r), slice(c - r, c + r))
        h = w = 2 * r
        U[sl] = 0.50 + self._rng.uniform(-0.05, 0.05, (h, w))
        V[sl] = 0.25 + self._rng.uniform(-0.05, 0.05, (h, w))
        self._upload(U, V)
        self.step_count = 0

    def set_preset(self, preset: str) -> None:
        self.preset_name = preset
        self._params = PRESETS[preset].copy()

    def set_param(self, key: str, value: float) -> None:
        if key in ("Du", "Dv", "F", "k"):
            self._params[key] = float(value)
        elif key == "steps_per_frame":
            self._params[key] = int(value)
        else:
            raise KeyError(f"Unknown parameter: {key!r}")

    def step(self, n: int = 1) -> None:
        """Advance n Euler steps; one fused kernel launch each, ping-ponging buffers."""
        Du = self._params["Du"]
        Dv = self._params["Dv"]
        F = self._params["F"]
        k = self._params["k"]
        for _ in range(n):
            cur, nxt = self._cur, 1 - self._cur
            self._step_kernel(
                self._U[cur], self._V[cur], self._U[nxt], self._V[nxt],
                Du, Dv, F, k,
            )
            self._cur = nxt
        self.step_count += n

    def inject(self, cx: int, cy: int, radius: int = 10) -> None:
        """Set V=1, U=0 in a circle -- seeds new pattern growth."""
        U, V = self._download()
        mask = self._disk_mask(cx, cy, radius)
        V[mask] = 1.0
        U[mask] = 0.0
        self._upload(U, V)

    def erase(self, cx: int, cy: int, radius: int = 10) -> None:
        """Set V=0, U=1 in a circle -- restores the trivial fixed point."""
        U, V = self._download()
        mask = self._disk_mask(cx, cy, radius)
        V[mask] = 0.0
        U[mask] = 1.0
        self._upload(U, V)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def U(self) -> np.ndarray:
        return self._U[self._cur].to_numpy()

    @property
    def V(self) -> np.ndarray:
        return self._V[self._cur].to_numpy()

    @property
    def params(self) -> dict:
        return self._params.copy()

    @property
    def steps_per_frame(self) -> int:
        return self._params["steps_per_frame"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _disk_mask(self, cx: int, cy: int, radius: int) -> np.ndarray:
        Y, X = np.ogrid[: self.size, : self.size]
        return (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2

    def _upload(self, U: np.ndarray, V: np.ndarray) -> None:
        self._U[self._cur].from_numpy(U.astype(self._np_dtype))
        self._V[self._cur].from_numpy(V.astype(self._np_dtype))

    def _download(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._U[self._cur].to_numpy().astype(np.float64),
            self._V[self._cur].to_numpy().astype(np.float64),
        )
