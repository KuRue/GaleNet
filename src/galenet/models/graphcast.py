"""Wrapper around the official `graphcast` package.

This module no longer implements the lightweight linear fallback used for
testing.  Instead it defers entirely to DeepMind's `graphcast` implementation
which provides a JAX/Flax model for global weather forecasting.  The wrapper
offers a minimal interface used by the rest of GaleNet while providing clear
error messages when the required dependency or checkpoint is missing.

Two public methods are exposed:

``infer``
    Apply the loaded GraphCast model to either a ``numpy.ndarray`` or an
    ``xarray.DataArray`` and return the same type.

``predict``
    Convenience method for running the model autoregressively for a number of
    steps.  The method simply chains calls to :meth:`infer` and returns the
    final array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    import graphcast as dm_graphcast  # type: ignore

    _GRAPHCAST_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    dm_graphcast = None  # type: ignore
    _GRAPHCAST_AVAILABLE = False


ArrayLike = Union[np.ndarray, xr.DataArray]


class GraphCastModel:
    """Thin wrapper around the official GraphCast implementation.

    Parameters
    ----------
    checkpoint_path:
        Path to a checkpoint file readable by ``graphcast.load_checkpoint``.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        if not _GRAPHCAST_AVAILABLE:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'graphcast' package is required for GraphCastModel but was not found"
            )

        path = Path(checkpoint_path)
        if not path.exists():  # pragma: no cover - defensive programming
            raise FileNotFoundError(f"GraphCast checkpoint not found at {path}")

        try:
            self._model = dm_graphcast.load_checkpoint(path)
        except Exception as exc:  # pragma: no cover - checkpoint issues
            raise RuntimeError(f"Failed to load GraphCast checkpoint: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(self, climate: ArrayLike) -> ArrayLike:
        """Run inference on ``climate`` data using the loaded model."""

        if isinstance(climate, xr.DataArray):
            arr = climate.values.astype(np.float32)
            try:
                out = self._model(arr)
            except Exception as exc:  # pragma: no cover - runtime failure
                raise RuntimeError(f"GraphCast inference failed: {exc}") from exc
            return xr.DataArray(
                np.asarray(out, dtype=np.float32),
                coords=climate.coords,
                dims=climate.dims,
            )

        arr = np.asarray(climate, dtype=np.float32)
        try:
            out = self._model(arr)
        except Exception as exc:  # pragma: no cover - runtime failure
            raise RuntimeError(f"GraphCast inference failed: {exc}") from exc
        return np.asarray(out, dtype=np.float32)

    def predict(self, climate: ArrayLike, num_steps: int, step: int) -> ArrayLike:
        """Run autoregressive forecasts for ``num_steps`` iterations."""

        arr: ArrayLike = climate
        for _ in range(num_steps):
            arr = self.infer(arr)
        return arr


__all__ = ["GraphCastModel"]
