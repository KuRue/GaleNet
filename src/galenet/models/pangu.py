"""Wrapper around the official `panguweather` package.

This module mirrors :mod:`graphcast`'s wrapper to provide a minimal interface
for the Pangu-Weather model.  The actual heavy lifting is delegated to the
external ``panguweather`` implementation.  ``PanguModel`` offers two public
methods:

``infer``
    Apply the loaded Pangu model to either a ``numpy.ndarray`` or an
    ``xarray.DataArray`` and return the same type.

``predict``
    Convenience method for running the model autoregressively for a number of
    steps.  The method chains calls to :meth:`infer` and returns the final
    array.
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
    import panguweather as dm_pangu  # type: ignore

    _PANGU_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    dm_pangu = None  # type: ignore
    _PANGU_AVAILABLE = False

ArrayLike = Union[np.ndarray, xr.DataArray]


class PanguModel:
    """Thin wrapper around the official Pangu-Weather implementation.

    Parameters
    ----------
    checkpoint_path:
        Path to a checkpoint file readable by ``panguweather``.  The wrapper
        performs minimal validation but delegates model construction and weight
        loading to the external package.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        if not _PANGU_AVAILABLE:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'panguweather' package is required for PanguModel but was not found"
            )

        path = Path(checkpoint_path)
        if not path.exists():  # pragma: no cover - defensive programming
            raise FileNotFoundError(f"Pangu checkpoint not found at {path}")

        try:
            # ``panguweather`` exposes a convenience ``load_model`` function that
            # returns a callable model when provided with a checkpoint path.  The
            # exact API may evolve, but we rely on it behaving similarly to the
            # official repository.  Any exception is caught and re-raised with a
            # clearer message for consumers of GaleNet.
            self._model = dm_pangu.load_model(path)  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - defensive programming
            # Older versions may expose ``load_checkpoint`` instead.
            try:
                self._model = dm_pangu.load_checkpoint(path)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - checkpoint issues
                raise RuntimeError(f"Failed to load Pangu checkpoint: {exc}") from exc
        except Exception as exc:  # pragma: no cover - checkpoint issues
            raise RuntimeError(f"Failed to load Pangu checkpoint: {exc}") from exc

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
                raise RuntimeError(f"Pangu inference failed: {exc}") from exc
            return xr.DataArray(
                np.asarray(out, dtype=np.float32), coords=climate.coords, dims=climate.dims
            )

        arr = np.asarray(climate, dtype=np.float32)
        try:
            out = self._model(arr)
        except Exception as exc:  # pragma: no cover - runtime failure
            raise RuntimeError(f"Pangu inference failed: {exc}") from exc
        return np.asarray(out, dtype=np.float32)

    def predict(self, climate: ArrayLike, num_steps: int, step: int) -> ArrayLike:
        """Run autoregressive forecasts for ``num_steps`` iterations."""

        arr: ArrayLike = climate
        for _ in range(num_steps):
            arr = self.infer(arr)
        return arr


__all__ = ["PanguModel"]
