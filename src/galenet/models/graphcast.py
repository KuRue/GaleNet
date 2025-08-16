"""GraphCast model wrapper for GaleNet.

This module provides a very small but reasonably faithful reimplementation of
`GraphCast <https://github.com/deepmind/graphcast>`_.  The real GraphCast model
from DeepMind is a large JAX model operating on global climate tensors.  For
testing and lightweight inference inside GaleNet we only require a deterministic
component that can load a checkpoint and perform forward passes on NumPy or
``xarray`` tensors.  The implementation below mimics the public interface of the
official model closely enough for our purposes.

The expected checkpoint format is a ``.npz`` file containing two arrays:

``w``
    Weight matrix with shape ``(C, C)`` where ``C`` is the number of climate
    channels.
``b``
    Bias vector with shape ``(C,)``.

These arrays are applied as ``x @ w + b`` along the feature dimension of the
input tensor.  Checkpoints exported from the official DeepMind implementation
contain a nested ``params`` dictionary; this wrapper understands both the simple
``w``/``b`` format used in the tests and the nested format by looking for these
keys during loading.

Two public methods are exposed:

``infer``
    Applies the linear transform to a NumPy ``ndarray`` or an ``xarray``
    ``DataArray`` and returns an object of the same type.

``predict``
    Provides a minimal hurricane-track forecasting interface compatible with
    :class:`galenet.inference.pipeline.GaleNetPipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import xarray as xr

ArrayLike = Union[np.ndarray, xr.DataArray]


class GraphCastModel:
    """Minimal GraphCast-style model based on a single linear layer."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():  # pragma: no cover - defensive programming
            raise FileNotFoundError(f"GraphCast checkpoint not found at {path}")

        data = np.load(path, allow_pickle=True)

        # DeepMind checkpoints often store parameters in a nested ``params``
        # dictionary.  For our simplified tests we also support direct ``w`` and
        # ``b`` arrays.
        if "w" in data and "b" in data:
            w, b = data["w"], data["b"]
        elif "params" in data:
            params = data["params"].item()
            w, b = params["w"], params["b"]
        else:  # pragma: no cover - unsupported format
            raise ValueError("Invalid GraphCast checkpoint format")

        self.w = np.asarray(w, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)

        if (
            self.w.ndim != 2
            or self.w.shape[0] != self.w.shape[1]
            or self.b.shape != (self.w.shape[0],)
        ):
            raise ValueError("GraphCast checkpoint has unexpected shapes")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply(self, array: np.ndarray) -> np.ndarray:
        """Apply the linear transform to ``array``."""

        flat = array.reshape(-1, array.shape[-1])
        out = flat @ self.w + self.b
        return out.reshape(array.shape)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(self, climate: ArrayLike) -> ArrayLike:
        """Run inference on ``climate`` data.

        Parameters
        ----------
        climate:
            ``numpy.ndarray`` or ``xarray.DataArray`` with the final dimension
            matching the checkpoint's channel size.  The returned object matches
            the input type.
        """

        if isinstance(climate, xr.DataArray):
            arr = climate.values.astype(np.float32)
            result = self._apply(arr)
            return xr.DataArray(result, coords=climate.coords, dims=climate.dims)

        arr = np.asarray(climate, dtype=np.float32)
        return self._apply(arr)

    def predict(self, features: pd.DataFrame, num_steps: int, step: int) -> pd.DataFrame:
        """Generate deterministic forecasts using the loaded parameters."""

        last = features.iloc[-1][["latitude", "longitude", "max_wind", "min_pressure"]]
        state = last.to_numpy(dtype=np.float32)

        rows: list[dict[str, Any]] = []
        for _ in range(num_steps):
            state = self._apply(state)  # type: ignore[arg-type]
            rows.append(
                {
                    "latitude": float(state[0]),
                    "longitude": float(state[1]),
                    "max_wind": float(state[2]),
                    "min_pressure": float(state[3]),
                }
            )
        return pd.DataFrame(rows)


__all__ = ["GraphCastModel"]

