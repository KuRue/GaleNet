"""Light‑weight GraphCast model wrapper used in tests and demos.

The real `GraphCast <https://github.com/deepmind/graphcast>`_ implementation is
a large JAX/Flax model.  Shipping the full dependency stack would make the
project heavy and slow to test, so GaleNet relies on a tiny but *interface
compatible* re-implementation.  When :mod:`torch` is available the model uses a
single ``torch.nn.Linear`` style transformation; otherwise NumPy is used as a
fallback.  The goal is simply to exercise the surrounding inference pipeline and
to provide a convenient hook for loading real DeepMind checkpoints when running
the project end‑to‑end.

Checkpoints are stored in ``.npz`` files.  We support the minimal format used in
our tests as well as the nested ``{"params": {"w": ..., "b": ...}}`` structure
exported by the official codebase.  The parameters represent a linear transform
``x @ w + b`` applied along the final feature dimension.

Two public methods are exposed:

``infer``
    Run the linear layer on a NumPy ``ndarray`` or an ``xarray`` ``DataArray``
    and return the same type.

``predict``
    Provide a deterministic hurricane‑track forecasting interface compatible
    with :class:`galenet.inference.pipeline.GaleNetPipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import xarray as xr

try:  # PyTorch is optional – fall back to NumPy if unavailable
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

ArrayLike = Union[np.ndarray, xr.DataArray]


class GraphCastModel:
    """Deterministic GraphCast‑style model based on a single linear layer."""

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

        w = np.asarray(w, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        if (
            w.ndim != 2
            or w.shape[0] != w.shape[1]
            or b.shape != (w.shape[0],)
        ):
            raise ValueError("GraphCast checkpoint has unexpected shapes")

        if _TORCH_AVAILABLE:
            self.w = torch.as_tensor(w)
            self.b = torch.as_tensor(b)
        else:  # NumPy fallback used in tests
            self.w = w
            self.b = b

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply(self, array: np.ndarray) -> np.ndarray:
        """Apply the linear transform to ``array`` using Torch when available."""

        flat = array.reshape(-1, array.shape[-1])
        if _TORCH_AVAILABLE:
            t = torch.as_tensor(flat, dtype=torch.float32)
            out = t @ self.w + self.b  # type: ignore[operator]
            return out.detach().cpu().numpy().reshape(array.shape)

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

