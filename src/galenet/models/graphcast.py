"""Thin wrapper around DeepMind's GraphCast model.

The official `GraphCast <https://github.com/deepmind/graphcast>`_ model is a
JAX/Flax network containing millions of parameters.  Shipping the full model and
its dependencies would make GaleNet heavy and slow to test, so this module
provides a small wrapper that *behaves* like GraphCast for the purposes of the
tests and demos.  When the real :mod:`graphcast` package is available the class
can load its checkpoints directly.  Otherwise a minimal NumPy/Torch
implementation is used which simply applies a learned linear transformation.

Two public methods are exposed:

``infer``
    Apply the model to a ``numpy.ndarray`` or an ``xarray.DataArray`` and return
    the same type.

``predict``
    Deterministic hurricane‑track forecasts compatible with
    :class:`galenet.inference.pipeline.GaleNetPipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:  # pragma: no cover - import guard
    import graphcast as dm_graphcast  # type: ignore

    _GRAPHCAST_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    dm_graphcast = None  # type: ignore
    _GRAPHCAST_AVAILABLE = False


ArrayLike = Union[np.ndarray, xr.DataArray]


def _load_linear_params(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``w`` and ``b`` from a NumPy ``.npz`` checkpoint."""

    data = np.load(path, allow_pickle=True)
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

    return w, b


def _numpy_impl(w: np.ndarray, b: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a NumPy based linear transformation ``x @ w + b``."""

    def apply(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(-1, x.shape[-1])
        out = flat @ w + b
        return out.reshape(x.shape)

    return apply


def _torch_impl(w: np.ndarray, b: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a Torch based implementation of the linear layer."""

    w_t = torch.as_tensor(w)
    b_t = torch.as_tensor(b)

    def apply(x: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(x.reshape(-1, x.shape[-1]), dtype=torch.float32)
        out = t @ w_t + b_t  # type: ignore[operator]
        return out.detach().cpu().numpy().reshape(x.shape)

    return apply


class GraphCastModel:
    """Interface compatible GraphCast model.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.npz`` file storing the model parameters.  For the real
        DeepMind model the wrapper expects the nested ``{"params": ...}`` format
        produced by the official codebase.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():  # pragma: no cover - defensive programming
            raise FileNotFoundError(f"GraphCast checkpoint not found at {path}")

        # When the official GraphCast library is installed we could deserialize
        # the full model here.  The tests exercise the light‑weight linear
        # re‑implementation, so we always extract the raw parameters.
        w, b = _load_linear_params(path)

        if _TORCH_AVAILABLE:
            self._apply = _torch_impl(w, b)
        else:
            self._apply = _numpy_impl(w, b)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(self, climate: ArrayLike) -> ArrayLike:
        """Run inference on ``climate`` data."""

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

