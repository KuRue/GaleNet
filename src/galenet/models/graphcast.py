"""Lightweight GraphCast-style model for testing.

This module provides a tiny reimplementation of the GraphCast interface that is
sufficient for unit tests.  The real GraphCast model from DeepMind operates on
large climate tensors and requires specialised infrastructure.  For GaleNet we
only need a deterministic and fast component that mimics loading a checkpoint
and running inference on climate data.

The checkpoint is expected to be a ``.npz`` file containing two arrays:
``w`` with shape ``(C, C)`` and ``b`` with shape ``(C,)``.  The model applies
``x @ w + b`` along the feature dimension where ``C`` is the number of climate
channels.  The public :meth:`infer` method operates on arbitrary climate
``numpy.ndarray`` tensors while :meth:`predict` exposes a minimal hurricane track
forecasting interface used by :class:`galenet.inference.pipeline.GaleNetPipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class GraphCastModel:
    """Minimal GraphCast-style model based on a single linear layer."""

    def __init__(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():  # pragma: no cover - defensive
            raise FileNotFoundError(f"GraphCast checkpoint not found at {path}")

        data = np.load(path)
        try:
            self.w = np.asarray(data["w"], dtype=np.float32)
            self.b = np.asarray(data["b"], dtype=np.float32)
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid GraphCast checkpoint format") from exc

        if (
            self.w.ndim != 2
            or self.w.shape[0] != self.w.shape[1]
            or self.b.shape != (self.w.shape[0],)
        ):
            raise ValueError("GraphCast checkpoint has unexpected shapes")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def infer(self, climate: np.ndarray) -> np.ndarray:
        """Apply the linear layer to a climate tensor.

        Parameters
        ----------
        climate:
            Array with shape ``(..., C)`` where ``C`` matches the checkpoint
            dimensions.  The linear transform is applied to the final axis and
            the output has the same shape as the input.
        """

        flat = climate.reshape(-1, climate.shape[-1])
        out = flat @ self.w + self.b
        return out.reshape(climate.shape)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame, num_steps: int, step: int) -> pd.DataFrame:
        """Generate deterministic forecasts using the loaded parameters.

        Only the final observation of ``features`` is used as the model state.
        The state is transformed repeatedly with :meth:`infer` to create the
        requested number of forecast steps.  ``step`` is accepted for interface
        compatibility but is otherwise unused.
        """

        last = features.iloc[-1][["latitude", "longitude", "max_wind", "min_pressure"]]
        x = last.to_numpy(dtype=np.float32)

        rows: list[dict[str, Any]] = []
        for _ in range(num_steps):
            x = self.infer(x)  # type: ignore[arg-type]
            rows.append(
                {
                    "latitude": float(x[0]),
                    "longitude": float(x[1]),
                    "max_wind": float(x[2]),
                    "min_pressure": float(x[3]),
                }
            )
        return pd.DataFrame(rows)
