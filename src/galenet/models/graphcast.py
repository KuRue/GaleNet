"""Lightweight wrapper around the official GraphCast model.

The real GraphCast architecture from DeepMind is extremely heavy and expects
large climate tensors as input.  For the purposes of GaleNet's testing suite we
only need a minimal interface that demonstrates how a GraphCast model would be
used.  This module therefore implements a tiny linear layer whose parameters are
stored using GraphCast's checkpoint format.  Loading and applying the layer
mimics the behaviour of restoring a trained GraphCast model and performing a
forward pass.

The class consumes hurricane track features and repeatedly applies the loaded
linear transformation to the final observation in an autoregressive manner.  It
produces a ``pandas.DataFrame`` containing deterministic forecast steps which is
sufficient for integration tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import pandas as pd
from graphcast import checkpoint


FeatureArray = jnp.ndarray


class GraphCastModel:
    """Minimal GraphCast style model using a single linear layer.

    Parameters are stored in the official GraphCast checkpoint format.  The
    checkpoint must contain two arrays: ``w`` with shape ``(4, 4)`` and ``b``
    with shape ``(4,)`` representing the weight matrix and bias for the linear
    layer.  This layout mirrors the expectation of the small test network used
    in the accompanying unit tests.
    """

    def __init__(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"GraphCast checkpoint not found at {path}")

        with path.open("rb") as f:
            params: Dict[str, Any] = checkpoint.load(f, dict)

        try:
            self.w = jnp.array(params["w"], dtype=jnp.float32)
            self.b = jnp.array(params["b"], dtype=jnp.float32)
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError("Invalid GraphCast checkpoint format") from exc

        if self.w.shape != (4, 4) or self.b.shape != (4,):  # pragma: no cover
            raise ValueError("GraphCast checkpoint has unexpected shapes")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _step(self, x: FeatureArray) -> FeatureArray:
        """Apply the linear layer for a single forecast step."""
        return x @ self.w + self.b

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame, num_steps: int, step: int) -> pd.DataFrame:
        """Generate deterministic forecasts using the loaded parameters.

        Parameters
        ----------
        features:
            Historical track features.  Only the final row is used as the model
            state.
        num_steps:
            Number of future steps to forecast.
        step:
            Lead time between steps in hours.  (Unused but included for
            compatibility with other models.)
        """

        # Convert the final observation into a JAX array in the expected order.
        last = features.iloc[-1][["latitude", "longitude", "max_wind", "min_pressure"]]
        x = jnp.array(last.to_numpy(jnp.float32))

        rows = []
        for _ in range(num_steps):
            x = self._step(x)
            rows.append(
                {
                    "latitude": float(x[0]),
                    "longitude": float(x[1]),
                    "max_wind": float(x[2]),
                    "min_pressure": float(x[3]),
                }
            )
        return pd.DataFrame(rows)
