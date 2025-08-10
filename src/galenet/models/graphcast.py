"""Placeholder GraphCast model for GaleNet.

This lightweight implementation loads parameters from a checkpoint file so the
inference pipeline can instantiate the model during tests.  It does not provide
actual GraphCast functionality; instead, it repeats the last input observation
(similar to a persistence baseline) when generating forecasts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class GraphCastModel:
    """Minimal GraphCast model wrapper."""

    def __init__(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"GraphCast checkpoint not found at {path}")

        # Load parameters from the provided checkpoint.  They are not used by the
        # dummy predict implementation but storing them verifies loading works.
        with np.load(path, allow_pickle=False) as data:
            self.params = {k: data[k] for k in data.files}

    def predict(self, features: pd.DataFrame, num_steps: int, step: int) -> pd.DataFrame:
        """Generate a simple forecast by repeating the last observation."""
        last = features.iloc[-1]
        rows = []
        for _ in range(num_steps):
            rows.append(
                {
                    "latitude": last.get("latitude", np.nan),
                    "longitude": last.get("longitude", np.nan),
                    "max_wind": last.get("max_wind", np.nan),
                    "min_pressure": last.get("min_pressure", np.nan),
                }
            )
        return pd.DataFrame(rows)
