"""Inference pipeline for generating hurricane forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..data import HurricaneDataPipeline
from ..data.processors import HurricanePreprocessor
from ..data.validators import HurricaneDataValidator
from ..utils.config import get_config
from ..utils.logging import setup_logging


@dataclass
class ForecastResult:
    """Container for forecast results.

    Attributes
    ----------
    track:
        DataFrame containing both the historical track and the forecasted
        points. A ``lead_time`` column stores the hour offset from the first
        observation.
    """

    track: pd.DataFrame

    def get_position(self, lead_time: int) -> tuple[float, float]:
        """Return the forecast position at a given lead time.

        Parameters
        ----------
        lead_time:
            Lead time in hours.
        """
        row = self.track[self.track["lead_time"] == lead_time]
        if row.empty:
            raise ValueError(f"Lead time {lead_time} not available")
        return float(row["latitude"].iloc[0]), float(row["longitude"].iloc[0])

    @property
    def max_intensity(self) -> float:
        """Maximum forecast intensity (max wind)."""
        return float(self.track["max_wind"].max())

    @property
    def track_cone(self) -> pd.DataFrame:
        """Simplified track cone as latitude/longitude points."""
        return self.track[["latitude", "longitude"]]


class GaleNetPipeline:
    """High-level pipeline coordinating data loading and forecasting."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = get_config(config_path)
        setup_logging(level=self.config.logging.level)

        # Core components
        self.data = HurricaneDataPipeline(config_path)
        self.preprocessor = HurricanePreprocessor()
        self.validator = HurricaneDataValidator()

    def forecast_storm(
        self,
        storm_id: str,
        forecast_hours: int = 120,
        source: str = "hurdat2",
    ) -> ForecastResult:
        """Generate a simple persistence-based forecast for a storm.

        This method loads the requested storm using :class:`HurricaneDataPipeline`
        and then extends the track forward in time using a persistence
        baseline (i.e. last observed position and intensity are repeated).
        The goal is not to provide a skillful forecast but to wire together
        the project components for testing and further development.
        """

        data = self.data.load_hurricane_for_training(
            storm_id, source=source, include_era5=False
        )
        track = data["track"].copy().sort_values("timestamp").reset_index(drop=True)

        # Validate the input track; log any issues but continue
        valid, errors = self.validator.validate_track(track)
        if not valid:
            logger.warning("Track validation issues detected: {}", errors)

        # Basic preprocessing to annotate lead times
        track["lead_time"] = (
            track["timestamp"] - track["timestamp"].iloc[0]
        ).dt.total_seconds() // 3600

        # Persistence forecast
        step = int(self.config.inference.time_step)
        num_steps = int(forecast_hours // step)
        last = track.iloc[-1]
        future_times = [
            last["timestamp"] + pd.Timedelta(hours=step * i)
            for i in range(1, num_steps + 1)
        ]
        future_rows = []
        for i, ts in enumerate(future_times, 1):
            future_rows.append(
                {
                    "storm_id": last["storm_id"],
                    "name": last.get("name", ""),
                    "timestamp": ts,
                    "latitude": last["latitude"],
                    "longitude": last["longitude"],
                    "max_wind": last.get("max_wind", np.nan),
                    "min_pressure": last.get("min_pressure", np.nan),
                    "lead_time": track["lead_time"].iloc[-1] + step * i,
                }
            )

        if future_rows:
            forecast_df = pd.DataFrame(future_rows)
            track = pd.concat([track, forecast_df], ignore_index=True)

        return ForecastResult(track=track)
