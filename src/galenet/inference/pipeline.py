"""Inference pipeline for generating hurricane forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
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
    fields:
        Optional environmental fields returned by GraphCast as an
        ``xarray`` object.
    """

    track: pd.DataFrame
    fields: xr.DataArray | xr.Dataset | None = None

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
        # Load forecasting model
        self.model = self._load_model()

    # ------------------------------------------------------------------
    # Model handling
    # ------------------------------------------------------------------
    def _load_model(self):
        """Instantiate the forecasting model specified in the config.

        The project does not yet ship with full model implementations, so we
        provide a simple baseline model that repeats the last observation.  This
        allows the pipeline to exercise the preprocessing, validation and
        ensemble code paths during testing.
        """

        model_name = getattr(self.config.model, "name", "")

        if model_name == "graphcast":
            from ..models.graphcast import GraphCastModel

            graphcast_cfg = getattr(self.config.model, "graphcast", {})
            checkpoint = getattr(graphcast_cfg, "checkpoint_path", "")
            try:
                return GraphCastModel(checkpoint)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to load GraphCast model: {}", exc)
                raise RuntimeError("GraphCast model could not be initialized") from exc

        if model_name == "pangu":
            from ..models.pangu import PanguModel

            pangu_cfg = getattr(self.config.model, "pangu", {})
            checkpoint = getattr(pangu_cfg, "checkpoint_path", "")
            try:
                return PanguModel(checkpoint)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to load Pangu model: {}", exc)
                raise RuntimeError("Pangu model could not be initialized") from exc

        if model_name in {"hurricane_ensemble", "ensemble"}:
            return _PersistenceModel()

        logger.warning("Unknown model '%s', falling back to persistence", model_name)
        return _PersistenceModel()

    # ------------------------------------------------------------------
    # Pre/Post processing helpers
    # ------------------------------------------------------------------
    def _preprocess_track(self, track: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to the raw track data."""

        try:
            norm = self.preprocessor.normalize_track_data(track, fit=False)
            features = self.preprocessor.create_track_features(norm)
            return features
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.error("Track preprocessing failed: {}", exc)
            raise

    def _post_process(self, track: pd.DataFrame, hist_len: int) -> pd.DataFrame:
        """Apply optional post-processing steps configured for inference."""

        post_cfg = getattr(self.config.inference, "post_process", {})

        # Optional smoothing of the forecast track only
        if getattr(post_cfg, "smooth_track", False):
            try:
                forecast_slice = track.iloc[hist_len:][["latitude", "longitude"]]
                smoothed = forecast_slice.rolling(window=3, min_periods=1).mean()
                track.loc[forecast_slice.index, ["latitude", "longitude"]] = smoothed
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Track smoothing failed: {}", exc)

        # Optional physics enforcement via validator
        if getattr(post_cfg, "enforce_physics", False):
            valid, errors = self.validator.validate_intensity_physics(track)
            if not valid:
                logger.warning("Physics validation issues detected: {}", errors)

        return track

    def forecast_storm(
        self,
        storm_id: str,
        forecast_hours: int = 120,
        source: str = "hurdat2",
    ) -> ForecastResult:
        """Generate a model-based hurricane forecast.

        The pipeline performs the following steps:

        1. Load historical track data for ``storm_id``.
        2. Validate and preprocess the track using
           :class:`HurricaneDataValidator` and
           :class:`HurricanePreprocessor`.
        3. Run the configured model to produce future track/intensity
           predictions.  Ensemble and post-processing options are applied if
           enabled in the configuration.
        """

        # ------------------------------------------------------------------
        # Load and validate historical track
        # ------------------------------------------------------------------
        from ..models.graphcast import GraphCastModel
        from ..models.pangu import PanguModel

        is_graphcast = isinstance(self.model, GraphCastModel)
        is_pangu = isinstance(self.model, PanguModel)

        data = self.data.load_hurricane_for_training(
            storm_id, source=source, include_era5=is_graphcast or is_pangu
        )
        track = data["track"].copy().sort_values("timestamp").reset_index(drop=True)

        # Validate the input track; log any issues but continue
        try:
            valid, errors = self.validator.validate_track(track)
            if not valid:
                logger.warning("Track validation issues detected: {}", errors)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Track validation failed: {}", exc)

        # Annotate lead times for historical observations
        track["lead_time"] = (
            track["timestamp"] - track["timestamp"].iloc[0]
        ).dt.total_seconds() // 3600

        # ------------------------------------------------------------------
        # Preprocess input for model
        # ------------------------------------------------------------------
        processed = self._preprocess_track(track)

        step = int(self.config.inference.time_step)
        num_steps = int(forecast_hours // step)

        if is_graphcast or is_pangu:
            era5 = data.get("era5")
            if era5 is None:
                model_name = "GraphCast" if is_graphcast else "Pangu"
                raise RuntimeError(
                    f"{model_name}Model requires ERA5 environmental fields at 0.25° resolution"
                )
            try:
                fields_forecast = self.model.infer(era5)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("%s inference failed: %s", type(self.model).__name__, exc)
                raise RuntimeError(f"{type(self.model).__name__} inference failed") from exc
            model = _PersistenceModel()
        else:
            fields_forecast = None
            model = self.model

        # ------------------------------------------------------------------
        # Run model (with optional ensemble)
        # ------------------------------------------------------------------
        def run_model():
            return model.predict(processed, num_steps, step)

        ensemble_cfg = getattr(self.config.inference, "ensemble", {})
        if getattr(ensemble_cfg, "enabled", False):
            size = int(getattr(ensemble_cfg, "size", 1))
            members = []
            for _ in range(size):
                try:
                    members.append(run_model())
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Model prediction failed: {}", exc)
                    raise
            preds = (
                pd.concat(members, axis=0).groupby(level=0).mean().reset_index(drop=True)
            )
        else:
            try:
                preds = run_model().reset_index(drop=True)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Model prediction failed: {}", exc)
                raise

        # ------------------------------------------------------------------
        # Merge predictions with historical track
        # ------------------------------------------------------------------
        last = track.iloc[-1]
        future_times = [
            last["timestamp"] + pd.Timedelta(hours=step * i)
            for i in range(1, num_steps + 1)
        ]
        preds["timestamp"] = future_times
        preds["storm_id"] = last["storm_id"]
        preds["name"] = last.get("name", "")
        preds["lead_time"] = [track["lead_time"].iloc[-1] + step * i for i in range(1, num_steps + 1)]

        full_track = pd.concat([track, preds], ignore_index=True)

        # ------------------------------------------------------------------
        # Post-processing & final validation
        # ------------------------------------------------------------------
        full_track = self._post_process(full_track, len(track))

        try:
            valid, errors = self.validator.validate_track(full_track)
            if not valid:
                logger.warning("Forecast track validation issues: {}", errors)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Forecast validation failed: {}", exc)

        return ForecastResult(track=full_track, fields=fields_forecast)


class _PersistenceModel:
    """Simple forecasting model that repeats the last observation."""

    def predict(self, features: pd.DataFrame, num_steps: int, step: int) -> pd.DataFrame:
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
