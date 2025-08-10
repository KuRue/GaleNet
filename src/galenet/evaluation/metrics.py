"""Evaluation metrics for GaleNet forecasts."""

from __future__ import annotations

from math import radians, cos
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import yaml

# Load default metrics from configuration without initializing Hydra
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default_config.yaml"
with open(_CONFIG_PATH) as _cfg:
    DEFAULT_METRICS: Iterable[str] = yaml.safe_load(_cfg)["evaluation"]["metrics"]

EARTH_RADIUS_KM = 6371.0


def _haversine(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorised haversine distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def track_error(pred: Sequence[Sequence[float]], truth: Sequence[Sequence[float]]) -> float:
    """Mean great-circle distance between predicted and true track in km."""
    pred_arr = np.asarray(pred)
    truth_arr = np.asarray(truth)
    distances = _haversine(
        pred_arr[:, 0], pred_arr[:, 1], truth_arr[:, 0], truth_arr[:, 1]
    )
    return float(np.mean(distances))


def _to_xy(lat: float | np.ndarray, lon: float | np.ndarray, ref_lat: float) -> np.ndarray:
    """Convert lat/lon to local Cartesian x/y in km using equirectangular projection."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_rad = radians(ref_lat)
    x = EARTH_RADIUS_KM * (lon_rad) * cos(ref_rad)
    y = EARTH_RADIUS_KM * (lat_rad)
    return np.stack([x, y], axis=-1)


def _along_cross_components(
    pred: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute along-track and cross-track error components for each step."""
    ref_lat = float(np.mean(truth[:, 0]))
    truth_xy = _to_xy(truth[:, 0], truth[:, 1], ref_lat)
    pred_xy = _to_xy(pred[:, 0], pred[:, 1], ref_lat)

    along_list = []
    cross_list = []
    for i in range(1, len(truth_xy)):
        dir_vec = truth_xy[i] - truth_xy[i - 1]
        norm = np.linalg.norm(dir_vec)
        if norm == 0:
            continue
        dir_unit = dir_vec / norm
        err_vec = pred_xy[i] - truth_xy[i]
        along_list.append(float(np.dot(err_vec, dir_unit)))
        cross_list.append(float(dir_unit[0] * err_vec[1] - dir_unit[1] * err_vec[0]))

    return np.asarray(along_list), np.asarray(cross_list)


def along_track_error(pred: Sequence[Sequence[float]], truth: Sequence[Sequence[float]]) -> float:
    """Mean absolute error component along the true track direction in km."""
    pred_arr = np.asarray(pred)
    truth_arr = np.asarray(truth)
    along, _ = _along_cross_components(pred_arr, truth_arr)
    return float(np.mean(np.abs(along)))


def cross_track_error(pred: Sequence[Sequence[float]], truth: Sequence[Sequence[float]]) -> float:
    """Mean absolute error component perpendicular to true track in km."""
    pred_arr = np.asarray(pred)
    truth_arr = np.asarray(truth)
    _, cross = _along_cross_components(pred_arr, truth_arr)
    return float(np.mean(np.abs(cross)))


def intensity_mae(pred: Sequence[float], truth: Sequence[float]) -> float:
    """Mean absolute error in storm intensity (e.g., max wind speed)."""
    pred_arr = np.asarray(pred)
    truth_arr = np.asarray(truth)
    return float(np.mean(np.abs(pred_arr - truth_arr)))


METRIC_FUNCTIONS = {
    "track_error": track_error,
    "along_track_error": along_track_error,
    "cross_track_error": cross_track_error,
    "intensity_mae": intensity_mae,
}


def compute_metrics(
    track_pred: Sequence[Sequence[float]],
    track_truth: Sequence[Sequence[float]],
    intensity_pred: Sequence[float],
    intensity_truth: Sequence[float],
    metrics: Iterable[str] | None = None,
) -> Dict[str, float]:
    """Compute selected metrics for a forecast.

    Args:
        track_pred: Predicted track coordinates ``(lat, lon)``.
        track_truth: True track coordinates ``(lat, lon)``.
        intensity_pred: Predicted intensity values.
        intensity_truth: True intensity values.
        metrics: Iterable of metric names. Defaults to
            ``configs/default_config.yaml::evaluation.metrics``.

    Returns:
        Dictionary of computed metric values.
    """
    metrics = metrics or DEFAULT_METRICS
    results: Dict[str, float] = {}
    for name in metrics:
        func = METRIC_FUNCTIONS.get(name)
        if func is None:
            continue
        if name == "intensity_mae":
            results[name] = func(intensity_pred, intensity_truth)
        else:
            results[name] = func(track_pred, track_truth)
    return results
