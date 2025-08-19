"""Evaluation metrics for GaleNet forecasts."""

from __future__ import annotations

from math import cos, radians
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import yaml  # type: ignore[import-untyped]

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


def track_error(
    pred: Sequence[Sequence[float]], truth: Sequence[Sequence[float]]
) -> float:
    """Mean great-circle distance between predicted and true track in km."""
    pred_arr = np.asarray(pred)
    truth_arr = np.asarray(truth)
    distances = _haversine(
        pred_arr[:, 0], pred_arr[:, 1], truth_arr[:, 0], truth_arr[:, 1]
    )
    return float(np.mean(distances))


def _to_xy(
    lat: float | np.ndarray, lon: float | np.ndarray, ref_lat: float
) -> np.ndarray:
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


def along_track_error(
    pred: Sequence[Sequence[float]], truth: Sequence[Sequence[float]]
) -> float:
    """Mean absolute error component along the true track direction in km."""
    pred_arr = np.asarray(pred)
    truth_arr = np.asarray(truth)
    along, _ = _along_cross_components(pred_arr, truth_arr)
    return float(np.mean(np.abs(along)))


def cross_track_error(
    pred: Sequence[Sequence[float]], truth: Sequence[Sequence[float]]
) -> float:
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


def rapid_intensification_skill(pred: Sequence[float], truth: Sequence[float]) -> float:
    """F1 skill score for predicting rapid intensification events.

    A rapid intensification (RI) event is defined as an increase in
    intensity of at least ``30`` kt over a 24 hour window (four 6 hour
    steps). The skill score measures how well the forecast identifies RI
    events compared to the truth using the F1 formulation.

    Args:
        pred: Sequence of predicted intensity values at 6-hourly steps.
        truth: Sequence of observed intensity values at 6-hourly steps.

    Returns:
        F1 score between 0 and 1. Returns 0.0 when no RI events are
        present in either prediction or truth or when precision/recall
        are undefined.
    """
    pred_arr = np.asarray(pred)
    truth_arr = np.asarray(truth)

    if len(pred_arr) < 5 or len(truth_arr) < 5:
        return 0.0

    pred_events = (pred_arr[4:] - pred_arr[:-4]) >= 30
    truth_events = (truth_arr[4:] - truth_arr[:-4]) >= 30

    tp = np.sum(pred_events & truth_events)
    fp = np.sum(pred_events & ~truth_events)
    fn = np.sum(~pred_events & truth_events)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


METRIC_FUNCTIONS: Mapping[str, Callable[..., float]] = {
    "track_error": track_error,
    "along_track_error": along_track_error,
    "cross_track_error": cross_track_error,
    "intensity_mae": intensity_mae,
    "rapid_intensification_skill": rapid_intensification_skill,
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
        if name in {"intensity_mae", "rapid_intensification_skill"}:
            results[name] = func(intensity_pred, intensity_truth)
        else:
            results[name] = func(track_pred, track_truth)
    return results


def compute_metrics_multi(
    track_preds: Sequence[Sequence[Sequence[float]]],
    track_truths: Sequence[Sequence[Sequence[float]]],
    intensity_preds: Sequence[Sequence[float]],
    intensity_truths: Sequence[Sequence[float]],
    metrics: Iterable[str] | None = None,
) -> Dict[str, float]:
    """Compute metrics averaged over multiple storms.

    Args:
        track_preds: Sequence of predicted tracks for each storm.
        track_truths: Sequence of true tracks for each storm.
        intensity_preds: Sequence of predicted intensity sequences.
        intensity_truths: Sequence of true intensity sequences.
        metrics: Iterable of metric names. Defaults to
            ``configs/default_config.yaml::evaluation.metrics``.

    Returns:
        Dictionary with mean metric values across all storms.
    """
    metrics = metrics or DEFAULT_METRICS
    accum: Dict[str, list[float]] = {m: [] for m in metrics}
    for t_pred, t_true, i_pred, i_true in zip(
        track_preds, track_truths, intensity_preds, intensity_truths
    ):
        result = compute_metrics(t_pred, t_true, i_pred, i_true, metrics)
        for name, value in result.items():
            accum.setdefault(name, []).append(float(value))
    return {name: float(np.mean(values)) for name, values in accum.items()}
