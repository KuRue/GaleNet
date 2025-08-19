"""Baseline forecast methods for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import yaml  # type: ignore[import-untyped]

from .metrics import DEFAULT_METRICS, compute_metrics

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default_config.yaml"
with open(_CONFIG_PATH) as _cfg:
    DEFAULT_BASELINES: Iterable[str] = yaml.safe_load(_cfg)["evaluation"]["baselines"]


def persistence_baseline(
    track: Sequence[Sequence[float]] | np.ndarray, forecast_steps: int
) -> np.ndarray:
    """Simple persistence baseline: repeat last known position and intensity."""
    track_arr = np.asarray(track)
    last = track_arr[-1]
    return np.repeat(last[None, :], forecast_steps, axis=0)


def cliper5_baseline(
    track: Sequence[Sequence[float]] | np.ndarray, forecast_steps: int
) -> np.ndarray:
    """CLIPER5 baseline using mean motion over last 5 steps."""
    track_arr = np.asarray(track)
    if len(track_arr) < 2:
        return persistence_baseline(track_arr, forecast_steps)

    displacements = np.diff(track_arr[:, :2], axis=0)
    if len(displacements) >= 5:
        mean_disp = displacements[-5:].mean(axis=0)
    else:
        mean_disp = displacements.mean(axis=0)

    last_pos = track_arr[-1, :2]
    intensity = track_arr[-1, 2]
    preds = []
    for step in range(1, forecast_steps + 1):
        pos = last_pos + mean_disp * step
        preds.append(np.array([pos[0], pos[1], intensity]))
    return np.stack(preds, axis=0)


def gfs_baseline(
    track: Sequence[Sequence[float]] | np.ndarray, forecast_steps: int
) -> np.ndarray:
    """GFS-like baseline using slightly accelerated motion."""
    track_arr = np.asarray(track)
    if len(track_arr) < 2:
        return persistence_baseline(track_arr, forecast_steps)

    displacements = np.diff(track_arr[:, :2], axis=0)
    if len(displacements) >= 5:
        mean_disp = displacements[-5:].mean(axis=0)
    else:
        mean_disp = displacements.mean(axis=0)

    last_pos = track_arr[-1, :2]
    intensity = track_arr[-1, 2]
    preds = []
    for step in range(1, forecast_steps + 1):
        pos = last_pos + mean_disp * 1.1 * step
        preds.append(np.array([pos[0], pos[1], intensity]))
    return np.stack(preds, axis=0)


def ecmwf_baseline(
    track: Sequence[Sequence[float]] | np.ndarray, forecast_steps: int
) -> np.ndarray:
    """ECMWF-like baseline using slightly slower motion."""
    track_arr = np.asarray(track)
    if len(track_arr) < 2:
        return persistence_baseline(track_arr, forecast_steps)

    displacements = np.diff(track_arr[:, :2], axis=0)
    if len(displacements) >= 5:
        mean_disp = displacements[-5:].mean(axis=0)
    else:
        mean_disp = displacements.mean(axis=0)

    last_pos = track_arr[-1, :2]
    intensity = track_arr[-1, 2]
    preds = []
    for step in range(1, forecast_steps + 1):
        pos = last_pos + mean_disp * 0.9 * step
        preds.append(np.array([pos[0], pos[1], intensity]))
    return np.stack(preds, axis=0)


BASELINE_FUNCTIONS = {
    "persistence": persistence_baseline,
    "cliper5": cliper5_baseline,
    "gfs": gfs_baseline,
    "ecmwf": ecmwf_baseline,
}


def run_baselines(
    track: Sequence[Sequence[float]] | np.ndarray,
    forecast_steps: int,
    baselines: Iterable[str] | None = None,
) -> Dict[str, np.ndarray]:
    """Run selected baseline methods on a storm track."""
    baselines = baselines or DEFAULT_BASELINES
    results: Dict[str, np.ndarray] = {}
    for name in baselines:
        func = BASELINE_FUNCTIONS.get(name)
        if func is None:
            continue
        results[name] = func(track, forecast_steps)
    return results


def evaluate_baselines(
    storms: Sequence[Sequence[Sequence[float]]],
    history_steps: int,
    forecast_steps: int,
    model_forecasts: Sequence[np.ndarray] | None = None,
    model_name: str = "model",
    baselines: Iterable[str] | None = None,
    metrics: Iterable[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Run baselines over multiple storms and summarise metrics.

    Args:
        storms: Sequence of complete storm tracks. Each track should contain
            ``history_steps + forecast_steps`` entries with ``(lat, lon, intensity)``.
        history_steps: Number of initial steps to use as history for forecasting.
        forecast_steps: Number of forecast steps to evaluate.
        model_forecasts:
            Optional sequence of model forecast arrays with shape
            ``(forecast_steps, 3)`` matching the storms. When provided, metrics
            are computed for this model alongside the baselines.
        model_name:
            Name under which to report model metrics. Defaults to ``"model"``.
        baselines:
            Iterable of baseline names. Defaults to
            ``configs/default_config.yaml::evaluation.baselines``.
        metrics:
            Iterable of metric names. Defaults to
            ``configs/default_config.yaml::evaluation.metrics``.

    Returns:
        Dictionary mapping baseline name to a dictionary of mean metric values
        across all storms.
    """
    baselines = baselines or DEFAULT_BASELINES
    metrics = metrics or DEFAULT_METRICS
    summary: Dict[str, Dict[str, list[float]]] = {
        b: {m: [] for m in metrics} for b in baselines
    }
    if model_forecasts is not None:
        if len(model_forecasts) != len(storms):
            raise ValueError("model_forecasts must match number of storms")
        summary[model_name] = {m: [] for m in metrics}

    for idx, storm in enumerate(storms):
        storm_arr = np.asarray(storm)
        history = storm_arr[:history_steps]
        truth = storm_arr[history_steps : history_steps + forecast_steps]
        forecasts = run_baselines(history, forecast_steps, baselines)
        for name, pred in forecasts.items():
            results = compute_metrics(
                pred[:, :2].tolist(),
                truth[:, :2].tolist(),
                pred[:, 2].tolist(),
                truth[:, 2].tolist(),
                metrics,
            )
            for metric_name, value in results.items():
                summary[name][metric_name].append(value)

        if model_forecasts is not None:
            pred = model_forecasts[idx]
            results = compute_metrics(
                pred[:, :2].tolist(),
                truth[:, :2].tolist(),
                pred[:, 2].tolist(),
                truth[:, 2].tolist(),
                metrics,
            )
            for metric_name, value in results.items():
                summary[model_name][metric_name].append(value)

    return {
        name: {m: float(np.mean(vals)) for m, vals in metrics_dict.items()}
        for name, metrics_dict in summary.items()
    }
