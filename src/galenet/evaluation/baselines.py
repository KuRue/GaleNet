"""Baseline forecast methods for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default_config.yaml"
with open(_CONFIG_PATH) as _cfg:
    DEFAULT_BASELINES: Iterable[str] = yaml.safe_load(_cfg)["evaluation"]["baselines"]


def persistence_baseline(track: Sequence[Sequence[float]], forecast_steps: int) -> np.ndarray:
    """Simple persistence baseline: repeat last known position and intensity."""
    track_arr = np.asarray(track)
    last = track_arr[-1]
    return np.repeat(last[None, :], forecast_steps, axis=0)


def cliper5_baseline(track: Sequence[Sequence[float]], forecast_steps: int) -> np.ndarray:
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


BASELINE_FUNCTIONS = {
    "persistence": persistence_baseline,
    "cliper5": cliper5_baseline,
}


def run_baselines(
    track: Sequence[Sequence[float]],
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
