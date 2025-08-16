"""Evaluation utilities for GaleNet."""

from .metrics import (
    track_error,
    along_track_error,
    cross_track_error,
    intensity_mae,
    compute_metrics,
    compute_metrics_multi,
)
from .baselines import (
    run_baselines,
    evaluate_baselines,
    persistence_baseline,
    cliper5_baseline,
)

__all__ = [
    "track_error",
    "along_track_error",
    "cross_track_error",
    "intensity_mae",
    "compute_metrics",
    "compute_metrics_multi",
    "run_baselines",
    "evaluate_baselines",
    "persistence_baseline",
    "cliper5_baseline",
]
