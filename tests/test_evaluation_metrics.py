import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))

from galenet.evaluation.metrics import (
    track_error,
    along_track_error,
    cross_track_error,
    intensity_mae,
    compute_metrics,
)


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(
        np.radians(lat2)
    ) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def test_metric_computations():
    truth = np.array([[0.0, 0.0], [0.0, 1.0]])
    pred = np.array([[0.0, 0.0], [1.0, 1.5]])
    expected_track = np.mean([
        0.0,
        _haversine(1.0, 1.5, 0.0, 1.0),
    ])
    assert track_error(pred, truth) == pytest.approx(expected_track, rel=1e-4)

    assert along_track_error(pred, truth) == pytest.approx(55.5, rel=0.02)
    assert cross_track_error(pred, truth) == pytest.approx(111.0, rel=0.02)

    intens_pred = np.array([50.0, 60.0])
    intens_true = np.array([52.0, 58.0])
    assert intensity_mae(intens_pred, intens_true) == pytest.approx(2.0)

    results = compute_metrics(pred, truth, intens_pred, intens_true)
    assert set(["track_error", "along_track_error", "cross_track_error", "intensity_mae"]) <= set(
        results.keys()
    )
    assert results["intensity_mae"] == pytest.approx(2.0)
