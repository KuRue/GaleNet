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
    rapid_intensification_skill,
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

    intens_pred = np.array([40.0, 45.0, 50.0, 55.0, 65.0, 90.0])
    intens_true = np.array([40.0, 45.0, 50.0, 55.0, 70.0, 75.0])
    assert intensity_mae(intens_pred, intens_true) == pytest.approx(3.3333, rel=1e-4)
    assert rapid_intensification_skill(intens_pred, intens_true) == pytest.approx(
        2 / 3, rel=1e-4
    )

    results = compute_metrics(pred, truth, intens_pred, intens_true)
    assert set(
        [
            "track_error",
            "along_track_error",
            "cross_track_error",
            "intensity_mae",
            "rapid_intensification_skill",
        ]
    ) <= set(results.keys())
    assert results["intensity_mae"] == pytest.approx(3.3333, rel=1e-4)
    assert results["rapid_intensification_skill"] == pytest.approx(2 / 3, rel=1e-4)
