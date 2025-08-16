import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))

from galenet.evaluation.baselines import run_baselines, evaluate_baselines
from galenet.evaluation.metrics import compute_metrics


def test_baseline_predictions():
    track = np.array([
        [0.0, 0.0, 40.0],
        [1.0, 1.0, 42.0],
        [2.0, 2.0, 44.0],
        [3.0, 3.0, 46.0],
        [4.0, 4.0, 48.0],
        [5.0, 5.0, 50.0],
    ])
    forecasts = run_baselines(
        track,
        forecast_steps=3,
        baselines=["persistence", "cliper5", "gfs", "ecmwf"],
    )
    assert np.allclose(
        forecasts["persistence"],
        np.array([[5.0, 5.0, 50.0], [5.0, 5.0, 50.0], [5.0, 5.0, 50.0]]),
    )
    assert np.allclose(
        forecasts["cliper5"],
        np.array([[6.0, 6.0, 50.0], [7.0, 7.0, 50.0], [8.0, 8.0, 50.0]]),
    )
    assert np.allclose(
        forecasts["gfs"],
        np.array([[6.1, 6.1, 50.0], [7.2, 7.2, 50.0], [8.3, 8.3, 50.0]]),
    )
    assert np.allclose(
        forecasts["ecmwf"],
        np.array([[5.9, 5.9, 50.0], [6.8, 6.8, 50.0], [7.7, 7.7, 50.0]]),
    )


def test_gfs_ecmwf_fallback_to_persistence():
    track = np.array([[10.0, 20.0, 60.0]])
    forecasts = run_baselines(track, forecast_steps=2, baselines=["gfs", "ecmwf"])
    expected = np.array([[10.0, 20.0, 60.0], [10.0, 20.0, 60.0]])
    assert np.allclose(forecasts["gfs"], expected)
    assert np.allclose(forecasts["ecmwf"], expected)


def test_evaluate_baselines_multi_storm():
    storms = [
        np.array(
            [
                [0.0, 0.0, 40.0],
                [1.0, 1.0, 41.0],
                [2.0, 2.0, 42.0],
                [3.0, 3.0, 43.0],
            ]
        ),
        np.array(
            [
                [10.0, 10.0, 30.0],
                [11.0, 10.0, 32.0],
                [12.0, 10.0, 34.0],
                [13.0, 10.0, 36.0],
            ]
        ),
    ]

    summary = evaluate_baselines(
        storms,
        history_steps=2,
        forecast_steps=2,
        baselines=["persistence"],
        metrics=["track_error", "intensity_mae"],
    )

    # compute expected values manually
    f1 = run_baselines(storms[0][:2], 2, baselines=["persistence"])["persistence"]
    m1 = compute_metrics(
        f1[:, :2],
        storms[0][2:, :2],
        f1[:, 2],
        storms[0][2:, 2],
        metrics=["track_error", "intensity_mae"],
    )

    f2 = run_baselines(storms[1][:2], 2, baselines=["persistence"])["persistence"]
    m2 = compute_metrics(
        f2[:, :2],
        storms[1][2:, :2],
        f2[:, 2],
        storms[1][2:, 2],
        metrics=["track_error", "intensity_mae"],
    )

    expected_track = (m1["track_error"] + m2["track_error"]) / 2
    expected_intensity = (m1["intensity_mae"] + m2["intensity_mae"]) / 2

    assert summary["persistence"]["track_error"] == pytest.approx(expected_track, rel=1e-4)
    assert summary["persistence"]["intensity_mae"] == pytest.approx(
        expected_intensity, rel=1e-4
    )
