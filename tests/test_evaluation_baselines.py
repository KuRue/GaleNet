import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent / "src" / "galenet"))
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from evaluation.baselines import evaluate_baselines, run_baselines
from evaluation.metrics import compute_metrics
import evaluate_baselines as eval_cli


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


def test_evaluate_baselines_with_model():
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

    model_forecasts = [
        np.array([[2.0, 2.0, 42.0], [3.0, 3.0, 42.0]]),
        np.array([[12.0, 10.0, 34.0], [13.0, 10.0, 34.0]]),
    ]

    summary = evaluate_baselines(
        storms,
        history_steps=2,
        forecast_steps=2,
        baselines=["persistence"],
        metrics=["track_error", "intensity_mae"],
        model_forecasts=model_forecasts,
        model_name="model",
    )

    # Baseline metrics
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

    # Model metrics
    m1_model = compute_metrics(
        model_forecasts[0][:, :2],
        storms[0][2:, :2],
        model_forecasts[0][:, 2],
        storms[0][2:, 2],
        metrics=["track_error", "intensity_mae"],
    )

    m2_model = compute_metrics(
        model_forecasts[1][:, :2],
        storms[1][2:, :2],
        model_forecasts[1][:, 2],
        storms[1][2:, 2],
        metrics=["track_error", "intensity_mae"],
    )

    expected_track_model = (m1_model["track_error"] + m2_model["track_error"]) / 2
    expected_intensity_model = (
        m1_model["intensity_mae"] + m2_model["intensity_mae"]
    ) / 2

    assert summary["model"]["track_error"] == pytest.approx(
        expected_track_model, rel=1e-4
    )
    assert summary["model"]["intensity_mae"] == pytest.approx(
    expected_intensity_model, rel=1e-4
    )


def test_cli_evaluate_baselines(tmp_path, monkeypatch):
    storms = {
        "S1": pd.DataFrame(
            [
                [0.0, 0.0, 40.0],
                [1.0, 1.0, 42.0],
                [2.0, 2.0, 44.0],
                [3.0, 3.0, 46.0],
                [4.0, 4.0, 48.0],
            ],
            columns=["latitude", "longitude", "max_wind"],
        ),
        "S2": pd.DataFrame(
            [
                [5.0, 5.0, 30.0],
                [6.0, 5.0, 32.0],
                [7.0, 5.0, 34.0],
                [8.0, 5.0, 36.0],
                [9.0, 5.0, 38.0],
            ],
            columns=["latitude", "longitude", "max_wind"],
        ),
    }

    class DummyDataPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def load_hurricane_for_training(self, storm_id, include_era5=False):
            return {"track": storms[storm_id].copy()}

    class DummyModel:
        def predict(self, df_hist, forecast, step):
            last = df_hist.iloc[-1][["latitude", "longitude", "max_wind"]].to_numpy()
            preds = np.repeat(last.reshape(1, 3), forecast, axis=0)
            return pd.DataFrame(preds, columns=["latitude", "longitude", "max_wind"])

    class DummyGaleNetPipeline:
        def __init__(self, *args, **kwargs):
            self.model = DummyModel()

    monkeypatch.setattr(eval_cli, "HurricaneDataPipeline", DummyDataPipeline)
    monkeypatch.setattr(eval_cli, "GaleNetPipeline", DummyGaleNetPipeline)

    # Run with CSV output
    csv_path = tmp_path / "summary.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_baselines.py",
            "S1",
            "S2",
            "--history",
            "3",
            "--forecast",
            "2",
            "--output",
            str(csv_path),
        ],
    )
    eval_cli.main()
    df_csv = pd.read_csv(csv_path, index_col=0)

    # Run with JSON output
    json_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_baselines.py",
            "S1",
            "S2",
            "--history",
            "3",
            "--forecast",
            "2",
            "--output",
            str(json_path),
        ],
    )
    eval_cli.main()
    df_json = pd.read_json(json_path, orient="index")

    tracks = [storms[sid].to_numpy() for sid in ["S1", "S2"]]
    model_forecasts = [
        run_baselines(t[:3], 2, baselines=["persistence"])["persistence"] for t in tracks
    ]
    expected = pd.DataFrame(
        evaluate_baselines(
            tracks,
            history_steps=3,
            forecast_steps=2,
            model_forecasts=model_forecasts,
            model_name="model",
        )
    ).T

    for df in [df_csv, df_json]:
        for name in ["persistence", "model"]:
            assert df.loc[name, "track_error"] == pytest.approx(
                expected.loc[name, "track_error"], rel=1e-4
            )
            assert df.loc[name, "intensity_mae"] == pytest.approx(
                expected.loc[name, "intensity_mae"], rel=1e-4
            )


def test_cli_evaluate_baselines_multi_model(tmp_path, monkeypatch):
    storms = {
        "S1": pd.DataFrame(
            [
                [0.0, 0.0, 40.0],
                [1.0, 1.0, 42.0],
                [2.0, 2.0, 44.0],
                [3.0, 3.0, 46.0],
                [4.0, 4.0, 48.0],
            ],
            columns=["latitude", "longitude", "max_wind"],
        ),
        "S2": pd.DataFrame(
            [
                [5.0, 5.0, 30.0],
                [6.0, 5.0, 32.0],
                [7.0, 5.0, 34.0],
                [8.0, 5.0, 36.0],
                [9.0, 5.0, 38.0],
            ],
            columns=["latitude", "longitude", "max_wind"],
        ),
    }

    class DummyDataPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def load_hurricane_for_training(self, storm_id, include_era5=False):
            return {"track": storms[storm_id].copy()}

    class DummyModel:
        def __init__(self, offset: float):
            self.offset = offset

        def predict(self, df_hist, forecast, step):
            last = df_hist.iloc[-1][["latitude", "longitude", "max_wind"]].to_numpy()
            preds = np.repeat(last.reshape(1, 3), forecast, axis=0)
            preds[:, 0] += self.offset
            preds[:, 1] += self.offset
            return pd.DataFrame(preds, columns=["latitude", "longitude", "max_wind"])

    class DummyGaleNetPipeline:
        def __init__(self, config, *args, **kwargs):
            self.model = DummyModel(float(config))

    monkeypatch.setattr(eval_cli, "HurricaneDataPipeline", DummyDataPipeline)
    monkeypatch.setattr(eval_cli, "GaleNetPipeline", DummyGaleNetPipeline)

    summary_path = tmp_path / "summary.csv"
    details_path = tmp_path / "details.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_baselines.py",
            "S1",
            "S2",
            "--history",
            "3",
            "--forecast",
            "2",
            "--model",
            "m1=0.0",
            "--model",
            "m2=1.0",
            "--output",
            str(summary_path),
            "--details",
            str(details_path),
        ],
    )

    eval_cli.main()

    df_summary = pd.read_csv(summary_path, index_col=0)
    df_details = pd.read_csv(details_path, index_col=[0, 1])

    tracks = [storms[sid].to_numpy() for sid in ["S1", "S2"]]
    baseline_forecasts = [
        run_baselines(t[:3], 2, baselines=["persistence"])["persistence"]
        for t in tracks
    ]
    model1_forecasts = baseline_forecasts
    model2_forecasts = [f + np.array([1.0, 1.0, 0.0]) for f in baseline_forecasts]

    expected_records = []
    for sid, track, b_pred, m1_pred, m2_pred in zip(
        ["S1", "S2"], tracks, baseline_forecasts, model1_forecasts, model2_forecasts
    ):
        truth = track[3:5]
        for name, pred in [
            ("persistence", b_pred),
            ("m1", m1_pred),
            ("m2", m2_pred),
        ]:
            res = compute_metrics(
                pred[:, :2], truth[:, :2], pred[:, 2], truth[:, 2]
            )
            rec = {"storm": sid, "forecast": name}
            rec.update(res)
            expected_records.append(rec)

    expected_details = pd.DataFrame(expected_records).set_index(["storm", "forecast"])
    expected_summary = expected_details.groupby("forecast").mean()

    for idx in expected_details.index:
        assert df_details.loc[idx, "track_error"] == pytest.approx(
            expected_details.loc[idx, "track_error"], rel=1e-4
        )
        assert df_details.loc[idx, "intensity_mae"] == pytest.approx(
            expected_details.loc[idx, "intensity_mae"], rel=1e-4
        )

    for idx in expected_summary.index:
        assert df_summary.loc[idx, "track_error"] == pytest.approx(
            expected_summary.loc[idx, "track_error"], rel=1e-4
        )
        assert df_summary.loc[idx, "intensity_mae"] == pytest.approx(
            expected_summary.loc[idx, "intensity_mae"], rel=1e-4
        )

