import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pandas_spec = importlib.util.find_spec("pandas")
numpy_spec = importlib.util.find_spec("numpy")

pytestmark = pytest.mark.skipif(
    pandas_spec is None or numpy_spec is None,
    reason="pandas and numpy are required",
)

if pandas_spec and numpy_spec:  # pragma: no branch
    import numpy as np
    import pandas as pd

STORM_DATA_CODE = textwrap.dedent(
    """
import pandas as pd

SYNTHETIC_STORMS = {
    "STORM1": [
        {"latitude": 10.0, "longitude": 20.0, "max_wind": 30.0, "timestamp": 0},
        {"latitude": 11.0, "longitude": 21.0, "max_wind": 35.0, "timestamp": 1},
        {"latitude": 12.0, "longitude": 22.0, "max_wind": 40.0, "timestamp": 2},
        {"latitude": 13.0, "longitude": 23.0, "max_wind": 45.0, "timestamp": 3},
        {"latitude": 14.0, "longitude": 24.0, "max_wind": 50.0, "timestamp": 4},
    ]
}

class HurricaneDataPipeline:
    def __init__(self, *args, **kwargs):
        pass

    def load_hurricane_for_training(self, storm_id, include_era5=False):
        return {"track": pd.DataFrame(SYNTHETIC_STORMS[storm_id])}

class _Model:
    def predict(self, df_hist, forecast_steps, n_members):
        last = df_hist.iloc[-1]
        preds = pd.DataFrame([last] * forecast_steps, columns=df_hist.columns)
        return preds

class GaleNetPipeline:
    def __init__(self, config=None):
        self.model = _Model()
"""
)

BASELINES_CODE = textwrap.dedent(
    """
import numpy as np
from typing import Sequence, Dict

DEFAULT_BASELINES = ["persistence"]

def persistence_baseline(track: Sequence[Sequence[float]], forecast_steps: int) -> np.ndarray:
    track_arr = np.asarray(track)
    last = track_arr[-1]
    return np.repeat(last[None, :], forecast_steps, axis=0)

def run_baselines(
    track: Sequence[Sequence[float]],
    forecast_steps: int,
    baselines=None,
) -> Dict[str, np.ndarray]:
    baselines = baselines or DEFAULT_BASELINES
    return {name: persistence_baseline(track, forecast_steps) for name in baselines}
"""
)

METRICS_CODE = textwrap.dedent(
    """
import numpy as np
from typing import Sequence, Dict

DEFAULT_METRICS = ["track_error", "intensity_mae"]

def compute_metrics(
    track_pred: Sequence[Sequence[float]],
    track_truth: Sequence[Sequence[float]],
    intensity_pred: Sequence[float],
    intensity_truth: Sequence[float],
    metrics=None,
) -> Dict[str, float]:
    metrics = metrics or DEFAULT_METRICS
    track_pred = np.asarray(track_pred)
    track_truth = np.asarray(track_truth)
    intensity_pred = np.asarray(intensity_pred)
    intensity_truth = np.asarray(intensity_truth)
    results = {}
    if "track_error" in metrics:
        results["track_error"] = float(
            np.mean(np.linalg.norm(track_pred - track_truth, axis=1))
        )
    if "intensity_mae" in metrics:
        results["intensity_mae"] = float(
            np.mean(np.abs(intensity_pred - intensity_truth))
        )
    return results
"""
)


@pytest.mark.parametrize("suffix", ["csv", "json"])
def test_evaluate_baselines_cli(tmp_path, suffix):
    stub_root = tmp_path / "stub"
    pkg_eval = stub_root / "galenet" / "evaluation"
    pkg_eval.mkdir(parents=True)

    (stub_root / "galenet" / "__init__.py").write_text(STORM_DATA_CODE)
    (pkg_eval / "__init__.py").write_text("")
    (pkg_eval / "baselines.py").write_text(BASELINES_CODE)
    (pkg_eval / "metrics.py").write_text(METRICS_CODE)

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "evaluate_baselines.py"
    output = tmp_path / f"summary.{suffix}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(stub_root)

    cmd = [
        sys.executable,
        str(script),
        "STORM1",
        "--history",
        "3",
        "--forecast",
        "2",
        "--no-model",
        "--output",
        str(output),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    assert output.exists(), "summary file should be created"
    if suffix == "csv":
        df = pd.read_csv(output, index_col=0)
    else:
        df = pd.read_json(output, orient="index")

    # The summary contains one baseline with two metrics
    assert df.index.tolist() == ["persistence"]
    assert set(df.columns) == {"track_error", "intensity_mae"}

    expected_track_error = np.mean([np.sqrt(2), np.sqrt(8)])
    expected_intensity_mae = np.mean([5, 10])
    assert df.loc["persistence", "track_error"] == pytest.approx(expected_track_error)
    assert df.loc["persistence", "intensity_mae"] == pytest.approx(expected_intensity_mae)
