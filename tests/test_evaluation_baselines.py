import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from galenet.evaluation.baselines import run_baselines


def test_baseline_predictions():
    track = np.array([
        [0.0, 0.0, 40.0],
        [1.0, 1.0, 42.0],
        [2.0, 2.0, 44.0],
        [3.0, 3.0, 46.0],
        [4.0, 4.0, 48.0],
        [5.0, 5.0, 50.0],
    ])
    forecasts = run_baselines(track, forecast_steps=3, baselines=["persistence", "cliper5"])
    assert np.allclose(
        forecasts["persistence"],
        np.array([[5.0, 5.0, 50.0], [5.0, 5.0, 50.0], [5.0, 5.0, 50.0]]),
    )
    assert np.allclose(
        forecasts["cliper5"],
        np.array([[6.0, 6.0, 50.0], [7.0, 7.0, 50.0], [8.0, 8.0, 50.0]]),
    )
