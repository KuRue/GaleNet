"""Tests for GraphCastModel when official graphcast package is available."""

# flake8: noqa

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is on the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import galenet.models.graphcast as gc  # type: ignore


class _DummyModel:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x + 1.0


class _DummyGraphCastModule:
    @staticmethod
    def load_checkpoint(path: Path) -> _DummyModel:  # type: ignore[override]
        return _DummyModel()


def test_inference_without_linear_params(monkeypatch, tmp_path):
    ckpt_path = tmp_path / "params.npz"
    np.savez(ckpt_path, something=np.array([1], dtype=np.float32))

    monkeypatch.setattr(gc, "dm_graphcast", _DummyGraphCastModule)
    monkeypatch.setattr(gc, "_GRAPHCAST_AVAILABLE", True)

    model = gc.GraphCastModel(str(ckpt_path))
    arr = np.zeros((1, 4), dtype=np.float32)
    out = model.infer(arr)
    assert np.allclose(out, arr + 1.0)
    assert model._w is None and model._b is None

    features = pd.DataFrame(
        {
            "latitude": [0.0],
            "longitude": [0.0],
            "max_wind": [0.0],
            "min_pressure": [0.0],
        }
    )
    with pytest.raises(RuntimeError, match="requires linear parameters"):
        model.predict(features, num_steps=1, step=6)
