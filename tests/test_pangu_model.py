import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Ensure src is on path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import galenet.models.pangu as pangu  # noqa: E402


def _setup_dummy(monkeypatch, tmp_path):
    class DummyPanguWeather:
        def load_model(self, checkpoint):
            return lambda arr: arr + 1

    monkeypatch.setattr(pangu, "_PANGU_AVAILABLE", True)
    monkeypatch.setattr(pangu, "dm_pangu", DummyPanguWeather())

    ckpt = tmp_path / "dummy.ckpt"
    ckpt.write_text("checkpoint")
    return ckpt


def test_infer_numpy(monkeypatch, tmp_path):
    ckpt = _setup_dummy(monkeypatch, tmp_path)
    model = pangu.PanguModel(ckpt)
    arr = np.zeros((2, 2), dtype=np.float32)
    out = model.infer(arr)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, arr + 1)


def test_predict_xarray(monkeypatch, tmp_path):
    ckpt = _setup_dummy(monkeypatch, tmp_path)
    model = pangu.PanguModel(ckpt)
    data = xr.DataArray(np.zeros((1, 2), dtype=np.float32), dims=("x", "y"))
    out = model.predict(data, num_steps=2, step=6)
    assert isinstance(out, xr.DataArray)
    assert np.array_equal(out.values, data.values + 2)
