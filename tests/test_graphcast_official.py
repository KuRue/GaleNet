"""Tests for the GraphCastModel wrapper when the official package is present."""

# flake8: noqa

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Ensure src is on the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import galenet.models.graphcast as gc  # type: ignore


class _DummyModel:
    def __call__(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - simple op
        return x + 1.0


class _DummyGraphCastModule:
    @staticmethod
    def load_checkpoint(path: Path) -> _DummyModel:  # type: ignore[override]
        return _DummyModel()


def test_inference_with_real_package(monkeypatch, tmp_path):
    """GraphCastModel should delegate inference to the graphcast package."""

    ckpt_path = tmp_path / "params.npz"
    np.savez(ckpt_path, dummy=np.array([1], dtype=np.float32))

    monkeypatch.setattr(gc, "dm_graphcast", _DummyGraphCastModule)
    monkeypatch.setattr(gc, "_GRAPHCAST_AVAILABLE", True)

    model = gc.GraphCastModel(str(ckpt_path))

    arr = np.zeros((2, 4), dtype=np.float32)
    out = model.infer(arr)
    assert np.allclose(out, arr + 1.0)

    da = xr.DataArray(arr, dims=["t", "c"])
    da_out = model.infer(da)
    assert isinstance(da_out, xr.DataArray)
    assert np.allclose(da_out.values, arr + 1.0)

    # Autoregressive prediction simply chains calls to infer
    out2 = model.predict(arr, num_steps=2, step=6)
    assert np.allclose(out2, arr + 2.0)

    da_out2 = model.predict(da, num_steps=2, step=6)
    assert isinstance(da_out2, xr.DataArray)
    assert np.allclose(da_out2.values, arr + 2.0)
