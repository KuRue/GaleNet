# flake8: noqa: E501
"""Tests for the GaleNet inference pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import OmegaConf

# Ensure src is on the path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from galenet.inference.pipeline import GaleNetPipeline  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml"


@pytest.fixture
def sample_track():
    """Create a minimal hurricane track DataFrame."""
    dates = pd.date_range("2023-01-01", periods=4, freq="6H")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "latitude": np.linspace(10, 13, 4),
            "longitude": np.linspace(-50, -47, 4),
            "max_wind": [30, 35, 40, 45],
            "min_pressure": [1005, 1000, 995, 990],
            "storm_id": ["AL012023"] * 4,
            "name": ["TEST"] * 4,
        }
    )


@pytest.fixture(autouse=True)
def _allow_env_resolver(monkeypatch):
    """Allow repeated registration of the 'env' resolver."""
    original = OmegaConf.register_new_resolver

    def register(name, resolver, *, replace=False, use_cache=False):
        return original(name, resolver, replace=True, use_cache=use_cache)

    monkeypatch.setattr(OmegaConf, "register_new_resolver", register)


def _mock_model_predict(num_steps):
    """Return a simple set of predictions for mocking."""
    return pd.DataFrame(
        {
            "latitude": np.full(num_steps, 20.0),
            "longitude": np.full(num_steps, -70.0),
            "max_wind": np.full(num_steps, 50),
            "min_pressure": np.full(num_steps, 950),
        }
    )


def test_forecast_length_matches_hours(monkeypatch, sample_track):
    """Forecast length should match the requested forecast horizon."""

    class DummyDataPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def load_hurricane_for_training(self, storm_id, source="hurdat2", include_era5=False):
            return {"track": sample_track}

    monkeypatch.setattr(
        "galenet.inference.pipeline.HurricaneDataPipeline", DummyDataPipeline
    )

    config = OmegaConf.load(CONFIG_PATH)
    config.model.name = "hurricane_ensemble"
    monkeypatch.setattr(
        "galenet.inference.pipeline.get_config", lambda *a, **k: config
    )
    pipeline = GaleNetPipeline(config_path=CONFIG_PATH)
    monkeypatch.setattr(
        pipeline.model,
        "predict",
        lambda features, num_steps, step: _mock_model_predict(num_steps),
    )
    monkeypatch.setattr(
        pipeline.preprocessor,
        "normalize_track_data",
        lambda track, fit=False: track,
    )

    result = pipeline.forecast_storm("AL012023", forecast_hours=12)

    # With a 6-hour step, 12 hours should yield 2 forecast points
    assert len(result.track) == len(sample_track) + 2


def test_lead_times_and_validation_warning(monkeypatch, sample_track):
    """New lead times should be appended and validation warnings issued."""

    class DummyDataPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def load_hurricane_for_training(self, storm_id, source="hurdat2", include_era5=False):
            return {"track": sample_track}

    monkeypatch.setattr(
        "galenet.inference.pipeline.HurricaneDataPipeline", DummyDataPipeline
    )

    config = OmegaConf.load(CONFIG_PATH)
    config.model.name = "hurricane_ensemble"
    monkeypatch.setattr(
        "galenet.inference.pipeline.get_config", lambda *a, **k: config
    )
    pipeline = GaleNetPipeline(config_path=CONFIG_PATH)
    monkeypatch.setattr(
        pipeline.model,
        "predict",
        lambda features, num_steps, step: _mock_model_predict(num_steps),
    )
    monkeypatch.setattr(
        pipeline.preprocessor,
        "normalize_track_data",
        lambda track, fit=False: track,
    )

    # Mock validator: first call (historical) passes, second (forecast) fails
    calls = {"n": 0}

    def mock_validate(df):
        calls["n"] += 1
        return (True, []) if calls["n"] == 1 else (False, ["bad track"])

    monkeypatch.setattr(pipeline.validator, "validate_track", mock_validate)

    warnings = []

    def fake_warning(msg, *args, **kwargs):
        warnings.append(msg.format(*args, **kwargs))

    monkeypatch.setattr("galenet.inference.pipeline.logger.warning", fake_warning)

    result = pipeline.forecast_storm("AL012023", forecast_hours=12)

    # Lead times should continue from the historical data in 6-hour steps
    assert list(result.track["lead_time"].tail(2).astype(int)) == [24, 30]
    # The validator failure should trigger a warning
    assert any("bad track" in w for w in warnings)


def test_graphcast_pipeline_environment_output(monkeypatch, tmp_path, sample_track):
    """Pipeline should pass ERA5 fields to GraphCast and store the outputs."""

    ckpt_path = tmp_path / "params.npz"
    np.savez(ckpt_path, dummy=np.array([1], dtype=np.float32))

    config = OmegaConf.load(CONFIG_PATH)
    config.model.name = "graphcast"
    config.model.graphcast.checkpoint_path = str(ckpt_path)
    config.inference.post_process.smooth_track = False

    monkeypatch.setattr(
        "galenet.inference.pipeline.get_config", lambda *args, **kwargs: config
    )

    era5 = xr.DataArray(
        np.zeros((1, 2, 2), dtype=np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": [0], "lat": [0.0, 0.25], "lon": [0.0, 0.25]},
    )

    class DummyDataPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def load_hurricane_for_training(self, storm_id, source="hurdat2", include_era5=False):
            assert include_era5 is True
            return {"track": sample_track, "era5": era5}

    monkeypatch.setattr(
        "galenet.inference.pipeline.HurricaneDataPipeline", DummyDataPipeline
    )

    import galenet.models.graphcast as gm

    class DummyModel:
        def __call__(self, x: np.ndarray) -> np.ndarray:
            return x + 1.0

    class DummyGraphCastModule:
        @staticmethod
        def load_checkpoint(path):  # type: ignore[override]
            return DummyModel()

    monkeypatch.setattr(gm, "dm_graphcast", DummyGraphCastModule)
    monkeypatch.setattr(gm, "_GRAPHCAST_AVAILABLE", True)

    pipeline = GaleNetPipeline(config_path=CONFIG_PATH)

    # Simplify preprocessing and validation
    monkeypatch.setattr(
        pipeline.preprocessor, "normalize_track_data", lambda track, fit=False: track
    )
    monkeypatch.setattr(
        pipeline.preprocessor, "create_track_features", lambda df: df
    )
    monkeypatch.setattr(pipeline.validator, "validate_track", lambda df: (True, []))

    result = pipeline.forecast_storm("AL012023", forecast_hours=12)

    # Environmental fields should be returned and transformed
    assert isinstance(result.fields, xr.DataArray)
    assert np.allclose(result.fields, era5 + 1.0)

    # Track forecast falls back to persistence
    assert len(result.track) == len(sample_track) + 2


def test_graphcast_numpy_and_xarray_inference(monkeypatch, tmp_path):
    """Model inference should work for NumPy arrays and xarray DataArrays."""

    ckpt_path = tmp_path / "params.npz"
    np.savez(ckpt_path, dummy=np.array([1], dtype=np.float32))

    import galenet.models.graphcast as gc

    class DummyModel:
        def __call__(self, x: np.ndarray) -> np.ndarray:
            return x + 1.0

    class DummyGraphCastModule:
        @staticmethod
        def load_checkpoint(path) -> DummyModel:  # type: ignore[override]
            return DummyModel()

    monkeypatch.setattr(gc, "dm_graphcast", DummyGraphCastModule)
    monkeypatch.setattr(gc, "_GRAPHCAST_AVAILABLE", True)

    model = gc.GraphCastModel(str(ckpt_path))

    arr = np.zeros((2, 4), dtype=np.float32)
    out = model.infer(arr)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, arr + 1.0)

    da = xr.DataArray(arr, dims=["t", "c"])
    da_out = model.infer(da)
    assert isinstance(da_out, xr.DataArray)
    assert np.allclose(da_out.values, arr + 1.0)
