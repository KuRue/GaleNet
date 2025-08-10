# flake8: noqa: E501
"""Tests for the GaleNet inference pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

# Ensure src is on the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from galenet.inference.pipeline import GaleNetPipeline  # noqa: E402
from galenet.models.graphcast import GraphCastModel  # noqa: E402


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


def test_graphcast_model_smoke(monkeypatch, tmp_path, sample_track):
    """Ensure the pipeline instantiates and runs the GraphCast model."""

    # Create a dummy checkpoint file for GraphCast
    ckpt_path = tmp_path / "params.npz"
    np.savez(ckpt_path, weights=np.array([1]))

    # Load base config and override model settings
    config = OmegaConf.load(CONFIG_PATH)
    config.model.name = "graphcast"
    config.model.graphcast.checkpoint_path = str(ckpt_path)

    monkeypatch.setattr(
        "galenet.inference.pipeline.get_config", lambda *args, **kwargs: config
    )

    class DummyDataPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def load_hurricane_for_training(self, storm_id, source="hurdat2", include_era5=False):
            return {"track": sample_track}

    monkeypatch.setattr(
        "galenet.inference.pipeline.HurricaneDataPipeline", DummyDataPipeline
    )

    pipeline = GaleNetPipeline(config_path=CONFIG_PATH)
    assert isinstance(pipeline.model, GraphCastModel)

    monkeypatch.setattr(
        pipeline.preprocessor, "normalize_track_data", lambda track, fit=False: track
    )
    monkeypatch.setattr(
        pipeline.model,
        "predict",
        lambda features, num_steps, step: _mock_model_predict(num_steps),
    )

    result = pipeline.forecast_storm("AL012023", forecast_hours=6)
    assert len(result.track) == len(sample_track) + 1
