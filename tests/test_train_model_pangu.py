import sys
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig

# Ensure src on path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import galenet.models.pangu as pangu  # noqa: E402

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location(
    "train_model", Path(__file__).parent.parent / "scripts" / "train_model.py"
)
train_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(train_model)  # type: ignore[attr-defined]


class DummyPipeline:
    def load_hurricane_for_training(self, storm_id, include_era5=True, patch_size=25.0):
        times = pd.date_range("2020-01-01", periods=3, freq="H")
        values = np.arange(3, dtype=float)
        track = pd.DataFrame(
            {
                "timestamp": times,
                "latitude": values,
                "longitude": values,
                "max_wind": values,
                "min_pressure": values,
            }
        )
        era5 = np.zeros((3, 4), dtype=np.float32)
        return {"track": track, "era5": era5}


def test_train_model_runs_one_epoch(monkeypatch, tmp_path):
    class DummyPanguWeather:
        def load_model(self, checkpoint):
            return lambda arr: arr

    monkeypatch.setattr(pangu, "_PANGU_AVAILABLE", True)
    monkeypatch.setattr(pangu, "dm_pangu", DummyPanguWeather())
    monkeypatch.setattr(train_model, "HurricaneDataPipeline", DummyPipeline)

    def dummy_build_model(cfg):
        class DummyModule(torch.nn.Module):
            def __init__(self, ckpt: str) -> None:
                super().__init__()
                self.inner = pangu.PanguModel(ckpt)
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, _tracks: torch.Tensor, era5: torch.Tensor) -> torch.Tensor:
                import numpy as np

                out = self.inner.infer(era5.detach().cpu().numpy())
                return torch.from_numpy(np.asarray(out, dtype=np.float32)) + self.dummy

        return DummyModule(cfg.model.pangu.checkpoint_path)

    monkeypatch.setattr(train_model, "build_model", dummy_build_model)

    ckpt = tmp_path / "dummy.ckpt"
    ckpt.write_text("checkpoint")

    cfg = DictConfig(
        {
            "model": {"name": "pangu", "pangu": {"checkpoint_path": str(ckpt)}},
            "training": {
                "storms": ["A"],
                "sequence_window": 1,
                "forecast_window": 1,
                "include_era5": True,
                "batch_size": 1,
                "shuffle": False,
                "epochs": 1,
                "learning_rate": 0.1,
            },
            "project": {"device": "cpu"},
            "checkpoint_dir": str(tmp_path),
            "metrics_file": "metrics.jsonl",
        }
    )

    train_model.main.__wrapped__(cfg)

    # Check that checkpoint and metrics were written
    assert (tmp_path / "epoch_1.pt").exists()
    assert (tmp_path / "metrics.jsonl").exists()
