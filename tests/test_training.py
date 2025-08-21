import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent / "src"))
torch = pytest.importorskip("torch")
sys.modules.pop("galenet.training", None)
from galenet.training import HurricaneDataset, Trainer, create_dataloader  # noqa: E402


class DummyPipeline:
    """Minimal stand-in for :class:`HurricaneDataPipeline`."""

    def load_hurricane_for_training(
        self,
        storm_id: str,
        include_era5: bool = True,
        include_satellite: bool = True,
        patch_size: float = 25.0,
    ):
        times = pd.date_range("2020-01-01", periods=5, freq="H")
        values = np.arange(1, 6, dtype=float)
        df = pd.DataFrame(
            {
                "timestamp": times,
                "latitude": values,
                "longitude": values,
                "max_wind": values,
                "min_pressure": values,
            }
        )
        data = {"track": df}
        lat = [0.0, 1.0]
        lon = [0.0, 1.0]
        shape = (len(times), 2, 2)
        if include_satellite:
            sat = xr.Dataset(
                {
                    "ir1": (
                        ("time", "latitude", "longitude"),
                        np.full(shape, 0.01, dtype=np.float32),
                    ),
                    "ir2": (
                        ("time", "latitude", "longitude"),
                        np.full(shape, 0.01, dtype=np.float32),
                    ),
                    "water_vapor": (
                        ("time", "latitude", "longitude"),
                        np.full(shape, 0.01, dtype=np.float32),
                    ),
                },
                coords={"time": times, "latitude": lat, "longitude": lon},
            )
            data["satellite"] = sat
        if include_era5:
            era5 = xr.Dataset(
                {
                    "u10": (
                        ("time", "latitude", "longitude"),
                        np.full(shape, 0.01, dtype=np.float32),
                    ),
                    "v10": (
                        ("time", "latitude", "longitude"),
                        np.full(shape, 0.01, dtype=np.float32),
                    ),
                    "msl": (
                        ("time", "latitude", "longitude"),
                        np.full(shape, 0.01, dtype=np.float32),
                    ),
                    "sst": (
                        ("time", "latitude", "longitude"),
                        np.full(shape, 0.01, dtype=np.float32),
                    ),
                },
                coords={"time": times, "latitude": lat, "longitude": lon},
            )
            data["era5"] = era5
        return data


def test_dataset_iteration() -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(pipeline, ["A", "B"], sequence_window=2, forecast_window=1)

    assert len(dataset) == 3
    patches, target = next(iter(dataset))
    assert patches.shape == (2, 2, 7, 2, 2)
    assert target.shape == (2, 1, 4)

    loader = create_dataloader(dataset, batch_size=1, shuffle=False)
    batch_patches, batch_target = next(iter(loader))
    assert batch_patches.shape == (1, 2, 2, 7, 2, 2)
    assert batch_target.shape == (1, 2, 1, 4)


def test_trainer_single_step_reduces_loss() -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(pipeline, ["TEST"], sequence_window=1, forecast_window=1)
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)

    model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=3), torch.nn.Linear(7 * 2 * 2, 4, bias=False)
    )
    torch.nn.init.zeros_(model[1].weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)

    batch_inputs, batch_targets = next(iter(loader))
    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(model(batch_inputs), batch_targets).item()

    metrics = list(trainer.train(loader, epochs=1))
    assert "train_loss" in metrics[0]

    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(batch_inputs), batch_targets).item()

    assert final < initial


def test_trainer_updates_weights() -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(pipeline, ["TEST"], sequence_window=1, forecast_window=1)
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)

    model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=3), torch.nn.Linear(7 * 2 * 2, 4, bias=False)
    )
    torch.nn.init.zeros_(model[1].weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)

    initial_weights = model[1].weight.detach().clone()
    list(trainer.train(loader, epochs=1))
    updated_weights = model[1].weight.detach()

    assert not torch.allclose(initial_weights, updated_weights)


def test_trainer_checkpoint_restore(tmp_path) -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(pipeline, ["TEST"], sequence_window=1, forecast_window=1)
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)

    model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=3), torch.nn.Linear(7 * 2 * 2, 4, bias=False)
    )
    torch.nn.init.zeros_(model[1].weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)

    list(trainer.train(loader, epochs=1))
    saved_weights = model[1].weight.detach().clone()
    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt, epoch=1)

    with torch.no_grad():
        model[1].weight.add_(1.0)

    list(trainer.train(loader, epochs=0, resume_from=ckpt))
    assert torch.allclose(model[1].weight, saved_weights)

    list(trainer.train(loader, epochs=1, start_epoch=1))
    assert not torch.allclose(model[1].weight, saved_weights)


def test_trainer_logs_metrics(tmp_path) -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(pipeline, ["TEST"], sequence_window=1, forecast_window=1)
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)

    model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=3), torch.nn.Linear(7 * 2 * 2, 4, bias=False)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics_file = tmp_path / "metrics.jsonl"
    trainer = Trainer(model, optimizer, metrics_file=metrics_file)

    list(trainer.train(loader, epochs=2))
    lines = metrics_file.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["epoch"] == 1 and "train_loss" in first


def test_trainer_validation_loop() -> None:
    pipeline = DummyPipeline()
    train_dataset = HurricaneDataset(pipeline, ["A"], sequence_window=1, forecast_window=1)
    val_dataset = HurricaneDataset(pipeline, ["B"], sequence_window=1, forecast_window=1)
    train_loader = create_dataloader(train_dataset, batch_size=1, shuffle=False)
    val_loader = create_dataloader(val_dataset, batch_size=1, shuffle=False)

    model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=3), torch.nn.Linear(7 * 2 * 2, 4, bias=False)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)

    metrics = list(trainer.train(train_loader, epochs=1, val_dataloader=val_loader))
    assert "val_loss" in metrics[0]
