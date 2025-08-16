import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))
# Ensure any stubs from other tests are removed before importing
import torch
sys.modules.pop("galenet.training", None)
from galenet.training import HurricaneDataset, Trainer, create_dataloader


class DummyPipeline:
    """Minimal stand-in for :class:`HurricaneDataPipeline`."""

    def load_hurricane_for_training(
        self, storm_id: str, include_era5: bool = False, patch_size: float = 25.0
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
        if include_era5:
            # Simple 1D ERA5 feature for each timestep
            data["era5"] = np.arange(len(times), dtype=np.float32).reshape(len(times), 1)
        return data


def test_dataset_iteration() -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(
        pipeline, ["TEST"], sequence_window=2, forecast_window=1, include_era5=False
    )

    assert len(dataset) == 3
    seq, target = next(iter(dataset))
    assert seq.shape == (2, 4)
    assert target.shape == (1, 4)

    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batch_seq, batch_target = next(iter(loader))
    assert batch_seq.shape == (2, 2, 4)
    assert batch_target.shape == (2, 1, 4)


def test_dataset_with_era5() -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(
        pipeline, ["TEST"], sequence_window=2, forecast_window=1, include_era5=True
    )

    seq, target, patch = next(iter(dataset))
    assert patch.shape == (1, 1)

    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batch_seq, batch_target, batch_patch = next(iter(loader))
    assert batch_patch.shape == (2, 1, 1)


def test_trainer_single_step_reduces_loss() -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(
        pipeline, ["TEST"], sequence_window=1, forecast_window=1, include_era5=False
    )
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 4, bias=False))
    torch.nn.init.zeros_(model[1].weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)

    batch_inputs, batch_targets = next(iter(loader))
    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(
            model(batch_inputs), batch_targets
        ).item()

    list(trainer.train(loader, epochs=1))

    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(batch_inputs), batch_targets).item()

    assert final < initial


def test_trainer_updates_weights() -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(
        pipeline, ["TEST"], sequence_window=1, forecast_window=1, include_era5=False
    )
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 4, bias=False))
    torch.nn.init.zeros_(model[1].weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)

    initial_weights = model[1].weight.detach().clone()
    list(trainer.train(loader, epochs=1))
    updated_weights = model[1].weight.detach()

    assert not torch.allclose(initial_weights, updated_weights)


def test_trainer_checkpoint_restore(tmp_path) -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(
        pipeline, ["TEST"], sequence_window=1, forecast_window=1, include_era5=False
    )
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 4, bias=False))
    torch.nn.init.zeros_(model[1].weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)

    list(trainer.train(loader, epochs=1))
    saved_weights = model[1].weight.detach().clone()
    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt, epoch=1)

    with torch.no_grad():
        model[1].weight.add_(1.0)

    epoch = trainer.load_checkpoint(ckpt)
    assert epoch == 1
    assert torch.allclose(model[1].weight, saved_weights)

    list(trainer.train(loader, epochs=1, start_epoch=epoch))
    assert not torch.allclose(model[1].weight, saved_weights)

