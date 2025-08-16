import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent / "src"))
from galenet.training import HurricaneDataset, Trainer, create_dataloader


class DummyPipeline:
    """Minimal stand-in for :class:`HurricaneDataPipeline`."""

    def load_hurricane_for_training(self, storm_id, include_era5=False, patch_size: float = 25.0):
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
        return {"track": df}


def test_dataset_iteration() -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(pipeline, ["TEST"], window=2, include_era5=False)

    assert len(dataset) == 3
    seq, target = dataset[0]
    assert seq.shape == (2, 4)
    assert target.shape == (4,)

    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    batch_seq, batch_target = next(iter(loader))
    assert batch_seq.shape == (2, 2, 4)
    assert batch_target.shape == (2, 4)


def test_trainer_single_step_reduces_loss() -> None:
    pipeline = DummyPipeline()
    dataset = HurricaneDataset(pipeline, ["TEST"], window=1, include_era5=False)
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 4, bias=False))
    torch.nn.init.zeros_(model[1].weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)

    batch_inputs, batch_targets = next(iter(loader))
    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(model(batch_inputs), batch_targets).item()

    list(trainer.train(loader, epochs=1))

    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(batch_inputs), batch_targets).item()

    assert final < initial


def test_trainer_updates_weights() -> None:
    """Weights should change after a training epoch."""

    pipeline = DummyPipeline()
    dataset = HurricaneDataset(pipeline, ["TEST"], window=1, include_era5=False)
    loader = create_dataloader(dataset, batch_size=1, shuffle=False)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 4, bias=False))
    torch.nn.init.zeros_(model[1].weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)

    initial_weights = model[1].weight.detach().clone()
    list(trainer.train(loader, epochs=1))
    updated_weights = model[1].weight.detach()

    assert not torch.allclose(initial_weights, updated_weights)
