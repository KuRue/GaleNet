"""PyTorch-based training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    """Simple trainer handling optimization and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or nn.MSELoss()
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    def train(self, dataloader: DataLoader, epochs: int = 1) -> Iterator[float]:
        """Yield average loss for each epoch."""

        for _ in range(epochs):
            self.model.train()
            total = 0.0
            count = 0
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:  # pragma: no cover - defensive
                    inputs, targets = batch
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)

                self.optimizer.zero_grad()
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)
                loss.backward()
                self.optimizer.step()

                total += float(loss.item())
                count += 1
            yield total / max(count, 1)

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> None:
        """Persist model and optimizer state to ``path``."""

        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(ckpt, Path(path))

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model and optimizer state from ``path``."""

        ckpt = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


__all__ = ["Trainer"]
