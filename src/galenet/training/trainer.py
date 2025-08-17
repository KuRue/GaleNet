"""PyTorch-based training utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    """Trainer handling optimization, logging and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module | None = None,
        device: torch.device | str = "cpu",
        grad_accum_steps: int = 1,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or nn.MSELoss()
        self.device = torch.device(device)
        self.grad_accum_steps = int(max(1, grad_accum_steps))
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 1,
        start_epoch: int = 0,
        resume_from: str | Path | None = None,
    ) -> Iterator[dict]:
        """Yield metrics for each epoch.

        If ``resume_from`` is provided, the trainer will load the checkpoint
        before starting training and continue from the stored epoch.
        """

        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)

        for epoch in range(start_epoch, start_epoch + epochs):
            self.model.train()
            total = 0.0
            count = 0
            self.optimizer.zero_grad()
            for step, batch in enumerate(dataloader, 1):
                if isinstance(batch, (list, tuple)):
                    inputs, targets, *extras = batch
                else:  # pragma: no cover - defensive
                    inputs, targets, *extras = batch, None, []  # type: ignore[misc]
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)
                extras = [e.to(self.device, dtype=torch.float32) for e in extras]

                preds = self.model(inputs, *extras)
                loss = self.loss_fn(preds, targets) / self.grad_accum_steps
                loss.backward()

                if step % self.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total += float(loss.item()) * self.grad_accum_steps
                count += 1
            if count % self.grad_accum_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            avg = total / max(count, 1)
            metrics = {"loss": avg}
            self.logger.info(
                "epoch %d " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()),
                epoch + 1,
            )
            yield metrics

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path, epoch: int = 0) -> None:
        """Persist model and optimizer state to ``path``."""

        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": int(epoch),
        }
        torch.save(ckpt, Path(path))

    def load_checkpoint(self, path: str | Path) -> int:
        """Load model and optimizer state from ``path``.

        Returns the epoch stored in the checkpoint.
        """

        ckpt = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("epoch", 0))


__all__ = ["Trainer"]
