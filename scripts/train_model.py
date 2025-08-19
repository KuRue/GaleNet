#!/usr/bin/env python
"""Command line interface for training GaleNet models with PyTorch."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch

from galenet.data import HurricaneDataPipeline
from galenet.training import HurricaneDataset, Trainer, create_dataloader

log = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> torch.nn.Module:
    """Instantiate a model based on ``cfg.model.name``.

    The helper returns wrappers around GraphCast or Pangu models when requested.
    For unknown model names a simple linear baseline is returned.
    """

    name = cfg.model.get("name", "").lower()
    if name == "graphcast":
        from galenet.models import GraphCastModel

        class _GraphCastModule(torch.nn.Module):
            def __init__(self, checkpoint: str) -> None:
                super().__init__()
                self.inner = GraphCastModel(checkpoint)
                # Dummy parameter so optimizers have something to work with
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(
                self, _tracks: torch.Tensor, era5: torch.Tensor
            ) -> torch.Tensor:  # pragma: no cover - heavy model
                import numpy as np

                out = self.inner.infer(era5.detach().cpu().numpy())
                return torch.from_numpy(np.asarray(out, dtype=np.float32))

        ckpt = cfg.model.graphcast.get("checkpoint_path", "")
        return _GraphCastModule(ckpt)

    if name == "pangu":
        from galenet.models import PanguModel

        class _PanguModule(torch.nn.Module):
            def __init__(self, checkpoint: str) -> None:
                super().__init__()
                self.inner = PanguModel(checkpoint)
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(
                self, _tracks: torch.Tensor, era5: torch.Tensor
            ) -> torch.Tensor:  # pragma: no cover - heavy model
                import numpy as np

                out = self.inner.infer(era5.detach().cpu().numpy())
                return torch.from_numpy(np.asarray(out, dtype=np.float32))

        ckpt = cfg.model.pangu.get("checkpoint_path", "")
        return _PanguModule(ckpt)

    # Fallback simple baseline
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 4))


@hydra.main(version_base=None, config_path="../configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    """Train a simple model using configuration from Hydra."""

    logging.basicConfig(level=logging.INFO)

    model_name = cfg.model.get("name", "").lower()
    needs_era5 = model_name in {"graphcast", "pangu"}

    # Data -----------------------------------------------------------------
    storms = cfg.training.get("storms", ["AL012011"])
    pipeline = HurricaneDataPipeline()
    dataset = HurricaneDataset(
        pipeline,
        storms,
        sequence_window=cfg.training.get("sequence_window", 1),
        forecast_window=cfg.training.get("forecast_window", 1),
        include_era5=needs_era5 or cfg.training.get("include_era5", False),
    )
    loader = create_dataloader(
        dataset,
        batch_size=cfg.training.get("batch_size", 1),
        shuffle=cfg.training.get("shuffle", True),
    )

    val_loader = None
    val_storms = cfg.training.get("val_storms")
    if val_storms:
        val_dataset = HurricaneDataset(
            pipeline,
            val_storms,
            sequence_window=cfg.training.get("sequence_window", 1),
            forecast_window=cfg.training.get("forecast_window", 1),
            include_era5=needs_era5 or cfg.training.get("include_era5", False),
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=cfg.training.get("batch_size", 1),
            shuffle=False,
        )

    # Model/optimizer ------------------------------------------------------
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Checkpoint directory
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model,
        optimizer,
        device=cfg.project.get("device", "cpu"),
        grad_accum_steps=cfg.training.get("gradient_accumulation_steps", 1),
        metrics_file=ckpt_dir / cfg.get("metrics_file", "metrics.jsonl"),
    )

    # Training loop --------------------------------------------------------
    epochs = cfg.training.get("epochs", 1)
    resume = cfg.training.get("resume_from")
    early_cfg = cfg.training.get("early_stopping")
    patience = early_cfg.get("patience") if early_cfg else None
    min_delta = early_cfg.get("min_delta", 0.0) if early_cfg else 0.0
    best = float("inf")
    wait = 0
    for epoch, metrics in enumerate(
        trainer.train(
            loader,
            epochs=epochs,
            resume_from=resume,
            val_dataloader=val_loader,
        ),
        1,
    ):
        log.info(
            "epoch %d %s",
            epoch,
            " ".join(f"{k}={v:.6f}" for k, v in metrics.items()),
        )
        trainer.save_checkpoint(ckpt_dir / f"epoch_{epoch}.pt", epoch=epoch)
        if patience is not None:
            monitored = metrics.get("val_loss", metrics.get("train_loss", float("inf")))
            if monitored + min_delta < best:
                best = monitored
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    log.info("Early stopping triggered at epoch %d", epoch)
                    break


if __name__ == "__main__":
    main()

