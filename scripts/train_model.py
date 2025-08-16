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


@hydra.main(version_base=None, config_path="../configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    """Train a simple model using configuration from Hydra."""

    logging.basicConfig(level=logging.INFO)

    # Data -----------------------------------------------------------------
    storm = cfg.get("storm_id", "AL012011")
    pipeline = HurricaneDataPipeline()
    dataset = HurricaneDataset(pipeline, [storm], window=1)
    loader = create_dataloader(dataset, batch_size=cfg.training.get("batch_size", 1), shuffle=True)

    # Model/optimizer ------------------------------------------------------
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 4))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    trainer = Trainer(model, optimizer)

    # Checkpoint directory
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop --------------------------------------------------------
    for epoch, loss in enumerate(trainer.train(loader, epochs=cfg.training.epochs), 1):
        log.info("epoch %d loss=%.6f", epoch, loss)
        trainer.save_checkpoint(ckpt_dir / f"epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
