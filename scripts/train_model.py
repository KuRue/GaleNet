#!/usr/bin/env python
"""Command line interface for training GaleNet models."""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add repository root/src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from galenet.data import HurricaneDataPipeline
from galenet.models import GraphCastModel
from galenet.training import HurricaneDataset, Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GaleNet model")
    parser.add_argument("storm_id", help="Storm identifier for training data")
    parser.add_argument("checkpoint", help="Path to GraphCast checkpoint (.npz)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    pipeline = HurricaneDataPipeline()
    dataset = HurricaneDataset(pipeline, [args.storm_id])
    model = GraphCastModel(args.checkpoint)
    trainer = Trainer(model, dataset, learning_rate=args.lr)

    for epoch, loss in enumerate(trainer.train(args.epochs), 1):
        print(f"Epoch {epoch}: loss={loss:.6f}")

    np.savez(args.checkpoint, w=model.w, b=model.b)


if __name__ == "__main__":
    main()
