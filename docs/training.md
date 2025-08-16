# Training Guide

This guide demonstrates how to train GaleNet models with the included
PyTorch utilities.

## Quick Start

Run the default training script with Hydra configuration:

```bash
python scripts/train_model.py training.epochs=5 training.batch_size=8
```

The script performs the following:

1. Loads a sample storm via `HurricaneDataPipeline`.
2. Builds a simple neural network and optimizer.
3. Logs loss values per epoch and saves checkpoints under `checkpoints/`.

## Customizing

Override configuration options on the command line, for example:

```bash
python scripts/train_model.py training.epochs=10 training.learning_rate=1e-4
```

Configuration files live in the `configs/` directory. See `default_config.yaml`
for all available parameters.
