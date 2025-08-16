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

## Evaluation

After training completes, run the baseline evaluation script to compare GaleNet
against reference models:

```bash
python scripts/evaluate_baselines.py data/sample_storms.json \
  --history 3 --forecast 2 --model-config configs/default_config.yaml
```

The command reports track and intensity errors and writes a summary under
`results/`.

## GraphCast Integration Notes

GaleNet can initialize weights from the GraphCast model for transfer learning.
Configure the GraphCast checkpoint in your YAML config:

```yaml
model:
  graphcast:
    checkpoint: "models/graphcast/params.npz"
    freeze_backbone: true
```

During training, GraphCast features are fused with GaleNet's storm-specific
heads, enabling rapid fine-tuning on hurricane data.
