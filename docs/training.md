# Training Guide

This guide demonstrates how to train GaleNet models with the included
PyTorch utilities. Ensure datasets are prepared as described in the
[Data Workflow](data_workflow.md) guide before starting.

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
`results/`. See the [Evaluation Guide](evaluation.md) for details.

## GraphCast Placeholder

Experimental hooks expose placeholder GraphCast weights for upcoming
integration. Configure the checkpoint in your YAML config:

```yaml
model:
  graphcast:
    checkpoint: "models/graphcast/params.npz"
    freeze_backbone: true
```

The weights are currently stubs and do not provide full GraphCast capability;
future releases will fuse these features into the training loop.
