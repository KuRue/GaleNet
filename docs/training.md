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

## GraphCast Model

GaleNet ships with a thin wrapper around DeepMind's GraphCast weather model.
To use it during training, point the configuration to a pre-trained checkpoint
and specify whether the GraphCast backbone should be frozen:

```yaml
model:
  name: graphcast
  graphcast:
    checkpoint_path: "models/graphcast/params.npz"
    freeze_backbone: true  # set to false to fine‑tune GraphCast
training:
  include_era5: true  # provide ERA5 fields for GraphCast
```

The dataset must supply ERA5 atmospheric variables at 0.25° resolution, which
the pipeline passes to GraphCast as input.  When `freeze_backbone` is true the
GraphCast weights remain fixed and only GaleNet's head is optimized; setting it
to false enables full fine‑tuning.

### Example: GraphCast-backed Training

Run a GraphCast-supported training session with:

```bash
python scripts/train_model.py model.name=graphcast \
    model.graphcast.checkpoint_path=/path/to/params.npz \
    model.graphcast.freeze_backbone=false \
    training.include_era5=true training.epochs=5
```

This command fine‑tunes GraphCast while training GaleNet's forecasting head.

## Pangu-Weather Model

The Pangu-Weather backbone offers an alternative physics-informed encoder. It
consumes 3D ERA5 cubes containing wind, temperature, humidity, and geopotential
fields on 13 pressure levels along with surface variables.

### Required ERA5 Variables

Ensure the data pipeline requests the following fields:

```yaml
data:
  era5:
    variables:
      - "geopotential"
      - "temperature"
      - "u_component_of_wind"
      - "v_component_of_wind"
      - "specific_humidity"
      - "mean_sea_level_pressure"
      - "10m_u_component_of_wind"
      - "10m_v_component_of_wind"
      - "2m_temperature"
    pressure_levels: [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50]
```

### Example: Pangu-backed Training

Run a Pangu-supported training session with:

```bash
python scripts/train_model.py model.name=pangu \
    model.pangu.checkpoint_path=/path/to/pangu_weights.npz \
    model.pangu.freeze_backbone=false \
    training.include_era5=true training.epochs=5
```

This command fine‑tunes Pangu while training GaleNet's forecasting head.
