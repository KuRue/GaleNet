# Model Architecture

GaleNet combines classic hurricane science with modern deep learning.

## Components

- **Data Pipeline** – normalizes HURDAT2/IBTrACS tracks and optionally merges
  ERA5 reanalysis patches.
- **Neural Network Core** – supports GraphCast and Pangu‑Weather backbones for
  physics‑informed feature extraction. Pangu operates in inference mode only.
- **Inference Pipeline** – the `GaleNetPipeline` class wraps preprocessing and
  model execution, optionally blending outputs from multiple backbones (e.g.,
  GraphCast and Pangu) to guide track forecasts.

### Backbone Options

The backbone is selected via `model.name` in the Hydra configuration. Multiple
backbones may be specified to ensemble their predictions:

- `graphcast` uses DeepMind's GraphCast weights and expects 0.25° ERA5 patches.
- `pangu` enables Microsoft's Pangu‑Weather transformer for inference. It
  consumes 3D ERA5 cubes with wind, temperature, humidity, and geopotential
  fields. The pretrained weights remain fixed; fine‑tuning is not supported.

Listing both names will blend their forecasts:

```yaml
model:
  name: [graphcast, pangu]
```

## System Requirements

- **GPU** – GraphCast and Pangu backbones expect an NVIDIA GPU with at least
  16 GB of memory. The repository falls back to CPU when no CUDA device is
  detected (`nvidia-smi` absent).
- **Storage** – Downloaded datasets and training checkpoints require
  roughly 50 GB of disk space; ensure at least ~25 GB free before running
  experiments. The current container reports about 22 GB available.

## Design Principles

1. **Modularity** – data loading, model definition, and training utilities live
   in separate subpackages under `src/galenet`.
2. **Hydra Configuration** – experiment settings are described in YAML files and
   overridden via command‑line arguments.
3. **Reproducibility** – training scripts log checkpoints and configuration so
   experiments can be reproduced.

See the [Training Guide](training.md) for how models are trained and the
[Evaluation Guide](evaluation.md) for benchmarking approaches.
