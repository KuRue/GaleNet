# Model Architecture

GaleNet combines classic hurricane science with modern deep learning.

## Components

- **Data Pipeline** – normalizes HURDAT2/IBTrACS tracks and optionally merges
  ERA5 reanalysis patches.
- **Neural Network Core** – placeholder network used during prototyping; future
  versions will integrate GraphCast and Pangu‑Weather backbones.
- **Inference Pipeline** – the `GaleNetPipeline` class wraps preprocessing and
  model execution to deliver track forecasts.

## Design Principles

1. **Modularity** – data loading, model definition, and training utilities live
   in separate subpackages under `src/galenet`.
2. **Hydra Configuration** – experiment settings are described in YAML files and
   overridden via command‑line arguments.
3. **Reproducibility** – training scripts log checkpoints and configuration so
   experiments can be reproduced.
