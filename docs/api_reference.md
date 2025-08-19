# API Reference

Key classes and functions exposed at the package level.

## Data Loading

- `HurricaneDataPipeline` – orchestrates loading and preprocessing of track and
  ERA5 data.
- `HURDAT2Loader` / `IBTrACSLoader` / `ERA5Loader` – individual dataset
  interfaces.

## Training Utilities

- `HurricaneDataset` – wraps a pipeline to produce PyTorch tensors.
- `Trainer` – lightweight training loop with checkpoint saving.
- `mse_loss` – default loss function used in examples.

## Inference

- `GaleNetPipeline` – high‑level interface for generating hurricane forecasts.

Refer to the source code in `src/galenet/` for full details on each component.

See the [Data Pipeline](data_pipeline.md), [Training Guide](training.md), and
[Evaluation Guide](evaluation.md) for end-to-end examples.
