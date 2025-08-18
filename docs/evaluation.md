# Evaluation Guide

This guide shows how to run baseline evaluations on prepared data.

## Prerequisites
- Data prepared as described in the [Data Workflow](data_workflow.md).
- (Optional) Trained model from the [Training Guide](training.md).

## Quick Start
Run the evaluation script with a sample storm list:

```bash
python scripts/evaluate_baselines.py data/sample_storms.json \
  --history 3 --forecast 2 --model-config configs/default_config.yaml
```

The command loads storms via `HurricaneDataPipeline`, executes baseline models,
and prints track error metrics. Results are written under `results/`.

## Next Steps
Use the generated metrics to compare model variants or to validate training
runs. For details on configuring the data pipeline, see the
[Data Pipeline](data_pipeline.md) reference.

