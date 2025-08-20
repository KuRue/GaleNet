# Evaluation Guide

This guide shows how to evaluate baseline and model forecasts on prepared hurricane data.

## Prerequisites
- Data prepared as described in the [Data Workflow](data_workflow.md).
- (Optional) Trained model from the [Training Guide](training.md).

## Running `scripts/evaluate_baselines.py`
The CLI accepts one or more storm identifiers and evaluates a set of baselines. For example, save storm IDs in a text file:

```text
# sample_storms.txt
AL012022
AL022022
```

Run the evaluator using the storm list:

```bash
python scripts/evaluate_baselines.py $(cat sample_storms.txt) \
  --history 3 --forecast 2 \
  --model mymodel=configs/default_config.yaml \
  --output results/summary.csv
```

### Key arguments
- `--history`: number of initial time steps used as input history.
- `--forecast`: number of steps to forecast and evaluate.
- `--model`: optional model specification in `NAME=CONFIG` form. Repeat to compare multiple models.
- `--output`: optional path to save the summary table (`.csv` or `.json`).

## Baselines
The evaluator includes several simple forecasting baselines:
- **persistence** – repeats the last observed position and intensity【F:src/galenet/evaluation/baselines.py†L18-L24】
- **cliper5** – uses the mean motion over the last five steps【F:src/galenet/evaluation/baselines.py†L27-L47】
- **gfs** – applies a slightly accelerated motion (110% of recent mean)【F:src/galenet/evaluation/baselines.py†L50-L70】
- **ecmwf** – applies a slightly slower motion (90% of recent mean)【F:src/galenet/evaluation/baselines.py†L73-L93】

## Reported metrics
For each forecast the script reports:
- **track_error** – mean great-circle distance between predicted and true tracks (km)【F:src/galenet/evaluation/metrics.py†L32-L41】
- **along_track_error** – mean absolute error along the true track direction (km)【F:src/galenet/evaluation/metrics.py†L79-L86】
- **cross_track_error** – mean absolute error perpendicular to the true track (km)【F:src/galenet/evaluation/metrics.py†L89-L96】
- **intensity_mae** – mean absolute error in predicted intensity (kt)【F:src/galenet/evaluation/metrics.py†L99-L103】
- **rapid_intensification_skill** – F1 score for detecting rapid intensification events【F:src/galenet/evaluation/metrics.py†L106-L143】

## Custom metrics
The evaluator reads the default metric list from `evaluation.metrics` in
`configs/default_config.yaml` and computes the functions registered in
`METRIC_FUNCTIONS`【F:src/galenet/evaluation/metrics.py†L12-L15】【F:src/galenet/evaluation/metrics.py†L146-L152】.
To track additional metrics, implement a new function, add it to that
dictionary, and list its name in your configuration.

## Example output
Running the command above yields console output similar to:

```text
Per-storm metrics:
                      track_error  along_track_error  cross_track_error  intensity_mae  rapid_intensification_skill
storm    forecast
AL012022 persistence       23.588             31.450              0.000          3.000                        0.000
         cliper5            0.000              0.000              0.000          3.000                        0.000
         gfs                2.359              3.145              0.000          3.000                        0.000
         ecmwf              2.359              3.145              0.000          3.000                        0.000
AL022022 persistence       16.679             22.239              0.000          3.000                        0.000
         cliper5            0.000              0.000              0.000          3.000                        0.000
         gfs                1.668              2.224              0.000          3.000                        0.000
         ecmwf              1.668              2.224              0.000          3.000                        0.000

Summary:
             track_error  along_track_error  cross_track_error  intensity_mae  rapid_intensification_skill
forecast
cliper5            0.000              0.000              0.000          3.000                        0.000
ecmwf              2.013              2.684              0.000          3.000                        0.000
gfs                2.013              2.684              0.000          3.000                        0.000
persistence       20.134             26.845              0.000          3.000                        0.000
```

The summary table is also saved to `results/summary.csv` when `--output` is provided.

## Evaluating GraphCast
1. **Configure weights** – ensure `configs/default_config.yaml` contains a valid
   `model.graphcast.checkpoint_path` pointing to the official GraphCast weights.
2. **Run the evaluator**:

   ```bash
   python scripts/evaluate_baselines.py $(cat sample_storms.txt) \
     --history 3 --forecast 2 \
     --model graphcast=configs/default_config.yaml \
     --output results/graphcast.csv
   ```

3. **Inspect output** – the console shows an additional `graphcast` row and the
   summary is written to `results/graphcast.csv`:

   ```text
   Summary:
                track_error  along_track_error  cross_track_error  intensity_mae
   forecast
   graphcast           1.234             1.567              0.321          2.345
   ```

## Evaluating Pangu-Weather
1. **Configure weights** – update `model.pangu.checkpoint_path` in
   `configs/default_config.yaml` to the location of the Pangu-Weather
   checkpoint.
2. **Run the evaluator**:

   ```bash
   python scripts/evaluate_baselines.py $(cat sample_storms.txt) \
     --history 3 --forecast 2 \
     --model pangu=configs/default_config.yaml \
     --output results/pangu.csv
   ```

3. **Inspect output** – the console now includes a `pangu` entry and metrics are
   saved to `results/pangu.csv`:

   ```text
   Summary:
                track_error  along_track_error  cross_track_error  intensity_mae
   forecast
   pangu               1.890             2.012              0.543          2.876
   ```

## Next Steps
Use the generated metrics to compare model variants or validate training runs. For details on configuring the data pipeline, see the [Data Pipeline](data_pipeline.md) reference.

