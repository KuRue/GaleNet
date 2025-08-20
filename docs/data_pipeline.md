# Data Pipeline

This document describes the data pipeline used by GaleNet, including dataset schemas, preprocessing stages, and ERA5 patch extraction. For a step-by-step walkthrough from raw archives to model-ready tensors see the [Full Data Pipeline Guide](data_pipeline_full.md). For instructions on gathering datasets, see the [Data Workflow](data_workflow.md). After data preparation, continue with the [Training Guide](training.md) or [Evaluation Guide](evaluation.md).

## End-to-End Workflow

1. **Download Raw Datasets** – Fetch track archives and a sample ERA5 cube:

   ```bash
   python scripts/setup_data.py --all --data-dir $HOME/data/galenet
   ```

   Output directories:

   ```
   $HOME/data/galenet/
   ├── hurdat2/hurdat2.txt
   ├── ibtracs/IBTrACS.ALL.v04r00.nc
   ├── era5/
   ├── models/
   └── cache/
   ```

   Runtime checkpoints: HURDAT2 (<1 min), IBTrACS (~5 min), sample ERA5 (~10 min per year).

2. **Extract ERA5 Patches** – Use the pipeline to cache reanalysis data around a storm track:

   ```bash
   python - <<'PY'
   from galenet import HurricaneDataPipeline
   pipeline = HurricaneDataPipeline("configs/default_config.yaml")
   storm = pipeline.load_hurricane_for_training(
       storm_id="AL092019",
       include_era5=True,
       patch_size=25.0,
   )
   PY
   ```

   ERA5 downloads are stored under `$HOME/data/galenet/era5` and reused on subsequent runs (first download ~5–10 min, cached loads <1 min).

3. **Assemble Final Dataset** – Persist the track and its ERA5 patches for training or evaluation:

   ```bash
   python - <<'PY'
   from pathlib import Path
   from galenet import HurricaneDataPipeline
   pipeline = HurricaneDataPipeline("configs/default_config.yaml")
   storm = pipeline.load_hurricane_for_training("AL092019", include_era5=True)
   out_dir = Path("$HOME/data/galenet/processed/AL092019")
   out_dir.mkdir(parents=True, exist_ok=True)
   storm["track"].to_csv(out_dir / "track.csv", index=False)
   storm["era5"].to_netcdf(out_dir / "era5.nc")
   PY
   ```

   Resulting structure:

   ```
   processed/AL092019/
   ├── track.csv
   └── era5.nc
   ```

   Writing the assembled dataset typically takes only a few seconds.

## Dataset Schemas

### HURDAT2 and IBTrACS Track Data

Storm track records are normalised to a common schema with the following core fields:

| Column | Description |
|--------|-------------|
| `storm_id` | Basin identifier and year (e.g., `AL092019`) |
| `name` | Storm name in uppercase |
| `timestamp` | Observation time in UTC |
| `latitude` | Degrees north (negative south) |
| `longitude` | Degrees east (negative west) |
| `max_wind` | Maximum sustained wind in knots |
| `min_pressure` | Minimum central pressure in hPa |
| `34kt_ne`, `34kt_se`, `34kt_sw`, `34kt_nw` | Optional wind radii |

IBTrACS provides the same physical quantities but may contain additional metadata. All fields are cast to numeric types and timestamps are parsed into ``pandas`` ``datetime`` objects.

### ERA5 Reanalysis

ERA5 fields are stored as an ``xarray.Dataset`` with dimensions ``time`` x ``latitude`` x ``longitude``. Typical variables include:

- Surface: `u10`, `v10`, `msl`, `t2m`, `sst`
- Pressure levels: `u{level}` and `v{level}` (e.g., `u850`)

Each file encodes CF-compliant attributes allowing direct use with `xarray` or `pint` for unit handling.

## Preprocessing Workflow

1. **Track Loading** – HURDAT2/IBTrACS records are read, sorted by time, and converted to the unified schema above.
2. **Normalization** – Latitudes, longitudes, and intensity values are scaled; categorical flags are one‑hot encoded.
3. **Feature Engineering** – Additional predictors such as forward speed, bearing, and environmental shear are derived.
4. **Quality Checks** – ``HurricaneDataValidator`` enforces physical limits (e.g., pressure ranges) and timestamps spacing.
5. **Dataset Assembly** – When ERA5 data are requested, atmospheric patches are merged with the track for downstream models.

## ERA5 Patch Extraction

ERA5 patches provide the model with environmental context around the storm path:

1. **Bounds** – For a given track, spatial bounds are computed from the min/max latitude and longitude with a configurable ``patch_size`` (default ``25°``) margin.
2. **Time Window** – Data can include a lead and lag window (``lead_time_hours`` and ``lag_time_hours``) around the observation period.
3. **Download** – ``ERA5Loader.download_data`` requests the required variables and time range from the Copernicus Climate Data Store, caching results on disk.
4. **Extraction** – ``ERA5Loader.extract_hurricane_patches`` or ``HurricaneDataPipeline`` slices the downloaded dataset into patches following the storm track. Dateline crossings are handled by stitching split patches.
5. **Output** – The final dataset is an ``xarray.Dataset`` aligned with track timestamps and ready for model ingestion.

This process enables consistent preprocessing across training and inference, ensuring storms share a common representation and environmental context.

## Evaluation Example

Validate the pipeline and generate baseline forecasts using:

```bash
python scripts/evaluate_baselines.py data/sample_storms.json \
  --history 3 --forecast 2 --model-config configs/default_config.yaml
```

The script loads tracks through `HurricaneDataPipeline`, runs baseline models,
and reports track error statistics. Additional options are documented in the
[Evaluation Guide](evaluation.md).

## Preparing GraphCast Inputs

GraphCast operates on ERA5 fields sampled at a 0.25° grid. To supply these
inputs through the data pipeline:

1. **Select Variables** – Request the surface and pressure‑level fields used by
   GraphCast:

   ```yaml
   data:
     era5:
       variables:
         - "10m_u_component_of_wind"
         - "10m_v_component_of_wind"
         - "mean_sea_level_pressure"
         - "2m_temperature"
         - "sea_surface_temperature"
         - "total_precipitation"
         - "convective_available_potential_energy"
       pressure_levels: [1000, 925, 850, 700, 500, 300, 200]
   ```

2. **Set Resolution** – Extract ERA5 patches on the 0.25° latitude/longitude
   grid expected by GraphCast:

   ```yaml
   model:
     graphcast:
       resolution: 0.25
   ```

3. **Align with ERA5** – During extraction the `HurricaneDataPipeline` resamples
   ERA5 data to this grid and snaps timestamps to the nearest analysis hour so
   the resulting `xarray.Dataset` matches GraphCast's expectations.

### Enabling GraphCast Extraction

Activate GraphCast features in the pipeline with the following configuration:

```yaml
model:
  name: graphcast
  graphcast:
    checkpoint_path: "models/graphcast/params.npz"
training:
  include_era5: true
```

This configuration instructs the pipeline to download the required ERA5
variables at 0.25° resolution and provide them to GraphCast during processing.

## Preparing Pangu Inputs

Pangu-Weather ingests a richer 3D ERA5 cube to supply atmospheric context. The
backbone is used for inference only, and GaleNet does not train or update Pangu
weights.

1. **Select Variables** – Request the surface and pressure‑level fields needed
   by Pangu:

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

2. **Set Resolution** – Extract ERA5 patches on Pangu's 0.25° grid:

   ```yaml
   model:
     pangu:
       resolution: 0.25
   ```

3. **Enable Pangu** – Activate the backbone and ensure ERA5 fields are passed
   through the pipeline:

   ```yaml
   model:
     name: pangu
     pangu:
       checkpoint_path: "models/pangu/params.npz"
   training:
     include_era5: true
   ```

This configuration prepares the required ERA5 data for Pangu during
preprocessing. The Pangu weights remain fixed and are only used for inference.
