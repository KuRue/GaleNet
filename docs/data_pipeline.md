# Data Pipeline

This document describes the data pipeline used by GaleNet, including dataset schemas, preprocessing stages, and ERA5 patch extraction.

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

