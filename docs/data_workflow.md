# Data Workflow

This guide outlines required datasets, ERA5 credential setup, and usage of the `HurricaneDataPipeline`.

## Required Datasets

GaleNet expects data in the root directory defined in [`configs/default_config.yaml`](../configs/default_config.yaml):

```
$HOME/data/galenet/
├── hurdat2/hurdat2.txt
├── ibtracs/IBTrACS.ALL.v04r00.nc
└── era5/
```

Use the setup script to create directories and download track datasets:

```bash
python scripts/setup_data.py --download-hurdat2 --download-ibtracs
```

ERA5 reanalysis data is optional but recommended. With credentials configured (see below), a sample can be fetched using:

```bash
python scripts/setup_data.py --download-era5
```

## ERA5 Credential Setup

Downloading ERA5 data requires Copernicus Climate Data Store (CDS) credentials:

1. Register at <https://cds.climate.copernicus.eu/user/register>.
2. Retrieve your API key from <https://cds.climate.copernicus.eu/api-how-to>.
3. Provide credentials via either method:

   **Environment variables**
   ```bash
   export CDSAPI_URL="https://cds.climate.copernicus.eu/api/v2"
   export CDSAPI_KEY="<UID>:<API_KEY>"
   ```

   **or `.cdsapirc` file**
   ```text
   url: https://cds.climate.copernicus.eu/api/v2
   key: <UID>:<API_KEY>
   ```

## Using `HurricaneDataPipeline`

Once the data directory and ERA5 credentials are in place, initialize the pipeline and load a storm:

```python
from galenet import HurricaneDataPipeline

pipeline = HurricaneDataPipeline("configs/default_config.yaml")
storm = pipeline.load_hurricane_for_training(
    storm_id="AL092023",
    include_era5=True,
    patch_size=25.0,
)

track = storm["track"]          # pandas DataFrame
era5_patch = storm["era5"]      # xarray Dataset
```

The call loads the HURDAT2 track and extracts ERA5 patches around the storm path, caching downloads under `$HOME/data/galenet/era5`.

## Advanced ERA5 Downloading

`ERA5Loader` now handles multi-year requests and caches each year's data on
disk. Subsequent downloads with overlapping periods reuse these cached files
and merge them into a single dataset.

```python
from datetime import datetime
from galenet.data.loaders import ERA5Loader

loader = ERA5Loader()
path = loader.download_data(
    datetime(2022, 6, 1),
    datetime(2023, 6, 30),
    bounds=(40, -80, 0, -30),
)
print(path)  # -> merged NetCDF file covering both years
```

If ERA5 credentials are missing or invalid the loader raises a descriptive
error before attempting any downloads, helping diagnose configuration issues.
