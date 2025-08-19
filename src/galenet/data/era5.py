"""Loader for ERA5 reanalysis data."""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import xarray as xr
from loguru import logger

from ..utils.config import get_config


class ERA5Loader:
    """Loader for ERA5 reanalysis data."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize ERA5 loader.

        Args:
            cache_dir: Directory to cache downloaded ERA5 data
        """
        if cache_dir is None:
            config = get_config()
            cache_dir = Path(config.data.era5_cache_dir)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(
        self,
        start_date: datetime,
        end_date: datetime,
        bounds: Tuple[float, float, float, float],
        variables: Optional[List[str]] = None,
    ) -> Path:
        """Return cache filepath for given request parameters.

        A short hash of the requested variable list is appended to the filename so
        that cached files for different variable combinations do not collide.
        """
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        bounds_str = f"{bounds[0]}N_{abs(bounds[1])}W_{bounds[2]}N_{abs(bounds[3])}W"
        var_hash = "default"
        if variables:
            key = ",".join(sorted(variables))
            var_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
        filename = f"era5_{date_str}_{bounds_str}_{var_hash}.nc"
        return self.cache_dir / filename

    def download_data(
        self,
        start_date: datetime,
        end_date: datetime,
        bounds: Tuple[float, float, float, float],
        variables: Optional[List[str]] = None,
    ) -> Path:
        """Download ERA5 data for a possibly multi-year period.

        The method caches each request on disk keyed by the date range, spatial
        bounds, and requested variable set.  When a range spans multiple years the
        yearly files are cached individually and then merged. Subsequent calls with
        overlapping periods or the same parameters reuse the cached files instead of
        re-downloading from the CDS API.
        """

        # If the range is contained within a single year we can download it
        # directly using the helper method.
        if start_date.year == end_date.year:
            return self._download_single_period(start_date, end_date, bounds, variables)

        # Path for the merged multi-year file, includes variable hash
        merged_path = self._cache_path(start_date, end_date, bounds, variables)
        if merged_path.exists():
            logger.info(f"ERA5 data already cached at {merged_path}")
            return merged_path

        # Download each year separately to benefit from caching and API limits
        datasets = []
        current_year = start_date.year
        while current_year <= end_date.year:
            period_start = max(start_date, datetime(current_year, 1, 1))
            period_end = min(end_date, datetime(current_year, 12, 31))
            path = self._download_single_period(period_start, period_end, bounds, variables)
            datasets.append(xr.open_dataset(path))
            current_year += 1

        merged = xr.concat(datasets, dim="time").sortby("time")
        merged.to_netcdf(merged_path)

        for ds in datasets:
            ds.close()

        return merged_path

    def _download_single_period(
        self,
        start_date: datetime,
        end_date: datetime,
        bounds: Tuple[float, float, float, float],
        variables: Optional[List[str]] = None,
    ) -> Path:
        """Download ERA5 data for a period confined to a single year."""
        import cdsapi
        import time

        if variables is None:
            # Default variables required by the preprocessing pipeline. This
            # includes both single-level fields and pressure-level winds used to
            # compute vertical shear and humidity-related diagnostics.
            single_level_vars = [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
                "2m_temperature",
                "2m_dewpoint_temperature",
                "sea_surface_temperature",
            ]
            pressure_level_vars = ["u_component_of_wind", "v_component_of_wind"]
            pressure_levels = ["200", "850"]
        else:
            # Separate provided variables into single-level and pressure-level
            single_level_vars = []
            pressure_level_vars = []
            for var in variables:
                if var in {
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "mean_sea_level_pressure",
                    "2m_temperature",
                    "2m_dewpoint_temperature",
                    "sea_surface_temperature",
                }:
                    single_level_vars.append(var)
                else:
                    pressure_level_vars.append(var)
            pressure_levels = ["200", "850"] if pressure_level_vars else []

        # Create filename for this sub-period (includes variable hash)
        filepath = self._cache_path(start_date, end_date, bounds, variables)

        if filepath.exists():
            logger.info(f"ERA5 data already cached at {filepath}")
            return filepath

        logger.info(f"Downloading ERA5 data to {filepath}")

        # Load API credentials from configuration
        config = get_config()
        era5_cfg = getattr(getattr(config, "data", None), "era5", None)
        api_url = getattr(era5_cfg, "api_url", None)
        api_key = getattr(era5_cfg, "api_key", None)
        if not api_url or not api_key:
            raise ValueError(
                "ERA5 API credentials not configured. "
                "Set data.era5.api_url and data.era5.api_key in your configuration."
            )

        try:
            c = cdsapi.Client(url=api_url, key=api_key, timeout=60, retry_max=0)
        except Exception as e:  # pragma: no cover - network issues
            raise RuntimeError(
                "Could not initialize CDS API client. "
                "Check your ERA5 API credentials and network connectivity."
            ) from e

        # Date range for this sub-period
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        # Build requests for single-level and pressure-level datasets
        requests = []
        if single_level_vars:
            requests.append(
                (
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "format": "netcdf",
                        "variable": single_level_vars,
                        "date": dates,
                        "time": [f"{h:02d}:00" for h in range(24)],
                        "area": list(bounds),
                    },
                    filepath.with_suffix(".single.nc"),
                )
            )

        if pressure_level_vars:
            requests.append(
                (
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "format": "netcdf",
                        "variable": pressure_level_vars,
                        "pressure_level": pressure_levels,
                        "date": dates,
                        "time": [f"{h:02d}:00" for h in range(24)],
                        "area": list(bounds),
                    },
                    filepath.with_suffix(".pressure.nc"),
                )
            )

        # Download with retries and exponential backoff for each request
        for dataset_name, request, outpath in requests:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    c.retrieve(dataset_name, request, str(outpath))
                    logger.success(f"Downloaded ERA5 data to {outpath}")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        msg = str(e).lower()
                        if "401" in msg or "unauthorized" in msg or "auth" in msg:
                            raise RuntimeError(
                                "ERA5 download failed: invalid API credentials."
                            ) from e
                        raise RuntimeError(
                            "ERA5 download failed after multiple attempts. "
                            "Check your network connection or API credentials."
                        ) from e

                    sleep_time = 2 ** attempt
                    logger.warning(
                        f"ERA5 download failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {sleep_time}s"
                    )
                    time.sleep(sleep_time)

        # Merge the downloaded datasets into a single file
        datasets = []
        for _, _, outpath in requests:
            if outpath.exists():
                datasets.append(xr.open_dataset(outpath))

        if datasets:
            merged = xr.merge(datasets)

            # If pressure level data were retrieved, rename to u200/v200/u850/v850
            if pressure_level_vars:
                for level in pressure_levels:
                    lvl = int(level)
                    if "u_component_of_wind" in merged and "v_component_of_wind" in merged:
                        u = merged["u_component_of_wind"].sel(pressure_level=lvl).drop(
                            "pressure_level"
                        )
                        v = merged["v_component_of_wind"].sel(pressure_level=lvl).drop(
                            "pressure_level"
                        )
                        merged[f"u{level}"] = u
                        merged[f"v{level}"] = v
                merged = merged.drop_vars(
                    [
                        v
                        for v in ["u_component_of_wind", "v_component_of_wind"]
                        if v in merged
                    ]
                )

            merged.to_netcdf(filepath)

        # Clean up temporary files
        for _, _, outpath in requests:
            if outpath.exists():
                outpath.unlink()

        return filepath

    def extract_hurricane_patches(
        self,
        track_df: pd.DataFrame,
        patch_size: float = 25.0,
        variables: Optional[List[str]] = None,
        lead_time_hours: int = 6,
        lag_time_hours: int = 6,
    ) -> xr.Dataset:
        """Extract ERA5 patches around hurricane track."""
        # Determine temporal bounds first
        start_date = track_df["timestamp"].min() - timedelta(hours=lead_time_hours)
        end_date = track_df["timestamp"].max() + timedelta(hours=lag_time_hours)

        # Determine spatial bounds and handle possible dateline crossings
        lat_min = track_df["latitude"].min() - patch_size / 2
        lat_max = track_df["latitude"].max() + patch_size / 2

        lon_vals = track_df["longitude"]
        lon_min = lon_vals.min() - patch_size / 2
        lon_max = lon_vals.max() + patch_size / 2

        # Detect dateline crossing when longitudes span more than 180 degrees
        crosses_dateline = lon_vals.max() - lon_vals.min() > 180

        if crosses_dateline:
            # Split into eastern (>=0) and western (<0) hemispheres
            east = lon_vals[lon_vals >= 0]
            west = lon_vals[lon_vals < 0]

            east_min = east.min() - patch_size / 2
            east_max = east.max() + patch_size / 2
            west_min = west.min() - patch_size / 2
            west_max = west.max() + patch_size / 2

            bounds_east = (lat_max, east_min, lat_min, east_max)
            bounds_west = (lat_max, west_min, lat_min, west_max)

            east_file = self.download_data(start_date, end_date, bounds_east, variables)
            west_file = self.download_data(start_date, end_date, bounds_west, variables)

            east_ds = self.load_file(east_file)
            west_ds = self.load_file(west_file)

            combined = xr.concat([west_ds, east_ds], dim="longitude").sortby(
                "longitude"
            )

            east_ds.close()
            west_ds.close()

            return combined

        bounds = (lat_max, lon_min, lat_min, lon_max)

        # Download data for single continuous region
        era5_file = self.download_data(start_date, end_date, bounds, variables)

        # Load and return
        return self.load_file(era5_file)

    def load_file(self, filepath: Path) -> xr.Dataset:
        """Load ERA5 data from NetCDF file."""
        return xr.open_dataset(filepath)
