"""Loader for satellite imagery tiles such as infrared and water vapor."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ..utils.config import get_config


class SatelliteLoader:
    """Fetch and cache satellite image tiles for hurricanes."""

    DEFAULT_SOURCES = ["ir1", "ir2", "water_vapor"]

    def __init__(self, cache_dir: Optional[Path] = None, sources: Optional[List[str]] = None):
        if cache_dir is None:
            config = get_config()
            cache_dir = Path(getattr(config.data, "satellite_cache_dir", "./sat_cache"))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sources = sources or self.DEFAULT_SOURCES

    def _cache_path(
        self,
        start_date: datetime,
        end_date: datetime,
        bounds: Tuple[float, float, float, float],
        sources: Optional[List[str]] = None,
    ) -> Path:
        """Return cache filepath for a given request."""
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        bounds_str = f"{bounds[0]}N_{abs(bounds[1])}W_{bounds[2]}N_{abs(bounds[3])}W"
        src_hash = "default"
        if sources:
            key = ",".join(sorted(sources))
            src_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
        filename = f"sat_{date_str}_{bounds_str}_{src_hash}.nc"
        return self.cache_dir / filename

    def extract_hurricane_tiles(
        self,
        track_df: pd.DataFrame,
        patch_size: float = 25.0,
        sources: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """Return satellite tiles around the hurricane track.

        The current implementation generates zero-filled arrays to act as
        placeholders for real satellite data and caches them on disk. The method
        still computes deterministic cache paths so that higher-level components
        can reason about caching behaviour.
        """
        sources = sources or self.sources

        start_date = track_df["timestamp"].min()
        end_date = track_df["timestamp"].max()

        lat_min = track_df["latitude"].min() - patch_size / 2
        lat_max = track_df["latitude"].max() + patch_size / 2
        lon_min = track_df["longitude"].min() - patch_size / 2
        lon_max = track_df["longitude"].max() + patch_size / 2
        bounds = (lat_max, lon_min, lat_min, lon_max)

        cache_path = self._cache_path(start_date, end_date, bounds, sources)
        if cache_path.exists():
            return xr.open_dataset(cache_path)

        lat = np.linspace(lat_min, lat_max, 2)
        lon = np.linspace(lon_min, lon_max, 2)
        time = track_df["timestamp"].to_numpy()
        shape = (len(time), len(lat), len(lon))
        data_vars = {
            src: (("time", "latitude", "longitude"), np.zeros(shape, dtype=np.float32))
            for src in sources
        }
        ds = xr.Dataset(data_vars, coords={"time": time, "latitude": lat, "longitude": lon})
        ds.to_netcdf(cache_path)
        return ds


__all__ = ["SatelliteLoader"]
