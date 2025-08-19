"""Loader for the IBTrACS global tropical cyclone database."""

from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr
from loguru import logger

from ..utils.config import get_config


class IBTrACSLoader:
    """Loader for IBTrACS global tropical cyclone database."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize IBTrACS loader.

        Args:
            data_path: Path to IBTrACS NetCDF file
        """
        if data_path is None:
            config = get_config()
            data_path = Path(config.data.ibtracs_path)

        self.data_path = Path(data_path)
        self._dataset: Optional[xr.Dataset] = None

    def load_data(self) -> xr.Dataset:
        """Load IBTrACS dataset.

        Returns:
            xarray Dataset with IBTrACS data
        """
        if self._dataset is not None:
            return self._dataset

        if not self.data_path.exists():
            raise FileNotFoundError(f"IBTrACS data not found at {self.data_path}")

        logger.info(f"Loading IBTrACS data from {self.data_path}")
        self._dataset = xr.open_dataset(self.data_path)

        num_storms = len(self._dataset.storm)
        logger.info(f"Loaded IBTrACS with {num_storms} storms")

        return self._dataset

    def get_storm(self, storm_id: str) -> xr.Dataset:
        """Get data for a specific storm.

        Args:
            storm_id: Storm identifier

        Returns:
            Dataset with storm data
        """
        if self._dataset is None:
            self.load_data()
        assert self._dataset is not None

        # Find storm by ID
        storm_idx = None
        for i, sid in enumerate(self._dataset.sid.values):
            if sid.decode("utf-8").strip() == storm_id:
                storm_idx = i
                break

        if storm_idx is None:
            raise ValueError(f"Storm {storm_id} not found in IBTrACS")

        return self._dataset.isel(storm=storm_idx)

    def to_dataframe(self, storm_id: str) -> pd.DataFrame:
        """Convert storm data to DataFrame format.

        Args:
            storm_id: Storm identifier

        Returns:
            DataFrame with storm track data
        """
        storm_ds = self.get_storm(storm_id)

        # Extract relevant variables
        times = pd.to_datetime(storm_ds.time.values)
        valid_times = ~pd.isna(times)

        name_val = storm_ds.name.values
        if hasattr(name_val, "decode"):
            name = name_val.decode("utf-8").strip()
        else:  # numpy 0-d arrays
            name = name_val.item().decode("utf-8").strip()

        data = {
            "storm_id": storm_id,
            "name": name,
            "timestamp": times[valid_times],
            "latitude": storm_ds.lat.values[valid_times],
            "longitude": storm_ds.lon.values[valid_times],
            "max_wind": storm_ds.wmo_wind.values[valid_times],
            "min_pressure": storm_ds.wmo_pres.values[valid_times],
        }

        df = pd.DataFrame(data)

        # Remove invalid rows
        df = df[df["latitude"].notna()].reset_index(drop=True)

        return df
