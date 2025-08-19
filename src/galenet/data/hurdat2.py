"""Loader for HURDAT2 Atlantic hurricane database."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.config import get_config


class HURDAT2Loader:
    """Loader for HURDAT2 Atlantic hurricane database."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize HURDAT2 loader.

        Args:
            data_path: Path to HURDAT2 data file
        """
        if data_path is None:
            config = get_config()
            data_path = Path(config.data.hurdat2_path)

        self.data_path = Path(data_path)
        self._data: Optional[pd.DataFrame] = None
        self._storms: Dict[str, str] = {}

    def load_data(self) -> pd.DataFrame:
        """Load HURDAT2 data from file.

        Returns:
            DataFrame with all hurricane data
        """
        if self._data is not None:
            return self._data

        if not self.data_path.exists():
            raise FileNotFoundError(f"HURDAT2 data not found at {self.data_path}")

        logger.info(f"Loading HURDAT2 data from {self.data_path}")

        records = []
        current_storm = None

        with open(self.data_path, "r") as f:
            for line in f:
                line = line.strip()

                # Header line
                if re.match(r"^[A-Z]{2}\d{6},", line):
                    parts = line.split(",")
                    storm_id = parts[0].strip()
                    storm_name = parts[1].strip()
                    num_records = int(parts[2].strip())

                    current_storm = {
                        "storm_id": storm_id,
                        "name": storm_name,
                        "num_records": num_records,
                    }

                # Data line
                else:
                    parts = [p.strip() for p in line.split(",")]

                    # Parse date and time
                    date_str = parts[0]
                    time_str = parts[1]
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    hour = int(time_str[:2])
                    minute = int(time_str[2:4]) if len(time_str) == 4 else 0

                    timestamp = datetime(year, month, day, hour, minute)

                    # Parse location
                    lat_str = parts[4]
                    lon_str = parts[5]

                    # Convert latitude
                    lat = float(lat_str[:-1])
                    if lat_str[-1] == "S":
                        lat = -lat

                    # Convert longitude
                    lon = float(lon_str[:-1])
                    if lon_str[-1] == "W":
                        lon = -lon

                    # Parse intensity
                    max_wind = int(parts[6]) if parts[6] else np.nan
                    min_pressure = (
                        int(parts[7]) if parts[7] and parts[7] != "-999" else np.nan
                    )

                    # Storm type
                    storm_type = parts[3]

                    # Create record
                    record = {
                        "storm_id": current_storm["storm_id"],
                        "name": current_storm["name"],
                        "timestamp": timestamp,
                        "record_identifier": parts[2],
                        "storm_type": storm_type,
                        "latitude": lat,
                        "longitude": lon,
                        "max_wind": max_wind,
                        "min_pressure": min_pressure,
                    }

                    # Add wind radii if available
                    if len(parts) > 8:
                        wind_radii = {
                            "34kt_ne": int(parts[8]) if parts[8] and parts[8] != "-999" else np.nan,
                            "34kt_se": int(parts[9]) if parts[9] and parts[9] != "-999" else np.nan,
                            "34kt_sw": int(parts[10]) if parts[10] and parts[10] != "-999" else np.nan,
                            "34kt_nw": int(parts[11]) if parts[11] and parts[11] != "-999" else np.nan,
                        }
                        record.update(wind_radii)

                    records.append(record)

        self._data = pd.DataFrame(records)
        logger.info(
            f"Loaded {len(self._data)} records from {len(self._data.storm_id.unique())} storms"
        )

        return self._data

    def get_storm(self, storm_id: str) -> pd.DataFrame:
        """Get data for a specific storm.

        Args:
            storm_id: Storm identifier (e.g., 'AL122005' for Katrina)

        Returns:
            DataFrame with storm track data
        """
        if self._data is None:
            self.load_data()

        storm_data = self._data[self._data["storm_id"] == storm_id].copy()

        if len(storm_data) == 0:
            raise ValueError(f"Storm {storm_id} not found in database")

        # Sort by timestamp
        storm_data = storm_data.sort_values("timestamp").reset_index(drop=True)

        return storm_data

    def get_storms_by_year(self, year: int) -> List[str]:
        """Get all storm IDs for a given year.

        Args:
            year: Year to query

        Returns:
            List of storm IDs
        """
        if self._data is None:
            self.load_data()

        year_data = self._data[self._data["timestamp"].dt.year == year]
        return sorted(year_data["storm_id"].unique())

    def get_storms_by_name(self, name: str) -> List[str]:
        """Get storm IDs by name.

        Args:
            name: Storm name (case insensitive)

        Returns:
            List of storm IDs
        """
        if self._data is None:
            self.load_data()

        name_upper = name.upper()
        matches = self._data[self._data["name"] == name_upper]
        return sorted(matches["storm_id"].unique())

    def get_available_storms(self) -> Dict[str, str]:
        """Get all available storms.

        Returns:
            Dictionary mapping storm_id to storm name
        """
        if self._data is None:
            self.load_data()

        storms: Dict[str, str] = {}
        for storm_id in self._data["storm_id"].unique():
            storm_data = self._data[self._data["storm_id"] == storm_id].iloc[0]
            storms[storm_id] = storm_data["name"]

        return storms
