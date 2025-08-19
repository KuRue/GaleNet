"""Main data pipeline combining hurricane track and ERA5 data."""

from typing import Dict, List, Optional, Union

import pandas as pd
import xarray as xr
from loguru import logger

from ..utils.config import get_config
from .era5 import ERA5Loader
from .hurdat2 import HURDAT2Loader
from .ibtracs import IBTrACSLoader


class HurricaneDataPipeline:
    """Main data pipeline combining all data sources."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize data pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)

        # Initialize data loaders
        self.hurdat2 = HURDAT2Loader()
        self.ibtracs = IBTrACSLoader()
        self.era5 = ERA5Loader()

        # Cache for loaded data
        self._cache: Dict[str, Dict[str, Union[pd.DataFrame, xr.Dataset]]] = {}

    def load_hurricane_for_training(
        self,
        storm_id: str,
        source: str = "hurdat2",
        include_era5: bool = True,
        patch_size: float = 25.0,
    ) -> Dict[str, Union[pd.DataFrame, xr.Dataset]]:
        """Load complete hurricane data for training."""
        logger.info(f"Loading hurricane {storm_id} from {source}")

        # Load track data
        if source == "hurdat2":
            track_df = self.hurdat2.get_storm(storm_id)
        elif source == "ibtracs":
            track_df = self.ibtracs.to_dataframe(storm_id)
        else:
            raise ValueError(f"Unknown source: {source}")

        result: Dict[str, Union[pd.DataFrame, xr.Dataset]] = {"track": track_df}

        # Load ERA5 data if requested
        if include_era5:
            logger.info(f"Extracting ERA5 patches for {storm_id}")
            try:
                era5_patches = self.era5.extract_hurricane_patches(
                    track_df, patch_size=patch_size
                )
                result["era5"] = era5_patches
            except Exception as e:  # pragma: no cover - network
                logger.warning(f"Could not load ERA5 data: {e}")
                logger.info("Continuing without ERA5 data")

        return result

    def prepare_training_dataset(
        self,
        years: List[int],
        min_intensity: int = 64,
        source: str = "hurdat2",
    ) -> pd.DataFrame:
        """Prepare dataset of hurricanes for training."""
        logger.info(f"Preparing training dataset for years {years}")

        storms: List[Dict[str, Union[str, int]]] = []

        for year in years:
            if source == "hurdat2":
                year_storms = self.hurdat2.get_storms_by_year(year)

                for sid in year_storms:
                    track = self.hurdat2.get_storm(sid)

                    # Check if storm reached hurricane intensity
                    if track["max_wind"].max() >= min_intensity:
                        storms.append(
                            {
                                "storm_id": sid,
                                "name": track["name"].iloc[0],
                                "year": year,
                                "max_intensity": track["max_wind"].max(),
                                "min_pressure": track["min_pressure"].min(),
                                "num_records": len(track),
                            }
                        )

        storms_df = pd.DataFrame(storms)
        logger.info(f"Found {len(storms_df)} hurricanes meeting criteria")

        return storms_df
