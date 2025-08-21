"""Main data pipeline combining hurricane track and ERA5 data."""

from typing import Dict, List, Optional, Union

import pandas as pd
import xarray as xr
from loguru import logger

from ..utils.config import get_config
from .era5 import ERA5Loader
from .hurdat2 import HURDAT2Loader
from .ibtracs import IBTrACSLoader
from .satellite import SatelliteLoader


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
        self.satellite = SatelliteLoader()

        # Cache for loaded data
        self._cache: Dict[str, Dict[str, Union[pd.DataFrame, xr.Dataset]]] = {}

    def load_hurricane_for_training(
        self,
        storm_id: str,
        source: str = "hurdat2",
        include_era5: bool = True,
        include_satellite: bool = False,
        patch_size: float = 25.0,
        era5_variables: Optional[List[str]] = None,
        sat_sources: Optional[List[str]] = None,
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
                try:
                    config_vars = getattr(self.config.data, "era5_single_level_vars", None)
                except AttributeError:  # pragma: no cover - config may be absent
                    config_vars = None
                vars_to_use = era5_variables or config_vars
                era5_patches = self.era5.extract_hurricane_patches(
                    track_df, patch_size=patch_size, variables=vars_to_use
                )
                result["era5"] = era5_patches
            except Exception as e:  # pragma: no cover - network
                logger.warning(f"Could not load ERA5 data: {e}")
                logger.info("Continuing without ERA5 data")

        # Load satellite data if requested
        if include_satellite:
            logger.info(f"Fetching satellite tiles for {storm_id}")
            try:
                try:
                    cfg_sources = getattr(self.config.data, "satellite_sources", None)
                except AttributeError:  # pragma: no cover - config may be absent
                    cfg_sources = None
                sources = sat_sources or cfg_sources
                sat_tiles = self.satellite.extract_hurricane_tiles(
                    track_df, patch_size=patch_size, sources=sources
                )
                result["satellite"] = sat_tiles
            except Exception as e:  # pragma: no cover - network
                logger.warning(f"Could not load satellite data: {e}")
                logger.info("Continuing without satellite data")

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
