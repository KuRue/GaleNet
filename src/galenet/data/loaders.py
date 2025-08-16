# type: ignore
# flake8: noqa
"""Data loaders for GaleNet hurricane forecasting system."""

import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
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
        self._data = None
        self._storms = {}
        
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
        
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Header line
                if re.match(r'^[A-Z]{2}\d{6},', line):
                    parts = line.split(',')
                    storm_id = parts[0].strip()
                    storm_name = parts[1].strip()
                    num_records = int(parts[2].strip())
                    
                    current_storm = {
                        'storm_id': storm_id,
                        'name': storm_name,
                        'num_records': num_records
                    }
                    
                # Data line
                else:
                    parts = [p.strip() for p in line.split(',')]
                    
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
                    if lat_str[-1] == 'S':
                        lat = -lat
                        
                    # Convert longitude  
                    lon = float(lon_str[:-1])
                    if lon_str[-1] == 'W':
                        lon = -lon
                    
                    # Parse intensity
                    max_wind = int(parts[6]) if parts[6] else np.nan
                    min_pressure = int(parts[7]) if parts[7] and parts[7] != '-999' else np.nan
                    
                    # Storm type
                    storm_type = parts[3]
                    
                    # Create record
                    record = {
                        'storm_id': current_storm['storm_id'],
                        'name': current_storm['name'],
                        'timestamp': timestamp,
                        'record_identifier': parts[2],
                        'storm_type': storm_type,
                        'latitude': lat,
                        'longitude': lon,
                        'max_wind': max_wind,
                        'min_pressure': min_pressure
                    }
                    
                    # Add wind radii if available
                    if len(parts) > 8:
                        wind_radii = {
                            '34kt_ne': int(parts[8]) if parts[8] and parts[8] != '-999' else np.nan,
                            '34kt_se': int(parts[9]) if parts[9] and parts[9] != '-999' else np.nan,
                            '34kt_sw': int(parts[10]) if parts[10] and parts[10] != '-999' else np.nan,
                            '34kt_nw': int(parts[11]) if parts[11] and parts[11] != '-999' else np.nan,
                        }
                        record.update(wind_radii)
                    
                    records.append(record)
        
        self._data = pd.DataFrame(records)
        logger.info(f"Loaded {len(self._data)} records from {len(self._data.storm_id.unique())} storms")
        
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
            
        storm_data = self._data[self._data['storm_id'] == storm_id].copy()
        
        if len(storm_data) == 0:
            raise ValueError(f"Storm {storm_id} not found in database")
            
        # Sort by timestamp
        storm_data = storm_data.sort_values('timestamp').reset_index(drop=True)
        
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
            
        year_data = self._data[self._data['timestamp'].dt.year == year]
        return sorted(year_data['storm_id'].unique())
    
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
        matches = self._data[self._data['name'] == name_upper]
        return sorted(matches['storm_id'].unique())
    
    def get_available_storms(self) -> Dict[str, str]:
        """Get all available storms.
        
        Returns:
            Dictionary mapping storm_id to storm name
        """
        if self._data is None:
            self.load_data()
            
        storms = {}
        for storm_id in self._data['storm_id'].unique():
            storm_data = self._data[self._data['storm_id'] == storm_id].iloc[0]
            storms[storm_id] = storm_data['name']
            
        return storms


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
        self._dataset = None
        
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
            
        # Find storm by ID
        storm_idx = None
        for i, sid in enumerate(self._dataset.sid.values):
            if sid.decode('utf-8').strip() == storm_id:
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
        
        data = {
            'storm_id': storm_id,
            'name': storm_ds.name.values.decode('utf-8').strip(),
            'timestamp': times[valid_times],
            'latitude': storm_ds.lat.values[valid_times],
            'longitude': storm_ds.lon.values[valid_times],
            'max_wind': storm_ds.wmo_wind.values[valid_times],
            'min_pressure': storm_ds.wmo_pres.values[valid_times]
        }
        
        df = pd.DataFrame(data)
        
        # Remove invalid rows
        df = df[df['latitude'].notna()].reset_index(drop=True)
        
        return df


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

        A short hash of the requested variable list is appended to the
        filename so that cached files for different variable combinations do
        not collide.
        """
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        bounds_str = (
            f"{bounds[0]}N_{abs(bounds[1])}W_{bounds[2]}N_{abs(bounds[3])}W"
        )
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
        bounds, and requested variable set.  When a range spans multiple years
        the yearly files are cached individually and then merged. Subsequent
        calls with overlapping periods or the same parameters reuse the cached
        files instead of re-downloading from the CDS API.
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
            # Default variables required by the preprocessing pipeline.  This
            # includes both single-level fields and pressure-level winds used to
            # compute vertical shear and humidity-related diagnostics.
            single_level_vars = [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'mean_sea_level_pressure',
                '2m_temperature',
                '2m_dewpoint_temperature',
                'sea_surface_temperature',
            ]
            pressure_level_vars = ['u_component_of_wind', 'v_component_of_wind']
            pressure_levels = ['200', '850']
        else:
            # Separate provided variables into single-level and pressure-level
            single_level_vars = []
            pressure_level_vars = []
            for var in variables:
                if var in {
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'mean_sea_level_pressure',
                    '2m_temperature',
                    '2m_dewpoint_temperature',
                    'sea_surface_temperature',
                }:
                    single_level_vars.append(var)
                else:
                    pressure_level_vars.append(var)
            pressure_levels = ['200', '850'] if pressure_level_vars else []

        # Create filename for this sub-period (includes variable hash)
        filepath = self._cache_path(start_date, end_date, bounds, variables)

        if filepath.exists():
            logger.info(f"ERA5 data already cached at {filepath}")
            return filepath

        logger.info(f"Downloading ERA5 data to {filepath}")

        # Load API credentials from configuration
        config = get_config()
        era5_cfg = getattr(getattr(config, 'data', None), 'era5', None)
        api_url = getattr(era5_cfg, 'api_url', None)
        api_key = getattr(era5_cfg, 'api_key', None)
        if not api_url or not api_key:
            raise ValueError(
                'ERA5 API credentials not configured. '
                'Set data.era5.api_url and data.era5.api_key in your configuration.'
            )

        try:
            c = cdsapi.Client(url=api_url, key=api_key, timeout=60, retry_max=0)
        except Exception as e:
            raise RuntimeError(
                'Could not initialize CDS API client. '
                'Check your ERA5 API credentials and network connectivity.'
            ) from e

        # Date range for this sub-period
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        # Build requests for single-level and pressure-level datasets
        requests = []
        if single_level_vars:
            requests.append(
                (
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': single_level_vars,
                        'date': dates,
                        'time': [f'{h:02d}:00' for h in range(24)],
                        'area': list(bounds),
                    },
                    filepath.with_suffix('.single.nc'),
                )
            )

        if pressure_level_vars:
            requests.append(
                (
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': pressure_level_vars,
                        'pressure_level': pressure_levels,
                        'date': dates,
                        'time': [f'{h:02d}:00' for h in range(24)],
                        'area': list(bounds),
                    },
                    filepath.with_suffix('.pressure.nc'),
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
                        if '401' in msg or 'unauthorized' in msg or 'auth' in msg:
                            raise RuntimeError(
                                'ERA5 download failed: invalid API credentials.'
                            ) from e
                        raise RuntimeError(
                            'ERA5 download failed after multiple attempts. '
                            'Check your network connection or API credentials.'
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
                    if 'u_component_of_wind' in merged and 'v_component_of_wind' in merged:
                        u = merged['u_component_of_wind'].sel(pressure_level=lvl).drop('pressure_level')
                        v = merged['v_component_of_wind'].sel(pressure_level=lvl).drop('pressure_level')
                        merged[f'u{level}'] = u
                        merged[f'v{level}'] = v
                merged = merged.drop_vars(
                    [v for v in ['u_component_of_wind', 'v_component_of_wind'] if v in merged]
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
        lag_time_hours: int = 6
    ) -> xr.Dataset:
        """Extract ERA5 patches around hurricane track.
        
        Args:
            track_df: DataFrame with hurricane track
            patch_size: Size of patch in degrees
            variables: ERA5 variables to extract
            lead_time_hours: Hours before first track point
            lag_time_hours: Hours after last track point
            
        Returns:
            Dataset with ERA5 patches
        """
        # Determine temporal bounds first
        start_date = track_df['timestamp'].min() - timedelta(hours=lead_time_hours)
        end_date = track_df['timestamp'].max() + timedelta(hours=lag_time_hours)

        # Determine spatial bounds and handle possible dateline crossings
        lat_min = track_df['latitude'].min() - patch_size / 2
        lat_max = track_df['latitude'].max() + patch_size / 2

        lon_vals = track_df['longitude']
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

            combined = xr.concat([west_ds, east_ds], dim="longitude").sortby("longitude")

            east_ds.close()
            west_ds.close()

            return combined

        bounds = (lat_max, lon_min, lat_min, lon_max)

        # Download data for single continuous region
        era5_file = self.download_data(start_date, end_date, bounds, variables)

        # Load and return
        return self.load_file(era5_file)
    
    def load_file(self, filepath: Path) -> xr.Dataset:
        """Load ERA5 data from NetCDF file.
        
        Args:
            filepath: Path to NetCDF file
            
        Returns:
            ERA5 dataset
        """
        return xr.open_dataset(filepath)


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
        self._cache = {}
        
    def load_hurricane_for_training(
        self,
        storm_id: str,
        source: str = 'hurdat2',
        include_era5: bool = True,
        patch_size: float = 25.0
    ) -> Dict[str, Union[pd.DataFrame, xr.Dataset]]:
        """Load complete hurricane data for training.
        
        Args:
            storm_id: Storm identifier
            source: Data source ('hurdat2' or 'ibtracs')
            include_era5: Whether to include ERA5 reanalysis data
            patch_size: Size of ERA5 patches in degrees
            
        Returns:
            Dictionary with track data and optional ERA5 data
        """
        logger.info(f"Loading hurricane {storm_id} from {source}")
        
        # Load track data
        if source == 'hurdat2':
            track_df = self.hurdat2.get_storm(storm_id)
        elif source == 'ibtracs':
            track_df = self.ibtracs.to_dataframe(storm_id)
        else:
            raise ValueError(f"Unknown source: {source}")
            
        result = {'track': track_df}
        
        # Load ERA5 data if requested
        if include_era5:
            logger.info(f"Extracting ERA5 patches for {storm_id}")
            try:
                era5_patches = self.era5.extract_hurricane_patches(
                    track_df,
                    patch_size=patch_size
                )
                result['era5'] = era5_patches
            except Exception as e:
                logger.warning(f"Could not load ERA5 data: {e}")
                logger.info("Continuing without ERA5 data")
            
        return result
    
    def prepare_training_dataset(
        self,
        years: List[int],
        min_intensity: int = 64,  # Hurricane strength
        source: str = 'hurdat2'
    ) -> pd.DataFrame:
        """Prepare dataset of hurricanes for training.
        
        Args:
            years: Years to include
            min_intensity: Minimum intensity threshold
            source: Data source
            
        Returns:
            DataFrame with storm metadata
        """
        logger.info(f"Preparing training dataset for years {years}")
        
        storms = []
        
        for year in years:
            if source == 'hurdat2':
                year_storms = self.hurdat2.get_storms_by_year(year)
                
                for storm_id in year_storms:
                    track = self.hurdat2.get_storm(storm_id)
                    
                    # Check if storm reached hurricane intensity
                    if track['max_wind'].max() >= min_intensity:
                        storms.append({
                            'storm_id': storm_id,
                            'name': track['name'].iloc[0],
                            'year': year,
                            'max_intensity': track['max_wind'].max(),
                            'min_pressure': track['min_pressure'].min(),
                            'num_records': len(track)
                        })
        
        storms_df = pd.DataFrame(storms)
        logger.info(f"Found {len(storms_df)} hurricanes meeting criteria")
        
        return storms_df
