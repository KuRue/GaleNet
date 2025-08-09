"""Data validation modules for GaleNet hurricane forecasting."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


class HurricaneDataValidator:
    """Validator for hurricane track and intensity data."""

    def __init__(self):
        """Initialize validator with physics-based constraints."""
        # Physical constraints
        self.max_wind_speed = 200  # knots
        self.min_pressure = 850  # mb
        self.max_pressure = 1050  # mb
        self.max_speed_of_motion = 70  # knots
        self.max_lat = 60  # degrees
        self.min_lat = -60  # degrees

    def validate_track(
        self,
        track_df: pd.DataFrame,
        strict: bool = False
    ) -> tuple[bool, List[str]]:
        """Validate hurricane track data.

        Args:
            track_df: DataFrame with track data
            strict: Whether to apply strict validation

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required columns
        required_cols = ['timestamp', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in track_df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, errors

        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(track_df['timestamp']):
            try:
                track_df['timestamp'] = pd.to_datetime(track_df['timestamp'])
            except Exception:
                errors.append("Cannot convert timestamp to datetime")

        # Check coordinate bounds
        if track_df['latitude'].min() < self.min_lat or track_df['latitude'].max() > self.max_lat:
            errors.append(f"Latitude out of bounds [{self.min_lat}, {self.max_lat}]")

        if track_df['longitude'].min() < -180 or track_df['longitude'].max() > 180:
            errors.append("Longitude out of bounds [-180, 180]")

        # Check for NaN values
        if track_df[['latitude', 'longitude']].isna().any().any():
            errors.append("NaN values found in coordinates")

        # Check intensity values if present
        if 'max_wind' in track_df.columns:
            if track_df['max_wind'].max() > self.max_wind_speed:
                errors.append(f"Maximum wind speed exceeds {self.max_wind_speed} knots")
            if track_df['max_wind'].min() < 0:
                errors.append("Negative wind speeds found")

        if 'min_pressure' in track_df.columns:
            if track_df['min_pressure'].min() < self.min_pressure:
                errors.append(f"Minimum pressure below {self.min_pressure} mb")
            if track_df['min_pressure'].max() > self.max_pressure:
                errors.append(f"Maximum pressure above {self.max_pressure} mb")

        # Check track continuity
        if len(track_df) > 1:
            time_diffs = track_df['timestamp'].diff().dt.total_seconds() / 3600
            if time_diffs.max() > 12 and strict:  # More than 12 hour gap
                errors.append("Large time gaps in track data")

            # Check speed of motion
            lat_diff = track_df['latitude'].diff()
            lon_diff = track_df['longitude'].diff()
            distance = np.sqrt(lat_diff**2 + lon_diff**2) * 60  # nautical miles
            speed = distance / time_diffs.fillna(1)

            if speed.max() > self.max_speed_of_motion:
                errors.append(f"Storm motion exceeds {self.max_speed_of_motion} knots")

        # Check minimum track length
        if len(track_df) < 4:
            errors.append("Track has fewer than 4 points")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_intensity_physics(
        self,
        track_df: pd.DataFrame
    ) -> tuple[bool, List[str]]:
        """Validate physical relationships in intensity data.

        Args:
            track_df: DataFrame with track and intensity data

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if 'max_wind' not in track_df.columns or 'min_pressure' not in track_df.columns:
            return True, []  # Skip if intensity data not present

        # Check wind-pressure relationship
        # Approximate relationship: P = 1013 - (V/14)^2
        for idx, row in track_df.iterrows():
            wind = row['max_wind']
            pressure = row['min_pressure']

            if pd.notna(wind) and pd.notna(pressure):
                expected_pressure = 1013 - (wind / 14) ** 2
                pressure_diff = abs(pressure - expected_pressure)

                if pressure_diff > 50:  # Allow 50 mb tolerance
                    errors.append(
                        f"Wind-pressure relationship violated at index {idx}: "
                        f"wind={wind}kt, pressure={pressure}mb"
                    )

        # Check rapid intensification limits
        if len(track_df) > 1:
            wind_change = track_df['max_wind'].diff()
            time_diff = track_df['timestamp'].diff().dt.total_seconds() / 3600

            # Maximum intensification rate ~50 kt/24hr
            max_rate = 50 / 24
            intensification_rate = wind_change / time_diff.fillna(1)

            if intensification_rate.max() > max_rate * 2:  # Allow some margin
                errors.append("Unrealistic intensification rate detected")

        is_valid = len(errors) == 0
        return is_valid, errors


def validate_track_continuity(
    track_df: pd.DataFrame,
    max_gap_hours: float = 12.0,
    max_jump_distance: float = 500.0  # nautical miles
) -> tuple[bool, List[str]]:
    """Validate track continuity.

    Args:
        track_df: DataFrame with track data
        max_gap_hours: Maximum allowed time gap
        max_jump_distance: Maximum allowed position jump

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if len(track_df) < 2:
        return True, []

    # Check time continuity
    time_gaps = track_df['timestamp'].diff().dt.total_seconds() / 3600
    large_gaps = time_gaps[time_gaps > max_gap_hours]

    if len(large_gaps) > 0:
        errors.append(
            f"Found {len(large_gaps)} time gaps larger than {max_gap_hours} hours"
        )

    # Check spatial continuity
    lat_diff = track_df['latitude'].diff()
    lon_diff = track_df['longitude'].diff()
    distances = np.sqrt(lat_diff**2 + lon_diff**2) * 60  # nautical miles

    large_jumps = distances[distances > max_jump_distance]
    if len(large_jumps) > 0:
        errors.append(
            f"Found {len(large_jumps)} position jumps larger than {max_jump_distance} nm"
        )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_intensity_physics(
    track_df: pd.DataFrame,
    validator: Optional[HurricaneDataValidator] = None
) -> tuple[bool, List[str]]:
    """Convenience function to validate intensity physics.

    Args:
        track_df: DataFrame with track and intensity data
        validator: Validator instance (creates new if None)

    Returns:
        Tuple of (is_valid, error_messages)
    """
    if validator is None:
        validator = HurricaneDataValidator()
    return validator.validate_intensity_physics(track_df)


def validate_era5_data(
    era5_data: xr.Dataset,
    required_vars: Optional[List[str]] = None
) -> tuple[bool, List[str]]:
    """Validate ERA5 reanalysis data.

    Args:
        era5_data: ERA5 dataset
        required_vars: Required variables

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if required_vars is None:
        required_vars = ['u10', 'v10', 'msl', 't2m']

    # Check for required variables
    missing_vars = [var for var in required_vars if var not in era5_data.data_vars]
    if missing_vars:
        errors.append(f"Missing required variables: {missing_vars}")

    # Check dimensions
    required_dims = ['time', 'latitude', 'longitude']
    missing_dims = [dim for dim in required_dims if dim not in era5_data.dims]
    if missing_dims:
        errors.append(f"Missing required dimensions: {missing_dims}")

    # Check for NaN values
    for var in era5_data.data_vars:
        nan_count = era5_data[var].isnull().sum().item()
        if nan_count > 0:
            total_count = era5_data[var].size
            nan_percent = (nan_count / total_count) * 100
            if nan_percent > 10:  # Allow up to 10% NaN values
                errors.append(f"Variable {var} has {nan_percent:.1f}% NaN values")

    # Check coordinate ranges
    if 'latitude' in era5_data.dims:
        lat_range = [era5_data.latitude.min().item(), era5_data.latitude.max().item()]
        if lat_range[0] < -90 or lat_range[1] > 90:
            errors.append(f"Invalid latitude range: {lat_range}")

    if 'longitude' in era5_data.dims:
        lon_range = [era5_data.longitude.min().item(), era5_data.longitude.max().item()]
        if lon_range[0] < -180 or lon_range[1] > 360:
            errors.append(f"Invalid longitude range: {lon_range}")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_training_data(
    data: Dict[str, Union[pd.DataFrame, xr.Dataset]],
    strict: bool = False
) -> bool:
    """Validate complete training data package.

    Args:
        data: Dictionary with 'track' DataFrame and optional 'era5' Dataset
        strict: Whether to apply strict validation

    Returns:
        Whether data is valid for training
    """
    validator = HurricaneDataValidator()
    all_valid = True

    # Validate track data
    if 'track' not in data:
        logger.error("No track data found")
        return False

    track_valid, track_errors = validator.validate_track(data['track'], strict=strict)
    if not track_valid:
        logger.error(f"Track validation failed: {track_errors}")
        all_valid = False

    # Validate intensity physics
    physics_valid, physics_errors = validator.validate_intensity_physics(data['track'])
    if not physics_valid:
        logger.warning(f"Physics validation warnings: {physics_errors}")
        if strict:
            all_valid = False

    # Validate ERA5 data if present
    if 'era5' in data:
        era5_valid, era5_errors = validate_era5_data(data['era5'])
        if not era5_valid:
            logger.error(f"ERA5 validation failed: {era5_errors}")
            all_valid = False

    return all_valid
