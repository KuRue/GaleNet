"""Data preprocessing modules for GaleNet hurricane forecasting."""

from typing import List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from sklearn.preprocessing import StandardScaler
import torch


class HurricanePreprocessor:
    """Preprocessor for hurricane track and intensity data."""

    def __init__(self):
        """Initialize preprocessor with standard scalers."""
        self.position_scaler = StandardScaler()
        self.intensity_scaler = StandardScaler()
        self.velocity_scaler = StandardScaler()
        self.fitted = False

    def normalize_track_data(
        self,
        track_df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """Normalize hurricane track data.

        Args:
            track_df: DataFrame with columns [timestamp, latitude, longitude,
                     max_wind, min_pressure]
            fit: Whether to fit the scalers (True for training data)

        Returns:
            Normalized DataFrame
        """
        normalized_df = track_df.copy()

        # Normalize positions (lat/lon)
        position_cols = ['latitude', 'longitude']
        if fit:
            # scikit-learn's StandardScaler uses population statistics (ddof=0)
            # which results in the sample standard deviation being greater than 1
            # when checked with pandas' default `std` (ddof=1).  This caused unit
            # tests that expect a sample std of exactly 1 to fail.  To avoid this
            # discrepancy we apply a correction factor so that the transformed
            # data has a sample standard deviation of 1.  The correction factor is
            # stored for use when transforming new data.
            scaled = self.position_scaler.fit_transform(normalized_df[position_cols])
            n = len(normalized_df[position_cols])
            self._pos_scale_factor = np.sqrt((n - 1) / n) if n > 1 else 1.0
            normalized_df[position_cols] = scaled * self._pos_scale_factor
        else:
            scaled = self.position_scaler.transform(normalized_df[position_cols])
            factor = getattr(self, "_pos_scale_factor", 1.0)
            normalized_df[position_cols] = scaled * factor

        # Normalize intensity (wind/pressure)
        intensity_cols = ['max_wind', 'min_pressure']
        if intensity_cols[0] in normalized_df.columns:
            if fit:
                scaled = self.intensity_scaler.fit_transform(normalized_df[intensity_cols])
                n = len(normalized_df[intensity_cols])
                self._int_scale_factor = np.sqrt((n - 1) / n) if n > 1 else 1.0
                normalized_df[intensity_cols] = scaled * self._int_scale_factor
            else:
                scaled = self.intensity_scaler.transform(normalized_df[intensity_cols])
                factor = getattr(self, "_int_scale_factor", 1.0)
                normalized_df[intensity_cols] = scaled * factor

        if fit:
            self.fitted = True

        return normalized_df

    def create_track_features(
        self,
        track_df: pd.DataFrame,
        include_temporal: bool = True
    ) -> pd.DataFrame:
        """Create engineered features from track data.

        Args:
            track_df: DataFrame with hurricane track data
            include_temporal: Whether to include temporal features

        Returns:
            DataFrame with engineered features
        """
        features_df = track_df.copy()

        # Calculate velocities
        features_df['lat_velocity'] = features_df['latitude'].diff() / 6  # 6-hour intervals
        features_df['lon_velocity'] = features_df['longitude'].diff() / 6
        features_df['speed'] = np.sqrt(
            features_df['lat_velocity']**2 + features_df['lon_velocity']**2
        )

        # Calculate acceleration
        features_df['lat_acceleration'] = features_df['lat_velocity'].diff()
        features_df['lon_acceleration'] = features_df['lon_velocity'].diff()

        # Intensity changes
        if 'max_wind' in features_df.columns:
            features_df['wind_change'] = features_df['max_wind'].diff()
            features_df['pressure_change'] = features_df['min_pressure'].diff()

            # Rapid intensification indicator
            features_df['rapid_intensification'] = (
                features_df['wind_change'] >= 30  # 30 kt in 24 hours
            ).astype(float)

        # Direction of motion
        features_df['heading'] = np.arctan2(
            features_df['lon_velocity'],
            features_df['lat_velocity']
        ) * 180 / np.pi

        # Temporal features
        if include_temporal and 'timestamp' in features_df.columns:
            features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
            features_df['day_of_year'] = pd.to_datetime(features_df['timestamp']).dt.dayofyear

            # Cyclical encoding
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
            features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)

        # Fill NaN values from diff operations
        features_df = features_df.fillna(0)

        return features_df

    def prepare_sequences(
        self,
        features_df: pd.DataFrame,
        sequence_length: int = 8,
        forecast_length: int = 20,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for model training.

        Args:
            features_df: DataFrame with features
            sequence_length: Input sequence length
            forecast_length: Forecast horizon length
            feature_cols: Columns to use as features (None for all)

        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        if feature_cols is None:
            # Default feature columns
            feature_cols = [
                'latitude', 'longitude', 'max_wind', 'min_pressure',
                'lat_velocity', 'lon_velocity', 'speed',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ]
            # Filter to available columns
            feature_cols = [col for col in feature_cols if col in features_df.columns]

        # Convert to numpy array
        data = features_df[feature_cols].values

        # Create sequences
        total_length = sequence_length + forecast_length
        num_sequences = len(data) - total_length + 1

        if num_sequences <= 0:
            logger.warning(
                f"Not enough data for sequences. Need {total_length} points, "
                f"have {len(data)}"
            )
            return np.array([]), np.array([])

        input_sequences = []
        target_sequences = []

        for i in range(num_sequences):
            input_seq = data[i:i + sequence_length]
            target_seq = data[i + sequence_length:i + total_length]

            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

        return np.array(input_sequences), np.array(target_sequences)


class ERA5Preprocessor:
    """Preprocessor for ERA5 reanalysis data."""

    def __init__(self):
        """Initialize ERA5 preprocessor."""
        self.variable_stats = {}

    def extract_patches(
        self,
        era5_data: xr.Dataset,
        center_lat: float,
        center_lon: float,
        patch_size: float = 25.0,
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """Extract spatial patches around hurricane center.

        Args:
            era5_data: ERA5 dataset
            center_lat: Hurricane center latitude
            center_lon: Hurricane center longitude
            patch_size: Size of patch in degrees
            variables: Variables to extract (None for all)

        Returns:
            Extracted patch dataset
        """
        # Define bounds
        lat_min = center_lat - patch_size / 2
        lat_max = center_lat + patch_size / 2
        lon_min = center_lon - patch_size / 2
        lon_max = center_lon + patch_size / 2

        # Handle longitude wrapping
        if lon_min < -180:
            lon_min += 360
        if lon_max > 180:
            lon_max -= 360

        # Extract patch
        if lon_min < lon_max:
            patch = era5_data.sel(
                latitude=slice(lat_max, lat_min),  # ERA5 has descending lats
                longitude=slice(lon_min, lon_max)
            )
        else:
            # Handle antimeridian crossing
            patch1 = era5_data.sel(
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, 180)
            )
            patch2 = era5_data.sel(
                latitude=slice(lat_max, lat_min),
                longitude=slice(-180, lon_max)
            )
            patch = xr.concat([patch1, patch2], dim='longitude')

        # Select variables if specified
        if variables:
            available_vars = [v for v in variables if v in patch.data_vars]
            patch = patch[available_vars]

        return patch

    def normalize_variables(
        self,
        data: xr.Dataset,
        fit: bool = True,
        variables: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """Normalize ERA5 variables to zero mean and unit variance.

        This method now supports normalizing a subset of variables rather than
        all variables present in the dataset, which is useful when working with
        heterogeneous collections of ERA5 fields.

        Args:
            data: ERA5 dataset
            fit: Whether to fit normalization parameters
            variables: Optional list of variables to normalize.  If ``None`` all
                variables in ``data`` will be normalized.

        Returns:
            Normalized dataset
        """
        normalized = data.copy()

        if variables is None:
            variables = list(data.data_vars)

        for var in variables:
            if var not in data.data_vars:
                # Skip variables that aren't present in the dataset
                logger.warning(f"Variable {var} not found in data, skipping")
                continue

            var_data = data[var].values

            if fit:
                mean = float(np.nanmean(var_data))
                std = float(np.nanstd(var_data))
                self.variable_stats[var] = {"mean": mean, "std": std}
            else:
                if var not in self.variable_stats:
                    logger.warning(f"No normalization stats for {var}, skipping")
                    continue
                mean = self.variable_stats[var]["mean"]
                std = self.variable_stats[var]["std"]

            # Normalize
            normalized[var] = (data[var] - mean) / (std + 1e-8)

        return normalized

    def compute_derived_fields(
        self,
        data: xr.Dataset
    ) -> xr.Dataset:
        """Compute derived meteorological fields.

        The routine augments raw ERA5 variables with additional diagnostics
        used for model training.  These include vertical wind shear between 200
        and 850 hPa, a suite of humidity indices (relative humidity, specific
        humidity and dewpoint depression) and basic kinematic quantities such as
        vorticity and convergence.

        Args:
            data: ERA5 dataset

        Returns:
            Dataset with added derived fields
        """
        enhanced = data.copy()

        # Wind speed and direction from u/v components
        if 'u10' in data and 'v10' in data:
            enhanced['wind_speed'] = np.sqrt(data['u10']**2 + data['v10']**2)
            enhanced['wind_direction'] = np.arctan2(data['v10'], data['u10']) * 180 / np.pi

        # Relative vorticity at 10m
        if 'u10' in data and 'v10' in data:
            # Simple finite difference approximation
            dvdx = data['v10'].differentiate('longitude')
            dudy = data['u10'].differentiate('latitude')
            enhanced['vorticity_10m'] = dvdx - dudy

        # Convergence
        if 'u10' in data and 'v10' in data:
            dudx = data['u10'].differentiate('longitude')
            dvdy = data['v10'].differentiate('latitude')
            enhanced['convergence_10m'] = -(dudx + dvdy)

        # Potential temperature (if temperature available)
        if 't2m' in data and 'msl' in data:
            # Simplified calculation
            enhanced['theta'] = data['t2m'] * (1000.0 / data['msl'])**(287.0/1004.0)

        # Vertical wind shear between 200 hPa and 850 hPa
        shear_vars = ['u200', 'v200', 'u850', 'v850']
        if all(var in data for var in shear_vars):
            du = data['u200'] - data['u850']
            dv = data['v200'] - data['v850']
            enhanced['vertical_wind_shear_u'] = du
            enhanced['vertical_wind_shear_v'] = dv
            enhanced['vertical_wind_shear'] = np.sqrt(du**2 + dv**2)

        # Relative humidity using 2m temperature and dew point
        if 't2m' in data and 'd2m' in data:
            t_c = data['t2m'] - 273.15
            td_c = data['d2m'] - 273.15
            numerator = np.exp((17.625 * td_c) / (243.04 + td_c))
            denominator = np.exp((17.625 * t_c) / (243.04 + t_c))
            enhanced['relative_humidity'] = (100.0 * numerator / denominator).clip(0, 100)
            # Dewpoint depression: difference between temperature and dew point
            enhanced['dewpoint_depression'] = data['t2m'] - data['d2m']

        # Specific humidity computed from dew point and pressure
        if 'd2m' in data and 'msl' in data:
            td_c = data['d2m'] - 273.15
            # Convert pressure to hPa
            p_hpa = data['msl'] / 100.0
            e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
            enhanced['specific_humidity'] = (0.622 * e) / (p_hpa - 0.378 * e)

        return enhanced

    def save_stats(self, path: str) -> None:
        """Persist variable statistics to disk as JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.variable_stats, f)

    def load_stats(self, path: str) -> None:
        """Load variable statistics from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            self.variable_stats = json.load(f)

    def to_tensor(
        self,
        data: xr.Dataset,
        variables: Optional[List[str]] = None,
        dtype: "Optional[torch.dtype]" = None,
    ) -> "torch.Tensor":
        """Convert xarray dataset to a PyTorch tensor.

        Args:
            data: xarray dataset
            variables: Variables to include (None for all)
            dtype: Tensor data type

        Returns:
            Tensor of shape (C, H, W) or (T, C, H, W) if time dimension exists
        """

        if variables is None:
            variables = list(data.data_vars)

        # Stack variables along channel dimension
        arrays = []
        for var in variables:
            if var in data.data_vars:
                arrays.append(data[var].values)

        # Stack arrays
        if arrays:
            stacked = np.stack(arrays, axis=0)
            tensor_dtype = dtype or torch.float32
            tensor = torch.from_numpy(stacked).to(tensor_dtype)
            return tensor
        else:
            raise ValueError("No valid variables found for tensor conversion")


def normalize_track_data(
    track_df: pd.DataFrame,
    preprocessor: Optional[HurricanePreprocessor] = None
) -> pd.DataFrame:
    """Convenience function to normalize track data.

    Args:
        track_df: Hurricane track DataFrame
        preprocessor: Preprocessor instance (creates new if None)

    Returns:
        Normalized DataFrame
    """
    if preprocessor is None:
        preprocessor = HurricanePreprocessor()
    return preprocessor.normalize_track_data(track_df)


def create_track_features(
    track_df: pd.DataFrame,
    preprocessor: Optional[HurricanePreprocessor] = None
) -> pd.DataFrame:
    """Convenience function to create track features.

    Args:
        track_df: Hurricane track DataFrame
        preprocessor: Preprocessor instance (creates new if None)

    Returns:
        DataFrame with engineered features
    """
    if preprocessor is None:
        preprocessor = HurricanePreprocessor()
    return preprocessor.create_track_features(track_df)
