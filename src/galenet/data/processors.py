"""Data preprocessing modules for GaleNet hurricane forecasting."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from loguru import logger


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
            normalized_df[position_cols] = self.position_scaler.fit_transform(
                normalized_df[position_cols]
            )
        else:
            normalized_df[position_cols] = self.position_scaler.transform(
                normalized_df[position_cols]
            )
        
        # Normalize intensity (wind/pressure)
        intensity_cols = ['max_wind', 'min_pressure']
        if intensity_cols[0] in normalized_df.columns:
            if fit:
                normalized_df[intensity_cols] = self.intensity_scaler.fit_transform(
                    normalized_df[intensity_cols]
                )
            else:
                normalized_df[intensity_cols] = self.intensity_scaler.transform(
                    normalized_df[intensity_cols]
                )
        
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
        fit: bool = True
    ) -> xr.Dataset:
        """Normalize ERA5 variables to zero mean and unit variance.
        
        Args:
            data: ERA5 dataset
            fit: Whether to fit normalization parameters
            
        Returns:
            Normalized dataset
        """
        normalized = data.copy()
        
        for var in data.data_vars:
            var_data = data[var].values
            
            if fit:
                mean = np.nanmean(var_data)
                std = np.nanstd(var_data)
                self.variable_stats[var] = {'mean': mean, 'std': std}
            else:
                if var not in self.variable_stats:
                    logger.warning(f"No normalization stats for {var}, skipping")
                    continue
                mean = self.variable_stats[var]['mean']
                std = self.variable_stats[var]['std']
            
            # Normalize
            normalized[var] = (data[var] - mean) / (std + 1e-8)
        
        return normalized
    
    def compute_derived_fields(
        self,
        data: xr.Dataset
    ) -> xr.Dataset:
        """Compute derived meteorological fields.
        
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
        
        return enhanced
    
    def to_tensor(
        self,
        data: xr.Dataset,
        variables: Optional[List[str]] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Convert xarray dataset to PyTorch tensor.
        
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
            tensor = torch.from_numpy(stacked).to(dtype)
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
