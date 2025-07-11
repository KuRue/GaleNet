"""Unit tests for GaleNet data loaders."""

import sys
from pathlib import Path
from datetime import datetime
import tempfile

import pytest
import pandas as pd
import numpy as np
import xarray as xr

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from galenet.data.loaders import HURDAT2Loader, IBTrACSLoader, ERA5Loader
from galenet.data.processors import HurricanePreprocessor, ERA5Preprocessor
from galenet.data.validators import (
    HurricaneDataValidator, 
    validate_track_continuity,
    validate_training_data
)


@pytest.fixture
def sample_hurdat2_data():
    """Create sample HURDAT2 format data."""
    data = """AL092023,         LEE,     80,
20230905, 0000,  , TD, 15.0N,  42.9W,  30, 1008,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20230905, 0600,  , TD, 15.2N,  44.0W,  30, 1008,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20230905, 1200,  , TD, 15.5N,  45.2W,  35, 1006,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20230905, 1800,  , TS, 15.8N,  46.5W,  40, 1004,   30,   30,   20,   20,    0,    0,    0,    0,    0,    0,    0,    0,
AL122005,     KATRINA,     34,
20050823, 1800,  , TD, 23.1N,  75.1W,  30, 1008,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20050824, 0000,  , TD, 23.4N,  75.7W,  30, 1007,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20050824, 0600,  , TD, 23.8N,  76.2W,  30, 1007,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20050824, 1200,  , TS, 24.5N,  76.5W,  35, 1006,   60,   60,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,"""
    return data


@pytest.fixture
def temp_hurdat2_file(sample_hurdat2_data):
    """Create temporary HURDAT2 file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_hurdat2_data)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink()


@pytest.fixture
def sample_track_df():
    """Create sample hurricane track DataFrame."""
    dates = pd.date_range('2023-09-05', periods=10, freq='6H')
    return pd.DataFrame({
        'timestamp': dates,
        'latitude': np.linspace(15.0, 25.0, 10),
        'longitude': np.linspace(-45.0, -65.0, 10),
        'max_wind': np.linspace(30, 150, 10),
        'min_pressure': np.linspace(1005, 920, 10),
        'storm_id': 'AL092023',
        'name': 'LEE'
    })


@pytest.fixture
def sample_era5_data():
    """Create sample ERA5 dataset."""
    # Create coordinates
    time = pd.date_range('2023-09-05', periods=24, freq='H')
    lat = np.arange(30, 10, -0.25)
    lon = np.arange(-70, -40, 0.25)
    
    # Create data variables
    shape = (len(time), len(lat), len(lon))
    
    ds = xr.Dataset(
        {
            'u10': xr.DataArray(np.random.randn(*shape), dims=['time', 'latitude', 'longitude']),
            'v10': xr.DataArray(np.random.randn(*shape), dims=['time', 'latitude', 'longitude']),
            'msl': xr.DataArray(np.random.randn(*shape) * 10 + 1013, dims=['time', 'latitude', 'longitude']),
            't2m': xr.DataArray(np.random.randn(*shape) * 5 + 298, dims=['time', 'latitude', 'longitude']),
            'sst': xr.DataArray(np.random.randn(*shape) * 2 + 300, dims=['time', 'latitude', 'longitude']),
        },
        coords={
            'time': time,
            'latitude': lat,
            'longitude': lon
        }
    )
    
    return ds


class TestHURDAT2Loader:
    """Test HURDAT2 data loader."""
    
    def test_load_data(self, temp_hurdat2_file):
        """Test loading HURDAT2 data."""
        loader = HURDAT2Loader(temp_hurdat2_file)
        df = loader.load_data()
        
        # Check data loaded
        assert len(df) > 0
        assert 'storm_id' in df.columns
        assert 'timestamp' in df.columns
        assert 'latitude' in df.columns
        
        # Check we have 2 storms
        assert len(df['storm_id'].unique()) == 2
        assert 'AL092023' in df['storm_id'].values
        assert 'AL122005' in df['storm_id'].values
    
    def test_get_storm(self, temp_hurdat2_file):
        """Test getting specific storm."""
        loader = HURDAT2Loader(temp_hurdat2_file)
        
        # Get Lee (2023)
        lee = loader.get_storm('AL092023')
        assert len(lee) == 4
        assert lee['name'].iloc[0] == 'LEE'
        assert lee['max_wind'].max() == 40
        
        # Get Katrina (2005)
        katrina = loader.get_storm('AL122005')
        assert len(katrina) == 4
        assert katrina['name'].iloc[0] == 'KATRINA'
    
    def test_get_storms_by_year(self, temp_hurdat2_file):
        """Test getting storms by year."""
        loader = HURDAT2Loader(temp_hurdat2_file)
        
        storms_2023 = loader.get_storms_by_year(2023)
        assert len(storms_2023) == 1
        assert storms_2023[0] == 'AL092023'
        
        storms_2005 = loader.get_storms_by_year(2005)
        assert len(storms_2005) == 1
        assert storms_2005[0] == 'AL122005'
    
    def test_missing_storm(self, temp_hurdat2_file):
        """Test error handling for missing storm."""
        loader = HURDAT2Loader(temp_hurdat2_file)
        
        with pytest.raises(ValueError):
            loader.get_storm('AL999999')


class TestDataProcessors:
    """Test data preprocessing functionality."""
    
    def test_normalize_track_data(self, sample_track_df):
        """Test track normalization."""
        preprocessor = HurricanePreprocessor()
        
        normalized = preprocessor.normalize_track_data(sample_track_df)
        
        # Check columns preserved
        assert set(normalized.columns) == set(sample_track_df.columns)
        
        # Check normalization applied
        assert abs(normalized['latitude'].mean()) < 1e-10
        assert abs(normalized['longitude'].mean()) < 1e-10
        assert abs(normalized['latitude'].std() - 1.0) < 1e-10
        assert abs(normalized['longitude'].std() - 1.0) < 1e-10
    
    def test_create_track_features(self, sample_track_df):
        """Test feature engineering."""
        preprocessor = HurricanePreprocessor()
        
        features = preprocessor.create_track_features(sample_track_df)
        
        # Check new features created
        assert 'lat_velocity' in features.columns
        assert 'lon_velocity' in features.columns
        assert 'speed' in features.columns
        assert 'heading' in features.columns
        assert 'wind_change' in features.columns
        assert 'hour_sin' in features.columns
        assert 'hour_cos' in features.columns
        
        # Check feature values reasonable
        assert features['speed'].min() >= 0
        assert features['heading'].min() >= -180
        assert features['heading'].max() <= 180
    
    def test_prepare_sequences(self, sample_track_df):
        """Test sequence preparation."""
        preprocessor = HurricanePreprocessor()
        
        features = preprocessor.create_track_features(sample_track_df)
        inputs, targets = preprocessor.prepare_sequences(
            features,
            sequence_length=4,
            forecast_length=2
        )
        
        # Check shapes
        expected_sequences = len(features) - 4 - 2 + 1
        assert inputs.shape[0] == expected_sequences
        assert inputs.shape[1] == 4  # sequence length
        assert targets.shape[0] == expected_sequences
        assert targets.shape[1] == 2  # forecast length
        
        # Check values are from original data
        assert np.allclose(inputs[0, 0, 0], features['latitude'].iloc[0])


class TestERA5Preprocessor:
    """Test ERA5 preprocessing functionality."""
    
    def test_extract_patches(self, sample_era5_data):
        """Test patch extraction."""
        preprocessor = ERA5Preprocessor()
        
        patch = preprocessor.extract_patches(
            sample_era5_data,
            center_lat=20.0,
            center_lon=-55.0,
            patch_size=10.0
        )
        
        # Check patch size
        lat_size = len(patch.latitude)
        lon_size = len(patch.longitude)
        
        # Should be approximately 10 degrees (40 points at 0.25 resolution)
        assert 35 <= lat_size <= 45
        assert 35 <= lon_size <= 45
        
        # Check all variables present
        for var in ['u10', 'v10', 'msl', 't2m', 'sst']:
            assert var in patch.data_vars
    
    def test_compute_derived_fields(self, sample_era5_data):
        """Test computation of derived fields."""
        preprocessor = ERA5Preprocessor()
        
        # Extract a patch first
        patch = preprocessor.extract_patches(
            sample_era5_data,
            center_lat=20.0,
            center_lon=-55.0,
            patch_size=10.0
        )
        
        # Compute derived fields
        enhanced = preprocessor.compute_derived_fields(patch)
        
        # Check new fields
        assert 'wind_speed' in enhanced
        assert 'wind_direction' in enhanced
        assert 'vorticity_10m' in enhanced
        assert 'convergence_10m' in enhanced
        
        # Check wind speed calculation
        expected_speed = np.sqrt(patch['u10']**2 + patch['v10']**2)
        np.testing.assert_allclose(
            enhanced['wind_speed'].values,
            expected_speed.values,
            rtol=1e-5
        )


class TestDataValidators:
    """Test data validation functionality."""
    
    def test_validate_track_valid(self, sample_track_df):
        """Test validation of valid track."""
        validator = HurricaneDataValidator()
        
        is_valid, errors = validator.validate_track(sample_track_df)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_track_invalid(self):
        """Test validation catches invalid data."""
        validator = HurricaneDataValidator()
        
        # Create invalid track
        bad_track = pd.DataFrame({
            'timestamp': pd.date_range('2023-09-05', periods=3, freq='6H'),
            'latitude': [15.0, 200.0, 25.0],  # Invalid latitude
            'longitude': [-45.0, -50.0, -55.0],
            'max_wind': [30, 40, 50]
        })
        
        is_valid, errors = validator.validate_track(bad_track)
        
        assert not is_valid
        assert len(errors) > 0
        assert any('latitude' in error.lower() for error in errors)
    
    def test_validate_intensity_physics(self):
        """Test physics validation."""
        validator = HurricaneDataValidator()
        
        # Create physically inconsistent data
        track = pd.DataFrame({
            'timestamp': pd.date_range('2023-09-05', periods=3, freq='6H'),
            'latitude': [15.0, 16.0, 17.0],
            'longitude': [-45.0, -46.0, -47.0],
            'max_wind': [150, 150, 150],  # Very high wind
            'min_pressure': [1000, 1000, 1000]  # Inconsistent pressure
        })
        
        is_valid, errors = validator.validate_intensity_physics(track)
        
        assert not is_valid
        assert any('wind-pressure' in error.lower() for error in errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
