# flake8: noqa: E501
"""Unit tests for GaleNet data loaders."""

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cdsapi
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Stub out heavy training/torch dependencies for lightweight tests
sys.modules.setdefault("torch", SimpleNamespace())
training_stub = SimpleNamespace(HurricaneDataset=None, Trainer=None, mse_loss=None)
sys.modules.setdefault("galenet.training", training_stub)

from galenet.data.loaders import (
    HURDAT2Loader,
    ERA5Loader,
    HurricaneDataPipeline,
)  # noqa: E402
from galenet.data.processors import ERA5Preprocessor  # noqa: E402
from galenet.data.processors import HurricanePreprocessor
from galenet.data.validators import HurricaneDataValidator  # noqa: E402


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
            'd2m': xr.DataArray(np.random.randn(*shape) * 5 + 293, dims=['time', 'latitude', 'longitude']),
            'u200': xr.DataArray(np.random.randn(*shape), dims=['time', 'latitude', 'longitude']),
            'v200': xr.DataArray(np.random.randn(*shape), dims=['time', 'latitude', 'longitude']),
            'u850': xr.DataArray(np.random.randn(*shape), dims=['time', 'latitude', 'longitude']),
            'v850': xr.DataArray(np.random.randn(*shape), dims=['time', 'latitude', 'longitude']),
            'sst': xr.DataArray(np.random.randn(*shape) * 2 + 300, dims=['time', 'latitude', 'longitude']),
        },
        coords={
            'time': time,
            'latitude': lat,
            'longitude': lon
        }
    )

    return ds


@pytest.fixture
def multi_year_storms():
    """Create synthetic storms across multiple years with varying intensity."""

    def make_storm(storm_id, name, year, winds):
        timestamps = pd.date_range(f"{year}-08-01", periods=len(winds), freq="6H")
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "latitude": np.linspace(10, 15, len(winds)),
                "longitude": np.linspace(-50, -55, len(winds)),
                "max_wind": winds,
                "min_pressure": np.linspace(1000, 950, len(winds)),
                "storm_id": storm_id,
                "name": name,
            }
        )

    storms = {
        "AL012022": make_storm("AL012022", "ALPHA", 2022, [40, 65, 70]),
        "AL022022": make_storm("AL022022", "BETA", 2022, [30, 35, 40]),
        "AL012023": make_storm("AL012023", "CHARLIE", 2023, [60, 70, 80]),
        "AL022023": make_storm("AL022023", "DELTA", 2023, [20, 25, 45]),
    }

    storms_by_year = {
        2022: ["AL012022", "AL022022"],
        2023: ["AL012023", "AL022023"],
    }

    return storms_by_year, storms


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
        assert 'vertical_wind_shear' in enhanced
        assert 'vertical_wind_shear_u' in enhanced
        assert 'vertical_wind_shear_v' in enhanced
        assert 'relative_humidity' in enhanced
        assert 'specific_humidity' in enhanced
        assert 'dewpoint_depression' in enhanced

        # Check wind speed calculation
        expected_speed = np.sqrt(patch['u10']**2 + patch['v10']**2)
        np.testing.assert_allclose(
            enhanced['wind_speed'].values,
            expected_speed.values,
            rtol=1e-5
        )

        # Check vertical wind shear calculation
        expected_shear = np.sqrt((patch['u200'] - patch['u850'])**2 + (patch['v200'] - patch['v850'])**2)
        np.testing.assert_allclose(
            enhanced['vertical_wind_shear'].values,
            expected_shear.values,
            rtol=1e-5
        )
        np.testing.assert_allclose(
            enhanced['vertical_wind_shear_u'].values,
            (patch['u200'] - patch['u850']).values,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            enhanced['vertical_wind_shear_v'].values,
            (patch['v200'] - patch['v850']).values,
            rtol=1e-5,
        )

        # Check relative humidity calculation
        t_c = patch['t2m'] - 273.15
        td_c = patch['d2m'] - 273.15
        expected_rh = 100.0 * np.exp((17.625 * td_c) / (243.04 + td_c)) / np.exp((17.625 * t_c) / (243.04 + t_c))
        expected_rh = xr.where(expected_rh > 100, 100, expected_rh)
        expected_rh = xr.where(expected_rh < 0, 0, expected_rh)
        np.testing.assert_allclose(
            enhanced['relative_humidity'].values,
            expected_rh.values,
            rtol=1e-5
        )

        # Check specific humidity calculation
        td_c = patch['d2m'] - 273.15
        p_hpa = patch['msl'] / 100.0
        e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
        expected_q = (0.622 * e) / (p_hpa - 0.378 * e)
        np.testing.assert_allclose(
            enhanced['specific_humidity'].values,
            expected_q.values,
            rtol=1e-5,
        )

        # Check dewpoint depression calculation
        expected_dd = patch['t2m'] - patch['d2m']
        np.testing.assert_allclose(
            enhanced['dewpoint_depression'].values,
            expected_dd.values,
            rtol=1e-5,
        )

    def test_variable_stats_persistence(self, sample_era5_data, tmp_path):
        """Ensure stats include derived variables and can be saved/loaded."""
        preprocessor = ERA5Preprocessor()
        patch = preprocessor.extract_patches(
            sample_era5_data,
            center_lat=20.0,
            center_lon=-55.0,
            patch_size=10.0,
        )
        enhanced = preprocessor.compute_derived_fields(patch)
        preprocessor.normalize_variables(enhanced, fit=True)

        stats_file = tmp_path / "stats.json"
        preprocessor.save_stats(stats_file)
        assert stats_file.exists()

        new_preprocessor = ERA5Preprocessor()
        new_preprocessor.load_stats(stats_file)
        for key in enhanced.data_vars:
            assert key in new_preprocessor.variable_stats

    def test_multi_variable_normalization(self, sample_era5_data):
        """Ensure multiple variables are normalized correctly."""
        preprocessor = ERA5Preprocessor()
        patch = preprocessor.extract_patches(
            sample_era5_data,
            center_lat=20.0,
            center_lon=-55.0,
            patch_size=10.0,
        )
        enhanced = preprocessor.compute_derived_fields(patch)

        vars_to_norm = ['u10', 'v10', 'specific_humidity', 'vertical_wind_shear']
        normalized = preprocessor.normalize_variables(enhanced, fit=True, variables=vars_to_norm)

        for var in vars_to_norm:
            np.testing.assert_allclose(float(normalized[var].mean()), 0.0, atol=1e-6)
            np.testing.assert_allclose(float(normalized[var].std()), 1.0, atol=1e-6)

        # Apply again without fitting to ensure same result
        normalized2 = preprocessor.normalize_variables(enhanced, fit=False, variables=vars_to_norm)
        for var in vars_to_norm:
            np.testing.assert_allclose(
                normalized[var].values,
                normalized2[var].values,
                rtol=1e-6,
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


class TestHurricaneDataPipeline:
    """Tests for the HurricaneDataPipeline utility methods."""

    def test_prepare_training_dataset_filters_min_intensity(self, multi_year_storms):
        storms_by_year, storms = multi_year_storms

        pipeline = HurricaneDataPipeline.__new__(HurricaneDataPipeline)
        pipeline.hurdat2 = MagicMock()
        pipeline.hurdat2.get_storms_by_year.side_effect = lambda y: storms_by_year.get(y, [])
        pipeline.hurdat2.get_storm.side_effect = lambda sid: storms[sid]

        result = pipeline.prepare_training_dataset([2022, 2023], min_intensity=64)
        assert set(result["storm_id"]) == {"AL012022", "AL012023"}

        expected_cols = {
            "storm_id",
            "name",
            "year",
            "max_intensity",
            "min_pressure",
            "num_records",
        }
        assert set(result.columns) == expected_cols

        empty = pipeline.prepare_training_dataset([2022, 2023], min_intensity=200)
        assert empty.empty


class TestERA5Loader:
    """Tests for ERA5Loader retry logic and error handling."""

    def _mock_config(self):
        era5 = SimpleNamespace(api_url="http://example.com", api_key="dummy")
        data = SimpleNamespace(era5=era5)
        return SimpleNamespace(data=data)

    def test_download_retries(self, tmp_path, monkeypatch):
        """Ensure download retries on failure."""
        monkeypatch.setattr(
            'galenet.data.loaders.get_config',
            lambda *args, **kwargs: self._mock_config(),
        )

        mock_client = MagicMock()
        mock_client.retrieve.side_effect = [Exception('boom'), None]
        monkeypatch.setattr(cdsapi, 'Client', lambda **kwargs: mock_client)
        monkeypatch.setattr('time.sleep', lambda s: None)

        loader = ERA5Loader(cache_dir=tmp_path)

        loader.download_data(
            datetime(2023, 1, 1),
            datetime(2023, 1, 1),
            (10, -20, 5, -15),
            variables=['var'],
        )

        assert mock_client.retrieve.call_count == 2

    def test_download_failure_propagates_error(self, tmp_path, monkeypatch):
        """Ensure errors propagate after max retries."""
        monkeypatch.setattr(
            'galenet.data.loaders.get_config',
            lambda *args, **kwargs: self._mock_config(),
        )

        mock_client = MagicMock()
        mock_client.retrieve.side_effect = Exception('401 Unauthorized')
        monkeypatch.setattr(cdsapi, 'Client', lambda **kwargs: mock_client)
        monkeypatch.setattr('time.sleep', lambda s: None)

        loader = ERA5Loader(cache_dir=tmp_path)

        with pytest.raises(RuntimeError) as exc:
            loader.download_data(
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                (10, -20, 5, -15),
                variables=['var'],
            )

        assert 'API credentials' in str(exc.value)
        assert mock_client.retrieve.call_count == 3

    def test_yearly_caching_reuses_files(self, tmp_path, monkeypatch):
        """Subsequent calls for a single year use cached files."""

        loader = ERA5Loader(cache_dir=tmp_path)

        calls = []

        def fake_download(self, start, end, bounds, variables):
            fname = tmp_path / (
                f"era5_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_"
                f"{bounds[0]}N_{abs(bounds[1])}W_{bounds[2]}N_{abs(bounds[3])}W.nc"
            )
            if fname.exists():
                return fname
            calls.append((start, end))
            times = pd.date_range(start, end, freq="6H")
            ds = xr.Dataset(
                {
                    "u10": xr.DataArray(
                        np.zeros((len(times), 1, 1)),
                        dims=["time", "latitude", "longitude"],
                    )
                },
                coords={"time": times, "latitude": [0], "longitude": [0]},
            )
            ds.to_netcdf(fname)
            return fname

        monkeypatch.setattr(ERA5Loader, "_download_single_period", fake_download)

        start = datetime(2022, 12, 30)
        end = datetime(2023, 1, 2)
        bounds = (10, -20, 5, -15)

        # Initial multi-year request populates yearly cache
        loader.download_data(start, end, bounds, variables=["u10"])
        assert len(calls) == 2

        calls.clear()
        # Request one of the years again; should hit cache and avoid new downloads
        loader.download_data(datetime(2022, 12, 30), datetime(2022, 12, 31), bounds, variables=["u10"])
        assert len(calls) == 0

    def test_multi_year_caching(self, tmp_path, monkeypatch):
        """Ensure multi-year ranges use yearly caching and merged file."""

        loader = ERA5Loader(cache_dir=tmp_path)

        calls = []

        def fake_download(self, start, end, bounds, variables):
            calls.append((start, end))
            fname = tmp_path / (
                f"era5_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_"
                f"{bounds[0]}N_{abs(bounds[1])}W_{bounds[2]}N_{abs(bounds[3])}W.nc"
            )
            times = pd.date_range(start, end, freq="6H")
            ds = xr.Dataset(
                {
                    "u10": xr.DataArray(
                        np.zeros((len(times), 1, 1)),
                        dims=["time", "latitude", "longitude"],
                    )
                },
                coords={"time": times, "latitude": [0], "longitude": [0]},
            )
            ds.to_netcdf(fname)
            return fname

        monkeypatch.setattr(ERA5Loader, "_download_single_period", fake_download)

        start = datetime(2022, 12, 30)
        end = datetime(2023, 1, 2)
        bounds = (10, -20, 5, -15)

        # First call should trigger downloads for two separate periods
        loader.download_data(start, end, bounds, variables=["u10"])
        assert len(calls) == 2

        calls.clear()
        # Second call should use cached merged file and not call fake_download
        loader.download_data(start, end, bounds, variables=["u10"])
        assert len(calls) == 0

    def test_extract_patches_dateline_crossing(self, tmp_path, monkeypatch):
        """Dateline-crossing tracks are stitched from two downloads."""

        loader = ERA5Loader(cache_dir=tmp_path)

        track_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=2, freq="6H"),
                "latitude": [0, 0],
                "longitude": [170, -170],
            }
        )

        calls = []

        def fake_download(start, end, bounds, variables):
            calls.append(bounds)
            lon_min, lon_max = bounds[1], bounds[3]
            times = pd.date_range(start, end, freq="6H")
            ds = xr.Dataset(
                {
                    "u10": xr.DataArray(
                        np.zeros((len(times), 1, 2)),
                        dims=["time", "latitude", "longitude"],
                    )
                },
                coords={"time": times, "latitude": [0], "longitude": [lon_min, lon_max]},
            )
            fname = tmp_path / f"era5_patch_{len(calls)}.nc"
            ds.to_netcdf(fname)
            return fname

        monkeypatch.setattr(loader, "download_data", fake_download)

        ds = loader.extract_hurricane_patches(track_df, patch_size=10, variables=["u10"])

        assert len(calls) == 2
        # Ensure we requested both sides of the dateline
        assert any(b[1] >= 0 or b[3] > 0 for b in calls)
        assert any(b[1] < 0 or b[3] < 0 for b in calls)
        # Combined dataset should have monotonically increasing longitudes
        assert np.all(np.diff(ds.longitude.values) > 0)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
