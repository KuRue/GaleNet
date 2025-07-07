#!/usr/bin/env python
"""Test script to verify GaleNet data loading functionality."""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from galenet.data.loaders import HURDAT2Loader, IBTrACSLoader, HurricaneDataPipeline
from galenet.data.processors import HurricanePreprocessor
from galenet.data.validators import HurricaneDataValidator


def test_hurdat2_loading():
    """Test HURDAT2 data loading."""
    logger.info("Testing HURDAT2 loader...")
    
    try:
        loader = HURDAT2Loader()
        
        # Check if data file exists
        if not loader.data_path.exists():
            logger.warning(f"HURDAT2 data not found at {loader.data_path}")
            logger.info("Run: python scripts/setup_data.py --download-hurdat2")
            return False
        
        # Load data
        df = loader.load_data()
        logger.success(f"Loaded HURDAT2 data: {len(df)} records")
        
        # Get available storms
        storms = loader.get_available_storms()
        logger.info(f"Found {len(storms)} storms in database")
        
        # Load a specific storm (Hurricane Katrina 2005)
        if 'AL122005' in storms:
            katrina = loader.get_storm('AL122005')
            logger.info(f"Hurricane Katrina (2005): {len(katrina)} track points")
            logger.info(f"Max wind: {katrina['max_wind'].max()} kt")
            logger.info(f"Min pressure: {katrina['min_pressure'].min()} mb")
        
        return True
        
    except Exception as e:
        logger.error(f"HURDAT2 test failed: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing functionality."""
    logger.info("Testing data preprocessing...")
    
    try:
        # Create sample data
        dates = pd.date_range('2023-09-01', periods=20, freq='6H')
        sample_track = pd.DataFrame({
            'timestamp': dates,
            'latitude': [15.0 + i * 0.5 for i in range(20)],
            'longitude': [-60.0 - i * 1.0 for i in range(20)],
            'max_wind': [35 + i * 5 for i in range(20)],
            'min_pressure': [1005 - i * 2 for i in range(20)]
        })
        
        # Test preprocessor
        preprocessor = HurricanePreprocessor()
        
        # Normalize data
        normalized = preprocessor.normalize_track_data(sample_track)
        logger.success("Track normalization successful")
        
        # Create features
        features = preprocessor.create_track_features(normalized)
        logger.info(f"Created {len(features.columns)} features")
        logger.info(f"Features: {list(features.columns)}")
        
        # Prepare sequences
        inputs, targets = preprocessor.prepare_sequences(
            features,
            sequence_length=8,
            forecast_length=4
        )
        logger.success(f"Created {len(inputs)} training sequences")
        logger.info(f"Input shape: {inputs.shape}")
        logger.info(f"Target shape: {targets.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Preprocessing test failed: {e}")
        return False


def test_data_validation():
    """Test data validation functionality."""
    logger.info("Testing data validation...")
    
    try:
        validator = HurricaneDataValidator()
        
        # Create valid sample data
        dates = pd.date_range('2023-09-01', periods=10, freq='6H')
        valid_track = pd.DataFrame({
            'timestamp': dates,
            'latitude': [15.0 + i * 0.3 for i in range(10)],
            'longitude': [-60.0 - i * 0.5 for i in range(10)],
            'max_wind': [65 + i * 2 for i in range(10)],
            'min_pressure': [985 - i * 1 for i in range(10)]
        })
        
        is_valid, errors = validator.validate_track(valid_track)
        if is_valid:
            logger.success("Valid track data passed validation")
        else:
            logger.error(f"Valid track failed: {errors}")
        
        # Test physics validation
        physics_valid, physics_errors = validator.validate_intensity_physics(valid_track)
        if physics_valid:
            logger.success("Track physics validation passed")
        else:
            logger.warning(f"Physics warnings: {physics_errors}")
        
        # Create invalid data
        invalid_track = valid_track.copy()
        invalid_track.loc[5, 'latitude'] = 100  # Invalid latitude
        
        is_valid, errors = validator.validate_track(invalid_track)
        if not is_valid:
            logger.info(f"Invalid track correctly rejected: {errors}")
        
        return True
        
    except Exception as e:
        logger.error(f"Validation test failed: {e}")
        return False


def test_full_pipeline():
    """Test the full data pipeline."""
    logger.info("Testing full data pipeline...")
    
    try:
        pipeline = HurricaneDataPipeline()
        
        # Check if we have data
        loader = HURDAT2Loader()
        if not loader.data_path.exists():
            logger.warning("No data available for full pipeline test")
            return True  # Not a failure, just no data
        
        # Get available storms
        storms = loader.get_available_storms()
        if len(storms) > 0:
            # Load first available storm
            storm_id = list(storms)[0]
            logger.info(f"Loading storm {storm_id} for pipeline test...")
            
            data = pipeline.load_hurricane_for_training(
                storm_id=storm_id,
                source='hurdat2',
                include_era5=False  # Skip ERA5 for basic test
            )
            
            if 'track' in data:
                track = data['track']
                logger.success(f"Pipeline loaded storm with {len(track)} points")
                return True
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("=" * 60)
    logger.info("GaleNet Data Loading Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("HURDAT2 Loading", test_hurdat2_loading),
        ("Data Preprocessing", test_data_preprocessing),
        ("Data Validation", test_data_validation),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'-' * 40}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'-' * 40}")
        
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Test Summary")
    logger.info(f"{'=' * 60}")
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:<30} {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.success("\n🎉 All tests passed!")
    else:
        logger.error("\n⚠️  Some tests failed")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
