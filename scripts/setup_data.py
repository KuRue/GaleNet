#!/usr/bin/env python
"""Setup script to download hurricane and model data for GaleNet."""

import os
import sys
import argparse
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
import cdsapi
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
        
    Returns:
        Success status
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def setup_directories(base_dir: Path) -> dict:
    """Create necessary directories.
    
    Args:
        base_dir: Base data directory
        
    Returns:
        Dictionary of created directories
    """
    dirs = {
        'base': base_dir,
        'hurdat2': base_dir / 'hurdat2',
        'ibtracs': base_dir / 'ibtracs',
        'era5': base_dir / 'era5',
        'models': base_dir / 'models',
        'cache': base_dir / 'cache',
        'checkpoints': base_dir / 'checkpoints'
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return dirs


def download_hurdat2(data_dir: Path) -> bool:
    """Download HURDAT2 Atlantic hurricane database.
    
    Args:
        data_dir: Directory to save data
        
    Returns:
        Success status
    """
    logger.info("Downloading HURDAT2 Atlantic hurricane database...")
    
    url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
    dest_path = data_dir / 'hurdat2' / 'hurdat2.txt'
    
    if dest_path.exists():
        logger.info("HURDAT2 data already exists, skipping download")
        return True
    
    success = download_file(url, dest_path, "HURDAT2")
    
    if success:
        logger.success(f"HURDAT2 data saved to {dest_path}")
    
    return success


def download_ibtracs(data_dir: Path) -> bool:
    """Download IBTrACS global tropical cyclone database.
    
    Args:
        data_dir: Directory to save data
        
    Returns:
        Success status
    """
    logger.info("Downloading IBTrACS database...")
    
    # Download netCDF version
    url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/IBTrACS.ALL.v04r00.nc"
    dest_path = data_dir / 'ibtracs' / 'IBTrACS.ALL.v04r00.nc'
    
    if dest_path.exists():
        logger.info("IBTrACS data already exists, skipping download")
        return True
    
    success = download_file(url, dest_path, "IBTrACS")
    
    if success:
        logger.success(f"IBTrACS data saved to {dest_path}")
    
    return success


def setup_era5_credentials() -> bool:
    """Setup CDS API credentials for ERA5 download.
    
    Returns:
        Whether credentials are configured
    """
    cdsapirc_path = Path.home() / '.cdsapirc'
    
    if cdsapirc_path.exists():
        logger.info("CDS API credentials already configured")
        return True
    
    logger.warning("CDS API credentials not found!")
    logger.info("""
To download ERA5 data, you need to:
1. Register at https://cds.climate.copernicus.eu/user/register
2. Get your API key from https://cds.climate.copernicus.eu/api-how-to
3. Create ~/.cdsapirc file with:
   
url: https://cds.climate.copernicus.eu/api/v2
key: UID:API-KEY

Replace UID and API-KEY with your credentials.
""")
    
    return False


def download_era5_sample(data_dir: Path) -> bool:
    """Download sample ERA5 data for testing.
    
    Args:
        data_dir: Directory to save data
        
    Returns:
        Success status
    """
    if not setup_era5_credentials():
        logger.warning("Skipping ERA5 download - no credentials")
        return False
    
    logger.info("Downloading sample ERA5 data...")
    
    # Create ERA5 directory
    era5_dir = data_dir / 'era5'
    era5_dir.mkdir(exist_ok=True)
    
    # Sample request for Hurricane Katrina period
    c = cdsapi.Client()
    
    output_file = era5_dir / 'era5_sample_katrina_2005.nc'
    
    if output_file.exists():
        logger.info("Sample ERA5 data already exists")
        return True
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'mean_sea_level_pressure',
                    '2m_temperature',
                    'sea_surface_temperature',
                    'total_precipitation'
                ],
                'year': '2005',
                'month': '08',
                'day': ['23', '24', '25', '26', '27', '28', '29', '30'],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'area': [35, -95, 20, -75],  # North, West, South, East
            },
            str(output_file)
        )
        logger.success(f"ERA5 sample data saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download ERA5 data: {e}")
        return False


def download_model_weights(data_dir: Path) -> bool:
    """Download pre-trained model weights.
    
    Args:
        data_dir: Directory to save models
        
    Returns:
        Success status
    """
    logger.info("Downloading model weights...")
    
    models_dir = data_dir / 'models'
    
    # Note: These are placeholder URLs - actual model weights would need to be hosted
    model_urls = {
        'graphcast': {
            'url': 'https://example.com/graphcast_hurricane_v1.pt',  # Placeholder
            'file': 'graphcast_hurricane_v1.pt',
            'size': '2.3GB'
        },
        'pangu': {
            'url': 'https://example.com/pangu_weather_v1.pt',  # Placeholder
            'file': 'pangu_weather_v1.pt',
            'size': '1.8GB'
        }
    }
    
    logger.warning("""
Model weights are not yet publicly available.
Please check the following resources:
- GraphCast: https://github.com/deepmind/graphcast
- Pangu-Weather: https://github.com/198808xc/Pangu-Weather

Once you have the model weights, place them in:
{}
""".format(models_dir))
    
    # Create placeholder files for testing
    for model_name, info in model_urls.items():
        model_path = models_dir / info['file']
        if not model_path.exists():
            # Create empty placeholder
            model_path.touch()
            logger.info(f"Created placeholder for {model_name} at {model_path}")
    
    return True


def verify_setup(data_dir: Path) -> bool:
    """Verify that all required data is present.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Whether setup is complete
    """
    logger.info("Verifying data setup...")
    
    required_files = {
        'HURDAT2': data_dir / 'hurdat2' / 'hurdat2.txt',
        'IBTrACS': data_dir / 'ibtracs' / 'IBTrACS.ALL.v04r00.nc',
    }
    
    optional_files = {
        'ERA5 Sample': data_dir / 'era5' / 'era5_sample_katrina_2005.nc',
        'GraphCast Model': data_dir / 'models' / 'graphcast_hurricane_v1.pt',
        'Pangu Model': data_dir / 'models' / 'pangu_weather_v1.pt',
    }
    
    all_good = True
    
    # Check required files
    for name, path in required_files.items():
        if path.exists():
            logger.success(f"✓ {name}: {path}")
        else:
            logger.error(f"✗ {name}: Missing")
            all_good = False
    
    # Check optional files
    for name, path in optional_files.items():
        if path.exists():
            logger.info(f"✓ {name}: {path}")
        else:
            logger.warning(f"○ {name}: Not found (optional)")
    
    return all_good


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup data for GaleNet hurricane forecasting"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path.home() / 'data' / 'galenet',
        help='Base directory for data storage'
    )
    parser.add_argument(
        '--download-hurdat2',
        action='store_true',
        help='Download HURDAT2 hurricane database'
    )
    parser.add_argument(
        '--download-ibtracs',
        action='store_true',
        help='Download IBTrACS database'
    )
    parser.add_argument(
        '--download-era5',
        action='store_true',
        help='Download sample ERA5 data'
    )
    parser.add_argument(
        '--download-models',
        action='store_true',
        help='Download model weights (placeholders)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all data'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("GaleNet Data Setup")
    logger.info("=" * 50)
    
    # Create directories
    dirs = setup_directories(args.data_dir)
    
    # Download data
    if args.all or args.download_hurdat2:
        download_hurdat2(args.data_dir)
    
    if args.all or args.download_ibtracs:
        download_ibtracs(args.data_dir)
    
    if args.all or args.download_era5:
        download_era5_sample(args.data_dir)
    
    if args.all or args.download_models:
        download_model_weights(args.data_dir)
    
    # Verify setup
    logger.info("\n" + "=" * 50)
    if verify_setup(args.data_dir):
        logger.success("\n✅ Data setup complete!")
        logger.info(f"\nData directory: {args.data_dir}")
        logger.info("\nYou can now run GaleNet with this data directory.")
    else:
        logger.error("\n❌ Data setup incomplete!")
        logger.info("\nPlease download missing required files.")


if __name__ == '__main__':
    main()
