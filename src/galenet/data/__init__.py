"""Data loading, preprocessing, and validation modules for GaleNet."""

# Re-export data loaders so users can access them from ``galenet.data``
from .era5 import ERA5Loader
from .hurdat2 import HURDAT2Loader
from .ibtracs import IBTrACSLoader
from .pipeline import HurricaneDataPipeline
from .satellite import SatelliteLoader
from .processors import (
    ERA5Preprocessor,
    HurricanePreprocessor,
    create_track_features,
    normalize_track_data,
)
from .validators import (
    HurricaneDataValidator,
    validate_era5_data,
    validate_intensity_physics,
    validate_track_continuity,
    validate_training_data,
)

__all__ = [
    # Loaders
    "HURDAT2Loader",
    "IBTrACSLoader",
    "ERA5Loader",
    "SatelliteLoader",
    "HurricaneDataPipeline",
    # Processors
    "HurricanePreprocessor",
    "ERA5Preprocessor",
    "normalize_track_data",
    "create_track_features",
    # Validators
    "HurricaneDataValidator",
    "validate_track_continuity",
    "validate_intensity_physics",
    "validate_era5_data",
    "validate_training_data",
]
