"""Data loading, preprocessing, and validation modules for GaleNet."""

# Re-export data loaders so users can access them from ``galenet.data``
from .loaders import (
    ERA5Loader,
    HURDAT2Loader,
    HurricaneDataPipeline,
    IBTrACSLoader,
)
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
