"""Configuration management for GaleNet."""

import os
from pathlib import Path
from typing import Optional, Union, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def get_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[list] = None
) -> DictConfig:
    """Load configuration from file or use defaults.

    Args:
        config_path: Path to config file or name of config
        overrides: List of config overrides in Hydra format

    Returns:
        Configuration object
    """
    if config_path is None:
        config_path = "default_config.yaml"

    config_path = Path(config_path)

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    if config_path.exists():
        # Load from file path
        config = OmegaConf.load(config_path)
    else:
        # Try to load from configs directory
        configs_dir = Path(__file__).parent.parent.parent.parent / "configs"

        if not configs_dir.exists():
            # Fallback to default config
            logger.warning(f"Config directory not found at {configs_dir}")
            config = get_default_config()
        else:
            # Initialize Hydra with config directory
            with initialize_config_dir(
                config_dir=str(configs_dir), version_base="1.3"
            ):
                config = cast(
                    DictConfig,
                    compose(
                        config_name=str(config_path).replace(".yaml", ""),
                        overrides=overrides or [],
                    ),
                )

    # Apply any overrides
    if overrides:
        for override in overrides:
            key, value = override.split('=', 1)
            OmegaConf.update(config, key, value)

    # Set environment variables
    OmegaConf.register_new_resolver("env", lambda x: os.environ.get(x, ''))

    return cast(DictConfig, config)


def get_default_config() -> DictConfig:
    """Get default configuration.

    Returns:
        Default configuration
    """
    default_config = {
        "project": {
            "name": "galenet",
            "version": "0.1.0",
            "seed": 42,
            "device": "cuda",
            "mixed_precision": True
        },
        "data": {
            "root_dir": f"{os.environ.get('HOME', '.')}/data/galenet",
            "hurdat2_path": "${data.root_dir}/hurdat2/hurdat2.txt",
            "ibtracs_path": "${data.root_dir}/ibtracs/IBTrACS.ALL.v04r00.nc",
            "era5_cache_dir": "${data.root_dir}/era5",
            "min_hurricane_intensity": 64,
            "training_years": list(range(2010, 2020)),
            "validation_years": [2020, 2021],
            "test_years": [2022, 2023],
            "pipeline": {
                "batch_size": 4,
                "num_workers": 4,
                "prefetch_factor": 2,
                "pin_memory": True,
                "shuffle": True
            }
        },
        "model": {
            "name": "hurricane_ensemble",
            "ensemble": {
                "size": 50,
                "method": "perturbation",
                "perturbation_scale": 0.01
            }
        },
        "training": {
            "epochs": 100,
            "learning_rate": 5e-5,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "gradient_accumulation_steps": 8
        },
        "logging": {
            "level": "INFO",
            "mlflow": {
                "enabled": True,
                "tracking_uri": "file://${data.root_dir}/mlruns",
                "experiment_name": "galenet_experiments"
            }
        }
    }

    config = OmegaConf.create(default_config)
    OmegaConf.set_struct(config, False)

    return config


def save_config(config: DictConfig, path: Union[str, Path]) -> None:
    """Save configuration to file.

    Args:
        config: Configuration to save
        path: Path to save to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        OmegaConf.save(config, f)

    logger.info(f"Configuration saved to {path}")


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configurations.

    Args:
        *configs: Configurations to merge

    Returns:
        Merged configuration
    """
    merged: DictConfig = cast(DictConfig, OmegaConf.create({}))

    for config in configs:
        merged = cast(DictConfig, OmegaConf.merge(merged, config))

    return merged


def validate_config(config: DictConfig) -> bool:
    """Validate configuration.

    Args:
        config: Configuration to validate

    Returns:
        Whether configuration is valid
    """
    required_keys = [
        "project.name",
        "data.root_dir",
        "model.name",
        "training.epochs"
    ]

    for key in required_keys:
        try:
            value = OmegaConf.select(config, key)
            if value is None:
                logger.error(f"Missing required config key: {key}")
                return False
        except Exception:
            logger.error(f"Invalid config key: {key}")
            return False

    return True
