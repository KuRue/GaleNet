"""Logging utilities for GaleNet."""

import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format: Optional[str] = None,
    colorize: bool = True,
    enqueue: bool = True
) -> None:
    """Setup logging configuration for GaleNet.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format: Log format string
        colorize: Whether to colorize console output
        enqueue: Whether to enqueue log messages (thread-safe)
    """
    # Remove default logger
    logger.remove()
    
    # Default format
    if format is None:
        if colorize:
            format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            )
        else:
            format = (
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} - "
                "{message}"
            )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=format,
        level=level,
        colorize=colorize,
        enqueue=enqueue
    )
    
    # Add file handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="50 MB",
            retention="10 days",
            compression="zip",
            enqueue=enqueue
        )
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logging initialized at {level} level")


def get_run_logger(
    run_name: Optional[str] = None,
    log_dir: Optional[Union[str, Path]] = None
) -> "logger":
    """Get a logger for a specific run.
    
    Args:
        run_name: Name of the run
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_file = log_dir / f"{run_name}.log"
        
        # Add file handler for this run
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            filter=lambda record: record["extra"].get("run_name") == run_name,
            enqueue=True
        )
    
    # Return logger with run_name context
    return logger.bind(run_name=run_name)


def log_config(config: dict, level: str = "INFO") -> None:
    """Log configuration in a readable format.
    
    Args:
        config: Configuration dictionary
        level: Logging level
    """
    import json
    
    logger.log(level, "Configuration:")
    logger.log(level, "-" * 50)
    
    config_str = json.dumps(config, indent=2, default=str)
    for line in config_str.split('\n'):
        logger.log(level, line)
    
    logger.log(level, "-" * 50)


def log_metrics(
    metrics: dict,
    step: Optional[int] = None,
    prefix: str = "",
    level: str = "INFO"
) -> None:
    """Log metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        step: Optional step number
        prefix: Prefix for metric names
        level: Logging level
    """
    header = f"Metrics"
    if step is not None:
        header += f" (Step {step})"
    if prefix:
        header += f" [{prefix}]"
    
    logger.log(level, header)
    logger.log(level, "-" * 40)
    
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.log(level, f"{name:<25}: {value:>10.4f}")
        else:
            logger.log(level, f"{name:<25}: {value:>10}")
    
    logger.log(level, "-" * 40)
