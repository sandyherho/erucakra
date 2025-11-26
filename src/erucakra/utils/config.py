"""Configuration management for erucakra."""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "scenarios": {
        "default": "ssp245",
        "available": ["ssp126", "ssp245", "ssp370", "ssp585", "overshoot"],
    },
    "model": {
        "damping": 0.2,
        "epsilon": 0.02,
        "beta": 0.8,
        "z_critical": 1.0,
    },
    "simulation": {
        "t_start": 0.0,
        "t_end": 150.0,
        "n_points": 12000,
        "initial_state": [0.05, 0.0, 0.3],
        "add_noise": True,
        "noise_level": 0.03,
        "noise_smoothing": 15.0,
        "rtol": 1e-10,
        "atol": 1e-12,
        "method": "RK45",
    },
    "outputs": {
        "formats": ["csv", "netcdf", "gif", "png"],
        "base_dir": "./outputs",
        "subdirs": {
            "csv": "csv",
            "netcdf": "netcdf",
            "gif": "gif",
            "png": "png",
        },
    },
    "logging": {
        "level": "INFO",
        "log_dir": "./logs",
        "include_timestamp": True,
        "format_style": "detailed",
    },
    "visualization": {
        "timeseries_dpi": 200,
        "gif_fps": 30,
        "gif_duration": 12,
        "gif_dpi": 100,
    },
}


def load_config(
    config_path: Optional[str | Path] = None,
    create_default: bool = True,
) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to config file. Defaults to ~/.erucakra/config.yaml
    create_default : bool, optional
        Create default config if not found. Default is True.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path is None:
        config_path = Path.home() / ".erucakra" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        
        # Merge with defaults
        config = _deep_merge(DEFAULT_CONFIG.copy(), user_config)
        return config
    
    if create_default:
        save_config(DEFAULT_CONFIG, config_path)
    
    return DEFAULT_CONFIG.copy()


def save_config(
    config: Dict[str, Any],
    config_path: Optional[str | Path] = None,
) -> None:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    config_path : str or Path, optional
        Path to config file. Defaults to ~/.erucakra/config.yaml
    """
    if config_path is None:
        config_path = Path.home() / ".erucakra" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Config saved to: {config_path}")


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
