"""Configuration management for erucakra."""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# UPDATED DEFAULT CONFIGURATION
# =============================================================================
# Key changes:
# - z_crit is now an absolute value (0.55) instead of threshold_fraction
# - A_scale is global (13.0 W/m²) for all scenarios
# - This ensures proper differentiation between SSP scenarios

DEFAULT_CONFIG: Dict[str, Any] = {
    "scenarios": {
        "default": "ssp245",
        "available": ["ssp126", "ssp245", "ssp370", "ssp585"],
    },
    # Custom forcing file - when set, overrides scenario
    "forcing_file": None,
    "model": {
        "damping": 0.2,
        "epsilon": 0.02,
        "beta": 0.8,
        # UPDATED: Use absolute z_crit instead of threshold_fraction
        "z_crit": 0.55,  # Absolute threshold in normalized units
        # DEPRECATED: threshold_fraction is no longer used by default
        # "threshold_fraction": 0.7,
    },
    "simulation": {
        "t_start": 0.0,
        "t_end": 600.0,
        "n_points": 48000,
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
        "format_style": "detailed",
    },
    "visualization": {
        "timeseries_dpi": 200,
        "gif_fps": 30,
        "gif_duration": 12,
        "gif_dpi": 100,
    },
    # Optional scenario metadata for custom forcings
    "scenario": {
        "key": None,
        "name": None,
        "subtitle": None,
        "description": None,
    },
    # Global normalization settings (NEW)
    "normalization": {
        # Global forcing scale (W/m²) - based on SSP5-8.5 peak
        "A_scale_global": 13.0,
        # Use absolute z_crit (True) or compute from threshold_fraction (False)
        "use_absolute_threshold": True,
    },
}


# =============================================================================
# EXPECTED BEHAVIOR WITH NEW DEFAULTS
# =============================================================================
# With z_crit = 0.55 and A_scale = 13.0 W/m²:
#
# Scenario  | Peak Forcing | Max A_norm | z_max (approx) | Outcome
# ----------|--------------|------------|----------------|----------
# SSP1-2.6  | 3.6 W/m²     | 0.28       | ~0.28          | STABLE
# SSP2-4.5  | 5.6 W/m²     | 0.43       | ~0.43          | MARGINAL
# SSP3-7.0  | 11.6 W/m²    | 0.89       | ~0.85          | TIPPING
# SSP5-8.5  | 13.2 W/m²    | 1.02       | ~1.0           | CATASTROPHIC
#
# The z variable relaxes toward A_normalized (minus feedback), so:
# - SSP1-2.6: z_max ~ 0.28 < 0.55 → Never tips
# - SSP2-4.5: z_max ~ 0.43 < 0.55 → Stays below but close (marginal)
# - SSP3-7.0: z_max ~ 0.85 > 0.55 → Tips around 2140
# - SSP5-8.5: z_max ~ 1.0 > 0.55 → Tips earlier, stays tipped


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
        
        if user_config is None:
            user_config = {}
        
        # Merge with defaults
        config = _deep_merge(DEFAULT_CONFIG.copy(), user_config)
        
        # Handle legacy threshold_fraction configs
        if "threshold_fraction" in config.get("model", {}):
            logger.warning(
                "Config contains deprecated 'threshold_fraction'. "
                "Consider using 'z_crit' (absolute value) instead."
            )
            # If z_crit not explicitly set, compute from threshold_fraction
            if "z_crit" not in user_config.get("model", {}):
                config["model"]["z_crit"] = None  # Will trigger legacy computation
        
        # Resolve relative paths for forcing_file relative to config file location
        if config.get("forcing_file"):
            forcing_path = Path(config["forcing_file"])
            if not forcing_path.is_absolute():
                # Make relative to config file directory
                config["forcing_file"] = str(config_path.parent / forcing_path)
        
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


def get_expected_outcomes() -> Dict[str, Dict[str, Any]]:
    """
    Get expected outcomes for each scenario with current defaults.
    
    Returns
    -------
    dict
        Expected behavior for each SSP scenario.
    """
    A_scale = DEFAULT_CONFIG["normalization"]["A_scale_global"]
    z_crit = DEFAULT_CONFIG["model"]["z_crit"]
    
    return {
        "ssp126": {
            "peak_forcing": 3.6,
            "max_A_normalized": 3.6 / A_scale,
            "expected_outcome": "STABLE",
            "will_tip": False,
            "reason": f"max_A_norm ({3.6/A_scale:.2f}) < z_crit ({z_crit})",
        },
        "ssp245": {
            "peak_forcing": 5.6,
            "max_A_normalized": 5.6 / A_scale,
            "expected_outcome": "MARGINAL",
            "will_tip": False,
            "reason": f"max_A_norm ({5.6/A_scale:.2f}) < z_crit ({z_crit}), but close",
        },
        "ssp370": {
            "peak_forcing": 11.6,
            "max_A_normalized": 11.6 / A_scale,
            "expected_outcome": "TIPPING",
            "will_tip": True,
            "reason": f"max_A_norm ({11.6/A_scale:.2f}) > z_crit ({z_crit})",
        },
        "ssp585": {
            "peak_forcing": 13.2,
            "max_A_normalized": 13.2 / A_scale,
            "expected_outcome": "CATASTROPHIC",
            "will_tip": True,
            "reason": f"max_A_norm ({13.2/A_scale:.2f}) >> z_crit ({z_crit})",
        },
    }
