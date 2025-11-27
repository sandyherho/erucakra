"""
SSP (Shared Socioeconomic Pathways) forcing scenarios.

Based on IPCC AR6 WG1 (2021) scenarios using CSV forcing data:
- SSP1-2.6: Sustainability
- SSP2-4.5: Middle of the Road
- SSP3-7.0: Regional Rivalry
- SSP5-8.5: Fossil-fueled Development

Forcing data extends from 1750 to 2500.
"""

from typing import Dict, Any, List, Callable, Optional
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d


# Path to forcing data files
FORCING_DIR = Path(__file__).parent.parent.parent.parent / "forcings"


def _load_forcing_data(scenario_key: str) -> tuple:
    """Load forcing data from CSV file."""
    csv_path = FORCING_DIR / f"{scenario_key}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Forcing file not found: {csv_path}")
    
    import pandas as pd
    df = pd.read_csv(csv_path, comment='#')
    return df['time'].values, df['forcing'].values


def create_forcing_from_csv(
    scenario_key: str,
    cache: Optional[Dict[str, Callable]] = None,
) -> Callable[[float], float]:
    """
    Create forcing function from CSV data.
    
    Parameters
    ----------
    scenario_key : str
        Scenario identifier (e.g., 'ssp126', 'ssp245').
    cache : dict, optional
        Cache dictionary for storing interpolators.
    
    Returns
    -------
    Callable
        Function f(t_normalized) -> forcing_value.
    """
    # Use cache if available
    if cache is not None and scenario_key in cache:
        return cache[scenario_key]
    
    times, values = _load_forcing_data(scenario_key)
    
    # Convert years from 1750 to normalized time
    # t_norm = (year - 2020) / 0.8
    t_norm = (times - 2020) / 0.8
    
    interpolator = interp1d(
        t_norm, values,
        kind='linear',
        bounds_error=False,
        fill_value=(values[0], values[-1]),
    )
    
    def forcing_func(t: float) -> float:
        return float(interpolator(t))
    
    if cache is not None:
        cache[scenario_key] = forcing_func
    
    return forcing_func


# Cache for forcing functions
_forcing_cache: Dict[str, Callable] = {}


def _get_forcing_func(scenario_key: str) -> Callable[[float], float]:
    """Get cached forcing function for scenario."""
    return create_forcing_from_csv(scenario_key, _forcing_cache)


# Scenario configurations
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "ssp126": {
        "name": "SSP1-2.6: Sustainability",
        "subtitle": 'Strong Mitigation — "Taking the Green Road"',
        "forcing_func": lambda t: _get_forcing_func("ssp126")(t),
        "color_primary": "#00E5CC",
        "color_secondary": "#00FF88",
        "color_trajectory": "#00FFFF",
        "color_forcing": "#88FF00",
        "cmap_name": "viridis",
        "expected_outcome": "STABLE",
        "description": "Aggressive early action keeps system below tipping threshold",
    },
    "ssp245": {
        "name": "SSP2-4.5: Middle Road",
        "subtitle": 'Moderate Action — "Muddling Through"',
        "forcing_func": lambda t: _get_forcing_func("ssp245")(t),
        "color_primary": "#FFD700",
        "color_secondary": "#FFA500",
        "color_trajectory": "#FFFF00",
        "color_forcing": "#FF8C00",
        "cmap_name": "plasma",
        "expected_outcome": "MARGINAL",
        "description": "System hovers near threshold — outcome uncertain",
    },
    "ssp370": {
        "name": "SSP3-7.0: Regional Rivalry",
        "subtitle": 'Fragmented Action — "A Rocky Road"',
        "forcing_func": lambda t: _get_forcing_func("ssp370")(t),
        "color_primary": "#FF6B35",
        "color_secondary": "#FF4444",
        "color_trajectory": "#FF8855",
        "color_forcing": "#FF2200",
        "cmap_name": "inferno",
        "expected_outcome": "TIPPING",
        "description": "Delayed action leads to threshold crossing and regime shift",
    },
    "ssp585": {
        "name": "SSP5-8.5: Fossil Development",
        "subtitle": 'No Mitigation — "Highway to Warming"',
        "forcing_func": lambda t: _get_forcing_func("ssp585")(t),
        "color_primary": "#FF0066",
        "color_secondary": "#CC0044",
        "color_trajectory": "#FF3388",
        "color_forcing": "#AA0033",
        "cmap_name": "magma",
        "expected_outcome": "CATASTROPHIC",
        "description": "Multiple tipping points crossed — chaotic high-amplitude regime",
    },
}


def get_scenario(key: str) -> Dict[str, Any]:
    """
    Get scenario configuration by key.
    
    Parameters
    ----------
    key : str
        Scenario key (e.g., "ssp126", "ssp245").
    
    Returns
    -------
    dict
        Scenario configuration.
    
    Raises
    ------
    KeyError
        If scenario not found.
    """
    # Normalize key
    key_lower = key.lower().replace("-", "").replace("_", "")
    
    # Direct match
    if key_lower in SCENARIOS:
        return SCENARIOS[key_lower]
    
    # Try alternative formats
    key_map = {
        "ssp126": "ssp126",
        "ssp12.6": "ssp126",
        "ssp1": "ssp126",
        "sustainability": "ssp126",
        "ssp245": "ssp245",
        "ssp24.5": "ssp245",
        "ssp2": "ssp245",
        "middleroad": "ssp245",
        "ssp370": "ssp370",
        "ssp37.0": "ssp370",
        "ssp3": "ssp370",
        "rivalry": "ssp370",
        "ssp585": "ssp585",
        "ssp58.5": "ssp585",
        "ssp5": "ssp585",
        "fossil": "ssp585",
    }
    
    if key_lower in key_map:
        return SCENARIOS[key_map[key_lower]]
    
    available = list(SCENARIOS.keys())
    raise KeyError(f"Unknown scenario '{key}'. Available: {available}")


def list_scenarios() -> List[str]:
    """
    List available scenario keys.
    
    Returns
    -------
    List[str]
        List of scenario identifiers.
    """
    return list(SCENARIOS.keys())
