"""
SSP (Shared Socioeconomic Pathways) forcing scenarios.

Based on IPCC AR6 WG1 (2021) scenarios:
- SSP1-2.6: Sustainability
- SSP2-4.5: Middle of the Road
- SSP3-7.0: Regional Rivalry
- SSP5-8.5: Fossil-fueled Development
- Overshoot: Net-zero with temporary exceedance
"""

from typing import Dict, Any, List, Callable
import numpy as np


def create_forcing_ssp126(t: float) -> float:
    """
    SSP1-2.6: Sustainability Pathway.
    
    Rapid decarbonization starting 2025, net-zero by 2075,
    peak forcing ~2.6 W/m² then decline.
    
    Parameters
    ----------
    t : float
        Normalized simulation time.
    
    Returns
    -------
    float
        Forcing value (normalized).
    """
    year = 2020 + t * 0.8
    
    if year < 2025:
        return 0.6 + 0.02 * (year - 2020)
    elif year < 2050:
        peak = 0.7
        return peak + 0.3 * np.sin(np.pi * (year - 2025) / 50)
    elif year < 2080:
        return 0.85 - 0.005 * (year - 2050)
    else:
        return 0.7


def create_forcing_ssp245(t: float) -> float:
    """
    SSP2-4.5: Middle of the Road.
    
    Moderate mitigation efforts, emissions peak around 2040-2050,
    stabilization at ~4.5 W/m² equivalent.
    
    Parameters
    ----------
    t : float
        Normalized simulation time.
    
    Returns
    -------
    float
        Forcing value (normalized).
    """
    year = 2020 + t * 0.8
    
    if year < 2040:
        return 0.6 + 0.015 * (year - 2020)
    elif year < 2070:
        base = 0.9
        return base + 0.4 * (1 - np.exp(-(year - 2040) / 30))
    else:
        return 1.25 + 0.05 * np.sin(0.1 * (year - 2070))


def create_forcing_ssp370(t: float) -> float:
    """
    SSP3-7.0: Regional Rivalry.
    
    Delayed and fragmented action, peak emissions around 2070-2080,
    high forcing with late stabilization.
    
    Parameters
    ----------
    t : float
        Normalized simulation time.
    
    Returns
    -------
    float
        Forcing value (normalized).
    """
    year = 2020 + t * 0.8
    
    if year < 2050:
        return 0.6 + 0.025 * (year - 2020)
    elif year < 2080:
        return 1.35 + 0.6 * np.sin(np.pi * (year - 2050) / 60)
    else:
        return 1.8 - 0.003 * (year - 2080)


def create_forcing_ssp585(t: float) -> float:
    """
    SSP5-8.5: Fossil-fueled Development.
    
    No significant mitigation, continuous high emissions growth,
    crosses multiple tipping thresholds.
    
    Parameters
    ----------
    t : float
        Normalized simulation time.
    
    Returns
    -------
    float
        Forcing value (normalized).
    """
    year = 2020 + t * 0.8
    
    if year < 2100:
        return 0.6 + 0.02 * (year - 2020) + 0.0003 * (year - 2020)**1.5
    else:
        return 2.8 + 0.1 * np.sin(0.05 * (year - 2100))


def create_forcing_overshoot(t: float) -> float:
    """
    Overshoot Scenario: Net-Zero with Temporary Exceedance.
    
    Delayed action until 2035, aggressive negative emissions after 2060,
    temporarily exceeds threshold, attempts return.
    
    Parameters
    ----------
    t : float
        Normalized simulation time.
    
    Returns
    -------
    float
        Forcing value (normalized).
    """
    year = 2020 + t * 0.8
    
    if year < 2035:
        return 0.6 + 0.03 * (year - 2020)
    elif year < 2060:
        return 1.05 + 0.02 * (year - 2035)
    elif year < 2090:
        peak = 1.55
        return peak - 0.025 * (year - 2060)
    else:
        return 0.8


# Scenario configurations
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "ssp126": {
        "name": "SSP1-2.6: Sustainability",
        "subtitle": 'Strong Mitigation — "Taking the Green Road"',
        "forcing_func": create_forcing_ssp126,
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
        "forcing_func": create_forcing_ssp245,
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
        "forcing_func": create_forcing_ssp370,
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
        "forcing_func": create_forcing_ssp585,
        "color_primary": "#FF0066",
        "color_secondary": "#CC0044",
        "color_trajectory": "#FF3388",
        "color_forcing": "#AA0033",
        "cmap_name": "magma",
        "expected_outcome": "CATASTROPHIC",
        "description": "Multiple tipping points crossed — chaotic high-amplitude regime",
    },
    "overshoot": {
        "name": "Overshoot & Return",
        "subtitle": 'Delayed Net-Zero — "Betting on the Future"',
        "forcing_func": create_forcing_overshoot,
        "color_primary": "#AA55FF",
        "color_secondary": "#7722CC",
        "color_trajectory": "#CC88FF",
        "color_forcing": "#00FF00",
        "cmap_name": "cool",
        "expected_outcome": "HYSTERESIS",
        "description": "Tests irreversibility — can we return after crossing threshold?",
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
        "overshoot": "overshoot",
        "netzero": "overshoot",
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
