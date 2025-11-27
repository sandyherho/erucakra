"""
erucakra - Climate Tipping Point Dynamics Toy Model

A physically-motivated dynamical system toy model for analyzing
climate tipping points under various SSP scenarios.
"""

__version__ = "0.0.1"

from erucakra.core.model import ClimateModel
from erucakra.core.results import SimulationResults
from erucakra.core.dynamics import (
    climate_tipping_model,
    add_climate_noise,
    compute_fixed_points,
    compute_effective_potential,
    FORCING_SCALES,
    DEFAULT_Z_CRIT,
)
from erucakra.scenarios import SCENARIOS, get_scenario, list_scenarios

__all__ = [
    "__version__",
    "ClimateModel",
    "SimulationResults",
    "climate_tipping_model",
    "add_climate_noise",
    "compute_fixed_points",
    "compute_effective_potential",
    "FORCING_SCALES",
    "DEFAULT_Z_CRIT",
    "SCENARIOS",
    "get_scenario",
    "list_scenarios",
]
