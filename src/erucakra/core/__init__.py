"""Core simulation components."""

from erucakra.core.model import ClimateModel
from erucakra.core.results import SimulationResults
from erucakra.core.dynamics import climate_tipping_model, add_climate_noise

__all__ = [
    "ClimateModel",
    "SimulationResults",
    "climate_tipping_model",
    "add_climate_noise",
]
