"""
erucakra - Climate Tipping Point Dynamics
==========================================

A physically-motivated dynamical systems model for analyzing climate
tipping points under various SSP (Shared Socioeconomic Pathways) scenarios.

Physical Model:
    dx/dt = y
    dy/dt = x(z - z_crit - x²) - cy
    dz/dt = ε(A(t) - z - βx²)

Where:
    x  → Fast climate variability (interannual-decadal oscillations)
    y  → Rate of change / momentum in climate system
    z  → Slow accumulated forcing (ocean heat content, ice sheet state)
    A  → Effective radiative forcing (W/m² scaled)

Author: Sandy H. S. Herho
License: MIT
"""

__version__ = "0.0.1"
__author__ = "Sandy H. S. Herho"
__license__ = "MIT"

from erucakra.core.model import ClimateModel
from erucakra.core.results import SimulationResults
from erucakra.scenarios import SCENARIOS, get_scenario
from erucakra.io.forcing import load_forcing_csv, load_forcing_txt
from erucakra.utils.logging import setup_logging

__all__ = [
    "ClimateModel",
    "SimulationResults",
    "SCENARIOS",
    "get_scenario",
    "load_forcing_csv",
    "load_forcing_txt",
    "setup_logging",
    "__version__",
    "__author__",
]
