"""
Built-in SSP forcing scenarios.
"""

from erucakra.scenarios.ssp import (
    SCENARIOS,
    get_scenario,
    list_scenarios,
    create_forcing_ssp126,
    create_forcing_ssp245,
    create_forcing_ssp370,
    create_forcing_ssp585,
    create_forcing_overshoot,
)

__all__ = [
    "SCENARIOS",
    "get_scenario",
    "list_scenarios",
    "create_forcing_ssp126",
    "create_forcing_ssp245",
    "create_forcing_ssp370",
    "create_forcing_ssp585",
    "create_forcing_overshoot",
]
