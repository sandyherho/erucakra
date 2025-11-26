"""CSV output writer."""

from typing import TYPE_CHECKING
from pathlib import Path
import logging

if TYPE_CHECKING:
    from erucakra.core.results import SimulationResults

logger = logging.getLogger(__name__)


def write_csv(
    results: "SimulationResults",
    filepath: str | Path,
    include_header: bool = True,
    float_format: str = "%.6f",
) -> None:
    """
    Write simulation results to CSV file.
    
    Parameters
    ----------
    results : SimulationResults
        Simulation results to export.
    filepath : str or Path
        Output file path.
    include_header : bool, optional
        Include metadata header. Default is True.
    float_format : str, optional
        Float format string. Default is "%.6f".
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing CSV to: {filepath}")
    
    df = results.to_dataframe()
    
    if include_header:
        # Build metadata header
        info = results.scenario_info or {}
        scenario_name = info.get("name", "Custom Forcing")
        description = info.get("description", "N/A")
        expected = info.get("expected_outcome", "N/A")
        
        header_lines = [
            "# ================================================================",
            "# Climate Tipping Point Model - Simulation Output",
            "# ================================================================",
            f"# Scenario: {scenario_name}",
            f"# Description: {description}",
            f"# Expected Outcome: {expected}",
            "# ",
            "# Model: dx/dt = y",
            "#        dy/dt = x(z - 1 - x²) - cy",
            "#        dz/dt = ε(A(t) - z - βx²)",
            "#",
            f"# Parameters: c={results.model_params.get('c', 0.2)}, "
            f"ε={results.model_params.get('epsilon', 0.02)}, "
            f"β={results.model_params.get('beta', 0.8)}, "
            f"z_crit={results.model_params.get('z_crit', 1.0)}",
            "# Time: normalized (t) and calendar year",
            "# Tipping threshold: z = 1.0 (approx 1.5°C warming)",
            "#",
            "# Columns:",
            "#   time_normalized    - Simulation time units",
            "#   year               - Calendar year (2020 baseline)",
            "#   x_variability      - Fast climate oscillations",
            "#   y_momentum         - Rate of change of x",
            "#   z_accumulated      - Slow accumulated forcing (ocean heat)",
            "#   A_forcing          - External radiative forcing",
            "#   warming_proxy      - Approximate warming in °C",
            "#   phase_velocity     - Speed in phase space",
            "#   distance_to_threshold - z - 1.0 (negative = safe)",
            "#   regime             - 'stable' or 'tipped'",
            "# ================================================================",
        ]
        header = "\n".join(header_lines) + "\n"
        
        with open(filepath, "w") as f:
            f.write(header)
            df.to_csv(f, index=False, float_format=float_format)
    else:
        df.to_csv(filepath, index=False, float_format=float_format)
    
    logger.info(f"CSV written: {len(df)} rows")
