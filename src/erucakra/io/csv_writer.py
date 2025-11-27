"""CSV output writer."""

from typing import TYPE_CHECKING
from pathlib import Path
import logging
import numpy as np

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
        Include column header. Default is True.
    float_format : str, optional
        Float format string. Default is "%.6f".
    """
    import pandas as pd
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing CSV to: {filepath}")
    
    # Create DataFrame with proper time columns
    df = pd.DataFrame({
        # Time columns - split into integer year and decimal year
        "year": np.floor(results.year).astype(int),
        "year_decimal": results.year,
        "month": ((results.year % 1) * 12 + 1).astype(int).clip(1, 12),
        "time_normalized": results.t,
        
        # State variables
        "x_variability": results.x,
        "y_momentum": results.y,
        "z_accumulated": results.z,
        
        # Forcing
        "A_forcing_Wm2": results.A,
        "A_normalized": results.A_normalized,
        
        # Derived quantities
        "warming_proxy_celsius": results.warming,
        "distance_to_threshold": results.distance_to_threshold,
        "phase_velocity": results.velocity,
        
        # Regime classification
        "regime": results.regime,
        "above_threshold": (results.z > results.z_crit).astype(int),
    })
    
    # Write CSV
    df.to_csv(filepath, index=False, float_format=float_format)
    
    logger.info(f"CSV written: {len(df)} rows, years {df['year'].iloc[0]}-{df['year'].iloc[-1]}")
