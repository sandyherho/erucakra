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
        Include column header. Default is True.
    float_format : str, optional
        Float format string. Default is "%.6f".
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing CSV to: {filepath}")
    
    df = results.to_dataframe()
    
    # Write CSV with column header only (no metadata comments)
    df.to_csv(filepath, index=False, float_format=float_format)
    
    logger.info(f"CSV written: {len(df)} rows")
