"""
Loading functions for custom forcing data.
"""

from typing import Tuple, Callable, Optional
from pathlib import Path
import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def load_forcing_csv(
    filepath: str | Path,
    time_col: str = "time",
    forcing_col: str = "forcing",
    delimiter: str = ",",
    skip_header: int = 0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load forcing data from CSV file.
    
    Expected format: time (years from 1750), forcing (W/m²)
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV file.
    time_col : str, optional
        Name of time column. Default is "time".
    forcing_col : str, optional
        Name of forcing column. Default is "forcing".
    delimiter : str, optional
        Column delimiter. Default is ",".
    skip_header : int, optional
        Number of header lines to skip. Default is 0.
    
    Returns
    -------
    Tuple[NDArray, NDArray]
        (times, forcing_values) arrays.
    
    Examples
    --------
    >>> times, forcing = load_forcing_csv("my_forcing.csv")
    >>> model.run(forcing=forcing, forcing_times=times)
    """
    import pandas as pd
    
    filepath = Path(filepath)
    logger.info(f"Loading forcing from CSV: {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Forcing file not found: {filepath}")
    
    df = pd.read_csv(filepath, delimiter=delimiter, skiprows=skip_header, comment='#')
    
    # Try to find columns
    cols = df.columns.str.lower()
    
    # Time column
    time_data = None
    for col_name in [time_col, "time", "year", "t", "years"]:
        matches = cols.str.contains(col_name.lower())
        if matches.any():
            time_data = df.iloc[:, matches.argmax()].values
            logger.debug(f"Found time column: {df.columns[matches.argmax()]}")
            break
    
    if time_data is None:
        # Assume first column is time
        time_data = df.iloc[:, 0].values
        logger.warning("Could not find time column, using first column")
    
    # Forcing column
    forcing_data = None
    for col_name in [forcing_col, "forcing", "rf", "radiative", "wm2", "w/m2"]:
        matches = cols.str.contains(col_name.lower())
        if matches.any():
            forcing_data = df.iloc[:, matches.argmax()].values
            logger.debug(f"Found forcing column: {df.columns[matches.argmax()]}")
            break
    
    if forcing_data is None:
        # Assume second column is forcing
        forcing_data = df.iloc[:, 1].values
        logger.warning("Could not find forcing column, using second column")
    
    logger.info(f"Loaded {len(time_data)} forcing data points")
    logger.debug(f"Time range: {time_data[0]:.1f} - {time_data[-1]:.1f}")
    logger.debug(f"Forcing range: {forcing_data.min():.3f} - {forcing_data.max():.3f} W/m²")
    
    return time_data.astype(np.float64), forcing_data.astype(np.float64)


def load_forcing_txt(
    filepath: str | Path,
    time_col: int = 0,
    forcing_col: int = 1,
    delimiter: Optional[str] = None,
    skip_header: int = 0,
    comments: str = "#",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load forcing data from plain text file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to text file.
    time_col : int, optional
        Column index for time. Default is 0.
    forcing_col : int, optional
        Column index for forcing. Default is 1.
    delimiter : str, optional
        Column delimiter. Default is whitespace.
    skip_header : int, optional
        Number of header lines to skip. Default is 0.
    comments : str, optional
        Comment character. Default is "#".
    
    Returns
    -------
    Tuple[NDArray, NDArray]
        (times, forcing_values) arrays.
    """
    filepath = Path(filepath)
    logger.info(f"Loading forcing from TXT: {filepath}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Forcing file not found: {filepath}")
    
    data = np.loadtxt(
        filepath,
        delimiter=delimiter,
        skiprows=skip_header,
        comments=comments,
    )
    
    time_data = data[:, time_col]
    forcing_data = data[:, forcing_col]
    
    logger.info(f"Loaded {len(time_data)} forcing data points")
    
    return time_data.astype(np.float64), forcing_data.astype(np.float64)


def create_forcing_function(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    kind: str = "linear",
) -> Callable[[float], float]:
    """
    Create interpolating forcing function from data.
    
    Parameters
    ----------
    times : NDArray
        Time array (years from 1750).
    values : NDArray
        Forcing values (W/m²).
    kind : str, optional
        Interpolation method. Default is "linear".
    
    Returns
    -------
    Callable
        Function f(t_normalized) -> forcing_value.
    """
    from scipy.interpolate import interp1d
    
    # Convert years from 1750 to normalized time
    # t_norm = (year - 2020) / 0.8
    t_norm = (times - 2020) / 0.8
    
    interpolator = interp1d(
        t_norm, values,
        kind=kind,
        bounds_error=False,
        fill_value=(values[0], values[-1]),
    )
    
    return lambda t: float(interpolator(t))
