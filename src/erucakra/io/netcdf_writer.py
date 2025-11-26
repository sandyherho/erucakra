"""NetCDF output writer (CF-compliant)."""

from typing import TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import logging
import numpy as np

if TYPE_CHECKING:
    from erucakra.core.results import SimulationResults

logger = logging.getLogger(__name__)


def write_netcdf(
    results: "SimulationResults",
    filepath: str | Path,
    compression: bool = True,
    compression_level: int = 4,
) -> None:
    """
    Write simulation results to NetCDF file.
    
    Creates a CF-compliant NetCDF4 file with full metadata.
    
    Parameters
    ----------
    results : SimulationResults
        Simulation results to export.
    filepath : str or Path
        Output file path.
    compression : bool, optional
        Enable zlib compression. Default is True.
    compression_level : int, optional
        Compression level (1-9). Default is 4.
    """
    import netCDF4 as nc
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing NetCDF to: {filepath}")
    
    # Compression settings
    comp_kwargs = {}
    if compression:
        comp_kwargs = {"zlib": True, "complevel": compression_level}
    
    with nc.Dataset(filepath, "w", format="NETCDF4") as ds:
        # Global attributes
        ds.title = "Climate Tipping Point Model Simulation"
        ds.institution = "erucakra"
        ds.source = "erucakra climate tipping point model v0.0.1"
        ds.history = f"Created {datetime.now().isoformat()} by erucakra"
        ds.Conventions = "CF-1.8"
        
        info = results.scenario_info or {}
        ds.scenario = info.get("name", "Custom Forcing")
        ds.scenario_description = info.get("description", "N/A")
        ds.expected_outcome = info.get("expected_outcome", "N/A")
        
        # Model parameters
        ds.model_damping_c = results.model_params.get("c", 0.2)
        ds.model_epsilon = results.model_params.get("epsilon", 0.02)
        ds.model_beta = results.model_params.get("beta", 0.8)
        ds.model_z_critical = results.model_params.get("z_crit", 1.0)
        
        # Dimensions
        n_time = len(results.t)
        ds.createDimension("time", n_time)
        
        # Time coordinate (normalized)
        time_var = ds.createVariable("time", "f8", ("time",), **comp_kwargs)
        time_var.units = "normalized_time_units"
        time_var.long_name = "Normalized simulation time"
        time_var.standard_name = "time"
        time_var[:] = results.t
        
        # Calendar year
        year_var = ds.createVariable("year", "f8", ("time",), **comp_kwargs)
        year_var.units = "year"
        year_var.long_name = "Calendar year"
        year_var.calendar = "proleptic_gregorian"
        year_var[:] = results.year
        
        # State variables
        x_var = ds.createVariable("x_variability", "f8", ("time",), **comp_kwargs)
        x_var.units = "1"
        x_var.long_name = "Climate variability (fast variable)"
        x_var.description = "Interannual-decadal climate oscillations"
        x_var[:] = results.x
        
        y_var = ds.createVariable("y_momentum", "f8", ("time",), **comp_kwargs)
        y_var.units = "1/time"
        y_var.long_name = "Rate of change (momentum)"
        y_var.description = "Time derivative of climate variability"
        y_var[:] = results.y
        
        z_var = ds.createVariable("z_accumulated", "f8", ("time",), **comp_kwargs)
        z_var.units = "1"
        z_var.long_name = "Accumulated forcing (slow variable)"
        z_var.description = "Ocean heat content / ice sheet state proxy"
        z_var.tipping_threshold = 1.0
        z_var[:] = results.z
        
        # Forcing
        A_var = ds.createVariable("A_forcing", "f8", ("time",), **comp_kwargs)
        A_var.units = "W m-2"
        A_var.long_name = "Effective radiative forcing"
        A_var.standard_name = "toa_incoming_shortwave_flux"
        A_var[:] = results.A
        
        # Derived quantities
        warm_var = ds.createVariable("warming_proxy", "f8", ("time",), **comp_kwargs)
        warm_var.units = "K"
        warm_var.long_name = "Warming proxy (approximate temperature anomaly)"
        warm_var[:] = results.warming
        
        vel_var = ds.createVariable("phase_velocity", "f8", ("time",), **comp_kwargs)
        vel_var.units = "1/time"
        vel_var.long_name = "Phase space velocity magnitude"
        vel_var[:] = results.velocity
        
        dist_var = ds.createVariable("distance_to_threshold", "f8", ("time",), **comp_kwargs)
        dist_var.units = "1"
        dist_var.long_name = "Distance from tipping threshold (z - 1)"
        dist_var.positive_means = "above_threshold"
        dist_var[:] = results.distance_to_threshold
        
        # Regime (as integer flags)
        regime_var = ds.createVariable("regime", "i2", ("time",), **comp_kwargs)
        regime_var.units = "1"
        regime_var.long_name = "Climate regime flag"
        regime_var.flag_values = np.array([0, 1], dtype=np.int16)
        regime_var.flag_meanings = "stable tipped"
        regime_var[:] = np.where(results.z > 1, 1, 0).astype(np.int16)
    
    logger.info(f"NetCDF written: {n_time} time steps")
