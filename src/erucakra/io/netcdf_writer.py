"""NetCDF output writer (CF-compliant) with threshold metadata."""

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
    
    Creates a CF-compliant NetCDF4 file with full metadata including
    the critical threshold z_crit and forcing scale A_scale.
    
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
        ds.scenario_key = results.scenario_key or "custom"
        ds.scenario_description = info.get("description", "N/A")
        ds.expected_outcome = info.get("expected_outcome", "N/A")
        
        # Model parameters - including z_crit and A_scale
        ds.model_damping_c = results.model_params.get("c", 0.2)
        ds.model_epsilon = results.model_params.get("epsilon", 0.02)
        ds.model_beta = results.model_params.get("beta", 0.8)
        ds.model_z_critical = results.z_crit
        ds.model_A_scale = results.A_scale
        
        # Diagnostics as attributes
        diag = results.diagnostics
        ds.max_z = diag.get("max_z", float(np.max(results.z)))
        ds.final_z = diag.get("final_z", float(results.z[-1]))
        ds.crossed_threshold = int(results.crossed_threshold)
        ds.time_above_threshold_pct = diag.get("time_above_threshold_pct", 0.0)
        if diag.get("first_crossing_year"):
            ds.first_crossing_year = diag["first_crossing_year"]
        ds.lyapunov_exponent = diag.get("lyapunov_exponent", 0.0)
        
        # Dimensions
        n_time = len(results.t)
        ds.createDimension("time", n_time)
        
        # Real calendar year as primary time coordinate
        year_var = ds.createVariable("year", "i4", ("time",), **comp_kwargs)
        year_var.units = "year"
        year_var.long_name = "Calendar year"
        year_var.calendar = "proleptic_gregorian"
        year_var.standard_name = "time"
        year_var[:] = results.real_year
        
        # Normalized time (secondary)
        time_var = ds.createVariable("time_normalized", "f8", ("time",), **comp_kwargs)
        time_var.units = "normalized_time_units"
        time_var.long_name = "Normalized simulation time"
        time_var.comment = "t_norm = (year - 2020) / 0.8"
        time_var[:] = results.t
        
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
        z_var.tipping_threshold = results.z_crit
        z_var.comment = f"Tipping occurs when z > z_crit = {results.z_crit:.3f}"
        z_var[:] = results.z
        
        # Forcing (raw)
        A_var = ds.createVariable("A_forcing", "f8", ("time",), **comp_kwargs)
        A_var.units = "W m-2"
        A_var.long_name = "Effective radiative forcing"
        A_var.standard_name = "toa_incoming_shortwave_flux"
        A_var[:] = results.A
        
        # Normalized forcing
        A_norm_var = ds.createVariable("A_normalized", "f8", ("time",), **comp_kwargs)
        A_norm_var.units = "1"
        A_norm_var.long_name = "Normalized radiative forcing"
        A_norm_var.comment = f"A_normalized = A_forcing / {results.A_scale:.2f}"
        A_norm_var.scale_factor_used = results.A_scale
        A_norm_var[:] = results.A_normalized
        
        # Derived quantities
        warm_var = ds.createVariable("warming_proxy", "f8", ("time",), **comp_kwargs)
        warm_var.units = "K"
        warm_var.long_name = "Warming proxy (approximate temperature anomaly)"
        warm_var.comment = "warming = z * 1.5 (rough approximation)"
        warm_var[:] = results.warming
        
        vel_var = ds.createVariable("phase_velocity", "f8", ("time",), **comp_kwargs)
        vel_var.units = "1/time"
        vel_var.long_name = "Phase space velocity magnitude"
        vel_var[:] = results.velocity
        
        dist_var = ds.createVariable("distance_to_threshold", "f8", ("time",), **comp_kwargs)
        dist_var.units = "1"
        dist_var.long_name = f"Distance from tipping threshold (z - {results.z_crit:.2f})"
        dist_var.positive_means = "above_threshold"
        dist_var.threshold_value = results.z_crit
        dist_var[:] = results.distance_to_threshold
        
        # Regime (as integer flags)
        regime_var = ds.createVariable("regime", "i2", ("time",), **comp_kwargs)
        regime_var.units = "1"
        regime_var.long_name = "Climate regime flag"
        regime_var.flag_values = np.array([0, 1], dtype=np.int16)
        regime_var.flag_meanings = "stable tipped"
        regime_var.threshold = results.z_crit
        regime_var[:] = np.where(results.z > results.z_crit, 1, 0).astype(np.int16)
    
    logger.info(f"NetCDF written: {n_time} time steps, z_crit={results.z_crit:.3f}")
