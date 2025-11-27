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
    
    Creates a CF-compliant NetCDF4 file with full metadata including
    the computed critical threshold z_crit and forcing scale A_scale.
    
    Time is stored in multiple formats:
    - year: Integer year (2020, 2021, etc.)
    - year_decimal: Decimal year (2020.5 = mid-2020)
    - month: Month of year (1-12)
    - time_normalized: Model's internal normalized time
    
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
    
    comp_kwargs = {}
    if compression:
        comp_kwargs = {"zlib": True, "complevel": compression_level}
    
    # Compute time arrays
    year_decimal = results.year
    year_int = np.floor(year_decimal).astype(np.int32)
    month = ((year_decimal % 1) * 12 + 1).astype(np.int32)
    month = np.clip(month, 1, 12)
    day_of_year = ((year_decimal % 1) * 365 + 1).astype(np.int32)
    day_of_year = np.clip(day_of_year, 1, 365)
    
    with nc.Dataset(filepath, "w", format="NETCDF4") as ds:
        # =====================================================================
        # Global attributes
        # =====================================================================
        ds.title = "Climate Tipping Point Model Simulation"
        ds.institution = "erucakra"
        ds.source = "erucakra climate tipping point model v0.0.1"
        ds.history = f"Created {datetime.now().isoformat()} by erucakra"
        ds.Conventions = "CF-1.8"
        
        info = results.scenario_info or {}
        ds.scenario = info.get("name", "Custom Forcing")
        ds.scenario_key = results.scenario_key or "custom"
        ds.scenario_description = info.get("description", "N/A")
        
        # Model parameters
        ds.model_damping_c = results.model_params.get("c", 0.2)
        ds.model_epsilon = results.model_params.get("epsilon", 0.02)
        ds.model_beta = results.model_params.get("beta", 0.8)
        ds.model_threshold_fraction = results.threshold_fraction
        
        # Computed parameters
        ds.computed_z_critical = results.z_crit
        ds.computed_A_scale = results.A_scale
        ds.z_crit_computation_method = "threshold_fraction * max(A_normalized)"
        
        # Forcing analysis
        fa = results.forcing_analysis
        if fa:
            ds.forcing_A_max = fa.get("A_max", np.nan)
            ds.forcing_A_min = fa.get("A_min", np.nan)
            ds.forcing_A_normalized_max = fa.get("A_normalized_max", np.nan)
        
        # Diagnostics
        diag = results.diagnostics
        ds.max_z = diag.get("max_z", float(np.max(results.z)))
        ds.final_z = diag.get("final_z", float(results.z[-1]))
        ds.tipped = int(results.tipped)
        ds.time_above_threshold_pct = diag.get("time_above_threshold_pct", 0.0)
        if diag.get("first_crossing_year"):
            ds.first_crossing_year = diag["first_crossing_year"]
        
        # =====================================================================
        # Dimensions
        # =====================================================================
        n_time = len(results.t)
        ds.createDimension("time", n_time)
        
        # =====================================================================
        # Time Variables - Multiple formats for convenience
        # =====================================================================
        
        # Integer year (2020, 2021, etc.)
        year_var = ds.createVariable("year", "i4", ("time",), **comp_kwargs)
        year_var.units = "year"
        year_var.long_name = "Calendar year (integer)"
        year_var.standard_name = "year"
        year_var[:] = year_int
        
        # Decimal year (2020.5 = mid-2020)
        year_dec_var = ds.createVariable("year_decimal", "f8", ("time",), **comp_kwargs)
        year_dec_var.units = "year"
        year_dec_var.long_name = "Decimal year (fractional)"
        year_dec_var.comment = "2020.5 represents mid-2020 (approximately July 1)"
        year_dec_var[:] = year_decimal
        
        # Month (1-12)
        month_var = ds.createVariable("month", "i2", ("time",), **comp_kwargs)
        month_var.units = "1"
        month_var.long_name = "Month of year"
        month_var.valid_range = np.array([1, 12], dtype=np.int16)
        month_var[:] = month.astype(np.int16)
        
        # Day of year (1-365)
        doy_var = ds.createVariable("day_of_year", "i2", ("time",), **comp_kwargs)
        doy_var.units = "1"
        doy_var.long_name = "Day of year"
        doy_var.valid_range = np.array([1, 365], dtype=np.int16)
        doy_var[:] = day_of_year.astype(np.int16)
        
        # Normalized time (model internal)
        time_var = ds.createVariable("time_normalized", "f8", ("time",), **comp_kwargs)
        time_var.units = "normalized_time_units"
        time_var.long_name = "Normalized simulation time"
        time_var.comment = "Model internal time; year = 2020 + t * 0.8"
        time_var[:] = results.t
        
        # =====================================================================
        # State Variables
        # =====================================================================
        
        x_var = ds.createVariable("x_variability", "f8", ("time",), **comp_kwargs)
        x_var.units = "1"
        x_var.long_name = "Climate variability (fast variable)"
        x_var.description = "Interannual-decadal climate oscillations"
        x_var[:] = results.x
        
        y_var = ds.createVariable("y_momentum", "f8", ("time",), **comp_kwargs)
        y_var.units = "1/time"
        y_var.long_name = "Rate of change (momentum)"
        y_var[:] = results.y
        
        z_var = ds.createVariable("z_accumulated", "f8", ("time",), **comp_kwargs)
        z_var.units = "1"
        z_var.long_name = "Accumulated forcing (slow variable)"
        z_var.description = "Ocean heat content / ice sheet state proxy"
        z_var.tipping_threshold = results.z_crit
        z_var.comment = f"Tipping occurs when z > z_crit = {results.z_crit:.3f} (computed)"
        z_var[:] = results.z
        
        # =====================================================================
        # Forcing Variables
        # =====================================================================
        
        A_var = ds.createVariable("A_forcing", "f8", ("time",), **comp_kwargs)
        A_var.units = "W m-2"
        A_var.long_name = "Effective radiative forcing"
        A_var.standard_name = "toa_incoming_shortwave_flux"
        A_var[:] = results.A
        
        A_norm_var = ds.createVariable("A_normalized", "f8", ("time",), **comp_kwargs)
        A_norm_var.units = "1"
        A_norm_var.long_name = "Normalized radiative forcing"
        A_norm_var.comment = f"A_normalized = A_forcing / {results.A_scale:.2f}"
        A_norm_var.scale_factor_used = results.A_scale
        A_norm_var[:] = results.A_normalized
        
        # =====================================================================
        # Derived Quantities
        # =====================================================================
        
        warm_var = ds.createVariable("warming_proxy", "f8", ("time",), **comp_kwargs)
        warm_var.units = "K"
        warm_var.long_name = "Warming proxy (approximate temperature anomaly)"
        warm_var.comment = "warming = z * 1.5"
        warm_var[:] = results.warming
        
        vel_var = ds.createVariable("phase_velocity", "f8", ("time",), **comp_kwargs)
        vel_var.units = "1/time"
        vel_var.long_name = "Phase space velocity magnitude"
        vel_var.comment = "sqrt(x^2 + y^2)"
        vel_var[:] = results.velocity
        
        dist_var = ds.createVariable("distance_to_threshold", "f8", ("time",), **comp_kwargs)
        dist_var.units = "1"
        dist_var.long_name = f"Distance from tipping threshold (z - z_crit)"
        dist_var.positive_means = "above_threshold"
        dist_var.threshold_value = results.z_crit
        dist_var[:] = results.distance_to_threshold
        
        # =====================================================================
        # Regime Classification
        # =====================================================================
        
        regime_var = ds.createVariable("regime", "i2", ("time",), **comp_kwargs)
        regime_var.units = "1"
        regime_var.long_name = "Climate regime flag"
        regime_var.flag_values = np.array([0, 1], dtype=np.int16)
        regime_var.flag_meanings = "not_tipped tipped"
        regime_var.threshold = results.z_crit
        regime_var[:] = np.where(results.z > results.z_crit, 1, 0).astype(np.int16)
        
        above_var = ds.createVariable("above_threshold", "i2", ("time",), **comp_kwargs)
        above_var.units = "1"
        above_var.long_name = "Above tipping threshold flag"
        above_var.flag_values = np.array([0, 1], dtype=np.int16)
        above_var.flag_meanings = "below above"
        above_var[:] = (results.z > results.z_crit).astype(np.int16)
    
    logger.info(f"NetCDF written: {n_time} time steps, "
                f"years {year_int[0]}-{year_int[-1]}, "
                f"z_crit={results.z_crit:.3f}")
