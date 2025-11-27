"""
Simulation results container with export functionality.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SimulationResults:
    """
    Container for climate simulation results.
    
    Attributes
    ----------
    t : NDArray
        Normalized time array.
    year : NDArray
        Calendar year array (decimal).
    x : NDArray
        Climate variability (fast variable).
    y : NDArray
        Momentum / rate of change.
    z : NDArray
        Accumulated forcing (slow variable).
    A : NDArray
        External radiative forcing (W/m²).
    A_normalized : NDArray
        Normalized forcing (A/A_scale).
    warming : NDArray
        Warming proxy in °C (z * 1.5).
    distance_to_threshold : NDArray
        z - z_crit (positive = above threshold).
    scenario_key : str, optional
        Scenario identifier.
    scenario_info : dict, optional
        Scenario metadata.
    model_params : dict
        Model parameters used (including effective z_crit).
    simulation_params : dict
        Simulation parameters used.
    diagnostics : dict
        Pre-computed diagnostic quantities.
    forcing_analysis : dict, optional
        Analysis of forcing data used to compute z_crit.
    """
    
    t: NDArray[np.float64]
    year: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    A: NDArray[np.float64]
    A_normalized: NDArray[np.float64]
    warming: NDArray[np.float64]
    distance_to_threshold: NDArray[np.float64]
    scenario_key: Optional[str] = None
    scenario_info: Optional[Dict[str, Any]] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    simulation_params: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    forcing_analysis: Optional[Dict[str, Any]] = None
    
    @property
    def z_crit(self) -> float:
        """Effective critical threshold used in simulation."""
        return self.model_params.get("z_crit_effective", 1.0)
    
    @property
    def A_scale(self) -> float:
        """Forcing normalization scale used."""
        return self.model_params.get("A_scale", 1.0)
    
    @property
    def threshold_fraction(self) -> float:
        """Threshold fraction used to compute z_crit."""
        return self.model_params.get("threshold_fraction", 0.7)
    
    @property
    def velocity(self) -> NDArray[np.float64]:
        """Phase space velocity magnitude sqrt(x² + y²)."""
        return np.sqrt(self.x**2 + self.y**2)
    
    @property
    def regime(self) -> NDArray[np.str_]:
        """Climate regime classification based on z vs z_crit."""
        return np.where(self.z > self.z_crit, "tipped", "not_tipped")
    
    @property
    def max_z(self) -> float:
        """Maximum accumulated forcing."""
        return self.diagnostics.get("max_z", float(np.max(self.z)))
    
    @property
    def time_above_threshold(self) -> float:
        """Percentage of time above tipping threshold."""
        return self.diagnostics.get("time_above_threshold_pct", 
                                    float(np.sum(self.z > self.z_crit) / len(self.z) * 100))
    
    @property
    def crossed_threshold(self) -> bool:
        """Whether the tipping threshold was ever crossed."""
        return bool(np.any(self.z > self.z_crit))
    
    @property
    def tipped(self) -> bool:
        """Whether the system tipped (z exceeded z_crit at any point)."""
        return self.diagnostics.get("tipped", self.crossed_threshold)
    
    @property
    def first_crossing_year(self) -> Optional[float]:
        """Year when threshold was first crossed (None if never crossed)."""
        return self.diagnostics.get("first_crossing_year")
    
    @property
    def year_int(self) -> NDArray[np.int32]:
        """Integer calendar year."""
        return np.floor(self.year).astype(np.int32)
    
    @property
    def month(self) -> NDArray[np.int32]:
        """Month of year (1-12)."""
        m = ((self.year % 1) * 12 + 1).astype(np.int32)
        return np.clip(m, 1, 12)
    
    @property
    def day_of_year(self) -> NDArray[np.int32]:
        """Day of year (1-365)."""
        d = ((self.year % 1) * 365 + 1).astype(np.int32)
        return np.clip(d, 1, 365)
    
    # Keep real_year as alias for backward compatibility
    @property
    def real_year(self) -> NDArray[np.int64]:
        """Real calendar year as integer (deprecated, use year_int)."""
        return self.year_int.astype(np.int64)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "scenario": self.scenario_key,
            "name": self.scenario_info.get("name", "Custom") if self.scenario_info else "Custom",
            "expected_outcome": self.scenario_info.get("expected_outcome", "N/A") if self.scenario_info else "N/A",
            "threshold_fraction": self.threshold_fraction,
            "z_crit": self.z_crit,
            "A_scale": self.A_scale,
            "max_z": self.max_z,
            "final_z": float(self.z[-1]),
            "max_A": float(np.max(self.A)),
            "max_A_normalized": float(np.max(self.A_normalized)),
            "time_above_threshold_pct": self.time_above_threshold,
            "crossed_threshold": self.crossed_threshold,
            "tipped": self.tipped,
            "first_crossing_year": self.first_crossing_year,
            "max_variability": self.diagnostics.get("max_variability", float(np.max(np.abs(self.x)))),
            "year_start": int(self.year_int[0]),
            "year_end": int(self.year_int[-1]),
            "n_points": len(self.t),
        }
    
    def get_regime_transitions(self) -> List[Dict[str, Any]]:
        """
        Find all regime transitions (crossings of z_crit).
        
        Returns
        -------
        List[Dict]
            List of transition events with year, direction, z value.
        """
        above = self.z > self.z_crit
        transitions = np.diff(above.astype(int))
        
        events = []
        
        # Upward crossings (not_tipped → tipped)
        up_indices = np.where(transitions > 0)[0]
        for idx in up_indices:
            events.append({
                "year": float(self.year[idx]),
                "year_int": int(self.year_int[idx]),
                "direction": "up",
                "z_before": float(self.z[idx]),
                "z_after": float(self.z[idx + 1]),
                "type": "tipping",
            })
        
        # Downward crossings (tipped → not_tipped)
        down_indices = np.where(transitions < 0)[0]
        for idx in down_indices:
            events.append({
                "year": float(self.year[idx]),
                "year_int": int(self.year_int[idx]),
                "direction": "down",
                "z_before": float(self.z[idx]),
                "z_after": float(self.z[idx + 1]),
                "type": "recovery",
            })
        
        # Sort by year
        events.sort(key=lambda e: e["year"])
        
        return events
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame with proper time columns."""
        import pandas as pd
        
        return pd.DataFrame({
            # Time columns
            "year": self.year_int,
            "year_decimal": self.year,
            "month": self.month,
            "day_of_year": self.day_of_year,
            "time_normalized": self.t,
            
            # State variables
            "x_variability": self.x,
            "y_momentum": self.y,
            "z_accumulated": self.z,
            
            # Forcing
            "A_forcing_Wm2": self.A,
            "A_normalized": self.A_normalized,
            
            # Derived quantities
            "warming_proxy_celsius": self.warming,
            "distance_to_threshold": self.distance_to_threshold,
            "phase_velocity": self.velocity,
            
            # Regime
            "regime": self.regime,
            "above_threshold": (self.z > self.z_crit).astype(int),
        })
    
    def to_csv(
        self,
        filepath: str | Path,
        include_header: bool = True,
        float_format: str = "%.6f",
    ) -> None:
        """
        Export results to CSV file.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path.
        include_header : bool, optional
            Include column header. Default is True.
        float_format : str, optional
            Float format string. Default is "%.6f".
        """
        from erucakra.io.csv_writer import write_csv
        write_csv(self, filepath, include_header, float_format)
    
    def to_netcdf(
        self,
        filepath: str | Path,
        compression: bool = True,
        compression_level: int = 4,
    ) -> None:
        """
        Export results to NetCDF file.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path.
        compression : bool, optional
            Enable compression. Default is True.
        compression_level : int, optional
            Compression level (1-9). Default is 4.
        """
        from erucakra.io.netcdf_writer import write_netcdf
        write_netcdf(self, filepath, compression, compression_level)
    
    def to_gif(
        self,
        filepath: str | Path,
        fps: int = 30,
        duration_seconds: int = 12,
        dpi: int = 100,
    ) -> None:
        """
        Create animated 3D phase space GIF.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path.
        fps : int, optional
            Frames per second. Default is 30.
        duration_seconds : int, optional
            Animation duration. Default is 12.
        dpi : int, optional
            Output resolution. Default is 100.
        """
        from erucakra.visualization.phase_space import create_phase_space_gif
        create_phase_space_gif(self, filepath, fps, duration_seconds, dpi)
    
    def to_png(
        self,
        filepath: str | Path,
        dpi: int = 200,
    ) -> None:
        """
        Create time series plot.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path.
        dpi : int, optional
            Output resolution. Default is 200.
        """
        from erucakra.visualization.timeseries import create_timeseries_plot
        create_timeseries_plot(self, filepath, dpi)
    
    def __repr__(self) -> str:
        name = self.scenario_info.get("name", "Custom") if self.scenario_info else "Custom"
        status = "TIPPED" if self.tipped else "NOT_TIPPED"
        return (
            f"SimulationResults(scenario='{name}', "
            f"years={self.year_int[0]}-{self.year_int[-1]}, "
            f"threshold_fraction={self.threshold_fraction:.2f}, "
            f"z_crit={self.z_crit:.3f}, max_z={self.max_z:.3f}, "
            f"status={status})"
        )
