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
        Calendar year array.
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
    
    @property
    def z_crit(self) -> float:
        """Effective critical threshold used in simulation."""
        return self.model_params.get("z_crit_effective", 1.0)
    
    @property
    def A_scale(self) -> float:
        """Forcing normalization scale used."""
        return self.model_params.get("A_scale", 1.0)
    
    @property
    def velocity(self) -> NDArray[np.float64]:
        """Phase space velocity magnitude sqrt(x² + y²)."""
        return np.sqrt(self.x**2 + self.y**2)
    
    @property
    def regime(self) -> NDArray[np.str_]:
        """Climate regime classification based on z vs z_crit."""
        return np.where(self.z > self.z_crit, "tipped", "stable")
    
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
        """Whether the tipping threshold was crossed."""
        return bool(np.any(self.z > self.z_crit))
    
    @property
    def first_crossing_year(self) -> Optional[float]:
        """Year when threshold was first crossed (None if never crossed)."""
        return self.diagnostics.get("first_crossing_year")
    
    @property
    def real_year(self) -> NDArray[np.int64]:
        """Real calendar year as integer."""
        return np.round(self.year).astype(np.int64)
    
    @property
    def lyapunov_exponent(self) -> float:
        """Estimated largest Lyapunov exponent (chaos indicator)."""
        return self.diagnostics.get("lyapunov_exponent", 0.0)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "scenario": self.scenario_key,
            "name": self.scenario_info.get("name", "Custom") if self.scenario_info else "Custom",
            "expected_outcome": self.scenario_info.get("expected_outcome", "N/A") if self.scenario_info else "N/A",
            "z_crit": self.z_crit,
            "A_scale": self.A_scale,
            "max_z": self.max_z,
            "final_z": float(self.z[-1]),
            "max_A": float(np.max(self.A)),
            "max_A_normalized": float(np.max(self.A_normalized)),
            "time_above_threshold_pct": self.time_above_threshold,
            "crossed_threshold": self.crossed_threshold,
            "first_crossing_year": self.first_crossing_year,
            "max_variability": self.diagnostics.get("max_variability", float(np.max(np.abs(self.x)))),
            "lyapunov_exponent": self.lyapunov_exponent,
            "year_start": int(np.round(self.year[0])),
            "year_end": int(np.round(self.year[-1])),
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
        
        # Upward crossings (stable → tipped)
        up_indices = np.where(transitions > 0)[0]
        for idx in up_indices:
            events.append({
                "year": float(self.year[idx]),
                "direction": "up",
                "z_before": float(self.z[idx]),
                "z_after": float(self.z[idx + 1]),
                "type": "tipping",
            })
        
        # Downward crossings (tipped → stable)
        down_indices = np.where(transitions < 0)[0]
        for idx in down_indices:
            events.append({
                "year": float(self.year[idx]),
                "direction": "down",
                "z_before": float(self.z[idx]),
                "z_after": float(self.z[idx + 1]),
                "type": "recovery",
            })
        
        # Sort by year
        events.sort(key=lambda e: e["year"])
        
        return events
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd
        
        return pd.DataFrame({
            "year": self.real_year,
            "time_normalized": self.t,
            "x_variability": self.x,
            "y_momentum": self.y,
            "z_accumulated": self.z,
            "A_forcing_Wm2": self.A,
            "A_normalized": self.A_normalized,
            "warming_proxy_celsius": self.warming,
            "distance_to_threshold": self.distance_to_threshold,
            "phase_velocity": self.velocity,
            "regime": self.regime,
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
        status = "TIPPED" if self.crossed_threshold else "STABLE"
        return (
            f"SimulationResults(scenario='{name}', "
            f"years={int(np.round(self.year[0]))}-{int(np.round(self.year[-1]))}, "
            f"z_crit={self.z_crit:.2f}, max_z={self.max_z:.3f}, "
            f"status={status})"
        )
