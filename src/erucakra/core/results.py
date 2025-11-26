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
        External radiative forcing.
    warming : NDArray
        Warming proxy in °C.
    scenario_key : str, optional
        Scenario identifier.
    scenario_info : dict, optional
        Scenario metadata.
    model_params : dict
        Model parameters used.
    simulation_params : dict
        Simulation parameters used.
    """
    
    t: NDArray[np.float64]
    year: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    A: NDArray[np.float64]
    warming: NDArray[np.float64]
    scenario_key: Optional[str] = None
    scenario_info: Optional[Dict[str, Any]] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    simulation_params: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def velocity(self) -> NDArray[np.float64]:
        """Phase space velocity magnitude."""
        return np.sqrt(self.x**2 + self.y**2)
    
    @property
    def regime(self) -> NDArray[np.str_]:
        """Climate regime classification."""
        return np.where(self.z > 1, "tipped", "stable")
    
    @property
    def distance_to_threshold(self) -> NDArray[np.float64]:
        """Distance from tipping threshold (z - 1)."""
        return self.z - 1.0
    
    @property
    def max_z(self) -> float:
        """Maximum accumulated forcing."""
        return float(np.max(self.z))
    
    @property
    def time_above_threshold(self) -> float:
        """Percentage of time above tipping threshold."""
        return float(np.sum(self.z > 1) / len(self.z) * 100)
    
    @property
    def crossed_threshold(self) -> bool:
        """Whether the tipping threshold was crossed."""
        return bool(np.any(self.z > 1))
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "scenario": self.scenario_key,
            "name": self.scenario_info.get("name", "Custom") if self.scenario_info else "Custom",
            "max_z": self.max_z,
            "final_z": float(self.z[-1]),
            "time_above_threshold_pct": self.time_above_threshold,
            "crossed_threshold": self.crossed_threshold,
            "max_variability": float(np.max(np.abs(self.x))),
            "year_start": float(self.year[0]),
            "year_end": float(self.year[-1]),
            "n_points": len(self.t),
        }
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd
        
        return pd.DataFrame({
            "time_normalized": self.t,
            "year": self.year,
            "x_variability": self.x,
            "y_momentum": self.y,
            "z_accumulated": self.z,
            "A_forcing": self.A,
            "warming_proxy_celsius": self.warming,
            "phase_velocity": self.velocity,
            "distance_to_threshold": self.distance_to_threshold,
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
            Include metadata header. Default is True.
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
        return (
            f"SimulationResults(scenario='{name}', "
            f"years={self.year[0]:.0f}-{self.year[-1]:.0f}, "
            f"n_points={len(self.t)}, "
            f"max_z={self.max_z:.3f})"
        )
