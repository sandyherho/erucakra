"""
Main climate model class for running simulations.
"""

from typing import Callable, Optional, Dict, Any, Union
import logging
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from tqdm import tqdm

from erucakra.core.dynamics import climate_tipping_model, add_climate_noise
from erucakra.core.results import SimulationResults
from erucakra.scenarios import SCENARIOS, get_scenario


logger = logging.getLogger(__name__)


class ClimateModel:
    """
    Climate Tipping Point Model.
    
    A three-variable dynamical system modeling climate tipping points
    with time-dependent radiative forcing.
    
    Parameters
    ----------
    c : float, optional
        Damping coefficient. Default is 0.2.
    epsilon : float, optional
        Timescale separation ratio. Default is 0.02.
    beta : float, optional
        Feedback strength. Default is 0.8.
    z_crit : float, optional
        Critical threshold for tipping. Default is 1.0.
    
    Attributes
    ----------
    params : dict
        Model parameters.
    
    Examples
    --------
    >>> model = ClimateModel()
    >>> results = model.run(scenario="ssp245")
    >>> results.to_csv("output.csv")
    """
    
    def __init__(
        self,
        c: float = 0.2,
        epsilon: float = 0.02,
        beta: float = 0.8,
        z_crit: float = 1.0,
    ):
        self.params = {
            "c": c,
            "epsilon": epsilon,
            "beta": beta,
            "z_crit": z_crit,
        }
        logger.info(f"Initialized ClimateModel with params: {self.params}")
    
    def run(
        self,
        scenario: Optional[str] = None,
        forcing: Optional[Union[Callable[[float], float], NDArray]] = None,
        forcing_times: Optional[NDArray] = None,
        t_start: float = 0.0,
        t_end: float = 150.0,
        n_points: int = 12000,
        initial_state: tuple = (0.05, 0.0, 0.3),
        add_noise: bool = True,
        noise_level: float = 0.03,
        noise_smoothing: float = 15.0,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        method: str = "RK45",
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> SimulationResults:
        """
        Run the climate tipping point simulation.
        
        Parameters
        ----------
        scenario : str, optional
            Built-in scenario name (e.g., "ssp126", "ssp245", "ssp370",
            "ssp585", "overshoot"). If None, must provide forcing.
        forcing : Callable or NDArray, optional
            Custom forcing function f(t) -> W/m² or array of forcing values.
            If array, must also provide forcing_times.
        forcing_times : NDArray, optional
            Time points for forcing array (years from 1750).
        t_start : float, optional
            Start time (normalized). Default is 0.0.
        t_end : float, optional
            End time (normalized). Default is 150.0.
        n_points : int, optional
            Number of output points. Default is 12000.
        initial_state : tuple, optional
            Initial (x, y, z) state. Default is (0.05, 0.0, 0.3).
        add_noise : bool, optional
            Whether to add climate noise. Default is True.
        noise_level : float, optional
            Noise amplitude. Default is 0.03.
        noise_smoothing : float, optional
            Noise smoothing sigma. Default is 15.0.
        rtol : float, optional
            Relative tolerance for ODE solver. Default is 1e-10.
        atol : float, optional
            Absolute tolerance for ODE solver. Default is 1e-12.
        method : str, optional
            ODE solver method. Default is "RK45".
        seed : int, optional
            Random seed for noise.
        show_progress : bool, optional
            Show progress bar. Default is True.
        
        Returns
        -------
        SimulationResults
            Container with simulation results and export methods.
        
        Raises
        ------
        ValueError
            If neither scenario nor forcing is provided.
        """
        logger.info("Starting simulation run")
        logger.debug(f"Parameters: scenario={scenario}, t_end={t_end}, n_points={n_points}")
        
        # Get forcing function
        scenario_info = None
        if scenario is not None:
            scenario_info = get_scenario(scenario)
            A_func = scenario_info["forcing_func"]
            logger.info(f"Using scenario: {scenario_info['name']}")
        elif forcing is not None:
            if callable(forcing):
                A_func = forcing
            else:
                # Interpolate from array
                if forcing_times is None:
                    raise ValueError("forcing_times required when forcing is an array")
                A_func = self._create_forcing_interpolator(forcing_times, forcing)
            logger.info("Using custom forcing")
        else:
            raise ValueError("Must provide either scenario or forcing")
        
        # Time evaluation points
        t_eval = np.linspace(t_start, t_end, n_points)
        
        # Progress tracking
        if show_progress:
            pbar = tqdm(total=100, desc="Integrating ODE", unit="%", leave=True)
            last_progress = [0]
            
            def progress_callback(t, y):
                progress = int((t - t_start) / (t_end - t_start) * 100)
                if progress > last_progress[0]:
                    pbar.update(progress - last_progress[0])
                    last_progress[0] = progress
        
        logger.debug("Starting ODE integration")
        
        # Solve ODE system
        solution = solve_ivp(
            climate_tipping_model,
            t_span=(t_start, t_end),
            y0=list(initial_state),
            args=(self.params["c"], self.params["epsilon"], A_func,
                  self.params["beta"], self.params["z_crit"]),
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        
        if show_progress:
            pbar.update(100 - last_progress[0])
            pbar.close()
        
        if not solution.success:
            logger.error(f"ODE integration failed: {solution.message}")
            raise RuntimeError(f"ODE integration failed: {solution.message}")
        
        logger.info(f"ODE integration successful: {len(solution.t)} points")
        
        # Extract solution
        t = solution.t
        y = solution.y
        
        # Add climate noise
        if add_noise:
            logger.debug(f"Adding climate noise: level={noise_level}, smoothing={noise_smoothing}")
            y = add_climate_noise(t, y, noise_level, noise_smoothing, seed)
        
        # Compute forcing profile
        logger.debug("Computing forcing profile")
        A_profile = np.array([A_func(ti) for ti in tqdm(t, desc="Computing forcing", 
                                                         disable=not show_progress)])
        
        # Calendar year (2020 baseline, t normalized)
        year = 2020 + t * 0.8
        
        # Warming proxy
        warming = y[2] * 1.5
        
        # Build results
        results = SimulationResults(
            t=t,
            year=year,
            x=y[0],
            y=y[1],
            z=y[2],
            A=A_profile,
            warming=warming,
            scenario_key=scenario,
            scenario_info=scenario_info,
            model_params=self.params.copy(),
            simulation_params={
                "t_start": t_start,
                "t_end": t_end,
                "n_points": n_points,
                "initial_state": initial_state,
                "add_noise": add_noise,
                "noise_level": noise_level,
                "rtol": rtol,
                "atol": atol,
                "method": method,
            },
        )
        
        logger.info("Simulation complete")
        return results
    
    def _create_forcing_interpolator(
        self,
        times: NDArray,
        values: NDArray,
    ) -> Callable[[float], float]:
        """Create interpolating function from forcing data."""
        from scipy.interpolate import interp1d
        
        # Convert years from 1750 to normalized time
        # t_norm = (year - 2020) / 0.8
        t_norm = (times - 2020) / 0.8
        
        interpolator = interp1d(
            t_norm, values,
            kind="linear",
            bounds_error=False,
            fill_value=(values[0], values[-1]),
        )
        
        return lambda t: float(interpolator(t))
    
    def __repr__(self) -> str:
        return f"ClimateModel(c={self.params['c']}, epsilon={self.params['epsilon']})"
