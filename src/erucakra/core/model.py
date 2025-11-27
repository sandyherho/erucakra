"""
Main climate model class for running simulations.
"""

from typing import Callable, Optional, Dict, Any, Union
import logging
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from tqdm import tqdm

from erucakra.core.dynamics import (
    climate_tipping_model,
    add_climate_noise,
    compute_fixed_points,
    compute_lyapunov_exponent,
    FORCING_SCALES,
    DEFAULT_Z_CRIT,
)
from erucakra.core.results import SimulationResults
from erucakra.scenarios import SCENARIOS, get_scenario


logger = logging.getLogger(__name__)


class ClimateModel:
    """
    Climate Tipping Point Model.
    
    A three-variable dynamical system modeling climate tipping points
    with time-dependent radiative forcing from SSP scenarios.
    
    The model exhibits a pitchfork bifurcation when the slow variable z
    crosses the critical threshold z_crit, transitioning from a stable
    single-well potential to a bistable double-well configuration.
    
    Parameters
    ----------
    c : float, optional
        Damping coefficient (energy dissipation rate). 
        Higher values = faster decay of oscillations.
        Typical range: 0.1-0.5. Default is 0.2.
    epsilon : float, optional
        Timescale separation ratio (slow/fast).
        Smaller values = stronger separation, slower z evolution.
        Typical range: 0.01-0.05. Default is 0.02.
    beta : float, optional
        Feedback strength (variability → accumulation coupling).
        Higher values = stronger negative feedback from x oscillations.
        Typical range: 0.5-1.5. Default is 0.8.
    z_crit : float, optional
        Critical threshold for tipping. If None, uses scenario-specific
        defaults that produce expected behavior (STABLE, MARGINAL, etc.).
        Default is None (auto-select based on scenario).
    
    Attributes
    ----------
    params : dict
        Model parameters.
    
    Examples
    --------
    >>> # Use default thresholds (scenario-specific)
    >>> model = ClimateModel()
    >>> results = model.run(scenario="ssp245")
    
    >>> # Override threshold for sensitivity analysis
    >>> model = ClimateModel(z_crit=0.75)
    >>> results = model.run(scenario="ssp245")
    
    >>> # Run all scenarios with same threshold
    >>> model = ClimateModel(z_crit=0.80)
    >>> for scenario in ["ssp126", "ssp245", "ssp370", "ssp585"]:
    ...     results = model.run(scenario=scenario)
    """
    
    def __init__(
        self,
        c: float = 0.2,
        epsilon: float = 0.02,
        beta: float = 0.8,
        z_crit: Optional[float] = None,
    ):
        self.params = {
            "c": c,
            "epsilon": epsilon,
            "beta": beta,
            "z_crit": z_crit,  # None means auto-select
        }
        self._validate_params()
        logger.info(f"Initialized ClimateModel with params: {self.params}")
    
    def _validate_params(self) -> None:
        """Validate model parameters are in physically reasonable ranges."""
        c = self.params["c"]
        epsilon = self.params["epsilon"]
        beta = self.params["beta"]
        z_crit = self.params["z_crit"]
        
        if not 0 < c < 2:
            logger.warning(f"Damping c={c} outside typical range (0, 2)")
        
        if not 0 < epsilon < 0.2:
            logger.warning(f"Timescale epsilon={epsilon} outside typical range (0, 0.2)")
        
        if not 0 < beta < 3:
            logger.warning(f"Feedback beta={beta} outside typical range (0, 3)")
        
        if z_crit is not None and z_crit <= 0:
            raise ValueError(f"z_crit must be positive, got {z_crit}")
    
    def get_z_crit(self, scenario_key: Optional[str] = None) -> float:
        """
        Get effective z_crit value.
        
        If z_crit was specified at initialization, uses that value.
        Otherwise, returns scenario-specific default.
        
        Parameters
        ----------
        scenario_key : str, optional
            Scenario identifier for default lookup.
        
        Returns
        -------
        float
            Effective critical threshold.
        """
        if self.params["z_crit"] is not None:
            return self.params["z_crit"]
        
        if scenario_key is not None and scenario_key in DEFAULT_Z_CRIT:
            return DEFAULT_Z_CRIT[scenario_key]
        
        return DEFAULT_Z_CRIT.get("custom", 0.80)
    
    def get_A_scale(self, scenario_key: Optional[str] = None) -> float:
        """
        Get forcing scale for normalization.
        
        Parameters
        ----------
        scenario_key : str, optional
            Scenario identifier.
        
        Returns
        -------
        float
            Forcing scale (W/m²).
        """
        if scenario_key is not None and scenario_key in FORCING_SCALES:
            return FORCING_SCALES[scenario_key]
        return FORCING_SCALES.get("custom", 5.0)
    
    def run(
        self,
        scenario: Optional[str] = None,
        forcing: Optional[Union[Callable[[float], float], NDArray]] = None,
        forcing_times: Optional[NDArray] = None,
        t_start: float = 0.0,
        t_end: float = 600.0,
        n_points: int = 48000,
        initial_state: tuple = (0.05, 0.0, 0.3),
        add_noise: bool = True,
        noise_level: float = 0.03,
        noise_smoothing: float = 15.0,
        noise_color: str = "red",
        rtol: float = 1e-10,
        atol: float = 1e-12,
        method: str = "RK45",
        seed: Optional[int] = None,
        show_progress: bool = True,
        z_crit_override: Optional[float] = None,
        A_scale_override: Optional[float] = None,
    ) -> SimulationResults:
        """
        Run the climate tipping point simulation.
        
        Parameters
        ----------
        scenario : str, optional
            Built-in scenario name ("ssp126", "ssp245", "ssp370", "ssp585").
            If None, must provide forcing.
        forcing : Callable or NDArray, optional
            Custom forcing function f(t) → W/m² or array of forcing values.
            If array, must also provide forcing_times.
        forcing_times : NDArray, optional
            Time points for forcing array (years from 1750).
        t_start : float, optional
            Start time (normalized). Default is 0.0 (year 2020).
        t_end : float, optional
            End time (normalized). Default is 600.0 (year 2500).
        n_points : int, optional
            Number of output points. Default is 48000.
        initial_state : tuple, optional
            Initial (x, y, z) state. Default is (0.05, 0.0, 0.3).
        add_noise : bool, optional
            Whether to add climate noise. Default is True.
        noise_level : float, optional
            Noise amplitude (std dev). Default is 0.03.
        noise_smoothing : float, optional
            Noise autocorrelation timescale. Default is 15.0.
        noise_color : str, optional
            Noise spectrum: "red", "pink", "white". Default is "red".
        rtol : float, optional
            Relative tolerance for ODE solver. Default is 1e-10.
        atol : float, optional
            Absolute tolerance for ODE solver. Default is 1e-12.
        method : str, optional
            ODE solver method. Default is "RK45".
        seed : int, optional
            Random seed for reproducible noise.
        show_progress : bool, optional
            Show progress bar. Default is True.
        z_crit_override : float, optional
            Override z_crit for this run only.
        A_scale_override : float, optional
            Override forcing scale for this run only.
        
        Returns
        -------
        SimulationResults
            Container with simulation results and export methods.
        
        Raises
        ------
        ValueError
            If neither scenario nor forcing is provided.
        RuntimeError
            If ODE integration fails.
        """
        logger.info("Starting simulation run")
        logger.debug(f"Parameters: scenario={scenario}, t_end={t_end}, n_points={n_points}")
        
        # Determine scenario key for defaults
        scenario_key = scenario if scenario else "custom"
        
        # Get effective z_crit and A_scale
        if z_crit_override is not None:
            z_crit = z_crit_override
        else:
            z_crit = self.get_z_crit(scenario_key)
        
        if A_scale_override is not None:
            A_scale = A_scale_override
        else:
            A_scale = self.get_A_scale(scenario_key)
        
        logger.info(f"Using z_crit={z_crit:.3f}, A_scale={A_scale:.2f} W/m²")
        
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
            
            def update_progress(t):
                progress = int((t - t_start) / (t_end - t_start) * 100)
                if progress > last_progress[0]:
                    pbar.update(progress - last_progress[0])
                    last_progress[0] = progress
        
        logger.debug("Starting ODE integration")
        
        # Wrapper to track progress during dense output
        def rhs(t, y):
            if show_progress:
                update_progress(t)
            return climate_tipping_model(
                t, y,
                self.params["c"],
                self.params["epsilon"],
                A_func,
                self.params["beta"],
                z_crit,
                A_scale,
            )
        
        # Solve ODE system
        solution = solve_ivp(
            rhs,
            t_span=(t_start, t_end),
            y0=list(initial_state),
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
            logger.debug(f"Adding {noise_color} noise: level={noise_level}")
            y = add_climate_noise(
                t, y, noise_level, noise_smoothing, seed, noise_color
            )
        
        # Compute forcing profile
        logger.debug("Computing forcing profile")
        A_profile = np.array([
            A_func(ti) for ti in tqdm(
                t, desc="Computing forcing", disable=not show_progress
            )
        ])
        
        # Normalized forcing (what z sees)
        A_normalized = A_profile / A_scale
        
        # Calendar year (2020 baseline)
        year = 2020 + t * 0.8
        
        # Warming proxy: scale z to approximate temperature anomaly
        # z=1 roughly corresponds to 1.5°C (Paris threshold)
        warming = y[2] * 1.5
        
        # Distance from threshold
        distance_to_threshold = y[2] - z_crit
        
        # Compute diagnostics
        logger.debug("Computing diagnostics")
        
        # Time above threshold
        above_threshold = y[2] > z_crit
        time_above_pct = np.sum(above_threshold) / len(t) * 100
        
        # First crossing time
        crossing_indices = np.where(np.diff(above_threshold.astype(int)) > 0)[0]
        first_crossing_year = None
        if len(crossing_indices) > 0:
            first_crossing_year = float(year[crossing_indices[0]])
        
        # Lyapunov exponent (chaos indicator)
        dt = (t_end - t_start) / n_points
        lyapunov = compute_lyapunov_exponent(y, dt)
        
        # Build results
        results = SimulationResults(
            t=t,
            year=year,
            x=y[0],
            y=y[1],
            z=y[2],
            A=A_profile,
            A_normalized=A_normalized,
            warming=warming,
            distance_to_threshold=distance_to_threshold,
            scenario_key=scenario,
            scenario_info=scenario_info,
            model_params={
                **self.params,
                "z_crit_effective": z_crit,
                "A_scale": A_scale,
            },
            simulation_params={
                "t_start": t_start,
                "t_end": t_end,
                "n_points": n_points,
                "initial_state": initial_state,
                "add_noise": add_noise,
                "noise_level": noise_level,
                "noise_color": noise_color,
                "rtol": rtol,
                "atol": atol,
                "method": method,
            },
            diagnostics={
                "time_above_threshold_pct": time_above_pct,
                "first_crossing_year": first_crossing_year,
                "lyapunov_exponent": lyapunov,
                "max_z": float(np.max(y[2])),
                "final_z": float(y[2, -1]),
                "max_variability": float(np.max(np.abs(y[0]))),
            },
        )
        
        logger.info(f"Simulation complete. Max z={results.diagnostics['max_z']:.3f}, "
                   f"crossed={results.crossed_threshold}")
        
        return results
    
    def _create_forcing_interpolator(
        self,
        times: NDArray,
        values: NDArray,
    ) -> Callable[[float], float]:
        """Create interpolating function from forcing data."""
        from scipy.interpolate import interp1d
        
        # Convert years from 1750 to normalized time
        t_norm = (times - 2020) / 0.8
        
        interpolator = interp1d(
            t_norm, values,
            kind="linear",
            bounds_error=False,
            fill_value=(values[0], values[-1]),
        )
        
        return lambda t: float(interpolator(t))
    
    def sensitivity_analysis(
        self,
        scenario: str,
        z_crit_range: tuple = (0.5, 1.2),
        n_samples: int = 10,
        **run_kwargs,
    ) -> list:
        """
        Run sensitivity analysis over z_crit values.
        
        Parameters
        ----------
        scenario : str
            Scenario to analyze.
        z_crit_range : tuple, optional
            (min, max) range for z_crit. Default is (0.5, 1.2).
        n_samples : int, optional
            Number of z_crit values to test. Default is 10.
        **run_kwargs
            Additional arguments passed to run().
        
        Returns
        -------
        list
            List of (z_crit, results) tuples.
        """
        z_crit_values = np.linspace(z_crit_range[0], z_crit_range[1], n_samples)
        results_list = []
        
        for z_crit in tqdm(z_crit_values, desc="Sensitivity analysis"):
            results = self.run(
                scenario=scenario,
                z_crit_override=z_crit,
                show_progress=False,
                **run_kwargs,
            )
            results_list.append((z_crit, results))
        
        return results_list
    
    def __repr__(self) -> str:
        z_crit_str = f"{self.params['z_crit']}" if self.params['z_crit'] else "auto"
        return (f"ClimateModel(c={self.params['c']}, epsilon={self.params['epsilon']}, "
                f"beta={self.params['beta']}, z_crit={z_crit_str})")
