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
    FORCING_SCALES,
    GLOBAL_A_SCALE,
    DEFAULT_Z_CRIT_ABSOLUTE,
)
from erucakra.core.results import SimulationResults
from erucakra.scenarios import SCENARIOS, get_scenario


logger = logging.getLogger(__name__)


# Import logging utilities - these are safe, no circular import
def _get_logging_utils():
    """Lazy import to avoid circular imports."""
    try:
        from erucakra.utils.logging import (
            get_timing_logger,
            start_step,
            end_step,
            log_error,
            log_calculation_issue,
        )
        return start_step, end_step, log_error, log_calculation_issue
    except ImportError:
        # Fallback if logging not available
        def noop(*args, **kwargs):
            pass
        def noop_end(*args, **kwargs):
            return 0.0
        return noop, noop_end, noop, noop


class ClimateModel:
    """
    Climate Tipping Point Model with absolute z_crit threshold.
    
    A three-variable dynamical system modeling climate tipping points
    with time-dependent radiative forcing from SSP scenarios.
    
    The model exhibits a pitchfork bifurcation when the slow variable z
    crosses the critical threshold z_crit, transitioning from a stable
    single-well potential to a bistable double-well configuration.
    
    Key Design Principle (UPDATED):
        z_crit is now an ABSOLUTE threshold (default 0.55) applied consistently
        across all scenarios. Combined with global A_scale normalization (13 W/m²),
        this ensures:
        - SSP1-2.6: STABLE (z never reaches threshold)
        - SSP2-4.5: MARGINAL (z hovers near threshold)
        - SSP3-7.0: TIPPING (z crosses threshold)
        - SSP5-8.5: CATASTROPHIC (z far exceeds threshold)
    
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
        Critical threshold for tipping (absolute value).
        Default is 0.55 (calibrated for global A_scale = 13 W/m²).
    threshold_fraction : float, optional
        DEPRECATED. Kept for backward compatibility but ignored when
        z_crit is set to absolute value. Default is None.
    use_absolute_threshold : bool, optional
        If True (default), use absolute z_crit. If False, compute from
        threshold_fraction (legacy behavior, not recommended).
    
    Attributes
    ----------
    params : dict
        Model parameters.
    z_crit_absolute : float
        Absolute threshold value.
    """
    
    def __init__(
        self,
        c: float = 0.2,
        epsilon: float = 0.02,
        beta: float = 0.8,
        z_crit: Optional[float] = None,
        threshold_fraction: Optional[float] = None,
        use_absolute_threshold: bool = True,
    ):
        # Determine z_crit value
        if z_crit is not None:
            # Explicit z_crit provided
            effective_z_crit = z_crit
        elif use_absolute_threshold:
            # Use default absolute threshold
            effective_z_crit = DEFAULT_Z_CRIT_ABSOLUTE
        else:
            # Legacy mode: will compute from threshold_fraction later
            effective_z_crit = None
        
        self.params = {
            "c": c,
            "epsilon": epsilon,
            "beta": beta,
            "z_crit": effective_z_crit,
        }
        
        # Store threshold_fraction for legacy compatibility
        self.threshold_fraction = threshold_fraction if threshold_fraction is not None else 0.7
        self.use_absolute_threshold = use_absolute_threshold
        self.z_crit_absolute = effective_z_crit
        
        self._validate_params()
        
        mode = "absolute" if use_absolute_threshold else "threshold_fraction"
        logger.info(f"Initialized ClimateModel with params: {self.params}, "
                   f"mode={mode}, z_crit={effective_z_crit}")
    
    def _validate_params(self) -> None:
        """Validate model parameters are in physically reasonable ranges."""
        start_step, end_step, log_error, log_calculation_issue = _get_logging_utils()
        
        c = self.params["c"]
        epsilon = self.params["epsilon"]
        beta = self.params["beta"]
        z_crit = self.params["z_crit"]
        
        issues = []
        
        if not 0 < c < 2:
            msg = f"Damping c={c} outside typical range (0, 2)"
            logger.warning(msg)
            issues.append(msg)
        
        if not 0 < epsilon < 0.2:
            msg = f"Timescale epsilon={epsilon} outside typical range (0, 0.2)"
            logger.warning(msg)
            issues.append(msg)
        
        if not 0 < beta < 3:
            msg = f"Feedback beta={beta} outside typical range (0, 3)"
            logger.warning(msg)
            issues.append(msg)
        
        if z_crit is not None and z_crit <= 0:
            raise ValueError(f"z_crit must be positive, got {z_crit}")
        
        if issues:
            log_calculation_issue(
                "Parameter validation",
                "Some parameters outside typical ranges",
                {"c": c, "epsilon": epsilon, "beta": beta, "z_crit": z_crit}
            )
    
    def get_A_scale(self, scenario_key: Optional[str] = None) -> float:
        """
        Get forcing scale for normalization.
        
        UPDATED: Now returns GLOBAL_A_SCALE for all scenarios.
        """
        # Always use global scale for consistent normalization
        return GLOBAL_A_SCALE
    
    def compute_z_crit_from_fraction(
        self,
        A_func: Callable[[float], float],
        A_scale: float,
        t_start: float,
        t_end: float,
        n_sample: int = 1000,
    ) -> tuple:
        """
        Compute z_crit from forcing data using threshold_fraction.
        
        DEPRECATED: This method is kept for backward compatibility but
        should not be used. Use absolute z_crit instead.
        
        Parameters
        ----------
        A_func : Callable
            Forcing function A(t).
        A_scale : float
            Forcing normalization scale.
        t_start : float
            Start time.
        t_end : float
            End time.
        n_sample : int
            Number of points to sample forcing.
        
        Returns
        -------
        tuple
            (z_crit, forcing_analysis_dict)
        """
        logger.warning("Using deprecated threshold_fraction method for z_crit computation")
        
        # Sample forcing over the simulation period
        t_sample = np.linspace(t_start, t_end, n_sample)
        A_sample = np.array([A_func(t) for t in t_sample])
        
        # Compute normalized forcing
        A_normalized = A_sample / A_scale
        A_normalized_max = np.max(A_normalized)
        A_normalized_min = np.min(A_normalized)
        
        # Compute z_crit as fraction of max normalized forcing
        z_crit = self.threshold_fraction * A_normalized_max
        
        # Store forcing analysis for metadata
        forcing_analysis = {
            "A_max": float(np.max(A_sample)),
            "A_min": float(np.min(A_sample)),
            "A_mean": float(np.mean(A_sample)),
            "A_scale": A_scale,
            "A_normalized_max": float(A_normalized_max),
            "A_normalized_min": float(A_normalized_min),
            "threshold_fraction": self.threshold_fraction,
            "z_crit_computed": float(z_crit),
            "method": "threshold_fraction (DEPRECATED)",
        }
        
        logger.info(f"Computed z_crit = {self.threshold_fraction:.2f} × {A_normalized_max:.3f} = {z_crit:.3f}")
        
        return z_crit, forcing_analysis
    
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
        threshold_fraction_override: Optional[float] = None,
    ) -> SimulationResults:
        """
        Run the climate tipping point simulation.
        
        UPDATED: Now uses absolute z_crit with global A_scale by default.
        
        Parameters
        ----------
        scenario : str, optional
            Built-in scenario key (ssp126, ssp245, ssp370, ssp585).
        forcing : Callable or NDArray, optional
            Custom forcing function or array.
        forcing_times : NDArray, optional
            Times for custom forcing array.
        t_start : float
            Start time (normalized). Default 0.0.
        t_end : float
            End time (normalized). Default 600.0.
        n_points : int
            Number of output points. Default 48000.
        initial_state : tuple
            Initial (x, y, z) state. Default (0.05, 0.0, 0.3).
        add_noise : bool
            Add climate noise. Default True.
        noise_level : float
            Noise amplitude. Default 0.03.
        noise_smoothing : float
            Noise smoothing sigma. Default 15.0.
        noise_color : str
            Noise type: 'red', 'white', 'pink'. Default 'red'.
        rtol : float
            Relative tolerance. Default 1e-10.
        atol : float
            Absolute tolerance. Default 1e-12.
        method : str
            ODE solver method. Default 'RK45'.
        seed : int, optional
            Random seed for reproducibility.
        show_progress : bool
            Show progress bars. Default True.
        z_crit_override : float, optional
            Override z_crit with this value.
        A_scale_override : float, optional
            Override A_scale (not recommended).
        threshold_fraction_override : float, optional
            DEPRECATED. Use z_crit_override instead.
        
        Returns
        -------
        SimulationResults
            Container with all simulation results and metadata.
        """
        # Get logging utilities
        start_step, end_step, log_error, log_calculation_issue = _get_logging_utils()
        
        # =====================================================================
        # STEP 1: Setup and Parameter Resolution
        # =====================================================================
        start_step("Setup and parameter resolution")
        
        try:
            logger.info("Starting simulation run")
            logger.debug(f"Parameters: scenario={scenario}, t_end={t_end}, n_points={n_points}")
            
            # Determine scenario key for defaults
            scenario_key = scenario if scenario else "custom"
            
            # Get forcing function first
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
            
            # Get A_scale (always global now)
            if A_scale_override is not None:
                A_scale = A_scale_override
                logger.warning(f"Using A_scale override: {A_scale}")
            else:
                A_scale = self.get_A_scale(scenario_key)
            
            # Determine z_crit
            forcing_analysis = None
            if z_crit_override is not None:
                z_crit = z_crit_override
                logger.info(f"Using z_crit override: {z_crit:.3f}")
            elif self.params["z_crit"] is not None:
                z_crit = self.params["z_crit"]
                logger.info(f"Using absolute z_crit: {z_crit:.3f}")
            elif threshold_fraction_override is not None:
                # Legacy mode with override
                logger.warning("Using deprecated threshold_fraction_override")
                self.threshold_fraction = threshold_fraction_override
                z_crit, forcing_analysis = self.compute_z_crit_from_fraction(
                    A_func, A_scale, t_start, t_end
                )
            else:
                # Fallback to default absolute threshold
                z_crit = DEFAULT_Z_CRIT_ABSOLUTE
                logger.info(f"Using default absolute z_crit: {z_crit:.3f}")
            
            # Compute forcing analysis for metadata
            if forcing_analysis is None:
                t_sample = np.linspace(t_start, t_end, 1000)
                A_sample = np.array([A_func(t) for t in t_sample])
                A_normalized = A_sample / A_scale
                forcing_analysis = {
                    "A_max": float(np.max(A_sample)),
                    "A_min": float(np.min(A_sample)),
                    "A_mean": float(np.mean(A_sample)),
                    "A_scale": A_scale,
                    "A_normalized_max": float(np.max(A_normalized)),
                    "A_normalized_min": float(np.min(A_normalized)),
                    "z_crit_used": float(z_crit),
                    "method": "absolute",
                }
            
            logger.info(f"Using z_crit={z_crit:.3f}, A_scale={A_scale:.2f} W/m²")
            logger.info(f"Normalized forcing will peak at: {forcing_analysis['A_normalized_max']:.3f}")
            
            # Predict outcome
            predicted_tip = forcing_analysis['A_normalized_max'] > z_crit
            logger.info(f"Predicted to tip: {predicted_tip} "
                       f"(max_A_norm={forcing_analysis['A_normalized_max']:.3f} vs z_crit={z_crit:.3f})")
            
            # Time evaluation points
            t_eval = np.linspace(t_start, t_end, n_points)
            
            end_step(success=True)
            
        except Exception as e:
            log_error(e, "Setup and parameter resolution")
            end_step(success=False)
            raise
        
        # =====================================================================
        # STEP 2: ODE Integration
        # =====================================================================
        start_step("ODE integration")
        
        try:
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
            
            # Track integration issues
            integration_warnings = []
            
            def rhs(t, y):
                if show_progress:
                    update_progress(t)
                
                # Check for NaN/Inf in current state
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    integration_warnings.append({
                        "t": t,
                        "y": y.copy(),
                        "issue": "NaN or Inf in state"
                    })
                
                result = climate_tipping_model(
                    t, y,
                    self.params["c"],
                    self.params["epsilon"],
                    A_func,
                    self.params["beta"],
                    z_crit,
                    A_scale,
                )
                
                # Check for NaN/Inf in derivatives
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    integration_warnings.append({
                        "t": t,
                        "y": y.copy(),
                        "dydt": result,
                        "issue": "NaN or Inf in derivatives"
                    })
                
                return result
            
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
            
            # Log any integration warnings
            if integration_warnings:
                log_calculation_issue(
                    "Integration warnings",
                    f"{len(integration_warnings)} issues during integration",
                    {"first_issue": integration_warnings[0]}
                )
            
            if not solution.success:
                logger.error(f"ODE integration failed: {solution.message}")
                raise RuntimeError(f"ODE integration failed: {solution.message}")
            
            logger.info(f"ODE integration successful: {len(solution.t)} points")
            
            # Extract solution
            t = solution.t
            y = solution.y
            
            # Validate solution
            self._validate_solution(t, y, "raw ODE solution", log_calculation_issue)
            
            end_step(success=True)
            
        except Exception as e:
            log_error(e, "ODE integration")
            if show_progress and 'pbar' in dir():
                pbar.close()
            end_step(success=False)
            raise
        
        # =====================================================================
        # STEP 3: Add Climate Noise
        # =====================================================================
        if add_noise:
            start_step("Adding climate noise")
            
            try:
                logger.debug(f"Adding {noise_color} noise: level={noise_level}")
                y = add_climate_noise(
                    t, y, noise_level, noise_smoothing, seed, noise_color
                )
                
                # Validate noisy solution
                self._validate_solution(t, y, "noisy solution", log_calculation_issue)
                
                end_step(success=True)
                
            except Exception as e:
                log_error(e, "Adding climate noise")
                end_step(success=False)
                raise
        
        # =====================================================================
        # STEP 4: Compute Forcing Profile
        # =====================================================================
        start_step("Computing forcing profile")
        
        try:
            logger.debug("Computing forcing profile")
            A_profile = np.array([
                A_func(ti) for ti in tqdm(
                    t, desc="Computing forcing", disable=not show_progress
                )
            ])
            
            # Validate forcing
            if np.any(np.isnan(A_profile)):
                nan_count = np.sum(np.isnan(A_profile))
                log_calculation_issue(
                    "NaN in forcing",
                    f"{nan_count} NaN values in forcing profile",
                    {"nan_count": int(nan_count), "total": len(A_profile)}
                )
            
            if np.any(np.isinf(A_profile)):
                inf_count = np.sum(np.isinf(A_profile))
                log_calculation_issue(
                    "Inf in forcing",
                    f"{inf_count} Inf values in forcing profile",
                    {"inf_count": int(inf_count), "total": len(A_profile)}
                )
            
            # Normalized forcing (what z sees)
            A_normalized = A_profile / A_scale
            
            logger.info(f"Forcing range: {np.nanmin(A_profile):.3f} to {np.nanmax(A_profile):.3f} W/m²")
            logger.info(f"Normalized forcing range: {np.nanmin(A_normalized):.3f} to {np.nanmax(A_normalized):.3f}")
            
            end_step(success=True)
            
        except Exception as e:
            log_error(e, "Computing forcing profile")
            end_step(success=False)
            raise
        
        # =====================================================================
        # STEP 5: Compute Derived Quantities
        # =====================================================================
        start_step("Computing derived quantities")
        
        try:
            # Calendar year (2020 baseline)
            year = 2020 + t * 0.8
            
            # Warming proxy: scale z to approximate temperature anomaly
            # z=1 roughly corresponds to 1.5°C (Paris threshold)
            warming = y[2] * 1.5
            
            # Distance from threshold
            distance_to_threshold = y[2] - z_crit
            
            logger.debug(f"Year range: {year[0]:.0f} to {year[-1]:.0f}")
            logger.debug(f"z range: {np.nanmin(y[2]):.4f} to {np.nanmax(y[2]):.4f}")
            logger.debug(f"Warming range: {np.nanmin(warming):.4f} to {np.nanmax(warming):.4f} °C")
            
            end_step(success=True)
            
        except Exception as e:
            log_error(e, "Computing derived quantities")
            end_step(success=False)
            raise
        
        # =====================================================================
        # STEP 6: Compute Diagnostics
        # =====================================================================
        start_step("Computing diagnostics")
        
        try:
            logger.debug("Computing diagnostics")
            
            # Time above threshold
            above_threshold = y[2] > z_crit
            time_above_pct = np.sum(above_threshold) / len(t) * 100
            
            # First crossing time
            crossing_indices = np.where(np.diff(above_threshold.astype(int)) > 0)[0]
            first_crossing_year = None
            if len(crossing_indices) > 0:
                first_crossing_year = float(year[crossing_indices[0]])
            
            # Check for extreme values
            max_z = float(np.nanmax(y[2]))
            final_z = float(y[2, -1])
            max_variability = float(np.nanmax(np.abs(y[0])))
            
            # Binary tipped status
            tipped = bool(np.any(above_threshold))
            
            if max_z > 10:
                log_calculation_issue(
                    "Extreme z value",
                    f"max_z={max_z:.3f} is unusually high",
                    {"max_z": max_z, "z_crit": z_crit}
                )
            
            if max_variability > 10:
                log_calculation_issue(
                    "Extreme variability",
                    f"max |x|={max_variability:.3f} is unusually high",
                    {"max_variability": max_variability}
                )
            
            logger.info(f"Time above threshold: {time_above_pct:.1f}%")
            logger.info(f"Max z: {max_z:.4f}, z_crit: {z_crit:.4f}")
            logger.info(f"Tipped: {tipped}")
            
            if first_crossing_year:
                logger.info(f"First threshold crossing: Year {int(first_crossing_year)}")
            
            end_step(success=True)
            
        except Exception as e:
            log_error(e, "Computing diagnostics")
            end_step(success=False)
            raise
        
        # =====================================================================
        # STEP 7: Build Results Object
        # =====================================================================
        start_step("Building results object")
        
        try:
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
                    "threshold_mode": "absolute" if self.use_absolute_threshold else "fraction",
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
                    "max_z": max_z,
                    "final_z": final_z,
                    "max_variability": max_variability,
                    "tipped": tipped,
                },
                forcing_analysis=forcing_analysis,
            )
            
            logger.info(f"Simulation complete. Max z={results.diagnostics['max_z']:.3f}, "
                       f"tipped={results.tipped}")
            
            end_step(success=True)
            
        except Exception as e:
            log_error(e, "Building results object")
            end_step(success=False)
            raise
        
        return results
    
    def _validate_solution(
        self,
        t: NDArray,
        y: NDArray,
        context: str,
        log_calculation_issue: Callable,
    ) -> None:
        """Validate solution array for numerical issues."""
        issues = []
        
        # Check for NaN
        nan_counts = [np.sum(np.isnan(y[i])) for i in range(3)]
        if any(nan_counts):
            issues.append(f"NaN values: x={nan_counts[0]}, y={nan_counts[1]}, z={nan_counts[2]}")
        
        # Check for Inf
        inf_counts = [np.sum(np.isinf(y[i])) for i in range(3)]
        if any(inf_counts):
            issues.append(f"Inf values: x={inf_counts[0]}, y={inf_counts[1]}, z={inf_counts[2]}")
        
        # Check for extreme values
        x_max, y_max, z_max = np.nanmax(np.abs(y[0])), np.nanmax(np.abs(y[1])), np.nanmax(np.abs(y[2]))
        if x_max > 100:
            issues.append(f"Extreme x values: max |x| = {x_max:.2f}")
        if y_max > 100:
            issues.append(f"Extreme y values: max |y| = {y_max:.2f}")
        if z_max > 100:
            issues.append(f"Extreme z values: max |z| = {z_max:.2f}")
        
        if issues:
            log_calculation_issue(
                f"Solution validation ({context})",
                "; ".join(issues),
                {
                    "nan_counts": nan_counts,
                    "inf_counts": inf_counts,
                    "max_abs": {"x": float(x_max), "y": float(y_max), "z": float(z_max)},
                }
            )
    
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
        z_crit_range: tuple = (0.3, 0.8),
        n_samples: int = 10,
        **run_kwargs,
    ) -> list:
        """
        Run sensitivity analysis over z_crit values.
        
        UPDATED: Now varies absolute z_crit instead of threshold_fraction.
        
        Parameters
        ----------
        scenario : str
            Scenario to analyze.
        z_crit_range : tuple
            (min, max) z_crit values to test.
        n_samples : int
            Number of samples.
        **run_kwargs
            Additional arguments passed to run().
        
        Returns
        -------
        list
            List of (z_crit, SimulationResults) tuples.
        """
        start_step, end_step, log_error, log_calculation_issue = _get_logging_utils()
        
        start_step(f"Sensitivity analysis: {scenario}")
        
        try:
            z_crit_values = np.linspace(z_crit_range[0], z_crit_range[1], n_samples)
            results_list = []
            
            for i, zc in enumerate(tqdm(z_crit_values, desc="Sensitivity analysis")):
                logger.debug(f"Running sensitivity sample {i+1}/{n_samples}: z_crit={zc:.3f}")
                
                try:
                    results = self.run(
                        scenario=scenario,
                        z_crit_override=zc,
                        show_progress=False,
                        **run_kwargs,
                    )
                    results_list.append((zc, results))
                    
                except Exception as e:
                    log_error(e, f"Sensitivity sample z_crit={zc:.3f}")
                    results_list.append((zc, None))
            
            end_step(success=True)
            return results_list
            
        except Exception as e:
            log_error(e, "Sensitivity analysis")
            end_step(success=False)
            raise
    
    def __repr__(self) -> str:
        z_crit_str = f"{self.params['z_crit']:.3f}" if self.params['z_crit'] else "auto"
        mode = "absolute" if self.use_absolute_threshold else "fraction"
        return (f"ClimateModel(c={self.params['c']}, epsilon={self.params['epsilon']}, "
                f"beta={self.params['beta']}, z_crit={z_crit_str}, mode={mode})")
