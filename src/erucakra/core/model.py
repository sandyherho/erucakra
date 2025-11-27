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
    Climate Tipping Point Model with enhanced logging and error tracking.
    
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
    
    def get_z_crit(self, scenario_key: Optional[str] = None) -> float:
        """Get effective z_crit value."""
        if self.params["z_crit"] is not None:
            return self.params["z_crit"]
        
        if scenario_key is not None and scenario_key in DEFAULT_Z_CRIT:
            return DEFAULT_Z_CRIT[scenario_key]
        
        return DEFAULT_Z_CRIT.get("custom", 0.80)
    
    def get_A_scale(self, scenario_key: Optional[str] = None) -> float:
        """Get forcing scale for normalization."""
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
        Run the climate tipping point simulation with comprehensive logging.
        
        All steps are timed and logged. Any calculation issues (NaN, Inf, etc.)
        are detected and logged for debugging.
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
            
            # Lyapunov exponent (chaos indicator)
            dt = (t_end - t_start) / n_points
            lyapunov = compute_lyapunov_exponent(y, dt)
            
            # Check for extreme values
            max_z = float(np.nanmax(y[2]))
            final_z = float(y[2, -1])
            max_variability = float(np.nanmax(np.abs(y[0])))
            
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
            
            if np.isnan(lyapunov):
                log_calculation_issue(
                    "Lyapunov calculation failed",
                    "Lyapunov exponent is NaN",
                    {}
                )
                lyapunov = 0.0
            
            logger.info(f"Time above threshold: {time_above_pct:.1f}%")
            logger.info(f"Max z: {max_z:.4f}")
            logger.info(f"Final z: {final_z:.4f}")
            logger.info(f"Lyapunov exponent: {lyapunov:.6f}")
            
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
                    "max_z": max_z,
                    "final_z": final_z,
                    "max_variability": max_variability,
                },
            )
            
            logger.info(f"Simulation complete. Max z={results.diagnostics['max_z']:.3f}, "
                       f"crossed={results.crossed_threshold}")
            
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
        z_crit_range: tuple = (0.5, 1.2),
        n_samples: int = 10,
        **run_kwargs,
    ) -> list:
        """
        Run sensitivity analysis over z_crit values.
        """
        start_step, end_step, log_error, log_calculation_issue = _get_logging_utils()
        
        start_step(f"Sensitivity analysis: {scenario}")
        
        try:
            z_crit_values = np.linspace(z_crit_range[0], z_crit_range[1], n_samples)
            results_list = []
            
            for i, z_crit in enumerate(tqdm(z_crit_values, desc="Sensitivity analysis")):
                logger.debug(f"Running sensitivity sample {i+1}/{n_samples}: z_crit={z_crit:.3f}")
                
                try:
                    results = self.run(
                        scenario=scenario,
                        z_crit_override=z_crit,
                        show_progress=False,
                        **run_kwargs,
                    )
                    results_list.append((z_crit, results))
                    
                except Exception as e:
                    log_error(e, f"Sensitivity sample z_crit={z_crit:.3f}")
                    results_list.append((z_crit, None))
            
            end_step(success=True)
            return results_list
            
        except Exception as e:
            log_error(e, "Sensitivity analysis")
            end_step(success=False)
            raise
    
    def __repr__(self) -> str:
        z_crit_str = f"{self.params['z_crit']}" if self.params['z_crit'] else "auto"
        return (f"ClimateModel(c={self.params['c']}, epsilon={self.params['epsilon']}, "
                f"beta={self.params['beta']}, z_crit={z_crit_str})")
