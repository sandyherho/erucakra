"""
Core dynamical system equations for climate tipping point model.

Physical Model:
    dx/dt = y
    dy/dt = x(z - z_crit - x²) - cy
    dz/dt = ε(A(t)/A_scale - z - βx²)
"""

from typing import Callable, List, Union, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# GLOBAL FORCING SCALE (CRITICAL CHANGE)
# =============================================================================
# Use a SINGLE global scale based on SSP5-8.5 peak forcing (~13 W/m²)
# This ensures all scenarios are normalized consistently
GLOBAL_A_SCALE = 13.0  # W/m²


# New approach: all scenarios use global scale
FORCING_SCALES = {
    "ssp126": GLOBAL_A_SCALE,
    "ssp245": GLOBAL_A_SCALE,
    "ssp370": GLOBAL_A_SCALE,
    "ssp585": GLOBAL_A_SCALE,
    "custom": GLOBAL_A_SCALE,
}


# =============================================================================
# ABSOLUTE Z_CRIT THRESHOLD
# =============================================================================


DEFAULT_Z_CRIT_ABSOLUTE = 0.55  # Absolute threshold in normalized units



def climate_tipping_model(
    t: float,
    state: List[float],
    c: float,
    epsilon: float,
    A_func: Union[Callable[[float], float], float],
    beta: float = 0.8,
    z_crit: float = DEFAULT_Z_CRIT_ABSOLUTE,
    A_scale: float = GLOBAL_A_SCALE,
) -> List[float]:
    """
    Climate Tipping Point Dynamical System with global forcing normalization.
    
    The model equations:
        dx/dt = y
        dy/dt = x(z - z_crit - x²) - cy  
        dz/dt = ε(A(t)/A_scale - z - βx²)
    
    The bifurcation occurs when z crosses z_crit:
    - z < z_crit: Single stable fixed point at x=0 (stable climate)
    - z > z_crit: Pitchfork bifurcation, x → ±√(z - z_crit) (tipped state)
    
    Parameters
    ----------
    t : float
        Current time.
    state : List[float]
        Current state vector [x, y, z].
    c : float
        Damping coefficient (energy dissipation, typically 0.1-0.5).
        Higher c = faster decay of oscillations.
    epsilon : float
        Timescale separation (slow/fast ratio, typically 0.01-0.05).
        Smaller ε = stronger separation, slower z dynamics.
    A_func : Callable or float
        Time-dependent radiative forcing function A(t) in W/m² or constant.
    beta : float, optional
        Feedback strength (variability → accumulation coupling).
        Higher β = stronger negative feedback from oscillations.
        Default is 0.8.
    z_crit : float, optional
        Critical threshold for tipping (absolute value in normalized units).
        Default is 0.55.
    A_scale : float, optional
        Global forcing normalization scale (W/m²).
        Default is 13.0 (based on SSP5-8.5 peak).
    
    Returns
    -------
    List[float]
        Time derivatives [dx/dt, dy/dt, dz/dt].
    
    Physical Interpretation
    -----------------------
    The (x, y) subsystem is a nonlinear oscillator whose stability depends on z:
    
    - x: Fast climate variability (ENSO-like interannual oscillations)
    - y: Rate of change / momentum (dx/dt)
    - z: Slow accumulated state (ocean heat content, ice sheet mass)
    
    The term x(z - z_crit - x²) creates:
    - Restoring force toward x=0 when z < z_crit (stable climate)
    - Repulsion from x=0 when z > z_crit (tipped, bistable climate)
    - Saturation via -x³ (prevents unbounded growth)
    
    The z equation shows:
    - z relaxes toward A(t)/A_scale on timescale 1/ε
    - βx² feedback: high variability reduces effective forcing
    - This creates hysteresis: tipping may not reverse when forcing decreases
    """
    x, y, z = state
    
    # Time-dependent or constant forcing (normalized by GLOBAL scale)
    if callable(A_func):
        A_normalized = A_func(t) / A_scale
    else:
        A_normalized = A_func / A_scale
    
    # System of ODEs
    # Fast subsystem (x, y): nonlinear oscillator
    dxdt = y
    dydt = x * (z - z_crit - x**2) - c * y
    
    # Slow subsystem (z): forced accumulation with feedback
    dzdt = epsilon * (A_normalized - z - beta * x**2)
    
    return [dxdt, dydt, dzdt]


def compute_fixed_points(z: float, z_crit: float) -> dict:
    """
    Compute fixed points and their stability for given z value.
    
    Parameters
    ----------
    z : float
        Current value of slow variable.
    z_crit : float
        Critical threshold.
    
    Returns
    -------
    dict
        Dictionary with fixed points and stability information.
    """
    if z <= z_crit:
        # Pre-tipping: single stable fixed point
        return {
            "regime": "stable",
            "fixed_points": [(0.0, 0.0)],
            "stability": ["stable"],
            "x_amplitude": 0.0,
        }
    else:
        # Post-tipping: pitchfork bifurcation
        x_eq = np.sqrt(z - z_crit)
        return {
            "regime": "tipped",
            "fixed_points": [(0.0, 0.0), (x_eq, 0.0), (-x_eq, 0.0)],
            "stability": ["unstable", "stable", "stable"],
            "x_amplitude": x_eq,
        }


def compute_effective_potential(x: NDArray, z: float, z_crit: float) -> NDArray:
    """
    Compute effective potential V(x) for the fast subsystem.
    
    The dynamics satisfy dx/dt = -dV/dx (in the overdamped limit).
    
    V(x) = -x²(z - z_crit)/2 + x⁴/4
    
    Parameters
    ----------
    x : NDArray
        Array of x values.
    z : float
        Current slow variable value.
    z_crit : float
        Critical threshold.
    
    Returns
    -------
    NDArray
        Potential values V(x).
    """
    return -x**2 * (z - z_crit) / 2 + x**4 / 4


def add_climate_noise(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    noise_level: float = 0.05,
    smoothing: float = 10.0,
    seed: Optional[int] = None,
    noise_color: str = "red",
) -> NDArray[np.float64]:
    """
    Add realistic climate variability (colored noise) to solution.
    
    Mimics natural oscillations like ENSO, PDO, AMO and other modes
    of internal climate variability not captured by the deterministic model.
    
    Parameters
    ----------
    t : NDArray
        Time array.
    y : NDArray
        Solution array of shape (3, n_points).
    noise_level : float, optional
        Amplitude of noise (standard deviation). Default is 0.05.
    smoothing : float, optional
        Gaussian smoothing sigma for temporal correlation. Default is 10.0.
        Higher values = more autocorrelated (redder) noise.
    seed : int, optional
        Random seed for reproducibility.
    noise_color : str, optional
        Type of noise: "red" (autocorrelated), "white", or "pink".
        Default is "red".
    
    Returns
    -------
    NDArray
        Solution array with noise added to x component.
    
    Notes
    -----
    Only the fast variable x receives noise, as it represents
    weather/interannual variability. The slow variable z represents
    integrated quantities (ocean heat) that average over fast noise.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    y_noisy = y.copy()
    
    if noise_color == "white":
        noise = rng.standard_normal(n) * noise_level
    elif noise_color == "pink":
        # 1/f noise via spectral method
        white = rng.standard_normal(n)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1e-10  # Avoid division by zero
        fft_pink = fft / np.sqrt(freqs)
        noise = np.fft.irfft(fft_pink, n)
        noise = noise / np.std(noise) * noise_level
    else:  # red noise (default)
        white_noise = rng.standard_normal(n)
        red_noise = gaussian_filter1d(white_noise, sigma=smoothing)
        if np.std(red_noise) > 0:
            noise = red_noise / np.std(red_noise) * noise_level
        else:
            noise = np.zeros(n)
    
    # Add to fast variable only
    y_noisy[0] = y_noisy[0] + noise
    
    return y_noisy
