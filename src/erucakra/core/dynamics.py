"""
Core dynamical system equations for climate tipping point model.

The key insight: z_crit must be scaled relative to the forcing A(t),
not as an absolute threshold.

Physical Model:
    dx/dt = y
    dy/dt = x(z - z_crit - x²) - cy
    dz/dt = ε(A(t)/A_scale - z - βx²)

Where A_scale normalizes forcing to O(1) values, allowing z_crit ~ 1
to represent a meaningful fraction of maximum forcing.
"""

from typing import Callable, List, Union, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


# Forcing scale factors for each scenario (approximate max forcing)
# This normalizes A(t) so that z evolves on O(1) scale
FORCING_SCALES = {
    "ssp126": 3.6,   # Peak ~3.6 W/m²
    "ssp245": 5.5,   # Stabilizes ~5.4 W/m²
    "ssp370": 8.5,   # Reaches ~8.2 W/m²
    "ssp585": 10.5,  # Reaches ~10+ W/m²
    "custom": 5.0,   # Default for custom forcing
}

# Default z_crit values tuned for each scenario to show expected behavior
# These represent the fraction of normalized forcing needed to tip
DEFAULT_Z_CRIT = {
    "ssp126": 0.95,  # Just above max normalized forcing → STABLE
    "ssp245": 0.82,  # Near max → MARGINAL  
    "ssp370": 0.65,  # Below max → TIPPING
    "ssp585": 0.55,  # Well below max → CATASTROPHIC
    "custom": 0.80,  # Default
}


def climate_tipping_model(
    t: float,
    state: List[float],
    c: float,
    epsilon: float,
    A_func: Union[Callable[[float], float], float],
    beta: float = 0.8,
    z_crit: float = 1.0,
    A_scale: float = 1.0,
) -> List[float]:
    """
    Climate Tipping Point Dynamical System with proper forcing normalization.
    
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
        Critical threshold for tipping (in normalized units).
        Default is 1.0.
    A_scale : float, optional
        Forcing normalization scale (W/m²). Divides A(t) to get O(1) values.
        Default is 1.0 (no scaling).
    
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
    
    # Time-dependent or constant forcing (normalized)
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


def compute_lyapunov_exponent(
    trajectory: NDArray,
    dt: float,
    transient: int = 1000,
) -> float:
    """
    Estimate largest Lyapunov exponent from trajectory.
    
    Uses the method of nearby trajectories divergence rate.
    Positive λ indicates chaos, negative indicates stable dynamics.
    
    Parameters
    ----------
    trajectory : NDArray
        State trajectory of shape (3, n_points).
    dt : float
        Time step.
    transient : int, optional
        Number of initial points to discard. Default is 1000.
    
    Returns
    -------
    float
        Estimated largest Lyapunov exponent.
    """
    x = trajectory[0, transient:]
    y = trajectory[1, transient:]
    
    # Compute local divergence rates
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Finite-time Lyapunov exponent approximation
    divergence = np.sqrt(dx**2 + dy**2)
    divergence = divergence[divergence > 1e-10]  # Avoid log(0)
    
    if len(divergence) < 10:
        return 0.0
    
    # Average log divergence rate
    lyap = np.mean(np.log(divergence)) / dt
    
    return float(lyap)
