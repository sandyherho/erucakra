"""
Core dynamical system equations for climate tipping point model.
"""

from typing import Callable, List, Union, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


def climate_tipping_model(
    t: float,
    state: List[float],
    c: float,
    epsilon: float,
    A_func: Union[Callable[[float], float], float],
    beta: float = 0.8,
    z_crit: float = 1.0,
) -> List[float]:
    """
    Climate Tipping Point Dynamical System.
    
    The model equations:
        dx/dt = y
        dy/dt = x(z - z_crit - x²) - cy
        dz/dt = ε(A(t) - z - βx²)
    
    Parameters
    ----------
    t : float
        Current time.
    state : List[float]
        Current state vector [x, y, z].
    c : float
        Damping coefficient (energy dissipation).
    epsilon : float
        Timescale separation (slow/fast ratio, typically 0.01-0.05).
    A_func : Callable or float
        Time-dependent radiative forcing function or constant value.
    beta : float, optional
        Feedback strength (how variability affects accumulation).
        Default is 0.8.
    z_crit : float, optional
        Critical threshold (normalized to 1.0 ≈ 1.5°C warming).
        Default is 1.0.
    
    Returns
    -------
    List[float]
        Time derivatives [dx/dt, dy/dt, dz/dt].
    
    Physical Interpretation
    -----------------------
    - x: Fast climate variability (interannual-decadal oscillations)
    - y: Rate of change / momentum in climate system
    - z: Slow accumulated forcing (ocean heat content, ice sheet state)
    - A: Effective radiative forcing (W/m² scaled)
    - z_crit = 1.0 represents ~1.5°C threshold (Paris target)
    - x² feedback: variability affects mean state
    - Slow z evolution: ocean thermal inertia (~decades)
    """
    x, y, z = state
    
    # Time-dependent or constant forcing
    current_A = A_func(t) if callable(A_func) else A_func
    
    # System of ODEs
    dxdt = y
    dydt = x * (z - z_crit - x**2) - c * y
    dzdt = epsilon * (current_A - z - beta * x**2)
    
    return [dxdt, dydt, dzdt]


def add_climate_noise(
    t: NDArray[np.float64],
    y: NDArray[np.float64],
    noise_level: float = 0.05,
    smoothing: float = 10.0,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Add realistic climate variability (red noise) to solution.
    
    Mimics natural oscillations like ENSO, PDO, and other modes
    of climate variability.
    
    Parameters
    ----------
    t : NDArray
        Time array.
    y : NDArray
        Solution array of shape (3, n_points).
    noise_level : float, optional
        Amplitude of noise. Default is 0.05.
    smoothing : float, optional
        Gaussian smoothing sigma for red noise. Default is 10.0.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    NDArray
        Solution array with noise added to x component.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(t)
    y_noisy = y.copy()
    
    # Generate red noise (autocorrelated)
    white_noise = np.random.randn(n)
    red_noise = gaussian_filter1d(white_noise, sigma=smoothing)
    
    # Normalize and scale
    if np.std(red_noise) > 0:
        red_noise = red_noise / np.std(red_noise) * noise_level
    
    # Add to fast variable only (weather/variability)
    y_noisy[0] = y_noisy[0] + red_noise
    
    return y_noisy
