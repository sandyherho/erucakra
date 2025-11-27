"""
Time series visualization with threshold indication.
"""

from typing import TYPE_CHECKING
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from erucakra.core.results import SimulationResults

logger = logging.getLogger(__name__)


def create_timeseries_plot(
    results: "SimulationResults",
    filepath: str | Path,
    dpi: int = 200,
) -> None:
    """
    Generate comprehensive time series visualization.
    
    Creates a 4-panel diagnostic plot showing:
    - Climate variability (x) with regime shading
    - System state (z) with threshold z_crit
    - Normalized forcing (A/A_scale) showing what z "sees"
    - Summary statistics and outcome
    
    Parameters
    ----------
    results : SimulationResults
        Simulation results to visualize.
    filepath : str or Path
        Output PNG file path.
    dpi : int, optional
        Resolution. Default is 200.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating time series plot: {filepath}")
    
    info = results.scenario_info or {}
    z_crit = results.z_crit
    A_scale = results.A_scale
    
    # Style settings
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("#050510")
    
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.3, 1, 0.5], hspace=0.25)
    
    def style_axis(ax):
        ax.set_facecolor("#0a0a15")
        ax.grid(True, alpha=0.12, color="white", linestyle="-", linewidth=0.4)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333355")
            spine.set_linewidth(0.5)
    
    # Colors from scenario
    color_trajectory = info.get("color_trajectory", "#00FFFF")
    color_primary = info.get("color_primary", "#00E5CC")
    color_forcing = info.get("color_forcing", "#88FF00")
    
    # Get regime transitions
    transitions = results.get_regime_transitions()
    
    # === PANEL 1: Climate Variability (x) ===
    ax1 = fig.add_subplot(gs[0])
    style_axis(ax1)
    
    ax1.plot(results.year, results.x, color=color_trajectory, lw=0.7, alpha=0.9)
    ax1.fill_between(results.year, results.x, 0, alpha=0.2, color=color_primary)
    ax1.axhline(0, color="white", alpha=0.3, linestyle="--", lw=0.5)
    
    # Shade tipped regions
    tipped = results.z > z_crit
    ax1.fill_between(
        results.year, ax1.get_ylim()[0], ax1.get_ylim()[1],
        where=tipped, alpha=0.1, color="#FF0000",
        transform=ax1.get_xaxis_transform(),
    )
    
    ax1.set_ylabel("Climate Variability (x)\n[normalized]", fontsize=10, color="white")
    ax1.set_xlim([results.year[0], results.year[-1]])
    
    # Mark threshold crossings
    for trans in transitions:
        color = "#FF4444" if trans["direction"] == "up" else "#44FF44"
        ax1.axvline(trans["year"], color=color, alpha=0.5, lw=1.5, linestyle=":")
    
    # === PANEL 2: System State (z) with z_crit ===
    ax2 = fig.add_subplot(gs[1])
    style_axis(ax2)
    
    ax2.plot(results.year, results.z, color="white", lw=2.5, label=f"System State (z)")
    ax2.axhline(z_crit, color="#FF3333", alpha=0.9, linestyle="--", lw=2.5, 
                label=f"Tipping Threshold (z_crit={z_crit:.2f})")
    
    # Regime zones relative to z_crit
    z_max = max(z_crit + 0.5, results.z.max() + 0.2)
    z_min = min(-0.1, results.z.min() - 0.1)
    
    # Danger zone: above z_crit
    ax2.axhspan(z_crit, z_max, alpha=0.15, color="#FF0000", label="Tipped Regime")
    # Warning zone: 80% to 100% of z_crit
    ax2.axhspan(z_crit * 0.8, z_crit, alpha=0.1, color="#FFAA00", label="Warning Zone")
    # Safe zone: below 80%
    ax2.axhspan(z_min, z_crit * 0.8, alpha=0.08, color="#00FF00", label="Safe Zone")
    
    # Fill when above threshold
    ax2.fill_between(
        results.year,
        results.z,
        z_crit,
        where=(results.z > z_crit),
        alpha=0.35,
        color="#FF0000",
        interpolate=True,
    )
    
    # Mark crossings
    for trans in transitions:
        color = "#FF4444" if trans["direction"] == "up" else "#44FF44"
        marker = "^" if trans["direction"] == "up" else "v"
        ax2.plot(trans["year"], z_crit, marker, color=color, markersize=12, 
                markeredgecolor="white", markeredgewidth=1.5, zorder=10)
        label = "TIPPING" if trans["direction"] == "up" else "Recovery"
        ax2.annotate(
            f"{label}\n{int(trans['year'])}",
            xy=(trans["year"], z_crit),
            xytext=(0, 25 if trans["direction"] == "up" else -35),
            textcoords="offset points",
            fontsize=8, color=color, ha="center",
            arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
        )
    
    ax2.set_ylabel(f"Accumulated State (z)\n[z_crit = {z_crit:.2f}]", fontsize=10, color="white")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.4, ncol=2)
    ax2.set_xlim([results.year[0], results.year[-1]])
    ax2.set_ylim([z_min, z_max])
    
    # === PANEL 3: Normalized Forcing A(t)/A_scale ===
    ax3 = fig.add_subplot(gs[2])
    style_axis(ax3)
    
    # Plot normalized forcing (what z equation sees)
    ax3.plot(results.year, results.A_normalized, color=color_forcing, lw=2.5, 
             label=f"Normalized Forcing (A/{A_scale:.1f})")
    ax3.fill_between(results.year, results.A_normalized, alpha=0.25, color=color_forcing)
    
    # Show z_crit level for reference
    ax3.axhline(z_crit, color="#FF3333", alpha=0.6, linestyle=":", lw=1.5,
                label=f"z_crit = {z_crit:.2f}")
    
    # Secondary axis for raw forcing
    ax3b = ax3.twinx()
    ax3b.plot(results.year, results.A, color="#AAAAAA", lw=1, alpha=0.4, linestyle="--")
    ax3b.set_ylabel("Raw Forcing (W/m²)", fontsize=9, color="#888888")
    ax3b.tick_params(colors="#666666", labelsize=8)
    
    ax3.set_ylabel(f"Normalized Forcing\n[A(t) / {A_scale:.1f} W/m²]", fontsize=10, color="white")
    ax3.set_xlabel("Year", fontsize=11, color="white")
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.4)
    ax3.set_xlim([results.year[0], results.year[-1]])
    
    # === PANEL 4: Summary ===
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor("#0a0a15")
    ax4.axis("off")
    
    summary = results.summary()
    expected = info.get("expected_outcome", "N/A")
    
    # Determine actual outcome
    if not results.crossed_threshold:
        actual_outcome = "STABLE"
    elif summary["time_above_threshold_pct"] > 50:
        actual_outcome = "TIPPED"
    else:
        actual_outcome = "MARGINAL"
    
    outcome_colors = {
        "STABLE": "#00FF88",
        "MARGINAL": "#FFDD00",
        "TIPPING": "#FF6600",
        "TIPPED": "#FF4444",
        "CATASTROPHIC": "#FF0044",
    }
    expected_color = outcome_colors.get(expected, "#AAAAAA")
    actual_color = outcome_colors.get(actual_outcome, "#FFFFFF")
    
    # Parameters box
    params_text = (
        f"Parameters:  z_crit = {z_crit:.2f}  |  "
        f"A_scale = {A_scale:.1f} W/m²  |  "
        f"ε = {results.model_params.get('epsilon', 0.02)}  |  "
        f"c = {results.model_params.get('c', 0.2)}  |  "
        f"β = {results.model_params.get('beta', 0.8)}"
    )
    
    ax4.text(
        0.5, 0.85, params_text,
        transform=ax4.transAxes, fontsize=9, color="#888888",
        ha="center", family="monospace",
    )
    
    # Expected vs actual
    ax4.text(
        0.3, 0.55, f"Expected: {expected}",
        transform=ax4.transAxes, fontsize=13, color=expected_color,
        ha="center", fontweight="bold",
    )
    ax4.text(
        0.7, 0.55, f"Actual: {actual_outcome}",
        transform=ax4.transAxes, fontsize=13, color=actual_color,
        ha="center", fontweight="bold",
    )
    
    # Statistics
    crossing_str = f"Year {int(summary['first_crossing_year'])}" if summary['first_crossing_year'] else "Never"
    stats_text = (
        f"Max z: {summary['max_z']:.3f}  |  "
        f"Final z: {summary['final_z']:.3f}  |  "
        f"Time above z_crit: {summary['time_above_threshold_pct']:.1f}%  |  "
        f"First crossing: {crossing_str}"
    )
    
    ax4.text(
        0.5, 0.2, stats_text,
        transform=ax4.transAxes, fontsize=10, color="#CCCCCC",
        ha="center", family="monospace",
    )
    
    # Title
    scenario_name = info.get("name", "Custom Scenario")
    subtitle = info.get("subtitle", "")
    fig.suptitle(
        f"{scenario_name}\n{subtitle}",
        fontsize=18, color="white", fontweight="bold", y=0.98,
    )
    
    plt.savefig(
        filepath, dpi=dpi, facecolor="#050510",
        edgecolor="none", bbox_inches="tight",
    )
    plt.close()
    
    logger.info(f"Time series plot saved: {filepath}")
