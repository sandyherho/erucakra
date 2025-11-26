"""Time series visualization."""

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
    - Climate variability (x)
    - System energy state (z) with thresholds
    - Radiative forcing (A)
    - Outcome summary statistics
    
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
    
    # Style settings
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#050510")
    
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.2, 1, 0.6], hspace=0.25)
    
    def style_axis(ax):
        ax.set_facecolor("#0a0a15")
        ax.grid(True, alpha=0.12, color="white", linestyle="-", linewidth=0.4)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333355")
            spine.set_linewidth(0.5)
    
    # Colors
    color_trajectory = info.get("color_trajectory", "#00FFFF")
    color_primary = info.get("color_primary", "#00E5CC")
    color_forcing = info.get("color_forcing", "#88FF00")
    
    # === PANEL 1: Climate Variability (x) ===
    ax1 = fig.add_subplot(gs[0])
    style_axis(ax1)
    
    ax1.plot(results.year, results.x, color=color_trajectory, lw=0.6, alpha=0.85)
    ax1.fill_between(results.year, results.x, 0, alpha=0.15, color=color_primary)
    ax1.axhline(0, color="white", alpha=0.2, linestyle="--", lw=0.5)
    ax1.set_ylabel("Climate Variability\n(normalized)", fontsize=10, color="white")
    ax1.set_xlim([results.year[0], results.year[-1]])
    
    # Mark threshold crossing
    if results.crossed_threshold:
        crossing_idx = np.where(results.z > 1)[0]
        if len(crossing_idx) > 0:
            cross_year = results.year[crossing_idx[0]]
            ax1.axvline(cross_year, color="#FF4444", alpha=0.4, lw=1.5, linestyle=":")
            ax1.annotate(
                "Threshold\nCrossed",
                xy=(cross_year, ax1.get_ylim()[1] * 0.7),
                fontsize=8,
                color="#FF6666",
                ha="center",
            )
    
    # === PANEL 2: System Energy State (z) ===
    ax2 = fig.add_subplot(gs[1])
    style_axis(ax2)
    
    ax2.plot(results.year, results.z, color="white", lw=2.5, label="System State (z)")
    ax2.axhline(1.0, color="#FF3333", alpha=0.8, linestyle="--", lw=2, label="Tipping Threshold (z=1)")
    
    z_max = max(1.5, results.z.max() + 0.2)
    ax2.axhspan(1.0, z_max, alpha=0.15, color="#FF0000", label="Danger Zone")
    ax2.axhspan(0.8, 1.0, alpha=0.1, color="#FFAA00", label="Warning Zone")
    ax2.axhspan(-0.5, 0.8, alpha=0.08, color="#00FF00", label="Safe Zone")
    
    ax2.fill_between(
        results.year,
        results.z,
        1.0,
        where=(results.z > 1),
        alpha=0.3,
        color="#FF0000",
        interpolate=True,
    )
    
    ax2.set_ylabel("Accumulated Forcing (z)\n[~Ocean Heat Content]", fontsize=10, color="white")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.3, ncol=2)
    ax2.set_xlim([results.year[0], results.year[-1]])
    ax2.set_ylim([min(-0.1, results.z.min() - 0.1), z_max])
    
    # === PANEL 3: Radiative Forcing A(t) ===
    ax3 = fig.add_subplot(gs[2])
    style_axis(ax3)
    
    ax3.plot(results.year, results.A, color=color_forcing, lw=2.5, label="Radiative Forcing A(t)")
    ax3.fill_between(results.year, results.A, alpha=0.2, color=color_forcing)
    ax3.axhline(1.0, color="#FF3333", alpha=0.5, linestyle=":", lw=1.5, label="Critical Forcing Level")
    
    ax3.set_ylabel("External Forcing A(t)\n[~W/m² normalized]", fontsize=10, color="white")
    ax3.set_xlabel("Year", fontsize=11, color="white")
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax3.set_xlim([results.year[0], results.year[-1]])
    
    # === PANEL 4: Outcome ===
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor("#0a0a15")
    ax4.axis("off")
    
    summary = results.summary()
    expected = info.get("expected_outcome", "N/A")
    
    outcome_colors = {
        "STABLE": "#00FF88",
        "MARGINAL": "#FFDD00",
        "TIPPING": "#FF6600",
        "CATASTROPHIC": "#FF0044",
        "HYSTERESIS": "#AA55FF",
    }
    outcome_color = outcome_colors.get(expected, "white")
    
    stats_text = (
        f"Max z: {summary['max_z']:.2f}  |  "
        f"Time above threshold: {summary['time_above_threshold_pct']:.1f}%  |  "
        f"Max variability: {summary['max_variability']:.2f}  |  "
        f"Final z: {summary['final_z']:.2f}"
    )
    
    ax4.text(
        0.5, 0.7, f"Outcome: {expected}",
        transform=ax4.transAxes, fontsize=14, color=outcome_color,
        ha="center", fontweight="bold",
    )
    ax4.text(
        0.5, 0.25, stats_text,
        transform=ax4.transAxes, fontsize=10, color="#AAAAAA",
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
