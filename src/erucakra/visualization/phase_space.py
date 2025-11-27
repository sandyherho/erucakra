"""
3D phase space animation.
"""

from typing import TYPE_CHECKING
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm

if TYPE_CHECKING:
    from erucakra.core.results import SimulationResults

logger = logging.getLogger(__name__)


# High-contrast colormaps for dark backgrounds
BRIGHT_COLORMAPS = {
    "ssp126": LinearSegmentedColormap.from_list(
        "bright_green",
        ["#00FF88", "#00FFCC", "#00FFFF", "#88FFFF", "#FFFFFF"],
    ),
    "ssp245": LinearSegmentedColormap.from_list(
        "bright_yellow",
        ["#FFFF00", "#FFDD00", "#FFAA00", "#FF8800", "#FF6600"],
    ),
    "ssp370": LinearSegmentedColormap.from_list(
        "bright_orange",
        ["#FF8800", "#FF6600", "#FF4400", "#FF2200", "#FF0000"],
    ),
    "ssp585": LinearSegmentedColormap.from_list(
        "bright_red",
        ["#FF0088", "#FF0066", "#FF0044", "#FF4444", "#FFAAAA"],
    ),
    "default": LinearSegmentedColormap.from_list(
        "bright_cyan",
        ["#00FFFF", "#00CCFF", "#0088FF", "#4444FF", "#8888FF"],
    ),
}


def create_phase_space_gif(
    results: "SimulationResults",
    filepath: str | Path,
    fps: int = 30,
    duration_seconds: int = 12,
    dpi: int = 100,
) -> None:
    """
    Create 3D phase space animation with persistent trajectory trace.
    
    Parameters
    ----------
    results : SimulationResults
        Simulation results to visualize.
    filepath : str or Path
        Output GIF file path.
    fps : int, optional
        Frames per second. Default is 30.
    duration_seconds : int, optional
        Animation duration in seconds. Default is 12.
    dpi : int, optional
        Resolution. Default is 100.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating phase space GIF: {filepath}")
    
    info = results.scenario_info or {}
    scenario_key = results.scenario_key or "default"
    z_crit = results.z_crit
    threshold_fraction = results.threshold_fraction
    
    total_frames = fps * duration_seconds
    
    n_points = min(len(results.t), total_frames * 8)
    indices = np.linspace(0, len(results.t) - 1, n_points).astype(int)
    
    x = results.x[indices]
    y = results.y[indices]
    z = results.z[indices]
    years = results.year[indices]
    
    logger.debug(f"Preparing {total_frames} frames with persistent trace")
    
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor("#000510")
    
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#000510")
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#202040")
    ax.yaxis.pane.set_edgecolor("#202040")
    ax.zaxis.pane.set_edgecolor("#202040")
    ax.grid(True, alpha=0.15, color="white", linestyle="-", linewidth=0.3)
    
    ax.set_xlabel("Climate Variability (x)", fontsize=10, color="#CCCCCC", labelpad=8)
    ax.set_ylabel("Rate of Change (y)", fontsize=10, color="#CCCCCC", labelpad=8)
    ax.set_zlabel("Accumulated Forcing (z)", fontsize=10, color="#CCCCCC", labelpad=8)
    ax.tick_params(colors="#888888", labelsize=7)
    
    pad = 0.15
    x_lim = [x.min() - pad * max(abs(x.min()), 0.5), x.max() + pad * max(abs(x.max()), 0.5)]
    y_lim = [y.min() - pad * max(abs(y.min()), 0.5), y.max() + pad * max(abs(y.max()), 0.5)]
    z_lim = [min(-0.1, z.min() - 0.1), max(z_crit + 0.5, z.max() + 0.2)]
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    
    # Tipping threshold plane
    xx_plane, yy_plane = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], 15),
        np.linspace(y_lim[0], y_lim[1], 15),
    )
    zz_plane = np.ones_like(xx_plane) * z_crit
    ax.plot_surface(
        xx_plane, yy_plane, zz_plane,
        alpha=0.15, color="#FF0000", shade=False, zorder=1,
    )
    
    border_x = [x_lim[0], x_lim[1], x_lim[1], x_lim[0], x_lim[0]]
    border_y = [y_lim[0], y_lim[0], y_lim[1], y_lim[1], y_lim[0]]
    border_z = [z_crit] * 5
    ax.plot(border_x, border_y, border_z, color="#FF4444", lw=1.5, alpha=0.7)
    
    ax.text(x_lim[1], y_lim[1], z_crit + 0.02, 
            f"z_crit = {z_crit:.3f}", color="#FF6666", fontsize=9)
    
    cmap = BRIGHT_COLORMAPS.get(scenario_key, BRIGHT_COLORMAPS["default"])
    norm = Normalize(vmin=years[0], vmax=years[-1])
    
    ax.scatter(
        [x[0]], [y[0]], [z[0]],
        c="#00FF00", s=120, marker="o",
        edgecolors="white", linewidths=2,
        label=f"Start ({int(years[0])})", zorder=20,
    )
    
    color_primary = info.get("color_primary", "#00FFFF")
    color_secondary = info.get("color_secondary", "#FFFFFF")
    
    head_point, = ax.plot(
        [], [], [], "o", markersize=16,
        color=color_secondary,
        markeredgecolor="white", markeredgewidth=2.5, zorder=25,
    )
    head_glow, = ax.plot(
        [], [], [], "o", markersize=28,
        color=color_secondary, alpha=0.4, zorder=24,
    )
    head_glow2, = ax.plot(
        [], [], [], "o", markersize=20,
        color="white", alpha=0.3, zorder=24,
    )
    
    scenario_name = info.get("name", "Custom Scenario")
    subtitle = info.get("subtitle", "")
    
    ax.text2D(
        0.5, 0.96, scenario_name,
        transform=ax.transAxes, fontsize=16,
        color="white", ha="center", fontweight="bold",
    )
    ax.text2D(
        0.5, 0.92, subtitle,
        transform=ax.transAxes, fontsize=11,
        color="#AAAAAA", ha="center",
    )
    
    status_text = ax.text2D(
        0.02, 0.06, "",
        transform=ax.transAxes, fontsize=10, color="white",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#000020", 
                  alpha=0.8, edgecolor="#444466"),
    )
    year_text = ax.text2D(
        0.98, 0.06, "",
        transform=ax.transAxes, fontsize=15, color="#FFFF00",
        ha="right", fontweight="bold", family="monospace",
    )
    regime_text = ax.text2D(
        0.98, 0.94, "",
        transform=ax.transAxes, fontsize=12, ha="right", fontweight="bold",
    )
    
    # Show threshold_fraction in corner
    ax.text2D(
        0.02, 0.94, f"threshold_fraction = {threshold_fraction:.2f}\nz_crit = {z_crit:.3f}",
        transform=ax.transAxes, fontsize=9, color="#FF8888",
        family="monospace",
    )
    
    collection_artist = [None]
    
    pbar = tqdm(total=total_frames, desc="Generating frames", leave=True)
    
    def init():
        head_point.set_data([], [])
        head_point.set_3d_properties([])
        head_glow.set_data([], [])
        head_glow.set_3d_properties([])
        head_glow2.set_data([], [])
        head_glow2.set_3d_properties([])
        status_text.set_text("")
        year_text.set_text("")
        regime_text.set_text("")
        return head_point, head_glow, head_glow2, status_text, year_text, regime_text
    
    def animate(frame):
        pbar.update(1)
        
        progress = frame / total_frames
        idx = int(progress * (len(x) - 1))
        idx = max(1, idx)
        
        if collection_artist[0] is not None:
            collection_artist[0].remove()
        
        points = np.array([x[: idx + 1], y[: idx + 1], z[: idx + 1]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        segment_colors = cmap(norm(years[:idx]))
        
        lc = Line3DCollection(
            segments, 
            colors=segment_colors, 
            linewidth=2.5,
            alpha=0.95,
        )
        collection_artist[0] = ax.add_collection3d(lc)
        
        curr_x, curr_y, curr_z = x[idx], y[idx], z[idx]
        
        head_point.set_data([curr_x], [curr_y])
        head_point.set_3d_properties([curr_z])
        head_glow.set_data([curr_x], [curr_y])
        head_glow.set_3d_properties([curr_z])
        head_glow2.set_data([curr_x], [curr_y])
        head_glow2.set_3d_properties([curr_z])
        
        # Binary regime status
        is_tipped = curr_z > z_crit
        regime_str = "TIPPED" if is_tipped else "NOT TIPPED"
        status_text.set_text(
            f"x={curr_x:+.3f}  y={curr_y:+.3f}  z={curr_z:.3f}\n"
            f"Δz = {curr_z - z_crit:+.3f}  |  {regime_str}"
        )
        
        year_text.set_text(f"Year: {int(years[idx])}")
        
        if is_tipped:
            regime_text.set_text("TIPPED")
            regime_text.set_color("#FF4444")
        else:
            regime_text.set_text("NOT TIPPED")
            regime_text.set_color("#00FF88")
        
        elev = 22 + 8 * np.sin(progress * 1.5 * np.pi)
        azim = -60 + progress * 200
        ax.view_init(elev=elev, azim=azim)
        
        return head_point, head_glow, head_glow2, status_text, year_text, regime_text
    
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000 / fps, blit=False,
    )
    
    writer = PillowWriter(fps=fps)
    anim.save(filepath, writer=writer, dpi=dpi)
    plt.close()
    
    pbar.close()
    logger.info(f"Phase space GIF saved: {filepath}")
