"""3D phase space animation."""

from typing import TYPE_CHECKING
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm

if TYPE_CHECKING:
    from erucakra.core.results import SimulationResults

logger = logging.getLogger(__name__)


def create_phase_space_gif(
    results: "SimulationResults",
    filepath: str | Path,
    fps: int = 30,
    duration_seconds: int = 12,
    dpi: int = 100,
) -> None:
    """
    Create 3D phase space animation with persistent trajectory trace.
    
    The entire history remains visible - trajectory builds up over time
    showing complete system evolution from start to current time.
    
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
    total_frames = fps * duration_seconds
    
    # Subsample for smooth animation
    n_points = min(len(results.t), total_frames * 8)
    indices = np.linspace(0, len(results.t) - 1, n_points).astype(int)
    
    x = results.x[indices]
    y = results.y[indices]
    z = results.z[indices]
    years = results.year[indices]
    
    logger.debug(f"Preparing {total_frames} frames with persistent trace")
    
    # Setup figure
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor("#000008")
    
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#000008")
    
    # Styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#151525")
    ax.yaxis.pane.set_edgecolor("#151525")
    ax.zaxis.pane.set_edgecolor("#151525")
    ax.grid(True, alpha=0.08, color="white", linestyle="-")
    
    # Labels
    ax.set_xlabel("Climate Variability (x)", fontsize=10, color="#AAAAAA", labelpad=8)
    ax.set_ylabel("Rate of Change (y)", fontsize=10, color="#AAAAAA", labelpad=8)
    ax.set_zlabel("Accumulated Forcing (z)", fontsize=10, color="#AAAAAA", labelpad=8)
    ax.tick_params(colors="#666666", labelsize=7)
    
    # Axis limits
    pad = 0.15
    x_lim = [x.min() - pad * max(abs(x.min()), 0.5), x.max() + pad * max(abs(x.max()), 0.5)]
    y_lim = [y.min() - pad * max(abs(y.min()), 0.5), y.max() + pad * max(abs(y.max()), 0.5)]
    z_lim = [min(-0.1, z.min() - 0.1), max(1.6, z.max() + 0.2)]
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    
    # Draw tipping threshold plane (z = 1)
    xx_plane, yy_plane = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], 15),
        np.linspace(y_lim[0], y_lim[1], 15),
    )
    zz_plane = np.ones_like(xx_plane)
    ax.plot_surface(
        xx_plane, yy_plane, zz_plane,
        alpha=0.12, color="#FF0000", shade=False, zorder=1,
    )
    
    # Threshold plane border
    border_x = [x_lim[0], x_lim[1], x_lim[1], x_lim[0], x_lim[0]]
    border_y = [y_lim[0], y_lim[0], y_lim[1], y_lim[1], y_lim[0]]
    border_z = [1, 1, 1, 1, 1]
    ax.plot(border_x, border_y, border_z, color="#FF3333", lw=1, alpha=0.5)
    
    # Colormap
    cmap = plt.get_cmap(info.get("cmap_name", "viridis"))
    norm = Normalize(vmin=years[0], vmax=years[-1])
    
    # Start marker
    ax.scatter(
        [x[0]], [y[0]], [z[0]],
        c="#00FF00", s=100, marker="o",
        edgecolors="white", linewidths=1.5,
        label=f"Start ({int(years[0])})", zorder=20,
    )
    
    # Colors
    color_secondary = info.get("color_secondary", "#FFA500")
    color_primary = info.get("color_primary", "#FFD700")
    
    # Current position markers
    head_point, = ax.plot(
        [], [], [], "o", markersize=14,
        color=color_secondary,
        markeredgecolor="white", markeredgewidth=2, zorder=25,
    )
    head_glow, = ax.plot(
        [], [], [], "o", markersize=24,
        color=color_secondary, alpha=0.3, zorder=24,
    )
    
    # Shadow
    shadow_line, = ax.plot(
        [], [], [], lw=0.6, alpha=0.2,
        color=color_primary, zorder=2,
    )
    
    # Text overlays
    scenario_name = info.get("name", "Custom Scenario")
    subtitle = info.get("subtitle", "")
    
    ax.text2D(
        0.5, 0.96, scenario_name,
        transform=ax.transAxes, fontsize=15,
        color="white", ha="center", fontweight="bold",
    )
    ax.text2D(
        0.5, 0.92, subtitle,
        transform=ax.transAxes, fontsize=10,
        color="#888888", ha="center",
    )
    
    status_text = ax.text2D(
        0.02, 0.06, "",
        transform=ax.transAxes, fontsize=10, color="white",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#000000", alpha=0.7, edgecolor="#333333"),
    )
    year_text = ax.text2D(
        0.98, 0.06, "",
        transform=ax.transAxes, fontsize=14, color="#FFFF00",
        ha="right", fontweight="bold", family="monospace",
    )
    threshold_text = ax.text2D(
        0.98, 0.94, "",
        transform=ax.transAxes, fontsize=11, ha="right", fontweight="bold",
    )
    
    collection_artist = [None]
    
    # Progress bar for frame generation
    pbar = tqdm(total=total_frames, desc="Generating frames", leave=True)
    
    def init():
        head_point.set_data([], [])
        head_point.set_3d_properties([])
        head_glow.set_data([], [])
        head_glow.set_3d_properties([])
        shadow_line.set_data([], [])
        shadow_line.set_3d_properties([])
        status_text.set_text("")
        year_text.set_text("")
        threshold_text.set_text("")
        return head_point, head_glow, shadow_line, status_text, year_text, threshold_text
    
    def animate(frame):
        pbar.update(1)
        
        progress = frame / total_frames
        idx = int(progress * (len(x) - 1))
        idx = max(1, idx)
        
        # Remove previous collection
        if collection_artist[0] is not None:
            collection_artist[0].remove()
        
        # Create segments for trajectory
        points = np.array([x[: idx + 1], y[: idx + 1], z[: idx + 1]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        segment_colors = cmap(norm(years[:idx]))
        
        lc = Line3DCollection(segments, colors=segment_colors, linewidth=1.5, alpha=0.9)
        collection_artist[0] = ax.add_collection3d(lc)
        
        # Update current position
        curr_x, curr_y, curr_z = x[idx], y[idx], z[idx]
        
        head_point.set_data([curr_x], [curr_y])
        head_point.set_3d_properties([curr_z])
        head_glow.set_data([curr_x], [curr_y])
        head_glow.set_3d_properties([curr_z])
        
        # Shadow
        shadow_line.set_data(x[: idx + 1], y[: idx + 1])
        shadow_line.set_3d_properties(np.full(idx + 1, z_lim[0]))
        
        # Status
        status = "ABOVE THRESHOLD" if curr_z > 1 else "Below Threshold"
        status_text.set_text(f"x={curr_x:+.2f}  y={curr_y:+.2f}  z={curr_z:.3f}\n{status}")
        
        year_text.set_text(f"Year: {int(years[idx])}")
        
        if curr_z > 1:
            threshold_text.set_text("TIPPING REGIME")
            threshold_text.set_color("#FF4444")
        elif curr_z > 0.8:
            threshold_text.set_text("WARNING ZONE")
            threshold_text.set_color("#FFAA00")
        else:
            threshold_text.set_text("SAFE ZONE")
            threshold_text.set_color("#00FF88")
        
        # Camera rotation
        elev = 22 + 8 * np.sin(progress * 1.5 * np.pi)
        azim = -60 + progress * 200
        ax.view_init(elev=elev, azim=azim)
        
        return head_point, head_glow, shadow_line, status_text, year_text, threshold_text
    
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000 / fps, blit=False,
    )
    
    writer = PillowWriter(fps=fps)
    anim.save(filepath, writer=writer, dpi=dpi)
    plt.close()
    
    pbar.close()
    logger.info(f"Phase space GIF saved: {filepath}")
