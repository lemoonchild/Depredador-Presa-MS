"""
Animation module for predator-prey simulations.

This module provides functions to create animations showing the temporal
evolution of the predator-prey system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from typing import Optional, Tuple, Callable
from pathlib import Path


def animate_spatial_evolution(
    R_timeseries: np.ndarray,
    F_timeseries: np.ndarray,
    times: np.ndarray,
    refuge_map: Optional[np.ndarray] = None,
    L: float = 10.0,
    title: str = "Predator-Prey Spatial Evolution",
    interval: int = 50,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    fps: int = 20,
    writer: str = 'pillow'
) -> FuncAnimation:
    """
    Create animation of spatial evolution of prey and predator populations.

    Parameters:
        R_timeseries (np.ndarray): Prey density over time (n_times, ny, nx)
        F_timeseries (np.ndarray): Predator density over time (n_times, ny, nx)
        times (np.ndarray): Time points
        refuge_map (np.ndarray, optional): Refuge map to overlay
        L (float): Domain size
        title (str): Animation title
        interval (int): Delay between frames in milliseconds
        figsize (tuple): Figure size
        save_path (str, optional): Path to save animation (supports .gif, .mp4)
        fps (int): Frames per second for saved animation
        writer (str): Animation writer ('pillow' for GIF, 'ffmpeg' for MP4)

    Returns:
        matplotlib.animation.FuncAnimation: The animation object
    """
    n_times = len(times)

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Compute global color limits for consistent scaling
    vmin_R, vmax_R = 0, np.max(R_timeseries) * 1.05
    vmin_F, vmax_F = 0, np.max(F_timeseries) * 1.05

    # Initialize plots
    im1 = ax1.imshow(
        R_timeseries[0],
        origin='lower',
        cmap='Blues',
        extent=[0, L, 0, L],
        vmin=vmin_R,
        vmax=vmax_R,
        interpolation='bilinear'
    )
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('Prey Density (R)', fontweight='bold', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Density', fontsize=10)

    im2 = ax2.imshow(
        F_timeseries[0],
        origin='lower',
        cmap='Reds',
        extent=[0, L, 0, L],
        vmin=vmin_F,
        vmax=vmax_F,
        interpolation='bilinear'
    )
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('Predator Density (F)', fontweight='bold', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Density', fontsize=10)

    # Add refuge contours if available
    if refuge_map is not None:
        ax1.contour(refuge_map, levels=[1.5], colors='green',
                   linewidths=2, alpha=0.6, extent=[0, L, 0, L])
        ax2.contour(refuge_map, levels=[1.5], colors='green',
                   linewidths=2, alpha=0.6, extent=[0, L, 0, L])

    # Time counter text
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=13, fontweight='bold')

    # Main title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    def update(frame):
        """Update function for animation."""
        im1.set_array(R_timeseries[frame])
        im2.set_array(F_timeseries[frame])
        time_text.set_text(f't = {times[frame]:.2f}')
        return im1, im2, time_text

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=n_times,
        interval=interval,
        blit=True,
        repeat=True
    )

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == '.gif':
            print(f"Saving animation as GIF to {save_path}...")
            writer_obj = PillowWriter(fps=fps)
        elif save_path.suffix == '.mp4':
            print(f"Saving animation as MP4 to {save_path}...")
            writer_obj = FFMpegWriter(fps=fps, bitrate=1800)
        else:
            raise ValueError("save_path must end with .gif or .mp4")

        anim.save(str(save_path), writer=writer_obj)
        print(f"Animation saved successfully!")

    return anim


def animate_phase_portrait(
    R_total_timeseries: np.ndarray,
    F_total_timeseries: np.ndarray,
    times: np.ndarray,
    title: str = "Phase Portrait Evolution",
    interval: int = 50,
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None,
    fps: int = 20,
    trail_length: Optional[int] = 50,
    writer: str = 'pillow'
) -> FuncAnimation:
    """
    Create animation of phase portrait trajectory over time.

    Parameters:
        R_total_timeseries (np.ndarray): Total prey population over time
        F_total_timeseries (np.ndarray): Total predator population over time
        times (np.ndarray): Time points
        title (str): Animation title
        interval (int): Delay between frames in milliseconds
        figsize (tuple): Figure size
        save_path (str, optional): Path to save animation
        fps (int): Frames per second for saved animation
        trail_length (int, optional): Length of trajectory trail. None for full trail
        writer (str): Animation writer ('pillow' for GIF, 'ffmpeg' for MP4)

    Returns:
        matplotlib.animation.FuncAnimation: The animation object
    """
    n_times = len(times)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set axis limits with some padding
    R_min, R_max = R_total_timeseries.min(), R_total_timeseries.max()
    F_min, F_max = F_total_timeseries.min(), F_total_timeseries.max()
    R_range = R_max - R_min
    F_range = F_max - F_min

    ax.set_xlim(R_min - 0.1*R_range, R_max + 0.1*R_range)
    ax.set_ylim(F_min - 0.1*F_range, F_max + 0.1*F_range)

    ax.set_xlabel('Total Prey Population (R)', fontsize=11)
    ax.set_ylabel('Total Predator Population (F)', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)

    # Initialize line and point
    line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
    point, = ax.plot([], [], 'ro', markersize=10, zorder=5)

    # Mark initial point
    ax.plot(R_total_timeseries[0], F_total_timeseries[0], 'go',
           markersize=12, label='Initial', zorder=4,
           markeredgecolor='darkgreen', markeredgewidth=2)

    # Time counter
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='best', fontsize=10)

    def update(frame):
        """Update function for animation."""
        # Determine trail start
        if trail_length is None:
            start = 0
        else:
            start = max(0, frame - trail_length)

        # Update trajectory line
        line.set_data(R_total_timeseries[start:frame+1],
                     F_total_timeseries[start:frame+1])

        # Update current point
        point.set_data([R_total_timeseries[frame]], [F_total_timeseries[frame]])

        # Update time text
        time_text.set_text(f't = {times[frame]:.2f}')

        return line, point, time_text

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=n_times,
        interval=interval,
        blit=True,
        repeat=True
    )

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == '.gif':
            print(f"Saving animation as GIF to {save_path}...")
            writer_obj = PillowWriter(fps=fps)
        elif save_path.suffix == '.mp4':
            print(f"Saving animation as MP4 to {save_path}...")
            writer_obj = FFMpegWriter(fps=fps, bitrate=1800)
        else:
            raise ValueError("save_path must end with .gif or .mp4")

        anim.save(str(save_path), writer=writer_obj)
        print(f"Animation saved successfully!")

    return anim


def animate_combined(
    R_timeseries: np.ndarray,
    F_timeseries: np.ndarray,
    times: np.ndarray,
    refuge_map: Optional[np.ndarray] = None,
    L: float = 10.0,
    title: str = "Predator-Prey System Evolution",
    interval: int = 50,
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[str] = None,
    fps: int = 20,
    writer: str = 'pillow'
) -> FuncAnimation:
    """
    Create combined animation with spatial distributions and phase portrait.

    This creates a comprehensive animation showing:
    - Prey spatial distribution
    - Predator spatial distribution
    - Time series of total populations
    - Phase portrait

    Parameters:
        R_timeseries (np.ndarray): Prey density over time (n_times, ny, nx)
        F_timeseries (np.ndarray): Predator density over time (n_times, ny, nx)
        times (np.ndarray): Time points
        refuge_map (np.ndarray, optional): Refuge map to overlay
        L (float): Domain size
        title (str): Animation title
        interval (int): Delay between frames in milliseconds
        figsize (tuple): Figure size
        save_path (str, optional): Path to save animation
        fps (int): Frames per second for saved animation
        writer (str): Animation writer ('pillow' for GIF, 'ffmpeg' for MP4)

    Returns:
        matplotlib.animation.FuncAnimation: The animation object
    """
    n_times = len(times)

    # Compute total populations
    R_totals = np.sum(R_timeseries, axis=(1, 2))
    F_totals = np.sum(F_timeseries, axis=(1, 2))

    # Create figure with grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax_R_spatial = fig.add_subplot(gs[0, 0])
    ax_F_spatial = fig.add_subplot(gs[0, 1])
    ax_timeseries = fig.add_subplot(gs[0, 2])
    ax_phase = fig.add_subplot(gs[1, :])

    # Compute global limits
    vmin_R, vmax_R = 0, np.max(R_timeseries) * 1.05
    vmin_F, vmax_F = 0, np.max(F_timeseries) * 1.05

    # Spatial distributions
    im_R = ax_R_spatial.imshow(
        R_timeseries[0],
        origin='lower',
        cmap='Blues',
        extent=[0, L, 0, L],
        vmin=vmin_R,
        vmax=vmax_R,
        interpolation='bilinear'
    )
    ax_R_spatial.set_title('Prey Density', fontweight='bold', fontsize=11)
    ax_R_spatial.set_xlabel('x')
    ax_R_spatial.set_ylabel('y')
    plt.colorbar(im_R, ax=ax_R_spatial, fraction=0.046)

    im_F = ax_F_spatial.imshow(
        F_timeseries[0],
        origin='lower',
        cmap='Reds',
        extent=[0, L, 0, L],
        vmin=vmin_F,
        vmax=vmax_F,
        interpolation='bilinear'
    )
    ax_F_spatial.set_title('Predator Density', fontweight='bold', fontsize=11)
    ax_F_spatial.set_xlabel('x')
    ax_F_spatial.set_ylabel('y')
    plt.colorbar(im_F, ax=ax_F_spatial, fraction=0.046)

    # Add refuge contours
    if refuge_map is not None:
        ax_R_spatial.contour(refuge_map, levels=[1.5], colors='green',
                            linewidths=1.5, alpha=0.6, extent=[0, L, 0, L])
        ax_F_spatial.contour(refuge_map, levels=[1.5], colors='green',
                            linewidths=1.5, alpha=0.6, extent=[0, L, 0, L])

    # Time series
    line_R, = ax_timeseries.plot([], [], 'b-', linewidth=2, label='Prey')
    line_F, = ax_timeseries.plot([], [], 'r-', linewidth=2, label='Predator')
    ax_timeseries.set_xlim(times[0], times[-1])
    ax_timeseries.set_ylim(0, max(R_totals.max(), F_totals.max()) * 1.1)
    ax_timeseries.set_xlabel('Time', fontsize=10)
    ax_timeseries.set_ylabel('Total Population', fontsize=10)
    ax_timeseries.set_title('Population Time Series', fontweight='bold', fontsize=11)
    ax_timeseries.grid(True, alpha=0.3)
    ax_timeseries.legend(loc='best', fontsize=9)

    # Phase portrait
    line_phase, = ax_phase.plot([], [], 'b-', linewidth=2, alpha=0.7)
    point_phase, = ax_phase.plot([], [], 'ro', markersize=8, zorder=5)
    ax_phase.plot(R_totals[0], F_totals[0], 'go', markersize=10,
                 label='Initial', zorder=4, markeredgecolor='darkgreen', markeredgewidth=2)

    R_range = R_totals.max() - R_totals.min()
    F_range = F_totals.max() - F_totals.min()
    ax_phase.set_xlim(R_totals.min() - 0.1*R_range, R_totals.max() + 0.1*R_range)
    ax_phase.set_ylim(F_totals.min() - 0.1*F_range, F_totals.max() + 0.1*F_range)
    ax_phase.set_xlabel('Total Prey Population', fontsize=10)
    ax_phase.set_ylabel('Total Predator Population', fontsize=10)
    ax_phase.set_title('Phase Portrait', fontweight='bold', fontsize=11)
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend(loc='best', fontsize=9)

    # Time counter
    time_text = fig.text(0.5, 0.96, '', ha='center', fontsize=13, fontweight='bold')

    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.99)

    def update(frame):
        """Update function for animation."""
        # Spatial distributions
        im_R.set_array(R_timeseries[frame])
        im_F.set_array(F_timeseries[frame])

        # Time series
        line_R.set_data(times[:frame+1], R_totals[:frame+1])
        line_F.set_data(times[:frame+1], F_totals[:frame+1])

        # Phase portrait
        line_phase.set_data(R_totals[:frame+1], F_totals[:frame+1])
        point_phase.set_data([R_totals[frame]], [F_totals[frame]])

        # Time text
        time_text.set_text(f't = {times[frame]:.2f}')

        return im_R, im_F, line_R, line_F, line_phase, point_phase, time_text

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=n_times,
        interval=interval,
        blit=True,
        repeat=True
    )

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == '.gif':
            print(f"Saving combined animation as GIF to {save_path}...")
            writer_obj = PillowWriter(fps=fps)
        elif save_path.suffix == '.mp4':
            print(f"Saving combined animation as MP4 to {save_path}...")
            writer_obj = FFMpegWriter(fps=fps, bitrate=1800)
        else:
            raise ValueError("save_path must end with .gif or .mp4")

        anim.save(str(save_path), writer=writer_obj)
        print(f"Animation saved successfully!")

    return anim
