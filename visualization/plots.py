"""
Static visualization module for predator-prey simulations.

This module provides functions to create publication-quality static plots
for analyzing simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, List
from pathlib import Path

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_spatial_distribution(
    R: np.ndarray,
    F: np.ndarray,
    time: float,
    refuge_map: Optional[np.ndarray] = None,
    L: float = 10.0,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    vmin_R: Optional[float] = None,
    vmax_R: Optional[float] = None,
    vmin_F: Optional[float] = None,
    vmax_F: Optional[float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot spatial distribution of prey and predator populations at a given time.

    Creates side-by-side heatmaps showing prey (R) and predator (F) densities,
    with optional refuge map overlay.

    Parameters:
        R (np.ndarray): Prey density distribution (ny, nx)
        F (np.ndarray): Predator density distribution (ny, nx)
        time (float): Time point for this snapshot
        refuge_map (np.ndarray, optional): Refuge map to overlay
        L (float): Domain size
        title (str, optional): Figure title
        figsize (tuple): Figure size (width, height)
        vmin_R, vmax_R (float, optional): Color scale limits for prey
        vmin_F, vmax_F (float, optional): Color scale limits for predator
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_plots = 3 if refuge_map is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 2:
        axes = list(axes)

    # Prey distribution
    im_R = axes[0].imshow(
        R,
        origin='lower',
        cmap='Blues',
        extent=[0, L, 0, L],
        interpolation='bilinear',
        vmin=vmin_R,
        vmax=vmax_R
    )
    axes[0].set_title(f'Prey Density (R)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('y', fontsize=11)
    cbar_R = plt.colorbar(im_R, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_R.set_label('Density', fontsize=10)

    # Overlay refuge contours on prey plot if available
    if refuge_map is not None:
        axes[0].contour(
            refuge_map,
            levels=[1.5],
            colors='green',
            linewidths=2,
            alpha=0.6,
            extent=[0, L, 0, L]
        )

    # Predator distribution
    im_F = axes[1].imshow(
        F,
        origin='lower',
        cmap='Reds',
        extent=[0, L, 0, L],
        interpolation='bilinear',
        vmin=vmin_F,
        vmax=vmax_F
    )
    axes[1].set_title(f'Predator Density (F)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('y', fontsize=11)
    cbar_F = plt.colorbar(im_F, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_F.set_label('Density', fontsize=10)

    # Overlay refuge contours on predator plot if available
    if refuge_map is not None:
        axes[1].contour(
            refuge_map,
            levels=[1.5],
            colors='green',
            linewidths=2,
            alpha=0.6,
            extent=[0, L, 0, L]
        )

    # Refuge map if provided
    if refuge_map is not None:
        im_refuge = axes[2].imshow(
            refuge_map,
            origin='lower',
            cmap='Greens',
            extent=[0, L, 0, L],
            interpolation='bilinear'
        )
        axes[2].set_title('Refuge Map', fontweight='bold', fontsize=12)
        axes[2].set_xlabel('x', fontsize=11)
        axes[2].set_ylabel('y', fontsize=11)
        cbar_refuge = plt.colorbar(im_refuge, ax=axes[2], fraction=0.046, pad=0.04)
        cbar_refuge.set_label('Growth Enhancement p(x,y)', fontsize=10)

    # Overall title
    if title is None:
        title = f'Spatial Distribution at t = {time:.2f}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_time_series(
    times: np.ndarray,
    R_total: np.ndarray,
    F_total: np.ndarray,
    R_std: Optional[np.ndarray] = None,
    F_std: Optional[np.ndarray] = None,
    title: str = "Population Time Series",
    figsize: Tuple[int, int] = (12, 5),
    show_confidence_bands: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series of total prey and predator populations.

    Parameters:
        times (np.ndarray): Time points
        R_total (np.ndarray): Total prey population at each time
        F_total (np.ndarray): Total predator population at each time
        R_std (np.ndarray, optional): Standard deviation of prey (for stochastic)
        F_std (np.ndarray, optional): Standard deviation of predator
        title (str): Figure title
        figsize (tuple): Figure size
        show_confidence_bands (bool): Show confidence bands if std provided
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Prey time series
    ax1.plot(times, R_total, 'b-', linewidth=2, label='Prey (R)')
    if R_std is not None and show_confidence_bands:
        ax1.fill_between(
            times,
            R_total - 2*R_std,
            R_total + 2*R_std,
            alpha=0.3,
            color='blue',
            label='95% Confidence'
        )
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Total Population', fontsize=11)
    ax1.set_title('Prey Population Over Time', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # Predator time series
    ax2.plot(times, F_total, 'r-', linewidth=2, label='Predator (F)')
    if F_std is not None and show_confidence_bands:
        ax2.fill_between(
            times,
            F_total - 2*F_std,
            F_total + 2*F_std,
            alpha=0.3,
            color='red',
            label='95% Confidence'
        )
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Total Population', fontsize=11)
    ax2.set_title('Predator Population Over Time', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_phase_portrait(
    R_total: np.ndarray,
    F_total: np.ndarray,
    times: Optional[np.ndarray] = None,
    title: str = "Phase Portrait",
    mark_initial: bool = True,
    mark_final: bool = True,
    color_by_time: bool = True,
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot phase portrait (R vs F) showing system trajectory.

    Parameters:
        R_total (np.ndarray): Total prey population trajectory
        F_total (np.ndarray): Total predator population trajectory
        times (np.ndarray, optional): Time points for color coding
        title (str): Figure title
        mark_initial (bool): Mark initial condition
        mark_final (bool): Mark final state
        color_by_time (bool): Color trajectory by time
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if color_by_time and times is not None:
        # Color by time
        points = np.array([R_total, F_total]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', linewidth=2)
        lc.set_array(times)
        line = ax.add_collection(lc)
        cbar = plt.colorbar(line, ax=ax)
        cbar.set_label('Time', fontsize=11)
    else:
        # Single color
        ax.plot(R_total, F_total, 'b-', linewidth=2, alpha=0.7)

    # Mark initial condition
    if mark_initial:
        ax.plot(R_total[0], F_total[0], 'go', markersize=12,
                label='Initial', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)

    # Mark final state
    if mark_final:
        ax.plot(R_total[-1], F_total[-1], 'r^', markersize=12,
                label='Final', zorder=5, markeredgecolor='darkred', markeredgewidth=2)

    ax.set_xlabel('Total Prey Population (R)', fontsize=11)
    ax.set_ylabel('Total Predator Population (F)', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)

    if mark_initial or mark_final:
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_scenario_comparison(
    results_dict: Dict[str, Dict],
    L: float = 10.0,
    figsize: Tuple[int, int] = (16, 14),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare final states of multiple scenarios in a grid layout.

    Parameters:
        results_dict (dict): Dictionary of scenario results
            Each value should have keys 'R', 'F', 'times', 'refuge_map'
        L (float): Domain size
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_scenarios = len(results_dict)
    n_cols = 2  # Prey and Predator
    n_rows = n_scenarios

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_scenarios == 1:
        axes = axes.reshape(1, -1)

    for idx, (scenario_name, result) in enumerate(results_dict.items()):
        R_final = result['R'][-1]
        F_final = result['F'][-1]
        refuge_map = result.get('refuge_map', None)

        # Prey
        im_R = axes[idx, 0].imshow(
            R_final,
            origin='lower',
            cmap='Blues',
            extent=[0, L, 0, L],
            interpolation='bilinear'
        )
        axes[idx, 0].set_title(f'{scenario_name}\nPrey', fontweight='bold', fontsize=11)
        axes[idx, 0].set_xlabel('x')
        axes[idx, 0].set_ylabel('y')
        plt.colorbar(im_R, ax=axes[idx, 0], fraction=0.046)

        # Overlay refuge contours if available
        if refuge_map is not None:
            axes[idx, 0].contour(
                refuge_map,
                levels=[1.5],
                colors='green',
                linewidths=1.5,
                alpha=0.5,
                extent=[0, L, 0, L]
            )

        # Predator
        im_F = axes[idx, 1].imshow(
            F_final,
            origin='lower',
            cmap='Reds',
            extent=[0, L, 0, L],
            interpolation='bilinear'
        )
        axes[idx, 1].set_title(f'{scenario_name}\nPredator', fontweight='bold', fontsize=11)
        axes[idx, 1].set_xlabel('x')
        axes[idx, 1].set_ylabel('y')
        plt.colorbar(im_F, ax=axes[idx, 1], fraction=0.046)

        # Overlay refuge contours if available
        if refuge_map is not None:
            axes[idx, 1].contour(
                refuge_map,
                levels=[1.5],
                colors='green',
                linewidths=1.5,
                alpha=0.5,
                extent=[0, L, 0, L]
            )

    fig.suptitle('Scenario Comparison - Final States', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_stochastic_ensemble(
    times: np.ndarray,
    R_all: np.ndarray,
    F_all: np.ndarray,
    n_trajectories_to_plot: int = 10,
    title: str = "Stochastic Ensemble",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ensemble of stochastic realizations with mean and percentiles.

    Parameters:
        times (np.ndarray): Time points
        R_all (np.ndarray): All realizations for prey (n_realizations, n_times, ny, nx)
        F_all (np.ndarray): All realizations for predator
        n_trajectories_to_plot (int): Number of individual trajectories to show
        title (str): Figure title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Compute total populations for each realization
    n_realizations = R_all.shape[0]
    R_totals = np.sum(R_all, axis=(2, 3))  # (n_realizations, n_times)
    F_totals = np.sum(F_all, axis=(2, 3))

    # Compute statistics
    R_mean = np.mean(R_totals, axis=0)
    F_mean = np.mean(F_totals, axis=0)
    R_percentiles = np.percentile(R_totals, [25, 50, 75], axis=0)
    F_percentiles = np.percentile(F_totals, [25, 50, 75], axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Prey
    # Individual trajectories (sample)
    indices = np.random.choice(n_realizations, min(n_trajectories_to_plot, n_realizations), replace=False)
    for i in indices:
        ax1.plot(times, R_totals[i], 'b-', alpha=0.15, linewidth=0.8)

    # Mean and percentiles
    ax1.plot(times, R_mean, 'b-', linewidth=2.5, label='Mean', zorder=10)
    ax1.fill_between(times, R_percentiles[0], R_percentiles[2],
                     alpha=0.3, color='blue', label='25-75 percentile')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Total Prey Population', fontsize=11)
    ax1.set_title(f'Prey - {n_realizations} Realizations', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # Predator
    # Individual trajectories (sample)
    for i in indices:
        ax2.plot(times, F_totals[i], 'r-', alpha=0.15, linewidth=0.8)

    # Mean and percentiles
    ax2.plot(times, F_mean, 'r-', linewidth=2.5, label='Mean', zorder=10)
    ax2.fill_between(times, F_percentiles[0], F_percentiles[2],
                     alpha=0.3, color='red', label='25-75 percentile')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Total Predator Population', fontsize=11)
    ax2.set_title(f'Predator - {n_realizations} Realizations', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_parameter_sensitivity(
    param_values: List[float],
    param_name: str,
    metric_values: np.ndarray,
    metric_name: str = "Metric",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot sensitivity analysis showing how a metric varies with a parameter.

    Parameters:
        param_values (list): Parameter values tested
        param_name (str): Name of the parameter
        metric_values (np.ndarray): Metric values for each parameter (can be 2D for prey/predator)
        metric_name (str): Name of the metric
        title (str, optional): Figure title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if metric_values.ndim == 1:
        # Single metric
        ax.plot(param_values, metric_values, 'o-', linewidth=2, markersize=8)
        ax.set_ylabel(metric_name, fontsize=11)
    else:
        # Two metrics (e.g., prey and predator)
        ax.plot(param_values, metric_values[0], 'bo-', linewidth=2,
                markersize=8, label='Prey')
        ax.plot(param_values, metric_values[1], 'ro-', linewidth=2,
                markersize=8, label='Predator')
        ax.set_ylabel(metric_name, fontsize=11)
        ax.legend(loc='best', fontsize=10)

    ax.set_xlabel(f'{param_name}', fontsize=11)
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f'Parameter Sensitivity: {param_name} vs {metric_name}'
    ax.set_title(title, fontweight='bold', fontsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
