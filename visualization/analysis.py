"""
Analysis module for predator-prey simulations.

This module provides quantitative metrics and analysis tools for
evaluating simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from scipy import fft
from scipy.spatial import distance


def compute_total_population(
    density: np.ndarray,
    dx: float,
    dy: float
) -> Union[float, np.ndarray]:
    """
    Compute total population by integrating density over domain.

    Approximates: ∫∫ density(x,y) dx dy

    Parameters:
        density (np.ndarray): Density field (ny, nx) or (n_times, ny, nx)
        dx (float): Spatial step in x
        dy (float): Spatial step in y

    Returns:
        float or np.ndarray: Total population (scalar if 2D input, array if 3D input)
    """
    # Integration factor
    dA = dx * dy

    if density.ndim == 2:
        # Single snapshot
        return np.sum(density) * dA
    elif density.ndim == 3:
        # Time series
        return np.sum(density, axis=(1, 2)) * dA
    else:
        raise ValueError("Density must be 2D or 3D array")


def compute_spatial_variance(
    density: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray
) -> Union[float, np.ndarray]:
    """
    Compute spatial variance of population distribution.

    Measures how spread out the population is in space.

    Parameters:
        density (np.ndarray): Density field (ny, nx) or (n_times, ny, nx)
        X (np.ndarray): Meshgrid of x-coordinates (ny, nx)
        Y (np.ndarray): Meshgrid of y-coordinates (ny, nx)

    Returns:
        float or np.ndarray: Spatial variance
    """
    def _variance_single(d):
        """Compute variance for single snapshot."""
        if np.sum(d) == 0:
            return 0.0

        # Compute center of mass
        x_cm = np.sum(X * d) / np.sum(d)
        y_cm = np.sum(Y * d) / np.sum(d)

        # Compute variance
        var = np.sum(((X - x_cm)**2 + (Y - y_cm)**2) * d) / np.sum(d)
        return var

    if density.ndim == 2:
        return _variance_single(density)
    elif density.ndim == 3:
        return np.array([_variance_single(density[i]) for i in range(density.shape[0])])
    else:
        raise ValueError("Density must be 2D or 3D array")


def compute_distance_to_equilibrium(
    R_timeseries: np.ndarray,
    F_timeseries: np.ndarray,
    R_eq: Optional[float] = None,
    F_eq: Optional[float] = None
) -> np.ndarray:
    """
    Compute Euclidean distance to equilibrium in phase space.

    Parameters:
        R_timeseries (np.ndarray): Total prey population over time
        F_timeseries (np.ndarray): Total predator population over time
        R_eq (float, optional): Equilibrium prey population (if None, uses final value)
        F_eq (float, optional): Equilibrium predator population

    Returns:
        np.ndarray: Distance to equilibrium at each time point
    """
    if R_eq is None:
        R_eq = R_timeseries[-1]
    if F_eq is None:
        F_eq = F_timeseries[-1]

    distances = np.sqrt((R_timeseries - R_eq)**2 + (F_timeseries - F_eq)**2)
    return distances


def compute_oscillation_frequency(
    timeseries: np.ndarray,
    times: np.ndarray,
    return_power_spectrum: bool = False
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Compute dominant oscillation frequency using FFT.

    Parameters:
        timeseries (np.ndarray): Time series data
        times (np.ndarray): Time points
        return_power_spectrum (bool): If True, also return full power spectrum

    Returns:
        float or tuple: Dominant frequency (Hz), or (frequency, frequencies, power_spectrum)
    """
    # Compute FFT
    dt = times[1] - times[0]
    n = len(timeseries)

    # Detrend by subtracting mean
    signal = timeseries - np.mean(timeseries)

    # Compute FFT
    fft_vals = fft.rfft(signal)
    fft_freq = fft.rfftfreq(n, dt)
    power_spectrum = np.abs(fft_vals)**2

    # Find dominant frequency (excluding DC component)
    dominant_idx = np.argmax(power_spectrum[1:]) + 1
    dominant_freq = fft_freq[dominant_idx]

    if return_power_spectrum:
        return dominant_freq, fft_freq, power_spectrum
    else:
        return dominant_freq


def compute_coefficient_of_variation(
    timeseries: np.ndarray
) -> float:
    """
    Compute coefficient of variation (CV = std/mean).

    Measures relative variability of the population.

    Parameters:
        timeseries (np.ndarray): Time series data

    Returns:
        float: Coefficient of variation
    """
    mean_val = np.mean(timeseries)
    if mean_val == 0:
        return np.inf
    return np.std(timeseries) / mean_val


def compute_persistence(
    timeseries: np.ndarray,
    threshold: float = 1.0
) -> float:
    """
    Compute persistence probability (fraction of time above threshold).

    Parameters:
        timeseries (np.ndarray): Time series data
        threshold (float): Minimum population for persistence

    Returns:
        float: Persistence probability [0, 1]
    """
    return np.mean(timeseries > threshold)


def analyze_single_scenario(
    result: Dict,
    scenario_name: str = "Scenario",
    L: float = 10.0
) -> Dict:
    """
    Perform comprehensive analysis on a single scenario result.

    Parameters:
        result (dict): Simulation result with keys 'R', 'F', 'times', etc.
        scenario_name (str): Name of the scenario
        L (float): Domain size

    Returns:
        dict: Dictionary of computed metrics
    """
    R = result['R']
    F = result['F']
    times = result['times']

    n_times, ny, nx = R.shape
    dx = L / (nx - 1)
    dy = L / (ny - 1)

    # Create meshgrid
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y)

    # Compute total populations
    R_total = compute_total_population(R, dx, dy)
    F_total = compute_total_population(F, dx, dy)

    # Compute spatial variances
    R_spatial_var = compute_spatial_variance(R, X, Y)
    F_spatial_var = compute_spatial_variance(F, X, Y)

    # Compute temporal statistics
    R_mean = np.mean(R_total)
    R_std = np.std(R_total)
    R_cv = compute_coefficient_of_variation(R_total)
    R_min = np.min(R_total)
    R_max = np.max(R_total)

    F_mean = np.mean(F_total)
    F_std = np.std(F_total)
    F_cv = compute_coefficient_of_variation(F_total)
    F_min = np.min(F_total)
    F_max = np.max(F_total)

    # Compute oscillation frequencies
    R_freq = compute_oscillation_frequency(R_total, times)
    F_freq = compute_oscillation_frequency(F_total, times)

    # Compute persistence
    R_persistence = compute_persistence(R_total, threshold=1.0)
    F_persistence = compute_persistence(F_total, threshold=1.0)

    # Distance to equilibrium
    distance_to_eq = compute_distance_to_equilibrium(R_total, F_total)
    final_distance = distance_to_eq[-1]

    # Final populations
    R_final = R_total[-1]
    F_final = F_total[-1]

    # Compile metrics
    metrics = {
        'scenario': scenario_name,
        # Prey metrics
        'R_mean': R_mean,
        'R_std': R_std,
        'R_cv': R_cv,
        'R_min': R_min,
        'R_max': R_max,
        'R_final': R_final,
        'R_freq': R_freq,
        'R_persistence': R_persistence,
        'R_spatial_var_mean': np.mean(R_spatial_var),
        'R_spatial_var_final': R_spatial_var[-1],
        # Predator metrics
        'F_mean': F_mean,
        'F_std': F_std,
        'F_cv': F_cv,
        'F_min': F_min,
        'F_max': F_max,
        'F_final': F_final,
        'F_freq': F_freq,
        'F_persistence': F_persistence,
        'F_spatial_var_mean': np.mean(F_spatial_var),
        'F_spatial_var_final': F_spatial_var[-1],
        # System metrics
        'final_distance_to_eq': final_distance,
        'simulation_time': times[-1],
    }

    return metrics


def analyze_stochastic_ensemble(
    result: Dict,
    scenario_name: str = "Stochastic Scenario",
    L: float = 10.0
) -> Dict:
    """
    Analyze ensemble of stochastic realizations.

    Parameters:
        result (dict): Result from StochasticSolver.solve_ensemble()
            with keys 'R_all', 'F_all', 'R_mean', 'F_mean', etc.
        scenario_name (str): Name of the scenario
        L (float): Domain size

    Returns:
        dict: Dictionary of ensemble metrics
    """
    R_all = result['R_all']
    F_all = result['F_all']
    times = result['times']
    n_realizations = result['n_realizations']

    n_realizations, n_times, ny, nx = R_all.shape
    dx = L / (nx - 1)
    dy = L / (ny - 1)

    # Compute total populations for each realization
    R_totals = np.zeros((n_realizations, n_times))
    F_totals = np.zeros((n_realizations, n_times))

    for i in range(n_realizations):
        R_totals[i] = compute_total_population(R_all[i], dx, dy)
        F_totals[i] = compute_total_population(F_all[i], dx, dy)

    # Ensemble statistics
    R_mean_trajectory = np.mean(R_totals, axis=0)
    F_mean_trajectory = np.mean(F_totals, axis=0)

    R_std_trajectory = np.std(R_totals, axis=0)
    F_std_trajectory = np.std(F_totals, axis=0)

    # Final population statistics
    R_final_mean = np.mean(R_totals[:, -1])
    R_final_std = np.std(R_totals[:, -1])

    F_final_mean = np.mean(F_totals[:, -1])
    F_final_std = np.std(F_totals[:, -1])

    # Extinction probability (< 1.0 population)
    R_extinction_prob = np.mean(R_totals[:, -1] < 1.0)
    F_extinction_prob = np.mean(F_totals[:, -1] < 1.0)

    # Mean coefficient of variation across realizations
    R_cv_list = [compute_coefficient_of_variation(R_totals[i]) for i in range(n_realizations)]
    F_cv_list = [compute_coefficient_of_variation(F_totals[i]) for i in range(n_realizations)]

    R_cv_mean = np.mean(R_cv_list)
    F_cv_mean = np.mean(F_cv_list)

    metrics = {
        'scenario': scenario_name,
        'n_realizations': n_realizations,
        # Prey ensemble metrics
        'R_final_mean': R_final_mean,
        'R_final_std': R_final_std,
        'R_mean_trajectory_mean': np.mean(R_mean_trajectory),
        'R_std_trajectory_mean': np.mean(R_std_trajectory),
        'R_cv_mean': R_cv_mean,
        'R_extinction_prob': R_extinction_prob,
        # Predator ensemble metrics
        'F_final_mean': F_final_mean,
        'F_final_std': F_final_std,
        'F_mean_trajectory_mean': np.mean(F_mean_trajectory),
        'F_std_trajectory_mean': np.mean(F_std_trajectory),
        'F_cv_mean': F_cv_mean,
        'F_extinction_prob': F_extinction_prob,
    }

    return metrics


def compare_scenarios(
    results_dict: Dict[str, Dict],
    L: float = 10.0,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare metrics across multiple scenarios.

    Parameters:
        results_dict (dict): Dictionary mapping scenario names to results
        L (float): Domain size
        save_path (str, optional): Path to save comparison table (CSV or Excel)

    Returns:
        pd.DataFrame: Comparison table
    """
    metrics_list = []

    for scenario_name, result in results_dict.items():
        # Check if stochastic ensemble
        if 'R_all' in result and 'n_realizations' in result:
            metrics = analyze_stochastic_ensemble(result, scenario_name, L)
        else:
            metrics = analyze_single_scenario(result, scenario_name, L)

        metrics_list.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(metrics_list)

    # Set scenario as index
    if 'scenario' in df.columns:
        df = df.set_index('scenario')

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == '.csv':
            df.to_csv(save_path)
        elif save_path.suffix in ['.xlsx', '.xls']:
            df.to_excel(save_path)
        else:
            raise ValueError("save_path must end with .csv, .xlsx, or .xls")

        print(f"Comparison table saved to: {save_path}")

    return df


def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str],
    title: str = "Metric Comparison Across Scenarios",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create bar plots comparing metrics across scenarios.

    Parameters:
        comparison_df (pd.DataFrame): Comparison table from compare_scenarios()
        metrics (list): List of metric names to compare
        title (str): Figure title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(metrics):
        if metric not in comparison_df.columns:
            axes[idx].text(0.5, 0.5, f'Metric "{metric}" not found',
                          ha='center', va='center')
            axes[idx].set_title(metric)
            continue

        data = comparison_df[metric]
        scenarios = comparison_df.index

        axes[idx].bar(range(len(scenarios)), data, alpha=0.7, edgecolor='black')
        axes[idx].set_xticks(range(len(scenarios)))
        axes[idx].set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
        axes[idx].set_ylabel(metric, fontsize=10)
        axes[idx].set_title(metric.replace('_', ' ').title(), fontweight='bold', fontsize=11)
        axes[idx].grid(True, alpha=0.3, axis='y')

    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_refuge_effect_analysis(
    result_no_refuge: Dict,
    result_with_refuge: Dict,
    L: float = 10.0,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze and visualize the effect of refuges on system dynamics.

    Parameters:
        result_no_refuge (dict): Result without refuges
        result_with_refuge (dict): Result with refuges
        L (float): Domain size
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Extract data
    times_no = result_no_refuge['times']
    times_yes = result_with_refuge['times']

    R_no = result_no_refuge['R']
    F_no = result_no_refuge['F']
    R_yes = result_with_refuge['R']
    F_yes = result_with_refuge['F']

    ny, nx = R_no.shape[1], R_no.shape[2]
    dx = L / (nx - 1)
    dy = L / (ny - 1)

    # Compute total populations
    R_total_no = compute_total_population(R_no, dx, dy)
    F_total_no = compute_total_population(F_no, dx, dy)
    R_total_yes = compute_total_population(R_yes, dx, dy)
    F_total_yes = compute_total_population(F_yes, dx, dy)

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Final spatial distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # No refuge - prey
    im1 = ax1.imshow(R_no[-1], origin='lower', cmap='Blues', extent=[0, L, 0, L])
    ax1.set_title('No Refuge - Prey', fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # With refuge - prey
    refuge_map = result_with_refuge.get('refuge_map', None)
    im2 = ax2.imshow(R_yes[-1], origin='lower', cmap='Blues', extent=[0, L, 0, L])
    ax2.set_title('With Refuge - Prey', fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    if refuge_map is not None:
        ax2.contour(refuge_map, levels=[1.5], colors='green', linewidths=2, alpha=0.6, extent=[0, L, 0, L])

    # Difference
    diff = R_yes[-1] - R_no[-1]
    im3 = ax3.imshow(diff, origin='lower', cmap='RdBu_r', extent=[0, L, 0, L])
    ax3.set_title('Difference (With - Without)', fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Row 2: Time series comparison
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(times_no, R_total_no, 'b--', linewidth=2, label='Prey (No Refuge)', alpha=0.7)
    ax4.plot(times_yes, R_total_yes, 'b-', linewidth=2, label='Prey (With Refuge)')
    ax4.plot(times_no, F_total_no, 'r--', linewidth=2, label='Predator (No Refuge)', alpha=0.7)
    ax4.plot(times_yes, F_total_yes, 'r-', linewidth=2, label='Predator (With Refuge)')
    ax4.set_xlabel('Time', fontsize=11)
    ax4.set_ylabel('Total Population', fontsize=11)
    ax4.set_title('Population Dynamics Comparison', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=10)

    # Row 3: Phase portraits
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(R_total_no, F_total_no, 'b-', linewidth=2, alpha=0.7)
    ax5.plot(R_total_no[0], F_total_no[0], 'go', markersize=10, label='Initial')
    ax5.plot(R_total_no[-1], F_total_no[-1], 'ro', markersize=10, label='Final')
    ax5.set_xlabel('Prey Population')
    ax5.set_ylabel('Predator Population')
    ax5.set_title('Phase Portrait - No Refuge', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best', fontsize=9)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(R_total_yes, F_total_yes, 'b-', linewidth=2, alpha=0.7)
    ax6.plot(R_total_yes[0], F_total_yes[0], 'go', markersize=10, label='Initial')
    ax6.plot(R_total_yes[-1], F_total_yes[-1], 'ro', markersize=10, label='Final')
    ax6.set_xlabel('Prey Population')
    ax6.set_ylabel('Predator Population')
    ax6.set_title('Phase Portrait - With Refuge', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='best', fontsize=9)

    # Metrics comparison
    ax7 = fig.add_subplot(gs[2, 2])
    metrics_no = analyze_single_scenario(result_no_refuge, "No Refuge", L)
    metrics_yes = analyze_single_scenario(result_with_refuge, "With Refuge", L)

    metric_names = ['R_final', 'F_final', 'R_cv', 'F_cv']
    x_pos = np.arange(len(metric_names))
    width = 0.35

    values_no = [metrics_no[m] for m in metric_names]
    values_yes = [metrics_yes[m] for m in metric_names]

    ax7.bar(x_pos - width/2, values_no, width, label='No Refuge', alpha=0.7)
    ax7.bar(x_pos + width/2, values_yes, width, label='With Refuge', alpha=0.7)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([m.replace('_', '\n') for m in metric_names], fontsize=9)
    ax7.set_ylabel('Value', fontsize=10)
    ax7.set_title('Key Metrics Comparison', fontweight='bold')
    ax7.legend(loc='best', fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Refuge Effect Analysis', fontsize=15, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
