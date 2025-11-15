"""
Complete example demonstrating all solver capabilities.

This script shows how to use:
- PDESolver: Base diffusion solver
- LotkaVolterraSolver: Reaction-diffusion with predator-prey dynamics
- StochasticSolver: Stochastic version with environmental noise

Author: Persona 1 - Core Solver & Métodos Numéricos
"""

import numpy as np
import matplotlib.pyplot as plt
from src.solver import PDESolver, LotkaVolterraSolver, StochasticSolver
from src.utils import (
    check_stability_comprehensive,
    save_results,
    load_results,
    compute_total_population
)


def example_1_pure_diffusion():
    """
    Example 1: Pure diffusion (no reaction terms).

    Demonstrates the base PDESolver with a Gaussian initial condition.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Pure Diffusion")
    print("=" * 70)

    # Create solver
    solver = PDESolver(L=10.0, nx=80, ny=80)

    # Initial condition: Gaussian blob
    x0, y0 = 5.0, 5.0
    sigma = 0.8
    u0 = 100.0 * np.exp(-((solver.X - x0)**2 + (solver.Y - y0)**2) / (2*sigma**2))

    # Solve
    print("Solving pure diffusion equation...")
    result = solver.solve_diffusion(u0, D=0.1, T=20.0, dt=0.005, save_every=200)

    print(f"✓ Simulation complete")
    print(f"  Final time: {result['times'][-1]:.2f}")
    print(f"  Number of saved snapshots: {len(result['times'])}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    times_to_plot = [0, len(result['times'])//2, -1]
    titles = ['Initial', 'Middle', 'Final']

    for ax, idx, title in zip(axes, times_to_plot, titles):
        im = ax.imshow(result['u'][idx], origin='lower', cmap='hot',
                      extent=[0, solver.L, 0, solver.L])
        ax.set_title(f'{title} (t={result["times"][idx]:.2f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)

    fig.suptitle('Pure Diffusion: Gaussian Spreading', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return result, fig


def example_2_lotka_volterra_deterministic():
    """
    Example 2: Deterministic Lotka-Volterra system.

    Shows predator-prey dynamics with spatial diffusion.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Deterministic Lotka-Volterra Predator-Prey")
    print("=" * 70)

    # Create solver
    solver = LotkaVolterraSolver(L=10.0, nx=60, ny=60)

    # Model parameters (ecological)
    params = {
        'demographic': {
            'r': 1.0,     # Prey growth rate
            'K': 100.0,   # Carrying capacity
            'm': 0.5,     # Predator mortality
            'e': 0.3      # Conversion efficiency
        },
        'interaction': {
            'a': 0.5,     # Attack rate
            'h': 0.1      # Handling time
        },
        'diffusion': {
            'D_R': 0.1,   # Prey diffusion
            'D_F': 0.05   # Predator diffusion (slower)
        }
    }

    # Initial conditions: prey and predators in center
    x0, y0 = 5.0, 5.0
    sigma_R = 1.5
    sigma_F = 1.0

    R0 = 50.0 * np.exp(-((solver.X - x0)**2 + (solver.Y - y0)**2) / (2*sigma_R**2))
    F0 = 10.0 * np.exp(-((solver.X - x0)**2 + (solver.Y - y0)**2) / (2*sigma_F**2))

    # Check stability
    print("\nChecking numerical stability...")
    dt = 0.01
    is_stable, _ = check_stability_comprehensive(params, dt, solver.dx, solver.dy, verbose=False)
    print(f"Stability check: {'PASS ✓' if is_stable else 'FAIL ✗'}")

    # Solve
    print("Solving Lotka-Volterra system...")
    result = solver.solve(R0, F0, params, T=100.0, dt=dt, save_every=200)

    # Compute total populations over time
    R_total = np.array([compute_total_population(result['R'][i], solver.dx, solver.dy)
                        for i in range(len(result['times']))])
    F_total = np.array([compute_total_population(result['F'][i], solver.dx, solver.dy)
                        for i in range(len(result['times']))])

    print(f"✓ Simulation complete")
    print(f"  Initial prey population: {R_total[0]:.2f}")
    print(f"  Final prey population: {R_total[-1]:.2f}")
    print(f"  Initial predator population: {F_total[0]:.2f}")
    print(f"  Final predator population: {F_total[-1]:.2f}")

    # Visualize
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Spatial distributions at different times
    times_idx = [0, len(result['times'])//2, -1]

    for i, idx in enumerate(times_idx):
        # Prey
        ax_r = fig.add_subplot(gs[0, i])
        im_r = ax_r.imshow(result['R'][idx], origin='lower', cmap='Blues',
                          extent=[0, solver.L, 0, solver.L])
        ax_r.set_title(f'Prey at t={result["times"][idx]:.1f}')
        plt.colorbar(im_r, ax=ax_r)

        # Predator
        ax_f = fig.add_subplot(gs[1, i])
        im_f = ax_f.imshow(result['F'][idx], origin='lower', cmap='Reds',
                          extent=[0, solver.L, 0, solver.L])
        ax_f.set_title(f'Predator at t={result["times"][idx]:.1f}')
        plt.colorbar(im_f, ax=ax_f)

    # Time series
    ax_ts = fig.add_subplot(gs[2, :2])
    ax_ts.plot(result['times'], R_total, 'b-', linewidth=2, label='Prey')
    ax_ts.plot(result['times'], F_total, 'r-', linewidth=2, label='Predator')
    ax_ts.set_xlabel('Time')
    ax_ts.set_ylabel('Total Population')
    ax_ts.set_title('Population Dynamics')
    ax_ts.legend()
    ax_ts.grid(True, alpha=0.3)

    # Phase portrait
    ax_pp = fig.add_subplot(gs[2, 2])
    ax_pp.plot(R_total, F_total, 'k-', linewidth=1.5, alpha=0.7)
    ax_pp.plot(R_total[0], F_total[0], 'go', markersize=10, label='Start')
    ax_pp.plot(R_total[-1], F_total[-1], 'ro', markersize=10, label='End')
    ax_pp.set_xlabel('Prey Population')
    ax_pp.set_ylabel('Predator Population')
    ax_pp.set_title('Phase Portrait')
    ax_pp.legend()
    ax_pp.grid(True, alpha=0.3)

    fig.suptitle('Deterministic Lotka-Volterra Dynamics', fontsize=16, fontweight='bold')

    # Save results
    print("\nSaving results...")
    save_results(result, 'data/results/example2_lotka_volterra.npz',
                metadata={'example': 'Deterministic LV', 'type': 'reaction-diffusion'})
    print("✓ Results saved to: data/results/example2_lotka_volterra.npz")

    return result, fig


def example_3_stochastic_ensemble():
    """
    Example 3: Stochastic Lotka-Volterra with ensemble averaging.

    Demonstrates environmental noise and statistical analysis.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Stochastic Lotka-Volterra with Ensemble")
    print("=" * 70)

    # Create solver
    solver = StochasticSolver(L=8.0, nx=50, ny=50)

    # Parameters with moderate noise
    params = {
        'demographic': {
            'r': 0.8,
            'K': 60.0,
            'm': 0.4,
            'e': 0.35
        },
        'interaction': {
            'a': 0.45,
            'h': 0.12
        },
        'diffusion': {
            'D_R': 0.12,
            'D_F': 0.06
        },
        'stochastic': {
            'sigma_R': 0.08,  # 8% noise for prey
            'sigma_F': 0.08   # 8% noise for predator
        }
    }

    # Initial conditions
    R0 = np.full((solver.ny, solver.nx), 30.0)
    F0 = np.full((solver.ny, solver.nx), 7.0)

    # Add small spatial perturbation
    R0 += 5.0 * np.random.randn(solver.ny, solver.nx)
    F0 += 2.0 * np.random.randn(solver.ny, solver.nx)
    R0 = np.maximum(R0, 0.0)
    F0 = np.maximum(F0, 0.0)

    print(f"Noise intensity: σ_R = {params['stochastic']['sigma_R']}")
    print(f"Noise intensity: σ_F = {params['stochastic']['sigma_F']}")

    # Run ensemble
    print("\nRunning stochastic ensemble...")
    n_realizations = 20
    ensemble = solver.solve_ensemble(
        R0, F0, params, T=60.0, dt=0.01, save_every=100,
        n_realizations=n_realizations, show_progress=True
    )

    # Compute total populations for each realization
    R_totals = np.array([[compute_total_population(ensemble['R_all'][r, t], solver.dx, solver.dy)
                         for t in range(len(ensemble['times']))]
                        for r in range(n_realizations)])

    F_totals = np.array([[compute_total_population(ensemble['F_all'][r, t], solver.dx, solver.dy)
                         for t in range(len(ensemble['times']))]
                        for r in range(n_realizations)])

    # Statistics
    R_mean_total = np.mean(R_totals, axis=0)
    R_std_total = np.std(R_totals, axis=0)
    F_mean_total = np.mean(F_totals, axis=0)
    F_std_total = np.std(F_totals, axis=0)

    print(f"✓ Ensemble complete")
    print(f"  Number of realizations: {n_realizations}")
    print(f"  Final prey (mean ± std): {R_mean_total[-1]:.2f} ± {R_std_total[-1]:.2f}")
    print(f"  Final predator (mean ± std): {F_mean_total[-1]:.2f} ± {F_std_total[-1]:.2f}")

    # Visualize
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Show individual realizations
    ax1 = fig.add_subplot(gs[0, 0])
    for r in range(min(10, n_realizations)):
        ax1.plot(ensemble['times'], R_totals[r], 'b-', alpha=0.3, linewidth=1)
    ax1.plot(ensemble['times'], R_mean_total, 'b-', linewidth=3, label='Mean')
    ax1.fill_between(ensemble['times'],
                     R_mean_total - R_std_total,
                     R_mean_total + R_std_total,
                     alpha=0.3, color='blue', label='±1 std')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Prey Population')
    ax1.set_title('Prey: Individual Realizations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for r in range(min(10, n_realizations)):
        ax2.plot(ensemble['times'], F_totals[r], 'r-', alpha=0.3, linewidth=1)
    ax2.plot(ensemble['times'], F_mean_total, 'r-', linewidth=3, label='Mean')
    ax2.fill_between(ensemble['times'],
                     F_mean_total - F_std_total,
                     F_mean_total + F_std_total,
                     alpha=0.3, color='red', label='±1 std')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Predator Population')
    ax2.set_title('Predator: Individual Realizations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mean spatial distribution
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(ensemble['R_mean'][-1], origin='lower', cmap='Blues',
                    extent=[0, solver.L, 0, solver.L])
    ax3.set_title('Mean Prey Distribution (final)')
    plt.colorbar(im3, ax=ax3)

    # Spatial standard deviation
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(ensemble['R_std'][-1], origin='lower', cmap='Reds',
                    extent=[0, solver.L, 0, solver.L])
    ax4.set_title('Prey Std Dev (final)')
    plt.colorbar(im4, ax=ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(ensemble['F_std'][-1], origin='lower', cmap='Reds',
                    extent=[0, solver.L, 0, solver.L])
    ax5.set_title('Predator Std Dev (final)')
    plt.colorbar(im5, ax=ax5)

    # Coefficient of variation
    ax6 = fig.add_subplot(gs[1, 2])
    cv = R_std_total / (R_mean_total + 1e-10)
    ax6.plot(ensemble['times'], cv, 'g-', linewidth=2)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Coefficient of Variation')
    ax6.set_title('Prey Population Variability')
    ax6.grid(True, alpha=0.3)

    fig.suptitle(f'Stochastic Dynamics (n={n_realizations} realizations)',
                fontsize=16, fontweight='bold')

    return ensemble, fig


def main():
    """
    Run all examples.
    """
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  COMPLETE SOLVER DEMONSTRATION".center(68) + "#")
    print("#" + "  Predator-Prey Reaction-Diffusion System".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # Run examples
    result1, fig1 = example_1_pure_diffusion()
    result2, fig2 = example_2_lotka_volterra_deterministic()
    result3, fig3 = example_3_stochastic_ensemble()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nSummary:")
    print("  ✓ Example 1: Pure diffusion solver")
    print("  ✓ Example 2: Deterministic Lotka-Volterra")
    print("  ✓ Example 3: Stochastic ensemble simulation")

    print("\nGenerated files:")
    print("  - data/results/example2_lotka_volterra.npz")

    print("\nTo view plots, the figures are stored in memory.")
    print("Add 'plt.show()' at the end if running interactively.")

    return result1, result2, result3


if __name__ == "__main__":
    results = main()

    # Uncomment to display plots interactively
    # plt.show()
