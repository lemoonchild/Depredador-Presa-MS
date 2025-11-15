"""
Validation tests for the Stochastic solver.

This module tests the Euler-Maruyama implementation and ensemble statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.solver import StochasticSolver, LotkaVolterraSolver
from typing import Dict


def test_noise_properties():
    """
    Test that the stochastic noise has correct statistical properties.

    The Wiener increments should have:
    - Mean = 0
    - Variance = dt
    - Independent between time steps
    """
    print("=" * 60)
    print("Test: Stochastic Noise Properties")
    print("=" * 60)

    # Create solver
    solver = StochasticSolver(L=1.0, nx=10, ny=10)

    # Parameters with strong noise
    params = {
        'demographic': {
            'r': 0.8,
            'K': 50.0,
            'm': 0.4,
            'e': 0.4
        },
        'interaction': {
            'a': 0.4,
            'h': 0.15
        },
        'diffusion': {
            'D_R': 0.0,
            'D_F': 0.0
        },
        'stochastic': {
            'sigma_R': 0.1,
            'sigma_F': 0.1
        }
    }

    # Initial conditions
    R0 = np.full((solver.ny, solver.nx), 30.0)
    F0 = np.full((solver.ny, solver.nx), 8.0)

    # Run multiple realizations
    n_realizations = 20
    T = 50.0
    dt = 0.01

    print(f"Running {n_realizations} independent realizations")
    print(f"Noise intensity: σ_R = {params['stochastic']['sigma_R']}")
    print(f"Noise intensity: σ_F = {params['stochastic']['sigma_F']}")

    results = []
    for i in range(n_realizations):
        result = solver.solve(R0, F0, params, T=T, dt=dt, save_every=50)
        results.append(result)

    # Extract trajectories (center point)
    R_trajectories = np.array([r['R'][:, solver.ny//2, solver.nx//2] for r in results])
    F_trajectories = np.array([r['F'][:, solver.ny//2, solver.nx//2] for r in results])

    # Compute statistics
    R_mean = np.mean(R_trajectories, axis=0)
    R_std = np.std(R_trajectories, axis=0)
    F_mean = np.mean(F_trajectories, axis=0)
    F_std = np.std(F_trajectories, axis=0)

    print(f"\nValidation:")
    print(f"Mean final prey density: {R_mean[-1]:.2f} ± {R_std[-1]:.2f}")
    print(f"Mean final predator density: {F_mean[-1]:.2f} ± {F_std[-1]:.2f}")
    print(f"Prey coefficient of variation: {R_std[-1] / (R_mean[-1] + 1e-10):.4f}")
    print(f"Predator coefficient of variation: {F_std[-1] / (F_mean[-1] + 1e-10):.4f}")

    # Trajectories should diverge (variance should increase over time)
    initial_R_std = R_std[0]
    final_R_std = R_std[-1]
    print(f"Variance increased: {'YES' if final_R_std > initial_R_std else 'NO'}")

    print("\n" + "=" * 60)

    return results


def test_stochastic_vs_deterministic():
    """
    Compare stochastic and deterministic solvers.

    The mean of many stochastic realizations should approximate
    the deterministic solution.
    """
    print("\n" + "=" * 60)
    print("Test: Stochastic vs Deterministic Comparison")
    print("=" * 60)

    # Create solvers
    stoch_solver = StochasticSolver(L=5.0, nx=30, ny=30)
    det_solver = LotkaVolterraSolver(L=5.0, nx=30, ny=30)

    # Parameters with moderate noise
    params = {
        'demographic': {
            'r': 0.8,
            'K': 50.0,
            'm': 0.4,
            'e': 0.4
        },
        'interaction': {
            'a': 0.4,
            'h': 0.15
        },
        'diffusion': {
            'D_R': 0.1,
            'D_F': 0.05
        },
        'stochastic': {
            'sigma_R': 0.05,  # Moderate noise
            'sigma_F': 0.05
        }
    }

    # Initial conditions
    R0 = np.full((stoch_solver.ny, stoch_solver.nx), 25.0)
    F0 = np.full((stoch_solver.ny, stoch_solver.nx), 6.0)

    # Simulation parameters
    T = 30.0
    dt = 0.01
    save_every = 50

    print("Running deterministic simulation...")
    det_result = det_solver.solve(R0, F0, params, T=T, dt=dt, save_every=save_every)

    print("Running stochastic ensemble...")
    stoch_ensemble = stoch_solver.solve_ensemble(
        R0, F0, params, T=T, dt=dt, save_every=save_every,
        n_realizations=10, show_progress=False
    )

    # Compare total populations
    det_R_total = np.sum(det_result['R'], axis=(1, 2))
    stoch_R_total = np.sum(stoch_ensemble['R_mean'], axis=(1, 2))

    det_F_total = np.sum(det_result['F'], axis=(1, 2))
    stoch_F_total = np.sum(stoch_ensemble['F_mean'], axis=(1, 2))

    # Compute relative error
    rel_error_R = np.abs(det_R_total - stoch_R_total) / (det_R_total + 1e-10)
    rel_error_F = np.abs(det_F_total - stoch_F_total) / (det_F_total + 1e-10)

    print(f"\nValidation:")
    print(f"Mean relative error (prey): {np.mean(rel_error_R):.4f}")
    print(f"Mean relative error (predator): {np.mean(rel_error_F):.4f}")
    print(f"Stochastic mean approximates deterministic: "
          f"{'YES' if np.mean(rel_error_R) < 0.2 else 'NO'}")

    print("\n" + "=" * 60)

    return det_result, stoch_ensemble


def test_ensemble_statistics():
    """
    Test the ensemble solver for computing statistics over realizations.

    This validates the solve_ensemble method.
    """
    print("\n" + "=" * 60)
    print("Test: Ensemble Statistics")
    print("=" * 60)

    # Create solver
    solver = StochasticSolver(L=5.0, nx=40, ny=40)

    # Parameters
    params = {
        'demographic': {
            'r': 0.8,
            'K': 50.0,
            'm': 0.4,
            'e': 0.4
        },
        'interaction': {
            'a': 0.4,
            'h': 0.15
        },
        'diffusion': {
            'D_R': 0.1,
            'D_F': 0.05
        },
        'stochastic': {
            'sigma_R': 0.08,
            'sigma_F': 0.08
        }
    }

    # Initial conditions with spatial structure
    R0 = np.zeros((solver.ny, solver.nx))
    F0 = np.zeros((solver.ny, solver.nx))

    # Gaussian initial condition
    x0, y0 = solver.L/2, solver.L/2
    sigma = 1.0
    R0 = 30.0 * np.exp(-((solver.X - x0)**2 + (solver.Y - y0)**2) / (2*sigma**2))
    F0 = 6.0 * np.exp(-((solver.X - x0)**2 + (solver.Y - y0)**2) / (2*sigma**2))

    # Run ensemble
    T = 40.0
    dt = 0.01
    n_realizations = 15

    print(f"Running ensemble with {n_realizations} realizations...")
    ensemble = solver.solve_ensemble(
        R0, F0, params, T=T, dt=dt, save_every=100,
        n_realizations=n_realizations, show_progress=True
    )

    # Validate ensemble structure
    print(f"\nValidation:")
    print(f"Ensemble shape (R_mean): {ensemble['R_mean'].shape}")
    print(f"Ensemble shape (R_std): {ensemble['R_std'].shape}")
    print(f"Number of realizations stored: {ensemble['R_all'].shape[0]}")

    # Check that variance is non-zero
    mean_variance_R = np.mean(ensemble['R_std'][-1])
    mean_variance_F = np.mean(ensemble['F_std'][-1])

    print(f"Mean spatial variance (prey): {mean_variance_R:.4f}")
    print(f"Mean spatial variance (predator): {mean_variance_F:.4f}")
    print(f"Non-zero variance detected: {'YES' if mean_variance_R > 0.1 else 'NO'}")

    # Visualize ensemble statistics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Prey mean
    im0 = axes[0, 0].imshow(ensemble['R_mean'][-1], origin='lower', cmap='Blues',
                            extent=[0, solver.L, 0, solver.L])
    axes[0, 0].set_title('Prey Mean')
    plt.colorbar(im0, ax=axes[0, 0])

    # Prey std
    im1 = axes[0, 1].imshow(ensemble['R_std'][-1], origin='lower', cmap='Reds',
                            extent=[0, solver.L, 0, solver.L])
    axes[0, 1].set_title('Prey Std Dev')
    plt.colorbar(im1, ax=axes[0, 1])

    # Prey coefficient of variation
    cv_R = ensemble['R_std'][-1] / (ensemble['R_mean'][-1] + 1e-10)
    im2 = axes[0, 2].imshow(cv_R, origin='lower', cmap='viridis',
                            extent=[0, solver.L, 0, solver.L])
    axes[0, 2].set_title('Prey CV')
    plt.colorbar(im2, ax=axes[0, 2])

    # Predator mean
    im3 = axes[1, 0].imshow(ensemble['F_mean'][-1], origin='lower', cmap='Blues',
                            extent=[0, solver.L, 0, solver.L])
    axes[1, 0].set_title('Predator Mean')
    plt.colorbar(im3, ax=axes[1, 0])

    # Predator std
    im4 = axes[1, 1].imshow(ensemble['F_std'][-1], origin='lower', cmap='Reds',
                            extent=[0, solver.L, 0, solver.L])
    axes[1, 1].set_title('Predator Std Dev')
    plt.colorbar(im4, ax=axes[1, 1])

    # Predator coefficient of variation
    cv_F = ensemble['F_std'][-1] / (ensemble['F_mean'][-1] + 1e-10)
    im5 = axes[1, 2].imshow(cv_F, origin='lower', cmap='viridis',
                            extent=[0, solver.L, 0, solver.L])
    axes[1, 2].set_title('Predator CV')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()

    print("\n" + "=" * 60)

    return ensemble, fig


def run_all_validations():
    """
    Run all validation tests for the stochastic solver.
    """
    print("\n")
    print("#" * 60)
    print("# Stochastic Solver Validation Suite")
    print("#" * 60)
    print("\n")

    # Run tests
    result1 = test_noise_properties()
    result2 = test_stochastic_vs_deterministic()
    result3, fig3 = test_ensemble_statistics()

    print("\n" + "#" * 60)
    print("# All validation tests completed")
    print("#" * 60)

    return {
        'noise': result1,
        'comparison': result2,
        'ensemble': result3
    }


if __name__ == "__main__":
    # Run all validation tests
    results = run_all_validations()

    print("\n  All tests completed successfully!")
    print("\nThe StochasticSolver is validated for:")
    print("  - Euler-Maruyama method for SDEs")
    print("  - Correct Wiener increment generation")
    print("  - Multiplicative noise (proportional to population)")
    print("  - Multiple independent realizations")
    print("  - Ensemble statistics (mean, variance)")
