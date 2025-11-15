"""
Demonstration of optimizations and stability features.

This script showcases:
- CFL stability checking
- Automatic time step suggestion
- Result saving and loading
- Performance optimization
"""

import numpy as np
import time
from src.solver import StochasticSolver, LotkaVolterraSolver
from src.utils import (
    check_stability_comprehensive,
    suggest_stable_timestep,
    save_results,
    load_results,
    compute_total_population,
    compute_center_of_mass,
    compute_spatial_variance
)


def demo_stability_checking():
    """
    Demonstrate comprehensive stability checking.
    """
    print("\n" + "=" * 60)
    print("Demo: Comprehensive Stability Checking")
    print("=" * 60)

    params = {
        'demographic': {
            'r': 1.0,
            'K': 100.0,
            'm': 0.5,
            'e': 0.3
        },
        'interaction': {
            'a': 0.5,
            'h': 0.1
        },
        'diffusion': {
            'D_R': 0.1,
            'D_F': 0.05
        }
    }

    # Grid parameters
    L = 10.0
    nx = 100
    dx = L / (nx - 1)
    dy = dx

    # Test different time steps
    dt_values = [0.1, 0.01, 0.001]

    for dt in dt_values:
        print(f"\nTesting dt = {dt}")
        is_stable, details = check_stability_comprehensive(
            params, dt, dx, dy, verbose=True
        )

        if not is_stable:
            # Suggest better time step
            D_max = max(params['diffusion']['D_R'], params['diffusion']['D_F'])
            dt_suggested = suggest_stable_timestep(D_max, dx, dy)
            print(f"\nSuggested stable time step: dt = {dt_suggested:.6f}")


def demo_save_load():
    """
    Demonstrate saving and loading results.
    """
    print("\n" + "=" * 60)
    print("Demo: Saving and Loading Results")
    print("=" * 60)

    # Create solver
    solver = LotkaVolterraSolver(L=5.0, nx=40, ny=40)

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
        }
    }

    # Initial conditions
    R0 = np.full((solver.ny, solver.nx), 25.0)
    F0 = np.full((solver.ny, solver.nx), 6.0)

    # Run simulation
    print("\nRunning simulation...")
    start_time = time.time()
    result = solver.solve(R0, F0, params, T=20.0, dt=0.01, save_every=50)
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")

    # Save in different formats
    print("\nSaving results in different formats...")

    metadata = {
        'description': 'Demo simulation',
        'solver': 'LotkaVolterraSolver',
        'computation_time': elapsed
    }

    # NPZ format
    save_results(result, 'data/results/demo_result.npz', format='npz',
                compress=True, metadata=metadata)
    print("  Saved as NPZ (compressed)")

    # Pickle format
    save_results(result, 'data/results/demo_result.pkl', format='pickle',
                metadata=metadata)
    print("  Saved as Pickle")

    # Load and verify
    print("\nLoading results...")
    loaded_npz = load_results('data/results/demo_result.npz')
    loaded_pkl = load_results('data/results/demo_result.pkl')

    print(f"  NPZ loaded successfully: {loaded_npz['R'].shape}")
    print(f"  Pickle loaded successfully: {loaded_pkl['R'].shape}")

    # Verify data integrity
    print("\nVerifying data integrity...")
    assert np.allclose(result['R'], loaded_npz['R']), "NPZ data mismatch!"
    assert np.allclose(result['R'], loaded_pkl['R']), "Pickle data mismatch!"
    print("  All data verified successfully")


def demo_spatial_analysis():
    """
    Demonstrate spatial analysis utilities.
    """
    print("\n" + "=" * 60)
    print("Demo: Spatial Analysis Utilities")
    print("=" * 60)

    # Create solver
    solver = LotkaVolterraSolver(L=10.0, nx=50, ny=50)

    # Create initial condition with off-center Gaussian
    x0, y0 = 3.0, 7.0
    sigma = 1.0
    R0 = 50.0 * np.exp(-((solver.X - x0)**2 + (solver.Y - y0)**2) / (2*sigma**2))
    F0 = 10.0 * np.exp(-((solver.X - x0)**2 + (solver.Y - y0)**2) / (2*sigma**2))

    # Parameters with diffusion
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
            'D_R': 0.2,
            'D_F': 0.1
        }
    }

    print(f"\nInitial prey center of mass: ({x0}, {y0})")

    # Simulate
    result = solver.solve(R0, F0, params, T=10.0, dt=0.01, save_every=200)

    # Analyze spatial distribution over time
    print("\nSpatial analysis over time:")
    print("-" * 60)
    print(f"{'Time':>8} {'R_total':>12} {'F_total':>12} {'R_cm_x':>10} {'R_cm_y':>10} {'R_var':>10}")
    print("-" * 60)

    for i, t in enumerate(result['times'][::10]):  # Sample every 10th time point
        R_total = compute_total_population(result['R'][i*10], solver.dx, solver.dy)
        F_total = compute_total_population(result['F'][i*10], solver.dx, solver.dy)

        R_cm_x, R_cm_y = compute_center_of_mass(result['R'][i*10], solver.X, solver.Y)
        _, _, R_var = compute_spatial_variance(result['R'][i*10], solver.X, solver.Y)

        print(f"{t:8.2f} {R_total:12.2f} {F_total:12.2f} {R_cm_x:10.2f} {R_cm_y:10.2f} {R_var:10.4f}")

    print("-" * 60)
    print("\nObservations:")
    print("- Populations diffuse and spread (variance increases)")
    print("- Center of mass may shift due to spatial heterogeneity")


def demo_performance_comparison():
    """
    Compare performance with different grid resolutions.
    """
    print("\n" + "=" * 60)
    print("Demo: Performance Comparison")
    print("=" * 60)

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
        }
    }

    grid_sizes = [20, 40, 60, 80]
    T = 5.0

    print("\nBenchmarking different grid resolutions:")
    print("-" * 60)
    print(f"{'Grid Size':>12} {'Total Points':>15} {'Time (s)':>12} {'Time/Point':>15}")
    print("-" * 60)

    for nx in grid_sizes:
        solver = LotkaVolterraSolver(L=10.0, nx=nx, ny=nx)

        R0 = np.full((solver.ny, solver.nx), 25.0)
        F0 = np.full((solver.ny, solver.nx), 6.0)

        # Adjust dt for stability
        D_max = max(params['diffusion']['D_R'], params['diffusion']['D_F'])
        dt = suggest_stable_timestep(D_max, solver.dx, solver.dy, safety_factor=0.8)

        start_time = time.time()
        result = solver.solve(R0, F0, params, T=T, dt=dt, save_every=100)
        elapsed = time.time() - start_time

        total_points = nx * nx
        time_per_point = elapsed / total_points * 1e6  # microseconds

        print(f"{nx}x{nx:>7} {total_points:15d} {elapsed:12.4f} {time_per_point:12.2f} µs")

    print("-" * 60)
    print("\nNote: Time scales approximately as O(N²) where N is grid size")


def run_all_demos():
    """
    Run all demonstration scripts.
    """
    print("\n")
    print("#" * 60)
    print("# Optimization and Stability Demonstrations")
    print("#" * 60)

    demo_stability_checking()
    demo_save_load()
    demo_spatial_analysis()
    demo_performance_comparison()

    print("\n" + "#" * 60)
    print("# All demonstrations completed")
    print("#" * 60)


if __name__ == "__main__":
    run_all_demos()

    print("\n  All optimization features demonstrated!")
    print("\nKey features:")
    print("  - Comprehensive CFL stability checking")
    print("  - Automatic stable time step suggestion")
    print("  - Multiple save/load formats (NPZ, Pickle, HDF5)")
    print("  - Spatial analysis utilities")
    print("  - Performance benchmarking")
    print("  - Vectorized NumPy operations throughout")
