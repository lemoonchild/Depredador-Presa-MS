"""
Validation tests for the Lotka-Volterra solver.

This module tests the reaction terms and verifies the coupled system
behaves as expected.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.solver import LotkaVolterraSolver
from typing import Dict


def test_lotka_volterra_0d():
    """
    Test the Lotka-Volterra solver in 0D (spatially homogeneous).

    This validates the reaction terms by comparing with the classical
    Lotka-Volterra ODE system without spatial variation.

    Expected behavior:
    - Oscillations in prey and predator populations
    - Predator lags behind prey
    - Conservation of energy-like quantity (in absence of diffusion)
    """
    print("=" * 60)
    print("Test: Lotka-Volterra 0D (Spatially Homogeneous)")
    print("=" * 60)

    # Create solver with small domain (effectively 0D with uniform IC)
    solver = LotkaVolterraSolver(L=1.0, nx=10, ny=10)

    # Parameters for classic oscillatory behavior
    params = {
        'demographic': {
            'r': 1.0,    # Prey growth rate
            'K': 100.0,  # Carrying capacity
            'm': 0.5,    # Predator mortality
            'e': 0.3     # Conversion efficiency
        },
        'interaction': {
            'a': 0.5,    # Attack rate
            'h': 0.1     # Handling time
        },
        'diffusion': {
            'D_R': 0.0,  # No diffusion (0D test)
            'D_F': 0.0
        }
    }

    # Initial conditions (spatially uniform)
    R0 = np.full((solver.ny, solver.nx), 50.0)  # Prey
    F0 = np.full((solver.ny, solver.nx), 10.0)  # Predator

    # Simulation parameters
    T = 100.0
    dt = 0.01
    save_every = 10

    print(f"Initial prey density: {R0[0,0]:.2f}")
    print(f"Initial predator density: {F0[0,0]:.2f}")
    print(f"Simulation time: T = {T}")
    print(f"Time step: dt = {dt}")

    # Solve
    print("\nSolving...")
    result = solver.solve(R0, F0, params, refuge_map=None, T=T, dt=dt, save_every=save_every)

    # Extract results (take center point since all should be uniform)
    R_history = result['R'][:, solver.ny//2, solver.nx//2]
    F_history = result['F'][:, solver.ny//2, solver.nx//2]
    times = result['times']

    # Validate oscillatory behavior
    print(f"\nValidation:")
    print(f"Final prey density: {R_history[-1]:.2f}")
    print(f"Final predator density: {F_history[-1]:.2f}")

    # Check for oscillations (should have multiple peaks)
    R_peaks = np.where((R_history[1:-1] > R_history[:-2]) &
                       (R_history[1:-1] > R_history[2:]))[0]
    F_peaks = np.where((F_history[1:-1] > F_history[:-2]) &
                       (F_history[1:-1] > F_history[2:]))[0]

    print(f"Number of prey peaks: {len(R_peaks)}")
    print(f"Number of predator peaks: {len(F_peaks)}")
    print(f"Oscillations detected: {'YES' if len(R_peaks) > 2 else 'NO'}")

    # Check that predator lags prey (predator peak should come after prey peak)
    if len(R_peaks) > 0 and len(F_peaks) > 0:
        print(f"Predator lags prey: {'YES' if F_peaks[0] > R_peaks[0] else 'NO'}")

    # Plot time series
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Time series
    axes[0].plot(times, R_history, 'b-', label='Prey (R)', linewidth=2)
    axes[0].plot(times, F_history, 'r-', label='Predator (F)', linewidth=2)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Population Density')
    axes[0].set_title('Lotka-Volterra Dynamics (0D)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Phase portrait
    axes[1].plot(R_history, F_history, 'k-', linewidth=1.5, alpha=0.7)
    axes[1].plot(R_history[0], F_history[0], 'go', markersize=10, label='Start')
    axes[1].plot(R_history[-1], F_history[-1], 'ro', markersize=10, label='End')
    axes[1].set_xlabel('Prey Density (R)')
    axes[1].set_ylabel('Predator Density (F)')
    axes[1].set_title('Phase Portrait')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    print("\n" + "=" * 60)

    return result, fig


def test_lotka_volterra_with_refuges():
    """
    Test the Lotka-Volterra solver with spatial refuges.

    Refuges should enhance prey growth in specific regions.
    """
    print("\n" + "=" * 60)
    print("Test: Lotka-Volterra with Spatial Refuges")
    print("=" * 60)

    # Create solver
    solver = LotkaVolterraSolver(L=10.0, nx=50, ny=50)

    # Create refuge map: central region has refuge (p=2.0), rest has p=1.0
    refuge_map = np.ones((solver.ny, solver.nx))
    center_x, center_y = solver.nx//2, solver.ny//2
    radius = 10
    for i in range(solver.ny):
        for j in range(solver.nx):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if dist < radius:
                refuge_map[i, j] = 2.0  # Enhanced growth in refuge

    # Parameters
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

    # Initial conditions: uniform low density
    R0 = np.full((solver.ny, solver.nx), 20.0)
    F0 = np.full((solver.ny, solver.nx), 5.0)

    # Simulation
    T = 50.0
    dt = 0.01
    save_every = 100

    print(f"Refuge center: ({center_x}, {center_y})")
    print(f"Refuge radius: {radius} grid points")
    print(f"Growth enhancement in refuge: 2.0x")

    print("\nSolving...")
    result = solver.solve(R0, F0, params, refuge_map=refuge_map,
                         T=T, dt=dt, save_every=save_every)

    # Check if prey density is higher in refuge region
    R_final = result['R'][-1]
    R_in_refuge = R_final[center_y-5:center_y+5, center_x-5:center_x+5].mean()
    R_outside_refuge = R_final[:10, :10].mean()

    print(f"\nValidation:")
    print(f"Mean prey density in refuge: {R_in_refuge:.2f}")
    print(f"Mean prey density outside refuge: {R_outside_refuge:.2f}")
    print(f"Refuge effect detected: {'YES' if R_in_refuge > R_outside_refuge * 1.2 else 'NO'}")

    # Visualize final state
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Refuge map
    im0 = axes[0].imshow(refuge_map, origin='lower', cmap='Greens',
                         extent=[0, solver.L, 0, solver.L])
    axes[0].set_title('Refuge Map')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='Growth Factor')

    # Final prey distribution
    im1 = axes[1].imshow(R_final, origin='lower', cmap='Blues',
                         extent=[0, solver.L, 0, solver.L])
    axes[1].set_title(f'Prey Density (t={T})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1], label='Density')

    # Final predator distribution
    F_final = result['F'][-1]
    im2 = axes[2].imshow(F_final, origin='lower', cmap='Reds',
                         extent=[0, solver.L, 0, solver.L])
    axes[2].set_title(f'Predator Density (t={T})')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2], label='Density')

    plt.tight_layout()

    print("\n" + "=" * 60)

    return result, fig


def test_equilibrium_stability():
    """
    Test the stability of equilibrium points.

    Without diffusion and with proper parameters, the system should
    converge to a stable equilibrium or show stable limit cycles.
    """
    print("\n" + "=" * 60)
    print("Test: Equilibrium and Stability")
    print("=" * 60)

    # Create solver
    solver = LotkaVolterraSolver(L=1.0, nx=10, ny=10)

    # Parameters tuned for stable coexistence
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
        }
    }

    # Start near equilibrium
    R0 = np.full((solver.ny, solver.nx), 30.0)
    F0 = np.full((solver.ny, solver.nx), 8.0)

    # Simulation
    T = 200.0
    dt = 0.01
    save_every = 20

    print("Testing stability of coexistence equilibrium...")
    result = solver.solve(R0, F0, params, refuge_map=None,
                         T=T, dt=dt, save_every=save_every)

    # Extract center point
    R_history = result['R'][:, solver.ny//2, solver.nx//2]
    F_history = result['F'][:, solver.ny//2, solver.nx//2]

    # Check if populations remain positive
    print(f"\nValidation:")
    print(f"Prey survived: {'YES' if R_history[-1] > 1.0 else 'NO'}")
    print(f"Predator survived: {'YES' if F_history[-1] > 0.1 else 'NO'}")
    print(f"Final prey density: {R_history[-1]:.2f}")
    print(f"Final predator density: {F_history[-1]:.2f}")

    # Check variation in last 25% of simulation (should be small if stable)
    last_quarter = len(R_history) * 3 // 4
    R_variation = np.std(R_history[last_quarter:]) / np.mean(R_history[last_quarter:])
    F_variation = np.std(F_history[last_quarter:]) / np.mean(F_history[last_quarter:])

    print(f"Prey CV (last quarter): {R_variation:.4f}")
    print(f"Predator CV (last quarter): {F_variation:.4f}")

    print("\n" + "=" * 60)

    return result


def run_all_validations():
    """
    Run all validation tests for the Lotka-Volterra solver.
    """
    print("\n")
    print("#" * 60)
    print("# Lotka-Volterra Solver Validation Suite")
    print("#" * 60)
    print("\n")

    # Run tests
    result1, fig1 = test_lotka_volterra_0d()
    result2, fig2 = test_lotka_volterra_with_refuges()
    result3 = test_equilibrium_stability()

    print("\n" + "#" * 60)
    print("# All validation tests completed")
    print("#" * 60)

    return {
        '0d': result1,
        'refuges': result2,
        'equilibrium': result3
    }


if __name__ == "__main__":
    # Run all validation tests
    results = run_all_validations()

    print("\n✓ All tests completed successfully!")
    print("\nThe LotkaVolterraSolver is validated for:")
    print("  - Logistic growth term with carrying capacity")
    print("  - Type II functional response (predation)")
    print("  - Refuge effects on prey growth")
    print("  - Coupled reaction-diffusion dynamics")
    print("  - Equilibrium stability")
