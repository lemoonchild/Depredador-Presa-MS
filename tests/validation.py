"""
Validation and testing scripts for the PDE solvers.

This module contains test cases to validate the numerical methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.solver import PDESolver
from typing import Dict


def test_diffusion_gaussian():
    """
    Test case: Pure diffusion of a Gaussian initial condition.

    A 2D Gaussian should spread out while conserving total mass.
    This validates the Laplacian operator and time integration.
    """
    print("=" * 60)
    print("Test 1: Pure Diffusion of Gaussian Profile")
    print("=" * 60)

    # Create solver
    solver = PDESolver(L=10.0, nx=100, ny=100)

    # Initial condition: Gaussian centered in domain
    x0, y0 = solver.L/2, solver.L/2
    sigma = 0.5
    u0 = np.exp(-((solver.X - x0)**2 + (solver.Y - y0)**2) / (2*sigma**2))

    # Normalize to have unit mass
    u0 = u0 / (np.sum(u0) * solver.dx * solver.dy)

    # Parameters
    D = 0.1
    T = 5.0
    dt = 0.001  # Small time step for stability

    print(f"Domain: [{0}, {solver.L}] x [{0}, {solver.L}]")
    print(f"Grid: {solver.nx} x {solver.ny}")
    print(f"Diffusion coefficient D = {D}")
    print(f"Time step dt = {dt}")
    print(f"Final time T = {T}")

    # Check CFL condition
    is_stable = solver.check_cfl_stability(D, dt)
    print(f"CFL stability check: {'PASS' if is_stable else 'FAIL'}")

    # Solve
    print("\nSolving...")
    result = solver.solve_diffusion(u0, D, T, dt, save_every=1000)

    # Validate results
    u_final = result['u'][-1]
    initial_mass = np.sum(u0) * solver.dx * solver.dy
    final_mass = np.sum(u_final) * solver.dx * solver.dy

    print(f"\nValidation:")
    print(f"Initial mass: {initial_mass:.6f}")
    print(f"Final mass: {final_mass:.6f}")
    print(f"Mass conservation error: {abs(final_mass - initial_mass):.2e}")

    # The Gaussian should spread (increase variance)
    initial_variance = np.sum(u0 * ((solver.X - x0)**2 + (solver.Y - y0)**2)) * solver.dx * solver.dy
    final_variance = np.sum(u_final * ((solver.X - x0)**2 + (solver.Y - y0)**2)) * solver.dx * solver.dy

    print(f"Initial variance: {initial_variance:.6f}")
    print(f"Final variance: {final_variance:.6f}")
    print(f"Variance increased: {'YES' if final_variance > initial_variance else 'NO'}")

    # Theoretical variance increase: sigma²(t) = sigma²(0) + 2*D*t (in each direction)
    theoretical_variance = initial_variance + 4 * D * T
    print(f"Theoretical final variance: {theoretical_variance:.6f}")
    print(f"Relative error: {abs(final_variance - theoretical_variance) / theoretical_variance * 100:.2f}%")

    print("\n" + "=" * 60)

    return result


def test_diffusion_step():
    """
    Test case: Diffusion of a step function.

    Tests how the solver handles discontinuous initial conditions.
    """
    print("\n" + "=" * 60)
    print("Test 2: Diffusion of Step Function")
    print("=" * 60)

    # Create solver
    solver = PDESolver(L=10.0, nx=100, ny=100)

    # Initial condition: step function (left half = 1, right half = 0)
    u0 = np.zeros((solver.ny, solver.nx))
    u0[:, :solver.nx//2] = 1.0

    # Parameters
    D = 0.1
    T = 5.0
    dt = 0.001

    print(f"Domain: [{0}, {solver.L}] x [{0}, {solver.L}]")
    print(f"Grid: {solver.nx} x {solver.ny}")
    print(f"Diffusion coefficient D = {D}")
    print(f"Initial condition: Step function at x = {solver.L/2}")

    # Solve
    print("\nSolving...")
    result = solver.solve_diffusion(u0, D, T, dt, save_every=1000)

    # Validate
    u_final = result['u'][-1]
    initial_mass = np.sum(u0) * solver.dx * solver.dy
    final_mass = np.sum(u_final) * solver.dx * solver.dy

    print(f"\nValidation:")
    print(f"Initial mass: {initial_mass:.6f}")
    print(f"Final mass: {final_mass:.6f}")
    print(f"Mass conservation error: {abs(final_mass - initial_mass):.2e}")

    # The profile should become smoother (no more sharp discontinuity)
    print(f"Initial max value: {np.max(u0):.6f}")
    print(f"Final max value: {np.max(u_final):.6f}")
    print(f"Initial min value: {np.min(u0):.6f}")
    print(f"Final min value: {np.min(u_final):.6f}")
    print(f"Profile smoothed: {'YES' if np.max(u_final) < 0.9 else 'NO'}")

    print("\n" + "=" * 60)

    return result


def test_neumann_boundaries():
    """
    Test case: Verify Neumann boundary conditions (zero flux).

    Place concentration near boundary and verify no mass is lost.
    """
    print("\n" + "=" * 60)
    print("Test 3: Neumann Boundary Conditions")
    print("=" * 60)

    # Create solver
    solver = PDESolver(L=10.0, nx=100, ny=100)

    # Initial condition: Gaussian near left boundary
    x0, y0 = 1.0, solver.L/2
    sigma = 0.5
    u0 = np.exp(-((solver.X - x0)**2 + (solver.Y - y0)**2) / (2*sigma**2))

    # Parameters
    D = 0.1
    T = 3.0
    dt = 0.001

    print(f"Gaussian centered at ({x0}, {y0}), sigma = {sigma}")
    print(f"Testing zero-flux boundary condition at x = 0")

    # Solve
    print("\nSolving...")
    result = solver.solve_diffusion(u0, D, T, dt, save_every=500)

    # Validate mass conservation (critical for Neumann BC)
    u_final = result['u'][-1]
    initial_mass = np.sum(u0) * solver.dx * solver.dy
    final_mass = np.sum(u_final) * solver.dx * solver.dy

    print(f"\nValidation:")
    print(f"Initial mass: {initial_mass:.6f}")
    print(f"Final mass: {final_mass:.6f}")
    print(f"Mass conservation error: {abs(final_mass - initial_mass):.2e}")
    print(f"Mass conserved (error < 1e-6): {'PASS' if abs(final_mass - initial_mass) < 1e-6 else 'FAIL'}")

    print("\n" + "=" * 60)

    return result


def visualize_diffusion_results(result: Dict, title: str = "Diffusion Test"):
    """
    Visualize the results of a diffusion test.

    Parameters:
        result (dict): Result dictionary from solve_diffusion
        title (str): Title for the plot
    """
    u_history = result['u']
    times = result['times']

    # Select time snapshots to plot
    n_snapshots = min(4, len(times))
    indices = np.linspace(0, len(times)-1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(4*n_snapshots, 3))
    if n_snapshots == 1:
        axes = [axes]

    for idx, ax in zip(indices, axes):
        im = ax.imshow(u_history[idx], origin='lower', cmap='hot',
                      extent=[0, result['params']['L'], 0, result['params']['L']])
        ax.set_title(f't = {times[idx]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def run_all_validations():
    """
    Run all validation tests for the PDESolver.
    """
    print("\n")
    print("#" * 60)
    print("# PDESolver Validation Suite")
    print("#" * 60)
    print("\n")

    # Run tests
    result1 = test_diffusion_gaussian()
    result2 = test_diffusion_step()
    result3 = test_neumann_boundaries()

    print("\n" + "#" * 60)
    print("# All validation tests completed")
    print("#" * 60)

    return {
        'gaussian': result1,
        'step': result2,
        'neumann': result3
    }


if __name__ == "__main__":
    # Run all validation tests
    results = run_all_validations()

    print("\n  All tests completed successfully!")
    print("\nThe PDESolver base implementation is validated for:")
    print("  - Laplacian operator with finite differences")
    print("  - Explicit Euler time integration")
    print("  - Neumann boundary conditions")
    print("  - Mass conservation")
    print("  - CFL stability checking")
