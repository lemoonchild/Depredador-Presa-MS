"""
Solver module for partial differential equations (PDEs).

This module implements numerical methods for solving reaction-diffusion PDEs
using finite difference methods and explicit time-stepping schemes.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable
import warnings


class PDESolver:
    """
    Base solver for 2D reaction-diffusion PDEs using finite differences.

    This solver implements:
    - Spatial discretization using finite differences
    - Explicit Euler method for time integration
    - Laplacian operator with Neumann boundary conditions

    Attributes:
        L (float): Domain size (assumes square domain [0, L] x [0, L])
        nx (int): Number of grid points in x-direction
        ny (int): Number of grid points in y-direction
        dx (float): Spatial step size in x-direction
        dy (float): Spatial step size in y-direction
        x (np.ndarray): x-coordinates of grid points
        y (np.ndarray): y-coordinates of grid points
        X (np.ndarray): 2D meshgrid of x-coordinates
        Y (np.ndarray): 2D meshgrid of y-coordinates
    """

    def __init__(self, L: float = 10.0, nx: int = 100, ny: int = 100):
        """
        Initialize the PDE solver with domain and grid parameters.

        Parameters:
            L (float): Domain size (square domain [0, L] x [0, L])
            nx (int): Number of grid points in x-direction
            ny (int): Number of grid points in y-direction
        """
        self.L = L
        self.nx = nx
        self.ny = ny

        # Compute spatial steps
        self.dx = L / (nx - 1)
        self.dy = L / (ny - 1)

        # Create spatial grid
        self.x = np.linspace(0, L, nx)
        self.y = np.linspace(0, L, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the Laplacian of a 2D field using finite differences.

        Uses central differences for interior points and Neumann boundary
        conditions (zero derivative) at the boundaries.

        The Laplacian is approximated as:
        ∇²u ≈ (u[i+1,j] - 2u[i,j] + u[i-1,j]) / dx² +
              (u[i,j+1] - 2u[i,j] + u[i,j-1]) / dy²

        Parameters:
            u (np.ndarray): 2D field of shape (ny, nx)

        Returns:
            np.ndarray: Laplacian of u with same shape as input
        """
        lap = np.zeros_like(u)

        # Interior points using central differences
        lap[1:-1, 1:-1] = (
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dx**2 +
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.dy**2
        )

        # Neumann boundary conditions (zero flux)
        # Left boundary (x = 0)
        lap[1:-1, 0] = (
            (2*u[1:-1, 1] - 2*u[1:-1, 0]) / self.dx**2 +
            (u[2:, 0] - 2*u[1:-1, 0] + u[:-2, 0]) / self.dy**2
        )

        # Right boundary (x = L)
        lap[1:-1, -1] = (
            (2*u[1:-1, -2] - 2*u[1:-1, -1]) / self.dx**2 +
            (u[2:, -1] - 2*u[1:-1, -1] + u[:-2, -1]) / self.dy**2
        )

        # Bottom boundary (y = 0)
        lap[0, 1:-1] = (
            (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2]) / self.dx**2 +
            (2*u[1, 1:-1] - 2*u[0, 1:-1]) / self.dy**2
        )

        # Top boundary (y = L)
        lap[-1, 1:-1] = (
            (u[-1, 2:] - 2*u[-1, 1:-1] + u[-1, :-2]) / self.dx**2 +
            (2*u[-2, 1:-1] - 2*u[-1, 1:-1]) / self.dy**2
        )

        # Corners (average of adjacent boundary conditions)
        lap[0, 0] = (
            (2*u[0, 1] - 2*u[0, 0]) / self.dx**2 +
            (2*u[1, 0] - 2*u[0, 0]) / self.dy**2
        )
        lap[0, -1] = (
            (2*u[0, -2] - 2*u[0, -1]) / self.dx**2 +
            (2*u[1, -1] - 2*u[0, -1]) / self.dy**2
        )
        lap[-1, 0] = (
            (2*u[-1, 1] - 2*u[-1, 0]) / self.dx**2 +
            (2*u[-2, 0] - 2*u[-1, 0]) / self.dy**2
        )
        lap[-1, -1] = (
            (2*u[-1, -2] - 2*u[-1, -1]) / self.dx**2 +
            (2*u[-2, -1] - 2*u[-1, -1]) / self.dy**2
        )

        return lap

    def check_cfl_stability(self, D: float, dt: float) -> bool:
        """
        Check CFL (Courant-Friedrichs-Lewy) stability condition.

        For explicit Euler method with diffusion, the stability condition is:
        D * dt / (dx²) + D * dt / (dy²) <= 0.5

        Parameters:
            D (float): Diffusion coefficient
            dt (float): Time step size

        Returns:
            bool: True if stable, False otherwise
        """
        cfl_x = D * dt / self.dx**2
        cfl_y = D * dt / self.dy**2
        cfl_total = cfl_x + cfl_y

        return cfl_total <= 0.5

    def solve_diffusion(
        self,
        u0: np.ndarray,
        D: float,
        T: float,
        dt: float,
        save_every: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Solve the pure diffusion equation: ∂u/∂t = D∇²u

        Uses explicit Euler method for time integration.

        Parameters:
            u0 (np.ndarray): Initial condition (ny, nx)
            D (float): Diffusion coefficient
            T (float): Final time
            dt (float): Time step size
            save_every (int): Save solution every N time steps

        Returns:
            dict: Dictionary containing:
                - 'u': Solution array of shape (n_saved_steps, ny, nx)
                - 'times': Time points where solution was saved
                - 'params': Dictionary of parameters used
        """
        # Check stability
        if not self.check_cfl_stability(D, dt):
            warnings.warn(
                f"CFL condition violated! D*dt/dx²={D*dt/self.dx**2:.4f}, "
                f"D*dt/dy²={D*dt/self.dy**2:.4f}. "
                f"Sum should be <= 0.5 for stability."
            )

        # Initialize
        n_steps = int(T / dt)
        n_saved = (n_steps // save_every) + 1

        u_history = np.zeros((n_saved, self.ny, self.nx))
        times = np.zeros(n_saved)

        u = u0.copy()
        u_history[0] = u
        times[0] = 0.0

        save_counter = 1

        # Time integration loop
        for n in range(n_steps):
            t = n * dt

            # Compute Laplacian
            lap_u = self.laplacian(u)

            # Explicit Euler step
            u = u + dt * D * lap_u

            # Ensure non-negative values
            u = np.maximum(u, 0.0)

            # Save if needed
            if (n + 1) % save_every == 0 and save_counter < n_saved:
                u_history[save_counter] = u
                times[save_counter] = t + dt
                save_counter += 1

        # Save final state if not already saved
        if save_counter < n_saved:
            u_history[save_counter] = u
            times[save_counter] = T

        return {
            'u': u_history,
            'times': times,
            'params': {
                'D': D,
                'T': T,
                'dt': dt,
                'L': self.L,
                'nx': self.nx,
                'ny': self.ny
            }
        }

    def solve(
        self,
        R0: np.ndarray,
        F0: np.ndarray,
        params: Dict,
        refuge_map: Optional[np.ndarray] = None,
        T: float = 200.0,
        dt: float = 0.01,
        save_every: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Solve the coupled reaction-diffusion system for prey (R) and predator (F).

        This is the main interface method that will be extended in derived classes.
        Base implementation only handles diffusion (no reaction terms).

        Equations:
        ∂R/∂t = D_R ∇²R + reaction_R(R, F)
        ∂F/∂t = D_F ∇²F + reaction_F(R, F)

        Parameters:
            R0 (np.ndarray): Initial prey distribution (ny, nx)
            F0 (np.ndarray): Initial predator distribution (ny, nx)
            params (dict): Model parameters including:
                - 'diffusion': {'D_R': float, 'D_F': float}
                - 'demographic': {'r': float, 'K': float, 'm': float, 'e': float}
                - 'interaction': {'a': float, 'h': float}
            refuge_map (np.ndarray, optional): Refuge function p(x,y) in [0,1]
            T (float): Final time
            dt (float): Time step size
            save_every (int): Save solution every N time steps

        Returns:
            dict: Dictionary containing:
                - 'R': Prey solution array (n_saved_steps, ny, nx)
                - 'F': Predator solution array (n_saved_steps, ny, nx)
                - 'times': Time points
                - 'params': Parameters used
        """
        # Extract diffusion coefficients
        D_R = params.get('diffusion', {}).get('D_R', 0.1)
        D_F = params.get('diffusion', {}).get('D_F', 0.05)

        # Check stability for both species
        if not self.check_cfl_stability(max(D_R, D_F), dt):
            warnings.warn(
                f"CFL condition may be violated for dt={dt}. "
                f"Consider reducing time step."
            )

        # Initialize
        n_steps = int(T / dt)
        n_saved = (n_steps // save_every) + 1

        R_history = np.zeros((n_saved, self.ny, self.nx))
        F_history = np.zeros((n_saved, self.ny, self.nx))
        times = np.zeros(n_saved)

        R = R0.copy()
        F = F0.copy()

        R_history[0] = R
        F_history[0] = F
        times[0] = 0.0

        save_counter = 1

        # Time integration loop
        for n in range(n_steps):
            t = n * dt

            # Compute diffusion terms
            lap_R = self.laplacian(R)
            lap_F = self.laplacian(F)

            # Compute reaction terms (to be overridden in derived classes)
            reaction_R = self._reaction_prey(R, F, params, refuge_map)
            reaction_F = self._reaction_predator(R, F, params)

            # Explicit Euler step
            R = R + dt * (D_R * lap_R + reaction_R)
            F = F + dt * (D_F * lap_F + reaction_F)

            # Ensure non-negative values
            R = np.maximum(R, 0.0)
            F = np.maximum(F, 0.0)

            # Save if needed
            if (n + 1) % save_every == 0 and save_counter < n_saved:
                R_history[save_counter] = R
                F_history[save_counter] = F
                times[save_counter] = t + dt
                save_counter += 1

        # Save final state if not already saved
        if save_counter < n_saved:
            R_history[save_counter] = R
            F_history[save_counter] = F
            times[save_counter] = T

        return {
            'R': R_history,
            'F': F_history,
            'times': times,
            'params': params,
            'refuge_map': refuge_map
        }

    def _reaction_prey(
        self,
        R: np.ndarray,
        F: np.ndarray,
        params: Dict,
        refuge_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute reaction term for prey population.

        Base implementation returns zero (pure diffusion).
        To be overridden in derived classes.

        Parameters:
            R (np.ndarray): Prey density
            F (np.ndarray): Predator density
            params (dict): Model parameters
            refuge_map (np.ndarray, optional): Refuge function

        Returns:
            np.ndarray: Reaction term for prey
        """
        return np.zeros_like(R)

    def _reaction_predator(
        self,
        R: np.ndarray,
        F: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """
        Compute reaction term for predator population.

        Base implementation returns zero (pure diffusion).
        To be overridden in derived classes.

        Parameters:
            R (np.ndarray): Prey density
            F (np.ndarray): Predator density
            params (dict): Model parameters

        Returns:
            np.ndarray: Reaction term for predator
        """
        return np.zeros_like(F)


class LotkaVolterraSolver(PDESolver):
    """
    Solver for the Lotka-Volterra predator-prey model with spatial diffusion.

    Implements the reaction-diffusion system:

    ∂R/∂t = D_R ∇²R + r·R·(1 - R/K)·p(x,y) - (a·R·F)/(1 + h·R)
    ∂F/∂t = D_F ∇²F + e·(a·R·F)/(1 + h·R) - m·F

    where:
    - R: prey density
    - F: predator density
    - r: intrinsic growth rate of prey
    - K: carrying capacity of prey
    - a: attack rate of predator
    - h: handling time of predator
    - e: conversion efficiency
    - m: mortality rate of predator
    - p(x,y): refuge function (prey growth enhancement in refuges)
    - D_R, D_F: diffusion coefficients
    """

    def _reaction_prey(
        self,
        R: np.ndarray,
        F: np.ndarray,
        params: Dict,
        refuge_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute reaction term for prey population.

        Implements:
        reaction_R = r·R·(1 - R/K)·p(x,y) - (a·R·F)/(1 + h·R)

        The first term is logistic growth modified by refuges.
        The second term is predation with Type II functional response.

        Parameters:
            R (np.ndarray): Prey density (ny, nx)
            F (np.ndarray): Predator density (ny, nx)
            params (dict): Model parameters
            refuge_map (np.ndarray, optional): Refuge function p(x,y) in [0,1]

        Returns:
            np.ndarray: Reaction term for prey
        """
        # Extract parameters
        r = params.get('demographic', {}).get('r', 1.0)
        K = params.get('demographic', {}).get('K', 100.0)
        a = params.get('interaction', {}).get('a', 0.5)
        h = params.get('interaction', {}).get('h', 0.1)

        # Refuge function (default to 1.0 everywhere if not provided)
        if refuge_map is None:
            p = 1.0
        else:
            p = refuge_map

        # Logistic growth with refuge enhancement
        growth = r * R * (1.0 - R / K) * p

        # Type II functional response for predation
        # Avoid division by zero
        predation = (a * R * F) / (1.0 + h * R + 1e-10)

        return growth - predation

    def _reaction_predator(
        self,
        R: np.ndarray,
        F: np.ndarray,
        params: Dict
    ) -> np.ndarray:
        """
        Compute reaction term for predator population.

        Implements:
        reaction_F = e·(a·R·F)/(1 + h·R) - m·F

        The first term is energy gain from predation.
        The second term is mortality.

        Parameters:
            R (np.ndarray): Prey density (ny, nx)
            F (np.ndarray): Predator density (ny, nx)
            params (dict): Model parameters

        Returns:
            np.ndarray: Reaction term for predator
        """
        # Extract parameters
        a = params.get('interaction', {}).get('a', 0.5)
        h = params.get('interaction', {}).get('h', 0.1)
        e = params.get('demographic', {}).get('e', 0.3)
        m = params.get('demographic', {}).get('m', 0.5)

        # Energy gain from predation (Type II functional response)
        gain = e * (a * R * F) / (1.0 + h * R + 1e-10)

        # Mortality
        mortality = m * F

        return gain - mortality


class StochasticSolver(LotkaVolterraSolver):
    """
    Stochastic solver for the Lotka-Volterra model with environmental noise.

    Implements the stochastic reaction-diffusion system using Euler-Maruyama method:

    dR = [D_R ∇²R + reaction_R(R,F)] dt + σ_R·R·dW_R
    dF = [D_F ∇²F + reaction_F(R,F)] dt + σ_F·F·dW_F

    where:
    - dW_R, dW_F are independent Wiener processes (Brownian motion)
    - σ_R, σ_F are noise intensities
    - The noise is multiplicative (proportional to population density)

    This models environmental stochasticity affecting birth and death rates.
    """

    def solve(
        self,
        R0: np.ndarray,
        F0: np.ndarray,
        params: Dict,
        refuge_map: Optional[np.ndarray] = None,
        T: float = 200.0,
        dt: float = 0.01,
        save_every: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Solve the stochastic reaction-diffusion system using Euler-Maruyama.

        Parameters:
            R0 (np.ndarray): Initial prey distribution (ny, nx)
            F0 (np.ndarray): Initial predator distribution (ny, nx)
            params (dict): Model parameters including:
                - 'stochastic': {'sigma_R': float, 'sigma_F': float}
            refuge_map (np.ndarray, optional): Refuge function p(x,y)
            T (float): Final time
            dt (float): Time step size
            save_every (int): Save solution every N time steps

        Returns:
            dict: Dictionary with same structure as deterministic solver
        """
        # Extract parameters
        D_R = params.get('diffusion', {}).get('D_R', 0.1)
        D_F = params.get('diffusion', {}).get('D_F', 0.05)
        sigma_R = params.get('stochastic', {}).get('sigma_R', 0.0)
        sigma_F = params.get('stochastic', {}).get('sigma_F', 0.0)

        # Check stability
        if not self.check_cfl_stability(max(D_R, D_F), dt):
            warnings.warn(
                f"CFL condition may be violated for dt={dt}. "
                f"Consider reducing time step."
            )

        # Initialize
        n_steps = int(T / dt)
        n_saved = (n_steps // save_every) + 1

        R_history = np.zeros((n_saved, self.ny, self.nx))
        F_history = np.zeros((n_saved, self.ny, self.nx))
        times = np.zeros(n_saved)

        R = R0.copy()
        F = F0.copy()

        R_history[0] = R
        F_history[0] = F
        times[0] = 0.0

        save_counter = 1

        # Precompute square root of dt for Wiener increments
        sqrt_dt = np.sqrt(dt)

        # Time integration loop (Euler-Maruyama)
        for n in range(n_steps):
            t = n * dt

            # Deterministic terms
            lap_R = self.laplacian(R)
            lap_F = self.laplacian(F)

            reaction_R = self._reaction_prey(R, F, params, refuge_map)
            reaction_F = self._reaction_predator(R, F, params)

            drift_R = D_R * lap_R + reaction_R
            drift_F = D_F * lap_F + reaction_F

            # Stochastic terms (Wiener increments)
            # dW = sqrt(dt) * N(0,1)
            dW_R = sqrt_dt * np.random.randn(self.ny, self.nx)
            dW_F = sqrt_dt * np.random.randn(self.ny, self.nx)

            # Multiplicative noise (proportional to population)
            diffusion_R = sigma_R * R * dW_R
            diffusion_F = sigma_F * F * dW_F

            # Euler-Maruyama step
            R = R + dt * drift_R + diffusion_R
            F = F + dt * drift_F + diffusion_F

            # Ensure non-negative values
            R = np.maximum(R, 0.0)
            F = np.maximum(F, 0.0)

            # Save if needed
            if (n + 1) % save_every == 0 and save_counter < n_saved:
                R_history[save_counter] = R
                F_history[save_counter] = F
                times[save_counter] = t + dt
                save_counter += 1

        # Save final state if not already saved
        if save_counter < n_saved:
            R_history[save_counter] = R
            F_history[save_counter] = F
            times[save_counter] = T

        return {
            'R': R_history,
            'F': F_history,
            'times': times,
            'params': params,
            'refuge_map': refuge_map
        }

    def solve_ensemble(
        self,
        R0: np.ndarray,
        F0: np.ndarray,
        params: Dict,
        refuge_map: Optional[np.ndarray] = None,
        T: float = 200.0,
        dt: float = 0.01,
        save_every: int = 100,
        n_realizations: int = 50,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Solve multiple independent realizations of the stochastic system.

        This is essential for computing ensemble statistics (mean, variance, etc.)
        of the stochastic process.

        Parameters:
            R0 (np.ndarray): Initial prey distribution
            F0 (np.ndarray): Initial predator distribution
            params (dict): Model parameters
            refuge_map (np.ndarray, optional): Refuge function
            T (float): Final time
            dt (float): Time step size
            save_every (int): Save solution every N time steps
            n_realizations (int): Number of independent realizations
            show_progress (bool): Show progress bar

        Returns:
            dict: Dictionary containing:
                - 'R_mean': Mean prey density over realizations (n_times, ny, nx)
                - 'F_mean': Mean predator density over realizations
                - 'R_std': Standard deviation of prey density
                - 'F_std': Standard deviation of predator density
                - 'R_all': All realizations for prey (n_realizations, n_times, ny, nx)
                - 'F_all': All realizations for predator
                - 'times': Time points
                - 'params': Parameters used
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(n_realizations), desc="Running realizations")
            except ImportError:
                iterator = range(n_realizations)
                print(f"Running {n_realizations} realizations...")
        else:
            iterator = range(n_realizations)

        # First realization to get dimensions
        result = self.solve(R0, F0, params, refuge_map, T, dt, save_every)

        n_times = len(result['times'])

        # Storage for all realizations
        R_all = np.zeros((n_realizations, n_times, self.ny, self.nx))
        F_all = np.zeros((n_realizations, n_times, self.ny, self.nx))

        R_all[0] = result['R']
        F_all[0] = result['F']
        times = result['times']

        # Run remaining realizations
        for i in iterator:
            if i == 0:
                continue  # Already computed

            result = self.solve(R0, F0, params, refuge_map, T, dt, save_every)
            R_all[i] = result['R']
            F_all[i] = result['F']

        # Compute statistics
        R_mean = np.mean(R_all, axis=0)
        F_mean = np.mean(F_all, axis=0)
        R_std = np.std(R_all, axis=0)
        F_std = np.std(F_all, axis=0)

        return {
            'R_mean': R_mean,
            'F_mean': F_mean,
            'R_std': R_std,
            'F_std': F_std,
            'R_all': R_all,
            'F_all': F_all,
            'times': times,
            'params': params,
            'refuge_map': refuge_map,
            'n_realizations': n_realizations
        }
