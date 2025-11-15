"""
Model configuration and initial conditions module.

This module provides tools for managing model parameters and generating
initial conditions for the predator-prey simulations.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
from pathlib import Path


class ModelParameters:
    """
    Manager for model parameters with validation and I/O capabilities.

    This class handles all parameters for the predator-prey model including
    spatial, temporal, demographic, interaction, diffusion, and stochastic
    parameters.

    Attributes:
        params (dict): Dictionary containing all model parameters
    """

    DEFAULT_PARAMS = {
        "spatial": {
            "L": 10.0,      # Domain size
            "nx": 100,      # Grid points in x
            "ny": 100       # Grid points in y
        },
        "temporal": {
            "T": 200.0,     # Final time
            "dt": 0.01      # Time step
        },
        "demographic": {
            "r": 1.0,       # Prey intrinsic growth rate
            "K": 100.0,     # Prey carrying capacity
            "m": 0.5,       # Predator mortality rate
            "e": 0.3        # Conversion efficiency
        },
        "interaction": {
            "a": 0.5,       # Attack rate
            "h": 0.1        # Handling time
        },
        "diffusion": {
            "D_R": 0.1,     # Prey diffusion coefficient
            "D_F": 0.05     # Predator diffusion coefficient
        },
        "stochastic": {
            "sigma_R": 0.05,        # Prey noise intensity
            "sigma_F": 0.05,        # Predator noise intensity
            "n_realizations": 50    # Number of realizations for ensemble
        }
    }

    VALID_RANGES = {
        "spatial": {
            "L": (0.1, 1000.0),
            "nx": (10, 1000),
            "ny": (10, 1000)
        },
        "temporal": {
            "T": (0.1, 10000.0),
            "dt": (1e-6, 1.0)
        },
        "demographic": {
            "r": (0.0, 10.0),
            "K": (1.0, 1000.0),
            "m": (0.0, 10.0),
            "e": (0.0, 1.0)
        },
        "interaction": {
            "a": (0.0, 10.0),
            "h": (0.0, 10.0)
        },
        "diffusion": {
            "D_R": (0.0, 10.0),
            "D_F": (0.0, 10.0)
        },
        "stochastic": {
            "sigma_R": (0.0, 1.0),
            "sigma_F": (0.0, 1.0),
            "n_realizations": (1, 1000)
        }
    }

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the parameter manager.

        Parameters:
            params (dict, optional): Custom parameters. If None, uses defaults.
        """
        if params is None:
            self.params = self._deep_copy_dict(self.DEFAULT_PARAMS)
        else:
            self.params = self._deep_copy_dict(self.DEFAULT_PARAMS)
            self._update_params(params)

        self.validate()

    def _deep_copy_dict(self, d: Dict) -> Dict:
        """Deep copy a dictionary (simple implementation for nested dicts)."""
        import copy
        return copy.deepcopy(d)

    def _update_params(self, new_params: Dict):
        """Update parameters recursively."""
        for key, value in new_params.items():
            if key in self.params:
                if isinstance(value, dict) and isinstance(self.params[key], dict):
                    self.params[key].update(value)
                else:
                    self.params[key] = value
            else:
                self.params[key] = value

    def validate(self) -> bool:
        """
        Validate all parameters are within acceptable ranges.

        Returns:
            bool: True if all parameters are valid

        Raises:
            ValueError: If any parameter is out of valid range
        """
        for category, params in self.params.items():
            if category not in self.VALID_RANGES:
                continue  # Skip unknown categories

            for param_name, value in params.items():
                if param_name not in self.VALID_RANGES[category]:
                    continue  # Skip unknown parameters

                min_val, max_val = self.VALID_RANGES[category][param_name]

                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Parameter {category}.{param_name} = {value} "
                        f"is out of valid range [{min_val}, {max_val}]"
                    )

        # dt should satisfy CFL condition approximately
        if 'temporal' in self.params and 'diffusion' in self.params and 'spatial' in self.params:
            dt = self.params['temporal']['dt']
            D_max = max(self.params['diffusion']['D_R'],
                       self.params['diffusion']['D_F'])
            L = self.params['spatial']['L']
            nx = self.params['spatial']['nx']
            dx = L / (nx - 1)

            cfl = D_max * dt / dx**2
            if cfl > 0.5:
                import warnings
                warnings.warn(
                    f"CFL condition may be violated: D*dt/dx² = {cfl:.4f} > 0.5. "
                    f"Consider reducing dt or increasing nx."
                )

        return True

    def get(self) -> Dict:
        """
        Get the current parameters.

        Returns:
            dict: Copy of current parameters
        """
        return self._deep_copy_dict(self.params)

    def set(self, category: str, param_name: str, value: float):
        """
        Set a specific parameter value.

        Parameters:
            category (str): Parameter category (e.g., 'demographic')
            param_name (str): Parameter name (e.g., 'r')
            value (float): New value
        """
        if category not in self.params:
            raise ValueError(f"Unknown parameter category: {category}")

        if param_name not in self.params[category]:
            raise ValueError(f"Unknown parameter {param_name} in category {category}")

        self.params[category][param_name] = value
        self.validate()

    def save(self, filename: str):
        """
        Save parameters to a JSON file.

        Parameters:
            filename (str): Path to save the parameters
        """
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(self.params, f, indent=2)

    @classmethod
    def load(cls, filename: str) -> 'ModelParameters':
        """
        Load parameters from a JSON file.

        Parameters:
            filename (str): Path to the parameter file

        Returns:
            ModelParameters: New instance with loaded parameters
        """
        with open(filename, 'r') as f:
            params_dict = json.load(f)

        return cls(params_dict)

    def __repr__(self) -> str:
        """String representation of parameters."""
        lines = ["ModelParameters:"]
        for category, params in self.params.items():
            lines.append(f"  {category}:")
            for param_name, value in params.items():
                lines.append(f"    {param_name}: {value}")
        return "\n".join(lines)


class InitialConditions:
    """
    Generator for initial conditions for the predator-prey system.

    This class provides various methods to generate spatial distributions
    for prey and predator populations.

    Attributes:
        L (float): Domain size
        nx (int): Grid points in x-direction
        ny (int): Grid points in y-direction
        X (np.ndarray): Meshgrid of x-coordinates
        Y (np.ndarray): Meshgrid of y-coordinates
    """

    def __init__(self, L: float = 10.0, nx: int = 100, ny: int = 100):
        """
        Initialize the initial conditions generator.

        Parameters:
            L (float): Domain size
            nx (int): Grid points in x-direction
            ny (int): Grid points in y-direction
        """
        self.L = L
        self.nx = nx
        self.ny = ny

        # Create spatial grid
        self.x = np.linspace(0, L, nx)
        self.y = np.linspace(0, L, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def uniform(
        self,
        R0: float = 50.0,
        F0: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate uniform initial conditions.

        Parameters:
            R0 (float): Constant prey density
            F0 (float): Constant predator density

        Returns:
            tuple: (R_initial, F_initial) arrays of shape (ny, nx)
        """
        R_initial = np.full((self.ny, self.nx), R0)
        F_initial = np.full((self.ny, self.nx), F0)

        return R_initial, F_initial

    def gaussian(
        self,
        R0: float = 50.0,
        F0: float = 10.0,
        center: Optional[Tuple[float, float]] = None,
        sigma: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Gaussian initial conditions centered in the domain.

        Parameters:
            R0 (float): Peak prey density
            F0 (float): Peak predator density
            center (tuple, optional): (x, y) center coordinates. If None, uses domain center
            sigma (float): Standard deviation of Gaussian

        Returns:
            tuple: (R_initial, F_initial) arrays of shape (ny, nx)
        """
        if center is None:
            center = (self.L / 2.0, self.L / 2.0)

        x0, y0 = center

        # Gaussian profile
        gaussian = np.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2 * sigma**2))

        R_initial = R0 * gaussian
        F_initial = F0 * gaussian

        return R_initial, F_initial

    def random_spatial(
        self,
        R_mean: float = 50.0,
        F_mean: float = 10.0,
        R_std: float = 10.0,
        F_std: float = 2.0,
        seed: Optional[int] = None,
        ensure_positive: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spatially random initial conditions.

        Parameters:
            R_mean (float): Mean prey density
            F_mean (float): Mean predator density
            R_std (float): Standard deviation of prey density
            F_std (float): Standard deviation of predator density
            seed (int, optional): Random seed for reproducibility
            ensure_positive (bool): If True, clip negative values to zero

        Returns:
            tuple: (R_initial, F_initial) arrays of shape (ny, nx)
        """
        if seed is not None:
            np.random.seed(seed)

        R_initial = np.random.normal(R_mean, R_std, (self.ny, self.nx))
        F_initial = np.random.normal(F_mean, F_std, (self.ny, self.nx))

        if ensure_positive:
            R_initial = np.maximum(R_initial, 0.0)
            F_initial = np.maximum(F_initial, 0.0)

        return R_initial, F_initial

    def prey_in_refuge_predators_outside(
        self,
        refuge_map: np.ndarray,
        R_in_refuge: float = 80.0,
        R_outside: float = 20.0,
        F_in_refuge: float = 2.0,
        F_outside: float = 15.0,
        refuge_threshold: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate initial conditions with prey concentrated in refuges
        and predators mostly outside.

        This represents a scenario where prey have already found refuges
        and predators are searching.

        Parameters:
            refuge_map (np.ndarray): Refuge map indicating refuge locations
            R_in_refuge (float): Prey density inside refuges
            R_outside (float): Prey density outside refuges
            F_in_refuge (float): Predator density inside refuges (low)
            F_outside (float): Predator density outside refuges (high)
            refuge_threshold (float): Threshold to determine refuge regions

        Returns:
            tuple: (R_initial, F_initial) arrays of shape (ny, nx)
        """
        # Check dimensions
        if refuge_map.shape != (self.ny, self.nx):
            raise ValueError(
                f"Refuge map shape {refuge_map.shape} does not match "
                f"expected shape ({self.ny}, {self.nx})"
            )

        # Create mask for refuge regions
        in_refuge = refuge_map >= refuge_threshold

        # Initialize arrays
        R_initial = np.full((self.ny, self.nx), R_outside)
        F_initial = np.full((self.ny, self.nx), F_outside)

        # Set values in refuge regions
        R_initial[in_refuge] = R_in_refuge
        F_initial[in_refuge] = F_in_refuge

        return R_initial, F_initial

    def multi_gaussian(
        self,
        centers: List[Tuple[float, float]],
        R_peaks: List[float],
        F_peaks: List[float],
        sigmas: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate initial conditions with multiple Gaussian peaks.

        Parameters:
            centers (list): List of (x, y) coordinates for peak centers
            R_peaks (list): Peak prey densities for each center
            F_peaks (list): Peak predator densities for each center
            sigmas (list): Standard deviations for each Gaussian

        Returns:
            tuple: (R_initial, F_initial) arrays of shape (ny, nx)
        """
        if not (len(centers) == len(R_peaks) == len(F_peaks) == len(sigmas)):
            raise ValueError("All input lists must have the same length")

        R_initial = np.zeros((self.ny, self.nx))
        F_initial = np.zeros((self.ny, self.nx))

        for (x0, y0), R_peak, F_peak, sigma in zip(centers, R_peaks, F_peaks, sigmas):
            gaussian = np.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2 * sigma**2))
            R_initial += R_peak * gaussian
            F_initial += F_peak * gaussian

        return R_initial, F_initial

    def visualize(
        self,
        R_initial: np.ndarray,
        F_initial: np.ndarray,
        refuge_map: Optional[np.ndarray] = None,
        title: str = "Initial Conditions",
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Visualize initial conditions.

        Parameters:
            R_initial (np.ndarray): Initial prey distribution
            F_initial (np.ndarray): Initial predator distribution
            refuge_map (np.ndarray, optional): Refuge map to overlay
            title (str): Figure title
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        n_plots = 3 if refuge_map is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        # Prey distribution
        im0 = axes[0].imshow(
            R_initial,
            origin='lower',
            cmap='Blues',
            extent=[0, self.L, 0, self.L],
            interpolation='bilinear'
        )
        axes[0].set_title('Prey (R) Initial Distribution', fontweight='bold')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0], label='Density')

        # Predator distribution
        im1 = axes[1].imshow(
            F_initial,
            origin='lower',
            cmap='Reds',
            extent=[0, self.L, 0, self.L],
            interpolation='bilinear'
        )
        axes[1].set_title('Predator (F) Initial Distribution', fontweight='bold')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1], label='Density')

        # Refuge map if provided
        if refuge_map is not None:
            im2 = axes[2].imshow(
                refuge_map,
                origin='lower',
                cmap='Greens',
                extent=[0, self.L, 0, self.L],
                interpolation='bilinear'
            )
            axes[2].set_title('Refuge Map', fontweight='bold')
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('y')
            plt.colorbar(im2, ax=axes[2], label='Growth Factor')

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        return fig
