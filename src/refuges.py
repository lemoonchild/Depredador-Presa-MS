"""
Refuge generation module for the predator-prey system.

This module provides tools to generate spatial refuge maps that enhance
prey growth in specific regions of the domain.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class RefugeGenerator:
    """
    Generator for spatial refuge configurations.

    This class provides methods to create refuge maps p(x,y) that modify
    prey growth rates in different regions of the spatial domain.

    Attributes:
        L (float): Domain size (square domain [0, L] x [0, L])
        nx (int): Number of grid points in x-direction
        ny (int): Number of grid points in y-direction
        dx (float): Spatial step size in x-direction
        dy (float): Spatial step size in y-direction
        refuge_map (np.ndarray): Current refuge configuration (ny, nx)
    """

    def __init__(self, L: float = 10.0, nx: int = 100, ny: int = 100):
        """
        Initialize the refuge generator.

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

        # Initialize refuge map (default: no refuge enhancement, p=1.0 everywhere)
        self.refuge_map = np.ones((ny, nx))

    def reset(self):
        """
        Reset the refuge map to default (p = 1.0 everywhere).
        """
        self.refuge_map = np.ones((self.ny, self.nx))

    def generate_circular_refuges(
        self,
        centers: List[Tuple[float, float]],
        radii: List[float],
        enhancement: float = 2.0,
        additive: bool = False
    ) -> np.ndarray:
        """
        Generate circular refuge regions.

        Parameters:
            centers (list): List of (x, y) coordinates for circle centers
            radii (list): List of radii for each circle
            enhancement (float): Growth enhancement factor in refuges (p value)
            additive (bool): If True, add to existing map; if False, reset first

        Returns:
            np.ndarray: Refuge map with circular refuges
        """
        if not additive:
            self.reset()

        if len(centers) != len(radii):
            raise ValueError("Number of centers must match number of radii")

        for (cx, cy), radius in zip(centers, radii):
            # Compute distance from center
            dist = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)

            # Set enhancement in circular region
            mask = dist <= radius
            self.refuge_map[mask] = enhancement

        return self.refuge_map.copy()

    def generate_rectangular_refuges(
        self,
        corners: List[Tuple[float, float]],
        sizes: List[Tuple[float, float]],
        enhancement: float = 2.0,
        additive: bool = False
    ) -> np.ndarray:
        """
        Generate rectangular refuge regions.

        Parameters:
            corners (list): List of (x, y) coordinates for bottom-left corners
            sizes (list): List of (width, height) for each rectangle
            enhancement (float): Growth enhancement factor in refuges
            additive (bool): If True, add to existing map; if False, reset first

        Returns:
            np.ndarray: Refuge map with rectangular refuges
        """
        if not additive:
            self.reset()

        if len(corners) != len(sizes):
            raise ValueError("Number of corners must match number of sizes")

        for (corner_x, corner_y), (width, height) in zip(corners, sizes):
            # Define rectangle boundaries
            x_min, x_max = corner_x, corner_x + width
            y_min, y_max = corner_y, corner_y + height

            # Create mask for rectangle
            mask = (self.X >= x_min) & (self.X <= x_max) & \
                   (self.Y >= y_min) & (self.Y <= y_max)

            self.refuge_map[mask] = enhancement

        return self.refuge_map.copy()

    def generate_central_strip(
        self,
        width: float,
        orientation: str = 'vertical',
        enhancement: float = 2.0,
        additive: bool = False
    ) -> np.ndarray:
        """
        Generate a central strip refuge (vertical or horizontal).

        Parameters:
            width (float): Width of the strip
            orientation (str): 'vertical' or 'horizontal'
            enhancement (float): Growth enhancement factor in refuge
            additive (bool): If True, add to existing map; if False, reset first

        Returns:
            np.ndarray: Refuge map with central strip
        """
        if not additive:
            self.reset()

        center = self.L / 2.0
        half_width = width / 2.0

        if orientation.lower() == 'vertical':
            # Vertical strip centered at x = L/2
            mask = (self.X >= center - half_width) & (self.X <= center + half_width)
        elif orientation.lower() == 'horizontal':
            # Horizontal strip centered at y = L/2
            mask = (self.Y >= center - half_width) & (self.Y <= center + half_width)
        else:
            raise ValueError("Orientation must be 'vertical' or 'horizontal'")

        self.refuge_map[mask] = enhancement

        return self.refuge_map.copy()

    def generate_corner_refuges(
        self,
        size: float,
        corners: List[str] = ['all'],
        enhancement: float = 2.0,
        additive: bool = False
    ) -> np.ndarray:
        """
        Generate square refuges in the corners of the domain.

        Parameters:
            size (float): Size of square refuge in each corner
            corners (list): Which corners to include: 'all', 'bottom-left',
                          'bottom-right', 'top-left', 'top-right'
            enhancement (float): Growth enhancement factor in refuges
            additive (bool): If True, add to existing map; if False, reset first

        Returns:
            np.ndarray: Refuge map with corner refuges
        """
        if not additive:
            self.reset()

        # Define corner positions and masks
        corner_configs = {
            'bottom-left': (0, 0),
            'bottom-right': (self.L - size, 0),
            'top-left': (0, self.L - size),
            'top-right': (self.L - size, self.L - size)
        }

        # Determine which corners to create
        if 'all' in corners:
            active_corners = list(corner_configs.keys())
        else:
            active_corners = corners

        # Create refuges in specified corners
        for corner_name in active_corners:
            if corner_name not in corner_configs:
                raise ValueError(f"Invalid corner name: {corner_name}")

            corner_x, corner_y = corner_configs[corner_name]

            mask = (self.X >= corner_x) & (self.X <= corner_x + size) & \
                   (self.Y >= corner_y) & (self.Y <= corner_y + size)

            self.refuge_map[mask] = enhancement

        return self.refuge_map.copy()

    def generate_random_refuges(
        self,
        num_refuges: int,
        min_size: float,
        max_size: float,
        shape: str = 'circular',
        enhancement: float = 2.0,
        additive: bool = False,
        avoid_overlap: bool = True,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate randomly placed refuges.

        Parameters:
            num_refuges (int): Number of refuges to generate
            min_size (float): Minimum refuge size (radius or side length)
            max_size (float): Maximum refuge size
            shape (str): 'circular' or 'rectangular'
            enhancement (float): Growth enhancement factor
            additive (bool): If True, add to existing map; if False, reset first
            avoid_overlap (bool): If True, attempt to avoid overlapping refuges
            seed (int, optional): Random seed for reproducibility

        Returns:
            np.ndarray: Refuge map with random refuges
        """
        if not additive:
            self.reset()

        if seed is not None:
            np.random.seed(seed)

        centers_list = []
        sizes_list = []

        for i in range(num_refuges):
            # Generate random size
            if shape == 'circular':
                size = np.random.uniform(min_size, max_size)
            else:
                width = np.random.uniform(min_size, max_size)
                height = np.random.uniform(min_size, max_size)
                size = (width, height)

            # Generate random position with margin to avoid boundaries
            margin = max_size
            max_attempts = 100

            for attempt in range(max_attempts):
                x = np.random.uniform(margin, self.L - margin)
                y = np.random.uniform(margin, self.L - margin)

                # Check for overlap if required
                if avoid_overlap and len(centers_list) > 0:
                    # Check distance to existing centers
                    min_dist = min([np.sqrt((x - cx)**2 + (y - cy)**2)
                                   for cx, cy in centers_list])

                    if shape == 'circular':
                        required_dist = size + max([s for s in sizes_list if isinstance(s, (int, float))], default=0)
                    else:
                        required_dist = max(size[0], size[1]) + margin

                    if min_dist > required_dist:
                        break
                else:
                    break

            centers_list.append((x, y))
            sizes_list.append(size)

        # Generate refuges based on shape
        if shape == 'circular':
            radii = sizes_list
            self.generate_circular_refuges(
                centers=centers_list,
                radii=radii,
                enhancement=enhancement,
                additive=True
            )
        elif shape == 'rectangular':
            # Convert to corners for rectangular refuges
            corners = [(x - w/2, y - h/2) for (x, y), (w, h) in zip(centers_list, sizes_list)]
            self.generate_rectangular_refuges(
                corners=corners,
                sizes=sizes_list,
                enhancement=enhancement,
                additive=True
            )
        else:
            raise ValueError("Shape must be 'circular' or 'rectangular'")

        return self.refuge_map.copy()

    def visualize(
        self,
        title: str = "Refuge Configuration",
        cmap: str = 'YlGn',
        figsize: Tuple[int, int] = (8, 7)
    ) -> plt.Figure:
        """
        Visualize the current refuge configuration.

        Parameters:
            title (str): Plot title
            cmap (str): Colormap name
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            self.refuge_map,
            origin='lower',
            cmap=cmap,
            extent=[0, self.L, 0, self.L],
            interpolation='bilinear'
        )

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Growth Enhancement Factor p(x,y)', fontsize=11)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        plt.tight_layout()

        return fig

    def get_refuge_map(self) -> np.ndarray:
        """
        Get the current refuge map.

        Returns:
            np.ndarray: Copy of the current refuge map
        """
        return self.refuge_map.copy()

    def save_refuge_map(self, filename: str):
        """
        Save the refuge map to a file.

        Parameters:
            filename (str): Path to save the refuge map (numpy .npy format)
        """
        np.save(filename, self.refuge_map)

    def load_refuge_map(self, filename: str) -> np.ndarray:
        """
        Load a refuge map from a file.

        Parameters:
            filename (str): Path to the refuge map file

        Returns:
            np.ndarray: Loaded refuge map
        """
        self.refuge_map = np.load(filename)

        # Validate dimensions
        if self.refuge_map.shape != (self.ny, self.nx):
            raise ValueError(
                f"Loaded refuge map shape {self.refuge_map.shape} "
                f"does not match expected shape ({self.ny}, {self.nx})"
            )

        return self.refuge_map.copy()