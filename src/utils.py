"""
Utility functions for the predator-prey simulation.

This module provides helper functions for:
- Saving and loading simulation results
- Computing stability criteria
- Performance optimization
- Result analysis
"""

import numpy as np
import json
import pickle
import os
from typing import Dict, Optional, Tuple
from datetime import datetime


def compute_cfl_number(D: float, dt: float, dx: float, dy: float) -> float:
    """
    Compute the CFL (Courant-Friedrichs-Lewy) number for diffusion.

    For explicit Euler method with diffusion, the stability condition is:
    CFL = D * dt * (1/dx² + 1/dy²) <= 0.5

    Parameters:
        D (float): Diffusion coefficient
        dt (float): Time step size
        dx (float): Spatial step in x-direction
        dy (float): Spatial step in y-direction

    Returns:
        float: CFL number
    """
    return D * dt * (1.0 / dx**2 + 1.0 / dy**2)


def suggest_stable_timestep(D: float, dx: float, dy: float, safety_factor: float = 0.9) -> float:
    """
    Suggest a stable time step size based on CFL criterion.

    Parameters:
        D (float): Diffusion coefficient
        dx (float): Spatial step in x-direction
        dy (float): Spatial step in y-direction
        safety_factor (float): Safety factor (< 1.0 for extra stability)

    Returns:
        float: Suggested time step size
    """
    cfl_limit = 0.5 * safety_factor
    dt_max = cfl_limit / (D * (1.0 / dx**2 + 1.0 / dy**2))
    return dt_max


def check_stability_comprehensive(
    params: Dict,
    dt: float,
    dx: float,
    dy: float,
    verbose: bool = True
) -> Tuple[bool, Dict]:
    """
    Comprehensive stability check for the reaction-diffusion system.

    Checks:
    1. CFL condition for diffusion
    2. Time step size relative to reaction time scales
    3. Spatial resolution relative to diffusion length

    Parameters:
        params (dict): Model parameters
        dt (float): Time step size
        dx (float): Spatial step in x-direction
        dy (float): Spatial step in y-direction
        verbose (bool): Print detailed information

    Returns:
        tuple: (is_stable, details_dict)
    """
    D_R = params.get('diffusion', {}).get('D_R', 0.1)
    D_F = params.get('diffusion', {}).get('D_F', 0.05)
    r = params.get('demographic', {}).get('r', 1.0)
    m = params.get('demographic', {}).get('m', 0.5)

    details = {}

    # Check CFL for both species
    cfl_R = compute_cfl_number(D_R, dt, dx, dy)
    cfl_F = compute_cfl_number(D_F, dt, dx, dy)

    details['cfl_R'] = cfl_R
    details['cfl_F'] = cfl_F
    details['cfl_stable'] = (cfl_R <= 0.5) and (cfl_F <= 0.5)

    # Check time step relative to reaction rates
    # For stability, dt should be much smaller than 1/r and 1/m
    reaction_timescale_R = 1.0 / r if r > 0 else np.inf
    reaction_timescale_F = 1.0 / m if m > 0 else np.inf

    details['dt_over_timescale_R'] = dt / reaction_timescale_R
    details['dt_over_timescale_F'] = dt / reaction_timescale_F
    details['reaction_stable'] = (dt < 0.1 * min(reaction_timescale_R, reaction_timescale_F))

    # Check spatial resolution
    # Diffusion length scale: sqrt(D/r)
    diffusion_length_R = np.sqrt(D_R / r) if r > 0 else np.inf
    diffusion_length_F = np.sqrt(D_F / m) if m > 0 else np.inf

    details['dx_over_diff_length_R'] = dx / diffusion_length_R if diffusion_length_R < np.inf else 0
    details['dx_over_diff_length_F'] = dx / diffusion_length_F if diffusion_length_F < np.inf else 0
    details['spatial_resolution_adequate'] = (
        dx < 0.5 * min(diffusion_length_R, diffusion_length_F)
    )

    # Overall stability
    is_stable = (
        details['cfl_stable'] and
        details['reaction_stable'] and
        details['spatial_resolution_adequate']
    )

    if verbose:
        print("=" * 60)
        print("Stability Analysis")
        print("=" * 60)
        print(f"CFL number (prey):     {cfl_R:.4f} {'CORRECT:' if cfl_R <= 0.5 else 'ERROR:'}")
        print(f"CFL number (predator): {cfl_F:.4f} {'CORRECT' if cfl_F <= 0.5 else 'ERROR:'}")
        print(f"Time step / reaction timescale (R): {details['dt_over_timescale_R']:.4f}")
        print(f"Time step / reaction timescale (F): {details['dt_over_timescale_F']:.4f}")
        print(f"Spatial resolution (R): {details['dx_over_diff_length_R']:.4f}")
        print(f"Spatial resolution (F): {details['dx_over_diff_length_F']:.4f}")
        print(f"\nOverall stability: {'STABLE' if is_stable else 'UNSTABLE'}")
        print("=" * 60)

    return is_stable, details


def save_results(
    result: Dict,
    filepath: str,
    format: str = 'npz',
    compress: bool = True,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save simulation results to disk.

    Supports multiple formats:
    - 'npz': NumPy compressed archive (efficient for arrays)
    - 'pickle': Python pickle (preserves all data structures)
    - 'hdf5': HDF5 format (requires h5py, good for very large datasets)

    Parameters:
        result (dict): Result dictionary from solver
        filepath (str): Path to save file
        format (str): File format ('npz', 'pickle', 'hdf5')
        compress (bool): Whether to compress data
        metadata (dict, optional): Additional metadata to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    if metadata is None:
        metadata = {}

    # Add timestamp
    metadata['timestamp'] = datetime.now().isoformat()

    if format == 'npz':
        # Save as compressed NumPy archive
        save_dict = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                save_dict[key] = value
            elif isinstance(value, dict):
                # Save dict as JSON string
                save_dict[key + '_json'] = np.array([json.dumps(value)])

        save_dict['metadata'] = np.array([json.dumps(metadata)])

        if compress:
            np.savez_compressed(filepath, **save_dict)
        else:
            np.savez(filepath, **save_dict)

    elif format == 'pickle':
        # Save as pickle
        result['metadata'] = metadata

        with open(filepath, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif format == 'hdf5':
        try:
            import h5py

            with h5py.File(filepath, 'w') as f:
                # Save arrays
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        if compress:
                            f.create_dataset(key, data=value, compression='gzip')
                        else:
                            f.create_dataset(key, data=value)
                    elif isinstance(value, dict):
                        # Save dict as attributes
                        grp = f.create_group(key)
                        for k, v in value.items():
                            grp.attrs[k] = v

                # Save metadata
                meta_grp = f.create_group('metadata')
                for k, v in metadata.items():
                    meta_grp.attrs[k] = v

        except ImportError:
            raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")

    else:
        raise ValueError(f"Unknown format: {format}. Use 'npz', 'pickle', or 'hdf5'.")


def load_results(filepath: str, format: str = None) -> Dict:
    """
    Load simulation results from disk.

    Parameters:
        filepath (str): Path to file
        format (str, optional): File format (auto-detected if None)

    Returns:
        dict: Result dictionary
    """
    # Auto-detect format from extension
    if format is None:
        ext = os.path.splitext(filepath)[1]
        if ext == '.npz':
            format = 'npz'
        elif ext in ['.pkl', '.pickle']:
            format = 'pickle'
        elif ext in ['.h5', '.hdf5']:
            format = 'hdf5'
        else:
            raise ValueError(f"Cannot auto-detect format from extension: {ext}")

    if format == 'npz':
        data = np.load(filepath, allow_pickle=True)
        result = {}

        for key in data.keys():
            if key.endswith('_json'):
                # Deserialize JSON
                original_key = key[:-5]
                result[original_key] = json.loads(str(data[key][0]))
            elif key == 'metadata':
                result['metadata'] = json.loads(str(data[key][0]))
            else:
                result[key] = data[key]

        return result

    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    elif format == 'hdf5':
        try:
            import h5py

            result = {}

            with h5py.File(filepath, 'r') as f:
                # Load arrays
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        result[key] = f[key][:]
                    elif isinstance(f[key], h5py.Group):
                        # Load dict from attributes
                        result[key] = dict(f[key].attrs)

            return result

        except ImportError:
            raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")

    else:
        raise ValueError(f"Unknown format: {format}")


def compute_total_population(field: np.ndarray, dx: float, dy: float) -> float:
    """
    Compute total population by integrating over space.

    Parameters:
        field (np.ndarray): Spatial distribution (ny, nx)
        dx (float): Spatial step in x
        dy (float): Spatial step in y

    Returns:
        float: Total population
    """
    return np.sum(field) * dx * dy


def compute_center_of_mass(field: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    """
    Compute center of mass of a spatial distribution.

    Parameters:
        field (np.ndarray): Spatial distribution (ny, nx)
        X (np.ndarray): X coordinates (meshgrid)
        Y (np.ndarray): Y coordinates (meshgrid)

    Returns:
        tuple: (x_cm, y_cm)
    """
    total = np.sum(field)
    if total < 1e-10:
        return np.nan, np.nan

    x_cm = np.sum(field * X) / total
    y_cm = np.sum(field * Y) / total

    return x_cm, y_cm


def compute_spatial_variance(
    field: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute spatial variance (spread) of a distribution.

    Parameters:
        field (np.ndarray): Spatial distribution (ny, nx)
        X (np.ndarray): X coordinates (meshgrid)
        Y (np.ndarray): Y coordinates (meshgrid)

    Returns:
        tuple: (var_x, var_y, var_total)
    """
    x_cm, y_cm = compute_center_of_mass(field, X, Y)
    total = np.sum(field)

    if total < 1e-10 or np.isnan(x_cm):
        return np.nan, np.nan, np.nan

    var_x = np.sum(field * (X - x_cm)**2) / total
    var_y = np.sum(field * (Y - y_cm)**2) / total
    var_total = var_x + var_y

    return var_x, var_y, var_total
