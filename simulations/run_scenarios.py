"""
Scenario execution system for predator-prey simulations.

This module provides functions to run all simulation scenarios defined
in the project specification.
"""

import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver import LotkaVolterraSolver, StochasticSolver
from src.model import ModelParameters, InitialConditions
from src.refuges import RefugeGenerator

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not available. Install with 'pip install tqdm' for progress bars.")


def run_single_scenario(
    params: Dict,
    refuge_config: Optional[Dict] = None,
    initial_conditions_config: Optional[Dict] = None,
    scenario_name: str = "unnamed",
    save_every: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Run a single simulation scenario.

    Parameters:
        params (dict): Model parameters
        refuge_config (dict, optional): Refuge configuration with keys:
            - 'type': Type of refuge ('none', 'circular', 'rectangular', etc.)
            - Additional parameters specific to refuge type
        initial_conditions_config (dict, optional): Initial conditions config with keys:
            - 'type': Type of IC ('uniform', 'gaussian', 'random_spatial', etc.)
            - Additional parameters specific to IC type
        scenario_name (str): Name for this scenario
        save_every (int): Save solution every N time steps
        verbose (bool): Print progress information

    Returns:
        dict: Simulation results including:
            - 'R': Prey density evolution
            - 'F': Predator density evolution
            - 'times': Time points
            - 'params': Parameters used
            - 'refuge_map': Refuge configuration
            - 'metadata': Scenario metadata
    """
    if verbose:
        print(f"\n === Running Scenario: {scenario_name} ===")

    start_time = time.time()

    # Extract spatial parameters
    L = params.get('spatial', {}).get('L', 10.0)
    nx = params.get('spatial', {}).get('nx', 100)
    ny = params.get('spatial', {}).get('ny', 100)

    # Generate refuge map
    refuge_map = None
    if refuge_config is not None and refuge_config.get('type') != 'none':
        if verbose:
            print(f"Generating refuge configuration: {refuge_config.get('type')}")

        gen = RefugeGenerator(L=L, nx=nx, ny=ny)

        if refuge_config['type'] == 'circular':
            centers = refuge_config.get('centers', [(L/2, L/2)])
            radii = refuge_config.get('radii', [2.0])
            enhancement = refuge_config.get('enhancement', 2.0)
            refuge_map = gen.generate_circular_refuges(centers, radii, enhancement)

        elif refuge_config['type'] == 'rectangular':
            corners = refuge_config.get('corners', [(2.0, 2.0)])
            sizes = refuge_config.get('sizes', [(3.0, 3.0)])
            enhancement = refuge_config.get('enhancement', 2.0)
            refuge_map = gen.generate_rectangular_refuges(corners, sizes, enhancement)

        elif refuge_config['type'] == 'central_strip':
            width = refuge_config.get('width', 2.0)
            orientation = refuge_config.get('orientation', 'vertical')
            enhancement = refuge_config.get('enhancement', 2.0)
            refuge_map = gen.generate_central_strip(width, orientation, enhancement)

        elif refuge_config['type'] == 'corners':
            size = refuge_config.get('size', 1.5)
            corners = refuge_config.get('corners', ['all'])
            enhancement = refuge_config.get('enhancement', 2.0)
            refuge_map = gen.generate_corner_refuges(size, corners, enhancement)

        elif refuge_config['type'] == 'random':
            num_refuges = refuge_config.get('num_refuges', 5)
            min_size = refuge_config.get('min_size', 0.5)
            max_size = refuge_config.get('max_size', 1.5)
            shape = refuge_config.get('shape', 'circular')
            enhancement = refuge_config.get('enhancement', 2.0)
            seed = refuge_config.get('seed', None)
            refuge_map = gen.generate_random_refuges(
                num_refuges, min_size, max_size, shape, enhancement, seed=seed
            )

    # Generate initial conditions
    if verbose:
        ic_type = initial_conditions_config.get('type', 'uniform') if initial_conditions_config else 'uniform'
        print(f"Generating initial conditions: {ic_type}")

    ic = InitialConditions(L=L, nx=nx, ny=ny)

    if initial_conditions_config is None or initial_conditions_config.get('type') == 'uniform':
        R0_val = initial_conditions_config.get('R0', 50.0) if initial_conditions_config else 50.0
        F0_val = initial_conditions_config.get('F0', 10.0) if initial_conditions_config else 10.0
        R0, F0 = ic.uniform(R0=R0_val, F0=F0_val)

    elif initial_conditions_config['type'] == 'gaussian':
        R0_val = initial_conditions_config.get('R0', 50.0)
        F0_val = initial_conditions_config.get('F0', 10.0)
        sigma = initial_conditions_config.get('sigma', 1.0)
        center = initial_conditions_config.get('center', None)
        R0, F0 = ic.gaussian(R0=R0_val, F0=F0_val, center=center, sigma=sigma)

    elif initial_conditions_config['type'] == 'random_spatial':
        R_mean = initial_conditions_config.get('R_mean', 50.0)
        F_mean = initial_conditions_config.get('F_mean', 10.0)
        R_std = initial_conditions_config.get('R_std', 10.0)
        F_std = initial_conditions_config.get('F_std', 2.0)
        seed = initial_conditions_config.get('seed', None)
        R0, F0 = ic.random_spatial(R_mean, F_mean, R_std, F_std, seed=seed)

    elif initial_conditions_config['type'] == 'prey_in_refuge':
        if refuge_map is None:
            raise ValueError("'prey_in_refuge' IC requires a refuge configuration")
        R_in = initial_conditions_config.get('R_in_refuge', 80.0)
        R_out = initial_conditions_config.get('R_outside', 20.0)
        F_in = initial_conditions_config.get('F_in_refuge', 2.0)
        F_out = initial_conditions_config.get('F_outside', 15.0)
        R0, F0 = ic.prey_in_refuge_predators_outside(
            refuge_map, R_in, R_out, F_in, F_out
        )

    # Determine if stochastic simulation
    is_stochastic = params.get('stochastic', {}).get('sigma_R', 0.0) > 0 or \
                    params.get('stochastic', {}).get('sigma_F', 0.0) > 0

    # Create solver
    if is_stochastic:
        solver = StochasticSolver(L=L, nx=nx, ny=ny)
        solver_type = "Stochastic"
    else:
        solver = LotkaVolterraSolver(L=L, nx=nx, ny=ny)
        solver_type = "Deterministic"

    if verbose:
        print(f"Solver type: {solver_type}")

    # Extract simulation parameters
    T = params.get('temporal', {}).get('T', 200.0)
    dt = params.get('temporal', {}).get('dt', 0.01)

    if verbose:
        print(f"Simulation time: T = {T}")
        print(f"Time step: dt = {dt}")
        print(f"Number of time steps: {int(T/dt)}")
        print("\nRunning simulation...")

    # Run simulation
    result = solver.solve(
        R0=R0,
        F0=F0,
        params=params,
        refuge_map=refuge_map,
        T=T,
        dt=dt,
        save_every=save_every
    )

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"Simulation completed in {elapsed_time:.2f} seconds")
        print(f"Final prey population: {np.sum(result['R'][-1]):.2f}")
        print(f"Final predator population: {np.sum(result['F'][-1]):.2f}")

    # Add metadata
    result['metadata'] = {
        'scenario_name': scenario_name,
        'solver_type': solver_type,
        'is_stochastic': is_stochastic,
        'elapsed_time': elapsed_time,
        'timestamp': datetime.now().isoformat(),
        'refuge_config': refuge_config,
        'initial_conditions_config': initial_conditions_config
    }

    return result


def run_all_scenarios(
    base_params_file: Optional[str] = None,
    output_dir: str = "data/results",
    save_every: int = 100,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Run all 5 predefined simulation scenarios.

    Scenarios:
        1. Baseline: No refuges, no noise
        2. Refuges only: With refuges, no noise
        3. Noise only: No refuges, with noise
        4. Complete: With refuges and noise
        5. Parametric variation: Test different parameter values

    Parameters:
        base_params_file (str, optional): Path to base parameters JSON.
                                          If None, uses defaults.
        output_dir (str): Directory to save results
        save_every (int): Save solution every N time steps
        verbose (bool): Print progress information

    Returns:
        dict: Dictionary with results for each scenario
    """

    print("=== RUNNING ALL SIMULATION SCENARIOS ===")
    # Load base parameters
    if base_params_file is not None:
        params_manager = ModelParameters.load(base_params_file)
        base_params = params_manager.get()
    else:
        params_manager = ModelParameters()
        base_params = params_manager.get()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Scenario 1: Baseline (No refuges, no noise)
    scenario_1_params = base_params.copy()
    scenario_1_params['stochastic'] = {
        'sigma_R': 0.0,
        'sigma_F': 0.0,
        'n_realizations': 1
    }

    results['scenario_1_baseline'] = run_single_scenario(
        params=scenario_1_params,
        refuge_config={'type': 'none'},
        initial_conditions_config={'type': 'uniform', 'R0': 50.0, 'F0': 10.0},
        scenario_name="Scenario 1: Baseline (No refuges, no noise)",
        save_every=save_every,
        verbose=verbose
    )

    # Scenario 2: With refuges, no noise
    scenario_2_params = base_params.copy()
    scenario_2_params['stochastic'] = {
        'sigma_R': 0.0,
        'sigma_F': 0.0,
        'n_realizations': 1
    }

    L = scenario_2_params['spatial']['L']
    results['scenario_2_refuges'] = run_single_scenario(
        params=scenario_2_params,
        refuge_config={
            'type': 'circular',
            'centers': [(L/2, L/2), (L/4, L/4), (3*L/4, 3*L/4)],
            'radii': [1.5, 1.0, 1.0],
            'enhancement': 2.5
        },
        initial_conditions_config={'type': 'uniform', 'R0': 40.0, 'F0': 8.0},
        scenario_name="Scenario 2: With Refuges, No Noise",
        save_every=save_every,
        verbose=verbose
    )

    # Scenario 3: No refuges, with noise
    scenario_3_params = base_params.copy()
    scenario_3_params['stochastic'] = {
        'sigma_R': 0.08,
        'sigma_F': 0.08,
        'n_realizations': 50
    }

    results['scenario_3_stochastic'] = run_single_scenario(
        params=scenario_3_params,
        refuge_config={'type': 'none'},
        initial_conditions_config={'type': 'uniform', 'R0': 50.0, 'F0': 10.0},
        scenario_name="Scenario 3: No Refuges, With Noise",
        save_every=save_every,
        verbose=verbose
    )

    # Scenario 4: Complete (With refuges and noise)
    scenario_4_params = base_params.copy()
    scenario_4_params['stochastic'] = {
        'sigma_R': 0.06,
        'sigma_F': 0.06,
        'n_realizations': 50
    }

    results['scenario_4_complete'] = run_single_scenario(
        params=scenario_4_params,
        refuge_config={
            'type': 'circular',
            'centers': [(L/2, L/2), (L/4, L/4), (3*L/4, 3*L/4)],
            'radii': [1.5, 1.0, 1.0],
            'enhancement': 2.5
        },
        initial_conditions_config={'type': 'gaussian', 'R0': 60.0, 'F0': 12.0, 'sigma': 1.5},
        scenario_name="Scenario 4: Complete (Refuges + Noise)",
        save_every=save_every,
        verbose=verbose
    )

    # Scenario 5: Parametric variation (different attack rates)
    if verbose:
        print("Scenario 5: Parametric Variation (Attack Rate)")
    attack_rates = [0.3, 0.5, 0.7]
    scenario_5_results = {}

    iterator = tqdm(attack_rates, desc="Attack rates") if TQDM_AVAILABLE else attack_rates

    for a_val in iterator:
        scenario_5_params = base_params.copy()
        scenario_5_params['interaction']['a'] = a_val
        scenario_5_params['stochastic'] = {
            'sigma_R': 0.0,
            'sigma_F': 0.0,
            'n_realizations': 1
        }

        result = run_single_scenario(
            params=scenario_5_params,
            refuge_config={
                'type': 'central_strip',
                'width': 2.0,
                'orientation': 'vertical',
                'enhancement': 2.0
            },
            initial_conditions_config={'type': 'uniform', 'R0': 45.0, 'F0': 9.0},
            scenario_name=f"Scenario 5: Parametric (a={a_val})",
            save_every=save_every,
            verbose=False
        )

        scenario_5_results[f'a_{a_val}'] = result

    results['scenario_5_parametric'] = scenario_5_results

    if verbose:
        print("=== SAVING RESULTS ====")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for scenario_name, result in results.items():
        if scenario_name == 'scenario_5_parametric':
            # Save each parametric variation
            for param_name, param_result in result.items():
                filename = output_path / f"{scenario_name}_{param_name}_{timestamp}.npz"
                _save_result(param_result, filename, verbose)
        else:
            filename = output_path / f"{scenario_name}_{timestamp}.npz"
            _save_result(result, filename, verbose)

    # Save metadata summary
    metadata_file = output_path / f"run_metadata_{timestamp}.json"
    metadata_summary = {
        'timestamp': timestamp,
        'base_parameters': base_params,
        'scenarios': {
            name: res.get('metadata', {}) if not isinstance(res, dict) or 'metadata' in res
            else {k: v.get('metadata', {}) for k, v in res.items()}
            for name, res in results.items()
        }
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata_summary, f, indent=2)

    if verbose:
        print(f"Metadata saved to: {metadata_file}")
        print("¡ALL SCENARIOS COMPLETED SUCCESSFULLY!")

    return results


def _save_result(result: Dict, filename: Path, verbose: bool = True):
    """
    Save a single result to file.

    Parameters:
        result (dict): Simulation result
        filename (Path): Output filename
        verbose (bool): Print save information
    """
    # Extract arrays and metadata
    R = result['R']
    F = result['F']
    times = result['times']
    refuge_map = result.get('refuge_map', None)

    # Prepare data for saving
    save_dict = {
        'R': R,
        'F': F,
        'times': times,
    }

    if refuge_map is not None:
        save_dict['refuge_map'] = refuge_map

    # Save arrays
    np.savez_compressed(filename, **save_dict)

    # Save metadata separately
    metadata_file = filename.with_suffix('.json')
    metadata = {
        'params': result.get('params', {}),
        'metadata': result.get('metadata', {})
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"Results saved to: {filename}")
        print(f"Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    # Run all scenarios
    results = run_all_scenarios(
        base_params_file="simulations/config/parameters.json",
        output_dir="data/results",
        save_every=100,
        verbose=True
    )

    print("Execution complete!")
    print(f"Results saved in: data/results/")
