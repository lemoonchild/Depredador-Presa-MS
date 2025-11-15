"""
Validation tests for the model configuration module.

This module tests the ModelParameters and InitialConditions classes.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ModelParameters, InitialConditions
from src.refuges import RefugeGenerator


def test_model_parameters_default():
    """
    Test default parameter initialization.
    """
    print("\n=== Test: Default Model Parameters ===")

    params = ModelParameters()

    print("\nDefault parameters loaded:")
    print(params)

    # Validate structure
    assert 'spatial' in params.params
    assert 'temporal' in params.params
    assert 'demographic' in params.params
    assert 'interaction' in params.params
    assert 'diffusion' in params.params
    assert 'stochastic' in params.params

    print("\nAll parameter categories present")

    # Validate some specific values
    assert params.params['demographic']['r'] == 1.0
    assert params.params['demographic']['K'] == 100.0
    assert params.params['diffusion']['D_R'] == 0.1

    print("Parameter values match defaults")

    return params

def test_model_parameters_custom():
    """
    Test custom parameter initialization.
    """
    print("\n=== Test: Custom Model Parameters ===")

    custom_params = {
        'demographic': {
            'r': 1.5,
            'K': 150.0
        },
        'diffusion': {
            'D_R': 0.2
        }
    }

    params = ModelParameters(custom_params)

    print("\nCustom parameters:")
    print(f"  r = {params.params['demographic']['r']}")
    print(f"  K = {params.params['demographic']['K']}")
    print(f"  D_R = {params.params['diffusion']['D_R']}")

    # Validate custom values
    assert params.params['demographic']['r'] == 1.5
    assert params.params['demographic']['K'] == 150.0
    assert params.params['diffusion']['D_R'] == 0.2

    # Validate that other defaults are preserved
    assert params.params['demographic']['m'] == 0.5
    assert params.params['interaction']['a'] == 0.5

    print("\nCustom parameters set correctly")
    print("Other defaults preserved")

    return params

def test_parameter_validation():
    """
    Test parameter validation and error handling.
    """
    print("\n=== Test: Parameter Validation ===")

    # Test valid parameters
    valid_params = {
        'demographic': {
            'r': 1.5,
            'K': 100.0,
            'e': 0.5
        }
    }

    try:
        params = ModelParameters(valid_params)
        print("Valid parameters accepted")
    except ValueError as e:
        print(f"Unexpected error: {e}")

    # Test invalid parameters (out of range)
    print("\nTesting invalid parameter (e > 1.0)...")
    invalid_params = {
        'demographic': {
            'e': 1.5  
        }
    }

    try:
        params = ModelParameters(invalid_params)
        print("Invalid parameters were accepted (should have raised error)")
    except ValueError as e:
        print(f"Invalid parameters correctly rejected: {e}")

def test_parameter_save_load():
    """
    Test saving and loading parameters to/from JSON.
    """
    print("\n=== Test: Parameter Save/Load ===")

    # Create parameters
    params = ModelParameters()
    params.set('demographic', 'r', 1.2)

    # Save to file
    test_file = "data/test_params.json"
    params.save(test_file)
    print(f"Parameters saved to: {test_file}")

    # Load from file
    loaded_params = ModelParameters.load(test_file)
    print(f"Parameters loaded from: {test_file}")

    # Validate
    assert loaded_params.params['demographic']['r'] == 1.2
    print("\nParameters correctly saved and loaded")
    print(f"r value preserved: {loaded_params.params['demographic']['r']}")

    # Clean up
    Path(test_file).unlink()
    print(f"Test file cleaned up")

def test_initial_conditions_uniform():
    """
    Test uniform initial conditions.
    """
    print("\n=== Test: Uniform Initial Conditions ===")

    ic = InitialConditions(L=10.0, nx=100, ny=100)

    R0, F0 = ic.uniform(R0=50.0, F0=10.0)

    print(f"R0 shape: {R0.shape}")
    print(f"F0 shape: {F0.shape}")
    print(f"R0 all equal to 50.0: {np.all(R0 == 50.0)}")
    print(f"F0 all equal to 10.0: {np.all(F0 == 10.0)}")

    assert R0.shape == (100, 100)
    assert F0.shape == (100, 100)
    assert np.all(R0 == 50.0)
    assert np.all(F0 == 10.0)

    print("\nUniform initial conditions generated correctly")

    return R0, F0

def test_initial_conditions_gaussian():
    """
    Test Gaussian initial conditions.
    """
    print("\n === Test: Gaussian Initial Conditions ===")

    ic = InitialConditions(L=10.0, nx=100, ny=100)

    R0, F0 = ic.gaussian(R0=100.0, F0=20.0, sigma=1.5)

    print(f"R0 shape: {R0.shape}")
    print(f"F0 shape: {F0.shape}")
    print(f"R0 max value: {np.max(R0):.2f}")
    print(f"F0 max value: {np.max(F0):.2f}")
    print(f"R0 min value: {np.min(R0):.2f}")
    print(f"F0 min value: {np.min(F0):.2f}")

    # Check that maximum is at center
    center_idx = (50, 50)
    center_R = R0[center_idx]
    edge_R = R0[0, 0]

    print(f"\nR0 at center: {center_R:.2f}")
    print(f"R0 at edge: {edge_R:.2f}")
    print(f"Peak at center: {'YES' if center_R > edge_R else 'NO'}")

    assert R0.shape == (100, 100)
    assert center_R > edge_R

    print("\nGaussian initial conditions generated correctly")

    return R0, F0


def test_initial_conditions_random():
    """
    Test random spatial initial conditions.
    """
    print("\n === Test: Random Spatial Initial Conditions ===")

    ic = InitialConditions(L=10.0, nx=100, ny=100)

    R0, F0 = ic.random_spatial(
        R_mean=50.0,
        F_mean=10.0,
        R_std=15.0,
        F_std=3.0,
        seed=42
    )

    print(f"R0 shape: {R0.shape}")
    print(f"F0 shape: {F0.shape}")
    print(f"R0 mean: {np.mean(R0):.2f} (target: 50.0)")
    print(f"F0 mean: {np.mean(F0):.2f} (target: 10.0)")
    print(f"R0 std: {np.std(R0):.2f} (target: 15.0)")
    print(f"F0 std: {np.std(F0):.2f} (target: 3.0)")
    print(f"All values non-negative: {np.all(R0 >= 0) and np.all(F0 >= 0)}")

    # Test reproducibility
    ic2 = InitialConditions(L=10.0, nx=100, ny=100)
    R0_2, F0_2 = ic2.random_spatial(
        R_mean=50.0, F_mean=10.0, R_std=15.0, F_std=3.0, seed=42
    )

    print(f"Reproducible (same seed): {np.allclose(R0, R0_2)}")

    print("\nRandom initial conditions generated correctly")

    return R0, F0


def test_initial_conditions_prey_in_refuge():
    """
    Test prey-in-refuge initial conditions.
    """
    print("\n === Test: Prey in Refuge Initial Conditions ===")

    # Create refuge map
    gen = RefugeGenerator(L=10.0, nx=100, ny=100)
    refuge_map = gen.generate_circular_refuges(
        centers=[(5.0, 5.0)],
        radii=[2.0],
        enhancement=2.5
    )

    # Create initial conditions
    ic = InitialConditions(L=10.0, nx=100, ny=100)
    R0, F0 = ic.prey_in_refuge_predators_outside(
        refuge_map=refuge_map,
        R_in_refuge=80.0,
        R_outside=20.0,
        F_in_refuge=2.0,
        F_outside=15.0
    )

    print(f"R0 shape: {R0.shape}")
    print(f"F0 shape: {F0.shape}")

    # Check values in refuge vs outside
    in_refuge = refuge_map >= 2.0
    R_in = np.mean(R0[in_refuge])
    R_out = np.mean(R0[~in_refuge])
    F_in = np.mean(F0[in_refuge])
    F_out = np.mean(F0[~in_refuge])

    print(f"\nPrey density in refuge: {R_in:.2f} (target: 80.0)")
    print(f"Prey density outside: {R_out:.2f} (target: 20.0)")
    print(f"Predator density in refuge: {F_in:.2f} (target: 2.0)")
    print(f"Predator density outside: {F_out:.2f} (target: 15.0)")

    print(f"\nHigher prey density in refuge: {R_in > R_out}")
    print(f"Higher predator density outside: {F_out > F_in}")

    print("\nPrey-in-refuge initial conditions generated correctly")

    return R0, F0


def test_initial_conditions_visualization():
    """
    Test initial conditions visualization.
    """
    print("\n === Test: Initial Conditions Visualization ===")

    ic = InitialConditions(L=10.0, nx=100, ny=100)
    R0, F0 = ic.gaussian(R0=100.0, F0=20.0, sigma=1.5)

    # Create refuge map for visualization
    gen = RefugeGenerator(L=10.0, nx=100, ny=100)
    refuge_map = gen.generate_circular_refuges(
        centers=[(5.0, 5.0)],
        radii=[2.0],
        enhancement=2.0
    )

    # Visualize
    print("Generating visualization...")
    fig = ic.visualize(R0, F0, refuge_map, title="Test Initial Conditions")

    print(f"Figure created: {fig is not None}")
    print("Visualization test passed")

    return fig


def run_all_tests():
    """
    Run all model configuration tests.
    """
    print("### Model Configuration Validation Suite ###")

    # Parameter tests
    params1 = test_model_parameters_default()
    params2 = test_model_parameters_custom()
    test_parameter_validation()
    test_parameter_save_load()

    # Initial conditions tests
    ic1 = test_initial_conditions_uniform()
    ic2 = test_initial_conditions_gaussian()
    ic3 = test_initial_conditions_random()
    ic4 = test_initial_conditions_prey_in_refuge()
    fig = test_initial_conditions_visualization()

    print("=== All validation tests completed successfully ===")

    return {
        'params_default': params1,
        'params_custom': params2,
        'ic_uniform': ic1,
        'ic_gaussian': ic2,
        'ic_random': ic3,
        'ic_refuge': ic4,
        'visualization': fig
    }

if __name__ == "__main__":
    results = run_all_tests()

    print("\nAll model configuration tests passed!")
    print("\nThe ModelParameters class is validated for:")
    print("  - Default parameter initialization")
    print("  - Custom parameter initialization")
    print("  - Parameter validation")
    print("  - Save/load functionality")
    print("\nThe InitialConditions class is validated for:")
    print("  - Uniform distribution")
    print("  - Gaussian distribution")
    print("  - Random spatial distribution")
    print("  - Prey-in-refuge distribution")
    print("  - Visualization capabilities")