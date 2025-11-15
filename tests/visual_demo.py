"""
Demonstration script for refuges and model functionality.

This script demonstrates all the features implemented:
- Refuge generation
- Parameter management
- Initial conditions
- Scenario execution
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.refuges import RefugeGenerator
from src.model import ModelParameters, InitialConditions
from simulations.run_scenarios import run_single_scenario

def demo_refuge_generation():
    """
    Demonstrate all refuge generation types.
    """
    print("\n === DEMO: Refuge Generation ===")

    gen = RefugeGenerator(L=10.0, nx=100, ny=100)

    # Create figure with all refuge types
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Circular refuges
    gen.reset()
    gen.generate_circular_refuges(
        centers=[(5.0, 5.0), (3.0, 7.0)],
        radii=[1.5, 1.0],
        enhancement=2.5
    )
    im1 = axes[0, 0].imshow(gen.refuge_map, origin='lower', cmap='YlGn',
                            extent=[0, 10, 0, 10])
    axes[0, 0].set_title('Circular Refuges', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])

    # Rectangular refuges
    gen.reset()
    gen.generate_rectangular_refuges(
        corners=[(1.0, 1.0), (6.0, 6.0)],
        sizes=[(2.0, 2.0), (1.5, 1.5)],
        enhancement=2.0
    )
    im2 = axes[0, 1].imshow(gen.refuge_map, origin='lower', cmap='YlGn',
                            extent=[0, 10, 0, 10])
    axes[0, 1].set_title('Rectangular Refuges', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])

    # Central strip
    gen.reset()
    gen.generate_central_strip(width=2.5, orientation='vertical', enhancement=2.0)
    im3 = axes[0, 2].imshow(gen.refuge_map, origin='lower', cmap='YlGn',
                            extent=[0, 10, 0, 10])
    axes[0, 2].set_title('Central Strip (Vertical)', fontweight='bold', fontsize=12)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 2])

    # Corner refuges
    gen.reset()
    gen.generate_corner_refuges(size=1.5, corners=['all'], enhancement=3.0)
    im4 = axes[1, 0].imshow(gen.refuge_map, origin='lower', cmap='YlGn',
                            extent=[0, 10, 0, 10])
    axes[1, 0].set_title('Corner Refuges', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0])

    # Random refuges
    gen.reset()
    gen.generate_random_refuges(
        num_refuges=6,
        min_size=0.4,
        max_size=1.0,
        shape='circular',
        enhancement=2.5,
        seed=42
    )
    im5 = axes[1, 1].imshow(gen.refuge_map, origin='lower', cmap='YlGn',
                            extent=[0, 10, 0, 10])
    axes[1, 1].set_title('Random Refuges', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 1])

    # Combined refuges
    gen.reset()
    gen.generate_circular_refuges(
        centers=[(5.0, 5.0)],
        radii=[1.5],
        enhancement=2.5,
        additive=False
    )
    gen.generate_corner_refuges(size=1.0, corners=['all'], enhancement=2.0, additive=True)
    im6 = axes[1, 2].imshow(gen.refuge_map, origin='lower', cmap='YlGn',
                            extent=[0, 10, 0, 10])
    axes[1, 2].set_title('Combined Refuges', fontweight='bold', fontsize=12)
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im6, ax=axes[1, 2])

    fig.suptitle('Refuge Generation Types', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('data/demo_refuges.png', dpi=150, bbox_inches='tight')
    print("Refuge types demonstrated")
    print("  Figure saved to: data/demo_refuges.png\n")

    return fig

def demo_initial_conditions():
    """
    Demonstrate all initial condition types.
    """
    print("\n === DEMO: Initial Conditions ===")

    ic = InitialConditions(L=10.0, nx=100, ny=100)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Uniform
    R0, F0 = ic.uniform(R0=50.0, F0=10.0)
    im1 = axes[0, 0].imshow(R0, origin='lower', cmap='Blues', extent=[0, 10, 0, 10])
    axes[0, 0].set_title('Uniform - Prey', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])
    im2 = axes[1, 0].imshow(F0, origin='lower', cmap='Reds', extent=[0, 10, 0, 10])
    axes[1, 0].set_title('Uniform - Predator', fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 0])

    # Gaussian
    R0, F0 = ic.gaussian(R0=100.0, F0=20.0, sigma=1.5)
    im3 = axes[0, 1].imshow(R0, origin='lower', cmap='Blues', extent=[0, 10, 0, 10])
    axes[0, 1].set_title('Gaussian - Prey', fontweight='bold')
    plt.colorbar(im3, ax=axes[0, 1])
    im4 = axes[1, 1].imshow(F0, origin='lower', cmap='Reds', extent=[0, 10, 0, 10])
    axes[1, 1].set_title('Gaussian - Predator', fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1])

    # Random spatial
    R0, F0 = ic.random_spatial(R_mean=50.0, F_mean=10.0, R_std=15.0, F_std=3.0, seed=42)
    im5 = axes[0, 2].imshow(R0, origin='lower', cmap='Blues', extent=[0, 10, 0, 10])
    axes[0, 2].set_title('Random Spatial - Prey', fontweight='bold')
    plt.colorbar(im5, ax=axes[0, 2])
    im6 = axes[1, 2].imshow(F0, origin='lower', cmap='Reds', extent=[0, 10, 0, 10])
    axes[1, 2].set_title('Random Spatial - Predator', fontweight='bold')
    plt.colorbar(im6, ax=axes[1, 2])

    # Prey in refuge
    gen = RefugeGenerator(L=10.0, nx=100, ny=100)
    refuge_map = gen.generate_circular_refuges([(5.0, 5.0)], [2.0], 2.5)
    R0, F0 = ic.prey_in_refuge_predators_outside(refuge_map, 80.0, 20.0, 2.0, 15.0)
    im7 = axes[0, 3].imshow(R0, origin='lower', cmap='Blues', extent=[0, 10, 0, 10])
    axes[0, 3].set_title('Prey in Refuge - Prey', fontweight='bold')
    plt.colorbar(im7, ax=axes[0, 3])
    im8 = axes[1, 3].imshow(F0, origin='lower', cmap='Reds', extent=[0, 10, 0, 10])
    axes[1, 3].set_title('Prey in Refuge - Predator', fontweight='bold')
    plt.colorbar(im8, ax=axes[1, 3])

    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.suptitle('Initial Condition Types', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('data/demo_initial_conditions.png', dpi=150, bbox_inches='tight')
    print("Initial condition types demonstrated")
    print("  Figure saved to: data/demo_initial_conditions.png\n")

    return fig

def demo_parameter_management():
    """
    Demonstrate parameter management.
    """
    print("\n === DEMO: Parameter Management ===")

    # Default parameters
    print("Default Parameters:")
    params = ModelParameters()
    print(params)
    print()

    # Custom parameters
    print("Custom Parameters:")
    custom = {
        'demographic': {'r': 1.5, 'K': 120.0},
        'diffusion': {'D_R': 0.15}
    }
    params_custom = ModelParameters(custom)
    print(f"  r = {params_custom.params['demographic']['r']}")
    print(f"  K = {params_custom.params['demographic']['K']}")
    print(f"  D_R = {params_custom.params['diffusion']['D_R']}")
    print()

    # Save and load
    print("Save/Load Functionality:")
    params.save('data/demo_params.json')
    print("  Saved to: data/demo_params.json")
    loaded = ModelParameters.load('data/demo_params.json')
    print("  Loaded successfully")
    print()

    print("Parameter management demonstrated\n")

    return params

def demo_scenario_execution():
    """
    Demonstrate scenario execution.
    """
    print("\n === DEMO: Scenario Execution ===")

    # Create simple parameters for fast demo
    params = ModelParameters()
    params_dict = params.get()

    # Reduce grid size and time for faster execution
    params_dict['spatial']['nx'] = 50
    params_dict['spatial']['ny'] = 50
    params_dict['temporal']['T'] = 50.0
    params_dict['stochastic']['sigma_R'] = 0.0
    params_dict['stochastic']['sigma_F'] = 0.0

    print("Running example scenario...")
    print("(Reduced grid size and time for demonstration)\n")

    result = run_single_scenario(
        params=params_dict,
        refuge_config={
            'type': 'circular',
            'centers': [(5.0, 5.0)],
            'radii': [2.0],
            'enhancement': 2.5
        },
        initial_conditions_config={
            'type': 'gaussian',
            'R0': 60.0,
            'F0': 12.0,
            'sigma': 1.5
        },
        scenario_name="Demo Scenario",
        save_every=50,
        verbose=True
    )

    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    times_to_plot = [0, len(result['times'])//2, -1]
    time_labels = ['Initial', 'Middle', 'Final']

    for i, (t_idx, label) in enumerate(zip(times_to_plot, time_labels)):
        # Prey
        im1 = axes[0, i].imshow(result['R'][t_idx], origin='lower', cmap='Blues',
                                extent=[0, 10, 0, 10])
        axes[0, i].set_title(f'Prey - {label} (t={result["times"][t_idx]:.1f})',
                            fontweight='bold')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, i])

        # Predator
        im2 = axes[1, i].imshow(result['F'][t_idx], origin='lower', cmap='Reds',
                                extent=[0, 10, 0, 10])
        axes[1, i].set_title(f'Predator - {label} (t={result["times"][t_idx]:.1f})',
                            fontweight='bold')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1, i])

    fig.suptitle('Scenario Execution Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('data/demo_scenario_execution.png', dpi=150, bbox_inches='tight')
    print("\nScenario execution demonstrated")
    print("  Figure saved to: data/demo_scenario_execution.png\n")

    return result, fig

def main():
    """
    Run all demonstrations.
    """
    print("### FUNCTIONALITY DEMONSTRATION ###")

    # Create output directory
    Path('data').mkdir(exist_ok=True)

    # Run demonstrations
    fig1 = demo_refuge_generation()
    fig2 = demo_initial_conditions()
    params = demo_parameter_management()
    result, fig3 = demo_scenario_execution()

    print("¡ALL DEMONSTRATIONS COMPLETED!")
    print("\nGenerated files:")
    print("  - data/demo_refuges.png")
    print("  - data/demo_initial_conditions.png")
    print("  - data/demo_params.json")
    print("  - data/demo_scenario_execution.png")
    print("\nAll functionality successfully demonstrated!")


if __name__ == "__main__":
    main()
    plt.show()
