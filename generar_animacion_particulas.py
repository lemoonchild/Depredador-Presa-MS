"""
Script para generar animaciones con partículas (puntitos) de presas y depredadores.

Muestra la evolución del sistema con:
- Presas = círculos azules
- Depredadores = triángulos rojos
- Comparación lado a lado: sin refugios vs con refugios
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.solver import LotkaVolterraSolver
from src.model import ModelParameters, InitialConditions
from src.refuges import RefugeGenerator


def generar_animacion_particulas():
    """
    Genera dos animaciones de partículas para comparar escenarios.
    """
    print("\n" + "="*70)
    print("GENERANDO ANIMACIONES DE PARTÍCULAS")
    print("="*70)

    # Configurar parámetros
    print("\n1. Configurando simulaciones...")
    params = ModelParameters()
    params_dict = params.get()

    # Reducir para rapidez
    params_dict['spatial']['nx'] = 40
    params_dict['spatial']['ny'] = 40
    params_dict['temporal']['T'] = 30.0
    params_dict['stochastic']['sigma_R'] = 0.0
    params_dict['stochastic']['sigma_F'] = 0.0

    L = params_dict['spatial']['L']
    nx = params_dict['spatial']['nx']
    ny = params_dict['spatial']['ny']

    # Condiciones iniciales
    ic = InitialConditions(L=L, nx=nx, ny=ny)
    R0, F0 = ic.gaussian(R0=60.0, F0=12.0, sigma=1.5)

    # ESCENARIO 1: Sin refugios
    print("\n2. Ejecutando simulación SIN refugios...")
    solver1 = LotkaVolterraSolver(L=L, nx=nx, ny=ny)
    result_no_refuge = solver1.solve(
        R0=R0,
        F0=F0,
        params=params_dict,
        refuge_map=None,
        T=params_dict['temporal']['T'],
        dt=params_dict['temporal']['dt'],
        save_every=30
    )
    print(f"   ✓ Completado ({len(result_no_refuge['times'])} frames)")

    # ESCENARIO 2: Con refugios
    print("\n3. Ejecutando simulación CON refugios...")
    gen = RefugeGenerator(L=L, nx=nx, ny=ny)
    refuge_map = gen.generate_circular_refuges(
        centers=[(L/2, L/2)],
        radii=[2.0],
        enhancement=2.5
    )

    solver2 = LotkaVolterraSolver(L=L, nx=nx, ny=ny)
    result_with_refuge = solver2.solve(
        R0=R0,
        F0=F0,
        params=params_dict,
        refuge_map=refuge_map,
        T=params_dict['temporal']['T'],
        dt=params_dict['temporal']['dt'],
        save_every=30
    )
    print(f"   ✓ Completado ({len(result_with_refuge['times'])} frames)")

    # Crear carpeta de salida
    output_dir = Path('animaciones')
    output_dir.mkdir(exist_ok=True)

    # ANIMACIÓN 1: Solo con refugios (individual)
    print("\n4. Generando animación individual CON refugios...")
    print("   (Esto puede tardar 1-2 minutos...)")
    anim1 = crear_animacion_particulas_individual(
        result_with_refuge['R'],
        result_with_refuge['F'],
        result_with_refuge['times'],
        refuge_map,
        L,
        n_particles_prey=200,
        n_particles_predator=50,
        save_path=output_dir / 'particulas_con_refugio.gif'
    )
    plt.close()

    # ANIMACIÓN 2: Comparación lado a lado
    print("\n5. Generando animación de COMPARACIÓN (sin vs con refugios)...")
    print("   (Esta puede tardar 2-3 minutos...)")
    anim2 = crear_animacion_comparacion(
        result_no_refuge,
        result_with_refuge,
        L,
        save_path=output_dir / 'comparacion_particulas.gif'
    )
    plt.close()

    print("\n" + "="*70)
    print("¡ANIMACIONES DE PARTÍCULAS GENERADAS!")
    print("="*70)
    print(f"\nArchivos creados en: {output_dir.absolute()}")
    print("\nAnimaciones generadas:")
    print("  1. particulas_con_refugio.gif - Partículas con refugios")
    print("  2. comparacion_particulas.gif - Comparación lado a lado")
    print("\nQué verás:")
    print("  • Círculos azules = Presas individuales")
    print("  • Triángulos rojos = Depredadores individuales")
    print("  • Zona verde = Refugio (solo en 'con refugio')")
    print("  • Observa cómo las presas se concentran en el refugio!")


def crear_animacion_particulas_individual(
    R_timeseries, F_timeseries, times, refuge_map, L,
    n_particles_prey=200, n_particles_predator=50,
    save_path=None
):
    """Crea animación de partículas individual."""
    n_times, ny, nx = R_timeseries.shape

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Fondo con refugio
    if refuge_map is not None:
        ax.imshow(refuge_map, origin='lower', extent=[0, L, 0, L],
                 cmap='Greens', alpha=0.3, interpolation='bilinear')
        ax.contour(refuge_map, levels=[1.5], colors='green',
                  linewidths=2, alpha=0.5, extent=[0, L, 0, L])

    # Scatter plots
    prey_scatter = ax.scatter([], [], c='blue', s=20, alpha=0.7,
                             marker='o', edgecolors='darkblue',
                             linewidths=0.5, label='Presas')
    pred_scatter = ax.scatter([], [], c='red', s=30, alpha=0.8,
                             marker='^', edgecolors='darkred',
                             linewidths=0.5, label='Depredadores')

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    pop_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                      fontsize=10, verticalalignment='top',
                      horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    fig.suptitle('Animación de Partículas - Con Refugios', fontsize=14, fontweight='bold')
    plt.tight_layout()

    def sample_particles(density, n_particles):
        """Muestrea partículas desde la densidad."""
        total = np.sum(density)
        if total < 1e-10:
            return np.array([]), np.array([])

        prob = density.flatten() / total
        indices = np.random.choice(len(prob), size=min(n_particles, int(total)),
                                  p=prob, replace=True)

        i_coords = indices // nx
        j_coords = indices % nx

        x_particles = X[i_coords, j_coords] + np.random.uniform(-L/(2*nx), L/(2*nx), len(i_coords))
        y_particles = Y[i_coords, j_coords] + np.random.uniform(-L/(2*ny), L/(2*ny), len(i_coords))

        return x_particles, y_particles

    def update(frame):
        """Actualizar frame."""
        x_prey, y_prey = sample_particles(R_timeseries[frame], n_particles_prey)
        x_pred, y_pred = sample_particles(F_timeseries[frame], n_particles_predator)

        if len(x_prey) > 0:
            prey_scatter.set_offsets(np.c_[x_prey, y_prey])
        else:
            prey_scatter.set_offsets(np.empty((0, 2)))

        if len(x_pred) > 0:
            pred_scatter.set_offsets(np.c_[x_pred, y_pred])
        else:
            pred_scatter.set_offsets(np.empty((0, 2)))

        time_text.set_text(f't = {times[frame]:.2f}')

        dx = L / (nx - 1)
        dy = L / (ny - 1)
        total_prey = np.sum(R_timeseries[frame]) * dx * dy
        total_pred = np.sum(F_timeseries[frame]) * dx * dy
        pop_text.set_text(f'Presas: {total_prey:.0f}\nDepred.: {total_pred:.0f}')

        return prey_scatter, pred_scatter, time_text, pop_text

    anim = FuncAnimation(fig, update, frames=n_times, interval=1000,
                        blit=True, repeat=True)

    if save_path:
        print(f"   Guardando en {save_path}...")
        writer = PillowWriter(fps=0.90)
        anim.save(str(save_path), writer=writer)
        print(f"   ✓ Guardado exitosamente!")

    return anim


def crear_animacion_comparacion(result_no_refuge, result_with_refuge, L, save_path=None):
    """Crea animación comparativa lado a lado."""
    R_no = result_no_refuge['R']
    F_no = result_no_refuge['F']
    times = result_no_refuge['times']

    R_yes = result_with_refuge['R']
    F_yes = result_with_refuge['F']
    refuge_map = result_with_refuge.get('refuge_map')

    n_times, ny, nx = R_no.shape

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y)

    fig, (ax_no, ax_yes) = plt.subplots(1, 2, figsize=(16, 7))

    # Izquierda - Sin refugio
    ax_no.set_xlim(0, L)
    ax_no.set_ylim(0, L)
    ax_no.set_xlabel('x', fontsize=11)
    ax_no.set_ylabel('y', fontsize=11)
    ax_no.set_title('SIN Refugio', fontweight='bold', fontsize=12)
    ax_no.set_aspect('equal')

    prey_scatter_no = ax_no.scatter([], [], c='blue', s=15, alpha=0.6,
                                   marker='o', label='Presas')
    pred_scatter_no = ax_no.scatter([], [], c='red', s=25, alpha=0.7,
                                   marker='^', label='Depredadores')
    ax_no.legend(loc='upper right', fontsize=9)

    # Derecha - Con refugio
    if refuge_map is not None:
        ax_yes.imshow(refuge_map, origin='lower', extent=[0, L, 0, L],
                     cmap='Greens', alpha=0.3, interpolation='bilinear')
        ax_yes.contour(refuge_map, levels=[1.5], colors='green',
                      linewidths=2, alpha=0.5, extent=[0, L, 0, L])

    ax_yes.set_xlim(0, L)
    ax_yes.set_ylim(0, L)
    ax_yes.set_xlabel('x', fontsize=11)
    ax_yes.set_ylabel('y', fontsize=11)
    ax_yes.set_title('CON Refugio', fontweight='bold', fontsize=12)
    ax_yes.set_aspect('equal')

    prey_scatter_yes = ax_yes.scatter([], [], c='blue', s=15, alpha=0.6,
                                     marker='o', label='Presas')
    pred_scatter_yes = ax_yes.scatter([], [], c='red', s=25, alpha=0.7,
                                     marker='^', label='Depredadores')
    ax_yes.legend(loc='upper right', fontsize=9)

    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=13, fontweight='bold')

    pop_text_no = ax_no.text(0.02, 0.98, '', transform=ax_no.transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    pop_text_yes = ax_yes.text(0.02, 0.98, '', transform=ax_yes.transAxes,
                              fontsize=9, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    fig.suptitle('Comparación de Partículas: Sin vs Con Refugio',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    def sample_particles(density, n_particles):
        total = np.sum(density)
        if total < 1e-10:
            return np.array([]), np.array([])

        prob = density.flatten() / total
        indices = np.random.choice(len(prob), size=min(n_particles, int(total)),
                                  p=prob, replace=True)

        i_coords = indices // nx
        j_coords = indices % nx

        x_particles = X[i_coords, j_coords] + np.random.uniform(-L/(2*nx), L/(2*nx), len(i_coords))
        y_particles = Y[i_coords, j_coords] + np.random.uniform(-L/(2*ny), L/(2*ny), len(i_coords))

        return x_particles, y_particles

    def update(frame):
        # Sin refugio
        x_prey_no, y_prey_no = sample_particles(R_no[frame], 150)
        x_pred_no, y_pred_no = sample_particles(F_no[frame], 40)

        if len(x_prey_no) > 0:
            prey_scatter_no.set_offsets(np.c_[x_prey_no, y_prey_no])
        else:
            prey_scatter_no.set_offsets(np.empty((0, 2)))

        if len(x_pred_no) > 0:
            pred_scatter_no.set_offsets(np.c_[x_pred_no, y_pred_no])
        else:
            pred_scatter_no.set_offsets(np.empty((0, 2)))

        # Con refugio
        x_prey_yes, y_prey_yes = sample_particles(R_yes[frame], 150)
        x_pred_yes, y_pred_yes = sample_particles(F_yes[frame], 40)

        if len(x_prey_yes) > 0:
            prey_scatter_yes.set_offsets(np.c_[x_prey_yes, y_prey_yes])
        else:
            prey_scatter_yes.set_offsets(np.empty((0, 2)))

        if len(x_pred_yes) > 0:
            pred_scatter_yes.set_offsets(np.c_[x_pred_yes, y_pred_yes])
        else:
            pred_scatter_yes.set_offsets(np.empty((0, 2)))

        time_text.set_text(f't = {times[frame]:.2f}')

        dx = L / (nx - 1)
        dy = L / (ny - 1)

        total_prey_no = np.sum(R_no[frame]) * dx * dy
        total_pred_no = np.sum(F_no[frame]) * dx * dy
        pop_text_no.set_text(f'Presas: {total_prey_no:.0f}\nDepred.: {total_pred_no:.0f}')

        total_prey_yes = np.sum(R_yes[frame]) * dx * dy
        total_pred_yes = np.sum(F_yes[frame]) * dx * dy
        pop_text_yes.set_text(f'Presas: {total_prey_yes:.0f}\nDepred.: {total_pred_yes:.0f}')

        return (prey_scatter_no, pred_scatter_no, prey_scatter_yes,
                pred_scatter_yes, time_text, pop_text_no, pop_text_yes)

    anim = FuncAnimation(fig, update, frames=n_times, interval=1000,
                        blit=True, repeat=True)

    if save_path:
        print(f"   Guardando en {save_path}...")
        writer = PillowWriter(fps=0.97)
        anim.save(str(save_path), writer=writer)
        print(f"   ✓ Guardado exitosamente!")

    return anim


if __name__ == "__main__":
    generar_animacion_particulas()
    print("\n¡Listo! Revisa la carpeta 'animaciones/' para ver tus GIFs.")
