"""
Script 1: Animación con refugios cuadrados en las esquinas
Muestra partículas (puntitos) con spawn aleatorio para presas y depredadores
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.solver import LotkaVolterraSolver
from src.model import ModelParameters
from src.refuges import RefugeGenerator


def generar_condiciones_iniciales_aleatorias(nx, ny, L, n_prey=800, n_predators=150):
    """
    Genera condiciones iniciales con posiciones aleatorias uniformemente distribuidas.

    Parameters:
        nx, ny: resolución de la malla
        L: tamaño del dominio
        n_prey: número inicial de presas
        n_predators: número inicial de depredadores

    Returns:
        R0, F0: arrays de densidades iniciales
    """
    R0 = np.zeros((ny, nx))
    F0 = np.zeros((ny, nx))

    # Spawn aleatorio de presas
    for _ in range(n_prey):
        i = np.random.randint(0, ny)
        j = np.random.randint(0, nx)
        R0[i, j] += 1.0

    # Spawn aleatorio de depredadores
    for _ in range(n_predators):
        i = np.random.randint(0, ny)
        j = np.random.randint(0, nx)
        F0[i, j] += 1.0

    # Suavizar ligeramente con un kernel gaussiano
    from scipy.ndimage import gaussian_filter
    R0 = gaussian_filter(R0, sigma=1.0)
    F0 = gaussian_filter(F0, sigma=1.0)

    return R0, F0


def sample_particles(density, n_particles, X, Y, L, nx, ny):
    """
    Muestrea partículas desde un campo de densidad.
    """
    total = np.sum(density)
    if total < 1e-10:
        return np.array([]), np.array([])

    prob = density.flatten() / total
    indices = np.random.choice(len(prob), size=min(n_particles, int(total*5)),
                               p=prob, replace=True)

    i_coords = indices // nx
    j_coords = indices % nx

    x_particles = X[i_coords, j_coords] + np.random.uniform(-L/(2*nx), L/(2*nx), len(i_coords))
    y_particles = Y[i_coords, j_coords] + np.random.uniform(-L/(2*ny), L/(2*ny), len(i_coords))

    return x_particles, y_particles


def crear_animacion_refugios_esquinas():
    """
    Crea animación con refugios cuadrados en las 4 esquinas.
    """
    print("\n" + "="*70)
    print("GENERANDO ANIMACIÓN: REFUGIOS EN ESQUINAS")
    print("="*70)

    # Parámetros
    L = 10.0
    nx, ny = 50, 50
    T = 30.0

    params = ModelParameters()
    params_dict = params.get()
    params_dict['spatial']['L'] = L
    params_dict['spatial']['nx'] = nx
    params_dict['spatial']['ny'] = ny
    params_dict['temporal']['T'] = T
    params_dict['stochastic']['sigma_R'] = 0.0
    params_dict['stochastic']['sigma_F'] = 0.0

    # Condiciones iniciales aleatorias
    print("\n1. Generando condiciones iniciales aleatorias...")
    R0, F0 = generar_condiciones_iniciales_aleatorias(nx, ny, L, n_prey=1000, n_predators=200)
    print(f"   ✓ Presas iniciales: {np.sum(R0):.0f}")
    print(f"   ✓ Depredadores iniciales: {np.sum(F0):.0f}")

    # Generar refugios en las 4 esquinas
    print("\n2. Generando refugios cuadrados en esquinas...")
    gen = RefugeGenerator(L=L, nx=nx, ny=ny)

    # Tamaño de cada refugio cuadrado
    square_size = 2.0

    # Esquinas: (x_corner, y_corner)
    corners = [
        (0.0, 0.0),                    # Esquina inferior izquierda
        (L - square_size, 0.0),        # Esquina inferior derecha
        (0.0, L - square_size),        # Esquina superior izquierda
        (L - square_size, L - square_size)  # Esquina superior derecha
    ]

    sizes = [(square_size, square_size)] * 4

    refuge_map = gen.generate_rectangular_refuges(
        corners=corners,
        sizes=sizes,
        enhancement=2.5
    )
    print(f"   ✓ 4 refugios cuadrados creados (tamaño: {square_size}x{square_size})")

    # Ejecutar simulación
    print("\n3. Ejecutando simulación...")
    solver = LotkaVolterraSolver(L=L, nx=nx, ny=ny)
    result = solver.solve(
        R0=R0,
        F0=F0,
        params=params_dict,
        refuge_map=refuge_map,
        T=T,
        dt=params_dict['temporal']['dt'],
        save_every=30
    )
    print(f"   ✓ Simulación completada ({len(result['times'])} frames)")

    # Crear animación
    print("\n4. Generando animación de partículas...")
    print("   (Esto puede tardar 1-2 minutos...)")

    R_timeseries = result['R']
    F_timeseries = result['F']
    times = result['times']
    n_times = len(times)

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y)

    # Calcular poblaciones totales para phase portrait
    dx = L / (nx - 1)
    dy = L / (ny - 1)
    R_total = [np.sum(R) * dx * dy for R in R_timeseries]
    F_total = [np.sum(F) * dx * dy for F in F_timeseries]

    # Configurar figura con 2 paneles
    fig = plt.figure(figsize=(16, 8))
    ax_spatial = plt.subplot(1, 2, 1)
    ax_phase = plt.subplot(1, 2, 2)

    # Panel izquierdo: Vista espacial
    refuge_display = ax_spatial.contourf(X, Y, refuge_map, levels=[1.0, 1.5, 3.0],
                                 colors=['white', 'lightgreen'], alpha=0.3)

    scatter_prey = ax_spatial.scatter([], [], c='blue', s=20, alpha=0.6, label='Presas')
    scatter_pred = ax_spatial.scatter([], [], c='red', s=30, alpha=0.8, marker='^', label='Depredadores')

    ax_spatial.set_xlim(0, L)
    ax_spatial.set_ylim(0, L)
    ax_spatial.set_xlabel('x', fontsize=12)
    ax_spatial.set_ylabel('y', fontsize=12)
    ax_spatial.set_title('Refugios en Esquinas - Spawn Aleatorio\nt = 0.00 s', fontsize=14, fontweight='bold')
    ax_spatial.legend(loc='upper right')
    ax_spatial.set_aspect('equal')

    # Panel derecho: Phase portrait
    ax_phase.plot(R_total, F_total, 'gray', alpha=0.3, linewidth=1, label='Trayectoria completa')
    line_phase, = ax_phase.plot([], [], 'b-', linewidth=2, label='Trayectoria actual')
    point_phase, = ax_phase.plot([], [], 'ro', markersize=10, label='Posición actual')
    ax_phase.plot(R_total[0], F_total[0], 'go', markersize=12, label='Inicio', zorder=5)

    ax_phase.set_xlim(0, max(R_total) * 1.1)
    ax_phase.set_ylim(0, max(F_total) * 1.1)
    ax_phase.set_xlabel('Población de Presas (R)', fontsize=12)
    ax_phase.set_ylabel('Población de Depredadores (F)', fontsize=12)
    ax_phase.set_title('Phase Portrait (R vs F)', fontsize=14, fontweight='bold')
    ax_phase.legend(loc='best', fontsize=9)
    ax_phase.grid(True, alpha=0.3)

    plt.tight_layout()

    def init():
        scatter_prey.set_offsets(np.empty((0, 2)))
        scatter_pred.set_offsets(np.empty((0, 2)))
        line_phase.set_data([], [])
        point_phase.set_data([], [])
        return scatter_prey, scatter_pred, line_phase, point_phase

    def update(frame):
        # Muestrear partículas
        x_prey, y_prey = sample_particles(R_timeseries[frame], 300, X, Y, L, nx, ny)
        x_pred, y_pred = sample_particles(F_timeseries[frame], 80, X, Y, L, nx, ny)

        # Actualizar scatter plots
        if len(x_prey) > 0:
            scatter_prey.set_offsets(np.column_stack([x_prey, y_prey]))
        else:
            scatter_prey.set_offsets(np.empty((0, 2)))

        if len(x_pred) > 0:
            scatter_pred.set_offsets(np.column_stack([x_pred, y_pred]))
        else:
            scatter_pred.set_offsets(np.empty((0, 2)))

        # Actualizar phase portrait
        line_phase.set_data(R_total[:frame+1], F_total[:frame+1])
        point_phase.set_data([R_total[frame]], [F_total[frame]])

        # Actualizar título
        ax_spatial.set_title(f'Refugios en Esquinas - Spawn Aleatorio\nt = {times[frame]:.2f} s',
                     fontsize=14, fontweight='bold')

        if frame % 10 == 0:
            print(f"   Frame {frame}/{n_times}")

        return scatter_prey, scatter_pred, line_phase, point_phase

    # Crear animación
    anim = FuncAnimation(fig, update, frames=n_times, init_func=init,
                        interval=1000, blit=True, repeat=True)

    # Guardar
    output_dir = Path('animaciones')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / '1_refugios_esquinas.gif'

    print(f"\n5. Guardando animación en: {output_path}")
    writer = PillowWriter(fps=0.90)
    anim.save(output_path, writer=writer)

    plt.close()

    print(f"\n✓ Animación guardada exitosamente!")
    print(f"  Archivo: {output_path}")
    print("="*70)


if __name__ == "__main__":
    crear_animacion_refugios_esquinas()
