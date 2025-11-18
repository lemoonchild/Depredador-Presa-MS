"""
Script 3: Comparación de parámetros influyentes
Compara lado a lado el efecto de diferentes parámetros:
- Panel izquierdo: Tasa de crecimiento baja (r = 0.5)
- Panel derecho: Tasa de crecimiento alta (r = 1.5)

Muestra cuál parámetro influye más en el comportamiento del sistema
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


def crear_comparacion_parametros():
    """
    Crea animación comparando diferentes valores de tasa de crecimiento.
    """
    print("\n" + "="*70)
    print("GENERANDO ANIMACIÓN: COMPARACIÓN DE PARÁMETROS")
    print("Comparando: Tasa de crecimiento r (baja vs alta)")
    print("="*70)

    # Parámetros comunes
    L = 10.0
    nx, ny = 50, 50
    T = 30.0

    # Condiciones iniciales aleatorias (iguales para ambos casos)
    print("\n1. Generando condiciones iniciales aleatorias...")
    np.random.seed(42)  # Misma semilla para comparación justa
    R0, F0 = generar_condiciones_iniciales_aleatorias(nx, ny, L, n_prey=1000, n_predators=200)
    print(f"   ✓ Presas iniciales: {np.sum(R0):.0f}")
    print(f"   ✓ Depredadores iniciales: {np.sum(F0):.0f}")

    # Refugio circular central (igual para ambos)
    print("\n2. Generando refugio circular central...")
    gen = RefugeGenerator(L=L, nx=nx, ny=ny)
    refuge_map = gen.generate_circular_refuges(
        centers=[(L/2, L/2)],
        radii=[2.0],
        enhancement=2.5
    )
    print(f"   ✓ Refugio circular creado")

    # SIMULACIÓN 1: Tasa de crecimiento BAJA (r = 0.5)
    print("\n3. Ejecutando simulación con r = 0.5 (BAJA)...")
    params1 = ModelParameters()
    params_dict1 = params1.get()
    params_dict1['spatial']['L'] = L
    params_dict1['spatial']['nx'] = nx
    params_dict1['spatial']['ny'] = ny
    params_dict1['temporal']['T'] = T
    params_dict1['demographic']['r'] = 0.5  # BAJA
    params_dict1['stochastic']['sigma_R'] = 0.0
    params_dict1['stochastic']['sigma_F'] = 0.0

    solver1 = LotkaVolterraSolver(L=L, nx=nx, ny=ny)
    result1 = solver1.solve(
        R0=R0.copy(),
        F0=F0.copy(),
        params=params_dict1,
        refuge_map=refuge_map,
        T=T,
        dt=params_dict1['temporal']['dt'],
        save_every=30
    )
    print(f"   ✓ Simulación 1 completada ({len(result1['times'])} frames)")

    # SIMULACIÓN 2: Tasa de crecimiento ALTA (r = 1.5)
    print("\n4. Ejecutando simulación con r = 1.5 (ALTA)...")
    params2 = ModelParameters()
    params_dict2 = params2.get()
    params_dict2['spatial']['L'] = L
    params_dict2['spatial']['nx'] = nx
    params_dict2['spatial']['ny'] = ny
    params_dict2['temporal']['T'] = T
    params_dict2['demographic']['r'] = 1.5  # ALTA
    params_dict2['stochastic']['sigma_R'] = 0.0
    params_dict2['stochastic']['sigma_F'] = 0.0

    solver2 = LotkaVolterraSolver(L=L, nx=nx, ny=ny)
    result2 = solver2.solve(
        R0=R0.copy(),
        F0=F0.copy(),
        params=params_dict2,
        refuge_map=refuge_map,
        T=T,
        dt=params_dict2['temporal']['dt'],
        save_every=30
    )
    print(f"   ✓ Simulación 2 completada ({len(result2['times'])} frames)")

    # Crear animación comparativa
    print("\n5. Generando animación comparativa...")
    print("   (Esto puede tardar 2-3 minutos...)")

    times = result1['times']
    n_times = len(times)

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y)

    # Calcular poblaciones totales para phase portraits
    dx = L / (nx - 1)
    dy = L / (ny - 1)
    R_total1 = [np.sum(R) * dx * dy for R in result1['R']]
    F_total1 = [np.sum(F) * dx * dy for F in result1['F']]
    R_total2 = [np.sum(R) * dx * dy for R in result2['R']]
    F_total2 = [np.sum(F) * dx * dy for F in result2['F']]

    # Configurar figura con 2x2 paneles (2 filas, 2 columnas)
    fig = plt.figure(figsize=(18, 14))

    # Fila superior: Vistas espaciales
    ax_spatial1 = plt.subplot(2, 2, 1)
    ax_spatial2 = plt.subplot(2, 2, 2)

    # Fila inferior: Phase portraits
    ax_phase1 = plt.subplot(2, 2, 3)
    ax_phase2 = plt.subplot(2, 2, 4)

    # Panel superior izquierdo: Vista espacial r = 0.5
    ax_spatial1.contourf(X, Y, refuge_map, levels=[1.0, 1.5, 3.0],
                 colors=['white', 'lightgreen'], alpha=0.3)
    scatter_prey1 = ax_spatial1.scatter([], [], c='blue', s=15, alpha=0.6, label='Presas')
    scatter_pred1 = ax_spatial1.scatter([], [], c='red', s=25, alpha=0.8, marker='^', label='Depredadores')
    ax_spatial1.set_xlim(0, L)
    ax_spatial1.set_ylim(0, L)
    ax_spatial1.set_xlabel('x', fontsize=11)
    ax_spatial1.set_ylabel('y', fontsize=11)
    ax_spatial1.set_title('Vista Espacial: r = 0.5 (BAJA)\nt = 0.00 s', fontsize=12, fontweight='bold')
    ax_spatial1.legend(loc='upper right', fontsize=9)
    ax_spatial1.set_aspect('equal')

    # Panel superior derecho: Vista espacial r = 1.5
    ax_spatial2.contourf(X, Y, refuge_map, levels=[1.0, 1.5, 3.0],
                 colors=['white', 'lightgreen'], alpha=0.3)
    scatter_prey2 = ax_spatial2.scatter([], [], c='blue', s=15, alpha=0.6, label='Presas')
    scatter_pred2 = ax_spatial2.scatter([], [], c='red', s=25, alpha=0.8, marker='^', label='Depredadores')
    ax_spatial2.set_xlim(0, L)
    ax_spatial2.set_ylim(0, L)
    ax_spatial2.set_xlabel('x', fontsize=11)
    ax_spatial2.set_ylabel('y', fontsize=11)
    ax_spatial2.set_title('Vista Espacial: r = 1.5 (ALTA)\nt = 0.00 s', fontsize=12, fontweight='bold')
    ax_spatial2.legend(loc='upper right', fontsize=9)
    ax_spatial2.set_aspect('equal')

    # Panel inferior izquierdo: Phase portrait r = 0.5
    ax_phase1.plot(R_total1, F_total1, 'gray', alpha=0.3, linewidth=1, label='Trayectoria completa')
    line_phase1, = ax_phase1.plot([], [], 'b-', linewidth=2, label='Trayectoria actual')
    point_phase1, = ax_phase1.plot([], [], 'ro', markersize=8, label='Posición actual')
    ax_phase1.plot(R_total1[0], F_total1[0], 'go', markersize=10, label='Inicio', zorder=5)
    ax_phase1.set_xlim(0, max(R_total1) * 1.1)
    ax_phase1.set_ylim(0, max(F_total1) * 1.1)
    ax_phase1.set_xlabel('Población de Presas (R)', fontsize=11)
    ax_phase1.set_ylabel('Población de Depredadores (F)', fontsize=11)
    ax_phase1.set_title('Phase Portrait: r = 0.5', fontsize=12, fontweight='bold')
    ax_phase1.legend(loc='best', fontsize=8)
    ax_phase1.grid(True, alpha=0.3)

    # Panel inferior derecho: Phase portrait r = 1.5
    ax_phase2.plot(R_total2, F_total2, 'gray', alpha=0.3, linewidth=1, label='Trayectoria completa')
    line_phase2, = ax_phase2.plot([], [], 'b-', linewidth=2, label='Trayectoria actual')
    point_phase2, = ax_phase2.plot([], [], 'ro', markersize=8, label='Posición actual')
    ax_phase2.plot(R_total2[0], F_total2[0], 'go', markersize=10, label='Inicio', zorder=5)
    ax_phase2.set_xlim(0, max(R_total2) * 1.1)
    ax_phase2.set_ylim(0, max(F_total2) * 1.1)
    ax_phase2.set_xlabel('Población de Presas (R)', fontsize=11)
    ax_phase2.set_ylabel('Población de Depredadores (F)', fontsize=11)
    ax_phase2.set_title('Phase Portrait: r = 1.5', fontsize=12, fontweight='bold')
    ax_phase2.legend(loc='best', fontsize=8)
    ax_phase2.grid(True, alpha=0.3)

    plt.suptitle('Comparación: Efecto de la Tasa de Crecimiento (r)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    def init():
        scatter_prey1.set_offsets(np.empty((0, 2)))
        scatter_pred1.set_offsets(np.empty((0, 2)))
        scatter_prey2.set_offsets(np.empty((0, 2)))
        scatter_pred2.set_offsets(np.empty((0, 2)))
        line_phase1.set_data([], [])
        point_phase1.set_data([], [])
        line_phase2.set_data([], [])
        point_phase2.set_data([], [])
        return scatter_prey1, scatter_pred1, scatter_prey2, scatter_pred2, line_phase1, point_phase1, line_phase2, point_phase2

    def update(frame):
        # Vistas espaciales
        # Panel 1: r = 0.5
        x_prey1, y_prey1 = sample_particles(result1['R'][frame], 250, X, Y, L, nx, ny)
        x_pred1, y_pred1 = sample_particles(result1['F'][frame], 60, X, Y, L, nx, ny)

        if len(x_prey1) > 0:
            scatter_prey1.set_offsets(np.column_stack([x_prey1, y_prey1]))
        else:
            scatter_prey1.set_offsets(np.empty((0, 2)))

        if len(x_pred1) > 0:
            scatter_pred1.set_offsets(np.column_stack([x_pred1, y_pred1]))
        else:
            scatter_pred1.set_offsets(np.empty((0, 2)))

        # Panel 2: r = 1.5
        x_prey2, y_prey2 = sample_particles(result2['R'][frame], 250, X, Y, L, nx, ny)
        x_pred2, y_pred2 = sample_particles(result2['F'][frame], 60, X, Y, L, nx, ny)

        if len(x_prey2) > 0:
            scatter_prey2.set_offsets(np.column_stack([x_prey2, y_prey2]))
        else:
            scatter_prey2.set_offsets(np.empty((0, 2)))

        if len(x_pred2) > 0:
            scatter_pred2.set_offsets(np.column_stack([x_pred2, y_pred2]))
        else:
            scatter_pred2.set_offsets(np.empty((0, 2)))

        # Actualizar phase portraits
        line_phase1.set_data(R_total1[:frame+1], F_total1[:frame+1])
        point_phase1.set_data([R_total1[frame]], [F_total1[frame]])
        line_phase2.set_data(R_total2[:frame+1], F_total2[:frame+1])
        point_phase2.set_data([R_total2[frame]], [F_total2[frame]])

        # Actualizar títulos
        ax_spatial1.set_title(f'Vista Espacial: r = 0.5 (BAJA)\nt = {times[frame]:.2f} s',
                     fontsize=12, fontweight='bold')
        ax_spatial2.set_title(f'Vista Espacial: r = 1.5 (ALTA)\nt = {times[frame]:.2f} s',
                     fontsize=12, fontweight='bold')

        if frame % 10 == 0:
            print(f"   Frame {frame}/{n_times}")

        return scatter_prey1, scatter_pred1, scatter_prey2, scatter_pred2, line_phase1, point_phase1, line_phase2, point_phase2

    # Crear animación
    anim = FuncAnimation(fig, update, frames=n_times, init_func=init,
                        interval=1000, blit=True, repeat=True)

    # Guardar
    output_dir = Path('animaciones')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / '3_comparacion_parametros.gif'

    print(f"\n6. Guardando animación en: {output_path}")
    writer = PillowWriter(fps=0.90)
    anim.save(output_path, writer=writer)

    plt.close()

    print(f"\n✓ Animación guardada exitosamente!")
    print(f"  Archivo: {output_path}")
    print("\nINTERPRETACIÓN:")
    print("  - r = 0.5 (izquierda): Crecimiento lento de presas")
    print("  - r = 1.5 (derecha): Crecimiento rápido de presas")
    print("  → Se observará mayor abundancia de presas con r alto")
    print("="*70)


if __name__ == "__main__":
    crear_comparacion_parametros()
