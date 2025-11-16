"""
Oscilador Armónico Cuántico en 3D
---------------------------------
Este programa implementa y visualiza el oscilador armónico cuántico tridimensional,
utilizando funciones de onda basadas en polinomios de Hermite.

Características:
- Calcula funciones de onda 1D y 3D normalizadas.
- Calcula las energías permitidas en 1D y 3D.
- Evalúa la función de onda en puntos específicos.
- Genera gráficas de la densidad de probabilidad |ψ(x,y,0)|² en el plano XY.
"""
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, sqrt, pi
from scipy.special import eval_hermite

# ----------------------------
# Funciones auxiliares
# ----------------------------

def psi_1d(n: int, x: np.ndarray, m: float = 1.0, w: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """Función de onda 1D del oscilador armónico cuántico."""
    xi = sqrt(m * w / hbar) * x
    norm = 1.0 / sqrt((2**n) * factorial(n)) * (m*w/(pi*hbar))**0.25
    return norm * np.exp(-xi**2 / 2) * eval_hermite(n, xi)


def energy_1d(n: int, w: float = 1.0, hbar: float = 1.0) -> float:
    """Energía 1D del oscilador armónico cuántico."""
    return hbar * w * (n + 0.5)


# ----------------------------
# Funciones principales
# ----------------------------

def psi_3d(nx: int, ny: int, nz: int, x: np.ndarray, y: np.ndarray, z: np.ndarray,
           m: float = 1.0, w: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """Función de onda 3D como producto de tres funciones 1D."""
    return psi_1d(nx, x, m, w, hbar) * psi_1d(ny, y, m, w, hbar) * psi_1d(nz, z, m, w, hbar)


def energy_3d(nx: int, ny: int, nz: int, w: float = 1.0, hbar: float = 1.0) -> float:
    """Energía total 3D como suma de tres energías 1D."""
    return energy_1d(nx, w, hbar) + energy_1d(ny, w, hbar) + energy_1d(nz, w, hbar)


# ----------------------------
# Prueba + Visualización
# ----------------------------
if __name__ == "__main__":
    # Números cuánticos
    nx, ny, nz = 1 , 0, 2
    
    # Grilla para graficar en el plano XY (z=0)
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # plano z=0

    # Calcular energía y función de onda en 2D
    E_total = energy_3d(nx, ny, nz)
    psi_xy = psi_3d(nx, ny, nz, X, Y, Z)
    prob_density = np.abs(psi_xy)**2

    # Imprimir valores
    psi_at_origin = psi_3d(nx, ny, nz, np.array([1.0]), np.array([0.0]), np.array([2.0]))[0]
    print(f"Energía total 3D (nx={nx}, ny={ny}, nz={nz}): {E_total:.3f}")
    print(f"ψ_3D evaluada en (1,0,2): {psi_at_origin:.3f}")

    # Graficar
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, psi_xy, levels=50, cmap="RdBu")
    plt.colorbar(label=r"$\psi_{3D}(x,y,0)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Función de onda 3D (nx={nx}, ny={ny}, nz={nz}) en z=0")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, prob_density, levels=50, cmap="viridis")
    plt.colorbar(label=r"$|\psi_{3D}(x,y,0)|^2$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Densidad de probabilidad (nx={nx}, ny={ny}, nz={nz}) en z=0")
    plt.show()
