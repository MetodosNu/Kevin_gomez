# -*- coding: utf-8 -*-
"""
Ejercicio 3.13 - Derivadas numéricas y extrapolación de Richardson
Autor: Emanuel Osorio (bro)
Descripción:
Se calcula la primera derivada de f(x) = e^{sin(2x)} mediante:
- Diferencia hacia adelante (Forward Difference)
- Diferencia central (Central Difference)
- Extrapolación de Richardson
y se comparan con la derivada analítica.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# (a) Definición de la función y su derivada analítica
# ============================================================

def f(x):
    """Función original f(x) = e^{sin(2x)}."""
    return np.exp(np.sin(2 * x))

def f_prime_analytical(x):
    """Derivada analítica f'(x) = 2cos(2x)e^{sin(2x)}."""
    return 2 * np.cos(2 * x) * np.exp(np.sin(2 * x))


# ============================================================
# (b) Diferencias finitas
# ============================================================

def forward_difference(x, y, h):
    """
    Diferencia hacia adelante:
        f'(x_i) ≈ [f(x_{i+1}) - f(x_i)] / h
    (No se calcula para el último punto)
    """
    dydx = np.zeros_like(y)
    dydx[:-1] = (y[1:] - y[:-1]) / h
    dydx[-1] = np.nan  # no se puede calcular en el extremo derecho
    return dydx

def central_difference(x, y, h):
    """
    Diferencia central:
        f'(x_i) ≈ [f(x_{i+1}) - f(x_{i-1})] / (2h)
    (No se calcula para los extremos)
    """
    dydx = np.zeros_like(y)
    dydx[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    dydx[0] = np.nan
    dydx[-1] = np.nan
    return dydx


# ============================================================
# (c) Extrapolación de Richardson
# ============================================================

def richardson_forward(x, f, h):
    """
    Aplica la extrapolación de Richardson a la derivada forward:
        R_fd = 2D_fd(h) - D_fd(2h)
    """
    y = f(x)
    D1 = forward_difference(x, y, h)
    D2 = forward_difference(x[::2], y[::2], 2 * h)  # usa pasos de 2h
    R = np.zeros_like(y)
    R[:-2:2] = 2 * D1[:-2:2] - D2[:-1]  # ajustar dimensiones
    R[np.isnan(D1)] = np.nan
    return R


# ============================================================
# (d) Ejecución y visualización
# ============================================================

def main():
    # Definición del dominio
    x = np.arange(0, 1.6 + 0.08, 0.08)
    y = f(x)
    h = x[1] - x[0]

    # Derivadas numéricas
    dydx_fd = forward_difference(x, y, h)
    dydx_cd = central_difference(x, y, h)
    dydx_richardson = richardson_forward(x, f, h)

    # Derivada analítica
    dydx_exact = f_prime_analytical(x)

    # --- Tabla de resultados ---
    print(" x\tf(x)\t\tForward\t\tCentral\t\tAnalytical")
    for i in range(len(x)):
        print(f"{x[i]:.2f}\t{y[i]:.6f}\t{dydx_fd[i]:.6f}\t{dydx_cd[i]:.6f}\t{dydx_exact[i]:.6f}")

    # --- Gráficas ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, dydx_exact, 'k-', lw=2, label="Analítica")
    plt.plot(x, dydx_fd, 'o-', label="Forward Difference")
    plt.plot(x, dydx_cd, 's-', label="Central Difference")
    plt.plot(x, dydx_richardson, 'd-', label="Richardson (Forward)")
    plt.xlabel("x")
    plt.ylabel("f'(x)")
    plt.title("Comparación de derivadas numéricas de f(x) = e^{sin(2x)}")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
