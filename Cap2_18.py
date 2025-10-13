
"""
Ejercicio 2.18 - Serie de Fourier de una onda cuadrada y fenómeno de Gibbs
Descripción:
Implementación de la función cuadrada y su serie de Fourier.
Se analiza el fenómeno de Gibbs y su comportamiento al aumentar n_max.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# (a) Definición de funciones
# ============================================================

def square_wave(x):
    """
    Define una onda cuadrada periódica de periodo 2π:
        f(x) =  1/2 si 0 < x < π
              -1/2 si -π < x < 0
    """
    x_mod = np.mod(x + np.pi, 2 * np.pi) - np.pi  # ajusta al rango [-π, π)
    return np.where(x_mod >= 0, 0.5, -0.5)


def fourier_square(x, n_max):
    """
    Calcula la expansión de Fourier truncada de la onda cuadrada:
        f(x) ≈ (2/π) * Σ_{n=1,3,5,...}^{n_max} sin(nx)/n
    Solo se suman los términos impares.
    """
    n_values = np.arange(1, n_max + 1, 2)  # solo n impares
    series_sum = np.zeros_like(x, dtype=float)
    for n in n_values:
        series_sum += np.sin(n * x) / n
    return (2 / np.pi) * series_sum


# ============================================================
# (b) Gráfica comparativa
# ============================================================

def plot_fourier_series():
    """
    Grafica la onda cuadrada y su serie de Fourier truncada
    para n_max = 1, 3, 5, 7, 9.
    """
    x = np.linspace(-np.pi, np.pi, 1000)
    f_true = square_wave(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, f_true, 'k', lw=2, label="Onda cuadrada original")

    for n_max in [1, 3, 5, 7, 9]:
        f_approx = fourier_square(x, n_max)
        plt.plot(x, f_approx, lw=1.8, label=f"n_max = {n_max}")

    plt.title("Serie de Fourier de la onda cuadrada y fenómeno de Gibbs")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# (c) Gibbs con valores grandes de n_max
# ============================================================

def plot_gibbs_effect():
    """
    Muestra la persistencia del fenómeno de Gibbs para n_max grandes.
    """
    x = np.linspace(-np.pi, np.pi, 1000)
    plt.figure(figsize=(10, 6))

    for n_max in [21, 51, 101]:
        f_approx = fourier_square(x, n_max)
        plt.plot(x, f_approx, lw=1.5, label=f"n_max = {n_max}")

    plt.plot(x, square_wave(x), 'k', lw=2, label="Onda cuadrada ideal")
    plt.title("Fenómeno de Gibbs con n grandes")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# (d) Ejecución principal
# ============================================================

def main():
    plot_fourier_series()   # Parte (a) y (b)
    plot_gibbs_effect()     # Parte (c)


if __name__ == "__main__":
    main()
