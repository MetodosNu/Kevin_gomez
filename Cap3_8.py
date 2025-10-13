
"""
Ejercicio 3.8 - Derivada segunda y cancelación numérica
Descripción:
Se estudia la precisión de la derivada segunda numérica de f(x) = (1 - cos x) / x²
usando el método de diferencias centrales y comparando con una versión
algebraicamente reescrita que evita cancelaciones.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# (a) Definición de la función y su derivada analítica
# ============================================================

def f_original(x):
    """Función original f(x) = (1 - cos x) / x²."""
    return (1 - np.cos(x)) / x**2

def f_rewritten(x):
    """
    Versión reescrita usando identidad trigonométrica:
        1 - cos(x) = 2 sin²(x/2)
    => f(x) = 2 * (sin(x/2)/x)²
    """
    return 2 * (np.sin(x/2) / x)**2

def f_second_derivative_exact(x):
    """
    Derivada segunda analítica de f(x).
    Resultado obtenido simbólicamente o manualmente:
        f''(x) = [ (x² * cos(x) - 2x * sin(x) - 2(1 - cos(x))) * 2 ] / x⁴
    """
    num = (x**2 * np.cos(x) - 2*x*np.sin(x) - 2*(1 - np.cos(x))) * 2
    den = x**4
    return num / den


# ============================================================
# (b) Derivada numérica por diferencias centrales
# ============================================================

def second_derivative_central(f, x, h):
    """
    Aproximación por diferencias centrales:
        f''(x) ≈ [f(x + h) - 2f(x) + f(x - h)] / h²
    """
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2


# ============================================================
# (c) Cálculo de errores y comparación
# ============================================================

def compute_errors(x0):
    """
    Calcula el error absoluto de la derivada segunda numérica
    para distintos valores de h, usando ambas versiones de f(x).
    """
    h_values = np.logspace(-1, -6, 50)
    f2_exact = f_second_derivative_exact(x0)

    errors_original = []
    errors_rewritten = []

    for h in h_values:
        approx_original = second_derivative_central(f_original, x0, h)
        approx_rewritten = second_derivative_central(f_rewritten, x0, h)

        errors_original.append(abs(approx_original - f2_exact))
        errors_rewritten.append(abs(approx_rewritten - f2_exact))

    return h_values, np.array(errors_original), np.array(errors_rewritten), f2_exact


# ============================================================
# (d) Gráfica de error log-log
# ============================================================

def plot_errors(h_values, err1, err2):
    plt.figure(figsize=(8, 6))
    plt.loglog(h_values, err1, 'o-', label='f(x) original', linewidth=2)
    plt.loglog(h_values, err2, 's-', label='f(x) reescrita', linewidth=2)
    plt.xlabel('Paso h')
    plt.ylabel('Error absoluto')
    plt.title('Error en la segunda derivada numérica (x = 0.004)')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.show()


# ============================================================
# (e) Ejecución principal
# ============================================================

def main():
    x0 = 0.004

    # Cálculo de errores
    h_values, err_orig, err_rew, f2_exact = compute_errors(x0)

    print(f"f''(x) analítica en x = {x0} → {f2_exact:.10e}")

    # Graficar resultados
    plot_errors(h_values, err_orig, err_rew)


if __name__ == "__main__":
    main()
