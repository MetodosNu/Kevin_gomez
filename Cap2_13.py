"""
Ejercicio 2.13 - Estudio de una función racional
Descripción:
Se compara la evaluación de una función racional f(x) mediante:
(a) Regla de Horner
(b) Fracción continua equivalente
(c) Evaluación numérica en x=10^7 y gráfico comparativo para x ∈ [0,4] y z = 2.4 + 2⁻⁵ⁱ (i=0..800)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# (a) Definición de funciones
# ============================================================

def f_rational_horner(x):
    """
    Evalúa la función racional:
        f(x) = (7x^4 - 101x^3 + 540x^2 - 1204x + 958) / (x^4 - 14x^3 + 72x^2 - 151x + 112)
    usando la regla de Horner para numerador y denominador.
    """
    # Numerador y denominador como listas de coeficientes descendentes
    num = [7, -101, 540, -1204, 958]
    den = [1, -14, 72, -151, 112]

    # Evaluación con Horner
    n_val = num[0]
    for c in num[1:]:
        n_val = n_val * x + c

    d_val = den[0]
    for c in den[1:]:
        d_val = d_val * x + c

    return n_val / d_val


def u_continued_fraction(x):
    """
    Evalúa la función equivalente:
        u(x) = 7 - 3 / (x - 2 - 1 / (x - 3 - 1 / (x - 4)))
    """
    return 7 - 3 / (x - 2 - 1 / (x - 3 - 1 / (x - 4)))


# ============================================================
# (b) Evaluación numérica
# ============================================================

def compare_large_x():
    """
    Compara f(x) y u(x) para x = 10^7, mostrando efectos de redondeo flotante.
    """
    x = 1e7
    f_val = f_rational_horner(x)
    u_val = u_continued_fraction(x)

    print("Evaluación en x = 10^7")
    print(f"f(x) via Horner = {f_val:.16f}")
    print(f"u(x) fracción continua = {u_val:.16f}")
    print("\nObserva si hay pérdida de precisión o overflow numérico.")


# ============================================================
# (c) Gráficas comparativas
# ============================================================

def plot_functions():
    """
    Grafica f(x) y u(x) en el rango [0, 4] y compara comportamiento.
    """
    x = np.linspace(0, 4, 400)
    fx = [f_rational_horner(xi) for xi in x]
    ux = [u_continued_fraction(xi) for xi in x]

    plt.figure(figsize=(8, 5))
    plt.plot(x, fx, label=r"$f(x)$ (Horner)", lw=2)
    plt.plot(x, ux, "--", label=r"$u(x)$ (Fracción continua)", lw=2)
    plt.xlabel("x")
    plt.ylabel("Valor de la función")
    plt.title("Comparación entre formulaciones de f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_complex_behavior():
    """
    Evalúa ambas formulaciones para z = 2.4 + 2^-5i (i = 0,...,800)
    y grafica su comportamiento en el plano complejo.
    """
    i_vals = np.arange(801)
    z_vals = 2.4 + np.power(2.0, -5 * i_vals)
    fz = [f_rational_horner(z) for z in z_vals]
    uz = [u_continued_fraction(z) for z in z_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(i_vals, fz, label="f(z) (Horner)")
    plt.plot(i_vals, uz, "--", label="u(z) (Fracción continua)")
    plt.xlabel("i (iteración)")
    plt.ylabel("Valor (magnitud aproximada)")
    plt.title("Comparación en valores complejos")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# (d) Ejecución principal
# ============================================================

def main():
    # Parte (a): gráfico inicial
    plot_functions()

    # Parte (b): comparación numérica
    compare_large_x()

    # Parte (c): gráfico para z = 2.4 + 2^-5i
    plot_complex_behavior()


if __name__ == "__main__":
    main()
