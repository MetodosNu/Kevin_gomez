import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Definición integral ---
def f_integral(n, x):
    integrand = lambda t: np.exp(-(x*t)**2) * t**(2*n)
    return quad(integrand, 0, 1)[0]

# --- Recurrencia hacia atrás ---
def f_backward(n_target, n_start, x):
    f = np.ones_like(x)  # f_{n_start}(x) = 1
    for n in range(n_start, n_target, -1):
        f = ((2*n - 1) * f - np.exp(-x)) / (2*x)
    return f

# --- Parámetros ---
x_vals = np.linspace(0.1, 0.5, 500)
f10_rec = f_backward(10, 30, x_vals)

# --- Integral para comparación ---
f10_int = np.array([f_integral(10, x) for x in x_vals])

# --- Graficar ---
plt.plot(x_vals, f10_rec, label="Recurrencia hacia atrás", lw=2)
plt.plot(x_vals, f10_int, "--", label="Integral numérica", lw=2)
plt.xlabel("x")
plt.ylabel(r"$f_{10}(x)$")
plt.title(r"Comparación $f_{10}(x)$: Recurrencia vs Integral")
plt.legend()
plt.grid(True)
plt.show()
