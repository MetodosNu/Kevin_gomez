
"""Minimiza la energía de Peierls E_L(Δ) y grafica E(Δ) con su mínimo."""  # Docstring descriptivo del script.

from dataclasses import dataclass  # Para definir una clase liviana de parámetros.
from math import asinh, pi, sqrt     # Funciones matemáticas usadas en la fórmula.
from typing import Callable, Tuple    # Tipos para anotar funciones.

import numpy as np                   
import matplotlib.pyplot as plt      

# ----------------------- Parámetros del modelo ------------------------------

@dataclass(frozen=True)              # Dataclass inmutable para evitar cambios accidentales.
class PeierlsParams:                
    EF: float = 0.5                  # Energía de Fermi (E_F).
    a: float = 1.0                   # Parámetro elástico (prefactor del término Δ^2).
    kF: float = 1.0                  # Número de onda de Fermi (k_F).
    b: float = 2.0                   # Parámetro geométrico (aparece como 1/b^2).

# ---------------------- Energía total por unidad de longitud ----------------

def peierls_energy(delta: float, p: PeierlsParams) -> float:  # Evalúa E_L(Δ) para un Δ dado.
    if delta <= 0.0:                 # Protección para evitar Δ≤0 (no físico y singularidades).
        return np.inf                # Retorna infinito para excluir esos valores del mínimo.
    elastic = p.a / (p.b**2) * delta**2  # Término elástico: (a/b^2) Δ^2.
    ratio = delta / (2.0 * p.EF)     # Cociente Δ/(2E_F) usado repetidamente.
    bracket = 1.0 - sqrt(1.0 + ratio**2) - (ratio**2) * asinh(1.0 / ratio)  # Parte entre corchetes del término electrónico.
    electronic = (2.0 * p.EF * p.kF / pi) * bracket  # Prefactor (2E_F k_F/π) multiplicando el corchete.
    return elastic + electronic      # Energía total: suma elástico + electrónico.

# ---------------------- Minimización por razón áurea ------------------------

def golden_section_minimize(         # Implementa búsqueda del mínimo sin derivadas.
    f: Callable[[float], float],     # Función escalar a minimizar.
    a: float,                        # Extremo izquierdo del intervalo.
    b: float,                        # Extremo derecho del intervalo.
    tol: float = 1e-8,               # Tolerancia de longitud final del intervalo.
    maxiter: int = 200,              # Tope de iteraciones.
) -> Tuple[float, float]:            # Devuelve (x_min, f(x_min)).
    gr = (sqrt(5.0) + 1.0) / 2.0     # Número áureo φ=(1+√5)/2.
    c = b - (b - a) / gr             # Punto interno c.
    d = a + (b - a) / gr             # Punto interno d (simétrico).
    fc = f(c)                        # Evalúa f en c.
    fd = f(d)                        # Evalúa f en d.
    for _ in range(maxiter):         # Itera encogiendo el intervalo.
        if abs(b - a) < tol:         # Criterio de parada por tamaño del intervalo.
            break                    # Sale cuando se alcanza la tolerancia.
        if fc < fd:                  # Si f(c) < f(d), el mínimo está en [a,d].
            b, d, fd = d, c, fc      # Mueve el extremo derecho y recicla valores.
            c = b - (b - a) / gr     # Recalcula c.
            fc = f(c)                # Reevalúa f(c).
        else:                        # Si f(d) ≤ f(c), el mínimo está en [c,b].
            a, c, fc = c, d, fd      # Mueve el extremo izquierdo y recicla valores.
            d = a + (b - a) / gr     # Recalcula d.
            fd = f(d)                # Reevalúa f(d).
    x = (a + b) / 2.0                # Toma el centro del intervalo final como estimador.
    return x, f(x)                   # Devuelve el punto y su valor de f.

# ----------------------- Ejecución y visualización --------------------------

if __name__ == "__main__":           # Solo se ejecuta si corres este archivo directamente.
    p = PeierlsParams()              # Instancia con los valores pedidos en el enunciado.
    grid = np.linspace(1e-4, 4.0, 600)  # Malla de Δ para ubicar el valle de energía.
    vals = np.array([peierls_energy(x, p) for x in grid])  # Evalúa E(Δ) en la malla.
    imin = int(np.nanargmin(vals))   # Índice del valor mínimo en la malla (aproximado).
    L = max(grid[max(imin - 5, 0)], 1e-6)  # Extremo izquierdo del intervalo de refinamiento.
    R = grid[min(imin + 5, len(grid) - 1)]  # Extremo derecho del intervalo de refinamiento.
    delta_star, Emin = golden_section_minimize(lambda d: peierls_energy(d, p), L, R)  # Refinamiento del mínimo con GSS.
    print(f"Δ* ≈ {delta_star:.10f}  |  E_min ≈ {Emin:.10f}")  # Reporte numérico del mínimo.
    plt.figure()                       # Crea figura para la curva E(Δ).
    plt.plot(grid, vals, label="E_L(Δ)")  # Traza la energía en la malla.
    plt.axvline(delta_star, ls="--", label=f"Δ* ≈ {delta_star:.6f}")  # Marca vertical en el Δ óptimo.
    plt.xlabel("Δ")                    # Etiqueta del eje x.
    plt.ylabel("E_L")                  # Etiqueta del eje y.
    plt.title("Inestabilidad de Peierls: energía vs Δ")  # Título de la figura.
    plt.legend()                       # Muestra leyenda.
    plt.tight_layout()                 # Ajusta márgenes.
    plt.show()                         # Renderiza la figura en pantalla.
