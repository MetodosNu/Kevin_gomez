"""
Ejercicio 1.7 - Comparación de precisión simple y doble
Descripción:
Se estudia la cantidad (1 + 1/n)^n para n = 10^1, 10^2, ..., 10^7
comparando resultados en precisión simple (float32) y doble (float64).
"""

import numpy as np

def compute_quantity(n_values):
    """
    Calcula (1 + 1/n)^n tanto en precisión simple como doble.

    Parámetros:
        n_values (list): lista de valores de n (enteros)
    Retorna:
        results (list): lista de tuplas (n, float32_value, float64_value)
    """
    results = []
    for n in n_values:
        single = np.float32(1 + 1/np.float32(n)) ** np.float32(n)
        double = np.float64(1 + 1/np.float64(n)) ** np.float64(n)
        results.append((n, single, double))
    return results


def print_table(results):
    """
    Imprime los resultados en una tabla formateada.
    """
    print(f"{'n':>10} | {'(1+1/n)^n (float32)':>25} | {'(1+1/n)^n (float64)':>25}")
    print("-" * 65)
    for n, single, double in results:
        print(f"{n:10d} | {single:25.10f} | {double:25.10f}")


def main():
    # Generar valores de n: 10¹, 10², ..., 10⁷
    n_values = [10**i for i in range(1, 8)]

    # Calcular resultados
    results = compute_quantity(n_values)

    # Mostrar tabla
    print_table(results)


if __name__ == "__main__":
    main()
