"""
Ejercicio 1.9 - Multiplicación de matrices sin usar numpy.dot() ni @.
Descripción: Tres métodos distintos para multiplicar matrices, con pruebas incluidas.
"""

from typing import List
import numpy as np

# =====================================================
# (a) Implementación con tres bucles
# =====================================================

def matrix_multiply_three_loops(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiplica dos matrices A y B usando tres bucles anidados.
    Parámetros:
        A (np.ndarray): Matriz de tamaño (n x m)
        B (np.ndarray): Matriz de tamaño (m x p)
    Retorna:
        C (np.ndarray): Resultado de la multiplicación (n x p)
    """
    n, m = A.shape
    m2, p = B.shape

    if m != m2:
        raise ValueError("Las dimensiones internas de las matrices no coinciden.")

    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C


# =====================================================
# (b) Implementación con dos bucles (usando producto punto)
# =====================================================

def matrix_multiply_two_loops(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiplica matrices A y B usando solo dos bucles,
    aplicando producto punto entre filas y columnas.
    """
    n, m = A.shape
    m2, p = B.shape

    if m != m2:
        raise ValueError("Las dimensiones internas de las matrices no coinciden.")

    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            C[i, j] = sum(A[i, k] * B[k, j] for k in range(m))
    return C


# =====================================================
# (c) Implementación con listas de listas (sin NumPy)
# =====================================================

def matrix_multiply_lists(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Multiplica matrices representadas como listas de listas.
    Parámetros:
        A (list): matriz n x m
        B (list): matriz m x p
    Retorna:
        C (list): matriz n x p
    """
    n = len(A)
    m = len(A[0])
    m2 = len(B)
    p = len(B[0])

    if m != m2:
        raise ValueError("Las dimensiones internas de las matrices no coinciden.")

    C = [[0 for _ in range(p)] for _ in range(n)]

    for i in range(n):
        for j in range(p):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(m))
    return C


# =====================================================
# (d) Pruebas con matrices 3x4 y 4x2
# =====================================================

def main():
    # Definición de matrices
    A = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    B = np.array([
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]
    ])

    print("Matriz A (3x4):\n", A)
    print("Matriz B (4x2):\n", B)

    # a) Tres bucles
    C1 = matrix_multiply_three_loops(A, B)
    print("\nResultado (3 bucles):\n", C1)

    # b) Dos bucles
    C2 = matrix_multiply_two_loops(A, B)
    print("\nResultado (2 bucles):\n", C2)

    # c) Listas de listas
    A_list = A.tolist()
    B_list = B.tolist()
    C3 = matrix_multiply_lists(A_list, B_list)
    print("\nResultado (listas de listas):\n", C3)


if __name__ == "__main__":
    main()
