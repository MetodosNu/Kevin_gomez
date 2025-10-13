# -*- coding: utf-8 -*-
"""
Descomposición en Valores Singulares (SVD)
Autor: Emanuel Osorio (bro)
Descripción:
Este programa implementa, explica y demuestra el uso de la descomposición
en valores singulares (SVD) paso a paso, mostrando:
  1. La descomposición A = U Σ V^T
  2. La reconstrucción de la matriz original
  3. La aproximación reducida de rango k
  4. Interpretación geométrica y aplicaciones a ML
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# (1) Definición de una matriz de ejemplo
# ============================================================

# Matriz de entrada (puede representar datos, relaciones, etc.)
A = np.array([
    [3, 1, 1],
    [-1, 3, 1]
], dtype=float)

print("Matriz original A:")
print(A)
print(f"Dimensiones: {A.shape}\n")

# ============================================================
# (2) Cálculo de la SVD usando NumPy
# ============================================================

# np.linalg.svd devuelve:
# U: matriz ortogonal (m x m)
# S: vector con valores singulares (r)
# Vt: transpuesta de la matriz ortogonal V (n x n)
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Convertimos S en una matriz diagonal rectangular (Σ)
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, S)

# ============================================================
# (3) Mostrar resultados intermedios
# ============================================================

print("Matriz U (vectores ortogonales de salida):")
print(U)
print(f"Verificación ortogonalidad: UᵀU = \n{U.T @ U}\n")

print("Valores singulares Σ (diagonal con σ₁, σ₂,...):")
print(Sigma)
print(f"Valores singulares individuales: {S}\n")

print("Matriz Vᵀ (vectores ortogonales de entrada):")
print(Vt)
print(f"Verificación ortogonalidad: VᵀV = \n{Vt.T @ Vt}\n")

# ============================================================
# (4) Reconstrucción de la matriz original
# ============================================================

A_reconstructed = U @ Sigma @ Vt

print("Reconstrucción de la matriz original A = UΣVᵀ:")
print(A_reconstructed)
print(f"Error de reconstrucción (norma Frobenius): {np.linalg.norm(A - A_reconstructed):.6e}\n")

# ============================================================
# (5) Reducción de dimensionalidad (SVD truncada)
# ============================================================

# Tomamos solo los k valores singulares más grandes
k = 1  # podemos cambiarlo por 1, 2, ...
U_k = U[:, :k]
Sigma_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

A_k = U_k @ Sigma_k @ Vt_k  # aproximación de rango k

print(f"Aproximación de rango k={k}:")
print(A_k)
print(f"Error de aproximación: {np.linalg.norm(A - A_k):.6e}\n")

# ============================================================
# (6) Interpretación geométrica
# ============================================================

# Los valores singulares representan el "estiramiento" en cada dirección
# U: rotación en el espacio de salida
# Σ: escalado
# V: rotación en el espacio de entrada

# Visualicemos cómo los valores singulares escalan los vectores base
x = np.linspace(0, 2*np.pi, 200)
circle = np.array([np.cos(x), np.sin(x), np.zeros_like(x)])  # círculo unitario

# Transformación A: el círculo se convierte en una elipse
ellipse = A @ circle

plt.figure(figsize=(7, 5))
plt.plot(circle[0], circle[1], 'b--', label="Círculo unitario (entrada)" )
plt.plot(ellipse[0], ellipse[1], 'r-', label="Transformación A (elipse)")
plt.axis('equal')
plt.title("Interpretación geométrica de A = UΣVᵀ")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


