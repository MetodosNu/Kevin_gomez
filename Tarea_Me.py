import numpy as np
import matplotlib.pyplot as plt

# Parámetros del péndulo
m = 1.0  
l = 1.0  
g = 9.8  

# Crear una malla de puntos para theta y theta_dot
# theta varía de -2π a +2π (ángulo desde la vertical)
theta = np.linspace(-2*np.pi, 2*np.pi, 400)
# theta_dot varía en un rango apropiado para velocidad angular
theta_dot = np.linspace(-8, 8, 400)

# Crear la malla bidimensional
THETA, THETA_DOT = np.meshgrid(theta, theta_dot)

# Calcular la energía total 
E = 0.5 * m * l**2 * THETA_DOT**2 + m * g * l * (1 - np.cos(THETA))

plt.figure(figsize=(12, 8))

# Generar el gráfico de contornos
# levels determina cuántas curvas de nivel mostrar
contour_levels = np.linspace(0, 50, 20)  # Niveles de energía desde 0 hasta 50 J
contours = plt.contour(THETA, THETA_DOT, E, levels=contour_levels, colors='blue', alpha=0.7)

# Agregar etiquetas a algunas curvas de nivel para mostrar los valores de energía
plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f J')

# Configurar los ejes
plt.xlabel(r'$\theta$ (radianes)', fontsize=12)
plt.ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
plt.title('Diagrama de Fases del Péndulo Simple\nContornos de Energía Constante', fontsize=14)

# Configurar las marcas del eje x para mostrar múltiplos de π
x_ticks = np.array([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
x_labels = [r'$-2\pi$', r'$-\pi$', '0', r'$\pi$', r'$2\pi$']
plt.xticks(x_ticks, x_labels)

# Agregar líneas de referencia
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Línea horizontal en θ̇ = 0
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)   # Línea vertical en θ = 0

# Agregar una cuadrícula para mejor visualización
plt.grid(True, alpha=0.3)

# Ajustar el diseño para que no se corten las etiquetas
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Interpretación física de las regiones:
print("INTERPRETACIÓN FÍSICA DEL DIAGRAMA:")
print("="*50)
print("1. CURVAS CERRADAS (cerca del centro, θ ≈ 0):")
print("   - Representan oscilaciones pequeñas del péndulo")
print("   - El péndulo oscila de un lado a otro sin dar vueltas completas")
print("   - Energía baja: movimiento periódico limitado")
print()
print("2. CURVAS ABIERTAS (arriba y abajo del diagrama):")
print("   - Representan rotaciones continuas del péndulo")
print("   - El péndulo tiene suficiente energía para dar vueltas completas")
print("   - Energía alta: movimiento rotacional continuo")
print()
print("3. SEPARATRIZ (curva en forma de 8):")
print("   - Separa las regiones de oscilación y rotación")
print("   - Corresponde a la energía crítica E = 2mgl")
print("   - Representa el movimiento límite donde el péndulo")
print("     llega exactamente a la posición invertida (θ = ±π)")
print()
print(f"Energía de la separatriz: E_sep = 2mgl = {2*m*g*l:.1f} J")

# Crear un segundo gráfico mostrando específicamente la separatriz
plt.figure(figsize=(12, 8))

# Calcular la energía de la separatriz
E_sep = 2 * m * g * l

# Generar contornos con más detalle cerca de la separatriz
levels_detailed = np.concatenate([
    np.linspace(0, E_sep-5, 10),      # Curvas cerradas
    [E_sep],                          # La separatriz
    np.linspace(E_sep+2, E_sep+20, 8) # Curvas abiertas
])

contours = plt.contour(THETA, THETA_DOT, E, levels=levels_detailed, colors='blue', alpha=0.7)

# Destacar la separatriz en rojo
separatrix = plt.contour(THETA, THETA_DOT, E, levels=[E_sep], colors='red', linewidths=2)
plt.clabel(separatrix, inline=True, fontsize=10, fmt='Separatriz\n(E=%.1f J)')

plt.xlabel(r'$\theta$ (radianes)', fontsize=12)
plt.ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
plt.title('Diagrama de Fases del Péndulo - Separatriz Destacada', fontsize=14)

plt.xticks(x_ticks, x_labels)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)

#anotaciones para las diferentes regiones
plt.annotate('Oscilaciones\n(curvas cerradas)', xy=(0, 2), xytext=(np.pi/2, 4),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, ha='center', color='green')

plt.annotate('Rotaciones\n(curvas abiertas)', xy=(0, 6), xytext=(np.pi, 7),
            arrowprops=dict(arrowstyle='->', color='orange'),
            fontsize=10, ha='center', color='orange')

plt.tight_layout()
plt.show()