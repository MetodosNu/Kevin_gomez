
"""Integra el problema restringido de los tres cuerpos (órbita de Arenstorf) con RK4 y grafica la trayectoria."""  # Descripción del script.

from dataclasses import dataclass    # Para empaquetar parámetros como dataclass.
import numpy as np                   # Librería numérica para vectores/arreglos.
import matplotlib.pyplot as plt      # Librería de graficación.

# ----------------------- Parámetros adimensionales --------------------------

@dataclass(frozen=True)              # Dataclass inmutable para seguridad.
class R3BPParams:                    # Contenedor de parámetros del sistema Tierra-Luna.
    M_over_m: float = 80.45          # Relación de masas M/m (Tierra/Luna).
    @property
    def u(self) -> float:            # u = m/(m+M), posición del primario pequeño a la izquierda.
        return 1.0 / (1.0 + self.M_over_m)  # Cálculo directo desde la razón de masas.
    @property
    def w(self) -> float:            # w = 1 - u, posición del primario grande a la derecha.
        return 1.0 - self.u          # Garantiza u + w = 1.

# ----------------------- Campo efectivo en marco co-rotante -----------------

def r3bp_accel(x: float, y: float, vx: float, vy: float, p: R3BPParams) -> tuple[float, float]:  # Aceleraciones (x¨, y¨).
    u, w = p.u, p.w                 # Obtiene parámetros geométricos (posiciones relativas).
    r1_sq = (x + u) ** 2 + y**2     # Distancia al primario en (-u,0) al cuadrado.
    r2_sq = (x - w) ** 2 + y**2     # Distancia al primario en (+w,0) al cuadrado.
    r1_32 = r1_sq ** 1.5            # r1^(3/2) requerido por el potencial newtoniano.
    r2_32 = r2_sq ** 1.5            # r2^(3/2) requerido por el potencial newtoniano.
    ax = x + 2.0 * vy - w * (x + u) / r1_32 - u * (x - w) / r2_32  # Ecuación 8.183 para x¨ (centrífuga + Coriolis + gravedad).
    ay = y - 2.0 * vx - w * y / r1_32 - u * y / r2_32              # Ecuación 8.183 para y¨ (centrífuga + Coriolis + gravedad).
    return ax, ay                    # Devuelve el par de aceleraciones.

# ----------------------- Integrador de cuarto orden (RK4) -------------------

def rk4_step(state: np.ndarray, dt: float, p: R3BPParams) -> np.ndarray:  # Un paso de RK4 para el sistema en 1er orden.
    def f(s: np.ndarray) -> np.ndarray:  # Campo de velocidades y aceleraciones.
        x, y, vx, vy = s               # Desempaqueta el estado.
        ax, ay = r3bp_accel(x, y, vx, vy, p)  # Calcula aceleraciones del modelo.
        return np.array([vx, vy, ax, ay], dtype=float)  # Devuelve [x˙, y˙, x¨, y¨].
    k1 = f(state)                      # Primer incremento (en el estado actual).
    k2 = f(state + 0.5 * dt * k1)      # Segundo incremento (a mitad de dt con k1).
    k3 = f(state + 0.5 * dt * k2)      # Tercer incremento (a mitad de dt con k2).
    k4 = f(state + dt * k3)            # Cuarto incremento (al final con k3).
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)  # Combina los cuatro para el paso RK4.

# ----------------------- Bucle de integración temporal ----------------------

def integrate_r3bp(t_end: float, dt: float, x0: float, y0: float, vx0: float, vy0: float, p: R3BPParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(np.ceil(t_end / dt)) + 1   # Número total de pasos (ceil para cubrir t_end).
    t = np.linspace(0.0, t_end, n)     # Vector de tiempos uniforme (útil para graficar).
    traj = np.zeros((n, 2), float)     # Arreglo para posiciones [x,y] en cada tiempo.
    vel = np.zeros((n, 2), float)      # Arreglo para velocidades [vx,vy] en cada tiempo.
    state = np.array([x0, y0, vx0, vy0], float)  # Estado inicial empaquetado.
    traj[0] = state[:2]                # Guarda posición inicial.
    vel[0] = state[2:]                 # Guarda velocidad inicial.
    for i in range(1, n):              # Recorre cada paso temporal.
        state = rk4_step(state, dt, p) # Avanza el estado usando RK4.
        traj[i] = state[:2]            # Registra posición en el paso i.
        vel[i] = state[2:]             # Registra velocidad en el paso i.
    return t, traj, vel                # Devuelve tiempos, trayectorias y velocidades.

# ----------------------- Ejecución y visualización --------------------------

if __name__ == "__main__":             # Punto de entrada al ejecutar este archivo.
    p = R3BPParams()                   # Instancia con M/m = 80.45 (Tierra/Luna).
    x0, y0 = 0.994, 0.0                # Condición inicial en posición (del enunciado).
    vx0, vy0 = 0.0, -2.00158510637908  # Condición inicial en velocidad (del enunciado).
    T_end = 17.06521656015796          # Duración ~un periodo de la órbita de Arenstorf.
    dt = 2e-4                          # Paso temporal (precisión razonable).
    t, traj, vel = integrate_r3bp(T_end, dt, x0, y0, vx0, vy0, p)  # Integra el sistema.
    final_state = np.array([traj[-1,0], traj[-1,1], vel[-1,0], vel[-1,1]])  # Estado al final de la integración.
    init_state = np.array([x0, y0, vx0, vy0])  # Estado inicial para comparar cierre.
    err = np.linalg.norm(final_state - init_state)  # Norma de la diferencia (qué tan “cerrada” quedó).
    print(f"u={p.u:.8f}, w={p.w:.8f}, ||estado(T)-estado(0)|| ≈ {err:.3e}, dt={dt:g}")  # Reporte de parámetros y error de cierre.
    plt.figure()                       # Figura de la trayectoria.
    plt.plot(traj[:,0], traj[:,1], label="Trayectoria")  # Curva (x(t), y(t)).
    plt.scatter([-p.u, p.w], [0.0, 0.0], s=30, marker="x", label="Primarios")  # Marca las posiciones de los primarios.
    plt.axis("equal")                  # Escalas iguales para apreciar la geometría.
    plt.xlabel("x")                    # Etiqueta eje x.
    plt.ylabel("y")                    # Etiqueta eje y.
    plt.title("Órbita de Arenstorf – problema restringido de 3 cuerpos")  # Título descriptivo.
    plt.legend()                       # Leyenda visible.
    plt.tight_layout()                 # Ajusta márgenes.
    plt.figure()                       # Figura para v_x(t).
    plt.plot(t, vel[:,0])              # Traza componente v_x frente a t.
    plt.xlabel("t")                    # Etiqueta eje x.
    plt.ylabel("v_x(t)")               # Etiqueta eje y.
    plt.title("Componente de velocidad v_x(t)")  # Título de la figura.
    plt.tight_layout()                 # Ajusta márgenes.
    plt.figure()                       # Figura para v_y(t).
    plt.plot(t, vel[:,1])              # Traza componente v_y frente a t.
    plt.xlabel("t")                    # Etiqueta eje x.
    plt.ylabel("v_y(t)")               # Etiqueta eje y.
    plt.title("Componente de velocidad v_y(t)")  # Título de la figura.
    plt.tight_layout()                 # Ajusta márgenes.
    plt.show()                         # Muestra todas las figuras creadas.
