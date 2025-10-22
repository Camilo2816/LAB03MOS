# Paso 2 — Gradiente y Hessiana analíticos
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa



x, y, z = sp.symbols('x y z', real=True)
f = (x - 1)**2 + (y - 2)**2 + (z - 3)**2

# Gradiente (vector 3x1) y Hessiana (3x3)
grad_f = sp.Matrix([sp.diff(f, x), sp.diff(f, y), sp.diff(f, z)])
H_f    = sp.hessian(f, (x, y, z))

print("f(x,y,z) =")
sp.pprint(f, use_unicode=False)

print("\n∇f(x,y,z) =")
sp.pprint(grad_f, use_unicode=False)

print("\nH_f(x,y,z) =")
sp.pprint(H_f, use_unicode=False)

# (para los siguientes pasos) funciones numéricas
f_func    = sp.lambdify((x, y, z), f, "numpy")
grad_func = sp.lambdify((x, y, z), grad_f, "numpy")
hess_func = sp.lambdify((x, y, z), H_f, "numpy")

# Paso 3 — Newton–Raphson 3D con criterio ||grad|| < ε

# Definición de la función, gradiente y Hessiana
def f_np(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-3)**2

def grad_np(x):
    return 2.0 * (x - np.array([1.0, 2.0, 3.0]))

def hess_np(x):
    return 2.0 * np.eye(3)

def newton3d(f, grad, hess, x0, tol=1e-10, max_iter=50):
    """
    Método de Newton–Raphson en R^3.
    Criterio de parada: ||∇f(x_k)||_2 < tol
    """
    xk = np.array(x0, dtype=float)
    history = []

    for k in range(max_iter):
        gk = grad(xk)
        Hk = hess(xk)
        grad_norm = np.linalg.norm(gk, 2)

        # Guardar información de la iteración
        history.append({
            "k": k,
            "x": xk.copy(),
            "f": f(xk),
            "grad_norm": grad_norm
        })

        # Criterio de parada (basado en la norma del gradiente)
        if grad_norm < tol:
            break

        # Paso de Newton
        sk = np.linalg.solve(Hk, -gk)
        xk = xk + sk

    return {"x_star": xk, "f_star": f(xk), "iterations": history}

# Ejecución
res = newton3d(f_np, grad_np, hess_np, x0=[0, 0, 0], tol=1e-10)
print("Punto final:", res["x_star"], "  f* =", res["f_star"])
print("Iteraciones:", len(res["iterations"]))

#Grafica de la superficie z = f(x,y,z) con el mínimo encontrado
x_star, y_star, z_star = res["x_star"]

# --- Superficie como corte a z = z* (no cambiamos Newton) ---
xv = np.linspace(x_star-2.5, x_star+2.5, 200)
yv = np.linspace(y_star-2.5, y_star+2.5, 200)
X, Y = np.meshgrid(xv, yv)
Zsurf = (X-1)**2 + (Y-2)**2 + (z_star-3)**2   # f(x,y, z=z*)

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

# Superficie f(x,y,z*) como “altura” (para visualizar el bowl alrededor del mínimo)
ax.plot_surface(X, Y, Zsurf, cmap='viridis', alpha=0.88, edgecolor='none')

# Marcar el mínimo encontrado (x*, y*, z*)
ax.scatter(x_star, y_star, z_star, c='r', s=90, label='Mínimo encontrado')

# Estética
ax.set_title("f(x,y,z) con corte z = z* y mínimo marcado")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.tight_layout()
plt.show()