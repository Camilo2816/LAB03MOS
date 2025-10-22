import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Variables simbólicas
x, y = sp.symbols('x y', real=True)

# Definición de la función de Rosenbrock
f = (x - 1)**2 + 100*(y - x**2)**2

# Gradiente (vector de derivadas parciales)
grad_f = sp.Matrix([sp.diff(f, x), sp.diff(f, y)]).applyfunc(sp.simplify)

# Hessiana (matriz de segundas derivadas)
H_f = sp.hessian(f, (x, y)).applyfunc(sp.simplify)

# Mostrar resultados 
sp.pprint(f, use_unicode=True)
print("\nGradiente ∇f(x, y):")
sp.pprint(grad_f, use_unicode=True)
print("\nHessiana Hf(x, y):")
sp.pprint(H_f, use_unicode=True)

# Lambdify para evaluaciones numéricas
f = sp.lambdify((x, y), f, 'numpy')
g = sp.lambdify((x, y), grad_f, 'numpy')
H = sp.lambdify((x, y), H_f, 'numpy')

def newton2d(f, g, H, x0, y0, tol=1e-8, max_iter=100, use_backtracking=True):
    """
    Newton–Raphson para f: R^2 -> R.
    Devuelve diccionario con iteraciones y el punto final.
    Criterios de parada: ||grad||_2 < tol o ||paso||_2 < tol.
    """
    xk = np.array([float(x0), float(y0)], dtype=float)
    history = []
    for k in range(max_iter):
        grad = np.asarray(g(xk[0], xk[1]), dtype=float).reshape(2)
        Hk   = np.asarray(H(xk[0], xk[1]), dtype=float)

        grad_norm = np.linalg.norm(grad, 2)

        # Paso de Newton resolviendo Hk * s = -grad
        try:
            sk = np.linalg.solve(Hk, -grad)
        except np.linalg.LinAlgError:
            # Regularización si la Hessiana es mal condicionada
            Hk_reg = Hk + 1e-6*np.eye(2)
            sk = np.linalg.solve(Hk_reg, -grad)

        
        t = 1.0
        if use_backtracking:
            c, beta = 1e-4, 0.5
            f0 = f(xk[0], xk[1])
            while f(xk[0] + t*sk[0], xk[1] + t*sk[1]) > f0 + c*t*grad.dot(sk) and t > 1e-12:
                t *= beta

        step = t*sk
        xk1 = xk + step

        history.append({
            "k": k,
            "x": xk[0], "y": xk[1],
            "f": float(f(xk[0], xk[1])),
            "grad_norm": float(grad_norm),
            "step_norm": float(np.linalg.norm(step, 2)),
            "t": float(t)
        })

        # Criterios de parada
        if grad_norm < tol or np.linalg.norm(step, 2) < tol:
            xk = xk1
            break

        xk = xk1

    result = {
        "x_star": xk[0],
        "y_star": xk[1],
        "f_star": float(f(xk[0], xk[1])),
        "iterations": history
    }
    return result

res = newton2d(f, g, H, x0=0, y0=10, tol=1e-10, max_iter=100, use_backtracking=True)

# Mostrar resumen
print("Punto final:")
print(f"  x* = {res['x_star']:.12f}, y* = {res['y_star']:.12f}, f* = {res['f_star']:.12e}")
print(f"Iteraciones realizadas: {len(res['iterations'])}")

for it in res["iterations"][:5]:
    print(f"k={it['k']:2d}  (x,y)=({it['x']:.6f},{it['y']:.6f})  "
          f"f={it['f']:.6e}  ||g||={it['grad_norm']:.3e}  "
          f"||step||={it['step_norm']:.3e}  t={it['t']:.2e}")
    
    
# --- Gráfica de la superficie z = f(x,y) con el mínimo encontrado ---


# Crear malla para la superficie
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Figura 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Superficie
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                rstride=10, cstride=10, edgecolor='none')

# Punto mínimo encontrado
ax.scatter(res['x_star'], res['y_star'], res['f_star'],
           color='r', s=100, label='Mínimo encontrado')

# Configuración
ax.set_title('Superficie de Rosenbrock con mínimo encontrado', fontsize=11)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
plt.tight_layout()
plt.show()

# --- Conjuntos de prueba (para pasos siguientes) ---
x0_list = np.linspace(-6, 6, 13)
y0_list = np.linspace(-6, 6, 13)
