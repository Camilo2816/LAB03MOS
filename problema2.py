import sympy as sp
import numpy as np 
import matplotlib.pyplot as plt

# Variable y función
x = sp.symbols('x', real=True)
f = x**5 - 8*x**3 + 10*x + 6

# Primera y segunda derivada
f1 = sp.diff(f, x)          # f'(x)
f2 = sp.diff(f, x, 2)       # f''(x)

# simplificación y factorización
f1_simp = sp.simplify(f1)
f2_simp = sp.simplify(f2)
f1_fact = sp.factor(f1_simp)
f2_fact = sp.factor(f2_simp)


# Mostrar resultados simbólicos
print("Función f(x):", f)
print("Primera derivada f'(x):", f1)
print("Segunda derivada f''(x):", f2)
print("Primera derivada simplificada f'(x):", f1_simp)
print("Segunda derivada simplificada f''(x):", f2_simp)


def newton_extremo(f_expr, x0, alpha=0.6, tol=1e-6, max_iter=100):
    """
    NewtonRaphson aplicado a g(x)=f'(x) para hallar extremos de f.
    Devuelve un diccionario con x*, tipo de extremo y el historial.
    """
    df_expr  = sp.diff(f_expr, x)
    d2f_expr = sp.diff(f_expr, x, 2)
    g  = sp.lambdify(x, df_expr,  'numpy')
    h  = sp.lambdify(x, d2f_expr, 'numpy')
    ff = sp.lambdify(x, f_expr,   'numpy')


    # Listas para almacenar el historial
    xs, fs, gs = [float(x0)], [ff(x0)], [g(x0)]
    xk = float(x0)

    for _ in range(max_iter):
        hk = h(xk)
        if abs(hk) < 1e-12:  # evitar que si f'' es 0 se divida por 0             
            break
        xk1 = xk - alpha * gs[-1] / hk   
        xs.append(xk1); fs.append(ff(xk1)); gs.append(g(xk1))
        if abs(gs[-1]) < tol:            # criterio: ||grad|| = |f'(x)| < tol
            break
        xk = xk1

    x_star = xs[-1]
    tipo = "mínimo" if h(x_star) > 0 else ("máximo" if h(x_star) < 0 else "indeterminado")
    return {
        "x_star": x_star,
        "f_x_star": ff(x_star),
        "tipo": tipo,
        "xs": np.array(xs),
        "fs": np.array(fs),
        "grads": np.array(gs),
    }
    
semillas = np.linspace(-3, 3, 25)        
resultados = []
for s in semillas:
    res = newton_extremo(f, s, alpha=0.6, tol=1e-8, max_iter=200)
    # guardar solo convergencias dentro del intervalo
    if -3 - 1e-6 <= res["x_star"] <= 3 + 1e-6 and np.isfinite(res["x_star"]):
        resultados.append(res)

# deduplicar raíces cercanas 
tol_root = 1e-5
unicos = []
for r in resultados:
    if not any(abs(r["x_star"] - u["x_star"]) < tol_root for u in unicos):
        unicos.append(r)

# ordenar por x*
unicos = sorted(unicos, key=lambda d: d["x_star"])

# mostrar resumen
for u in unicos:
    print(f"x* = {u['x_star']:.10f}   f(x*) = {u['f_x_star']:.10f}   tipo = {u['tipo']}")
    

for r in resultados:
    if not any(abs(r["x_star"] - u["x_star"]) < tol_root for u in unicos):
        unicos.append(r)

unicos = sorted(unicos, key=lambda d: d["x_star"])
unicos = [u for u in unicos if abs(u["x_star"]) > 1e-5]


f_l = sp.lambdify(x, f, 'numpy')
# Tabla: x*, f(x*), tipo
print("x*\t\t\tf(x*)\t\t\ttipo")
tabla_13 = []
for u in unicos:
    xi = float(u["x_star"])
    fxi = float(f_l(xi))
    tipo = u["tipo"]
    tabla_13.append((xi, fxi, tipo))
    print(f"{xi: .9f}\t {fxi: .9f}\t {tipo}")
    
    
criticos = sorted([u["x_star"] for u in unicos])



candidatos = criticos + [-3.0, 3.0]

# Evaluar f en todos los candidatos
evals = [(xi, float(f_l(xi))) for xi in candidatos]

# Elegir globales
x_max, f_max = max(evals, key=lambda t: t[1])
x_min, f_min = min(evals, key=lambda t: t[1])

print("Candidatos (x, f(x)):")
for xi, fxi in sorted(evals):
    print(f"x = {xi: .9f}   f(x) = {fxi: .9f}")

print("\n>>> Máximo global:  x* = {:.9f}   f(x*) = {:.9f}".format(x_max, f_max))
print(">>> Mínimo global:  x* = {:.9f}   f(x*) = {:.9f}".format(x_min, f_min))


xs = np.linspace(-3, 3, 600)
ys = f_l(xs)

# 4) Plot
plt.figure(figsize=(8,5))
plt.plot(xs, ys, label=r"$f(x)=x^5-8x^3+10x+6$")
# Extremos locales (negro)
xl = [xi for xi in criticos]
yl = [f_l(xi) for xi in criticos]

plt.scatter(xl, yl, s=60, color="black", zorder=3, label="Extremos locales")

# Globales (rojo)
plt.scatter([x_max, x_min], [f_max, f_min], s=90, color="red", zorder=4, label="Extremos globales")

# Anotaciones útiles
plt.axhline(0, linewidth=0.8, alpha=0.6)
plt.axvline(0, linewidth=0.8, alpha=0.6)
plt.xlim(-3, 3)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Función y extremos en [-3, 3]")
plt.grid(True, alpha=0.25)
plt.legend(loc="best")
plt.tight_layout()
plt.show()