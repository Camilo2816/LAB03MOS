from networkx import display
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


x = sp.symbols('x', real=True)
f_expr = 3*x**3 - 10*x**2 - 56*x + 50

df_expr  = sp.diff(f_expr, x)        # f'(x)
d2f_expr = sp.diff(f_expr, x, 2)     # f''(x)

f   = sp.lambdify(x, f_expr,  'numpy')
df  = sp.lambdify(x, df_expr, 'numpy')
d2f = sp.lambdify(x, d2f_expr,'numpy')

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


# --- Conjuntos a probar ---
x0_list  = np.linspace(-6, 6, 13)            
alphas   = [1.0, 0.8, 0.6, 0.4]               
tol      = 1e-6
max_iter = 80

# --- Ejecutar barrido y construir tabla ---
rows = []
traj_por_caso = {}  

for a in alphas:
    for x0 in x0_list:
        res = newton_extremo(f_expr, x0=float(x0), alpha=a, tol=tol, max_iter=max_iter)
        iters = len(res["xs"]) - 1
        grad_final = abs(res["grads"][-1])
        converged = grad_final < tol
        rows.append({
            "x0": float(x0),
            "alpha": a,
            "x_star": res["x_star"],
            "f_x_star": res["f_x_star"],
            "tipo": res["tipo"],
            "iteraciones": iters,
            "grad_final": grad_final,
            "convergió": converged
        })
        traj_por_caso[(round(float(x0),3), a)] = res["grads"]  # guarda |f'(x_k)|

df = pd.DataFrame(rows).sort_values(["alpha","x0"]).reset_index(drop=True)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9, 5))

# Curva de la función y extremos encontrados por los barridos
x_plot = np.linspace(-6, 6, 600)
y_plot = f(x_plot)
ax.plot(x_plot, y_plot, color='#2c7fb8', lw=2.2, label='f(x)')

# Tomar solo casos convergentes y deduplicar por x* (con redondeo)
conv = df[df["convergió"]].copy()
if not conv.empty:
    conv["x_round"] = conv["x_star"].round(6)
    unicos = conv.drop_duplicates("x_round")

    maximos = unicos[unicos["tipo"] == "máximo"]
    minimos = unicos[unicos["tipo"] == "mínimo"]

    # Dispersión de extremos con bordes para mejor contraste
    if not maximos.empty:
        ax.scatter(maximos["x_star"], maximos["f_x_star"], color='#d62728', edgecolor='white',
                   linewidths=0.7, marker='^', s=90, zorder=3, label='Máximos')
    if not minimos.empty:
        ax.scatter(minimos["x_star"], minimos["f_x_star"], color='#2ca02c', edgecolor='white',
                   linewidths=0.7, marker='o', s=90, zorder=3, label='Mínimos')

    # Ajuste de límites con margen superior para evitar solape con el título
    ys_ext = [np.min(y_plot), np.max(y_plot)]
    ys_pts = unicos["f_x_star"].to_numpy()
    y_min = float(min(np.min(ys_ext), np.min(ys_pts)))
    y_max = float(max(np.max(ys_ext), np.max(ys_pts)))
    y_rng = y_max - y_min if y_max > y_min else 1.0
    ax.set_ylim(y_min - 0.05*y_rng, y_max + 0.18*y_rng)

    # Anotaciones dinámicas: arriba o abajo según cercanía al borde superior
    for _, r in unicos.iterrows():
        y = float(r["f_x_star"])
        y_frac = (y - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        dy = -10 if y_frac > 0.88 else 10  # si muy arriba, anotar hacia abajo
        color = '#d62728' if r['tipo'] == 'máximo' else ('#2ca02c' if r['tipo'] == 'mínimo' else '#333')
        ax.annotate(f"x={r['x_star']:.3f}", (r["x_star"], y),
                    textcoords='offset points', xytext=(0, dy), ha='center', fontsize=9,
                    color=color, zorder=4)

# Títulos y ejes con más aire para el título
ax.set_title('f(x) con máximos y mínimos encontrados', pad=14)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.grid(True, alpha=0.35)

# Leyenda más limpia
leg = ax.legend(frameon=True, framealpha=0.9, fancybox=True, borderpad=0.6)
for text in leg.get_texts():
    text.set_fontsize(9)

# Ajuste de márgenes: más espacio superior para evitar solapes con anotaciones/título
fig.subplots_adjust(top=0.88)
plt.show()




