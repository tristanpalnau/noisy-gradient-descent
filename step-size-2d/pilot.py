import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def f(x, y):
    return (x - 1) ** 2 + 4 * (y + 2) ** 2

def grad(x, y):
    return np.array([2 * (x - 1), 8 * (y + 2)])

def gradient_descent(x0, y0, alpha, max_iter=200, tol=1e-6):
    p = np.array([x0, y0], dtype=float)
    path = [p.copy()]
    for _ in range(max_iter):
        g = grad(*p)
        if np.linalg.norm(g) < tol:
            break
        p = p - alpha * g
        path.append(p.copy())
        if np.linalg.norm(p) > 1e6:  # diverged, stop early
            break
    return np.array(path)

x0, y0 = 2.0, 0.0
alphas = [0.02, 0.1, 0.2, 0.3]
# Show limited iterations for the divergent case so the picture stays readable
max_iters = {0.02: 200, 0.1: 200, 0.2: 200, 0.3: 6}

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, a in zip(axes.ravel(), alphas):
    path = gradient_descent(x0, y0, a, max_iter=max_iters[a])
    xs, ys = path[:, 0], path[:, 1]

    # axis limits: at least the region R, expanded to fit the path
    xlo, xhi = min(-1, xs.min() - 0.3), max(3, xs.max() + 0.3)
    ylo, yhi = min(-3, ys.min() - 0.5), max(1, ys.max() + 0.5)

    X, Y = np.meshgrid(np.linspace(xlo, xhi, 400), np.linspace(ylo, yhi, 400))
    cs = ax.contour(X, Y, f(X, Y), levels=np.geomspace(0.05, f(X, Y).max(), 14),
                    cmap="viridis", linewidths=0.9)
    ax.plot(xs, ys, "o-", color="crimson", ms=3.5, lw=1.2, label="iterates")
    ax.plot(xs[0], ys[0], "s", color="black", ms=7, label="start (2, 0)")
    ax.plot(1, -2, "*", color="gold", mec="black", ms=15, label="minimizer (1, -2)")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_title(f"alpha = {a}  ({len(path) - 1} steps shown)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)

fig.suptitle("Gradient descent paths on f(x,y) = (x-1)^2 + 4(y+2)^2", fontsize=13)
fig.tight_layout()
fig.savefig("gd_paths.png", dpi=150)

# Numerical observations
print("alpha | x factor (1-2a) | y factor (1-8a) | steps to ||grad||<1e-6 | final point")
for a in alphas:
    p = gradient_descent(x0, y0, a, max_iter=200)
    conv = np.linalg.norm(grad(*p[-1])) < 1e-6
    print(f"{a:5} | {1 - 2 * a:15.3f} | {1 - 8 * a:15.3f} | "
          f"{(len(p) - 1) if conv else 'no convergence':>22} | {p[-1]}")
