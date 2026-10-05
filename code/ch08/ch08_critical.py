# DEP 803 -- Chapter 8: Optimisation
# Locate and classify stationary points with the gradient and Hessian
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root


def f(v):
    x, y = v
    return x**3 - 3 * x + y**2 - 2 * x * y**2 / 4


def grad(v):
    x, y = v
    return np.array([3 * x**2 - 3 - 0.5 * y**2, 2 * y - x * y])


def hess(v):
    x, y = v
    return np.array([[6 * x, -y], [-y, 2 - x]])


rng = np.random.default_rng(803)
pts = []
for s in rng.uniform(-3, 3, size=(60, 2)):
    sol = root(grad, s, jac=hess)
    sol.x[np.abs(sol.x) < 1e-10] = 0.0
    if sol.success and not any(np.allclose(sol.x, q, atol=1e-6) for q in pts):
        pts.append(sol.x)
print(f"{'x':>8}{'y':>8}{'f':>9}   Hessian eigenvalues    type")
for q in sorted(pts, key=lambda t: (t[0], t[1])):
    lam = np.linalg.eigvalsh(hess(q))
    kind = ("local min" if lam.min() > 0 else
            "local max" if lam.max() < 0 else
            "saddle" if lam.min() < 0 < lam.max() else "degenerate")
    print(f"{q[0]:>8.3f}{q[1]:>8.3f}{f(q):>9.3f}   "
          f"[{lam[0]:>7.3f}, {lam[1]:>7.3f}]   {kind}")

xx, yy = np.meshgrid(np.linspace(-2.5, 3.5, 300), np.linspace(-3, 3, 300))
zz = f((xx, yy))
plt.figure(figsize=(6, 3.8))
cs = plt.contour(xx, yy, zz, levels=np.linspace(-4, 6, 26), cmap="viridis")
for q in pts:
    lam = np.linalg.eigvalsh(hess(q))
    mk = "o" if lam.min() > 0 else ("^" if lam.max() < 0 else "x")
    plt.plot(q[0], q[1], mk, color="crimson", ms=8, mew=2)
plt.gca().set(xlabel="x", ylabel="y",
              title="Stationary points: o min, ^ max, x saddle")
plt.savefig("ch08_critical.pdf", bbox_inches="tight")
