# DEP 803 -- Chapter 8: Optimisation
# Local versus global minima: multi-start search on a non-convex risk
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy.optimize import minimize


def f(x):
    """A non-convex one-dimensional 'risk' with several local minima."""
    return 0.1 * x**2 + np.sin(3 * x) + 0.5 * np.cos(5 * x)


rng = np.random.default_rng(803)
starts = rng.uniform(-6, 6, 12)
found = []
for s in starts:
    res = minimize(lambda z: f(z[0]), [s], method="BFGS")
    found.append((s, res.x[0], res.fun))
print(f"{'start':>8}{'local min x*':>14}{'f(x*)':>10}")
for s, xs, fs in sorted(found, key=lambda t: t[0]):
    print(f"{s:>8.3f}{xs:>14.4f}{fs:>10.4f}")
xs = np.array([t[1] for t in found])
uniq = np.unique(np.round(xs, 3))
print("distinct local minima found:", len(uniq))
best = min(found, key=lambda t: t[2])
grid = np.linspace(-6, 6, 200001)
print(f"best multi-start: x = {best[1]:.4f}, f = {best[2]:.4f}")
print(f"grid search     : x = {grid[np.argmin(f(grid))]:.4f}, "
      f"f = {f(grid).min():.4f}")
