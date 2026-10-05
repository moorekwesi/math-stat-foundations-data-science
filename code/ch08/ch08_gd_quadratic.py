# DEP 803 -- Chapter 8: Optimisation
# Gradient descent on quadratics: condition number, rates and zig-zag
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt


def gd(A, x0, step, tol=1e-8, maxit=200000):
    """GD on f(x) = x'Ax/2; returns iterates until ||x|| < tol*||x0||."""
    x, path = x0.copy(), [x0.copy()]
    for k in range(maxit):
        x = x - step * (A @ x)
        path.append(x.copy())
        if np.linalg.norm(x) < tol * np.linalg.norm(x0):
            break
    return np.array(path)


x0 = np.array([1.0, 1.0])
print(f"{'kappa':>7}{'iters 1/L':>11}{'iters 2/(mu+L)':>16}"
      f"{'rate (k-1)/(k+1)':>18}{'predicted':>11}")
for kappa in [1, 10, 100, 1000]:
    A = np.diag([1.0, kappa])
    n1 = len(gd(A, x0, 1 / kappa)) - 1
    n2 = len(gd(A, x0, 2 / (1 + kappa))) - 1
    rate = (kappa - 1) / (kappa + 1)
    pred = np.ceil(np.log(1e-8) / np.log(rate)) if kappa > 1 else 1
    print(f"{kappa:>7}{n1:>11}{n2:>16}{rate:>18.4f}{pred:>11.0f}")

# zig-zag picture for kappa = 10 with x0 chosen on the "bad" direction
A = np.diag([1.0, 10.0])
x0 = np.array([10.0, 1.0])
path = gd(A, x0, 2 / 11)[:25]
mom = [x0.copy()]
x, v = x0.copy(), np.zeros(2)
beta, alpha = ((np.sqrt(10) - 1) / (np.sqrt(10) + 1))**2, \
    4 / (1 + np.sqrt(10))**2
for k in range(24):
    v = beta * v - alpha * (A @ x)
    x = x + v
    mom.append(x.copy())
mom = np.array(mom)
print(f"after 24 steps: GD ||x|| = {np.linalg.norm(path[-1]):.2e}, "
      f"heavy-ball ||x|| = {np.linalg.norm(mom[-1]):.2e}")

xx, yy = np.meshgrid(np.linspace(-11, 11, 300), np.linspace(-3.5, 3.5, 300))
plt.figure(figsize=(6, 3.8))
plt.contour(xx, yy, 0.5 * (xx**2 + 10 * yy**2), levels=15, cmap="Greys",
            linewidths=0.7)
plt.plot(path[:, 0], path[:, 1], "o-", ms=3, label="gradient descent")
plt.plot(mom[:, 0], mom[:, 1], "s-", ms=3, label="heavy-ball momentum")
plt.gca().set(xlabel="x1 (flat direction, eigenvalue 1)",
              ylabel="x2 (steep, eigenvalue 10)", title="Zig-zag of GD on an elongated quadratic (kappa = 10)")
plt.legend(frameon=False, loc="upper right")
plt.savefig("ch08_zigzag.pdf", bbox_inches="tight")
