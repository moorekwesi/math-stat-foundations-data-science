# DEP 803 -- Chapter 8: Optimisation
# GD, Armijo backtracking, heavy-ball and Nesterov on least squares
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
n = 2000
# simulated household survey: size, land (acres), years of schooling of head
hh = rng.poisson(4, n) + 1.0
land = rng.gamma(2.0, 1.5, n)
sch = rng.uniform(0, 18, n)
X = np.column_stack([np.ones(n), hh, land, sch])
y = X @ np.array([2.0, 0.3, 0.5, 0.1]) + rng.normal(0, 1, n)


def make(X):
    f = lambda b: 0.5 * np.mean((y - X @ b)**2)
    g = lambda b: -X.T @ (y - X @ b) / len(y)
    eig = np.linalg.eigvalsh(X.T @ X / len(y))
    return f, g, eig.min(), eig.max()


def run(X, method, iters=3000):
    f, g, mu, L = make(X)
    fstar = f(np.linalg.lstsq(X, y, rcond=None)[0])
    b = np.zeros(X.shape[1])
    b_prev, hist = b.copy(), []
    t = 1.0
    for k in range(iters):
        hist.append(f(b) - fstar)
        if method == "GD 1/L":
            b = b - g(b) / L
        elif method == "GD + Armijo":
            gb, t = g(b), min(1.0, 4 * t)      # backtracking line search
            while f(b - t * gb) > f(b) - 0.5 * t * gb @ gb:
                t *= 0.5
            b = b - t * gb
        elif method == "heavy ball":
            q = np.sqrt(L / mu)
            a, m = 4 / (np.sqrt(L) + np.sqrt(mu))**2, ((q - 1) / (q + 1))**2
            b, b_prev = b - a * g(b) + m * (b - b_prev), b
        elif method == "Nesterov":
            q = np.sqrt(L / mu)
            z = b + (q - 1) / (q + 1) * (b - b_prev)
            b, b_prev = z - g(z) / L, b
    return np.maximum(np.array(hist), 1e-16), L / mu


methods = ["GD 1/L", "GD + Armijo", "heavy ball", "Nesterov"]
plt.figure(figsize=(6, 3.8))
print(f"{'method':<14}{'gap after 3000':>16}{'iters to 1e-8':>15}")
for m in methods:
    h, kappa = run(X, m)
    hit = str(np.argmax(h < 1e-8)) if (h < 1e-8).any() else "> 3000"
    print(f"{m:<14}{h[-1]:>16.2e}{hit:>15}")
    plt.semilogy(h, label=m)
print(f"condition number of X'X/n (raw): {kappa:,.0f}")

Xs = X.copy()
Xs[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
h, kappa = run(Xs, "GD 1/L", 200)
print(f"standardised: kappa = {kappa:.2f}, GD 1/L reaches 1e-8 in "
      f"{np.argmax(h < 1e-8)} iterations")
plt.gca().set(xlabel="iteration k", ylabel=r"$f(\beta_k) - f^\star$")
plt.ylim(1e-14, 1e2)
plt.legend(frameon=False)
plt.title("First-order methods on raw (unscaled) least squares")
plt.savefig("ch08_gd_rates.pdf", bbox_inches="tight")
