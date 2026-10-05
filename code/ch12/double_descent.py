# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Double descent with random ReLU features and minimum-norm least squares.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
n, d, reps, sigma = 40, 5, 30, 0.2
f = lambda X: np.sin(X @ np.ones(d) / np.sqrt(d)) + 0.5 * X[:, 0]
widths = np.unique(np.r_[np.arange(2, 60, 2), np.geomspace(60, 1200, 18)]
                   .astype(int))
test_mse = np.zeros((reps, widths.size))
for r in range(reps):
    X = rng.normal(size=(n, d))
    y = f(X) + sigma * rng.normal(size=n)
    Xt = rng.normal(size=(1000, d))
    yt = f(Xt)
    Wall = rng.normal(size=(d, widths.max())) / np.sqrt(d)
    for k, p in enumerate(widths):
        Phi = np.maximum(X @ Wall[:, :p], 0)            # random ReLU features
        Phit = np.maximum(Xt @ Wall[:, :p], 0)
        w = np.linalg.pinv(Phi) @ y                     # min-norm solution
        test_mse[r, k] = np.mean((yt - Phit @ w) ** 2)
med = np.median(test_mse, axis=0)
for p in [10, 20, 30, 40, 50, 100, 400, 1200]:
    k = np.argmin(np.abs(widths - p))
    print(f"features p = {widths[k]:>5}  (p/n = {widths[k] / n:5.2f})"
          f"  median test MSE = {med[k]:9.4f}")
k_peak = np.argmax(med)
print(f"peak at p = {widths[k_peak]} (interpolation threshold p = n = {n})")

plt.figure(figsize=(6, 3.8))
plt.loglog(widths, med, "o-", ms=3)
plt.axvline(n, color="gray", ls="--", label="interpolation threshold p = n")
plt.xlabel("number of random features p")
plt.ylabel("median test MSE (log scale)")
plt.title("Double descent: min-norm least squares, n = 40")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.savefig("ch12_double_descent.pdf", bbox_inches="tight")
