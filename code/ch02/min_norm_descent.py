# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Test error of minimum-norm least squares as p crosses n (double descent)
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
n, d, n_test, reps, sigma = 40, 200, 500, 60, 0.5
beta = 1 / np.sqrt(np.arange(1, d + 1))          # decaying true coefficients
beta *= 2.0 / np.linalg.norm(beta)
ps = np.unique(np.r_[np.arange(2, 36, 3), np.arange(36, 46), 50, 60, 80,
                     100, 130, 160, 200])
risk = np.zeros((reps, ps.size))
norms = np.zeros((reps, ps.size))
for r in range(reps):
    X = rng.normal(size=(n, d))
    y = X @ beta + sigma * rng.normal(size=n)
    Xt = rng.normal(size=(n_test, d))
    yt = Xt @ beta + sigma * rng.normal(size=n_test)
    for j, p in enumerate(ps):
        b = np.linalg.lstsq(X[:, :p], y, rcond=None)[0]   # min-norm if p > n
        risk[r, j] = np.mean((yt - Xt[:, :p] @ b) ** 2)
        norms[r, j] = np.linalg.norm(b)
med = np.median(risk, axis=0)
print("  p   p/n   median test MSE   median ||b||")
for p, m, nb in zip(ps, med, np.median(norms, axis=0)):
    if p in (5, 20, 35, 38, 40, 42, 45, 60, 100, 200):
        print(f"{p:4d}  {p / n:4.2f}  {m:14.3f}  {nb:12.3f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogy(ps / n, med, "o-", ms=4, label="median test MSE")
ax.axvline(1.0, color="k", ls=":", lw=1)
ax.axhline(sigma ** 2 + beta @ beta, color="gray", ls="--", lw=1,
           label="null model (b = 0)")
ax.set_xlabel("p / n  (number of regressors / sample size)")
ax.set_ylabel("test mean squared error")
ax.text(1.05, med.max() / 2, "interpolation\nthreshold p = n", fontsize=8)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("ch02_min_norm_descent.pdf", bbox_inches="tight")
