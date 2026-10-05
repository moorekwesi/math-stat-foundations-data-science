# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Covariance matrices are PSD; the variance of a composite index is w' S w.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)

# Simulated district indicators: schooling, health, assets, water
names = ["school", "health", "assets", "water"]
R = np.array([[1.0, 0.6, 0.5, 0.3],
              [0.6, 1.0, 0.4, 0.4],
              [0.5, 0.4, 1.0, 0.2],
              [0.3, 0.4, 0.2, 1.0]])
sd = np.array([1.0, 0.8, 1.5, 1.2])
S = np.outer(sd, sd) * R                       # population covariance
X = rng.multivariate_normal(np.zeros(4), S, size=5000)
S_hat = np.cov(X, rowvar=False)
print("eigenvalues of S_hat:", np.round(np.linalg.eigvalsh(S_hat), 3))

w = np.full(4, 0.25)                            # equal-weight composite index
index = X @ w
print(f"Var(index): w'S w = {w @ S @ w:.4f}  w'S_hat w = {w @ S_hat @ w:.4f}"
      f"  sample = {index.var(ddof=1):.4f}")
print(f"  ignoring covariances: sum w_i^2 s_ii = {np.sum(w**2 * np.diag(S)):.4f}")

# Minimum-variance weights (sum to one): w* proportional to S^{-1} 1
one = np.ones(4)
wmin = np.linalg.solve(S, one)
wmin /= wmin.sum()
print("min-variance weights:", dict(zip(names, np.round(wmin, 3))))
print(f"  variance at w*: {wmin @ S @ wmin:.4f}")

# A 'correlation matrix' assembled from inconsistent pairwise estimates
B = np.array([[1.0, 0.9, 0.9],
              [0.9, 1.0, -0.9],
              [0.9, -0.9, 1.0]])
lam, V = np.linalg.eigh(B)
print("\ninconsistent matrix eigenvalues:", np.round(lam, 3))
v = V[:, 0]
print(f"  'variance' of v'X for v = {np.round(v, 3)}: {v @ B @ v:.3f}  (< 0!)")
