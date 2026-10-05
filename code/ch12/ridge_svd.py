# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Ridge: three equivalent solutions, p > n, SVD shrinkage, bias and variance.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)

# ---- p > n: X'X is singular but X'X + lambda I is not
n, p, lam = 30, 60, 2.0
X = rng.normal(size=(n, p))
y = X[:, :3] @ np.array([2.0, -1.0, 1.0]) + rng.normal(size=n)
print(f"n = {n}, p = {p}: rank(X'X) = {np.linalg.matrix_rank(X.T @ X)}")
b1 = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)        # normal eq.
U, d, Vt = np.linalg.svd(X, full_matrices=False)
b2 = Vt.T @ (d / (d**2 + lam) * (U.T @ y))                      # SVD form
Xa = np.vstack([X, np.sqrt(lam) * np.eye(p)])                   # augmented
b3 = np.linalg.lstsq(Xa, np.r_[y, np.zeros(p)], rcond=None)[0]
print(f"max |normal - SVD| = {np.abs(b1 - b2).max():.1e}, "
      f"max |normal - augmented| = {np.abs(b1 - b3).max():.1e}")
b_small = Vt.T @ (d / (d**2 + 1e-10) * (U.T @ y))
print("lambda -> 0 gives the min-norm solution pinv(X)y:",
      np.allclose(b_small, np.linalg.pinv(X) @ y, atol=1e-6))
print("shrinkage factors d^2/(d^2+lambda), largest and smallest d:",
      np.round((d**2 / (d**2 + lam))[[0, -1]], 4))

# ---- bias and variance of ridge with a fixed, collinear design
n, p, sigma = 100, 10, 1.0
C = 0.9 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
X = rng.multivariate_normal(np.zeros(p), C, size=n)
beta = rng.normal(size=p) * 0.5
G = X.T @ X
R = 4000
Y = X @ beta + sigma * rng.normal(size=(R, n))                  # R samples
print(f"\n{'lambda':>7}{'bias^2':>9}{'trace Var':>11}{'MSE theory':>12}"
      f"{'MSE MC':>9}")
for lam in [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
    A = np.linalg.inv(G + lam * np.eye(p))
    bias = -lam * A @ beta                                      # E b - beta
    V = sigma**2 * A @ G @ A
    B = Y @ X @ A                                               # rows = b_hat
    mse_mc = np.mean(np.sum((B - beta) ** 2, axis=1))
    print(f"{lam:>7.1f}{bias @ bias:>9.4f}{np.trace(V):>11.4f}"
          f"{bias @ bias + np.trace(V):>12.4f}{mse_mc:>9.4f}")
