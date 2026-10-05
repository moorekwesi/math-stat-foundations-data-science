# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Moore-Penrose pseudoinverse: Penrose conditions and minimum-norm least squares
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
A = rng.normal(size=(6, 3)) @ rng.normal(size=(3, 5))   # 6 x 5, rank 3
Ap = np.linalg.pinv(A)

# pinv from the SVD: A^+ = V diag(1/s_i) U^T over the non-zero s_i
U, s, Vt = np.linalg.svd(A, full_matrices=False)
r = int(np.sum(s > s[0] * 1e-12))
Ap_svd = Vt[:r].T @ np.diag(1 / s[:r]) @ U[:, :r].T
print(f"rank(A) = {r},  max|pinv - SVD formula| = {np.abs(Ap - Ap_svd).max():.1e}")

checks = {"(1) A A+ A   = A ": A @ Ap @ A - A,
          "(2) A+ A A+  = A+": Ap @ A @ Ap - Ap,
          "(3) (A A+)^T = A A+": (A @ Ap).T - A @ Ap,
          "(4) (A+ A)^T = A+ A": (Ap @ A).T - Ap @ A}
for name, D in checks.items():
    print(f"Penrose {name:22s} max error = {np.abs(D).max():.1e}")

# Naive inverse of A^T A fails for rank-deficient A
print(f"cond(A^T A) = {np.linalg.cond(A.T @ A):.1e}  (singular in exact arithmetic)")

# Minimum-norm least squares: all minimisers are x+ + null(A)
b = rng.normal(size=6)
x_plus = Ap @ b
x_lstsq, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
N = Vt[r:].T                          # basis of null(A)
x_other = x_plus + N @ rng.normal(size=N.shape[1])
print(f"\n||x_pinv - x_lstsq|| = {np.linalg.norm(x_plus - x_lstsq):.1e}")
print(f"residual norms: x+ {np.linalg.norm(A @ x_plus - b):.6f}, "
      f"x+ + null {np.linalg.norm(A @ x_other - b):.6f}")
print(f"solution norms: x+ {np.linalg.norm(x_plus):.4f}, "
      f"x+ + null {np.linalg.norm(x_other):.4f}")
print(f"x+ in row(A)?  ||(I - A+A) x+|| = "
      f"{np.linalg.norm(x_plus - Ap @ A @ x_plus):.1e}")

# Underdetermined p > n: interpolation with the minimum-norm solution
n, p = 20, 50
X = rng.normal(size=(n, p))
y = X[:, :5] @ np.ones(5) + 0.1 * rng.normal(size=n)
b_mn = np.linalg.pinv(X) @ y
print(f"\np = {p} > n = {n}: training residual = "
      f"{np.linalg.norm(X @ b_mn - y):.1e} (interpolates)")
lam = 1e-8
b_ridge = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
print(f"ridge, lambda = {lam:.0e}: ||b_ridge - b_minnorm|| = "
      f"{np.linalg.norm(b_ridge - b_mn):.1e}")
