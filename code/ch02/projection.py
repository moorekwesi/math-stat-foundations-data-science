# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Hat matrix P and annihilator M: symmetry, idempotency, trace and Pythagoras
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n, k = 60, 4
X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
y = X @ np.array([2.0, 1.0, -0.5, 0.0]) + rng.normal(size=n)

P = X @ np.linalg.solve(X.T @ X, X.T)     # hat (projection) matrix
M = np.eye(n) - P                         # annihilator (residual maker)

print("Property                      P            M")
print(f"symmetric  max|A - A^T|  {np.abs(P - P.T).max():9.1e}  "
      f"{np.abs(M - M.T).max():9.1e}")
print(f"idempotent max|A^2 - A|  {np.abs(P @ P - P).max():9.1e}  "
      f"{np.abs(M @ M - M).max():9.1e}")
print(f"trace                    {np.trace(P):9.4f}  {np.trace(M):9.4f}")
print(f"rank                     {np.linalg.matrix_rank(P):9d}  "
      f"{np.linalg.matrix_rank(M):9d}")
ev = np.linalg.eigvalsh(P)
print(f"eigenvalues of P: {np.sum(np.isclose(ev, 1))} ones, "
      f"{np.sum(np.isclose(ev, 0, atol=1e-10))} zeros")
print(f"max|PM| = {np.abs(P @ M).max():.1e},  max|MX| = {np.abs(M @ X).max():.1e}")

yhat, e = P @ y, M @ y
print(f"\nyhat' e = {yhat @ e: .2e}   (orthogonal pieces)")
print(f"||y||^2 = {y @ y:.4f}   ||yhat||^2 + ||e||^2 = {yhat @ yhat + e @ e:.4f}")

# ANOVA decomposition about the mean (X contains a constant)
ybar = y.mean()
TSS = np.sum((y - ybar) ** 2)
ESS = np.sum((yhat - ybar) ** 2)
RSS = e @ e
print(f"TSS = {TSS:.4f}  ESS + RSS = {ESS + RSS:.4f}  R^2 = {ESS / TSS:.4f}")

# Projection onto a single vector a: P_a = a a' / a'a
a = X[:, 1]
Pa = np.outer(a, a) / (a @ a)
print(f"\nrank-one projector: trace = {np.trace(Pa):.4f}, "
      f"idempotent err = {np.abs(Pa @ Pa - Pa).max():.1e}")
# Oblique (non-orthogonal) projector: idempotent but not symmetric
Q = np.array([[1.0, 1.0], [0.0, 0.0]])
print(f"oblique Q: Q^2 = Q? {np.allclose(Q @ Q, Q)}, symmetric? "
      f"{np.allclose(Q, Q.T)}")
