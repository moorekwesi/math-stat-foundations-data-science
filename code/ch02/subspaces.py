# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Four fundamental subspaces of a rank-deficient matrix and solvability of Ax = b
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
m, n, r = 4, 5, 2
A = rng.integers(-3, 4, size=(m, r)) @ rng.integers(-3, 4, size=(r, n))
A = A.astype(float)
print("A =\n", A)

U, s, Vt = np.linalg.svd(A)
tol = max(A.shape) * np.finfo(float).eps * s[0]
rk = int(np.sum(s > tol))
col_basis = U[:, :rk]          # basis of col(A)   (subspace of R^m)
lnull_basis = U[:, rk:]        # basis of null(A^T)
row_basis = Vt[:rk].T          # basis of col(A^T) (subspace of R^n)
null_basis = Vt[rk:].T         # basis of null(A)

print(f"\nsingular values: {np.round(s, 4)}")
print(f"rank r = {rk}, m = {m}, n = {n}")
print(f"dim col(A)   = {col_basis.shape[1]}   dim null(A^T) = {lnull_basis.shape[1]}"
      f"   (sum = m = {m})")
print(f"dim row(A)   = {row_basis.shape[1]}   dim null(A)   = {null_basis.shape[1]}"
      f"   (sum = n = {n})")
print(f"max |A N|          = {np.abs(A @ null_basis).max():.2e}")
print(f"max |row^T null|   = {np.abs(row_basis.T @ null_basis).max():.2e}")
print(f"max |col^T lnull|  = {np.abs(col_basis.T @ lnull_basis).max():.2e}")


def classify(A, b):
    """Existence / uniqueness of solutions of Ax = b via ranks."""
    rA = np.linalg.matrix_rank(A)
    rAb = np.linalg.matrix_rank(np.column_stack([A, b]))
    if rAb > rA:
        return "no solution (b not in col(A))"
    if rA == A.shape[1]:
        return "unique solution"
    return f"infinitely many solutions ({A.shape[1] - rA} free parameters)"


b_in = A @ rng.normal(size=n)              # consistent right-hand side
b_out = b_in + lnull_basis[:, 0]           # add a left-null-space component
print("\nb in col(A)    :", classify(A, b_in))
print("b not in col(A):", classify(A, b_out))
X_full = rng.normal(size=(6, 3))
print("tall full-rank :", classify(X_full, X_full @ np.ones(3)))

# Any solution = particular solution + element of null(A)
x_p = np.linalg.lstsq(A, b_in, rcond=None)[0]
x_alt = x_p + null_basis @ rng.normal(size=n - rk)
print(f"\n||A x_p - b||   = {np.linalg.norm(A @ x_p - b_in):.2e}")
print(f"||A x_alt - b|| = {np.linalg.norm(A @ x_alt - b_in):.2e}")
print(f"||x_p|| = {np.linalg.norm(x_p):.4f} <= ||x_alt|| = "
      f"{np.linalg.norm(x_alt):.4f}")
