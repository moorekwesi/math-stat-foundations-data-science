# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# Characteristic polynomial, eigenpairs, diagonalisation and matrix powers.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

np.set_printoptions(precision=4, suppress=True)
A = np.array([[4.0, 1.0],
              [2.0, 3.0]])

# Characteristic polynomial det(lambda I - A) = lambda^2 - tr(A) lambda + det(A)
coef = np.poly(A)                      # coefficients, highest power first
print("char. poly coefficients :", coef)
print("roots of polynomial     :", np.sort(np.roots(coef)))

lam, V = np.linalg.eig(A)              # columns of V are eigenvectors
order = np.argsort(lam)
lam, V = lam[order], V[:, order]
print("eigenvalues (eig)       :", lam)
print("trace = sum, det = prod :", np.trace(A), lam.sum(),
      round(np.linalg.det(A), 10), lam.prod())

# Check A v = lambda v for each pair
resid = np.linalg.norm(A @ V - V * lam, axis=0)
print("residuals ||Av - lv||   :", resid)

# Diagonalisation A = V D V^{-1} and the power A^10
D = np.diag(lam)
Vinv = np.linalg.inv(V)
print("||A - V D V^-1||        :", np.linalg.norm(A - V @ D @ Vinv))
k = 10
Ak_eig = V @ np.diag(lam ** k) @ Vinv
Ak_dir = np.linalg.matrix_power(A, k)
print("A^10 via eigen:\n", Ak_eig)
print("max |difference| from matrix_power:", np.abs(Ak_eig - Ak_dir).max())

# A defective (non-diagonalisable) matrix: Jordan block
J = np.array([[2.0, 1.0],
              [0.0, 2.0]])
lamJ, VJ = np.linalg.eig(J)
print("Jordan block eigenvalues:", lamJ)
print("eigenvector matrix rank :", np.linalg.matrix_rank(VJ, tol=1e-8))
print("cond(VJ) (huge => defective):", f"{np.linalg.cond(VJ):.2e}")
