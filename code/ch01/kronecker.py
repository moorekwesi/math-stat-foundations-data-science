# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# The Kronecker product, the vec operator and panel-data dummy matrices.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])
print("A kron B =\n", np.kron(A, B))

# mixed-product property (A kron B)(C kron D) = (AC) kron (BD)
A, C = rng.standard_normal((2, 3, 3))
B, D = rng.standard_normal((2, 2, 2))
print("mixed product rule  :",
      np.allclose(np.kron(A, B) @ np.kron(C, D), np.kron(A @ C, B @ D)))
print("inverse rule        :", np.allclose(np.linalg.inv(np.kron(A, B)),
      np.kron(np.linalg.inv(A), np.linalg.inv(B))))


def vec(M):
    """Stack the columns of M into one long vector (column-major order)."""
    return M.reshape(-1, order="F")


# vec(A X B) = (B^T kron A) vec(X)
X = rng.standard_normal((3, 2))
print("vec(AXB) identity   :", np.allclose(vec(A @ X @ B), np.kron(B.T, A) @ vec(X)))

# panel data: N = 3 districts observed for T = 4 years, stacked by district
N, T = 3, 4
Dunit = np.kron(np.eye(N), np.ones((T, 1)))     # district dummies (NT x N)
Dtime = np.kron(np.ones((N, 1)), np.eye(T))     # year dummies (NT x T)
print("district dummies D = I_N kron 1_T, shape", Dunit.shape)
print(Dunit.astype(int).T)
print("year dummies 1_N kron I_T, shape", Dtime.shape)
y = rng.normal(10, 2, N * T)
Q = np.eye(N * T) - Dunit @ np.linalg.solve(Dunit.T @ Dunit, Dunit.T)
print("within transform Q y removes district means:",
      np.allclose((Q @ y).reshape(N, T).mean(axis=1), 0))
