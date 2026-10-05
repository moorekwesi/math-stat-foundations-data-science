# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Four views of the matrix product and checks on special matrices.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
A = rng.integers(-3, 4, size=(3, 2)).astype(float)
B = rng.integers(-3, 4, size=(2, 4)).astype(float)
C = A @ B
# (1) entry view: c_ij = <row i of A, column j of B>
C1 = np.array([[A[i] @ B[:, j] for j in range(4)] for i in range(3)])
# (2) column view: column j of C = A times column j of B
C2 = np.column_stack([A @ B[:, j] for j in range(4)])
# (3) row view: row i of C = row i of A times B
C3 = np.vstack([A[i] @ B for i in range(3)])
# (4) outer-product view: C = sum_k a_k b_k^T (columns of A, rows of B)
C4 = sum(np.outer(A[:, k], B[k]) for k in range(2))
print("C = AB =\n", C)
print("all four views agree:",
      all(np.array_equal(C, M) for M in (C1, C2, C3, C4)))
print("rank of C (a sum of 2 outer products) =", np.linalg.matrix_rank(C))

# transpose and inverse rules
M = rng.standard_normal((4, 4))
N = rng.standard_normal((4, 4))
print("(MN)^T = N^T M^T :", np.allclose((M @ N).T, N.T @ M.T))
print("MN = NM ?        :", np.allclose(M @ N, N @ M))
print("(MN)^-1 = N^-1 M^-1:",
      np.allclose(np.linalg.inv(M @ N), np.linalg.inv(N) @ np.linalg.inv(M)))

# orthogonal matrix from a QR factorisation preserves lengths and angles
Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
v = rng.standard_normal(4)
print("Q^T Q = I        :", np.allclose(Q.T @ Q, np.eye(4)),
      "  ||Qv|| - ||v|| =", f"{np.linalg.norm(Q @ v) - np.linalg.norm(v):.1e}")
print("det(Q)           =", round(np.linalg.det(Q), 12))

# idempotent "hat" matrix P = X (X^T X)^{-1} X^T
X = np.column_stack([np.ones(6), rng.standard_normal((6, 2))])
P = X @ np.linalg.solve(X.T @ X, X.T)
print("P symmetric      :", np.allclose(P, P.T),
      "  P^2 = P :", np.allclose(P @ P, P))
print("trace(P) =", round(np.trace(P), 10), " = rank(X) =",
      np.linalg.matrix_rank(X))
print("eigenvalues of P :", np.round(np.linalg.eigvalsh(P), 10) + 0.0)
