# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Partitioned inverse via the Schur complement; trace and determinant rules.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
Z = rng.standard_normal((50, 5))
M = Z.T @ Z / 50 + 0.5 * np.eye(5)         # symmetric positive definite
k = 2                                      # partition sizes 2 and 3
A, B = M[:k, :k], M[:k, k:]
C, D = M[k:, :k], M[k:, k:]
Ai = np.linalg.inv(A)
S = D - C @ Ai @ B                         # Schur complement of A in M
Si = np.linalg.inv(S)
Minv = np.block([[Ai + Ai @ B @ Si @ C @ Ai, -Ai @ B @ Si],
                 [-Si @ C @ Ai,              Si]])
print("block formula equals inv(M):", np.allclose(Minv, np.linalg.inv(M)))
print("max abs difference        : %.2e" % np.abs(Minv - np.linalg.inv(M)).max())
print("det(M) = %.6f,  det(A) det(S) = %.6f"
      % (np.linalg.det(M), np.linalg.det(A) * np.linalg.det(S)))
print("lower-right block of inv(M) equals inv(S):",
      np.allclose(np.linalg.inv(M)[k:, k:], Si))

# trace rules
P = rng.standard_normal((3, 5))
Q = rng.standard_normal((5, 3))
print("tr(PQ) = %.6f, tr(QP) = %.6f" % (np.trace(P @ Q), np.trace(Q @ P)))
ev = np.linalg.eigvals(M)
print("tr(M) = %.6f = sum of eigenvalues %.6f" % (np.trace(M), ev.sum().real))
print("det(M) = %.6f = prod of eigenvalues %.6f" % (np.linalg.det(M),
                                                    ev.prod().real))
# determinant rules
N = rng.standard_normal((5, 5))
print("det(MN) - det(M)det(N) = %.1e"
      % (np.linalg.det(M @ N) - np.linalg.det(M) * np.linalg.det(N)))
print("det(3M) / det(M) = %.1f  (= 3^5 = %d)"
      % (np.linalg.det(3 * M) / np.linalg.det(M), 3**5))
sign, logdet = np.linalg.slogdet(1000 * np.eye(200))
print("det(1000 I_200) overflows:", np.linalg.det(1000 * np.eye(200)),
      "  slogdet gives log|det| =", round(logdet, 3))
