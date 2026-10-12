# DEP 803 -- Chapter 0: Mathematical Refresher
# Vectors and matrices: the hand calculations of Section 0.6 checked in NumPy.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

u = np.array([1, 2, 3])
v = np.array([4, -1, 2])
print("u + v =", u + v, "  2u - v =", 2 * u - v, "  u.v =", u @ v)

A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])
print("AB =", (A @ B).tolist(), "  BA =", (B @ A).tolist(),
      "  A^T =", A.T.tolist())     # .tolist() prints a matrix on one line

C = np.array([[2.0, 1.0], [5.0, 3.0]])
Cinv = np.linalg.inv(C)
print("det(C) =", round(np.linalg.det(C), 6))
print("inverse of C =", Cinv.round(6).tolist(),
      "  C @ Cinv =", ((C @ Cinv).round(6) + 0.0).tolist())

# market: demand q + p = 10, supply q - 2p = -2   (unknowns q, p)
M = np.array([[1.0, 1.0], [1.0, -2.0]])
rhs = np.array([10.0, -2.0])
q, p = np.linalg.solve(M, rhs)
print(f"equilibrium: q = {q:.2f}, p = {p:.2f}")

# reciprocal cost allocation between two service departments
# S1 = 10000 + 0.2 S2 ,  S2 = 8000 + 0.1 S1
K = np.array([[1.0, -0.2], [-0.1, 1.0]])
direct = np.array([10_000.0, 8_000.0])
S = np.linalg.solve(K, direct)
print("reciprocal costs S1, S2 =", S.round(2))

# NPV as a dot product of cash flows and discount factors
cash = np.array([-1000.0, 300.0, 400.0, 500.0])
disc = 1.10 ** -np.arange(4)
print("discount factors =", disc.round(4), "  NPV =", round(cash @ disc, 2))
