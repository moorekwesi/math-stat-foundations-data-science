# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# A three-sector Leontief input-output model with illustrative coefficients.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

sectors = ["agriculture", "industry", "services"]
# technical coefficients a_ij: input from sector i per unit output of sector j
# (hypothetical numbers for teaching, not an official input-output table)
A = np.array([[0.15, 0.10, 0.05],
              [0.20, 0.30, 0.15],
              [0.10, 0.20, 0.25]])
f = np.array([400.0, 600.0, 900.0])      # final demand (GHS million)
I = np.eye(3)
L = np.linalg.inv(I - A)                 # Leontief inverse
x = np.linalg.solve(I - A, f)            # gross output: (I - A) x = f
print("column sums of A:", A.sum(axis=0), "(all < 1: productive economy)")
print("Leontief inverse L = (I - A)^-1 =\n", L.round(4))
print("gross output x =", x.round(2))
print("check x - Ax = f:", np.allclose(x - A @ x, f))
mult = L.sum(axis=0)                     # output multipliers
for s, m in zip(sectors, mult):
    print(f"output multiplier of {s:12s} = {m:.4f}")

# policy shock: +100 in final demand for services (e.g. public health spending)
df = np.array([0.0, 0.0, 100.0])
dx = L @ df
print("change in output by sector:", dx.round(2), " total =", dx.sum().round(2))

# employment: labour coefficients (jobs per GHS million of output, hypothetical)
ell = np.array([12.0, 4.0, 7.0])
print("jobs created by the shock = ell^T L df =", round(ell @ dx, 1))

# Neumann series: L = I + A + A^2 + ... (converges because rho(A) < 1)
rho = max(abs(np.linalg.eigvals(A)))
S, term = I.copy(), I.copy()
for k in range(1, 31):
    term = term @ A
    S += term
print(f"spectral radius of A = {rho:.4f}")
print("max |L - sum_{k<=30} A^k| = %.2e" % np.abs(L - S).max())
