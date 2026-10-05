# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# LU for repeated solves (input-output model) and Cholesky for simulation
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import linalg

rng = np.random.default_rng(803)

# --- LU: a small Leontief input-output model (illustrative coefficients) ---
sectors = ["agric", "industry", "services"]
A = np.array([[0.15, 0.10, 0.05],      # input of row sector per unit
              [0.20, 0.30, 0.15],      # of column sector's output
              [0.10, 0.20, 0.25]])
L_mat = np.eye(3) - A
P, L, U = linalg.lu(L_mat)
print("LU of (I - A):  max|PLU - (I - A)| =",
      f"{np.abs(P @ L @ U - L_mat).max():.1e}")
lu_piv = linalg.lu_factor(L_mat)       # factor once ...
demands = np.array([[100, 80, 120],    # ... solve for many scenarios
                    [110, 80, 120],
                    [100, 95, 130]], dtype=float)
for d in demands:
    x = linalg.lu_solve(lu_piv, d)
    print(f"final demand {d} -> gross output {np.round(x, 1)}")

# --- Cholesky: simulate correlated (income, consumption, savings) shocks ---
sd = np.array([1.0, 0.8, 0.5])
R = np.array([[1.0, 0.7, 0.4],
              [0.7, 1.0, -0.2],
              [0.4, -0.2, 1.0]])
Sigma = np.outer(sd, sd) * R
Lc = np.linalg.cholesky(Sigma)         # Sigma = Lc Lc^T, Lc lower triangular
print("\nCholesky factor L:\n", np.round(Lc, 4))
Z = rng.standard_normal(size=(100_000, 3))
Xsim = Z @ Lc.T                        # rows are draws from N(0, Sigma)
print("target correlation (upper):", np.round(R[np.triu_indices(3, 1)], 3))
print("sample correlation (upper):",
      np.round(np.corrcoef(Xsim.T)[np.triu_indices(3, 1)], 3))

# Cholesky is a cheap positive-definiteness test
bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
print("\neigenvalues of 'bad' correlation matrix:",
      np.round(np.linalg.eigvalsh(bad), 3))
try:
    np.linalg.cholesky(bad)
except np.linalg.LinAlgError as err:
    print("np.linalg.cholesky(bad) fails:", err)

# Solving an SPD system with Cholesky (as in the normal equations)
b = rng.normal(size=3)
c_and_low = linalg.cho_factor(Sigma)
x = linalg.cho_solve(c_and_low, b)
print(f"\nSPD solve residual ||Sigma x - b|| = {np.linalg.norm(Sigma @ x - b):.1e}")
