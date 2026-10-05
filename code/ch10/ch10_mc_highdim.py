# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Monte Carlo in high dimension and simulated choice probabilities.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)
n = 100_000

# Two integrands on [0,1]^d with known integral 1:
#   f1 = (3/d) sum u_j^2            (variance falls with d)
#   f2 = prod (pi/2) sin(pi u_j)    (variance grows like (pi^2/8)^d)
print("  d   f1 estimate  s.e.      f2 estimate  s.e.      Var f2 (theory)")
for d in (1, 5, 20, 50):
    u = rng.random((n, d))
    f1 = 3.0 * (u**2).mean(axis=1)
    f2 = np.prod(0.5 * np.pi * np.sin(np.pi * u), axis=1)
    v2 = (np.pi**2 / 8) ** d - 1
    print(f"{d:3d}   {f1.mean():9.5f}  {f1.std() / np.sqrt(n):.1e}"
          f"   {f2.mean():9.5f}  {f2.std() / np.sqrt(n):.1e}   {v2:10.3e}")

# Simulated choice probability: multinomial probit with 3 alternatives.
# U_j = V_j + e_j, e ~ N(0, Omega); P(choose 1) = P(U1 > U2, U1 > U3).
V = np.array([0.5, 0.2, 0.0])
Om = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
D = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])   # differences U1-U2, U1-U3
mD, SD = D @ V, D @ Om @ D.T
# P(D e > -D V) = P(W < D V) with W = -D e ~ N(0, SD)
exact = stats.multivariate_normal(mean=[0, 0], cov=SD).cdf(mD)
L = np.linalg.cholesky(Om)
print("\nmultinomial probit P(choose alternative 1)")
print(f"   bivariate normal cdf: {exact:.5f}")
for R in (100, 1000, 10000, 100000):
    e = rng.standard_normal((R, 3)) @ L.T
    U = V + e
    ind = (U.argmax(axis=1) == 0).astype(float)
    p, se = ind.mean(), ind.std(ddof=1) / np.sqrt(R)
    print(f"   frequency simulator R={R:6d}: {p:.5f}  (s.e. {se:.5f})")
