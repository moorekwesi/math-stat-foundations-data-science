# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# How tight are Markov's and Chebyshev's inequalities? Exact tails versus bounds.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

# Markov: P(X >= a) <= E[X]/a for X >= 0. Exponential(1) has E[X] = 1.
print("Markov, X ~ Exponential(1)")
print(f"{'a':>5}{'exact':>12}{'bound 1/a':>12}")
for a in [1, 2, 4, 8]:
    print(f"{a:5d}{stats.expon.sf(a):12.5f}{1 / a:12.5f}")

# Chebyshev: P(|X - mu| >= k sigma) <= 1/k^2
dists = {"Normal": stats.norm(),
         "Uniform": stats.uniform(-np.sqrt(3), 2 * np.sqrt(3)),
         "Laplace": stats.laplace(scale=1 / np.sqrt(2)),
         "t(3)": stats.t(3, scale=1 / np.sqrt(3))}
print("\nChebyshev: P(|X - mu| >= k sd) for unit-variance laws")
print(f"{'k':>5}" + "".join(f"{d:>10}" for d in dists) + f"{'1/k^2':>9}")
for k in [1.5, 2, 3, 4]:
    row = "".join(f"{2 * D.sf(k):10.5f}" for D in dists.values())
    print(f"{k:5.1f}{row}{1 / k**2:9.4f}")

# Chebyshev is attained by a three-point law: +-k w.p. 1/(2k^2), 0 otherwise
k = 2.0
vals = np.array([-k, 0.0, k])
pr = np.array([1 / (2 * k**2), 1 - 1 / k**2, 1 / (2 * k**2)])
print(f"\nthree-point law: Var = {pr @ vals**2:.3f},"
      f" P(|X| >= {k}) = {pr[0] + pr[2]:.3f} = 1/k^2")
