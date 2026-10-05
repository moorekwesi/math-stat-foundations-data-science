# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Convergence in probability of the sample mean: Monte Carlo versus Chebyshev.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
reps, eps = 20_000, 0.1
mu, var = 1.0, 1.0                     # Exponential(1)
print(f"P(|Xbar_n - mu| > {eps}) for Exponential(1), {reps} replications")
print(f"{'n':>6}{'Monte Carlo':>13}{'Chebyshev':>11}{'MSE':>10}{'var/n':>9}")
for n in [10, 50, 100, 500, 1000, 2000]:
    # the sum of n Exponential(1) draws is Gamma(n, 1): sample it directly
    xbar = rng.gamma(n, 1.0, size=reps) / n
    p = np.mean(np.abs(xbar - mu) > eps)
    cheb = min(1.0, var / (n * eps**2))
    mse = np.mean((xbar - mu)**2)
    print(f"{n:6d}{p:13.4f}{cheb:11.4f}{mse:10.5f}{var / n:9.5f}")
