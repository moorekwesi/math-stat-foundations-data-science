# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Law of iterated expectations and law of total variance (within/between).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
# Simulated household survey: region shares and region-specific log incomes
regions = ["North", "Middle", "Coast"]
share = np.array([0.25, 0.45, 0.30])
mu = np.array([7.2, 7.8, 8.1])          # mean of log income by region
sig = np.array([0.7, 0.6, 0.8])
n = 200_000
r = rng.choice(3, size=n, p=share)
y = rng.normal(mu[r], sig[r])            # log income

cond_mean = np.array([y[r == k].mean() for k in range(3)])
cond_var = np.array([y[r == k].var() for k in range(3)])
p_hat = np.bincount(r) / n
print(f"{'region':>8}{'P(R=r)':>9}{'E[Y|R]':>9}{'Var(Y|R)':>10}")
for k in range(3):
    print(f"{regions[k]:>8}{p_hat[k]:9.3f}{cond_mean[k]:9.3f}{cond_var[k]:10.3f}")

lie = p_hat @ cond_mean
within = p_hat @ cond_var
between = p_hat @ (cond_mean - lie)**2
print(f"\nE[Y] directly           = {y.mean():.4f}")
print(f"E[E[Y|R]] (LIE)         = {lie:.4f}")
print(f"Var(Y) directly         = {y.var():.4f}")
print(f"E[Var(Y|R)] (within)    = {within:.4f}")
print(f"Var(E[Y|R]) (between)   = {between:.4f}")
print(f"within + between        = {within + between:.4f}")
print(f"share explained by region (R^2 of CEF) = {between / y.var():.3f}")
