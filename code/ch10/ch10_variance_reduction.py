# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Antithetic variates and control variates for a rainfall index-insurance payout.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)

# Warm-up: theta = E[exp(U)], U ~ U(0,1), theta = e - 1
n = 10_000
u = rng.random(n)
plain = np.exp(u)
anti = 0.5 * (np.exp(u[: n // 2]) + np.exp(1 - u[: n // 2]))   # n/2 pairs
c = -np.cov(plain, u)[0, 1] / np.var(u, ddof=1)
cv = plain + c * (u - 0.5)
print("E[exp(U)] = e - 1 =", f"{np.e - 1:.6f}")
print(f"  plain      {plain.mean():.6f}  var/eval {plain.var():.5f}")
print(f"  antithetic {anti.mean():.6f}  var/eval {anti.var() / 2:.5f}")
print(f"  control    {cv.mean():.6f}  var/eval {cv.var():.5f}  c* = {c:.4f}")

# Index insurance: seasonal rainfall R = exp(m + s Z); payout max(K - R, 0)
m, s, K = np.log(600.0), 0.3, 500.0
d = (np.log(K) - m) / s
exact = K * stats.norm.cdf(d) - np.exp(m + s**2 / 2) * stats.norm.cdf(d - s)
z = rng.standard_normal(n)
pay = lambda zz: np.maximum(K - np.exp(m + s * zz), 0.0)
y = pay(z)
ya = 0.5 * (pay(z[: n // 2]) + pay(-z[: n // 2]))
# control variate: rainfall R itself, whose mean is known exactly
r_ = np.exp(m + s * z)
c = -np.cov(y, r_)[0, 1] / np.var(r_, ddof=1)
ycv = y + c * (r_ - np.exp(m + s**2 / 2))
rho = np.corrcoef(y, r_)[0, 1]
print(f"\nexpected payout, exact: {exact:.4f}")
print("method        estimate    s.e.     var/eval   efficiency gain")
v0 = y.var()
for name, est, se, v in [
        ("plain", y.mean(), y.std() / np.sqrt(n), v0),
        ("antithetic", ya.mean(), ya.std() / np.sqrt(n // 2), ya.var() / 2),
        ("control var", ycv.mean(), ycv.std() / np.sqrt(n), ycv.var())]:
    print(f"{name:12s}  {est:9.4f}  {se:7.4f}  {v:10.2f}   {v0 / v:6.2f}")
print(f"corr(payout, rainfall) = {rho:.3f}; theory gain 1/(1-rho^2) ="
      f" {1 / (1 - rho**2):.2f}")
