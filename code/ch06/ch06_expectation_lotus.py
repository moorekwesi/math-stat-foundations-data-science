# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Expectations computed three ways: exact sums, quadrature (LOTUS), Monte Carlo.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import integrate, stats

rng = np.random.default_rng(803)

# 1. Discrete: household size X on {1,...,8} (simulated pmf)
x = np.arange(1, 9)
p = np.array([0.08, 0.12, 0.17, 0.18, 0.16, 0.12, 0.10, 0.07])
EX = np.sum(x * p)
EX2 = np.sum(x**2 * p)                      # LOTUS with g(x) = x^2
print("Household size (discrete)")
print(f"  E[X] = {EX:.4f}  E[X^2] = {EX2:.4f}  Var(X) = {EX2 - EX**2:.4f}")
print(f"  E[1/X] = {np.sum(p / x):.4f}  versus 1/E[X] = {1 / EX:.4f}")

# 2. Continuous: lognormal income Y = exp(Z), Z ~ N(mu, s^2)
mu, s = 7.0, 0.8
fy = stats.lognorm(s=s, scale=np.exp(mu)).pdf
fz = stats.norm(mu, s).pdf
EY_exact = np.exp(mu + s**2 / 2)
# (a) definition: integrate y * f_Y(y)  (split range for accuracy)
EY_def = sum(integrate.quad(lambda y: y * fy(y), a, b)[0]
             for a, b in [(0, 1e3), (1e3, 1e4), (1e4, np.inf)])
# (b) LOTUS: integrate exp(z) * f_Z(z), no density of Y needed
EY_lotus = integrate.quad(lambda z: np.exp(z) * fz(z),
                         mu - 12 * s, mu + 12 * s)[0]
# (c) Monte Carlo with a standard error
y = np.exp(rng.normal(mu, s, size=1_000_000))
print("\nLognormal income, mu = 7, s = 0.8 (simulated units)")
print(f"  exact  E[Y] = exp(mu + s^2/2) = {EY_exact:10.3f}")
print(f"  quad   int y f_Y(y) dy        = {EY_def:10.3f}")
print(f"  LOTUS  int e^z f_Z(z) dz      = {EY_lotus:10.3f}")
print(f"  Monte Carlo (n = 1e6)         = {y.mean():10.3f}"
      f"  (s.e. {y.std(ddof=1) / np.sqrt(y.size):.3f})")
print(f"  E[log Y] = {np.log(y).mean():.4f}  log E[Y] = {np.log(EY_exact):.4f}")
