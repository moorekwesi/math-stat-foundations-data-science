# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Adaptive quadrature with scipy.integrate.quad and dblquad.
# (c) 2026 Stephen E. Moore. MIT Licence.
import warnings
import numpy as np
from scipy import integrate, stats

# Expected shortfall E[max(c - Y, 0)] of income Y below a line c,
# Y lognormal: infinite range, kinked integrand
mu, s, c = 7.0, 0.8, 900.0
pdf = lambda y: stats.lognorm.pdf(y, s, scale=np.exp(mu))
out = integrate.quad(lambda y: max(c - y, 0.0) * pdf(y), 0, np.inf,
                     full_output=True)
print(f"shortfall, quad on [0,inf): {out[0]:.6f}  est.err {out[1]:.1e}"
      f"  evals {out[2]['neval']}")
val2, err2 = integrate.quad(lambda y: (c - y) * pdf(y), 0, c)
d1 = (np.log(c) - mu) / s
closed = c * stats.norm.cdf(d1) - np.exp(mu + s**2 / 2) * stats.norm.cdf(d1 - s)
print(f"shortfall, quad on [0,c]  : {val2:.6f}  closed form {closed:.6f}")

# Bivariate normal probability with dblquad versus scipy's mvn cdf
rho, a1, a2 = 0.6, 0.5, -0.2
mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1.0, rho], [rho, 1.0]])
p_dbl, e_dbl = integrate.dblquad(lambda y, x: mvn.pdf([x, y]),
                                 -8, a1, lambda x: -8, lambda x: a2)
print(f"P(X1<{a1}, X2<{a2}), rho={rho}:")
print(f"   dblquad {p_dbl:.8f} (est.err {e_dbl:.1e}),"
      f" mvn.cdf {mvn.cdf([a1, a2]):.8f}")

# Oscillatory integrand: too few subintervals triggers a warning
f = lambda x: np.cos(200 * x) * np.exp(-x)
exact = (1 + 200 * np.exp(-10) * np.sin(2000) - np.exp(-10) * np.cos(2000))
exact /= 1 + 200**2
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    v1 = integrate.quad(f, 0, 10, limit=20)[0]
v2 = integrate.quad(f, 0, 10, limit=500)[0]
print(f"oscillatory, limit=20 : {v1: .10e}  warnings: {len(caught)}")
print(f"oscillatory, limit=500: {v2: .10e}")
print(f"exact                 : {exact: .10e}")
