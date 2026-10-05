# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Gauss-Legendre exactness and Gauss-Hermite expectations under the normal.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

# Gauss-Legendre on [-1,1]: n nodes integrate polynomials of degree 2n-1
n = 3
x, w = np.polynomial.legendre.leggauss(n)
print(f"Gauss-Legendre n={n}: nodes {np.round(x, 6)}")
print(f"                   weights {np.round(w, 6)}")
for deg in range(2 * n + 2):
    exact = 2.0 / (deg + 1) if deg % 2 == 0 else 0.0
    err = abs(w @ x**deg - exact)
    print(f"  x^{deg}: error {err:8.1e}", "(exact)" if err < 1e-14 else "")


def gh_expect(g, mu, sigma, n):
    """E[g(X)], X ~ N(mu, sigma^2), by n-point Gauss-Hermite quadrature."""
    t, w = np.polynomial.hermite.hermgauss(n)
    return (w @ g(mu + np.sqrt(2.0) * sigma * t)) / np.sqrt(np.pi)


# Expected CRRA utility of income Y = exp(X), X ~ N(mu, s^2), gamma = 2
mu, s, gam = 7.0, 0.8, 2.0
u = lambda xx: np.exp((1 - gam) * xx) / (1 - gam)
eu = np.exp((1 - gam) * mu + (1 - gam) ** 2 * s**2 / 2) / (1 - gam)
# Random-coefficient probit: P = E_b[Phi(a + b z)], b ~ N(m, sb^2)
a0, m, sb, z = -0.5, 1.0, 0.7, 0.8
pr = stats.norm.cdf((a0 + m * z) / np.sqrt(1 + sb**2 * z**2))
g = lambda bb: stats.norm.cdf(a0 + bb * z)
print("\nGauss-Hermite relative errors")
print(" nodes   E[u(Y)] CRRA   mixed-probit P")
for nn in (1, 2, 3, 5, 10, 20):
    e1 = abs(gh_expect(u, mu, s, nn) / eu - 1)
    e2 = abs(gh_expect(g, m, sb, nn) / pr - 1)
    print(f"{nn:6d}   {e1:12.2e}   {e2:12.2e}")
print(f"exact: E[u(Y)] = {eu:.6e},  P(y=1) = {pr:.6f}")
