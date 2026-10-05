# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Moments from the MGF by differentiation; characteristic functions when MGFs fail.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)

# Poisson(lam): M(t) = exp(lam (e^t - 1)); derivatives at 0 give raw moments
lam = 3.0
M = lambda t: np.exp(lam * (np.exp(t) - 1))
h = 1e-3
m1 = (M(h) - M(-h)) / (2 * h)
m2 = (M(h) - 2 * M(0) + M(-h)) / h**2
print("Poisson(3) via central differences of the MGF")
print(f"  M'(0)  = {m1:.4f}  (exact E[X]   = {lam:.4f})")
print(f"  M''(0) = {m2:.4f}  (exact E[X^2] = {lam + lam**2:.4f})")
x = rng.poisson(lam, size=200_000)
for t in [0.2, 0.5, 1.0]:
    print(f"  t = {t:3.1f}: empirical MGF {np.mean(np.exp(t * x)):9.4f}"
          f"  exact {M(t):9.4f}")

# Cauchy: no MGF (E[e^{tX}] = inf for t != 0) but phi(t) = exp(-|t|) exists
xc = rng.standard_cauchy(size=200_000)
print("\nStandard Cauchy: empirical characteristic function")
for t in [0.5, 1.0, 2.0]:
    phi_hat = np.mean(np.exp(1j * t * xc))
    print(f"  t = {t:3.1f}: Re phi_hat = {phi_hat.real:7.4f}"
          f"  Im = {phi_hat.imag:7.4f}  exact exp(-|t|) = {np.exp(-t):.4f}")
print("  empirical MGF at t = 0.1 for growing n (does not settle):")
with np.errstate(over="ignore"):
    for n in [10**3, 10**4, 10**5, 2 * 10**5]:
        print(f"    n = {n:6d}: {np.mean(np.exp(0.1 * xc[:n])):.3e}")
