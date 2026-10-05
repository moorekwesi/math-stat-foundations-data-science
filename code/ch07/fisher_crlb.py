# DEP 803 -- Chapter 7: Estimation Principles
# Cramer-Rao bound check and observed vs expected information (Cauchy location).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import optimize

rng = np.random.default_rng(803)
R = 20000

# Part A: Poisson mean and exponential rate versus the Cramer-Rao bound
lam, n = 2.0, 50
xbar = rng.poisson(lam, size=(R, n)).mean(axis=1)
print(f"Part A: R = {R} samples of size n = {n}")
print(f"  Poisson mean    : Var(xbar) = {xbar.var():.5f},"
      f" CRLB lam/n = {lam / n:.5f}")
rate = 0.5
rhat = 1 / rng.exponential(1 / rate, size=(R, n)).mean(axis=1)
print(f"  Exponential rate: Var(1/xbar) = {rhat.var():.6f},"
      f" CRLB rate^2/n = {rate**2 / n:.6f}")
print(f"                    bias = {rhat.mean() - rate:.4f}"
      f" (theory rate/(n-1) = {rate / (n - 1):.4f})")

# Part B: Cauchy location; expected information n/2, observed varies
n, R2 = 20, 4000


def mle_cauchy(x):
    nll = lambda t: np.sum(np.log1p((x - t) ** 2))
    return optimize.minimize_scalar(nll, bracket=(np.median(x) - 1,
                                                  np.median(x) + 1)).x


def obs_info(x, t):
    d = x - t
    return np.sum(2 * (1 - d**2) / (1 + d**2) ** 2)


est, J = np.empty(R2), np.empty(R2)
for r in range(R2):
    x = rng.standard_cauchy(n)
    est[r] = mle_cauchy(x)
    J[r] = obs_info(x, est[r])
z_exp = est / np.sqrt(2 / n)                 # standardise by expected info
z_obs = est * np.sqrt(J)                     # standardise by observed info
print(f"\nPart B: Cauchy location, n = {n}, R = {R2}")
print(f"  Var(MLE) = {est.var():.4f}; 1/I(theta) = 2/n = {2 / n:.4f}")
print(f"  observed information: min {J.min():.2f}, median"
      f" {np.median(J):.2f}, max {J.max():.2f} (expected {n / 2:.1f})")
print("  coverage of 95% Wald intervals:")
print(f"    expected information: {np.mean(np.abs(z_exp) < 1.96):.3f}")
print(f"    observed information: {np.mean(np.abs(z_obs) < 1.96):.3f}")
# conditional behaviour: split by observed information
lo = J < np.median(J)
print("  Var(MLE | J below median) = %.4f, mean 1/J = %.4f"
      % (est[lo].var(), np.mean(1 / J[lo])))
print("  Var(MLE | J above median) = %.4f, mean 1/J = %.4f"
      % (est[~lo].var(), np.mean(1 / J[~lo])))
