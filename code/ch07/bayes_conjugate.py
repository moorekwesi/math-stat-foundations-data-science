# DEP 803 -- Chapter 7: Estimation Principles
# Conjugate Bayesian updating: beta-binomial, gamma-Poisson, normal-normal.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)


def report(name, post, mle, se, mode):
    lo, hi = post.ppf([0.025, 0.975])
    print(f"{name:<22} {post.mean():8.4f} {mode:8.4f} [{lo:.4f}, {hi:.4f}]"
          f"  MLE {mle:.4f} +/- {1.96 * se:.4f}")


print(f"{'model (simulated)':<22} {'post mean':>8} {'MAP':>8}"
      f" {'95% credible interval':>20}")
# 1. Beta-binomial: programme take-up in a pilot of 40 households
a0, b0, n = 2.0, 8.0, 40                # prior mean 0.2 from earlier rounds
y = rng.binomial(n, 0.35)
post = stats.beta(a0 + y, b0 + n - y)
mode = (a0 + y - 1) / (a0 + b0 + n - 2)
ph = y / n
report(f"take-up, y={y}/{n}", post, ph, np.sqrt(ph * (1 - ph) / n), mode)
jeff = stats.beta(0.5 + y, 0.5 + n - y)
report("  same, Jeffreys prior", jeff, ph, np.sqrt(ph * (1 - ph) / n),
       (y - 0.5) / (n - 1))

# 2. Gamma-Poisson: weekly malaria cases at a clinic, 12 weeks
a0, r0, m = 20.0, 2.0, 12               # prior mean 10 cases per week
cases = rng.poisson(13.0, m)
post = stats.gamma(a0 + cases.sum(), scale=1 / (r0 + m))
lam = cases.mean()
report("cases/week, 12 weeks", post, lam, np.sqrt(lam / m),
       (a0 + cases.sum() - 1) / (r0 + m))

# 3. Normal-normal: mean log income, known sigma = 1.2
mu0, tau0, sig, k = 5.0, 0.5, 1.2, 30
x = rng.normal(5.6, sig, k)
prec = 1 / tau0**2 + k / sig**2
mpost = (mu0 / tau0**2 + x.sum() / sig**2) / prec
post = stats.norm(mpost, 1 / np.sqrt(prec))
report("mean log income, k=30", post, x.mean(), sig / np.sqrt(k), mpost)
w = (k / sig**2) / prec
print(f"  shrinkage weight on the data = {w:.3f}")

# Bernstein-von Mises: posterior sd / MLE se -> 1 as n grows (beta-binomial)
print("\nBeta(2, 8) prior, true p = 0.35:  n, posterior sd / Wald se,"
      " |post mean - MLE|")
for n in (10, 40, 200, 1000, 5000):
    y = rng.binomial(n, 0.35)
    post = stats.beta(2 + y, 8 + n - y)
    ph = y / n
    se = np.sqrt(ph * (1 - ph) / n)
    print(f"  n = {n:>5}: {post.std() / se:.4f}, {abs(post.mean() - ph):.4f}")
