# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Exponential family: moments from derivatives of the log-partition function
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

# natural parameter eta, log-partition A(eta), and the mean/variance it implies
families = {
    "Bernoulli(p=0.3)": (np.log(0.3 / 0.7), lambda e: np.log1p(np.exp(e)),
                         stats.bernoulli(0.3)),
    "Poisson(lam=4)": (np.log(4.0), lambda e: np.exp(e), stats.poisson(4.0)),
    "Exponential(rate=2)": (-2.0, lambda e: -np.log(-e), stats.expon(scale=0.5)),
    "Geometric(p=0.3)*": (np.log(0.7), lambda e: -np.log1p(-np.exp(e)),
                          stats.geom(0.3, loc=-1)),
}
h = 1e-4
print(f"{'family':22s}{'eta':>8s}{'A1(eta)':>9s}{'mean':>8s}"
      f"{'A2(eta)':>9s}{'var':>8s}")
for name, (eta, A, d) in families.items():
    d1 = (A(eta + h) - A(eta - h)) / (2 * h)           # A'(eta)
    d2 = (A(eta + h) - 2 * A(eta) + A(eta - h)) / h**2  # A''(eta)
    print(f"{name:22s}{eta:8.3f}{d1:9.4f}{d.mean():8.4f}{d2:9.4f}{d.var():8.4f}")
print("* geometric counted as failures before first success (support 0,1,...)")

# sufficiency in action: the Poisson MLE depends on the data only via sum(y)
rng = np.random.default_rng(803)
y1 = rng.poisson(4.0, 50)
y2 = np.zeros(50, dtype=int)
y2[0] = y1.sum()                  # a very different sample, same total
print(f"two samples with T = sum(y) = {y1.sum()} and {y2.sum()}:")
for lam in np.linspace(3, 5, 5):
    diff = stats.poisson.logpmf(y1, lam).sum() - stats.poisson.logpmf(y2, lam).sum()
    print(f"  lam = {lam:.2f}  loglik difference = {diff:.6f}")
