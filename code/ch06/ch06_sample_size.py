# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Survey sample sizes for a proportion: CLT, Chebyshev, Hoeffding, union bound.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)


def n_clt(eps, delta, p=0.5):
    return np.ceil(stats.norm.ppf(1 - delta / 2)**2 * p * (1 - p) / eps**2)


def n_cheb(eps, delta, p=0.5):
    return np.ceil(p * (1 - p) / (delta * eps**2))


def n_hoef(eps, delta):
    return np.ceil(np.log(2 / delta) / (2 * eps**2))


print("Sample size for margin eps with confidence 1 - delta (worst case p=0.5)")
print(f"{'eps':>6}{'delta':>7}{'CLT':>8}{'Chebyshev':>11}{'Hoeffding':>11}")
for eps, delta in [(0.03, 0.05), (0.02, 0.05), (0.01, 0.05), (0.02, 0.01)]:
    print(f"{eps:6.2f}{delta:7.2f}{n_clt(eps, delta):8.0f}"
          f"{n_cheb(eps, delta):11.0f}{n_hoef(eps, delta):11.0f}")

# Simultaneous precision in m = 10 regions: union bound uses delta/m each
m, eps, delta, deff = 10, 0.03, 0.05, 2.0
print(f"\nm = {m} regions, eps = {eps}, overall delta = {delta}")
print(f"  per region, CLT, delta:      {n_clt(eps, delta):6.0f}")
print(f"  per region, CLT, delta/m:    {n_clt(eps, delta / m):6.0f}")
print(f"  per region, Hoeffding, d/m:  {n_hoef(eps, delta / m):6.0f}")
print(f"  x design effect {deff} (cluster sample): "
      f"{deff * n_clt(eps, delta / m):6.0f}")

# Check by simulation: simulated regional poverty rates, simple random samples
p_true = rng.uniform(0.15, 0.55, size=m)
reps = 20_000
for lab, n in [("delta", n_clt(eps, delta)), ("delta/m", n_clt(eps, delta / m))]:
    phat = rng.binomial(int(n), p_true, size=(reps, m)) / n
    miss_any = np.mean(np.any(np.abs(phat - p_true) > eps, axis=1))
    print(f"  n per region = {n:5.0f} ({lab:>7}): "
          f"P(some region misses by > eps) = {miss_any:.3f}")
