# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Order statistics: minimum and maximum of n iid variables.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)
reps = 100_000

# Minimum of n iid Exp(rate lam) is Exp(rate n lam): first of n loans to default
lam = 0.1                                    # default hazard per month
print("Months until the FIRST default among n loans, each Exp(rate 0.1)")
print(f"{'n':>4} {'E[min] sim':>11} {'1/(n lam)':>10}")
for n in [1, 5, 20, 100]:
    m = rng.exponential(1 / lam, (reps, n)).min(axis=1)
    print(f"{n:>4} {m.mean():11.3f} {1 / (n * lam):10.3f}")

# Maximum of n iid U(0,1): CDF x^n, mean n/(n+1)
print("\nMaximum of n iid Uniform(0,1): P(max <= 0.9) = 0.9^n")
print(f"{'n':>4} {'sim':>8} {'exact':>8} {'E[max] sim':>11} {'n/(n+1)':>8}")
for n in [2, 10, 50]:
    M = rng.random((reps, n)).max(axis=1)
    print(f"{n:>4} {np.mean(M <= 0.9):8.4f} {0.9**n:8.4f} {M.mean():11.4f}"
          f" {n / (n + 1):8.4f}")

# Worst rainfall shortfall over a 30-year horizon (annual maxima, N(0,1))
n = 30
Mx = rng.standard_normal((reps, n)).max(axis=1)
print(f"\nMax of 30 iid N(0,1): mean {Mx.mean():.3f}, "
      f"P(max > 2) = {np.mean(Mx > 2):.3f}  (exact 1 - Phi(2)^30 = "
      f"{1 - stats.norm.cdf(2) ** 30:.3f})")
