# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Many hypotheses: family-wise error, the union bound and the max of m means.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)
alpha, n, reps = 0.05, 400, 2000
print("m independent true-null z-tests on Bernoulli(0.5) means, n = 400")
print(f"{'m':>5}{'FWER':>8}{'1-(1-a)^m':>11}{'m*a':>7}{'FWER Bonf':>11}"
      f"{'E max|dev|':>12}{'Hoeffding':>11}")
for m in [1, 5, 20, 100, 500]:
    phat = rng.binomial(n, 0.5, size=(reps, m)) / n
    z = (phat - 0.5) / np.sqrt(0.25 / n)
    fwer = np.mean(np.any(np.abs(z) > 1.96, axis=1))
    crit = stats.norm.ppf(1 - alpha / (2 * m))          # Bonferroni cut-off
    bonf = np.mean(np.any(np.abs(z) > crit, axis=1))
    maxdev = np.mean(np.max(np.abs(phat - 0.5), axis=1))
    hoef = np.sqrt(np.log(2 * m / alpha) / (2 * n))      # 95% bound on max
    print(f"{m:5d}{fwer:8.3f}{1 - (1 - alpha)**m:11.3f}{min(1, m * alpha):7.2f}"
          f"{bonf:11.3f}{maxdev:12.4f}{hoef:11.4f}")
print("Hoeffding column: max deviation bound holding with prob >= 0.95")
