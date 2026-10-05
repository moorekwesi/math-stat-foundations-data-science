# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# How many bootstrap replications? Monte Carlo error of se_B and quantiles.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n = 100
x = rng.lognormal(7.0, 0.9, size=n)          # one fixed simulated sample

# Ideal bootstrap (B -> infinity) approximated with a very large B
big = np.median(x[rng.integers(0, n, size=(200_000, n))], axis=1)
se_inf, q_inf = big.std(ddof=1), np.quantile(big, 0.975)
print(f"median = {np.median(x):.1f}; ideal bootstrap se = {se_inf:.2f}, "
      f"97.5% quantile = {q_inf:.1f}")
print("Variability of the bootstrap output over 300 repeated bootstraps")
print(f"{'B':>6}{'mean se_B':>11}{'sd se_B':>9}{'CV(se_B)':>10}"
      f"{'sd q975':>9}")
for B in (25, 50, 100, 200, 500, 1000, 2000):
    se_B, q_B = np.empty(300), np.empty(300)
    for r in range(300):
        t = np.median(x[rng.integers(0, n, size=(B, n))], axis=1)
        se_B[r], q_B[r] = t.std(ddof=1), np.quantile(t, 0.975)
    print(f"{B:6d}{se_B.mean():11.2f}{se_B.std():9.2f}"
          f"{se_B.std() / se_B.mean():10.3f}{q_B.std():9.2f}")
