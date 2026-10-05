# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Poisson limit of the binomial and Bortkiewicz's horse-kick data
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

lam = 2.0
k = np.arange(0, 60)
print("Total variation distance between Bin(n, lam/n) and Poisson(lam):")
for n in [5, 10, 50, 100, 1000, 10000]:
    diff = stats.binom.pmf(k, n, lam / n) - stats.poisson.pmf(k, lam)
    tv = 0.5 * np.abs(diff).sum()
    print(f"  n = {n:6d}   TV = {tv:.5f}   bound lam^2/n = {lam**2 / n:.5f}")

# Bortkiewicz (1898): deaths by horse kick, 10 Prussian corps x 20 years
deaths = np.array([0, 1, 2, 3, 4])
corps_years = np.array([109, 65, 22, 3, 1])
N = corps_years.sum()
mean = (deaths * corps_years).sum() / N
var = ((deaths - mean) ** 2 * corps_years).sum() / (N - 1)
expected = N * stats.poisson.pmf(deaths, mean)
print(f"\nHorse kicks: N = {N}, mean = {mean:.3f}, variance = {var:.3f}")
print(" deaths  observed  expected(Poisson)")
for d, o, e in zip(deaths, corps_years, expected):
    print(f"{d:7d}{o:10d}{e:12.1f}")
