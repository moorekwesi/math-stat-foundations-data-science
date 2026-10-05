# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Simulated household incomes: log-normal fit, Lorenz curve, Gini, poverty headcount
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
n = 20_000                                  # simulated survey households
mu, sigma = np.log(12_000), 0.8             # parameters of log income
income = rng.lognormal(mu, sigma, n)        # annual income, simulated units


def gini(y):
    """Sample Gini coefficient via the sorted-rank formula."""
    y = np.sort(y)
    k = np.arange(1, len(y) + 1)
    return 2 * (k * y).sum() / (len(y) * y.sum()) - (len(y) + 1) / len(y)


# maximum likelihood fit of a log-normal (location fixed at zero)
s_hat, _, scale_hat = stats.lognorm.fit(income, floc=0)
print(f"true   mu = {mu:.4f}, sigma = {sigma:.4f}")
print(f"fitted mu = {np.log(scale_hat):.4f}, sigma = {s_hat:.4f}")
print(f"mean   = {income.mean():8.1f}  (theory {np.exp(mu + sigma**2 / 2):8.1f})")
print(f"median = {np.median(income):8.1f}  (theory {np.exp(mu):8.1f})")

g_theory = 2 * stats.norm.cdf(sigma / np.sqrt(2)) - 1
print(f"Gini: sample = {gini(income):.4f}, log-normal formula = {g_theory:.4f}")

z = 6_000                                   # an illustrative poverty line
h_model = stats.norm.cdf((np.log(z) - np.log(scale_hat)) / s_hat)
print(f"poverty headcount at z = {z}: sample = {(income < z).mean():.4f}, "
      f"model = {h_model:.4f}")
gap = np.mean(np.maximum(z - income, 0) / z)
print(f"poverty gap index (FGT1) = {gap:.4f}")

# Lorenz curve: cumulative share of income held by the poorest fraction p
y = np.sort(income)
p = np.arange(1, n + 1) / n
L = np.cumsum(y) / y.sum()
pp = np.linspace(0, 1, 200)
L_theory = stats.norm.cdf(stats.norm.ppf(pp) - sigma)
print(f"bottom 40% share = {L[int(0.4 * n) - 1]:.4f}, "
      f"top 10% share = {1 - L[int(0.9 * n) - 1]:.4f}")

fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect equality")
ax.plot(p, L, color="#1f4e79", lw=2.5, label="simulated sample")
ax.plot(pp, L_theory, color="#cc9900", lw=1.5, ls=":", label="log-normal theory")
ax.fill_between(p, L, p, color="#1f4e79", alpha=0.12)
ax.set_xlabel("cumulative share of households")
ax.set_ylabel("cumulative share of income")
ax.legend(frameon=False, fontsize=8)
plt.savefig("ch05_lorenz.pdf", bbox_inches="tight")
