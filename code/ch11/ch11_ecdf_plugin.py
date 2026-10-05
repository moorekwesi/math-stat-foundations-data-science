# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# The empirical distribution function, the DKW band and plug-in estimators.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)


def gini(x):
    """Plug-in Gini coefficient of the empirical distribution of x."""
    x = np.sort(x)
    n = x.size
    i = np.arange(1, n + 1)
    return np.sum((2 * i - n - 1) * x) / (n * np.sum(x))


def fgt(x, z, alpha):
    """Foster-Greer-Thorbecke poverty index P_alpha with poverty line z."""
    gap = np.clip((z - x) / z, 0, None)
    return np.mean(gap**alpha * (x < z))


mu, sigma, n = 7.0, 0.9, 200
x = rng.lognormal(mu, sigma, size=n)     # simulated household incomes
z = 600.0                                # illustrative poverty line
pop = stats.lognorm(s=sigma, scale=np.exp(mu))
big = pop.rvs(size=2_000_000, random_state=rng)   # for "true" FGT values

rows = [("mean", x.mean(), pop.mean()),
        ("median", np.median(x), pop.median()),
        ("Gini", gini(x), 2 * stats.norm.cdf(sigma / np.sqrt(2)) - 1),
        ("P0 headcount", fgt(x, z, 0), pop.cdf(z)),
        ("P1 poverty gap", fgt(x, z, 1), fgt(big, z, 1)),
        ("P2 severity", fgt(x, z, 2), fgt(big, z, 2))]
print(f"Plug-in estimates from one simulated survey (n = {n})")
print(f"{'functional':<16}{'plug-in T(F_n)':>16}{'true T(F)':>12}")
for name, est, tru in rows:
    print(f"{name:<16}{est:16.4f}{tru:12.4f}")

# Kolmogorov distance and the DKW 95% band
xs = np.sort(x)
Fn = np.arange(1, n + 1) / n
F = pop.cdf(xs)
D = np.max(np.maximum(np.abs(Fn - F), np.abs(Fn - 1 / n - F)))
eps = np.sqrt(np.log(2 / 0.05) / (2 * n))
print(f"sup|F_n - F| = {D:.4f};  DKW 95% half-width = {eps:.4f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
grid = np.linspace(0, xs.max(), 400)
ax.step(xs, Fn, where="post", color="steelblue", label="ECDF $F_n$")
ax.fill_between(xs, np.clip(Fn - eps, 0, 1), np.clip(Fn + eps, 0, 1),
                step="post", color="steelblue", alpha=0.18, label="DKW 95% band")
ax.plot(grid, pop.cdf(grid), color="black", lw=1, ls="--", label="true $F$")
ax.axvline(z, color="firebrick", lw=1, ls=":", label="poverty line $z$")
ax.set_xlim(0, np.quantile(xs, 0.98))
ax.set_xlabel("monthly income (simulated, arbitrary units)")
ax.set_ylabel("cumulative probability")
ax.legend(frameon=False, loc="lower right")
plt.savefig("ch11_ecdf.pdf", bbox_inches="tight")
