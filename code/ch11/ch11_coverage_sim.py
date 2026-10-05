# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Frequentist coverage of the 95% t-interval for a mean of skewed incomes.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)

# Simulated "population" of monthly incomes: lognormal (arbitrary units)
mu, sigma = 7.0, 1.0
true_mean = np.exp(mu + sigma**2 / 2)
R = 20000                       # Monte Carlo replications per sample size

print("Coverage of the nominal 95% t-interval for the mean income")
print(f"true mean = {true_mean:.1f}, replications = {R}")
print(f"{'n':>6} {'coverage':>9} {'miss low':>9} {'miss high':>10} {'avg len':>9}")
for n in (10, 30, 100, 500):
    x = rng.lognormal(mu, sigma, size=(R, n))
    xbar, s = x.mean(axis=1), x.std(axis=1, ddof=1)
    half = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    lo, hi = xbar - half, xbar + half
    cover = np.mean((lo <= true_mean) & (true_mean <= hi))
    below = np.mean(hi < true_mean)          # interval entirely below truth
    above = np.mean(lo > true_mean)          # interval entirely above truth
    print(f"{n:6d} {cover:9.3f} {below:9.3f} {above:10.3f} {np.mean(2*half):9.1f}")

# Figure: 60 realised intervals for n = 30
n, K = 30, 60
x = rng.lognormal(mu, sigma, size=(K, n))
xbar, s = x.mean(axis=1), x.std(axis=1, ddof=1)
half = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
miss = (xbar - half > true_mean) | (xbar + half < true_mean)
fig, ax = plt.subplots(figsize=(6, 3.8))
for k in range(K):
    col = "firebrick" if miss[k] else "steelblue"
    ax.plot([k, k], [xbar[k] - half[k], xbar[k] + half[k]], color=col, lw=1.4)
    ax.plot(k, xbar[k], "o", color=col, ms=2.5)
ax.axhline(true_mean, color="black", lw=1, ls="--", label="true mean")
ax.set_xlabel("replicated survey (n = 30 households each)")
ax.set_ylabel("95% t-interval for mean income")
ax.legend(frameon=False, loc="upper right")
plt.savefig("ch11_coverage.pdf", bbox_inches="tight")
print(f"figure: {miss.sum()} of {K} intervals miss the true mean")
