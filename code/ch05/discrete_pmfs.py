# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Means, variances and PMFs of the main discrete distributions with scipy.stats
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# (name, frozen distribution, closed-form mean, closed-form variance)
p, n, lam, r = 0.3, 20, 4.0, 3
dists = [
    ("Bernoulli(0.3)", stats.bernoulli(p), p, p * (1 - p)),
    ("Binomial(20,0.3)", stats.binom(n, p), n * p, n * p * (1 - p)),
    ("Poisson(4)", stats.poisson(lam), lam, lam),
    ("Geometric(0.3)", stats.geom(p), 1 / p, (1 - p) / p**2),
    ("NegBin(3,0.3)", stats.nbinom(r, p), r * (1 - p) / p, r * (1 - p) / p**2),
]
print(f"{'distribution':18s}{'mean':>8s}{'formula':>9s}{'var':>9s}{'formula':>9s}")
for name, d, m, v in dists:
    print(f"{name:18s}{d.mean():8.3f}{m:9.3f}{d.var():9.3f}{v:9.3f}")
print("note: scipy geom counts trials (support 1,2,...);")
print("      scipy nbinom counts failures before the r-th success.")

# multinomial: sector choice of 10 new firms with probs (0.5, 0.3, 0.2)
mn = stats.multinomial(10, [0.5, 0.3, 0.2])
print("P(5,3,2) under Multinomial(10; .5,.3,.2) =", round(mn.pmf([5, 3, 2]), 4))
print("Cov matrix of the counts:")
print(np.round(mn.cov(), 2))

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
k = np.arange(0, 16)
for ax, (name, d, _, _) in zip(axes, [dists[1], dists[2], dists[4]]):
    ax.bar(k, d.pmf(k), color="#1f4e79", width=0.7)
    ax.axvline(d.mean(), color="#cc9900", lw=2, label="mean")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("k")
axes[0].set_ylabel("P(X = k)")
axes[0].legend(frameon=False)
fig.tight_layout()
plt.savefig("ch05_discrete_pmfs.pdf", bbox_inches="tight")
