# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Gaussian copula: same dependence, different marginals (Sklar's theorem).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
n, r = 4_000, 0.7
Z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], size=n)
U = stats.norm.cdf(Z)                        # copula sample: uniform marginals
income = stats.lognorm(s=0.8, scale=np.exp(1.0)).ppf(U[:, 0])
loss = stats.expon(scale=2.0).ppf(U[:, 1])  # e.g. shock-related losses

print("Marginal checks (KS p-values against Uniform(0,1)):")
for j in range(2):
    print(f"  U{j + 1}: p = {stats.kstest(U[:, j], 'uniform').pvalue:.3f}")
rows = [("normal scores Z", Z[:, 0], Z[:, 1]),
        ("copula scale U", U[:, 0], U[:, 1]),
        ("income, loss", income, loss)]
print(f"\n{'scale':<17} {'Pearson':>8} {'Spearman':>9} {'Kendall':>8}")
for name, a, b in rows:
    pr = np.corrcoef(a, b)[0, 1]
    sp = stats.spearmanr(a, b).correlation
    kt = stats.kendalltau(a, b).correlation
    print(f"{name:<17} {pr:8.3f} {sp:9.3f} {kt:8.3f}")
print(f"\nTheory: Spearman = (6/pi) asin(r/2) = {6 / np.pi * np.arcsin(r / 2):.3f},"
      f" Kendall = (2/pi) asin(r) = {2 / np.pi * np.arcsin(r):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
axes[0].scatter(U[:, 0], U[:, 1], s=2, alpha=0.3)
axes[0].set_title("copula sample (U1, U2)")
axes[0].set(xlabel="U1", ylabel="U2")
axes[1].scatter(income, loss, s=2, alpha=0.3, color="C1")
axes[1].set_xlim(0, 15)
axes[1].set_ylim(0, 12)
axes[1].set_title("same copula, new marginals")
axes[1].set(xlabel="income (lognormal)", ylabel="loss (exponential)")
for a in axes:
    a.grid(alpha=0.3)
fig.tight_layout()
plt.savefig("ch04_copula.pdf", bbox_inches="tight")
