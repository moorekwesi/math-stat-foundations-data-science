# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Densities and moments of common continuous distributions with scipy.stats
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

dists = {
    "Uniform(0,1)": stats.uniform(0, 1),
    "Normal(0,1)": stats.norm(0, 1),
    "Exponential(rate 2)": stats.expon(scale=1 / 2),
    "Gamma(3, rate 2)": stats.gamma(a=3, scale=1 / 2),
    "Beta(2,5)": stats.beta(2, 5),
    "LogNormal(0,0.5)": stats.lognorm(s=0.5, scale=1.0),
    "Pareto(alpha 3)": stats.pareto(b=3),
    "Weibull(k 1.5)": stats.weibull_min(c=1.5),
}
print(f"{'distribution':21s}{'mean':>8s}{'var':>8s}{'skew':>8s}{'median':>8s}")
for name, d in dists.items():
    mean, var, skew = d.stats(moments="mvs")
    print(f"{name:21s}{float(mean):8.3f}{float(var):8.3f}"
          f"{float(skew):8.3f}{d.median():8.3f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
x = np.linspace(0.001, 4, 400)
for name in ["Exponential(rate 2)", "Gamma(3, rate 2)", "LogNormal(0,0.5)",
             "Weibull(k 1.5)"]:
    axes[0].plot(x, dists[name].pdf(x), lw=2, label=name)
axes[0].set_xlabel("x")
axes[0].set_ylabel("density")
axes[0].legend(frameon=False, fontsize=8)
u = np.linspace(0.001, 0.999, 400)
for a, b in [(1, 1), (2, 5), (5, 2), (0.5, 0.5)]:
    axes[1].plot(u, stats.beta(a, b).pdf(u), lw=2, label=f"Beta({a},{b})")
axes[1].set_xlabel("proportion")
axes[1].set_ylim(0, 3.5)
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.savefig("ch05_continuous_pdfs.pdf", bbox_inches="tight")
