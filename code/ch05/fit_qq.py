# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Model checking: fit candidate duration models with .fit, compare AIC, QQ plots
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
# simulated unemployment spells (months) from a gamma model with shape 1.8
spells = stats.gamma(a=1.8, scale=4.0).rvs(800, random_state=rng)

cands = {"exponential": stats.expon, "gamma": stats.gamma,
         "Weibull": stats.weibull_min, "log-normal": stats.lognorm}
fits = {}
print(f"{'model':12s}{'params (shape, loc, scale)':>34s}{'loglik':>10s}{'AIC':>9s}")
for name, fam in cands.items():
    par = fam.fit(spells, floc=0)           # MLE with location fixed at 0
    ll = fam.logpdf(spells, *par).sum()
    kpar = len(par) - 1                     # loc was not estimated
    fits[name] = fam(*par)
    ps = ", ".join(f"{v:.3f}" for v in par)
    print(f"{name:12s}{ps:>34s}{ll:10.1f}{2 * kpar - 2 * ll:9.1f}")

ks = stats.kstest(spells, fits["exponential"].cdf)
print(f"KS test vs fitted exponential: D = {ks.statistic:.3f}, "
      f"p = {ks.pvalue:.2g} (nominal: params were estimated)")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
pr = (np.arange(1, len(spells) + 1) - 0.5) / len(spells)
xs = np.sort(spells)
for ax, name in zip(axes, ["exponential", "gamma"]):
    q = fits[name].ppf(pr)
    ax.scatter(q, xs, s=6, color="#1f4e79")
    lim = [0, max(q.max(), xs.max())]
    ax.plot(lim, lim, color="#cc9900", lw=1.5)
    ax.set_title(f"QQ plot: {name} fit", fontsize=10)
    ax.set_xlabel("theoretical quantile")
    ax.set_ylabel("sample quantile")
fig.tight_layout()
plt.savefig("ch05_qqplots.pdf", bbox_inches="tight")
