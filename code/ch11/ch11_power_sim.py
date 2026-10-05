# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Power of permutation tests vs the t-test for a cash-transfer RCT.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
n1 = n0 = 30
n, R, M = n1 + n0, 500, 499
effects = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])   # in units of outcome sd
z = np.r_[np.ones(n1, bool), np.zeros(n0, bool)]
perms = np.argsort(rng.random((M, n)), axis=1)        # reused permutations
Zp = z[perms]                                         # (M, n) assignments


def perm_p(y):
    """Two-sided permutation p-values for difference in means and in ranks."""
    out = []
    for v in (y, stats.rankdata(y)):
        t0 = v[z].mean() - v[~z].mean()
        tp = (Zp * v).sum(1) / n1 - (~Zp * v).sum(1) / n0
        out.append((1 + np.sum(np.abs(tp) >= abs(t0))) / (M + 1))
    return out


def draw(kind, size):
    if kind == "normal":
        return rng.normal(0, 1, size)
    e = rng.lognormal(0, 1, size)                       # skewed consumption
    return (e - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)


power = {}
for kind in ("normal", "lognormal"):
    pw = np.zeros((len(effects), 3))
    for i, eff in enumerate(effects):
        for r in range(R):
            y = draw(kind, n) + eff * z
            p_mean, p_rank = perm_p(y)
            p_t = stats.ttest_ind(y[z], y[~z]).pvalue
            pw[i] += np.array([p_t, p_mean, p_rank]) < 0.05
    power[kind] = pw / R

print(f"Rejection rates at 5% (n = {n1}+{n0}, {R} reps, {M} permutations)")
for kind, pw in power.items():
    print(f"{kind} outcomes:  effect   t-test  perm(mean)  perm(rank)")
    for eff, row in zip(effects, pw):
        print(f"{'':18}{eff:5.1f}{row[0]:9.3f}{row[1]:12.3f}{row[2]:12.3f}")

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4), sharey=True)
for ax, (kind, pw) in zip(axes, power.items()):
    for k, lab in enumerate(["t-test", "permutation (mean)",
                             "permutation (ranks)"]):
        ax.plot(effects, pw[:, k], marker="o", ms=3, label=lab)
    ax.axhline(0.05, color="black", lw=0.8, ls="--")
    ax.set_title(f"{kind} outcomes", fontsize=10)
    ax.set_xlabel("effect size (sd units)")
axes[0].set_ylabel("power")
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
plt.savefig("ch11_power.pdf", bbox_inches="tight")
