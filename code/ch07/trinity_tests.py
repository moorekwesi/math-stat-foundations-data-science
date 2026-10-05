# DEP 803 -- Chapter 7: Estimation Principles
# Null distributions of Wald, likelihood-ratio and score tests (exponential rate).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
lam0, R = 1.0, 50000
crit = stats.chi2.ppf(0.95, 1)


def trinity(x, lam0):
    n = x.shape[1]
    xbar = x.mean(axis=1)
    lhat = 1 / xbar
    wald_r = n * (lhat - lam0) ** 2 / lhat**2        # rate scale, I = 1/lam^2
    mu0 = 1 / lam0
    wald_m = n * (xbar - mu0) ** 2 / xbar**2         # mean scale, I = 1/mu^2
    lr = 2 * n * (np.log(lhat / lam0) - 1 + lam0 * xbar)
    score = n * (1 - lam0 * xbar) ** 2               # U^2 / I(lam0)
    return wald_r, wald_m, lr, score


print(f"H0: rate = {lam0}; R = {R}; nominal size 0.05 (chi2_1 crit"
      f" {crit:.3f})")
print(f"{'n':>5} {'Wald(rate)':>11} {'Wald(mean)':>11} {'LR':>8} {'score':>8}")
for n in (5, 10, 30, 200):
    x = rng.exponential(1 / lam0, size=(R, n))
    rej = [np.mean(t > crit) for t in trinity(x, lam0)]
    print(f"{n:>5} {rej[0]:11.4f} {rej[1]:11.4f} {rej[2]:8.4f} {rej[3]:8.4f}")

# Power at a local alternative (n = 30, true rate 1.4)
x = rng.exponential(1 / 1.4, size=(R, 30))
pw = [np.mean(t > crit) for t in trinity(x, lam0)]
print("power, n = 30, rate = 1.4: Wald(rate) %.3f, Wald(mean) %.3f,"
      " LR %.3f, score %.3f" % tuple(pw))

# Figure: QQ plot of the three statistics against chi2_1 for n = 10
x = rng.exponential(1 / lam0, size=(R, 10))
wr, wm, l, s = trinity(x, lam0)
q = np.linspace(0.01, 0.995, 200)
theo = stats.chi2.ppf(q, 1)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6, 3.8))
for stat, lab in ((wr, "Wald (rate) = score"), (wm, "Wald (mean)"),
                  (l, "likelihood ratio")):
    ax.plot(theo, np.quantile(stat, q), lw=1.6, label=lab)
ax.plot(theo, theo, "k--", lw=0.8, label="chi-square(1)")
ax.set_xlim(0, 8)
ax.set_ylim(0, 14)
ax.set_xlabel("chi-square(1) quantile")
ax.set_ylabel("simulated quantile (n = 10)")
ax.legend(frameon=False)
plt.savefig("ch07_trinity_qq.pdf", bbox_inches="tight")
