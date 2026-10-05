# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Simpson's paradox: a training programme helps within regions, 'hurts' pooled.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
# Simulated: hours of training T; earnings Y; region 0 (richer) or 1 (poorer).
# The programme targets the poorer region, which trains more and earns less.
n = 600
region = rng.integers(0, 2, size=n)
base = np.where(region == 0, 60.0, 30.0)
T = np.where(region == 0, rng.uniform(0, 10, n), rng.uniform(10, 25, n))
Y = base + 0.8 * T + rng.normal(0, 4, n)


def slope(x, y):
    return np.cov(x, y)[0, 1] / np.var(x, ddof=1)


print(f"pooled slope of Y on T:        {slope(T, Y): .3f}")
for r, lab in [(0, "region A"), (1, "region B")]:
    m = region == r
    print(f"slope within {lab}:         {slope(T[m], Y[m]): .3f}  (n = {m.sum()})")
# Adjusting for region: regress Y on T and a region dummy
Z = np.column_stack([np.ones(n), T, region])
b = np.linalg.lstsq(Z, Y, rcond=None)[0]
print(f"slope on T controlling for region: {b[1]: .3f}")

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(6, 3.8))
cols = ["tab:blue", "tab:orange"]
for r, lab in [(0, "region A (richer)"), (1, "region B (poorer)")]:
    m = region == r
    ax.scatter(T[m], Y[m], s=8, alpha=0.5, color=cols[r], label=lab)
    tt = np.linspace(T[m].min(), T[m].max(), 2)
    ax.plot(tt, Y[m].mean() + slope(T[m], Y[m]) * (tt - T[m].mean()),
            color=cols[r], lw=2)
tt = np.linspace(0, 25, 2)
ax.plot(tt, Y.mean() + slope(T, Y) * (tt - T.mean()), "k--", lw=2,
        label="pooled fit")
ax.set_xlabel("training hours T (simulated)")
ax.set_ylabel("earnings Y (simulated)")
ax.legend(frameon=False, fontsize=9)
plt.savefig("ch06_simpson.pdf", bbox_inches="tight")
