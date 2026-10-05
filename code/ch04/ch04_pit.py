# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Probability integral transform: calibrated and miscalibrated forecasts.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
n = 5_000
# Outcomes (e.g. monthly inflation surprises) drawn from a Student-t, df = 5
y = stats.t(df=5).rvs(size=n, random_state=rng)
models = {"correct: t(5)": stats.t(df=5),
          "too narrow: N(0, 0.8^2)": stats.norm(0, 0.8),
          "too wide: N(0, 2^2)": stats.norm(0, 2.0)}
print(f"{'forecast model':<24} {'KS p-value':>11} {'P(PIT<0.05)':>12}"
      f" {'P(PIT>0.95)':>12}")
pits = {}
for name, m in models.items():
    u = m.cdf(y)                             # PIT values U = F(Y)
    pits[name] = u
    p = stats.kstest(u, "uniform").pvalue
    print(f"{name:<24} {p:11.3g} {np.mean(u < 0.05):12.3f}"
          f" {np.mean(u > 0.95):12.3f}")
print("A calibrated forecast puts 5% of PIT values in each tail.")

fig, axes = plt.subplots(1, 3, figsize=(8, 2.9), sharey=True)
for a, (name, u) in zip(axes, pits.items()):
    a.hist(u, bins=20, range=(0, 1), density=True, alpha=0.7)
    a.axhline(1, color="k", ls="--", lw=1)
    a.set_title(name, fontsize=9)
    a.set_xlabel("PIT value F(y)")
    a.grid(alpha=0.3)
axes[0].set_ylabel("density")
fig.tight_layout()
plt.savefig("ch04_pit.pdf", bbox_inches="tight")
