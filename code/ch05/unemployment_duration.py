# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Memorylessness of the exponential and Weibull hazards for unemployment spells
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
n = 200_000
expo = rng.exponential(scale=6.0, size=n)            # mean spell 6 months
weib = stats.weibull_min(c=0.7, scale=5.0).rvs(n, random_state=rng)

print("P(T > s + 3 | T > s) for simulated spells (months), s = 0, 3, 6, 12")
for name, T in [("exponential", expo), ("Weibull k=0.7", weib)]:
    row = [((T > s + 3).sum() / (T > s).sum()) for s in [0, 3, 6, 12]]
    print(f"  {name:14s}" + "  ".join(f"{c:.3f}" for c in row))
print(f"  exponential theory exp(-3/6) = {np.exp(-0.5):.3f}")

# hazard h(t) = f(t) / S(t)
t = np.linspace(0.05, 24, 300)
fig, ax = plt.subplots(figsize=(6, 3.8))
for c, lab in [(0.7, "Weibull k=0.7 (negative duration dependence)"),
               (1.0, "Weibull k=1 = exponential (constant)"),
               (1.5, "Weibull k=1.5 (positive duration dependence)")]:
    d = stats.weibull_min(c=c, scale=6.0)
    ax.plot(t, d.pdf(t) / d.sf(t), lw=2, label=lab)
ax.set_ylim(0, 0.6)
ax.set_xlabel("months unemployed")
ax.set_ylabel("hazard of leaving unemployment")
ax.legend(frameon=False, fontsize=8)
plt.savefig("ch05_hazards.pdf", bbox_inches="tight")
