# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Box-Muller transform: two uniforms in, two independent standard normals out.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
n = 500_000
U1, U2 = rng.random(n), rng.random(n)
R = np.sqrt(-2 * np.log1p(-U1))              # radius: R^2 ~ Exp(mean 2)
theta = 2 * np.pi * U2                       # angle ~ Uniform(0, 2 pi)
Z1, Z2 = R * np.cos(theta), R * np.sin(theta)

print(f"{'statistic':<20} {'Z1':>8} {'Z2':>8} {'N(0,1)':>8}")
for name, f, t in [("mean", np.mean, 0.0), ("variance", np.var, 1.0),
                   ("skewness", stats.skew, 0.0),
                   ("excess kurtosis", stats.kurtosis, 0.0)]:
    print(f"{name:<20} {f(Z1):8.4f} {f(Z2):8.4f} {t:8.4f}")
print(f"corr(Z1, Z2)         {np.corrcoef(Z1, Z2)[0, 1]:8.4f}")
print(f"corr(Z1^2, Z2^2)     {np.corrcoef(Z1**2, Z2**2)[0, 1]:8.4f}")
print(f"KS p-value Z1: {stats.kstest(Z1[:20_000], 'norm').pvalue:.3f}, "
      f"Z2: {stats.kstest(Z2[:20_000], 'norm').pvalue:.3f}")
print(f"P(|Z1| > 1.96) = {np.mean(abs(Z1) > 1.96):.4f}  (0.0500)")

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
axes[0].scatter(U1[:2000], U2[:2000], s=2, alpha=0.5)
axes[0].set_title("input (U1, U2)")
axes[0].set(xlabel="U1", ylabel="U2")
axes[1].scatter(Z1[:2000], Z2[:2000], s=2, alpha=0.5, color="C1")
axes[1].set_title("output (Z1, Z2)")
axes[1].set(xlabel="Z1", ylabel="Z2")
axes[1].set_aspect("equal")
for a in axes:
    a.grid(alpha=0.3)
fig.tight_layout()
plt.savefig("ch04_boxmuller.pdf", bbox_inches="tight")
