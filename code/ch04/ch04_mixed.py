# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# A mixed distribution: zero-inflated household health expenditure (simulated).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
pi0 = 0.40                                   # P(no health spending last month)
pos = stats.lognorm(s=1.0, scale=np.exp(3.0))  # spending given spending > 0


def cdf(x):
    """F(x) = pi0 * 1{x >= 0} + (1 - pi0) * G(x): a jump of size pi0 at zero."""
    x = np.asarray(x, dtype=float)
    return np.where(x < 0, 0.0, pi0 + (1 - pi0) * pos.cdf(np.maximum(x, 0)))


def quantile(p):
    """Generalised inverse F^{-1}(p) = inf{x : F(x) >= p}."""
    p = np.asarray(p, dtype=float)
    q = pos.ppf(np.clip((p - pi0) / (1 - pi0), 0, 1))
    return np.where(p <= pi0, 0.0, q)


n = 100_000
spend = np.where(rng.random(n) < pi0, 0.0, pos.rvs(size=n, random_state=rng))
print(f"P(X = 0): model {pi0:.3f}   sample {np.mean(spend == 0):.3f}")
print(f"P(X <= 50): model {cdf(50.0):.3f}   sample {np.mean(spend <= 50):.3f}")
mean_model = (1 - pi0) * pos.mean()
print(f"E[X]:  model {mean_model:7.2f}   sample {spend.mean():7.2f}")
m_pos = spend[spend > 0].mean()
print(f"E[X | X > 0]: model {pos.mean():7.2f}   sample {m_pos:7.2f}")
print("\n  p   quantile F^-1(p)   sample quantile")
for p in [0.25, 0.40, 0.50, 0.75, 0.95]:
    print(f"{p:4.2f}   {quantile(p):12.2f}   {np.quantile(spend, p):12.2f}")
print("\nNo density exists: the point x = 0 carries positive probability.")

fig, ax = plt.subplots(figsize=(6, 3.8))
grid = np.linspace(0, 150, 600)
ax.plot([-20, 0], [0, 0], "C0", lw=2)
ax.plot(grid, cdf(grid), "C0", lw=2, label="CDF of spending X")
ax.plot(0, 0, "o", mfc="white", mec="C0")
ax.plot(0, pi0, "o", color="C0")
ax.annotate("jump of size P(X = 0) = 0.4", xy=(0, 0.2), xytext=(25, 0.15),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlim(-20, 150)
ax.set(xlabel="monthly health expenditure (simulated, cedi)", ylabel="F(x)")
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.3)
plt.savefig("ch04_mixed.pdf", bbox_inches="tight")
