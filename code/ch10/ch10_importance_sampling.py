# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Importance sampling for a rare-event probability; effective sample size.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
n, a = 10_000, 4.0
exact = stats.norm.sf(a)
print(f"target P(Z > {a}) = {exact:.4e}, n = {n}")

z = rng.standard_normal(n)
h = (z > a).astype(float)
print(f"plain MC           : {h.mean():.4e}  hits {int(h.sum())}")


def importance(mu, sd):
    x = rng.normal(mu, sd, n)
    w = stats.norm.pdf(x) / stats.norm.pdf(x, mu, sd)      # p / q
    hw = (x > a) * w
    est, se = hw.mean(), hw.std(ddof=1) / np.sqrt(n)
    ess = w.sum() ** 2 / (w**2).sum()              # Kong ESS of the weights
    ess_h = hw.sum() ** 2 / (hw**2).sum()          # ESS for this integrand
    return est, se, ess, ess_h


print("proposal          estimate     s.e.      rel.err  ESS(w)  ESS(h*w)")
for mu, sd in [(4.0, 1.0), (4.5, 1.0), (3.0, 2.0), (0.0, 2.0), (6.0, 0.5)]:
    est, se, ess, ess_h = importance(mu, sd)
    print(f"N({mu:.1f}, {sd:.1f}^2)       {est:.4e}  {se:.2e}  "
          f"{abs(est / exact - 1):7.4f}  {ess:6.0f}  {ess_h:8.0f}")

# Optimal proposal q* ~ |h| p: here the normal truncated to (a, inf)
x = stats.truncnorm.rvs(a, np.inf, size=n, random_state=rng)
w = stats.norm.pdf(x) / stats.truncnorm.pdf(x, a, np.inf)
print(f"optimal q* (trunc) : {w.mean():.4e}  s.e. {w.std():.1e} (zero variance)")

# Figure: integrand p(x) 1{x>a}, and two proposals
xx = np.linspace(-3, 8, 500)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(xx, stats.norm.pdf(xx), color="navy", label="p = N(0,1)")
ax.fill_between(xx, 0, stats.norm.pdf(xx) * (xx > a) * 1e3, color="firebrick",
                alpha=0.4, label="p(x) 1{x>4}  (x 1000)")
ax.plot(xx, stats.norm.pdf(xx, 4, 1), color="darkorange", ls="--",
        label="good q = N(4,1)")
ax.plot(xx, stats.norm.pdf(xx, 6, 0.5), color="gray", ls=":",
        label="poor q = N(6,0.5^2)")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.legend(fontsize=8)
plt.savefig("ch10_importance_sampling.pdf", bbox_inches="tight")
