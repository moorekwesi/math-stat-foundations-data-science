# DEP 803 -- Chapter 7: Estimation Principles
# Grid approximation of a two-parameter logit posterior: MAP, MLE, posterior mean.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

rng = np.random.default_rng(803)
n = 30                                   # small pilot survey (simulated)
dist = rng.uniform(0, 5, n)              # km to enrolment office
y = rng.binomial(1, special.expit(1.0 - 0.6 * dist))

a = np.linspace(-3, 5, 401)              # intercept grid
b = np.linspace(-2.5, 1.0, 401)          # slope grid
A, B = np.meshgrid(a, b, indexing="ij")
eta = A[..., None] + B[..., None] * dist           # (401, 401, n)
loglik = np.sum(y * eta - np.logaddexp(0, eta), axis=-1)
tau = 2.5                                # N(0, tau^2) priors on a and b
logprior = -(A**2 + B**2) / (2 * tau**2)
logpost = loglik + logprior
post = np.exp(logpost - logpost.max())
post /= post.sum()                       # normalise on the grid

pa, pb = post.sum(axis=1), post.sum(axis=0)        # marginals
mean_a, mean_b = pa @ a, pb @ b
sd_b = np.sqrt(pb @ (b - mean_b) ** 2)
cdf_b = np.cumsum(pb)
lo, hi = b[np.searchsorted(cdf_b, 0.025)], b[np.searchsorted(cdf_b, 0.975)]
i, j = np.unravel_index(np.argmax(logpost), logpost.shape)
k, l = np.unravel_index(np.argmax(loglik), loglik.shape)
p_neg = pb[b < 0].sum()
print(f"Logit take-up posterior on a {a.size} x {b.size} grid, n = {n}")
print(f"  MLE  (grid)      : a = {a[k]:6.3f}, b = {b[l]:6.3f}")
print(f"  MAP  (grid)      : a = {a[i]:6.3f}, b = {b[j]:6.3f}")
print(f"  posterior mean   : a = {mean_a:6.3f}, b = {mean_b:6.3f}")
print(f"  posterior sd of b = {sd_b:.3f}; 95% credible interval for b:"
      f" [{lo:.3f}, {hi:.3f}]")
print(f"  P(b < 0 | data)  = {p_neg:.4f}")
# Posterior predictive take-up probability at 4 km from the office
pr4 = np.sum(post * special.expit(A + 4 * B))
print(f"  posterior predictive P(take-up | 4 km) = {pr4:.3f}")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6, 3.8))
lev = np.exp(-0.5 * np.array([9.21, 5.99, 2.41]))    # chi2_2 levels
ax.contour(A, B, np.exp(logpost - logpost.max()), levels=lev,
           colors=["tab:blue"], linewidths=[0.8, 1.2, 1.6])
ax.plot(a[k], b[l], "k^", label="MLE")
ax.plot(a[i], b[j], "r*", ms=10, label="MAP")
ax.plot(mean_a, mean_b, "go", label="posterior mean")
ax.set_xlabel("intercept a")
ax.set_ylabel("slope b (per km)")
ax.set_title("Posterior contours (approx. 70%, 95%, 99%)")
ax.legend(frameon=False, fontsize=9)
plt.savefig("ch07_bayes_grid.pdf", bbox_inches="tight")
