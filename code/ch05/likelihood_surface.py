# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Log-likelihood surfaces: normal (mu, sigma) and logistic regression (b0, b1)
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

rng = np.random.default_rng(803)
y = rng.normal(2.0, 1.5, 40)                       # simulated sample

# normal log-likelihood on a grid
M, S = np.meshgrid(np.linspace(1, 3.2, 200), np.linspace(0.9, 2.6, 200))
ll_norm = stats.norm.logpdf(y[:, None, None], M, S).sum(axis=0)
i, j = np.unravel_index(ll_norm.argmax(), ll_norm.shape)
print(f"normal: grid max at mu = {M[i, j]:.3f}, sigma = {S[i, j]:.3f}")
print(f"        closed form mu = {y.mean():.3f}, sigma = {y.std(ddof=0):.3f}")

# logistic regression log-likelihood
n = 150
x = rng.normal(0, 1, n)
yb = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.2 * x))))


def negll(b):
    eta = b[0] + b[1] * x
    return -(yb * eta - np.logaddexp(0, eta)).sum()


res = optimize.minimize(negll, np.zeros(2), method="BFGS")
print(f"logistic: MLE b0 = {res.x[0]:.3f}, b1 = {res.x[1]:.3f}, "
      f"max loglik = {-res.fun:.2f}")
B0, B1 = np.meshgrid(np.linspace(-1.5, 0.5, 150), np.linspace(0.2, 2.4, 150))
eta = B0[None] + B1[None] * x[:, None, None]
ll_log = (yb[:, None, None] * eta - np.logaddexp(0, eta)).sum(axis=0)
# likelihood-ratio 95% region: 2 (l_max - l) <= chi2_2(0.95)
crit = stats.chi2(2).ppf(0.95) / 2
print(f"95% LR region: loglik >= max - {crit:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
lev = ll_norm.max() - np.array([8, 4, 2, 1, 0.5])
axes[0].contour(M, S, ll_norm, levels=np.sort(lev), cmap="viridis")
axes[0].plot(y.mean(), y.std(), "r*", ms=10)
axes[0].set_xlabel("mu")
axes[0].set_ylabel("sigma")
axes[0].set_title("normal log-likelihood", fontsize=10)
lev2 = -res.fun - np.array([10, 5, crit, 1])
axes[1].contour(B0, B1, ll_log, levels=np.sort(lev2), cmap="viridis")
axes[1].contour(B0, B1, ll_log, levels=[-res.fun - crit], colors="#b22222")
axes[1].plot(*res.x, "r*", ms=10)
axes[1].set_xlabel("b0")
axes[1].set_ylabel("b1")
axes[1].set_title("logistic log-likelihood (red: 95% LR region)", fontsize=10)
fig.tight_layout()
plt.savefig("ch05_likelihood_contours.pdf", bbox_inches="tight")
