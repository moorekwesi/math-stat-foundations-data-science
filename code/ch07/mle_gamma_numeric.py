# DEP 803 -- Chapter 7: Estimation Principles
# Numerical MLE for the gamma distribution: log-parametrisation, MoM start, contours.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, special, stats

rng = np.random.default_rng(803)
x = rng.gamma(2.5, 1.6, size=300)       # simulated household consumption
n, sx, slx = x.size, x.sum(), np.log(x).sum()


def negll(t):
    """Negative gamma log-likelihood in (log k, log theta)."""
    k, th = np.exp(t)
    return -((k - 1) * slx - sx / th - n * special.gammaln(k)
             - n * k * np.log(th))


def grad(t):
    k, th = np.exp(t)
    dk = slx - n * special.digamma(k) - n * np.log(th)
    dth = sx / th**2 - n * k / th
    return -np.array([dk * k, dth * th])       # chain rule for log-params


m, v = x.mean(), x.var()
start = np.log([m**2 / v, v / m])              # method-of-moments start
res = optimize.minimize(negll, start, jac=grad, method="BFGS")
k_hat, th_hat = np.exp(res.x)
print(f"MoM start       : k = {m**2 / v:.4f}, theta = {v / m:.4f}")
print(f"MLE (BFGS)      : k = {k_hat:.4f}, theta = {th_hat:.4f}")
print(f"  converged = {res.success}, iterations = {res.nit},"
      f" max |grad| = {np.abs(res.jac).max():.1e}")
kf, _, thf = stats.gamma.fit(x, floc=0)
print(f"scipy gamma.fit : k = {kf:.4f}, theta = {thf:.4f}")
print(f"Score eq. check : k*theta = {k_hat * th_hat:.4f},"
      f" mean(x) = {m:.4f}")

# Figure: log-likelihood contours in (k, theta)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
kk = np.linspace(1.5, 4.0, 200)
tt = np.linspace(0.9, 2.6, 200)
K, TH = np.meshgrid(kk, tt)
LL = ((K - 1) * slx - sx / TH - n * special.gammaln(K) - n * K * np.log(TH))
fig, ax = plt.subplots(figsize=(6, 3.8))
lev = LL.max() - np.array([20.0, 10.0, 5.0, 2.0, 0.5])
cs = ax.contour(K, TH, LL, levels=lev, cmap="viridis")
ax.clabel(cs, fmt=lambda z: f"{z - LL.max():.1f}", fontsize=7)
ax.plot(k_hat, th_hat, "r*", ms=11, label="MLE")
ax.plot(m**2 / v, v / m, "ko", ms=5, label="method of moments")
ax.plot(kk, m / kk, "k--", lw=0.8, label="k * theta = sample mean")
ax.set_ylim(0.9, 2.6)
ax.set_xlabel("shape k")
ax.set_ylabel("scale theta")
ax.set_title("Gamma log-likelihood (relative to maximum)")
ax.legend(frameon=False, fontsize=9)
plt.savefig("ch07_gamma_loglik.pdf", bbox_inches="tight")
