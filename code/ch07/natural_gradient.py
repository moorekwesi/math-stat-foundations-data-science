# DEP 803 -- Chapter 7: Estimation Principles
# Ordinary vs natural gradient ascent for the normal log-likelihood in (mu, sigma).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
x = rng.normal(3.0, 0.5, size=200)
m1, m2 = x.mean(), x.var()             # MLE: mu = m1, sigma = sqrt(m2)


def avg_ll(mu, s):
    return -np.log(s) - (m2 + (m1 - mu) ** 2) / (2 * s**2)


def grad(t):
    mu, s = t
    d2 = m2 + (m1 - mu) ** 2
    return np.array([(m1 - mu) / s**2, -1 / s + d2 / s**3])


def fisher_inv(t):
    s = t[1]
    return np.diag([s**2, s**2 / 2])   # inverse of diag(1/s^2, 2/s^2)


def ascend(t0, eta, natural, iters=5000):
    t, path = np.array(t0, float), [np.array(t0, float)]
    for k in range(iters):
        g = grad(t)
        step = fisher_inv(t) @ g if natural else g
        t = t + eta * step
        path.append(t.copy())
        if np.linalg.norm(t - [m1, np.sqrt(m2)]) < 1e-6:
            return np.array(path), k + 1
        if not np.all(np.isfinite(t)) or t[1] <= 0:
            return np.array(path), np.inf
    return np.array(path), iters


start = (1.0, 0.4)
print(f"MLE: mu = {m1:.4f}, sigma = {np.sqrt(m2):.4f}; start = {start}")
print(f"{'method':<18} {'step':>6} {'iterations to 1e-6':>20}")
paths = {}
for eta in (0.01, 0.05, 0.2):
    p, k = ascend(start, eta, natural=False)
    paths[("ordinary", eta)] = p
    print(f"{'ordinary gradient':<18} {eta:6.2f} {k:>20}")
for eta in (0.2, 0.5, 1.0):
    p, k = ascend(start, eta, natural=True)
    paths[("natural", eta)] = p
    print(f"{'natural gradient':<18} {eta:6.2f} {k:>20}")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
MU, S = np.meshgrid(np.linspace(0, 3.6, 300), np.linspace(0.2, 3.8, 300))
fig, ax = plt.subplots(figsize=(6, 3.8))
LLg = avg_ll(MU, S)
lev = LLg.max() - np.logspace(2, -1.5, 12)     # log-spaced drops from max
ax.contour(MU, S, LLg, levels=lev, colors="0.65", linewidths=0.7)
po, pn = paths[("ordinary", 0.05)], paths[("natural", 0.2)]
ax.plot(po[:, 0], po[:, 1], ".-", ms=2, lw=1, color="tab:red",
        label="ordinary gradient (step 0.05)")
ax.plot(pn[:, 0], pn[:, 1], ".-", ms=2, lw=1, color="tab:blue",
        label="natural gradient (step 0.2)")
ax.plot(m1, np.sqrt(m2), "k*", ms=11, label="MLE")
ax.set_xlabel("mu")
ax.set_ylabel("sigma")
ax.legend(frameon=False, fontsize=9, loc="upper right")
plt.savefig("ch07_natural_gradient.pdf", bbox_inches="tight")
