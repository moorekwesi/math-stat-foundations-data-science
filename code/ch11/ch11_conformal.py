# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Prediction intervals: normal theory, bootstrap and split conformal.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
alpha, n, B = 0.10, 400, 999


def simulate(size):
    """Log earnings vs experience; skewed, heteroskedastic errors."""
    x = rng.uniform(0, 30, size)
    e = (rng.gamma(2.0, 1.0, size) - 2.0) * (0.10 + 0.012 * x)
    return x, 1.5 + 0.06 * x - 0.001 * x**2 + e


def design(x):
    return np.column_stack([np.ones_like(x), x, x**2])


def intervals(x, y, x_new):
    """Return dict of (lo, hi) at x_new for three methods."""
    out = {}
    X, Xn = design(x), design(x_new)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    s = np.sqrt(r @ r / (len(y) - 3))
    lev = np.einsum("ij,jk,ik->i", Xn, np.linalg.inv(X.T @ X), Xn)
    h = stats.t.ppf(1 - alpha / 2, len(y) - 3) * s * np.sqrt(1 + lev)
    out["normal theory"] = (Xn @ b - h, Xn @ b + h)
    # residual bootstrap: refit on y* and add a fresh resampled residual
    ys = X @ b + r[rng.integers(0, len(y), (B, len(y)))]
    bs = np.linalg.lstsq(X, ys.T, rcond=None)[0]            # (3, B)
    pred = (Xn @ bs).T + r[rng.integers(0, len(y), (B, len(x_new)))]
    out["residual bootstrap"] = tuple(np.quantile(pred, [alpha / 2,
                                                         1 - alpha / 2], 0))
    # split conformal: fit on half, calibrate absolute residuals on half
    tr = rng.permutation(len(y))
    fit, cal = tr[: len(y) // 2], tr[len(y) // 2:]
    b1 = np.linalg.lstsq(X[fit], y[fit], rcond=None)[0]
    score = np.abs(y[cal] - X[cal] @ b1)
    k = int(np.ceil((len(cal) + 1) * (1 - alpha)))
    qhat = np.sort(score)[k - 1]
    out["split conformal"] = (Xn @ b1 - qhat, Xn @ b1 + qhat)
    # locally weighted conformal: scale scores by a fitted spread model
    g = np.linalg.lstsq(design(x[fit]), np.abs(y[fit] - X[fit] @ b1),
                        rcond=None)[0]
    sd = lambda z: np.maximum(design(z) @ g, 1e-3)
    sc = score / sd(x[cal])
    q2 = np.sort(sc)[k - 1]
    out["weighted conformal"] = (Xn @ b1 - q2 * sd(x_new),
                                 Xn @ b1 + q2 * sd(x_new))
    return out


R, m = 300, 500
bins = [(0, 10), (10, 20), (20, 30)]
cov = {}
for r in range(R):
    x, y = simulate(n)
    xn, yn = simulate(m)
    for k, (lo, hi) in intervals(x, y, xn).items():
        hit = (lo <= yn) & (yn <= hi)
        row = [hit.mean()] + [hit[(xn >= a) & (xn < c)].mean() for a, c in bins]
        cov.setdefault(k, []).append(row + [np.mean(hi - lo)])
print(f"Nominal 90% prediction intervals; n = {n}, {R} replications")
print(f"{'method':<20}{'overall':>8}{'x<10':>7}{'10-20':>7}{'x>20':>7}"
      f"{'length':>8}{'sd cov':>8}")
for k, v in cov.items():
    v = np.array(v)
    mc = v.mean(0)
    print(f"{k:<20}{mc[0]:8.3f}" + "".join(f"{c:7.3f}" for c in mc[1:4])
          + f"{v[:, 4].mean():8.3f}{v[:, 0].std():8.3f}")

x, y = simulate(n)
grid = np.linspace(0, 30, 200)
iv = intervals(x, y, grid)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.scatter(x, y, s=5, color="gray", alpha=0.5)
for k, c in [("normal theory", "black"), ("split conformal", "firebrick"),
             ("weighted conformal", "steelblue")]:
    for j, b in enumerate(iv[k]):
        ax.plot(grid, b, color=c, lw=1.2, label=k if j == 0 else None)
ax.set_xlabel("years of experience (simulated)")
ax.set_ylabel("log hourly earnings (simulated)")
ax.legend(frameon=False, loc="lower right")
plt.savefig("ch11_conformal.pdf", bbox_inches="tight")
