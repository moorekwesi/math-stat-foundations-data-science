# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Pseudo-out-of-sample forecast evaluation and a bootstrap fan chart.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
T, T0, B, H = 300, 150, 499, 12

# Simulated monthly inflation-like series driven by a persistent indicator
z = np.zeros(T)
infl = np.zeros(T)
for t in range(2, T):
    z[t] = 0.8 * z[t - 1] + rng.normal(0, 1)
    infl[t] = (0.8 + 0.5 * infl[t - 1] + 0.2 * infl[t - 2] + 0.15 * z[t - 1]
               + rng.normal(0, 0.5) * (1 + 0.5 * (t > 220)))  # volatility jump


def lagmat(y, x, t_end):
    """Regressors for y_t on (1, y_{t-1}, y_{t-2}[, x_{t-1}]), t < t_end."""
    rows = np.arange(2, t_end)
    Z = np.column_stack([np.ones(rows.size), y[rows - 1], y[rows - 2]])
    if x is not None:
        Z = np.column_stack([Z, x[rows - 1]])
    return Z, y[rows]


models = {"historical mean": None, "random walk": None, "AR(2)": None,
          "AR(2) + indicator": z}
err = {k: [] for k in models}
hits80 = []
for t in range(T0, T):                       # forecast infl[t] with data < t
    err["historical mean"].append(infl[t] - infl[:t].mean())
    err["random walk"].append(infl[t] - infl[t - 1])
    for k in ("AR(2)", "AR(2) + indicator"):
        Z, yy = lagmat(infl, models[k], t)
        b, *_ = np.linalg.lstsq(Z, yy, rcond=None)
        zt = np.r_[1, infl[t - 1], infl[t - 2]]
        if models[k] is not None:
            zt = np.r_[zt, z[t - 1]]
        err[k].append(infl[t] - zt @ b)
        if k == "AR(2)":                     # bootstrap 80% interval, h = 1
            u = yy - Z @ b
            q = np.quantile(zt @ b + u[rng.integers(0, u.size, B)], [0.1, 0.9])
            hits80.append(q[0] <= infl[t] <= q[1])
rmse_ar = np.sqrt(np.mean(np.square(err["AR(2)"])))
print(f"Pseudo-out-of-sample, expanding window: {T - T0} one-step forecasts")
print(f"{'model':<20}{'RMSE':>8}{'relative to AR(2)':>19}")
for k, e in err.items():
    rm = np.sqrt(np.mean(np.square(e)))
    print(f"{k:<20}{rm:8.3f}{rm / rmse_ar:19.3f}")
# Diebold-Mariano (h = 1, squared loss): AR(2)+indicator vs AR(2)
dl = np.square(err["AR(2) + indicator"]) - np.square(err["AR(2)"])
dm = dl.mean() / (dl.std(ddof=1) / np.sqrt(dl.size))
print(f"Diebold-Mariano statistic (indicator vs AR(2)) = {dm:.2f}")
hits80 = np.array(hits80)
print(f"bootstrap 80% interval coverage: before t=220 {hits80[:70].mean():.3f},"
      f" after {hits80[70:].mean():.3f}")

# Fan chart: residual bootstrap of AR(2) with parameter re-estimation
Z, yy = lagmat(infl, None, T)
b, *_ = np.linalg.lstsq(Z, yy, rcond=None)
u = yy - Z @ b
paths = np.empty((B, H))
for j in range(B):
    ys = infl.copy()                          # recursive bootstrap sample
    us = u[rng.integers(0, u.size, u.size)]
    for t in range(2, T):
        ys[t] = b[0] + b[1] * ys[t - 1] + b[2] * ys[t - 2] + us[t - 2]
    Zs, ys2 = lagmat(ys, None, T)
    bs, *_ = np.linalg.lstsq(Zs, ys2, rcond=None)
    hist = [infl[-2], infl[-1]]
    for h in range(H):
        hist.append(bs @ [1, hist[-1], hist[-2]] + rng.choice(u))
    paths[j] = hist[2:]
qs = np.quantile(paths, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], axis=0)
print("fan chart median path (h=1,6,12):",
      " ".join(f"{qs[3, h]:.2f}" for h in (0, 5, 11)),
      "| 90% band at h=12:", f"[{qs[0, -1]:.2f}, {qs[-1, -1]:.2f}]")

fig, ax = plt.subplots(figsize=(6, 3.8))
past = np.arange(T - 48, T)
fut = np.arange(T - 1, T + H)
ax.plot(past, infl[past], color="black", lw=1.2, label="observed (simulated)")
for (lo, hi), a in zip([(0, 6), (1, 5), (2, 4)], (0.2, 0.3, 0.45)):
    ax.fill_between(fut, np.r_[infl[-1], qs[lo]], np.r_[infl[-1], qs[hi]],
                    color="firebrick", alpha=a, lw=0)
ax.plot(fut, np.r_[infl[-1], qs[3]], color="firebrick", lw=1.2,
        label="median and 50/80/90% bands")
ax.set_xlabel("month")
ax.set_ylabel("inflation rate (simulated, %)")
ax.legend(frameon=False, loc="upper center", ncol=2,
          bbox_to_anchor=(0.5, -0.18))
plt.savefig("ch11_fanchart.pdf", bbox_inches="tight")
