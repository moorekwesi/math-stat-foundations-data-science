# DEP 803 -- Chapter 13: Problem Set Clinic and Applied Foundations Review
# Pseudo-out-of-sample forecasting with many predictors: AR, PCA, ridge, lasso.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

rng = np.random.default_rng(803)
T, N, r = 300, 100, 3                               # months, indicators, factors


def simulate_panel(rng):
    """Simulated factor-structured panel of macro indicators and a target."""
    phi = np.array([0.8, 0.5, 0.3])
    F = np.zeros((T, r))
    for t in range(1, T):
        F[t] = phi * F[t - 1] + rng.standard_normal(r)
    Lam = rng.standard_normal((N, r))
    E = np.zeros((T, N))
    for t in range(1, T):
        E[t] = 0.3 * E[t - 1] + rng.standard_normal(N)
    X = F @ Lam.T + 1.5 * E
    y = np.zeros(T)
    for t in range(1, T):                     # target depends on lagged factors
        y[t] = (0.3 * y[t - 1] + 0.5 * F[t - 1, 0] - 0.3 * F[t - 1, 1]
                + 0.8 * rng.standard_normal())
    return X, y


def ols(Z, v):
    return np.linalg.lstsq(Z, v, rcond=None)[0]


def ridge(Z, v, lam):
    """Ridge with unpenalised intercept (Z is already standardised)."""
    zm, vm = Z.mean(0), v.mean()
    Zc = Z - zm
    b = np.linalg.solve(Zc.T @ Zc + lam * np.eye(Z.shape[1]), Zc.T @ (v - vm))
    return vm - zm @ b, b


def tune(fit_predict, grid, Z, v, n_val=36):
    """Choose a penalty on the last n_val points of the estimation window."""
    errs = [np.mean((v[-n_val:] - fit_predict(Z[:-n_val], v[:-n_val],
                                              Z[-n_val:], g))**2) for g in grid]
    return grid[int(np.argmin(errs))]


def ridge_fp(Z, v, Znew, lam):
    a, b = ridge(Z, v, lam)
    return a + Znew @ b


def lasso_fp(Z, v, Znew, alpha):
    return Lasso(alpha=alpha, max_iter=5000).fit(Z, v).predict(Znew)


X, y = simulate_panel(rng)
t0 = 150                                            # first forecast origin
lam_grid = np.logspace(0, 4, 13)
alpha_grid = np.logspace(-2.5, -0.5, 9)
fc = {m: [] for m in ["AR(1)", "PCA (DI)", "ridge", "lasso"]}
actual = []
for t in range(t0, T - 1):                          # origin t, forecast y[t+1]
    Xs = (X[:t + 1] - X[:t + 1].mean(0)) / X[:t + 1].std(0)
    ylag, ytar = y[:t], y[1:t + 1]                  # regress y[s+1] on info at s
    one = np.ones(t)
    b = ols(np.column_stack([one, ylag]), ytar)
    fc["AR(1)"].append(b[0] + b[1] * y[t])
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    Fh = U[:, :3] * S[:3]                           # PCA diffusion indexes
    b = ols(np.column_stack([one, ylag, Fh[:t]]), ytar)
    fc["PCA (DI)"].append(b @ np.r_[1.0, y[t], Fh[t]])
    Z = np.column_stack([ylag, Xs[:t]])
    znew = np.r_[y[t], Xs[t]][None, :]
    if (t - t0) % 12 == 0:                          # re-tune once a year
        lam = tune(ridge_fp, lam_grid, Z, ytar)
        alpha = tune(lasso_fp, alpha_grid, Z, ytar)
    fc["ridge"].append(ridge_fp(Z, ytar, znew, lam)[0])
    fc["lasso"].append(lasso_fp(Z, ytar, znew, alpha)[0])
    actual.append(y[t + 1])

actual = np.array(actual)
err = {m: actual - np.array(f) for m, f in fc.items()}
P = actual.size
rmse_ar = np.sqrt(np.mean(err["AR(1)"]**2))


def dm_stat(d, lags=4):
    """Diebold-Mariano statistic with a Newey-West long-run variance."""
    dc = d - d.mean()
    lrv = dc @ dc / P
    for j in range(1, lags + 1):
        lrv += 2 * (1 - j / (lags + 1)) * (dc[j:] @ dc[:-j]) / P
    return d.mean() / np.sqrt(lrv / P)


def block_boot_ratio(e1, e0, B=2000, L=8):
    """Moving-block bootstrap interval for RMSE(model) / RMSE(AR)."""
    starts = rng.integers(0, P - L + 1, size=(B, int(np.ceil(P / L))))
    idx = (starts[:, :, None] + np.arange(L)).reshape(B, -1)[:, :P]
    ratio = np.sqrt(np.mean(e1[idx]**2, 1) / np.mean(e0[idx]**2, 1))
    return np.quantile(ratio, [0.05, 0.95])


print(f"T = {T}, N = {N} indicators, {P} expanding-window forecasts (h = 1)")
print(f"{'model':<10}{'RMSE':>8}{'rel. AR':>9}{'DM':>7}{'90% block-boot CI':>21}")
for m, e in err.items():
    rm = np.sqrt(np.mean(e**2))
    if m == "AR(1)":
        print(f"{m:<10}{rm:8.3f}{1:9.3f}")
        continue
    lo, hi = block_boot_ratio(e, err["AR(1)"])
    dm = dm_stat(err["AR(1)"]**2 - e**2)
    print(f"{m:<10}{rm:8.3f}{rm / rmse_ar:9.3f}{dm:7.2f}      [{lo:.3f}, {hi:.3f}]")
print(f"last tuned ridge lambda = {lam:.0f}, lasso alpha = {alpha:.4f}")
print("DM > 0 means the model beats AR(1); compare with N(0,1) quantiles")

fig, ax = plt.subplots(figsize=(6, 3.8))
for m in ["PCA (DI)", "ridge", "lasso"]:
    ax.plot(np.arange(t0 + 1, T), np.cumsum(err["AR(1)"]**2 - err[m]**2), label=m)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("forecast target month")
ax.set_ylabel("cumulative SE(AR) - SE(model)")
ax.set_title("Rising line = model outperforming the AR(1) benchmark")
ax.legend()
plt.savefig("ch13_forecast_csfe.pdf", bbox_inches="tight")
