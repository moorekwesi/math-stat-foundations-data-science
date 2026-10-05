# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Pseudo-out-of-sample forecasting with many predictors: OLS, ridge, lasso, PCR.
# (c) 2026 Stephen E. Moore. MIT Licence.
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import (Lasso, LassoCV, LinearRegression, Ridge,
                                  RidgeCV)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
rng = np.random.default_rng(803)
# ---- simulated monthly macro panel: 3 AR(1) factors drive 100 indicators
T, N, r = 300, 100, 3
F = np.zeros((T, r))
for t in range(1, T):
    F[t] = np.array([0.8, 0.6, 0.4]) * F[t - 1] + rng.normal(size=r)
Lam = rng.normal(size=(N, r))
Xall = F @ Lam.T + 1.0 * rng.normal(size=(T, N))      # indicators
# target ("inflation", h = 1 month ahead) depends on factors + 2 indicators
y = np.zeros(T)
y[1:] = (F[:-1] @ np.array([0.6, -0.4, 0.3]) + 0.3 * Xall[:-1, 0]
         + 0.2 * Xall[:-1, 1] + 0.8 * rng.normal(size=T - 1))
# predictors dated t: indicators and own lag; target dated t + 1
Z = np.column_stack([Xall[:-1], y[:-1]])
target = y[1:]

cv = KFold(5, shuffle=True, random_state=803)
ols = lambda: LinearRegression()
pcr = lambda: make_pipeline(StandardScaler(), PCA(3), LinearRegression())
ridge_cv = lambda: make_pipeline(StandardScaler(),
                                 RidgeCV(alphas=np.logspace(-1, 4, 30)))
lasso_cv = lambda: make_pipeline(StandardScaler(), LassoCV(
    n_alphas=25, cv=cv, max_iter=5000, tol=1e-3))
start, retune = 120, 12                 # first origin; re-tune yearly
err = {m: [] for m in ["AR(1)", "OLS-all", "Ridge", "Lasso", "PCR(3)"]}
for t in range(start, len(target)):     # expanding window: data up to t
    Zt, yt, znew = Z[:t], target[:t], Z[t:t + 1]
    if (t - start) % retune == 0:       # cross-validate the penalties
        lam_r = ridge_cv().fit(Zt, yt)[-1].alpha_
        lam_l = lasso_cv().fit(Zt, yt)[-1].alpha_
    fits = {"AR(1)": ols().fit(Zt[:, [-1]], yt).predict(znew[:, [-1]]),
            "OLS-all": ols().fit(Zt, yt).predict(znew),
            "Ridge": make_pipeline(StandardScaler(), Ridge(lam_r))
            .fit(Zt, yt).predict(znew),
            "Lasso": make_pipeline(StandardScaler(), Lasso(lam_l))
            .fit(Zt, yt).predict(znew),
            "PCR(3)": pcr().fit(Zt, yt).predict(znew)}
    for m, f_hat in fits.items():
        err[m].append(target[t] - f_hat[0])
err = {m: np.array(e) for m, e in err.items()}
rmse_ar = np.sqrt(np.mean(err["AR(1)"] ** 2))
print(f"{len(err['AR(1)'])} one-step forecasts, origins t = {start}..."
      f"{len(target) - 1}, p = {Z.shape[1]} predictors")
print(f"last tuned penalties: ridge {lam_r:.2f}, lasso {lam_l:.4f}")
print(f"{'model':<9}{'RMSE':>8}{'rel. to AR':>12}")
for m, e in err.items():
    rm = np.sqrt(np.mean(e ** 2))
    print(f"{m:<9}{rm:>8.3f}{rm / rmse_ar:>12.3f}")

plt.figure(figsize=(6, 3.8))
for m in ["OLS-all", "Ridge", "Lasso", "PCR(3)"]:
    plt.plot(np.cumsum(err["AR(1)"] ** 2 - err[m] ** 2), label=m)
plt.axhline(0, color="k", lw=0.6)
plt.xlabel("forecast origin (months after start of evaluation)")
plt.ylabel("cumulative SSE(AR) - SSE(model)")
plt.title("Rising line = model beats the AR(1) benchmark")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.savefig("ch12_forecast_poos.pdf", bbox_inches="tight")
