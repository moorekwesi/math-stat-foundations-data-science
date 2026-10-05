# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# PCA of a simulated macro panel; scree plot; diffusion-index inflation forecast.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
T, N, r = 240, 40, 2                    # 20 years of monthly data, 40 series
# Two latent AR(1) factors: "real activity" and "price pressure"
F = np.zeros((T, r))
for t in range(1, T):
    F[t] = np.array([0.8, 0.9]) * F[t - 1] + rng.standard_normal(r)
Lam = np.zeros((N, r))
Lam[:25, 0] = rng.uniform(0.5, 1.5, 25)            # activity block
Lam[25:, 1] = rng.uniform(0.5, 1.5, 15)            # price block
Lam[20:30] += rng.uniform(-0.4, 0.4, (10, r))      # some mixed series
scale = rng.uniform(0.5, 20.0, N)                  # different units
X = (F @ Lam.T + rng.standard_normal((T, N)) * 1.2) * scale

# PCA via the SVD of the standardised panel
Z = (X - X.mean(0)) / X.std(0, ddof=1)
U, s, Vt = np.linalg.svd(Z, full_matrices=False)
evr = s ** 2 / np.sum(s ** 2)
print("explained variance ratio (first 6):", np.round(evr[:6], 3))
print("cumulative (first 3)              :", np.round(np.cumsum(evr)[:3], 3))
Fhat = U[:, :r] * s[:r] / np.sqrt(T - 1)            # principal component scores
for j in range(r):
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(T), Fhat], F[:, j], rcond=None)
    res = F[:, j] - np.c_[np.ones(T), Fhat] @ beta
    r2 = 1 - res.var() / F[:, j].var()
    print(f"R^2 of true factor {j + 1} on 2 PCs: {r2:.3f}")

# Without standardisation the high-variance series dominate PC1
Xc = X - X.mean(0)
_, s_raw, Vt_raw = np.linalg.svd(Xc, full_matrices=False)
top = np.argsort(-np.abs(Vt_raw[0]))[:3]
print("unstandardised PC1 top series:", top, "scales:", np.round(scale[top], 1))
print("unstandardised EVR(PC1): %.3f" % (s_raw[0] ** 2 / np.sum(s_raw ** 2)))

# Diffusion-index forecast of an inflation series (series 35), h = 1
y = Z[:, 35]
start = 120
err_ar, err_di = [], []
for t in range(start, T - 1):
    U_t, s_t, _ = np.linalg.svd(Z[:t + 1], full_matrices=False)
    f_t = U_t[:, :r] * s_t[:r]                     # factors estimated in real time
    Xar = np.c_[np.ones(t), y[:t]]
    Xdi = np.c_[Xar, f_t[:t]]
    b_ar, *_ = np.linalg.lstsq(Xar, y[1:t + 1], rcond=None)
    b_di, *_ = np.linalg.lstsq(Xdi, y[1:t + 1], rcond=None)
    err_ar.append(y[t + 1] - np.r_[1, y[t]] @ b_ar)
    err_di.append(y[t + 1] - np.r_[1, y[t], f_t[t]] @ b_di)
rmse_ar = np.sqrt(np.mean(np.square(err_ar)))
rmse_di = np.sqrt(np.mean(np.square(err_di)))
print(f"out-of-sample RMSE: AR(1) {rmse_ar:.3f}, AR + 2 factors {rmse_di:.3f}"
      f", ratio {rmse_di / rmse_ar:.3f}")

fig, ax = plt.subplots(figsize=(6, 3.6))
k = np.arange(1, 16)
ax.bar(k, evr[:15], color="C0", alpha=0.7, label="share of variance")
ax.plot(k, np.cumsum(evr)[:15], "o-", color="C3", label="cumulative")
ax.set_xlabel("principal component")
ax.set_ylabel("explained variance ratio")
ax.set_title("Scree plot: simulated panel of 40 macro indicators")
ax.legend(frameon=False)
plt.savefig("ch03_scree_macro.pdf", bbox_inches="tight")
