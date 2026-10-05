# DEP 803 -- Chapter 13: Problem Set Clinic and Applied Foundations Review
# Pairs and wild bootstrap confidence intervals for the return to schooling.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt


def simulate_lfs(rng, n=4000, n_reg=10):
    """Same simulated labour force survey as ch13_mincer_ols.py."""
    region = rng.integers(0, n_reg, n)
    reg_fe = np.linspace(-0.3, 0.3, n_reg)
    reg_edu = np.linspace(-2.0, 2.0, n_reg)
    female = (rng.random(n) < 0.5).astype(float)
    age = rng.integers(18, 61, n)
    educ = np.clip(np.round(9 + reg_edu[region] + 3.5 * rng.standard_normal(n)),
                   0, 20)
    exper = np.maximum(age - educ - 6, 0)
    sd = 0.10 + 0.045 * educ
    logw = (1.0 + reg_fe[region] + 0.10 * educ + 0.035 * exper
            - 0.0006 * exper**2 - 0.12 * female + sd * rng.standard_normal(n))
    return logw, educ, exper, female, region


rng = np.random.default_rng(803)
logw, educ, exper, female, region = simulate_lfs(rng)
n = logw.size
D = (region[:, None] == np.arange(1, 10)).astype(float)
X = np.column_stack([np.ones(n), educ, exper, exper**2, female, D])
k = X.shape[1]
Q, R = np.linalg.qr(X)
beta = np.linalg.solve(R, Q.T @ logw)
e = logw - X @ beta
Rinv = np.linalg.inv(R)
XtX_inv = Rinv @ Rinv.T
V = n / (n - k) * XtX_inv @ ((X * e[:, None]**2).T @ X) @ XtX_inv
se_hc1 = np.sqrt(V[1, 1])
# row 1 of (X'X)^{-1} X' maps any outcome vector to its educ coefficient
a1 = XtX_inv[1] @ X.T

B = 1999
b_pairs = np.empty(B)
b_wild = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n, n)                       # resample workers
    Xb, yb = X[idx], logw[idx]
    b_pairs[b] = np.linalg.lstsq(Xb, yb, rcond=None)[0][1]
    v = rng.choice([-1.0, 1.0], n)                    # Rademacher weights
    b_wild[b] = beta[1] + a1 @ (e * v)                # fixed design, y* = Xb + e v

z = 1.959964
print(f"beta_educ = {beta[1]:.5f}, HC1 se = {se_hc1:.5f}")
print(f"{'method':<22}{'se':>9}{'95% lower':>11}{'95% upper':>11}")
print(f"{'normal, HC1':<22}{se_hc1:9.5f}{beta[1] - z * se_hc1:11.5f}"
      f"{beta[1] + z * se_hc1:11.5f}")
for name, bs in [("pairs, percentile", b_pairs), ("wild, percentile", b_wild)]:
    lo, hi = np.quantile(bs, [0.025, 0.975])
    print(f"{name:<22}{bs.std(ddof=1):9.5f}{lo:11.5f}{hi:11.5f}")
lo, hi = np.quantile(b_pairs, [0.025, 0.975])
print(f"{'pairs, basic':<22}{'':9}{2 * beta[1] - hi:11.5f}{2 * beta[1] - lo:11.5f}")
pct = 100 * (np.exp(np.quantile(b_pairs, [0.025, 0.975])) - 1)
print(f"implied % wage gain per year (pairs): [{pct[0]:.2f}%, {pct[1]:.2f}%]")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(b_pairs, bins=50, density=True, alpha=0.55, label="pairs bootstrap")
ax.hist(b_wild, bins=50, density=True, alpha=0.55, label="wild bootstrap")
grid = np.linspace(b_pairs.min(), b_pairs.max(), 200)
ax.plot(grid, np.exp(-0.5 * ((grid - beta[1]) / se_hc1)**2)
        / (se_hc1 * np.sqrt(2 * np.pi)), "k-", lw=1.5, label="Normal(b, HC1 se)")
ax.axvline(beta[1], color="k", ls="--", lw=1)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.set_xlabel("estimated return to one year of schooling")
ax.set_ylabel("density")
ax.legend(fontsize=9)
plt.savefig("ch13_mincer_boot.pdf", bbox_inches="tight")
