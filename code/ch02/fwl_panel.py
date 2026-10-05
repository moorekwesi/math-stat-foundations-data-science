# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Frisch-Waugh-Lovell: fixed effects by dummies (LSDV) = within (demeaning)
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
N, T, G, beta = 400, 5, 16, 0.10           # workers, years, regions, true slope
region = rng.integers(0, G, size=N)
mu_region = rng.normal(0, 0.5, size=G)     # regional wage levels (simulated)
alpha = mu_region[region] + rng.normal(0, 0.3, size=N)   # worker effects
ids = np.repeat(np.arange(N), T)
year = np.tile(np.arange(T), N)
# training hours are higher where wages are high: correlated with alpha
x = 2.0 * alpha[ids] + rng.normal(0, 1, size=N * T)
y = beta * x + alpha[ids] + 0.03 * year + rng.normal(0, 0.2, size=N * T)

Dw = np.eye(N)[ids]                         # N*T x N worker dummies
Dt = np.eye(T)[year][:, 1:]                 # year dummies (base year dropped)
ols = lambda X, v: np.linalg.lstsq(X, v, rcond=None)[0]

b_pool = ols(np.column_stack([np.ones(N * T), x, Dt]), y)[1]
X_lsdv = np.column_stack([x, Dt, Dw])
coef_lsdv = ols(X_lsdv, y)
e_lsdv = y - X_lsdv @ coef_lsdv


def demean(v, g):
    """Subtract group means: this is M_D v for the group dummy matrix D."""
    means = np.bincount(g, weights=v) / np.bincount(g)
    return v - means[g]


yw, xw = demean(y, ids), demean(x, ids)
Dtw = np.column_stack([demean(c, ids) for c in Dt.T])
coef_w = ols(np.column_stack([xw, Dtw]), yw)
e_w = yw - np.column_stack([xw, Dtw]) @ coef_w
b_reg = ols(np.column_stack([demean(x, region[ids]), Dt]),
            demean(y, region[ids]))[0]

print(f"design sizes: LSDV {X_lsdv.shape},  within {(N * T, 1 + T - 1)}")
print(f"true beta                      = {beta:.4f}")
print(f"pooled OLS (ignores alpha_i)   = {b_pool:.4f}")
print(f"region fixed effects only      = {b_reg:.4f}")
print(f"LSDV with {N} worker dummies   = {coef_lsdv[0]:.6f}")
print(f"within (demeaned, FWL)         = {coef_w[0]:.6f}")
print(f"max |e_LSDV - e_within|        = {np.abs(e_lsdv - e_w).max():.1e}")
print(f"year effects equal?            {np.allclose(coef_lsdv[1:T], coef_w[1:])}")
dof = N * T - N - (T - 1) - 1
print(f"residual dof: correct = {dof}, naive within = {N * T - T}")

# Figure: pooled scatter vs within scatter (subsample of 6 regions)
sel = np.isin(region[ids], np.arange(6))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=False)
for g in range(6):
    m = sel & (region[ids] == g)
    ax[0].scatter(x[m], y[m], s=5, alpha=0.5)
    ax[1].scatter(xw[m], yw[m], s=5, alpha=0.5)
for a, (xx, yy, b, ttl) in zip(ax, [(x, y, b_pool, "pooled OLS"),
                                    (xw, yw, coef_w[0], "within (demeaned)")]):
    grid = np.linspace(xx[sel].min(), xx[sel].max(), 2)
    c0 = yy[sel].mean() - b * xx[sel].mean()
    a.plot(grid, c0 + b * grid, "k-", lw=2)
    a.set_title(f"{ttl}: slope = {b:.3f}", fontsize=10)
    a.set_xlabel("training hours (simulated)")
ax[0].set_ylabel("log wage (simulated)")
fig.tight_layout()
fig.savefig("ch02_fwl_panel.pdf", bbox_inches="tight")
