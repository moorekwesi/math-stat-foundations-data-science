# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Cluster bootstrap for a simulated survey clustered by enumeration area.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
G, m, B = 40, 15, 1999           # enumeration areas, households per EA
tau, s_ea, s_hh = 0.10, 0.25, 0.50


def survey():
    """Log consumption; a programme is assigned to half the EAs."""
    d = np.repeat(rng.permutation(np.r_[np.ones(G // 2), np.zeros(G // 2)]), m)
    ea = np.repeat(rng.normal(0, s_ea, G), m)            # shared EA shock
    return d, 5.0 + tau * d + ea + rng.normal(0, s_hh, G * m)


def effect(d, y):
    """Difference in means = OLS coefficient on d (last axis = households)."""
    return (np.sum(d * y, -1) / np.sum(d, -1)
            - np.sum((1 - d) * y, -1) / np.sum(1 - d, -1))


d, y = survey()
n = G * m
b = effect(d, y)
u = y - np.where(d == 1, y[d == 1].mean(), y[d == 0].mean())
# naive iid standard error (two-sample formula)
se_iid = np.sqrt(y[d == 1].var(ddof=1) / np.sum(d)
                 + y[d == 0].var(ddof=1) / np.sum(1 - d))
# cluster-robust (CR1) sandwich: sum scores within each EA
X = np.column_stack([np.ones(n), d])
XtXi = np.linalg.inv(X.T @ X)
scores = (X * u[:, None]).reshape(G, m, 2).sum(axis=1)
V = XtXi @ (scores.T @ scores) @ XtXi * G / (G - 1) * (n - 1) / (n - 2)
se_cr1 = np.sqrt(V[1, 1])
# pairs cluster bootstrap: resample whole EAs with replacement
gidx = rng.integers(0, G, size=(B, G))
rows = (gidx[:, :, None] * m + np.arange(m)).reshape(B, n)
with np.errstate(invalid="ignore", divide="ignore"):
    bb = effect(d[rows], y[rows])
ok = np.isfinite(bb)              # drop rare draws with no treated EA
se_cboot = bb[ok].std(ddof=1)
idx = rng.integers(0, n, size=(B, n))   # ignores clustering: wrong
se_hhboot = effect(d[idx], y[idx]).std(ddof=1)

# intra-cluster correlation (ANOVA estimator) and Monte Carlo truth
ybar_g = u.reshape(G, m).mean(1)
msb = m * np.sum(ybar_g**2) / (G - 2)
msw = np.sum((u.reshape(G, m) - ybar_g[:, None])**2) / (G * (m - 1))
icc = (msb - msw) / (msb + (m - 1) * msw)
sims = np.array([effect(*survey()) for _ in range(4000)])
print(f"G = {G} EAs x m = {m} households; estimated effect = {b:.4f}")
print(f"estimated ICC = {icc:.3f}; Moulton factor 1+(m-1)ICC = "
      f"{1 + (m - 1) * icc:.2f}")
print(f"{'standard error':<30}{'value':>8}")
for k, v in [("naive iid formula", se_iid),
             ("household bootstrap (wrong)", se_hhboot),
             ("cluster-robust CR1", se_cr1),
             ("cluster (EA) bootstrap", se_cboot),
             ("Monte Carlo truth", sims.std(ddof=1))]:
    print(f"{k:<30}{v:8.4f}")
print(f"cluster draws dropped (no treated EA): {np.sum(~ok)}")
