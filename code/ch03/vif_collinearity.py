# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# Multicollinearity among simulated policy covariates: VIF, eigenvalues, ridge.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
names = ["educ_spend", "health_spend", "road_density", "pupil_teacher"]


def simulate(n, rho, rng):
    """Simulated district covariates: education and health spending share a
    common budget component; correlation between them approximately rho."""
    budget = rng.standard_normal(n)
    e1 = np.sqrt(rho) * budget + np.sqrt(1 - rho) * rng.standard_normal(n)
    e2 = np.sqrt(rho) * budget + np.sqrt(1 - rho) * rng.standard_normal(n)
    road = 0.3 * budget + rng.standard_normal(n)
    ptr = -0.2 * e1 + rng.standard_normal(n)
    return np.c_[e1, e2, road, ptr]


n = 200
X = simulate(n, 0.95, rng)
R = np.corrcoef(X, rowvar=False)
vif = np.diag(np.linalg.inv(R))
# VIF_j = 1 / (1 - R_j^2) from the auxiliary regression of x_j on the others
Xs = (X - X.mean(0)) / X.std(0)
for j, name in enumerate(names):
    others = np.c_[np.ones(n), np.delete(Xs, j, axis=1)]
    g, *_ = np.linalg.lstsq(others, Xs[:, j], rcond=None)
    r2 = 1 - np.var(Xs[:, j] - others @ g) / np.var(Xs[:, j])
    print(f"{name:13s} VIF = {vif[j]:6.2f}   1/(1-R^2) = {1 / (1 - r2):6.2f}")
lam, V = np.linalg.eigh(R)
print("eigenvalues of R :", np.round(lam, 4))
print("sum VIF = sum 1/lambda:", round(vif.sum(), 3), round((1 / lam).sum(), 3))
print("condition indices sqrt(lmax/l):", np.round(np.sqrt(lam[-1] / lam), 2))
print("smallest eigvec  :", np.round(V[:, 0], 3))

# Sampling spread of OLS vs ridge coefficients as collinearity rises
beta = np.array([1.0, 1.0, 0.5, -0.5])
rhos = np.array([0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99])
k_ridge = 10.0
sd_ols, sd_rdg = [], []
for rho in rhos:
    b_o, b_r = [], []
    for rep in range(500):
        Xr = simulate(n, rho, rng)
        y = Xr @ beta + 2.0 * rng.standard_normal(n)
        G = Xr.T @ Xr
        b_o.append(np.linalg.solve(G, Xr.T @ y))
        b_r.append(np.linalg.solve(G + k_ridge * np.eye(4), Xr.T @ y))
    sd_ols.append(np.std(np.array(b_o)[:, 0]))
    sd_rdg.append(np.std(np.array(b_r)[:, 0]))
print(" rho   sd(OLS b1)  sd(ridge b1)  1/sqrt(1-rho^2)")
for r_, so, sr in zip(rhos, sd_ols, sd_rdg):
    print(f"{r_:4.2f}   {so:8.3f}    {sr:8.3f}      {1 / np.sqrt(1 - r_ ** 2):6.2f}")
G = X.T @ X
print(f"cond(X'X) = {np.linalg.cond(G):.1f};"
      f" cond(X'X + {k_ridge:.0f} I) = "
      f"{np.linalg.cond(G + k_ridge * np.eye(4)):.1f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(rhos, sd_ols, "o-", label="OLS")
ax.plot(rhos, sd_rdg, "s-", label=f"ridge (k = {k_ridge:.0f})")
ax.set_xlabel("correlation between education and health spending")
ax.set_ylabel("sampling sd of coefficient on educ_spend")
ax.set_title("Multicollinearity inflates variance; ridge stabilises")
ax.legend(frameon=False)
plt.savefig("ch03_vif_ridge.pdf", bbox_inches="tight")
