# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Ridge and lasso regularisation paths on the (standardised) diabetes data.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lasso_path

X, y = load_diabetes(return_X_y=True)
names = load_diabetes().feature_names
X = (X - X.mean(0)) / X.std(0)            # standardise: penalty is scale-aware
y = y - y.mean()
n, p = X.shape

# ridge path via one SVD:  b(lambda) = V diag(d/(d^2+lambda)) U'y
U, d, Vt = np.linalg.svd(X, full_matrices=False)
lams = np.logspace(-2, 5, 100)
ridge = np.array([Vt.T @ (d / (d**2 + l) * (U.T @ y)) for l in lams])
edf = np.array([np.sum(d**2 / (d**2 + l)) for l in lams])

# lasso path (sklearn coordinate descent with warm starts)
alphas, coefs, _ = lasso_path(X, y, n_alphas=200, eps=1e-4)
coefs = coefs.T                            # (n_alphas, p), alphas decreasing
entry = [np.argmax(np.abs(coefs[:, j]) > 0) for j in range(p)]
order = np.argsort(entry)
print("order in which variables enter the lasso path:")
for j in order:
    print(f"  {names[j]:>4} enters at alpha = {alphas[entry[j]]:8.3f}")
print(f"ridge edf at lambda = 1e-2: {edf[0]:.2f};  at 1e5: {edf[-1]:.3f}")
print("ridge: any exact zeros along the path?", bool(np.any(ridge == 0)))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
ax[0].semilogx(lams, ridge)
ax[0].set_xlabel(r"ridge penalty $\lambda$")
ax[0].set_ylabel("coefficient (standardised X)")
ax[0].set_title("Ridge: smooth shrinkage")
ax[1].semilogx(alphas, coefs)
ax[1].set_xlabel(r"lasso penalty $\alpha$")
ax[1].set_title("Lasso: shrinkage and selection")
for j in order[:4]:
    ax[1].annotate(names[j], (alphas[-1], coefs[-1, j]), fontsize=8,
                   xytext=(-18, 0), textcoords="offset points")
for a in ax:
    a.grid(alpha=0.3)
    a.axhline(0, color="k", lw=0.6)
plt.tight_layout()
plt.savefig("ch12_reg_paths.pdf", bbox_inches="tight")
