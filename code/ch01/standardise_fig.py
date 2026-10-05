# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Figure: raw versus standardised simulated survey data (same data as data_matrix).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
n = 500
# simulated household survey (illustrative only)
school = rng.integers(0, 19, n).astype(float)
hhsize = rng.poisson(4.5, n) + 1.0
dist = rng.gamma(2.0, 4.0, n)                        # km to nearest market
loginc = 1.2 + 0.09 * school - 0.04 * dist + 0.05 * hhsize \
    + rng.normal(0, 0.45, n)
df = pd.DataFrame({"log_income": loginc, "schooling": school,
                   "hh_size": hhsize, "dist_market": dist})
X = df.to_numpy()
Xc = X - X.mean(axis=0)
d = X.std(axis=0, ddof=1)
Z = Xc / d
R = np.corrcoef(X, rowvar=False)
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
axes[0].scatter(X[:, 1], X[:, 0], s=6, alpha=0.5)
axes[0].set_xlabel("schooling (years)")
axes[0].set_ylabel("log income")
axes[0].set_title("raw data", fontsize=10)
axes[1].scatter(Z[:, 1], Z[:, 0], s=6, alpha=0.5, color="C1")
axes[1].axhline(0, color="grey", lw=0.5)
axes[1].axvline(0, color="grey", lw=0.5)
axes[1].set_xlabel("schooling (standardised)")
axes[1].set_ylabel("log income (standardised)")
axes[1].set_title(f"standardised: slope of fit = r = {R[0, 1]:.3f}", fontsize=10)
plt.tight_layout()
plt.savefig("ch01_standardise.pdf", bbox_inches="tight")
