# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Joint, marginal and conditional PMFs: household size and poverty status.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import pandas as pd

rng = np.random.default_rng(803)
sizes = ["1-2", "3-4", "5-6", "7+"]
# Illustrative joint PMF p(x, y): rows = size class, columns = (non-poor, poor)
P = np.array([[0.150, 0.020],
              [0.260, 0.060],
              [0.200, 0.110],
              [0.090, 0.110]])
assert np.isclose(P.sum(), 1.0)
joint = pd.DataFrame(P, index=sizes, columns=["non-poor", "poor"])
px, py = P.sum(axis=1), P.sum(axis=0)        # marginals
print("Joint PMF with marginals")
tab = joint.copy()
tab["P(size)"] = px
print(tab.round(3).to_string())
print(f"P(poor) = {py[1]:.3f}\n")

cond = pd.DataFrame({"P(poor | size)": P[:, 1] / px,
                     "P(size | poor)": P[:, 1] / py[1]}, index=sizes)
print(cond.round(3).to_string())
indep = np.outer(px, py)
print(f"\nmax |p(x,y) - p(x)p(y)| = {np.max(abs(P - indep)):.3f}  -> dependent")

# Sampling from the joint PMF and recovering it by cross-tabulation
n = 50_000
cells = rng.choice(P.size, size=n, p=P.ravel())
size_i, poor_i = np.unravel_index(cells, P.shape)
ct = pd.crosstab(np.array(sizes)[size_i], poor_i, normalize=True)
print(f"max |empirical - true| over 8 cells (n = {n}) = "
      f"{np.max(abs(ct.values - P)):.4f}")
