# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Regression on a treatment dummy = difference of conditional means (ATE intuition).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n = 20_000
# Potential outcomes for a simulated cash-transfer programme
ability = rng.normal(size=n)
y0 = 50 + 8 * ability + rng.normal(0, 5, n)
y1 = y0 + 6 + 2 * ability                       # heterogeneous effect
print(f"true ATE = E[Y1 - Y0] = {np.mean(y1 - y0):.3f}")


def ols_dummy(d, y):
    Z = np.column_stack([np.ones(len(d)), d])
    return np.linalg.lstsq(Z, y, rcond=None)[0][1]


# (a) Randomised assignment: D independent of (Y0, Y1)
d = rng.integers(0, 2, n)
y = np.where(d == 1, y1, y0)
print("\nRandomised assignment")
print(f"  E[Y|D=1] - E[Y|D=0] = {y[d == 1].mean() - y[d == 0].mean():.3f}")
print(f"  OLS slope on D      = {ols_dummy(d, y):.3f}")

# (b) Self-selection: high-ability people enrol more often
d = (ability + rng.normal(size=n) > 0.5).astype(int)
y = np.where(d == 1, y1, y0)
naive = y[d == 1].mean() - y[d == 0].mean()
att = np.mean((y1 - y0)[d == 1])
sel = y0[d == 1].mean() - y0[d == 0].mean()
print("\nSelf-selection on ability")
print(f"  naive difference    = {naive:.3f}")
print(f"  = ATT {att:.3f} + selection bias {sel:.3f}")
print(f"  (ATT exceeds ATE because gains rise with ability)")
