# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Multivariate change of variables: total and share of two exponential incomes.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)
lam, n = 0.5, 200_000
X1 = rng.exponential(1 / lam, n)             # income of earner 1
X2 = rng.exponential(1 / lam, n)             # income of earner 2
T = X1 + X2                                  # household total
S = X1 / T                                   # share of earner 1

# Jacobian method: (T, S) has density lam^2 t exp(-lam t) * 1 on (0,inf)x(0,1)
# so T ~ Gamma(2, 1/lam), S ~ Uniform(0, 1) and T, S are independent.
print("Kolmogorov-Smirnov p-values")
p_T = stats.kstest(T, "gamma", (2, 0, 2)).pvalue
print(f"  T against Gamma(2, scale=2): {p_T:.3f}")
print(f"  S against Uniform(0, 1)    : {stats.kstest(S, 'uniform').pvalue:.3f}")
print(f"\ncorr(T, S)   = {np.corrcoef(T, S)[0, 1]: .4f}")
print(f"corr(T, S^2) = {np.corrcoef(T, S**2)[0, 1]: .4f}")
# Independence check on rectangles: P(T<=t, S<=s) vs P(T<=t)P(S<=s)
print(f"\n{'t':>4} {'s':>5} {'joint':>8} {'product':>8}")
for t, s in [(2, 0.25), (4, 0.5), (8, 0.9)]:
    joint = np.mean((T <= t) & (S <= s))
    print(f"{t:>4} {s:5.2f} {joint:8.4f} {np.mean(T <= t) * np.mean(S <= s):8.4f}")
