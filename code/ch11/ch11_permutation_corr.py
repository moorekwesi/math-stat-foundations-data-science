# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Permutation test of a correlation, by hand and with scipy (pairings).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)
n, M = 24, 20000
# Simulated districts: health spending per head (skewed) and an outcome
spend = rng.lognormal(3.0, 0.7, n)
outcome = 50 + 0.1 * spend + rng.normal(0, 5, n)
r_obs = np.corrcoef(spend, outcome)[0, 1]

# By hand: permute one variable, keep the other fixed
perm = np.argsort(rng.random((M, n)), axis=1)
yp = outcome[perm]
xc = spend - spend.mean()
yc = yp - yp.mean(axis=1, keepdims=True)
r_perm = (yc @ xc) / np.sqrt(np.sum(yc**2, 1) * np.sum(xc**2))
p_hand = (1 + np.sum(np.abs(r_perm) >= abs(r_obs))) / (M + 1)

# scipy: permutation_type="pairings" permutes the pairing of x and y
res = stats.permutation_test((spend,), lambda x: np.corrcoef(x, outcome)[0, 1],
                             permutation_type="pairings", n_resamples=M,
                             random_state=rng)
p_pearson = stats.pearsonr(spend, outcome)[1]
rho, p_spear = stats.spearmanr(spend, outcome)
print(f"n = {n} districts; Pearson r = {r_obs:.4f}, Spearman rho = {rho:.4f}")
print(f"permutation p (by hand, M = {M})     = {p_hand:.4f}")
print(f"permutation p (scipy, pairings)      = {res.pvalue:.4f}")
print(f"normal-theory t-test p (pearsonr)    = {p_pearson:.4f}")
print(f"Spearman rank-correlation p          = {p_spear:.4f}")
print(f"sd of permutation r = {r_perm.std():.4f}; 1/sqrt(n-1) = "
      f"{1 / np.sqrt(n - 1):.4f}")
