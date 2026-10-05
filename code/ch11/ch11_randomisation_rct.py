# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Fisher randomisation inference for a simulated school-feeding trial.
# (c) 2026 Stephen E. Moore. MIT Licence.
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)

# (a) Pilot: 12 schools, 6 randomly assigned to school feeding.
N, N1 = 12, 6
y0 = rng.normal(80, 6, N)                       # attendance rate (%) if control
z = rng.permutation(np.r_[np.ones(N1), np.zeros(N - N1)])
y = y0 + 5.0 * z                                # true effect: +5 points
t_obs = y[z == 1].mean() - y[z == 0].mean()
# Exact randomisation distribution under the sharp null y_i(1) = y_i(0)
assign = np.array([np.isin(np.arange(N), c) for c in combinations(range(N), N1)])
t_all = (assign * y).sum(1) / N1 - (~assign * y).sum(1) / (N - N1)
p_exact = np.mean(np.abs(t_all) >= abs(t_obs) - 1e-12)
p_welch = stats.ttest_ind(y[z == 1], y[z == 0], equal_var=False).pvalue
print("(a) pilot with 12 schools, 6 treated")
print(f"  observed difference = {t_obs:.3f} points; "
      f"{len(t_all)} possible assignments")
print(f"  exact randomisation p = {p_exact:.4f}; Welch t-test p = {p_welch:.4f}")
print(f"  smallest attainable p-value = {2 / len(t_all):.4f}")

# (b) Pupil-level trial: 160 pupils, skewed test-score gains, effect +3
n = 160
d = rng.permutation(np.r_[np.ones(n // 2), np.zeros(n // 2)]).astype(bool)
gain = rng.gamma(2.0, 6.0, n) + 3.0 * d
diff = lambda a, b, axis=-1: a.mean(axis) - b.mean(axis)
res = stats.permutation_test((gain[d], gain[~d]), diff, vectorized=True,
                             n_resamples=9999, random_state=rng)
p_t = stats.ttest_ind(gain[d], gain[~d], equal_var=False).pvalue
print("(b) pupil-level trial, n = 160")
print(f"  difference in means = {res.statistic:.3f}")
print(f"  Monte Carlo randomisation p (9999 draws) = {res.pvalue:.4f}; "
      f"Welch p = {p_t:.4f}")

# Fisher interval: invert sharp nulls y_i(1) - y_i(0) = tau0 over a grid
perm = np.array([rng.permutation(n) for _ in range(4000)])
taus, keep = np.round(np.arange(-2, 9.01, 0.05), 2), []
for tau0 in taus:
    y0_imp = gain - tau0 * d                    # imputed control outcomes
    dp = d[perm]
    tp = (y0_imp * dp).sum(1) / dp.sum(1) - (y0_imp * ~dp).sum(1) / (~dp).sum(1)
    t0 = y0_imp[d].mean() - y0_imp[~d].mean()
    keep.append(np.mean(np.abs(tp) >= abs(t0)) > 0.05)
acc = taus[np.array(keep)]
print(f"  95% Fisher interval for a constant effect: [{acc.min():.2f}, "
      f"{acc.max():.2f}]")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(t_all, bins=40, color="steelblue", alpha=0.75)
ax.axvline(t_obs, color="firebrick", lw=1.5, label="observed")
ax.axvline(-t_obs, color="firebrick", lw=1.5, ls=":")
ax.set_xlabel("difference in mean attendance under each assignment")
ax.set_ylabel("number of assignments")
ax.legend(frameon=False)
plt.savefig("ch11_randomisation.pdf", bbox_inches="tight")
