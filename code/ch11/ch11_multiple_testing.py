# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Bonferroni, Holm and Benjamini-Hochberg for 20 outcomes of one RCT.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)
m, m1, n, R, q = 20, 5, 100, 4000, 0.05     # outcomes, true effects, per arm


def holm(p, a):
    """Holm step-down: reject the k smallest while p_(j) <= a/(m-j+1)."""
    o = np.argsort(p, axis=-1)
    ps = np.take_along_axis(p, o, -1)
    ok = ps <= a / (p.shape[-1] - np.arange(p.shape[-1]))
    k = np.argmin(np.c_[ok, np.zeros((p.shape[0], 1), bool)], axis=-1)
    rej = np.arange(p.shape[-1]) < k[:, None]           # first k sorted
    out = np.zeros_like(rej)
    np.put_along_axis(out, o, rej, -1)
    return out


def bh(p, a):
    """Benjamini-Hochberg step-up: largest k with p_(k) <= k a / m."""
    mm = p.shape[-1]
    o = np.argsort(p, axis=-1)
    ps = np.take_along_axis(p, o, -1)
    ok = ps <= a * np.arange(1, mm + 1) / mm
    k = np.where(ok.any(-1), mm - np.argmax(ok[:, ::-1], -1), 0)
    rej = np.arange(mm) < k[:, None]
    out = np.zeros_like(rej)
    np.put_along_axis(out, o, rej, -1)
    return out


delta = np.r_[np.full(m1, 0.4), np.zeros(m - m1)]     # effects in sd units
treat = rng.normal(0, 1, (R, n, m)) + delta
ctrl = rng.normal(0, 1, (R, n, m))
p = stats.ttest_ind(treat, ctrl, axis=1).pvalue          # (R, m) p-values
null = delta == 0

rules = {"no correction": p <= q, "Bonferroni": p <= q / m,
         "Holm": holm(p, q), "Benjamini-Hochberg": bh(p, q)}
print(f"m = {m} outcomes ({m1} truly affected), {R} simulated trials")
print(f"{'procedure':<20}{'FWER':>7}{'FDR':>7}{'power':>8}{'avg rej':>9}")
for name, rej in rules.items():
    V = rej[:, null].sum(1)                              # false rejections
    Rn = rej.sum(1)
    fdr = np.mean(np.where(Rn > 0, V / np.maximum(Rn, 1), 0))
    pw = rej[:, ~null].mean()
    print(f"{name:<20}{np.mean(V > 0):7.3f}{fdr:7.3f}{pw:8.3f}{Rn.mean():9.2f}")

p1 = np.sort(p[0])
print("trial 1, five smallest p-values:", " ".join(f"{v:.4f}" for v in p1[:5]))
print(f"Bonferroni cut-off q/m = {q / m:.4f}")
print("BH cut-offs k*q/m, k = 1..5:     ",
      " ".join(f"{k * q / m:.4f}" for k in range(1, 6)))
