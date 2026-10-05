# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Coverage of bootstrap intervals for a skewed mean: a Monte Carlo study.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
sigma, R, B, alpha = 1.0, 2000, 499, 0.05
true_mean = np.exp(sigma**2 / 2)              # lognormal(0, 1) mean
methods = ["t-interval", "normal", "percentile", "basic", "studentised", "BCa"]
z = stats.norm.ppf(1 - alpha / 2)
zq = stats.norm.ppf([alpha / 2, 1 - alpha / 2])


def quant(srt, lev):
    """Row-wise linear-interpolation quantiles of sorted rows at levels lev."""
    pos = lev * (srt.shape[1] - 1)
    lo = np.floor(pos).astype(int)
    frac = pos - lo
    hi = np.minimum(lo + 1, srt.shape[1] - 1)
    r = np.arange(srt.shape[0])[:, None]
    return srt[r, lo] * (1 - frac) + srt[r, hi] * frac


def intervals(x):
    """All six intervals for a (chunk, n) array of samples -> (chunk, 6, 2)."""
    c, n = x.shape
    m, s = x.mean(axis=1), x.std(axis=1, ddof=1)
    xb = np.take_along_axis(x[:, None, :].repeat(B, 1),
                            rng.integers(0, n, size=(c, B, n)), axis=2)
    mb, sb = xb.mean(axis=2), xb.std(axis=2, ddof=1)
    srt = np.sort(mb, axis=1)
    lv = np.tile([alpha / 2, 1 - alpha / 2], (c, 1))
    qlo, qhi = quant(srt, lv).T
    tlo, thi = quant(np.sort((mb - m[:, None]) / (sb / np.sqrt(n)), 1), lv).T
    se = mb.std(axis=1, ddof=1)
    z0 = stats.norm.ppf(np.clip(np.mean(mb < m[:, None], 1), 1e-4, 1 - 1e-4))
    d = x - m[:, None]                      # jackknife influence for the mean
    a = np.sum(d**3, 1) / (6 * np.sum(d**2, 1)**1.5)
    z0, a = z0[:, None], a[:, None]
    lev = stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
    tq = stats.t.ppf(1 - alpha / 2, n - 1) * s / np.sqrt(n)
    rn = s / np.sqrt(n)
    out = [(m - tq, m + tq), (m - z * se, m + z * se), (qlo, qhi),
           (2 * m - qhi, 2 * m - qlo), (m - thi * rn, m - tlo * rn),
           tuple(quant(srt, lev).T)]
    return np.stack([np.stack(p, axis=1) for p in out], axis=1)


sizes = (10, 25, 100)
cover = np.zeros((len(sizes), len(methods)))
length = np.zeros_like(cover)
chunk = 50
for j, n in enumerate(sizes):
    for r in range(R // chunk):
        ci = intervals(rng.lognormal(0, sigma, size=(chunk, n)))
        hit = (ci[..., 0] <= true_mean) & (true_mean <= ci[..., 1])
        cover[j] += hit.sum(axis=0)
        length[j] += np.sum(ci[..., 1] - ci[..., 0], 0)
cover /= R
length /= R

print(f"Coverage of nominal 95% CIs for a lognormal mean ({R} reps, B = {B})")
print(f"{'method':<13}" + "".join(f"{'n=' + str(n):>9}" for n in sizes)
      + "   avg length (n=25)")
for k, mth in enumerate(methods):
    print(f"{mth:<13}" + "".join(f"{cover[j, k]:9.3f}" for j in range(3))
          + f"{length[1, k]:12.3f}")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
w = 0.26
pos = np.arange(len(methods))
for j, n in enumerate(sizes):
    ax.bar(pos + (j - 1) * w, cover[j], width=w, label=f"n = {n}")
ax.axhline(0.95, color="black", lw=1, ls="--")
ax.set_xticks(pos)
ax.set_xticklabels(methods, rotation=20)
ax.set_ylim(0.7, 1.0)
ax.set_ylabel("empirical coverage")
ax.legend(frameon=False, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, -0.22))
plt.savefig("ch11_ci_coverage.pdf", bbox_inches="tight")
