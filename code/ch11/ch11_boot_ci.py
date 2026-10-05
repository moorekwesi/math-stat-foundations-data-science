# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Five bootstrap confidence intervals for a Gini coefficient, plus scipy.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)


def gini(x, axis=-1):
    """Plug-in Gini coefficient along the last axis (`axis` is for scipy)."""
    x = np.sort(x, axis=-1)
    n = x.shape[-1]
    w = 2 * np.arange(1, n + 1) - n - 1
    return np.sum(w * x, axis=-1) / (n * np.sum(x, axis=-1))


def jackknife(x, stat):
    """Leave-one-out replicates theta_(i), i = 1..n."""
    n = x.size
    keep = ~np.eye(n, dtype=bool)
    return stat(np.broadcast_to(x, (n, n))[keep].reshape(n, n - 1))


def jack_se(x, stat):
    th = jackknife(x, stat)
    n = x.size
    return np.sqrt((n - 1) / n * np.sum((th - th.mean())**2))


sigma, n, B, alpha = 0.9, 150, 4999, 0.05
x = rng.lognormal(7.0, sigma, size=n)          # simulated incomes
g_true = 2 * stats.norm.cdf(sigma / np.sqrt(2)) - 1
g_hat = gini(x)
idx = rng.integers(0, n, size=(B, n))
g_star = gini(x[idx])
se_b = g_star.std(ddof=1)
z = stats.norm.ppf(1 - alpha / 2)
qlo, qhi = np.quantile(g_star, [alpha / 2, 1 - alpha / 2])

ci = {}
ci["normal"] = (g_hat - z * se_b, g_hat + z * se_b)
ci["percentile"] = (qlo, qhi)
ci["basic"] = (2 * g_hat - qhi, 2 * g_hat - qlo)

# studentised (bootstrap-t): jackknife se inside every bootstrap sample
se_hat = jack_se(x, gini)
t_star = np.array([(g_star[b] - g_hat) / jack_se(x[idx[b]], gini)
                   for b in range(B)])
tlo, thi = np.quantile(t_star, [alpha / 2, 1 - alpha / 2])
ci["studentised"] = (g_hat - thi * se_hat, g_hat - tlo * se_hat)

# BCa: bias correction z0 and jackknife acceleration a
z0 = stats.norm.ppf(np.mean(g_star < g_hat))
th = jackknife(x, gini)
d = th.mean() - th
a = np.sum(d**3) / (6 * np.sum(d**2)**1.5)
zq = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
adj = stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
ci["BCa"] = tuple(np.quantile(g_star, adj))

print(f"n = {n}, B = {B}; sample Gini = {g_hat:.4f} (population {g_true:.4f})")
print(f"bootstrap se = {se_b:.4f}, jackknife se = {se_hat:.4f}")
print(f"BCa constants: z0 = {z0:.4f}, a = {a:.4f}; "
      f"quantile levels = {adj[0]:.4f}, {adj[1]:.4f}")
print(f"{'method':<14}{'lower':>9}{'upper':>9}{'length':>9}{'skew ratio':>12}")
for k, (lo, hi) in ci.items():
    print(f"{k:<14}{lo:9.4f}{hi:9.4f}{hi - lo:9.4f}"
          f"{(hi - g_hat) / (g_hat - lo):12.3f}")

print("scipy.stats.bootstrap (n_resamples = 4999):")
for m in ("percentile", "basic", "BCa"):
    res = stats.bootstrap((x,), gini, n_resamples=B, method=m,
                          random_state=np.random.default_rng(11))
    lo, hi = res.confidence_interval
    print(f"  {m:<12}{lo:9.4f}{hi:9.4f}   se = {res.standard_error:.4f}")
