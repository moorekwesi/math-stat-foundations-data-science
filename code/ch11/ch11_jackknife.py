# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Jackknife bias and variance estimates, compared with the bootstrap.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)


def loo(x):
    """(n, n-1) array whose row i is the sample with observation i removed."""
    n = x.shape[0]
    keep = ~np.eye(n, dtype=bool)
    return np.broadcast_to(x, (n,) + x.shape)[keep].reshape((n, n - 1)
                                                            + x.shape[1:])


def jackknife(x, stat):
    n = x.shape[0]
    th_i = stat(loo(x))
    th = stat(x[None])[0]
    bias = (n - 1) * (th_i.mean() - th)
    se = np.sqrt((n - 1) / n * np.sum((th_i - th_i.mean())**2))
    return th, bias, se


def gini(x):
    x = np.sort(x, axis=-1)
    n = x.shape[-1]
    w = 2 * np.arange(1, n + 1) - n - 1
    return np.sum(w * x, -1) / (n * np.sum(x, -1))


n, B = 60, 4000
inc = rng.lognormal(7.0, 0.9, n) / 1000      # income in thousands
food = 0.45 * inc**0.85 * rng.lognormal(0, 0.3, n)
xy = np.column_stack([food, inc])
cases = {
    "plug-in variance": (inc, lambda s: s.var(axis=-1)),
    "Gini": (inc, gini),
    "ratio food/income": (xy, lambda s: s[..., 0].mean(-1) / s[..., 1].mean(-1)),
    "median": (inc, lambda s: np.median(s, axis=-1)),
}
print(f"n = {n}; bootstrap with B = {B}")
print(f"{'statistic':<19}{'estimate':>11}{'jack bias':>11}{'boot bias':>11}"
      f"{'jack se':>10}{'boot se':>10}")
for name, (x, f) in cases.items():
    th, jb, js = jackknife(x, f)
    tb = f(x[rng.integers(0, n, size=(B, n))])
    print(f"{name:<19}{th:11.4f}{jb:11.4f}{tb.mean() - th:11.4f}"
          f"{js:10.4f}{tb.std(ddof=1):10.4f}")
s2 = inc.var(ddof=1)
print(f"check: jackknife-corrected plug-in variance = "
      f"{inc.var() - jackknife(inc, lambda s: s.var(axis=-1))[1]:.6f}, "
      f"s^2 = {s2:.6f}")

# Jackknife se of the median is inconsistent: compare with the truth
R = 2000
xs = rng.lognormal(7.0, 0.9, (R, n)) / 1000
true_sd = np.median(xs, axis=1).std()
js_med = np.array([jackknife(x, lambda s: np.median(s, axis=-1))[2]
                   for x in xs[:500]])
print(f"median: true sd = {true_sd:.4f}; mean jackknife se = {js_med.mean():.4f},"
      f" CV across samples = {js_med.std() / js_med.mean():.2f}")
