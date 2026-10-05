# DEP 803 -- Chapter 7: Estimation Principles
# Sample mean versus sample median as estimators of a centre: light vs heavy tails.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
R = 20000
draws = {"normal": lambda size: rng.normal(size=size),
         "t(3)": lambda size: rng.standard_t(3, size=size),
         "Cauchy": lambda size: rng.standard_cauchy(size=size)}

def iqr(v):
    q1, q3 = np.percentile(v, [25, 75])
    return q3 - q1

print("Spread of estimators of the centre (true centre = 0)")
print("sqrt(n) * IQR of the sampling distribution, R = 20000")
print(f"{'model':>7} {'n':>5} {'mean':>9} {'median':>9} {'ratio':>7}")
for name, draw in draws.items():
    for n in (25, 400):
        x = draw((R, n))
        m, md = x.mean(axis=1), np.median(x, axis=1)
        a, b = np.sqrt(n) * iqr(m), np.sqrt(n) * iqr(md)
        print(f"{name:>7} {n:>5} {a:9.3f} {b:9.3f} {a / b:7.2f}")
print("Theory (normal): ARE(median, mean) = 2/pi =", round(2 / np.pi, 3))

# Figure: running mean and running median for one Cauchy sequence
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
N = 20000
x = rng.standard_cauchy(N)
k = np.arange(1, N + 1)
run_mean = np.cumsum(x) / k
idx = np.unique(np.logspace(0, np.log10(N), 300).astype(int))
run_med = np.array([np.median(x[:i]) for i in idx])
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(k, run_mean, lw=1, color="tab:red", label="running mean")
ax.plot(idx, run_med, lw=1.6, color="tab:blue", label="running median")
ax.axhline(0, color="k", lw=0.6, ls="--")
ax.set_xscale("log")
ax.set_xlabel("sample size n")
ax.set_ylabel("estimate of the centre")
ax.set_title("Cauchy data: the median is consistent, the mean is not")
ax.legend(frameon=False)
plt.savefig("ch07_mean_median.pdf", bbox_inches="tight")
