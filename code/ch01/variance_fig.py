# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Figure: error of one-pass, two-pass and Welford variances against the offset.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
rng = np.random.default_rng(803)
z = rng.standard_normal(10_000)
z = (z - z.mean()) / z.std(ddof=1)        # sample variance exactly 1


def one_pass(x):
    return ((x * x).sum() - x.sum() ** 2 / x.size) / (x.size - 1)


def two_pass(x):
    d = x - x.mean()
    return (d @ d) / (x.size - 1)


def welford(x):
    mean, m2 = 0.0, 0.0
    for k, xk in enumerate(x, start=1):
        delta = xk - mean
        mean += delta / k
        m2 += delta * (xk - mean)
    return m2 / (len(x) - 1)


c = np.logspace(1, 9, 25)
errs = np.array([[abs(f(ci + z) - 1.0) for f in (one_pass, two_pass)]
                 + [abs(welford((ci + z).tolist()) - 1.0)] for ci in c])
errs = np.maximum(errs, 1e-17)            # show exact zeros on the log axis
fig, ax = plt.subplots(figsize=(6, 3.8))
for j, lab in enumerate(["one-pass (textbook)", "two-pass", "Welford"]):
    ax.loglog(c, errs[:, j], "o-", ms=3, label=lab)
ax.loglog(c, np.finfo(float).eps * c**2, "k--", lw=1, label="eps x offset^2")
ax.set_ylim(1e-17, 1e3)
ax.set_xlabel("offset c added to data with variance 1")
ax.set_ylabel("absolute error in variance")
ax.set_title("The one-pass formula loses digits like eps c^2")
ax.legend(frameon=False, fontsize=8)
plt.savefig("ch01_variance.pdf", bbox_inches="tight")
print("largest one-pass error: %.2e" % errs[:, 0].max())
