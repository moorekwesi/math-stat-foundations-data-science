# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Figure: relative error of the naive (exp(x)-1)/x versus the eps/x model.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
# figure: relative error of the naive formula over many magnitudes
xs = np.logspace(-16, -1, 400)
rel_naive = np.abs((np.exp(xs) - 1) / xs - np.expm1(xs) / xs) / (np.expm1(xs) / xs)
eps = np.finfo(float).eps
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.loglog(xs, rel_naive + 1e-18, ".", ms=3, label="naive (exp(x)-1)/x")
ax.loglog(xs, eps / xs, "k--", lw=1.2, label="eps / x  (error model)")
ax.axhline(eps, color="grey", lw=0.8, ls=":", label="machine epsilon")
ax.set_ylim(1e-18, 10)
ax.set_xlabel("x")
ax.set_ylabel("relative error")
ax.set_title("Cancellation: error grows like eps/x as x -> 0")
ax.legend(frameon=False, fontsize=8)
plt.savefig("ch01_cancellation.pdf", bbox_inches="tight")
