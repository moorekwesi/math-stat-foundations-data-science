# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# High-dimensional concentration: norms of Gaussian vectors, near-orthogonality.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
k = 200                                     # number of random vectors
print(f"{'d':>6}{'mean |X|/sqrt(d)':>18}{'sd |X|':>9}{'mean |cos|':>12}"
      f"{'max |cos|':>11}{'1/sqrt(d)':>11}")
cosines = {}
for d in [2, 10, 100, 1000, 10000]:
    X = rng.normal(size=(k, d))
    norms = np.linalg.norm(X, axis=1)
    U = X / norms[:, None]
    C = U @ U.T
    off = C[np.triu_indices(k, 1)]          # all k(k-1)/2 pairwise cosines
    cosines[d] = off
    print(f"{d:6d}{np.mean(norms / np.sqrt(d)):18.4f}{norms.std():9.4f}"
          f"{np.mean(np.abs(off)):12.4f}{np.max(np.abs(off)):11.4f}"
          f"{1 / np.sqrt(d):11.4f}")
print("(sd of |X| stays near 1/sqrt(2) = 0.7071 in every dimension)")

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(6, 3.8))
for d, col in [(10, "firebrick"), (100, "darkorange"), (1000, "navy")]:
    ax.hist(cosines[d], bins=80, range=(-1, 1), density=True, histtype="step",
            lw=1.5, color=col, label=f"d = {d}")
ax.set_xlabel("cosine of the angle between two random vectors")
ax.set_ylabel("density")
ax.legend(frameon=False)
plt.savefig("ch06_highdim.pdf", bbox_inches="tight")
