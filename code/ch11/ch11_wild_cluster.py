# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Size of cluster-robust t-tests vs the wild cluster bootstrap, few clusters.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
R, B = 400, 399


def cr1_t(X, Y, XtXi, starts, G):
    """Cluster-robust (CR1) t-statistics for coef 1; Y has shape (k, n)."""
    n = X.shape[0]
    beta = Y @ X @ XtXi                                  # (k, 2)
    U = Y - beta @ X.T
    S = np.add.reduceat(U[:, :, None] * X[None], starts, axis=1)  # (k, G, 2)
    A = S @ XtXi[:, 1]                                   # (k, G)
    c = G / (G - 1) * (n - 1) / (n - 2)
    return beta[:, 1] / np.sqrt(c * np.sum(A**2, axis=1))


def one_experiment(G):
    """Null true; unbalanced EA sizes; programme in a third of the EAs."""
    sizes = rng.integers(5, 40, size=G)
    g = np.repeat(np.arange(G), sizes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    treated = np.zeros(G)
    treated[rng.choice(G, size=max(G // 3, 1), replace=False)] = 1
    d = treated[g]
    shock = rng.standard_t(3, size=G) * (1 + treated)    # heterogeneous EAs
    y = 2.0 + shock[g] + rng.normal(0, 1, g.size)
    X = np.column_stack([np.ones(g.size), d])
    XtXi = np.linalg.inv(X.T @ X)
    t = cr1_t(X, y[None], XtXi, starts, G)[0]
    # wild cluster restricted bootstrap: impose beta_1 = 0
    u_r = y - y.mean()
    w = rng.choice([-1.0, 1.0], size=(B, G))              # Rademacher by EA
    Ystar = y.mean() + w[:, g] * u_r
    tstar = cr1_t(X, Ystar, XtXi, starts, G)
    p_boot = np.mean(np.abs(tstar) >= abs(t))
    p_norm = 2 * stats.norm.sf(abs(t))
    p_tdist = 2 * stats.t.sf(abs(t), G - 1)
    return np.array([p_norm, p_tdist, p_boot]) < 0.05


Gs = (5, 8, 12, 20, 40)
rej = np.array([np.mean([one_experiment(G) for _ in range(R)], axis=0)
                for G in Gs])
print(f"Rejection rates of a true null at the 5% level ({R} reps, B = {B})")
print(f"{'G':>4}{'CR1 + N(0,1)':>14}{'CR1 + t(G-1)':>14}{'wild cluster':>14}")
for G, r in zip(Gs, rej):
    print(f"{G:4d}{r[0]:14.3f}{r[1]:14.3f}{r[2]:14.3f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
for k, lab in enumerate(["CR1, normal critical values",
                         "CR1, t(G-1) critical values",
                         "wild cluster bootstrap (WCR)"]):
    ax.plot(Gs, rej[:, k], marker="o", label=lab)
ax.axhline(0.05, color="black", ls="--", lw=1)
ax.set_xscale("log")
ax.minorticks_off()
ax.set_xticks(Gs)
ax.set_xticklabels([str(G) for G in Gs])
ax.set_xlabel("number of clusters G")
ax.set_ylabel("rejection rate under the null")
ax.legend(frameon=False)
plt.savefig("ch11_wild_cluster.pdf", bbox_inches="tight")
