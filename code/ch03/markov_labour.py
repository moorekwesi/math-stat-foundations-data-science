# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# A simulated labour-market Markov chain: eigenvalues and stationary distribution.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
states = ["Formal", "Informal", "Unemployed"]
# Simulated quarterly transition matrix, rows = origin, columns = destination
P = np.array([[0.94, 0.04, 0.02],
              [0.05, 0.85, 0.10],
              [0.08, 0.32, 0.60]])
assert np.allclose(P.sum(axis=1), 1.0)

# Stationary distribution: left eigenvector of P for eigenvalue 1
lam, W = np.linalg.eig(P.T)
order = np.argsort(-np.abs(lam))
lam, W = lam[order], W[:, order]
pi = np.real(W[:, 0] / W[:, 0].sum())
print("eigenvalues of P      :", np.round(np.real(lam), 4))
print("stationary pi         :", np.round(pi, 4))
print("check pi P = pi       :", np.allclose(pi @ P, pi))
print("half-life of shocks   : %.1f quarters"
      % (np.log(0.5) / np.log(np.abs(lam[1]))))

# Distribution path from "everyone unemployed"
x = np.array([0.0, 0.0, 1.0])
path = [x]
for t in range(40):
    x = x @ P
    path.append(x)
path = np.array(path)
print("after 4 quarters      :", np.round(path[4], 4))
print("after 40 quarters     :", np.round(path[40], 4))
gap = np.abs(path - pi).sum(axis=1)
print("TV gap at t=10, 20    : %.2e  %.2e" % (gap[10] / 2, gap[20] / 2))

# Simulate 2000 workers for 200 quarters and compare long-run shares
n, T = 2000, 200
s = rng.integers(0, 3, size=n)
cum = P.cumsum(axis=1)
for t in range(T):
    u = rng.random(n)
    s = (u[:, None] > cum[s]).sum(axis=1)
shares = np.bincount(s, minlength=3) / n
print("simulated shares T=200:", np.round(shares, 4))
print("expected spell (quarters):",
      dict(zip(states, np.round(1 / (1 - np.diag(P)), 1))))

fig, ax = plt.subplots(figsize=(6, 3.8))
for j, name in enumerate(states):
    ax.plot(path[:, j], lw=2, label=name)
    ax.axhline(pi[j], ls=":", color="gray", lw=1)
ax.set_xlabel("quarter t")
ax.set_ylabel("share of labour force")
ax.set_title("Convergence of x0 P^t to the stationary distribution")
ax.legend(frameon=False)
plt.savefig("ch03_markov_convergence.pdf", bbox_inches="tight")
