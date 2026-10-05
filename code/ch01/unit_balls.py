# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Draw the unit balls of the l1, l2 and l-infinity norms in R^2.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10})
t = np.linspace(0, 2 * np.pi, 721)
d = np.column_stack([np.cos(t), np.sin(t)])          # unit directions
fig, ax = plt.subplots(figsize=(4.6, 4.2))
for o, lab, ls in [(1, "1", "-"), (2, "2", "--"), (np.inf, "inf", ":")]:
    pts = d / np.linalg.norm(d, ord=o, axis=1, keepdims=True)  # rescale to norm 1
    ax.plot(pts[:, 0], pts[:, 1], ls, lw=2, label=f"||x||_{lab} = 1")
    area = 0.5 * np.abs(np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]))
    print(f"area enclosed by the l{lab:3s} unit ball = {area:.4f}")
ax.axhline(0, color="grey", lw=0.5)
ax.axvline(0, color="grey", lw=0.5)
ax.set_aspect("equal")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
plt.savefig("ch01_unitballs.pdf", bbox_inches="tight")
print("exact areas: l1 = 2, l2 = pi = %.4f, l-inf = 4" % np.pi)
