# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# The determinant as the signed area scaling factor of a linear map.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10})
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float).T
maps = {"A = [[2, 1], [0.5, 1.5]]": np.array([[2.0, 1.0], [0.5, 1.5]]),
        "B = [[1, 2], [0.5, 1]]": np.array([[1.0, 2.0], [0.5, 1.0]])}
fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharey=True)
for ax, (lab, M) in zip(axes, maps.items()):
    img = M @ square
    ax.fill(square[0], square[1], alpha=0.25, label="unit square")
    ax.fill(img[0], img[1], alpha=0.45, label="image M(square)")
    ax.plot(img[0], img[1], "k-", lw=1)
    ax.set_title(f"{lab}\ndet = {np.linalg.det(M):.2f}", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 3.3)
    ax.set_ylim(-0.2, 2.3)
    ax.set_xlabel("x1")
    x, y = img[0, :4], img[1, :4]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    print(f"{lab:26s} det = {np.linalg.det(M):6.3f}  image area = {area:.3f}")
axes[0].set_ylabel("x2")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
plt.savefig("ch01_determinant.pdf", bbox_inches="tight")
print("B is singular: its columns are parallel, the square collapses to a line")
