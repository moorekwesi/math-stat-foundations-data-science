# DEP 803 -- Chapter 0: Mathematical Refresher
# Graphs of exp(x) and its inverse ln(x), mirror images in the line y = x.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

# graphs of exp and its inverse ln
x = np.linspace(-2, 2.5, 200)
xp = np.linspace(0.05, 6, 200)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(x, np.exp(x), label="y = exp(x)")
ax.plot(xp, np.log(xp), label="y = ln(x)")
ax.plot([-2, 6], [-2, 6], "k--", lw=0.8, label="y = x (mirror line)")
ax.plot(x, 1 + x, ":", color="gray", label="tangent y = 1 + x")
ax.set_xlim(-2, 6)
ax.set_ylim(-2.5, 6)
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.legend(frameon=False, fontsize=8, loc="lower right")
plt.savefig("ch00_explog.pdf", bbox_inches="tight")
print("figure ch00_explog.pdf written")
