# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Geometry of the Lagrange condition: tangency of level set and budget line.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

a, p1, p2, m = 0.4, 2.0, 5.0, 100.0          # Cobb-Douglas share, prices, income


def u(x1, x2):
    return x1**a * x2**(1 - a)


def grad_u(x1, x2):
    return np.array([a * u(x1, x2) / x1, (1 - a) * u(x1, x2) / x2])


grad_g = np.array([p1, p2])                  # gradient of g(x) = p1 x1 + p2 x2
xs = np.array([a * m / p1, (1 - a) * m / p2])  # Marshallian demand
lam = grad_u(*xs)[0] / p1

print("Optimum x* = (%.2f, %.2f),  u(x*) = %.4f" % (xs[0], xs[1], u(*xs)))
print("grad u(x*) =", np.round(grad_u(*xs), 4), "  grad g =", grad_g)
print("ratio grad u / grad g (componentwise):", np.round(grad_u(*xs) / grad_g, 4))
print("Lagrange multiplier lambda = %.4f" % lam)

xb = np.array([35.0, (m - p1 * 35.0) / p2])  # another point on the budget line
gu = grad_u(*xb)
cosang = gu @ grad_g / (np.linalg.norm(gu) * np.linalg.norm(grad_g))
t = np.array([p2, -p1]) / np.hypot(p1, p2)   # tangent direction of budget line
print("\nAt xb = (%.1f, %.1f): u = %.4f" % (xb[0], xb[1], u(*xb)))
print("angle between grad u and grad g = %.2f degrees" %
      np.degrees(np.arccos(cosang)))
print("directional derivative of u along budget line = %.4f" % (gu @ t))

# ---- figure -----------------------------------------------------------
x1 = np.linspace(1, 50, 300)
X1, X2 = np.meshgrid(x1, np.linspace(1, 25, 300))
fig, ax = plt.subplots(figsize=(6, 3.8))
levels = [u(*xb) * 0.8, u(*xb), u(*xs), u(*xs) * 1.15]
cs = ax.contour(X1, X2, u(X1, X2), levels=sorted(levels), colors="grey",
                linewidths=0.9)
ax.clabel(cs, fmt="%.1f", fontsize=7)
ax.plot(x1, (m - p1 * x1) / p2, color="C0", lw=2, label="budget line g(x)=m")
for pt, col, nm in [(xs, "C3", "x*"), (xb, "C2", "xb")]:
    ax.plot(*pt, "o", color=col)
    ax.annotate(nm, pt + np.array([0.8, 0.6]), color=col)
    gu_ = grad_u(*pt) / np.linalg.norm(grad_u(*pt)) * 5
    gg_ = grad_g / np.linalg.norm(grad_g) * 5
    ax.arrow(*pt, *gu_, color=col, width=0.08, head_width=0.6)
    ax.arrow(*pt, *gg_, color="C0", width=0.08, head_width=0.6, alpha=0.6)
ax.set_xlim(0, 50)
ax.set_ylim(0, 25)
ax.set_xlabel("$x_1$ (good 1)")
ax.set_ylabel("$x_2$ (good 2)")
ax.set_title("Tangency: grad u parallel to grad g at x*", fontsize=10)
ax.legend(loc="upper right", fontsize=8)
plt.savefig("ch09_lagrange_geometry.pdf", bbox_inches="tight")
