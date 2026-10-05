# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Inner products, norms, distances, angles and the Cauchy-Schwarz inequality.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
x = np.array([3.0, -4.0, 1.0, 2.0])
y = np.array([1.0, 2.0, -2.0, 5.0])
print("x =", x, "  y =", y)
print("<x, y> =", x @ y)
for name, o in [("1", 1), ("2", 2), ("inf", np.inf)]:
    print(f"||x||_{name:3s} = {np.linalg.norm(x, o):8.4f}"
          f"    d_{name:3s}(x, y) = {np.linalg.norm(x - y, o):8.4f}")
cos = x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
print(f"cosine = {cos:.4f}, angle = {np.degrees(np.arccos(cos)):.2f} degrees")

# Cauchy-Schwarz and triangle inequality on 100,000 random pairs in R^10
U = rng.standard_normal((100_000, 10))
V = rng.standard_normal((100_000, 10))
nu, nv = np.linalg.norm(U, axis=1), np.linalg.norm(V, axis=1)
ratio = np.abs(np.sum(U * V, axis=1)) / (nu * nv)
tri = np.linalg.norm(U + V, axis=1) / (nu + nv)
print(f"max |<u,v>|/(||u|| ||v||) = {ratio.max():.6f}  (must be <= 1)")
print(f"max ||u+v||/(||u||+||v||) = {tri.max():.6f}  (must be <= 1)")
print(f"ratio when v = -3u        = {abs(U[0] @ (-3*U[0])) / (3*nu[0]**2):.6f}")

# norm equivalence: ||x||_inf <= ||x||_2 <= ||x||_1 <= sqrt(n) ||x||_2
n1, n2, ni = (np.linalg.norm(U, o, axis=1) for o in (1, 2, np.inf))
ok = np.all((ni <= n2) & (n2 <= n1) & (n1 <= np.sqrt(10) * n2 * (1 + 1e-12)))
print("norm-equivalence chain holds for all 100,000 vectors:", bool(ok))
print(f"largest ||u||_1/||u||_2 seen = {np.max(n1 / n2):.4f}"
      f"  (bound sqrt(10) = {np.sqrt(10):.4f})")

# cosine similarity of (simulated) household budget-share vectors
# columns: food, housing, transport, education, other
shares = np.array([[0.55, 0.15, 0.10, 0.05, 0.15],    # rural household
                   [0.48, 0.20, 0.12, 0.06, 0.14],    # peri-urban household
                   [0.25, 0.30, 0.15, 0.15, 0.15]])   # urban household
Sn = shares / np.linalg.norm(shares, axis=1, keepdims=True)
print("cosine similarity of budget shares (rural, peri-urban, urban):")
print(np.round(Sn @ Sn.T, 4))
