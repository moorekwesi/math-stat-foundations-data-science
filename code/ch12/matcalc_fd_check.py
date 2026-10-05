# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Verify matrix-calculus identities with central finite differences.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n = 4


def num_grad(f, X, h=1e-6):
    """Central-difference gradient of a scalar function f at array X."""
    G = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        E = np.zeros_like(X)
        E[idx] = h
        G[idx] = (f(X + E) - f(X - E)) / (2 * h)
    return G


a = rng.normal(size=n)
A = rng.normal(size=(n, n))
B = rng.normal(size=(n, n))
x = rng.normal(size=n)
X = rng.normal(size=(n, n))
M = rng.normal(size=(n, n))
S = M @ M.T + n * np.eye(n)              # symmetric positive definite
Si = np.linalg.inv(S)
Z = rng.normal(size=(10, n))             # design matrix for least squares
y = rng.normal(size=10)

cases = [
    ("a'x", lambda v: a @ v, a, x),
    ("x'Ax", lambda v: v @ A @ v, (A + A.T) @ x, x),
    ("||y - Zb||^2", lambda b: np.sum((y - Z @ b) ** 2),
     -2 * Z.T @ (y - Z @ x), x),
    ("tr(BX)", lambda W: np.trace(B @ W), B.T, X),
    ("tr(X'BX)", lambda W: np.trace(W.T @ B @ W), (B + B.T) @ X, X),
    ("log det S", lambda W: np.linalg.slogdet(W)[1], Si.T, S),
    ("tr(B S^-1)", lambda W: np.trace(B @ np.linalg.inv(W)),
     -(Si @ B @ Si).T, S),
    ("a'S^-1 a", lambda W: a @ np.linalg.solve(W, a),
     -Si.T @ np.outer(a, a) @ Si.T, S),
]

print(f"{'function':<14}{'formula':>28}{'max abs error':>16}")
formulas = ["a", "(A + A')x", "-2Z'(y - Zb)", "B'", "(B + B')X",
            "S^-T", "-(S^-1 B S^-1)'", "-S^-T aa' S^-T"]
for (name, f, g_exact, at), form in zip(cases, formulas):
    err = np.max(np.abs(num_grad(f, at) - g_exact))
    print(f"{name:<14}{form:>28}{err:>16.2e}")

# Hessian of the quadratic form: differentiate the gradient numerically
H_num = np.column_stack([
    num_grad(lambda v: v @ A @ v, x + 1e-4 * e) for e in np.eye(n)])
H_num = (H_num - np.column_stack([
    num_grad(lambda v: v @ A @ v, x - 1e-4 * e) for e in np.eye(n)])) / 2e-4
print(f"\nHessian of x'Ax vs A + A': max abs error "
      f"{np.max(np.abs(H_num - (A + A.T))):.2e}")
