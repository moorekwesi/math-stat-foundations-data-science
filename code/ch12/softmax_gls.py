# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Multinomial-logit gradient check, and GLS vs OLS under heteroskedasticity.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)

# ---- softmax regression: loss = -sum_i log p_{i, y_i},  grad = X'(P - Y)
n, p, K = 300, 4, 3                  # simulated choice of transport mode
X = rng.normal(size=(n, p))
W_true = rng.normal(size=(p, K))
U = X @ W_true + rng.gumbel(size=(n, K))
yc = U.argmax(axis=1)
Y = np.eye(K)[yc]                    # one-hot labels


def loss(W):
    Eta = X @ W
    lse = np.log(np.exp(Eta - Eta.max(1, keepdims=True)).sum(1)) + Eta.max(1)
    return np.sum(lse - np.sum(Y * Eta, axis=1))


def grad(W):
    Eta = X @ W
    P = np.exp(Eta - Eta.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    return X.T @ (P - Y)


W0 = rng.normal(size=(p, K))
G_num = np.zeros_like(W0)
for idx in np.ndindex(W0.shape):
    E = np.zeros_like(W0)
    E[idx] = 1e-6
    G_num[idx] = (loss(W0 + E) - loss(W0 - E)) / 2e-6
err = np.abs(grad(W0) - G_num).max()
print(f"softmax gradient X'(P - Y): max abs error {err:.2e}")

# ---- GLS: Var(e_i) = sigma_i^2 known; compare sampling variance with OLS
n, R = 200, 2000
x = rng.uniform(1, 10, size=n)       # e.g. firm size
Xd = np.column_stack([np.ones(n), x])
sig = 0.3 * x                        # error s.d. grows with size
w = 1 / sig**2                       # Omega^{-1} is diagonal
b_ols, b_gls = np.zeros((R, 2)), np.zeros((R, 2))
for r in range(R):
    y = Xd @ np.array([1.0, 0.5]) + sig * rng.normal(size=n)
    b_ols[r] = np.linalg.lstsq(Xd, y, rcond=None)[0]
    b_gls[r] = np.linalg.solve(Xd.T @ (w[:, None] * Xd), Xd.T @ (w * y))

V_gls = np.linalg.inv(Xd.T @ (w[:, None] * Xd))
XtXi = np.linalg.inv(Xd.T @ Xd)
V_ols = XtXi @ Xd.T @ (sig[:, None] ** 2 * Xd) @ XtXi    # sandwich
print(f"\n{'':18}{'MC var(slope)':>14}{'theory':>12}")
print(f"{'OLS':18}{b_ols[:, 1].var():>14.3e}{V_ols[1, 1]:>12.3e}")
print(f"{'GLS (= WLS)':18}{b_gls[:, 1].var():>14.3e}{V_gls[1, 1]:>12.3e}")
print(f"efficiency gain of GLS: {V_ols[1, 1] / V_gls[1, 1]:.2f}x smaller variance")
