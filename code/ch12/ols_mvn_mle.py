# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# OLS from the gradient; Gaussian MLE of mean and covariance, checked numerically.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)

# ---- OLS: simulated wage equation (log wage on schooling, experience, ...)
n, p = 500, 4
X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
beta = np.array([1.5, 0.08, 0.03, -0.2])
y = X @ beta + 0.4 * rng.normal(size=n)

b_hat = np.linalg.solve(X.T @ X, X.T @ y)          # normal equations
grad = -2 * X.T @ (y - X @ b_hat)                   # gradient of RSS
H = 2 * X.T @ X                                     # Hessian of RSS
print("OLS beta_hat         :", np.round(b_hat, 4))
print("||gradient|| at b_hat:", f"{np.linalg.norm(grad):.2e}")
print("min eigenvalue of H  :", f"{np.linalg.eigvalsh(H).min():.1f} (> 0)")

# ---- Multivariate normal MLE: simulated (income, consumption, savings)
mu = np.array([2.0, 1.5, 0.5])
L = np.array([[1.0, 0, 0], [0.8, 0.5, 0], [0.2, -0.3, 0.4]])
Sigma = L @ L.T
m = 2000
Y = rng.multivariate_normal(mu, Sigma, size=m)


def loglik(mu_, Sig_):
    R = Y - mu_
    _, logdet = np.linalg.slogdet(Sig_)
    quad = np.einsum("ij,jk,ik->", R, np.linalg.inv(Sig_), R)
    return -0.5 * (m * logdet + quad + m * 3 * np.log(2 * np.pi))


mu_hat = Y.mean(axis=0)
S_hat = (Y - mu_hat).T @ (Y - mu_hat) / m            # divide by m, not m-1
print("\nmu_hat               :", np.round(mu_hat, 4))
print("Sigma_hat equals np.cov(bias=True):",
      np.allclose(S_hat, np.cov(Y.T, bias=True)))

# analytic score w.r.t. Sigma: 0.5 * (S^-1 W S^-1 - m S^-1), W = sum r r'
Si = np.linalg.inv(S_hat)
W = (Y - mu_hat).T @ (Y - mu_hat)
score = 0.5 * (Si @ W @ Si - m * Si)
num = np.zeros((3, 3))
h = 1e-6
for i in range(3):
    for j in range(3):
        E = np.zeros((3, 3))
        E[i, j] = h
        num[i, j] = (loglik(mu_hat, S_hat + E) - loglik(mu_hat, S_hat - E)) / (2 * h)
print("max |score| at MLE (analytic):", f"{np.abs(score).max():.2e}")
print("max |score| at MLE (numeric) :", f"{np.abs(num).max():.2e}")
print("loglik at MLE vs at true    :",
      f"{loglik(mu_hat, S_hat):.2f} >= {loglik(mu, Sigma):.2f}")
