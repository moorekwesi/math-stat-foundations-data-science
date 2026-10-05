# DEP 803 -- Chapter 8: Optimisation
# Numerical convexity checks: logit Hessian, ridge constants mu and L
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n, p = 1000, 5
X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
X[:, 1] *= 10                                   # a badly scaled covariate
y = (rng.uniform(size=n) < 0.4).astype(float)


def logit_hessian(b):
    """Hessian of the average logit negative log-likelihood."""
    pr = 1 / (1 + np.exp(-X @ b))
    w = pr * (1 - pr)
    return (X * w[:, None]).T @ X / n


print("1. Logit NLL: smallest Hessian eigenvalue at random points")
for k in range(5):
    b = rng.normal(scale=2, size=p)
    lam = np.linalg.eigvalsh(logit_hessian(b))
    print(f"   point {k}: min eig = {lam.min():.3e}, max eig = {lam.max():.3e}")

print("2. Ridge risk (1/2n)||y - Xb||^2 + (lam/2)||b||^2")
G = X.T @ X / n
eig = np.linalg.eigvalsh(G)
for lam in [0.0, 0.1, 1.0]:
    mu, L = eig.min() + lam, eig.max() + lam
    print(f"   lam = {lam:>4}: mu = {mu:8.4f}, L = {L:8.4f}, "
          f"kappa = L/mu = {L / mu:8.1f}")
Xs = X.copy()
Xs[:, 1:] = (Xs[:, 1:] - Xs[:, 1:].mean(0)) / Xs[:, 1:].std(0)
e2 = np.linalg.eigvalsh(Xs.T @ Xs / n)
print(f"   after standardising: kappa = {e2.max() / e2.min():.2f}")

print("3. Jensen check for f(x) = log(1 + exp(x)) on random pairs")
f = lambda t: np.logaddexp(0, t)
a, b = rng.normal(size=(2, 100000)) * 5
th = rng.uniform(size=100000)
gap = th * f(a) + (1 - th) * f(b) - f(th * a + (1 - th) * b)
print(f"   min of theta f(a) + (1-theta) f(b) - f(mix) = {gap.min():.2e}")

print("4. Cobb-Douglas u = x1^0.3 x2^0.5: concave? quasi-concave?")
def hess_cd(x1, x2, a=0.3, c=0.5):
    u = x1**a * x2**c
    return np.array([[a * (a - 1) * u / x1**2, a * c * u / (x1 * x2)],
                     [a * c * u / (x1 * x2), c * (c - 1) * u / x2**2]])
lam = np.linalg.eigvalsh(hess_cd(2.0, 3.0))
print(f"   Hessian eigenvalues at (2,3): {lam.round(4)} -> concave")
lam = np.linalg.eigvalsh(hess_cd(2.0, 3.0, a=0.9, c=0.8))
print(f"   with a=0.9, c=0.8 (sum > 1): {lam.round(4)} -> not concave")
x1 = rng.uniform(0.1, 5, (2, 50000))
x2 = rng.uniform(0.1, 5, (2, 50000))
u = lambda s, t: s**0.9 * t**0.8
mix1, mix2 = th[:50000] * x1[0] + (1 - th[:50000]) * x1[1], \
    th[:50000] * x2[0] + (1 - th[:50000]) * x2[1]
qc = u(mix1, mix2) - np.minimum(u(x1[0], x2[0]), u(x1[1], x2[1]))
print(f"   but min u(mix) - min(u_a, u_b) = {qc.min():.2e} >= 0"
      " -> quasi-concave")
