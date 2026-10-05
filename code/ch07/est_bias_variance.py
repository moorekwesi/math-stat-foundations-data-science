# DEP 803 -- Chapter 7: Estimation Principles
# Monte Carlo bias, variance and MSE of three estimators of a variance.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
sigma2, R = 4.0, 40000          # true variance, Monte Carlo replications
divisors = {"n-1 (S^2)": 1, "n (MLE)": 0, "n+1": -1}   # divisor = n - ddof

print(f"True sigma^2 = {sigma2}, normal data, R = {R} replications")
print(f"{'n':>4} {'divisor':>10} {'bias':>8} {'var':>8} {'MSE':>8}"
      f" {'bias^2+var':>11} {'MSE theory':>11}")
for n in (5, 20, 100):
    x = rng.normal(0.0, np.sqrt(sigma2), size=(R, n))
    ss = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    for name, ddof in divisors.items():
        d = n - ddof
        est = ss / d
        bias = est.mean() - sigma2
        var = est.var()
        mse = np.mean((est - sigma2) ** 2)
        # exact MSE: SS/sigma^2 ~ chi^2_{n-1}
        theory = sigma2**2 * (2 * (n - 1) + (n - 1 - d) ** 2) / d**2
        print(f"{n:>4} {name:>10} {bias:8.3f} {var:8.3f} {mse:8.3f}"
              f" {bias**2 + var:11.3f} {theory:11.3f}")
