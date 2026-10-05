# DEP 803 -- Chapter 13: Problem Set Clinic and Applied Foundations Review
# Monte Carlo: ability bias and measurement error in OLS; an IV preview.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
R, n, beta_true = 1000, 2000, 0.08
gamma_ability, me_sd = 0.10, 1.5          # ability effect, measurement-error sd


def slope(x, y):
    xc = x - x.mean()
    return xc @ (y - y.mean()) / (xc @ xc)


def iv(z, x, y):
    zc = z - z.mean()
    return zc @ (y - y.mean()) / (zc @ (x - x.mean()))


est = np.empty((R, 4))
for r in range(R):
    ability = rng.standard_normal(n)
    z = (rng.random(n) < 0.5).astype(float)            # exposure to a reform
    educ = 9 + 1.0 * z + 1.5 * ability + 2.5 * rng.standard_normal(n)
    logw = (1 + beta_true * educ + gamma_ability * ability
            + 0.4 * rng.standard_normal(n))
    educ_obs = educ + me_sd * rng.standard_normal(n)  # survey recall error
    fs = slope(z, educ_obs)                             # first stage
    resid = educ_obs - educ_obs.mean() - fs * (z - z.mean())
    F = fs**2 / (resid @ resid / (n - 2) / ((z - z.mean()) @ (z - z.mean())))
    est[r] = [slope(educ, logw), slope(educ_obs, logw), iv(z, educ_obs, logw), F]

var_e = 1.5**2 + 2.5**2 + 0.25
plim_ols = beta_true + gamma_ability * 1.5 / var_e
plim_me = (beta_true * var_e + gamma_ability * 1.5) / (var_e + me_sd**2)
print(f"true return = {beta_true}, {R} samples of n = {n}")
print(f"{'estimator':<32}{'mean':>8}{'sd':>8}{'plim':>8}")
print(f"{'OLS, true schooling':<32}{est[:, 0].mean():8.4f}{est[:, 0].std():8.4f}"
      f"{plim_ols:8.4f}")
print(f"{'OLS, mismeasured schooling':<32}{est[:, 1].mean():8.4f}"
      f"{est[:, 1].std():8.4f}{plim_me:8.4f}")
print(f"{'IV (reform exposure)':<32}{np.median(est[:, 2]):8.4f}"
      f"{est[:, 2].std():8.4f}{beta_true:8.4f}")
print(f"median first-stage F = {np.median(est[:, 3]):.1f}")
cover = np.mean(np.abs(est[:, 2] - beta_true) < 1.96 * est[:, 2].std())
print(f"share of IV estimates within 1.96 sd of truth: {cover:.3f}")
print("(IV row reports the median: IV has heavy tails in finite samples)")
