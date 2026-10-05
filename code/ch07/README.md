# Chapter 7 -- Estimation Principles: Moments, Maximum Likelihood and Information Geometry

All data are simulated (fixed seed 803) except `map_ridge.py`, which uses the diabetes
data bundled with scikit-learn. Run `python run_all.py ch07` from the book root to
regenerate the `.out` files and the figures in `figures/`.

| Script | Book section | What it demonstrates |
|---|---|---|
| `est_bias_variance.py` | 7.1 | Monte Carlo bias, variance and MSE of variance estimators (divisors n-1, n, n+1) |
| `mean_vs_median.py` | 7.1 | Efficiency of mean vs median under normal, t(3) and Cauchy data; figure `ch07_mean_median.pdf` |
| `mom_gamma_beta.py` | 7.2 | Method-of-moments estimators for gamma and beta families and their sampling properties |
| `gmm_iv_schooling.py` | 7.2 | Returns to schooling: OLS, IV, 2SLS, two-step efficient GMM, Hansen J-test, first-stage F |
| `iv_montecarlo.py` | 7.2 | Sampling distributions of OLS and IV, strong vs weak instrument; figure `ch07_iv_montecarlo.pdf` |
| `mle_closed_form.py` | 7.3 | Closed-form MLEs (Bernoulli, Poisson, normal, exponential) vs numerical; invariance |
| `mle_gamma_numeric.py` | 7.3 | Numerical gamma MLE in log-parameters with analytic gradient; figure `ch07_gamma_loglik.pdf` |
| `mle_glm.py` | 7.3 | Logit and probit take-up models, marginal effects; Poisson demand with exposure offset |
| `mle_separation.py` | 7.3 | Complete separation: non-existent logit MLE, misleading convergence, penalised fix |
| `fisher_crlb.py` | 7.4 | Cramer-Rao bound checks; observed vs expected information (Cauchy location) |
| `logit_se.py` | 7.4 | Logit standard errors: analytic Hessian, BFGS hess_inv, Monte Carlo coverage |
| `sandwich_qmle.py` | 7.4 | Poisson quasi-MLE with overdispersion: model-based vs sandwich standard errors |
| `trinity_tests.py` | 7.4 | Null distributions and power of Wald, LR and score tests; figure `ch07_trinity_qq.pdf` |
| `kl_pseudo_true.py` | 7.5 | KL divergence, local Fisher approximation, pseudo-true parameter; figure `ch07_kl_pseudotrue.pdf` |
| `natural_gradient.py` | 7.5 | Ordinary vs natural gradient ascent for the normal model; figure `ch07_natural_gradient.pdf` |
| `bayes_conjugate.py` | 7.6 | Beta-binomial, gamma-Poisson, normal-normal updating; Jeffreys prior; Bernstein-von Mises |
| `map_ridge.py` | 7.6 | MAP with a Gaussian prior equals ridge regression (diabetes data) |
| `bayes_grid.py` | 7.6 | Grid approximation of a two-parameter logit posterior; figure `ch07_bayes_grid.pdf` |
