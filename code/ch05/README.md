# Chapter 5 -- Common Distributions in Data Science and Econometrics; Likelihood

Run all scripts from the book root with `python run_all.py ch05` (figures go to `figures/`).
All data are simulated except `normal_reg_ls.py` (scikit-learn's bundled `load_diabetes`)
and the historical horse-kick counts in `poisson_limit.py`.

| Script | Book section | What it demonstrates |
|---|---|---|
| `poisson_limit.py` | 5.1 Discrete distributions | Binomial-to-Poisson total variation distance vs Le Cam bound; Bortkiewicz horse-kick data |
| `discrete_pmfs.py` | 5.1 Discrete distributions | Means/variances of Bernoulli, binomial, Poisson, geometric, negative binomial; multinomial; PMF figure |
| `overdispersion.py` | 5.1 Discrete distributions | Gamma-Poisson clinic visits; Poisson vs negative binomial fits and log-likelihoods |
| `unemployment_duration.py` | 5.2 Continuous distributions | Memorylessness of the exponential; Weibull hazards and duration dependence |
| `income_inequality.py` | 5.2 Continuous distributions | Log-normal income fit, Gini (sample vs formula), Lorenz curve, poverty headcount and gap |
| `pareto_tail.py` | 5.2 Continuous distributions | Pareto top tail, log-log survival plot, Hill estimator, top-1% share |
| `continuous_pdfs.py` | 5.2 Continuous distributions | Moments of uniform, normal, exponential, gamma, beta, log-normal, Pareto, Weibull |
| `sampling_dists.py` | 5.3 Sampling distributions | Chi-square, t and F built from normals; Cochran's theorem; t vs normal figure |
| `cauchy_mean.py` | 5.3 Sampling distributions | Running means fail for the Cauchy; mean of Cauchy draws is Cauchy |
| `mvn_cholesky.py` | 5.4 Multivariate normal | Cholesky simulation, conditional mean/variance (Schur complement), Mahalanobis distance |
| `expfam_moments.py` | 5.5 Exponential family and GLMs | Moments from derivatives of the log-partition function; sufficiency of the Poisson total |
| `glm_irls.py` | 5.5 Exponential family and GLMs | Logistic and Poisson regression by IRLS, checked against scikit-learn |
| `normal_reg_ls.py` | 5.6 Likelihood | Numerical Gaussian MLE equals OLS; sigma_ML = sqrt(RSS/n) |
| `likelihood_surface.py` | 5.6 Likelihood | Log-likelihood contours for normal (mu, sigma) and logistic (b0, b1); LR region |
| `fit_qq.py` | 5.6 Likelihood | Fitting duration models with `.fit`, AIC comparison, KS test, QQ plots |
| `beta_binomial.py` | 5.6 Likelihood | Bayesian preview: beta prior x binomial likelihood = beta posterior |
