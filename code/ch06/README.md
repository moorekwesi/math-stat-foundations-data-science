# Chapter 6 -- Expectation, Covariance, Limit Theorems and Concentration

Run all scripts from the book root with `python run_all.py ch06`
(figures are written to `figures/`, printed output to `code/ch06/*.out`).
All data are simulated; no number is a statistic of any real country.

| Script | Book section | What it demonstrates |
|---|---|---|
| `ch06_expectation_lotus.py` | 6.1 Expectation | Discrete and continuous expectations; quadrature, LOTUS and Monte Carlo; E[1/X] vs 1/E[X] |
| `ch06_jensen_ce.py` | 6.1 Expectation | Jensen's inequality, CRRA certainty equivalents, risk premium and value of insurance (figure `ch06_jensen.pdf`) |
| `ch06_mgf_cf.py` | 6.1 Expectation | Moments from the MGF by differentiation; empirical CF of Cauchy data where the MGF fails |
| `ch06_covariance_psd.py` | 6.2 Covariance | PSD covariance matrices, variance of a composite index, minimum-variance weights, a non-PSD "correlation" matrix |
| `ch06_simpson.py` | 6.2 Covariance | Simpson's paradox for a simulated training programme (figure `ch06_simpson.pdf`) |
| `ch06_iterated_expectations.py` | 6.3 Conditional expectation | Law of iterated expectations and within/between variance decomposition |
| `ch06_cef_blp.py` | 6.3 Conditional expectation | CEF as best predictor vs linear projection; orthogonality of errors (figure `ch06_cef.pdf`) |
| `ch06_ate_regression.py` | 6.3 Conditional expectation | Difference in means = OLS on a dummy; ATE under randomisation, ATT plus selection bias otherwise |
| `ch06_markov_chebyshev.py` | 6.4 LLN | Exact tails vs Markov and Chebyshev bounds; the extremal three-point law |
| `ch06_running_means.py` | 6.4 LLN | Running means for Bernoulli, exponential, Pareto(1.5) and Cauchy laws (figure `ch06_lln.pdf`) |
| `ch06_wlln_prob.py` | 6.4 LLN | Convergence in probability of the mean: Monte Carlo vs Chebyshev, MSE = var/n |
| `ch06_clt_sampling.py` | 6.5 CLT | Sampling distributions of standardised means, KS distance vs Berry-Esseen bound (figure `ch06_clt.pdf`) |
| `ch06_delta_method.py` | 6.5 CLT | Delta-method standard errors for a ratio of means and an arc elasticity, with coverage |
| `ch06_pareto_clt.py` | 6.5 CLT | Coverage of CLT intervals for Pareto incomes; heavy tails slow convergence (figure `ch06_pareto.pdf`) |
| `ch06_concentration_bounds.py` | 6.6 Concentration | Exact binomial tail vs CLT, Chebyshev, Hoeffding and Chernoff bounds (figure `ch06_concentration.pdf`) |
| `ch06_sample_size.py` | 6.6 Concentration | Survey sample sizes (CLT, Chebyshev, Hoeffding), union bound across regions, design effect |
| `ch06_union_bound.py` | 6.6 Concentration | Family-wise error with many tests, Bonferroni, growth of the maximum deviation |
| `ch06_high_dim.py` | 6.6 Concentration | Norm concentration and near-orthogonality of random vectors (figure `ch06_highdim.pdf`) |
