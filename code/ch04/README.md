# Chapter 4 -- Probability Foundations, Random Variables and Transformations

All scripts use simulated data and a fixed seed (`np.random.default_rng(803)`).
Run them with `python run_all.py ch04` from the book root; outputs go to `*.out`,
figures to `../../figures/`.

| Script | Book section | What it demonstrates |
|---|---|---|
| `ch04_inclexcl.py` | 4.1 Probability spaces | Matching problem: inclusion-exclusion vs simulation; continuity of probability |
| `ch04_relfreq.py` | 4.1 Probability spaces | Relative frequencies of a six converge to 1/6; addition rule (figure) |
| `ch04_malaria.py` | 4.2 Conditional probability | PPV/NPV of a diagnostic test vs prevalence; sequential Bayes updating (figure used in slides) |
| `ch04_targeting.py` | 4.2 Conditional probability | Proxy means test: inclusion/exclusion errors via Bayes; coverage trade-off (figure) |
| `ch04_montyhall.py` | 4.2 Conditional probability | Monty Hall with k doors: stick vs switch |
| `ch04_naive_bayes.py` | 4.2 Conditional probability | Naive Bayes poverty classifier from counts, checked against scikit-learn |
| `ch04_ecdf.py` | 4.3 Random variables | Empirical CDF, Glivenko-Cantelli sup-distance, DKW band, sample quantiles (figure) |
| `ch04_mixed.py` | 4.3 Random variables | Zero-inflated expenditure: mixed CDF, generalised inverse, moments (figure) |
| `ch04_joint_discrete.py` | 4.4 Joint distributions | Joint/marginal/conditional PMFs of household size and poverty; independence check |
| `ch04_bivariate.py` | 4.4 Joint distributions | Bivariate normal log income and log consumption; conditional mean (figure) |
| `ch04_copula.py` | 4.4 Joint distributions | Gaussian copula, Sklar's theorem, invariance of rank correlations (figure) |
| `ch04_lognormal.py` | 4.5 Transformations | Change of variables: log-normal incomes from normal log incomes (figure) |
| `ch04_jacobian.py` | 4.5 Transformations | Jacobian method: total ~ Gamma, share ~ Uniform, independent |
| `ch04_convolution.py` | 4.5 Transformations | Numerical convolution: sums of exponentials are Gamma; sum of two dice (figure) |
| `ch04_order_stats.py` | 4.5 Transformations | Minimum and maximum of iid samples |
| `ch04_pit.py` | 4.5 Transformations | Probability integral transform and calibration of density forecasts (figure) |
| `ch04_prng.py` | 4.6 Generating random variables | Middle-square, RANDU planes, PCG64 seeds and spawned streams (figure) |
| `ch04_inverse.py` | 4.6 Generating random variables | Inverse transform sampling: exponential, Pareto, discrete |
| `ch04_boxmuller.py` | 4.6 Generating random variables | Box-Muller transform with moment and KS checks (figure) |
| `ch04_accept_reject.py` | 4.6 Generating random variables | Acceptance-rejection sampling from a Beta density (figure) |
