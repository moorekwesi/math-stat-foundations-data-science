# Chapter 11 -- Resampling: the Bootstrap, Permutation Tests and Uncertainty Intervals

All data are simulated (or `load_diabetes` from scikit-learn); no numbers refer to any real
country. Run everything with `python run_all.py ch11` from the book root; figures are written
to `figures/`.

| Script | Book section | What it demonstrates |
|---|---|---|
| `ch11_coverage_sim.py` | 11.1 | Coverage of the 95% t-interval for a skewed (lognormal) income mean; figure of realised intervals |
| `ch11_ecdf_plugin.py` | 11.1 | ECDF, DKW band, plug-in mean, median, Gini and FGT poverty indices |
| `ch11_bootstrap_basic.py` | 11.2 | Bootstrap standard errors and bias for mean, median, Gini and a ratio vs Monte Carlo truth |
| `ch11_boot_reps.py` | 11.2 | Monte Carlo error of bootstrap se and quantiles as a function of B |
| `ch11_boot_failure.py` | 11.2 | Failure for the sample maximum (parametric bootstrap fix) and for infinite-variance means |
| `ch11_boot_ci.py` | 11.3 | Normal, percentile, basic, studentised and BCa intervals for a Gini; `scipy.stats.bootstrap` |
| `ch11_ci_coverage.py` | 11.3 | Monte Carlo coverage study of six intervals for a lognormal mean |
| `ch11_regression_boot.py` | 11.4 | Pairs, residual and wild (Rademacher) bootstrap vs classical and HC1 se under heteroskedasticity |
| `ch11_cluster_boot.py` | 11.4 | Cluster (enumeration-area) bootstrap, CR1 sandwich, ICC and Moulton factor |
| `ch11_wild_cluster.py` | 11.4 | Size of CR1 t-tests vs the wild cluster restricted bootstrap with few clusters |
| `ch11_block_boot.py` | 11.4 | Moving block bootstrap for the mean of an AR(1) series; block-length sensitivity |
| `ch11_randomisation_rct.py` | 11.5 | Exact and Monte Carlo randomisation tests for a school-feeding trial; Fisher interval |
| `ch11_permutation_corr.py` | 11.5 | Permutation test of a correlation, by hand and with `scipy.stats.permutation_test` |
| `ch11_power_sim.py` | 11.5 | Power of t-test and permutation tests (means, ranks) with normal and skewed outcomes |
| `ch11_multiple_testing.py` | 11.5 | FWER, FDR and power of Bonferroni, Holm and Benjamini-Hochberg |
| `ch11_jackknife.py` | 11.6 | Jackknife bias and variance vs bootstrap; inconsistency for the median |
| `ch11_cv.py` | 11.6 | Exact LOO shortcut, 5- and 10-fold CV for ridge regression on `load_diabetes` |
| `ch11_conformal.py` | 11.6 | Normal-theory, residual-bootstrap, split and locally weighted conformal prediction intervals |
| `ch11_forecast.py` | 11.6 | Pseudo-out-of-sample forecast evaluation, Diebold-Mariano test, bootstrap fan chart |

Environment: Python 3.9, NumPy 1.23, SciPy 1.9, Matplotlib 3.6, scikit-learn 1.1.
Each script uses a fixed seed (`np.random.default_rng(803)`) and is deterministic.
