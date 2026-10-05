# Chapter 13 -- Problem Set Clinic and Applied Foundations Review

All data are simulated (no downloads). Run from the book root with
`python run_all.py ch13`; figures are written to `figures/`.

| Script | Book section | What it demonstrates |
|---|---|---|
| `ch13_verify_formulas.py` | 13.1 How to solve problems | Monte Carlo check of Var(s^2) (normal vs exponential); analytic vs finite-difference logistic gradient; step-size error curve (`ch13_fd_error.pdf`) |
| `ch13_reproducible.py` | 13.1 Reproducible workflow | Seeds, `SeedSequence.spawn` streams, result fingerprints (SHA-256), recording the environment |
| `ch13_mincer_ols.py` | 13.2 Case study 1 | Mincer regression on a simulated labour force survey: OLS via QR, FWL and within estimator for regional fixed effects, classical vs HC1 standard errors by hand |
| `ch13_mincer_bootstrap.py` | 13.2 Case study 1 | Pairs and wild (Rademacher) bootstrap intervals for the return to schooling (`ch13_mincer_boot.pdf`) |
| `ch13_mincer_iv.py` | 13.2 Case study 1 | Monte Carlo of ability bias, measurement-error attenuation and an IV estimator |
| `ch13_pmt_logit.py` | 13.3 Case study 2 | PCA asset index vs logit proxy-means test fitted by Newton/IRLS, checked against scikit-learn |
| `ch13_pmt_targeting.py` | 13.3 Case study 2 | L1/L2 logit chosen by CV, inclusion/exclusion errors at a budget threshold, ROC/AUC, fairness audit by area (`ch13_pmt_roc.pdf`) |
| `ch13_forecast.py` | 13.4 Case study 3 | Factor-structured panel; expanding-window AR, PCA diffusion index, ridge and lasso forecasts; Diebold-Mariano and block-bootstrap comparison (`ch13_forecast_csfe.pdf`) |
| `ch13_budget.py` | 13.5 Case study 4 | Budget allocation with uncertain concave returns: sample average approximation, SLSQP, risk adjustment, KKT shadow price, sensitivity to the budget (`ch13_budget_shares.pdf`) |
| `ch13_review_checks.py` | 13.6 Comprehensive review | Numerical checks of the ten worked exam-style problems covering Chapters 1-12 |
