# Chapter 10 -- Numerical Integration and Monte Carlo Simulation

Run all scripts from the book root with `python run_all.py ch10`
(outputs go to `*.out`, figures to `figures/`). All data are simulated.

| Script | Book section | What it demonstrates |
|---|---|---|
| `ch10_integrals_closed_form.py` | 10.1 | Consumer surplus, present value, lognormal mean, probit probability, beta-binomial marginal likelihood and posterior mean: closed forms vs `quad`; consumer-surplus figure |
| `ch10_newton_cotes.py` | 10.2 | Composite trapezoidal and Simpson rules; observed rates 2 and 4, and loss of rate for `sqrt(x)` |
| `ch10_gauss_quadrature.py` | 10.2 | Gauss-Legendre degree of exactness; Gauss-Hermite for expected CRRA utility and a mixed probit probability |
| `ch10_scipy_quad.py` | 10.2 | `quad` on an infinite range with a kink, `dblquad` for a bivariate normal probability, subdivision-limit warning |
| `ch10_curse_dimensionality.py` | 10.2 | Tensor-product Gauss rules vs Monte Carlo as dimension grows; cost of 10-point tensor grids |
| `ch10_mc_pi.py` | 10.3 | Monte Carlo estimate of pi with standard errors, CIs, a coverage check and a running-estimate figure |
| `ch10_mc_highdim.py` | 10.3 | Dimension-free rate but dimension-dependent variance; multinomial probit frequency simulator |
| `ch10_npv_risk.py` | 10.3 | Monte Carlo risk assessment of a hypothetical public investment (NPV distribution, P(NPV<0), sensitivity) |
| `ch10_variance_reduction.py` | 10.4 | Antithetic and control variates for E[exp(U)] and a rainfall index-insurance payout |
| `ch10_importance_sampling.py` | 10.4 | Importance sampling for P(Z>4); good, poor and optimal proposals; effective sample sizes |
| `ch10_stratified.py` | 10.4 | Stratified Monte Carlo; proportional and Neyman allocation in a simulated three-region survey |
| `ch10_qmc.py` | 10.4 | Scrambled Sobol and Halton (`scipy.stats.qmc`) vs Monte Carlo: RMSE rates, discrepancy, point-set figure |
| `ch10_sampling.py` | 10.5 | Inverse-CDF sampling (Pareto, discrete) and acceptance-rejection (Beta, truncated normal) |
| `ch10_mh_logit.py` | 10.5 | Random-walk Metropolis for a Bayesian logistic regression: 4 chains, ESS, split R-hat, policy quantities, trace/ACF figure |
| `ch10_gibbs.py` | 10.5 | Gibbs sampler for a bivariate normal; lag-1 autocorrelation rho^2 and AR(1) ESS theory |
| `ch10_hmc.py` | 10.5 | Minimal Hamiltonian Monte Carlo vs random-walk Metropolis on a correlated 10-d Gaussian |
| `ch10_sim_study.py` | 10.6 | ADEMP simulation study: classical vs HC1 confidence intervals for OLS, with Monte Carlo standard errors |
| `ch10_microsim.py` | 10.6 | Microsimulation of a PMT-targeted cash transfer on FGT poverty indices under parameter uncertainty |

Scripts with tight Python loops set `OPENBLAS_NUM_THREADS=1` before importing NumPy;
this avoids severe slow-downs from thread oversubscription on small matrices.
