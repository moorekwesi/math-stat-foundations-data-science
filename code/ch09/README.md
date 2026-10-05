# Chapter 9 -- Constrained Optimisation and Lagrange Multipliers in Policy Problems

Run all scripts from the book root with `python run_all.py ch09` (figures are written to
`figures/`, printed output to the matching `.out` file). All data are simulated or bundled
with scikit-learn; no numbers are real statistics of any country.

| Script | Book section | What it demonstrates |
|---|---|---|
| `ch09_lagrange_geometry.py` | 9.1 | Tangency of level set and budget line; gradients parallel at the optimum (figure) |
| `ch09_consumer_demand.py` | 9.1 | Cobb-Douglas and CES Marshallian demand: closed form vs SLSQP vs root-finding on the FOC |
| `ch09_envelope.py` | 9.2 | Envelope theorem (dV/dB = lambda), comparative statics, bordered Hessian (figure) |
| `ch09_kkt_examples.py` | 9.3 | Hand-solved KKT examples, corner solution with quasi-linear utility, LICQ failure (figure) |
| `ch09_duality.py` | 9.4 | Dual functions: zero gap for a convex QP, duality gap for 0-1 project selection (figure) |
| `ch09_diet_lp.py` | 9.4 | Least-cost nutritious diet with `linprog` (HiGHS); nutrient shadow prices, dual LP, parametric cost curve (figure) |
| `ch09_restricted_ls.py` | 9.5 | Restricted least squares, multiplier, F test of constant returns |
| `ch09_lasso_ridge.py` | 9.5 | Penalised vs constrained ridge and lasso on the diabetes data; lasso path (figure) |
| `ch09_svm_dual.py` | 9.5 | Soft-margin linear SVM solved through its dual QP, compared with scikit-learn (figure) |
| `ch09_projected_gradient.py` | 9.5 | Projected gradient, quadratic penalty and log-barrier methods (figure) |
| `ch09_budget_allocation.py` | 9.6 | District budget allocation with diminishing returns; water-filling KKT solution, expansion path (figure) |
| `ch09_social_planner.py` | 9.6 | Equity-weighted transfers with inequality aversion; fill-from-the-bottom solution |
| `ch09_poverty_targeting.py` | 9.6 | Poverty-gap-minimising regional transfers as an LP; shadow price of the budget (figure) |
| `ch09_ramsey.py` | 9.6 | Ramsey commodity taxes, inverse elasticity rule, marginal cost of public funds |
| `ch09_sensitivity.py` | 9.6 | Implicit-function-theorem sensitivities and Monte Carlo regret of an optimal allocation (figure) |
