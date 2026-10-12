# DEP 803 -- Chapter 0: Mathematical Refresher
# Lab 0, part 1: variables, lists, loops and functions (a loan schedule).
# (c) 2026 Stephen E. Moore. MIT Licence.

# variables: a name on the left, a value on the right
loan = 10_000.0        # GHS borrowed
rate = 0.24            # 24% a year
years = 3
print("loan:", loan, " rate:", rate, " type of years:", type(years).__name__)

# a list holds several values in order; indexing starts at 0
cash_flows = [-1000, 300, 400, 500]
print("first element:", cash_flows[0], " last element:", cash_flows[-1],
      " length:", len(cash_flows))


# a function packages a calculation so it can be reused
def annuity_payment(principal, r, n):
    """Equal annual payment that repays `principal` in n years at rate r."""
    return principal * r / (1 - (1 + r) ** (-n))


pay = annuity_payment(loan, rate, years)
print(f"annual payment: {pay:.2f}")

# a for-loop repeats the indented block once per year
balance = loan
print("year  interest  principal   balance")
for year in range(1, years + 1):
    interest = balance * rate
    principal_part = pay - interest
    balance = balance - principal_part
    shown = round(balance, 2) + 0.0      # + 0.0 turns -0.0 into 0.0
    print(f"{year:4d} {interest:9.2f} {principal_part:10.2f} {shown:9.2f}")

# if/else makes decisions
npv = sum(cf / 1.10**t for t, cf in enumerate(cash_flows))
decision = "accept" if npv > 0 else "reject"
print(f"NPV at 10%: {npv:.2f} -> {decision} the project")
