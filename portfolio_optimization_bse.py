import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Step 1: Load Data
# Replace 'your_data.xlsx' with your actual file name
# Assumes: first column = Date, other columns = stock closing prices
data = pd.read_excel('your_data.xlsx', index_col=0, parse_dates=True)

# Step 2: Calculate Daily Returns
returns = data.pct_change().dropna()

# Step 3: Set up Markowitz Optimization
n = returns.shape[1]
mean_returns = returns.mean()
cov_matrix = returns.cov()
budget = 500000

# Variables: asset weights (fraction of budget in each stock)
w = cp.Variable(n)

# Objective: Maximize expected return (for a given budget)
expected_return = mean_returns.values @ w
# No short selling: weights >= 0
# Budget constraint: sum(weights) == 1 (will multiply by budget later)
constraints = [
    w >= 0,
    cp.sum(w) == 1
]

# Set up the optimization problem (maximize expected return)
prob = cp.Problem(cp.Maximize(expected_return), constraints)
prob.solve()

# Step 4: Get Results
optimal_weights = w.value
investment_per_stock = optimal_weights * budget
portfolio_expected_return = expected_return.value * budget

print("Optimal portfolio weights (fractions):")
for stock, weight in zip(returns.columns, optimal_weights):
    print(f"{stock}: {weight:.4f}")

print("\nInvestment per stock (₹):")
for stock, investment in zip(returns.columns, investment_per_stock):
    print(f"{stock}: ₹{investment:.2f}")

print(f"\nExpected portfolio return per period: ₹{portfolio_expected_return:.2f}")

# Optional: Visualize
plt.figure(figsize=(10,6))
plt.bar(returns.columns, optimal_weights)
plt.title("Optimal Portfolio Weights (No Short Selling)")
plt.ylabel("Weight")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()