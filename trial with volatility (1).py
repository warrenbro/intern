import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# --- 1. Load Data ---
df = pd.read_excel("Original - Copy.xlsx")
tickers = sorted(set(col.rsplit(' ', 1)[0] for col in df.columns if col.endswith('Close')))
interval_low, interval_high = [], []

for ticker in tickers:
    H = df[f"{ticker} High"].values
    L = df[f"{ticker} Low"].values
    O = df[f"{ticker} Open"].values
    C = df[f"{ticker} Close"].values

    C_prev = np.roll(C, 1)
    C_prev[0] = O[0]
    x = np.minimum(O, C_prev)
    y = np.maximum(O, C_prev)
    U = (H - x) / x
    L_ = (L - y) / y
    interval_high.append(U)
    interval_low.append(L_)

interval_low = np.array(interval_low).T
interval_high = np.array(interval_high).T
interval_mid = (interval_low + interval_high) / 2

returns_df = pd.DataFrame(interval_mid, columns=tickers)

# --- 2. HMM Regime Detection ---
X = returns_df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
hmm_model.fit(X_scaled)
states = hmm_model.predict(X_scaled)
current_state = states[-1]
mask = (states == current_state)

R_mid = returns_df[mask]
R_low = pd.DataFrame(interval_low, columns=tickers)[mask]
R_high = pd.DataFrame(interval_high, columns=tickers)[mask]

# --- 3. HRP Allocation ---
def correl_dist(corr):
    return np.sqrt(0.5 * (1 - corr))

def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = [link[-1, 0], link[-1, 1]]
    num_items = link[-1, 3]
    while any(i >= num_items for i in sort_ix):
        new_sort_ix = []
        for i in sort_ix:
            if i >= num_items:
                i = int(i)
                new_sort_ix += [link[i - num_items, 0], link[i - num_items, 1]]
            else:
                new_sort_ix.append(i)
        sort_ix = new_sort_ix
    return sort_ix

def get_cluster_var(cov, items):
    cov_ = cov[np.ix_(items, items)]
    w_ = np.ones(len(items)) / len(items)
    return float(np.dot(w_, np.dot(cov_, w_)))

def hrp_allocation(returns):
    corr = returns.corr().values
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    cov = returns.cov().values
    dist = correl_dist(corr)
    link = linkage(squareform(dist), 'single')
    sort_ix = get_quasi_diag(link)
    sorted_tickers = [returns.columns[i] for i in sort_ix]

    weights = pd.Series(1.0, index=sorted_tickers)
    clusters = [sorted_tickers]

    while len(clusters) > 0:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            c1 = cluster[:split]
            c2 = cluster[split:]
            c1_idx = [sorted_tickers.index(x) for x in c1]
            c2_idx = [sorted_tickers.index(x) for x in c2]
            var1 = get_cluster_var(cov, c1_idx)
            var2 = get_cluster_var(cov, c2_idx)
            alpha = 1 - var1 / (var1 + var2)
            weights[c1] *= alpha
            weights[c2] *= 1 - alpha
            new_clusters += [c1, c2]
        clusters = new_clusters
    return weights / weights.sum()

hrp_weights = hrp_allocation(R_mid).reindex(tickers).fillna(0).values

import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

# Compute correlation and convert to distance
corr_matrix = R_mid.corr()
distance_matrix = correl_dist(corr_matrix.values)
linkage_matrix = linkage(squareform(distance_matrix), method='single')

# Plot dendrogram
plt.figure(figsize=(12, 5))
dendrogram(linkage_matrix, labels=corr_matrix.columns, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram (HRP)")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# --- 4. Interval + CVaR + Volatility Optimization ---
if R_mid.empty or R_low.empty or R_high.empty:
    print("Error: Empty data after HMM filter.")
    exit(1)

n_assets = len(tickers)
T = R_low.shape[0]

w = cp.Variable(n_assets)
portfolio_return_lows = R_low.values @ w
portfolio_return_highs = R_high.values @ w
portfolio_return_mid = R_mid.values @ w

alpha = 0.95
z = cp.Variable(T)
VaR = cp.Variable()
cvar_term = VaR + (1 / ((1 - alpha) * T)) * cp.sum(z)
reg_strength = 0.01
vol_penalty = cp.sum_squares(portfolio_return_mid - cp.sum(portfolio_return_mid)/T)
hrp_reg = reg_strength * cp.norm(w - hrp_weights, 2)

min_portfolio_return = cp.minimum(*[portfolio_return_lows[t] for t in range(T)])

constraints = [cp.sum(w) == 1, w >= 0]
for t in range(T):
    constraints += [
        portfolio_return_lows[t] <= portfolio_return_highs[t],
        z[t] >= 0,
        z[t] >= -portfolio_return_mid[t] - VaR
    ]

objective = cp.Maximize(min_portfolio_return - 0.06 * cvar_term - 0.14 * hrp_reg -  0.2 * vol_penalty)
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)

if w.value is None:
    print("❌ Optimization failed.")
    exit(1)

final_weights = np.clip(w.value, 0, 1)
final_weights /= final_weights.sum()

# --- 5. Output ---
print("\n✅ Optimization Successful.")
print(f"Selected HMM regime: {current_state}")
print("\nHRP Initial Weights:")
for asset, weight in zip(tickers, hrp_weights):
    print(f"{asset}: {weight:.4f}")

print("\nOptimized Final Weights (Interval Robust + CVaR + Volatility):")
for asset, weight in zip(tickers, final_weights):
    print(f"{asset}: {weight:.4f}")

print("\nWorst-case Monthly Portfolio Return (min over intervals):", float(min_portfolio_return.value))
for t in range(T):
    print(f"Month {t+1}: Portfolio return interval = ["
          f"{portfolio_return_lows.value[t]:.4f}, {portfolio_return_highs.value[t]:.4f}]")

# Plot interval returns
port_L = interval_low @ final_weights
port_U = interval_high @ final_weights
port_mid = (port_L + port_U) / 2
months = np.arange(1, len(port_L) + 1)

plt.figure(figsize=(14, 6))
plt.fill_between(months, port_L, port_U, color='skyblue', alpha=0.4, label='Return Interval')
plt.plot(months, port_mid, color='blue', linestyle='--', label='Midpoint Return')
plt.axhline(y=np.min(port_L), color='red', linestyle='--', linewidth=1.5, label='Worst-case Return')
plt.title("Monthly Portfolio Return Intervals with Worst-case Return", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Return", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

def walk_forward_validation(data_low, data_high, data_mid, tickers,
                            lambda_cvar,
                            lambda_hrp,
                            lambda_vol,
                            window_size=36,
                            test_size=1,
                            step_size=1):

    n_assets = len(tickers)
    alpha = 0.95

    returns = []
    all_metrics = []

    for start in range(0, len(data_mid) - window_size - test_size + 1, step_size):
        end = start + window_size

        train_mid = data_mid.iloc[start:end]
        train_low = data_low.iloc[start:end]
        train_high = data_high.iloc[start:end]

        # HRP weights (used for regularization)
        cov = train_mid.cov().values
        w_hrp = np.ones(n_assets) / n_assets  # Simplified HRP prior

        # Optimization variables
        w = cp.Variable(n_assets)
        port_ret_low = train_low.values @ w
        port_ret_high = train_high.values @ w
        port_ret_mid = train_mid.values @ w

        T = len(train_mid)
        z = cp.Variable(T)
        VaR = cp.Variable()

        cvar = VaR + (1 / ((1 - alpha) * T)) * cp.sum(z)
        volatility = cp.sum_squares(port_ret_mid - cp.sum(port_ret_mid)/T)
        reg_hrp = cp.norm(w - w_hrp, 2)

        min_return = cp.minimum(*[port_ret_low[t] for t in range(T)])
        constraints = [cp.sum(w) == 1, w >= 0]
        for t in range(T):
            constraints += [
                port_ret_low[t] <= port_ret_high[t],
                z[t] >= 0,
                z[t] >= -port_ret_mid[t] - VaR
            ]

        objective = cp.Maximize(min_return - lambda_cvar * cvar - lambda_hrp * reg_hrp - lambda_vol * volatility)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        if w.value is None:
            continue

        weights = np.clip(w.value, 0, 1)
        weights /= weights.sum()

        # Evaluate on next period
        r_low_next = data_low.iloc[end]
        r_high_next = data_high.iloc[end]
        r_mid_next = data_mid.iloc[end]

        port_return = np.dot((r_low_next + r_high_next) / 2, weights)
        returns.append(port_return)

    returns = np.array(returns)

    if len(returns) == 0:
        print("❌ No returns generated. Check data dimensions or constraints.")
        return {}

    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe = mean_return / std_dev if std_dev != 0 else 0

    # Max Drawdown
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns) / peak
    max_drawdown = np.max(drawdown)

    # CVaR 95%
    sorted_returns = np.sort(returns)
    var_95 = np.percentile(returns, 5)
    cvar_95 = np.mean(sorted_returns[sorted_returns <= var_95])

    metrics = {
        "Mean Return": round(mean_return, 4),
        "Std Dev": round(std_dev, 4),
        "Sharpe Ratio": round(sharpe, 4),
        "Max Drawdown": round(max_drawdown, 4),
        "CVaR 95%": round(cvar_95, 4)
    }

    print("\nWalk-Forward Validation Summary:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics

metrics = walk_forward_validation(R_low, R_high, R_mid, tickers,
                            lambda_cvar=0.06,
                            lambda_hrp=0.14,
                            lambda_vol=0.2,
                            window_size=36,
                            test_size=1,
                            step_size=1)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# --- 1. Split data into train/test ---
train_size = 114
test_size = 20

train_low = interval_low[:train_size]
train_high = interval_high[:train_size]
train_mid = interval_mid[:train_size]

test_low = interval_low[train_size:]
test_high = interval_high[train_size:]
test_mid = interval_mid[train_size:]

# --- 2. Compute returns for custom model using final_weights ---
port_L_test = test_low @ final_weights
port_U_test = test_high @ final_weights
port_mid_test = (port_L_test + port_U_test) / 2

# Training returns for model
port_L_train = train_low @ final_weights
port_U_train = train_high @ final_weights
port_mid_train = (port_L_train + port_U_train) / 2

# --- 3. Equal-weight benchmark ---
n_assets = interval_low.shape[1]
equal_weights = np.ones(n_assets) / n_assets

port_L_eq_train = train_low @ equal_weights
port_U_eq_train = train_high @ equal_weights
port_mid_eq_train = (port_L_eq_train + port_U_eq_train) / 2

port_L_eq_test = test_low @ equal_weights
port_U_eq_test = test_high @ equal_weights
port_mid_eq_test = (port_L_eq_test + port_U_eq_test) / 2

# --- 4. Baseline: Mean of individual asset returns ---
mean_asset_return_test = test_mid.mean(axis=1)
mean_asset_return_train = train_mid.mean(axis=1)

# --- 5. Evaluation function ---
def evaluate_performance(returns):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = mean_return / std_return if std_return > 0 else np.nan
    max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
    cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)])
    return {
        "Mean Return": mean_return,
        "Std Dev": std_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "CVaR 95%": cvar_95
    }

# --- 6. Evaluate and display ---
print("\n📊 Performance Summary (Training and Testing):")

print("\n🔷 Custom Model - Training (114 months):")
for k, v in evaluate_performance(port_mid_train).items():
    print(f"{k}: {v:.4f}")

print("\n🔷 Custom Model - Testing (20 months):")
for k, v in evaluate_performance(port_mid_test).items():
    print(f"{k}: {v:.4f}")

print("\n🟩 Equal-Weight Benchmark - Training:")
for k, v in evaluate_performance(port_mid_eq_train).items():
    print(f"{k}: {v:.4f}")

print("\n🟩 Equal-Weight Benchmark - Testing:")
for k, v in evaluate_performance(port_mid_eq_test).items():
    print(f"{k}: {v:.4f}")

print("\n🟨 Baseline (Mean of Individual Asset Returns - Testing):")
for k, v in evaluate_performance(mean_asset_return_test).items():
    print(f"{k}: {v:.4f}")

# --- 7. Plot Comparison on Testing Period ---
months_test = np.arange(1, len(port_mid_test) + 1)
plt.figure(figsize=(12, 5))
plt.plot(months_test, port_mid_test, label="Custom Model", color='blue')
plt.plot(months_test, port_mid_eq_test, label="Equal Weight", linestyle='--', color='green')
plt.plot(months_test, mean_asset_return_test, label="Mean Asset Return", linestyle=':', color='orange')

plt.fill_between(months_test, port_L_test, port_U_test, color='skyblue', alpha=0.2, label='Custom Model Interval')
plt.title("Out-of-sample Mid Returns (20 months)")
plt.xlabel("Month")
plt.ylabel("Return")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
