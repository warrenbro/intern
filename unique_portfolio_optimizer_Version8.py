import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. Load Data ---
df = pd.read_excel("Original - Copy.xlsx")
close_cols = [col for col in df.columns if str(col).strip().lower().endswith('close')]
returns_df = df[close_cols].apply(pd.to_numeric, errors='coerce').pct_change().dropna()

# --- 2. Regime Detection (Simple: Volatility Clustering with KMeans) ---
window = 6  # months
vols = returns_df.rolling(window).std().dropna()
scaler = StandardScaler()
vols_scaled = scaler.fit_transform(vols)
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(vols_scaled)
regimes = kmeans.labels_
current_regime = regimes[-1]
regime_mask = (regimes == current_regime)
regime_returns = returns_df.iloc[-len(regimes):][regime_mask]

# --- 3. HRP Implementation ---
def correl_dist(corr):
    """Convert correlation matrix to distance matrix."""
    return np.sqrt(0.5 * (1 - corr))

def get_quasi_diag(link):
    """Quasi-diagonalization to order clustered items."""
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
    """Compute cluster variance (for HRP allocation)."""
    cov_ = cov[np.ix_(items, items)]
    w_ = np.ones(len(items)) / len(items)
    return float(np.dot(w_, np.dot(cov_, w_)))

def hrp_allocation(returns):
    """Compute Hierarchical Risk Parity weights."""
    corr = returns.corr().values
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

hrp_weights = hrp_allocation(regime_returns).reindex(close_cols).fillna(0).values

# --- 4. CVaR-constrained Optimization with Regularization ---
n_assets = len(close_cols)
mean_returns = np.nanmean(regime_returns, axis=0)
cov = np.cov(regime_returns, rowvar=False)
alpha = 0.05

w = cp.Variable(n_assets)
z = cp.Variable(regime_returns.shape[0])
VaR = cp.Variable()
portfolio_loss = -regime_returns.values @ w

# Regularization
reg_strength = 0.01

constraints = [
    cp.sum(w) == 1,
    w >= 0,
    z >= 0,
    z >= portfolio_loss - VaR,
]

CVaR = VaR + 1/(alpha * regime_returns.shape[0]) * cp.sum(z)

risk_aversion = 0.5  # 0 = only return, 1 = only risk
lambda_cvar = 0.7    # 0 = ignore CVaR, 1 = CVaR only

objective = cp.Maximize(
    (1 - risk_aversion) * (mean_returns @ w)
    - risk_aversion * cp.quad_form(w, cov)
    - lambda_cvar * CVaR
    - reg_strength * cp.norm(w - hrp_weights, 2)
)

prob = cp.Problem(objective, constraints)
prob.solve()

final_weights = w.value
final_weights = np.clip(final_weights, 0, 1)
final_weights /= final_weights.sum()

# --- 5. Output ---
print("Selected regime:", current_regime)
print("HRP initial weights:")
for asset, weight in zip(close_cols, hrp_weights):
    print(f"{asset}: {weight:.4f}")
print("\nOptimized weights (CVaR-constrained + regime, regularized):")
for asset, weight in zip(close_cols, final_weights):
    print(f"{asset}: {weight:.4f}")
print("\nStats:")
print("Expected monthly return:", float(mean_returns @ final_weights))
print("Variance:", float(final_weights @ cov @ final_weights))
print("CVaR (5%):", float(CVaR.value))
