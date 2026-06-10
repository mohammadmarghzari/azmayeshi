import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Portfolio Analytics Pro",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# CUSTOM CSS (Carbon Blue Theme)
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0F172A;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

.block-container {
    padding-top: 1rem;
}

.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 12px;
    backdrop-filter: blur(10px);
}

h1,h2,h3 {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("📊 Portfolio Analytics Pro")

st.markdown("""
Professional Portfolio Optimization,
Sharpe Analysis,
and Covered Call Platform
""")

# ==========================================
# DEFAULT ASSETS
# ==========================================

DEFAULT_SYMBOLS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F"
}

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("⚙️ Settings")

selected_assets = st.sidebar.multiselect(
    "Select Assets",
    options=list(DEFAULT_SYMBOLS.keys()),
    default=["Bitcoin", "Ethereum"]
)

custom_symbol = st.sidebar.text_input(
    "Custom Yahoo Symbol",
    ""
)

period = st.sidebar.selectbox(
    "Analysis Period",
    [
        "1mo",
        "2mo",
        "3mo",
        "6mo",
        "1y"
    ]
)

risk_free_rate = st.sidebar.number_input(
    "Risk Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=4.5
)

num_portfolios = st.sidebar.slider(
    "Number of Portfolios",
    100,
    5000,
    500,
    100
)

# ==========================================
# BUILD SYMBOL LIST
# ==========================================

symbols = []

for asset in selected_assets:
    symbols.append(DEFAULT_SYMBOLS[asset])

if custom_symbol.strip():
    symbols.append(custom_symbol.strip().upper())

# ==========================================
# DATA DOWNLOAD
# ==========================================

@st.cache_data
def download_data(symbols, period):

    data = yf.download(
        symbols,
        period=period,
        auto_adjust=True,
        progress=False
    )

    return data

# ==========================================
# LOAD DATA
# ==========================================

if len(symbols) == 0:
    st.warning("Please select at least one asset.")
    st.stop()

try:

    raw_data = download_data(symbols, period)

    prices = raw_data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

except Exception as e:

    st.error(f"Error downloading data: {e}")
    st.stop()

# ==========================================
# PRICE CHART
# ==========================================

st.subheader("📈 Price History")

fig_price = px.line(
    prices,
    title="Historical Prices"
)

fig_price.update_layout(
    template="plotly_dark",
    height=600
)

st.plotly_chart(
    fig_price,
    use_container_width=True
)

# ==========================================
# RETURNS
# ==========================================

returns = prices.pct_change().dropna()

# ==========================================
# SHARPE RATIO
# ==========================================

def calculate_sharpe(series, rf):

    annual_return = series.mean() * 252

    annual_volatility = series.std() * np.sqrt(252)

    rf = rf / 100

    sharpe = (
        annual_return - rf
    ) / annual_volatility

    return (
        annual_return,
        annual_volatility,
        sharpe
    )

# ==========================================
# ANALYTICS TABLE
# ==========================================

analytics = []

for col in returns.columns:

    annual_return, annual_vol, sharpe = calculate_sharpe(
        returns[col],
        risk_free_rate
    )

    analytics.append({
        "Asset": col,
        "Annual Return %": round(
            annual_return * 100,
            2
        ),
        "Volatility %": round(
            annual_vol * 100,
            2
        ),
        "Sharpe Ratio": round(
            sharpe,
            3
        )
    })

analytics_df = pd.DataFrame(
    analytics
)

# ==========================================
# METRICS
# ==========================================

st.subheader("📊 Sharpe Analysis")

st.dataframe(
    analytics_df,
    use_container_width=True
)

# ==========================================
# SHARPE BAR CHART
# ==========================================

fig_sharpe = px.bar(
    analytics_df,
    x="Asset",
    y="Sharpe Ratio",
    title="Sharpe Ratio Comparison"
)

fig_sharpe.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    fig_sharpe,
    use_container_width=True
)

# ==========================================
# SECTION DIVIDER
# ==========================================

st.markdown("---")

st.header("🚀 Portfolio Optimization")
# ==========================================
# PORTFOLIO OPTIMIZATION
# ==========================================

if len(returns.columns) < 2:

    st.info(
        "Portfolio optimization requires at least 2 assets."
    )

else:

    mean_returns = returns.mean() * 252

    cov_matrix = returns.cov() * 252

    portfolio_returns = []
    portfolio_volatility = []
    portfolio_sharpe = []
    portfolio_weights = []

    # ======================================
    # RANDOM PORTFOLIOS
    # ======================================

    progress_bar = st.progress(0)

    for i in range(num_portfolios):

        weights = np.random.random(
            len(returns.columns)
        )

        weights /= np.sum(weights)

        port_return = np.sum(
            mean_returns * weights
        )

        port_vol = np.sqrt(
            np.dot(
                weights.T,
                np.dot(
                    cov_matrix,
                    weights
                )
            )
        )

        sharpe = (
            port_return -
            (risk_free_rate / 100)
        ) / port_vol

        portfolio_returns.append(
            port_return
        )

        portfolio_volatility.append(
            port_vol
        )

        portfolio_sharpe.append(
            sharpe
        )

        portfolio_weights.append(
            weights
        )

        if i % 10 == 0:
            progress_bar.progress(
                i / num_portfolios
            )

    progress_bar.empty()

    # ======================================
    # RESULTS DATAFRAME
    # ======================================

    portfolio_df = pd.DataFrame({
        "Return":
            portfolio_returns,
        "Volatility":
            portfolio_volatility,
        "Sharpe":
            portfolio_sharpe
    })

    # ======================================
    # BEST SHARPE PORTFOLIO
    # ======================================

    best_idx = portfolio_df[
        "Sharpe"
    ].idxmax()

    best_return = portfolio_returns[
        best_idx
    ]

    best_volatility = portfolio_volatility[
        best_idx
    ]

    best_sharpe = portfolio_sharpe[
        best_idx
    ]

    best_weights = portfolio_weights[
        best_idx
    ]

    # ======================================
    # METRICS
    # ======================================

    st.subheader(
        "🏆 Best Portfolio"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Expected Return",
            f"{best_return*100:.2f}%"
        )

    with c2:

        st.metric(
            "Volatility",
            f"{best_volatility*100:.2f}%"
        )

    with c3:

        st.metric(
            "Sharpe Ratio",
            f"{best_sharpe:.3f}"
        )

    # ======================================
    # ALLOCATION TABLE
    # ======================================

    st.subheader(
        "📋 Optimal Allocation"
    )

    allocation_df = pd.DataFrame({
        "Asset":
            returns.columns,
        "Weight %":
            np.round(
                best_weights * 100,
                2
            )
    })

    allocation_df = allocation_df.sort_values(
        "Weight %",
        ascending=False
    )

    st.dataframe(
        allocation_df,
        use_container_width=True
    )

    # ======================================
    # PIE CHART
    # ======================================

    fig_pie = px.pie(
        allocation_df,
        names="Asset",
        values="Weight %",
        title="Optimal Portfolio Allocation"
    )

    fig_pie.update_layout(
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    # ======================================
    # EFFICIENT FRONTIER
    # ======================================

    st.subheader(
        "📉 Efficient Frontier"
    )

    frontier = go.Figure()

    frontier.add_trace(
        go.Scatter(
            x=portfolio_volatility,
            y=portfolio_returns,
            mode="markers",
            marker=dict(
                size=6,
                color=portfolio_sharpe,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Sharpe"
                )
            ),
            name="Portfolios"
        )
    )

    frontier.add_trace(
        go.Scatter(
            x=[best_volatility],
            y=[best_return],
            mode="markers",
            marker=dict(
                size=18,
                symbol="star"
            ),
            name="Best Sharpe"
        )
    )

    frontier.update_layout(
        template="plotly_dark",
        title="Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        height=700
    )

    st.plotly_chart(
        frontier,
        use_container_width=True
    )

    # ======================================
    # WEIGHT VISUALIZATION
    # ======================================

    st.subheader(
        "📊 Weight Distribution"
    )

    fig_weight = px.bar(
        allocation_df,
        x="Asset",
        y="Weight %",
        title="Asset Weights"
    )

    fig_weight.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig_weight,
        use_container_width=True
    )

# ==========================================
# END OF OPTIMIZER SECTION
# ==========================================

st.markdown("---")

st.header(
    "📞 Covered Call Calculator"
)