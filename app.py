import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from options.black_scholes import call_price
from options.greeks import greeks
from scipy.stats import norm

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

    T = expiration_days / 365

sigma = iv / 100

bs_price = call_price(
    cc_stock_price,
    cc_strike_price,
    T,
    risk_free_rate / 100,
    sigma
)
greek_values = greeks(
    cc_stock_price,
    cc_strike_price,
    T,
    risk_free_rate / 100,
    sigma
)
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
# ==========================================
# COVERED CALL SIDEBAR INPUTS
# ==========================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Option Parameters"
)

iv = st.sidebar.slider(
    "Implied Volatility %",
    1,
    300,
    60
)

expiration_days = st.sidebar.slider(
    "Days To Expiration",
    1,
    365,
    30
)
st.sidebar.markdown("---")
st.sidebar.header("📞 Covered Call Inputs")

cc_stock_price = st.sidebar.number_input(
    "Current Stock Price",
    min_value=0.01,
    value=100.0,
    step=1.0
)

cc_strike_price = st.sidebar.number_input(
    "Strike Price",
    min_value=0.01,
    value=110.0,
    step=1.0
)

cc_premium = st.sidebar.number_input(
    "Premium Received",
    min_value=0.0,
    value=3.0,
    step=0.1
)

cc_shares = st.sidebar.number_input(
    "Shares Owned",
    min_value=100,
    value=100,
    step=100
)

cc_days = st.sidebar.number_input(
    "Days To Expiration",
    min_value=1,
    value=30
)

commission = st.sidebar.number_input(
    "Commission",
    min_value=0.0,
    value=0.0
)

# ==========================================
# COVERED CALL CALCULATIONS
# ==========================================

premium_income = cc_premium * cc_shares

max_profit = (
    (cc_strike_price - cc_stock_price)
    * cc_shares
) + premium_income - commission

breakeven = (
    cc_stock_price
    - cc_premium
)

capital_required = (
    cc_stock_price
    * cc_shares
)

roi = (
    max_profit
    / capital_required
) * 100

annualized_return = (
    roi
    * (365 / cc_days)
)

# ==========================================
# COVERED CALL METRICS
# ==========================================
st.subheader(
    "⚡ Option Analytics"
)

g1, g2, g3, g4, g5 = st.columns(5)

with g1:

    st.metric(
        "BS Price",
        f"${bs_price:.2f}"
    )

with g2:

    st.metric(
        "Delta",
        f"{greek_values['Delta']:.3f}"
    )

with g3:

    st.metric(
        "Gamma",
        f"{greek_values['Gamma']:.4f}"
    )

with g4:

    st.metric(
        "Theta",
        f"{greek_values['Theta']:.2f}"
    )

with g5:

    st.metric(
        "Vega",
        f"{greek_values['Vega']:.2f}"
    )
st.subheader(
    "📊 Covered Call Metrics"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Max Profit",
        f"${max_profit:,.2f}"
    )

with col2:
    st.metric(
        "Breakeven",
        f"${breakeven:,.2f}"
    )

with col3:
    st.metric(
        "ROI",
        f"{roi:.2f}%"
    )

with col4:
    st.metric(
        "Annualized Return",
        f"{annualized_return:.2f}%"
    )

# ==========================================
# PAYOFF FUNCTION
# ==========================================

def covered_call_payoff(
    stock_price_expiration,
    stock_price_today,
    strike,
    premium,
    shares
):

    stock_pnl = (
        stock_price_expiration
        - stock_price_today
    ) * shares

    call_pnl = np.where(
        stock_price_expiration > strike,
        -(stock_price_expiration - strike)
        * shares,
        0
    )

    premium_income = (
        premium
        * shares
    )

    return (
        stock_pnl
        + call_pnl
        + premium_income
    )

# ==========================================
# PAYOFF CURVE
# ==========================================

price_range = np.linspace(
    cc_stock_price * 0.5,
    cc_stock_price * 1.8,
    300
)

payoff = covered_call_payoff(
    price_range,
    cc_stock_price,
    cc_strike_price,
    cc_premium,
    cc_shares
)

# ==========================================
# PAYOFF CHART
# ==========================================

st.subheader(
    "📈 Covered Call Payoff Diagram"
)

payoff_fig = go.Figure()

payoff_fig.add_trace(
    go.Scatter(
        x=price_range,
        y=payoff,
        mode="lines",
        name="Covered Call"
    )
)

payoff_fig.add_hline(
    y=0,
    line_dash="dash"
)

payoff_fig.add_vline(
    x=cc_strike_price,
    line_dash="dash"
)

payoff_fig.update_layout(
    template="plotly_dark",
    height=650,
    title="Profit / Loss At Expiration",
    xaxis_title="Underlying Price At Expiration",
    yaxis_title="Profit / Loss ($)"
)

st.plotly_chart(
    payoff_fig,
    use_container_width=True
)

# ==========================================
# SCENARIO TABLE
# ==========================================

st.subheader(
    "📋 Scenario Analysis"
)

scenario_prices = [
    cc_stock_price * 0.7,
    cc_stock_price * 0.8,
    cc_stock_price * 0.9,
    cc_stock_price,
    cc_stock_price * 1.1,
    cc_stock_price * 1.2,
    cc_stock_price * 1.3,
]

scenario_profit = []

for price in scenario_prices:

    pnl = covered_call_payoff(
        price,
        cc_stock_price,
        cc_strike_price,
        cc_premium,
        cc_shares
    )

    scenario_profit.append(
        round(float(pnl), 2)
    )

scenario_df = pd.DataFrame({
    "Underlying Price":
        np.round(
            scenario_prices,
            2
        ),
    "Profit/Loss":
        scenario_profit
})

st.dataframe(
    scenario_df,
    use_container_width=True
)

# ==========================================
# LIVE PRICE SIMULATOR
# ==========================================

st.subheader(
    "🎯 Expiration Simulator"
)

sim_price = st.slider(
    "Move Underlying Price",
    min_value=float(
        cc_stock_price * 0.5
    ),
    max_value=float(
        cc_stock_price * 1.8
    ),
    value=float(
        cc_stock_price
    ),
)

sim_profit = covered_call_payoff(
    sim_price,
    cc_stock_price,
    cc_strike_price,
    cc_premium,
    cc_shares
)

st.metric(
    "Simulated Profit/Loss",
    f"${float(sim_profit):,.2f}"
)

# ==========================================
# END COVERED CALL
# ==========================================

st.markdown("---")
st.success(
    "Portfolio Analytics Pro Loaded Successfully"
)