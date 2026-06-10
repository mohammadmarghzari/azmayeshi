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