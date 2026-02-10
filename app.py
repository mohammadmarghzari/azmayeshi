import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import yfinance as yf
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ==================== دانلود داده ====================
@st.cache_data(show_spinner=False)
def download_data(tickers_str, period="max"):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    data = {}
    failed = []
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, auto_adjust=True)
            if len(df) > 50 and "Close" in df.columns:
                data[t] = df["Close"]
            else:
                failed.append(t)
        except:
            failed.append(t)
    if not data:
        st.error("هیچ داده‌ای دانلود نشد.")
        return None
    prices = pd.DataFrame(data).ffill().bfill()
    if failed:
        st.sidebar.warning(f"دانلود نشد: {', '.join(failed)}")
    return prices

# ==================== توابع کمکی ====================
def calculate_recovery_time(ret_series):
    if len(ret_series) == 0: return 0
    cum = (1 + ret_series).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    recoveries = []
    in_dd = False
    start = None
    for i in range(1, len(cum)):
        if dd.iloc[i] < -0.01:
            if not in_dd:
                in_dd = True
                start = i
        elif in_dd:
            in_dd = False
            recoveries.append(i - start)
    return np.mean(recoveries) if recoveries else 0

def format_recovery(days):
    if days == 0 or np.isnan(days): return "بدون افت جدی"
    months = int(days / 21)
    years, months = divmod(months, 12)
    if years and months: return f"{years} سال و {months} ماه"
    if years: return f"{years} سال"
    if months: return f"{months} ماه"
    return "کمتر از ۱ ماه"

# ==================== تخصیص سرمایه ====================
def allocate_capital(weights, assets, total_usd):
    rate_toman = 200_000_000 / 1200
    df = pd.DataFrame({
        "دارایی": assets,
        "وزن (%)": np.round(weights * 100, 2),
        "دلار ($)": np.round(weights * total_usd, 2),
        "تومان": np.round(weights * total_usd * rate_toman, 0),
        "ریال": np.round(weights * total_usd * rate_toman * 10, 0)
    })
    return df.sort_values("وزن (%)", ascending=False)

# ==================== پیش‌بینی قیمت (Monte Carlo) ====================
def forecast_price_series(price_series, days=63, sims=400):
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    mu, sigma = log_ret.mean(), log_ret.std()
    last_price = price_series.iloc[-1]

    paths = np.zeros((days, sims))
    for i in range(sims):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * np.exp(mu + sigma * np.random.normal()))
        paths[:, i] = prices[1:]
    return paths

def plot_forecast(prices, asset):
    series = prices[asset]
    ma150 = series.rolling(150).mean()

    paths = forecast_price_series(series, 63)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series, name="قیمت واقعی"))
    fig.add_trace(go.Scatter(y=ma150, name="MA 150", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(
        y=np.percentile(paths, 50, axis=1),
        name="پیش‌بینی نرمال (۳ ماه)",
        line=dict(color="orange")
    ))
    fig.add_trace(go.Scatter(
        y=np.percentile(paths, 85, axis=1),
        name="سناریوی خوش‌بینانه",
        line=dict(dash="dot", color="green")
    ))
    fig.add_trace(go.Scatter(
        y=np.percentile(paths, 15, axis=1),
        name="سناریوی بدبینانه",
        line=dict(dash="dot", color="red")
    ))

    fig.update_layout(title=f"پیش‌بینی قیمت {asset}", height=500)
    return fig

# ==================== محاسبه پرتفوی ====================
@st.fragment
def calculate_portfolio():
    if "prices" not in st.session_state:
        st.info("لطفاً داده‌ها را دانلود کنید.")
        return
prices = st.session_state.prices
    returns = prices.pct_change().dropna()
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    rf = st.session_state.rf_rate / 100
    n = len(mean_ret)

    weights = np.ones(n) / n

    st.success("پرتفوی محاسبه شد")
    df_w = pd.DataFrame({"دارایی": prices.columns, "وزن (%)": weights * 100})
    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی"), use_container_width=True)

    # تخصیص سرمایه
    st.markdown("### 💰 تخصیص سرمایه")
    capital = st.number_input("کل سرمایه (دلار)", 100, 1_000_000, 1200)
    alloc = allocate_capital(weights, prices.columns, capital)
    st.dataframe(alloc, use_container_width=True)

    # پیش‌بینی
    st.markdown("### 🔮 پیش‌بینی قیمت دارایی‌ها")
    asset = st.selectbox("انتخاب دارایی", prices.columns)
    st.plotly_chart(plot_forecast(prices, asset), use_container_width=True)

# ==================== UI ====================
st.set_page_config(page_title="Portfolio360 Ultimate Pro", layout="wide")
st.markdown("<h1 style='text-align:center;color:#00d2d3'>Portfolio360 Ultimate Pro</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("دانلود داده")
    tickers = st.text_input("نمادها", "BTC-USD, GC=F, ETH-USD")
    if st.button("دانلود"):
        st.session_state.prices = download_data(tickers)
        st.rerun()

    st.header("تنظیمات")
    st.session_state.rf_rate = st.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, 18.0)

calculate_portfolio()
st.caption("Portfolio360 Ultimate Pro | Prediction Enabled")
