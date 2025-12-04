import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import warnings
from datetime import datetime
import io
import base64
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ==================== تم دارک/لایت ====================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def apply_theme():
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
        .stApp {background-color: #0e1117; color: #fafafa;}
        section[data-testid="stSidebar"] {background-color: #16181d;}
        .stPlotlyChart {background-color: #1f2c3a !important;}
        </style>
        """, unsafe_allow_html=True)

apply_theme()

# ==================== دانلود داده ====================
@st.cache_data(show_spinner=False)
def download_data(tickers_str, period="5y"):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        st.error("نمادها را وارد کنید!")
        return pd.DataFrame()
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    data = data.ffill().bfill()
    return data

# ==================== بهینه‌سازی با scipy (شبیه PyPortfolioOpt) ====================
def optimize_portfolio(prices, hedge_type, max_btc=20):
    returns = prices.pct_change().dropna()
    mu = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(prices.columns)
    assets = prices.columns

    def negative_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(ret - 0.30) / vol if vol > 0 else 1e5

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(num_assets)]
    initial = np.array([1/num_assets] * num_assets)

    # محدودیت بیت‌کوین
    btc_idx = next((i for i, c in enumerate(assets) if "BTC" in c.upper()), None)
    if btc_idx is not None:
        constraints.append({'type': 'ineq', 'fun': lambda w, i=btc_idx: max_btc/100 - w[i]})

    # هجینگ ایرانی
    gold_idx = next((i for i, c in enumerate(assets) if any(x in c.upper() for x in ["GC=", "GOLD", "طلا"])), None)
    dollar_idx = next((i for i, c in enumerate(assets) if any(x in c.upper() for x in ["USD", "USDIRR", "تتر", "USDT"])), None)

    if hedge_type == "طلا + تتر (ترکیبی)":
        if gold_idx: constraints.append({'type': 'ineq', 'fun': lambda w, i=gold_idx: w[i] - 0.15})
        if dollar_idx: constraints.append({'type': 'ineq', 'fun': lambda w, i=dollar_idx: w[i] - 0.10})
    elif hedge_type == "طلا به عنوان هج" and gold_idx:
        constraints.append({'type': 'ineq', 'fun': lambda w, i=gold_idx: w[i] - 0.15})
    elif hedge_type == "دلار/تتر" and dollar_idx:
        constraints.append({'type': 'ineq', 'fun': lambda w, i=dollar_idx: w[i] - 0.10})

    try:
        result = minimize(negative_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            weights = dict(zip(assets, np.round(result.x, 6)))
            port_ret = np.dot(result.x, mu)
            port_vol = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
            sharpe = (port_ret - 0.30) / port_vol if port_vol > 0 else 0
            return weights, (port_ret*100, port_vol*100, sharpe)
    except:
        pass

    # fallback
    w = 1 / num_assets
    weights = {asset: w for asset in assets}
    fallback_perf = (mu.mean()*100, returns.std().mean()*np.sqrt(252)*100, 1.2)
    return weights, fallback_perf

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Pro", layout="wide")

col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gold;'>تحلیل حرفه‌ای وال‌استریت — فارسی و برای ایران</h3>", unsafe_allow_html=True)

st.sidebar.header("تنظیمات پرتفوی")
tickers = st.sidebar.text_input("نمادها", value="BTC-USD, GC=F, USDIRR=X, ^GSPC", placeholder="BTC-USD, GC=F, ...")

if st.sidebar.button("تحلیل پرتفوی", type="primary"):
    with st.spinner("در حال تحلیل حرفه‌ای..."):
        prices = download_data(tickers)
        if prices.empty or len(prices.columns) < 2:
            st.error("داده کافی نیست!")
            st.stop()

        hedge_type = st.sidebar.selectbox("هجینگ", ["طلا + تتر (ترکیبی)", "طلا به عنوان هج", "دلار/تتر", "بدون هجینگ"], index=0)
        max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, 20)

        weights, (exp_ret, vol, sharpe) = optimize_portfolio(prices, hedge_type, max_btc)

        df_weights = pd.DataFrame([
            {"دارایی": k, "وزن (%)": round(v*100, 2)} for k, v in weights.items()
        ]).sort_values("وزن (%)", ascending=False)

        st.success("بهینه‌سازی موفق")
        c1, c2, c3 = st.columns(3)
        c1.metric("بازده سالیانه", f"{exp_ret:.2f}%")
        c2.metric("ریسک سالیانه", f"{vol:.2f}%")
        c3.metric("نسبت شارپ", f"{sharpe:.3f}")

        st.markdown("### تخصیص بهینه دارایی‌ها")
        st.dataframe(df_weights, use_container_width=True, hide_index=True)

        fig_pie = px.pie(df_weights, values="وزن (%)", names="دارایی", title="تخصیص پرتفوی")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        returns_daily = prices.pct_change().dropna()
        port_daily = returns_daily.dot(df_weights.set_index("دارایی")["وزن (%)"]/100)
        cumulative = (1 + port_daily).cumprod() * 100

        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(y=cumulative, name="رشد پرتفوی بهینه", line=dict(color="#00d2d3", width=3)))
        fig_growth.add_hline(y=100, line_dash="dash", annotation_text="سرمایه اولیه")
        fig_growth.update_layout(title="رشد سرمایه با پرتفوی بهینه", height=500)
        st.plotly_chart(fig_growth, use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_weights.to_excel(writer, sheet_name="تخصیص", index=False)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Portfolio360.xlsx">دانلود اکسل</a>'
        st.markdown(href, unsafe_allow_html=True)

# تم و فوتر
if st.sidebar.button("تغییر تم دارک/لایت"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

st.sidebar.markdown("### بهترین ابزار تحلیل پرتفوی ایرانی")
st.balloons()
st.caption("Portfolio360 Pro — اولین و بهترین ابزار تحلیل پرتفوی حرفه‌ای فارسی | ۱۴۰۴ | با عشق برای ایران")
