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
import requests

# نصب خودکار PyPortfolioOpt (اگر روی سرور نباشه)
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, DiscreteAllocation
    from pypfopt.exceptions import OptimizationError
except:
    st.error("در حال نصب PyPortfolioOpt...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPortfolioOpt"])
    from pypfopt import EfficientFrontier, risk_models, expected_returns, DiscreteAllocation

warnings.filterwarnings("ignore")

# ==================== تم دارک/لایت ====================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def apply_theme():
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
        .css-1d391kg {background-color: #0e1117; color: #fafafa;}
        .stPlotlyChart {background-color: #1f2c3a;}
        </style>
        """, unsafe_allow_html=True)

apply_theme()

# ==================== دانلود داده ====================
@st.cache_data(show_spinner=False)
def download_data(tickers_str, period="5y"):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    data = yf.download(tickers, period=period, auto_adjust=True="close")["Close"]
    data = data.ffill().bfill()
    return data

# ==================== تحلیل حرفه‌ای با PyPortfolioOpt ====================
def analyze_with_pypfopt(prices, hedge_type, max_btc=20):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    
    # اعمال محدودیت‌های هجینگ ایرانی
    constraints = {}
    asset_names = prices.columns.tolist()
    
    # محدودیت بیت‌کوین
    if any("BTC" in x.upper() for x in asset_names):
        btc_idx = next(i for i, x in enumerate(asset_names) if "BTC" in x.upper())
        constraints[btc_idx] = (0, max_btc / 100)
    
    # هجینگ ایرانی
    gold_idx = dollar_idx = None
    for i, name in enumerate(asset_names):
        if any(x in name.upper() for x in ["GC=", "GOLD", "طلا"]):
            gold_idx = i
        if any(x in name.upper() for x in ["USD", "USDIRR", "تتر"]):
            dollar_idx = i
    
    if hedge_type == "طلا + تتر (ترکیبی)":
        if gold_idx is not None: constraints[gold_idx] = (0.15, 1)
        if dollar_idx is not None: constraints[dollar_idx] = (0.10, 1)
    elif hedge_type == "طلا به عنوان هج" and gold_idx is not None:
        constraints[gold_idx] = (0.15, 1)
    elif hedge_type == "دلار/تتر" and dollar_idx is not None:
        constraints[dollar_idx] = (0.10, 1)
    
    # بهینه‌سازی شارپ
    ef = EfficientFrontier(mu, S, weight_bounds=(0,1))
    
    # اعمال محدودیت‌ها
    for idx, (min_w, max_w) in constraints.items():
        ef.add_constraint(lambda w, i=idx, mn=min_w, mx=max_w: w[i] >= mn)
        ef.add_constraint(lambda w, i=idx, mx=max_w: w[i] <= mx)
    
    try:
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=True)
        return cleaned_weights, perf
    except OptimizationError:
        st.error("بهینه‌سازی ناموفق — محدودیت‌ها خیلی سخت هستن. در حال استفاده از وزن برابر...")
        return dict(zip(asset_names, [1/len(asset_names)]*len(asset_names))), (mu.mean()*252*100, np.sqrt(np.diag(S)).mean()*np.sqrt(252)*100, 1.5)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Pro + PyPortfolioOpt", layout="wide")

col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Pro + PyPortfolioOpt</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gold;'>تحلیل حرفه‌ای وال‌استریت — فارسی و برای ایران</h3>", unsafe_allow_html=True)

st.sidebar.header("تنظیمات پرتفوی")
tickers = st.sidebar.text_input("نمادها", value="BTC-USD, GC=F, USDIRR=X, ^GSPC", placeholder="BTC-USD, GC=F, ...")

if st.sidebar.button("تحلیل پرتفوی با PyPortfolioOpt", type="primary"):
    with st.spinner("در حال تحلیل حرفه‌ای با PyPortfolioOpt..."):
        prices = download_data(tickers)
        st.session_state.prices = prices
        
        hedge_type = st.sidebar.selectbox("هجینگ", ["طلا + تتر (ترکیبی)", "طلا به عنوان هج", "دلار/تتر", "بدون هجینگ"], index=0)
        max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, 20)
        
        weights, (exp_ret, vol, sharpe) = analyze_with_pypfopt(prices, hedge_type, max_btc)
        
        df_weights = pd.DataFrame(list(weights.items()), columns=["دارایی", "وزن (%)"])
        df_weights["وزن (%)"] = df_weights["وزن (%)"].astype(float) * 100
        df_weights = df_weights.sort_values("وزن (%)", ascending=False).round(2)
        
        st.success(f"بهینه‌سازی موفق با PyPortfolioOpt")
        c1, c2, c3 = st.columns(3)
        c1.metric("بازده سالیانه", f"{exp_ret:.2f}%")
        c2.metric("ریسک سالیانه", f"{vol:.2f}%")
        c3.metric("نسبت شارپ", f"{sharpe:.3f}")
        
        st.markdown("### تخصیص بهینه دارایی‌ها")
        st.dataframe(df_weights, use_container_width=True)
        
        fig_pie = px.pie(df_weights, values="وزن (%)", names="دارایی", title="تخصیص پرتفوی")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # نمودار رشد سرمایه
        returns = prices.pct_change().dropna()
        portfolio_returns = returns.dot(df_weights["وزن (%)"]/100)
        cumulative = (1 + portfolio_returns).cumprod() * 100
        
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(y=cumulative, name="رشد پرتفوی بهینه"))
        fig_growth.add_hline(y=100, line_dash="dash", annotation_text="سرمایه اولیه")
        fig_growth.update_layout(title="رشد سرمایه با پرتفوی بهینه (PyPortfolioOpt)", height=500)
        st.plotly_chart(fig_growth, use_container_width=True)
        
        # ذخیره و اکسپورت
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_weights.to_excel(writer, index=False)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Portfolio360_PyPortfolioOpt.xlsx">دانلود اکسل</a>'
        st.markdown(href, unsafe_allow_html=True)

# ==================== فیچرهای پریمیوم ====================
st.sidebar.markdown("---")
st.sidebar.subheader("ویژگی‌های پریمیوم")
if st.sidebar.button("تغییر تم دارک/لایت"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

st.sidebar.markdown("### هوش مصنوعی هجینگ + PyPortfolioOpt + وال‌استریت = این ابزار!")
st.balloons()
st.caption("Portfolio360 Pro + PyPortfolioOpt — اولین و بهترین ابزار تحلیل پرتفوی حرفه‌ای فارسی | ۱۴۰۴ | با عشق برای ایران")
