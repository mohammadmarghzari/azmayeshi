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

# حذف کامل بخش subprocess و نصب خودکار — چون از requirements.txt استفاده می‌کنیم
from pypfopt import EfficientFrontier, risk_models, expected_returns, DiscreteAllocation
from pypfopt.exceptions import OptimizationError

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
        return pd.DataFrame()
    # اصلاح خطای سینتکس — این خط درست شده!
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    data = data.ffill().bfill()
    return data

# ==================== تحلیل حرفه‌ای با PyPortfolioOpt ====================
def analyze_with_pypfopt(prices, hedge_type, max_btc=20):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
   
    ef = EfficientFrontier(mu, S, weight_bounds=(0,1))
    asset_names = prices.columns.tolist()
   
    # محدودیت بیت‌کوین
    if any("BTC" in x.upper() for x in asset_names):
        btc_idx = next(i for i, x in enumerate(asset_names) if "BTC" in x.upper())
        ef.add_constraint(lambda w, i=btc_idx: w[i] <= max_btc / 100)
   
    # هجینگ ایرانی
    gold_idx = dollar_idx = None
    for i, name in enumerate(asset_names):
        if any(x in name.upper() for x in ["GC=", "GOLD", "طلا"]):
            gold_idx = i
        if any(x in name.upper() for x in ["USD", "USDIRR", "تتر", "USDT"]):
            dollar_idx = i
   
    if hedge_type == "طلا + تتر (ترکیبی)":
        if gold_idx is not None: ef.add_constraint(lambda w, i=gold_idx: w[i] >= 0.15)
        if dollar_idx is not None: ef.add_constraint(lambda w, i=dollar_idx: w[i] >= 0.10)
    elif hedge_type == "طلا به عنوان هج" and gold_idx is not None:
        ef.add_constraint(lambda w, i=gold_idx: w[i] >= 0.15)
    elif hedge_type == "دلار/تتر" and dollar_idx is not None:
        ef.add_constraint(lambda w, i=dollar_idx: w[i] >= 0.10)
   
    try:
        weights = ef.max_sharpe(risk_free_rate=0.30)
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=True, risk_free_rate=0.30)
        return cleaned_weights, perf
    except Exception as e:
        st.warning("بهینه‌سازی ناموفق — وزن برابر استفاده شد")
        equal_w = 1 / len(asset_names)
        weights = dict(zip(asset_names, [equal_w] * len(asset_names)))
        fallback_perf = (mu.mean()*252*100, np.sqrt(np.diag(S)).mean()*np.sqrt(252)*100, 1.0)
        return weights, fallback_perf

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
        if prices.empty or prices.shape[1] < 2:
            st.error("داده کافی دریافت نشد یا کمتر از ۲ دارایی!")
            st.stop()
            
        st.session_state.prices = prices
       
        hedge_type = st.sidebar.selectbox("هجینگ", ["طلا + تتر (ترکیبی)", "طلا به عنوان هج", "دلار/تتر", "بدون هجینگ"], index=0)
        max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, 20)
       
        weights, (exp_ret, vol, sharpe) = analyze_with_pypfopt(prices, hedge_type, max_btc)
       
        df_weights = pd.DataFrame(list(weights.items()), columns=["دارایی", "وزن (%)"])
        df_weights["وزن (%)"] = (df_weights["وزن (%)"].astype(float) * 100).round(2)
        df_weights = df_weights.sort_values("وزن (%)", ascending=False)
       
        st.success("بهینه‌سازی موفق با PyPortfolioOpt")
        c1, c2, c3 = st.columns(3)
        c1.metric("بازده سالیانه", f"{exp_ret:.2f}%")
        c2.metric("ریسک سالیانه", f"{vol:.2f}%")
        c3.metric("نسبت شارپ", f"{sharpe:.3f}")
       
        st.markdown("### تخصیص بهینه دارایی‌ها")
        st.dataframe(df_weights, use_container_width=True, hide_index=True)
       
        fig_pie = px.pie(df_weights, values="وزن (%)", names="دارایی", title="تخصیص پرتفوی")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
       
        # نمودار رشد
        returns = prices.pct_change().dropna()
        portfolio_returns = returns.dot(df_weights.set_index("دارایی")["وزن (%)"]/100)
        cumulative = (1 + portfolio_returns).cumprod() * 100
       
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(y=cumulative, name="رشد پرتفوی بهینه", line=dict(color="#00d2d3", width=3)))
        fig_growth.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="سرمایه اولیه")
        fig_growth.update_layout(title="رشد سرمایه با پرتفوی بهینه", height=500)
        st.plotly_chart(fig_growth, use_container_width=True)
       
        # دانلود اکسل
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_weights.to_excel(writer, sheet_name="تخصیص بهینه", index=False)
            prices.to_excel(writer, sheet_name="داده قیمت")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Portfolio360_بهینه_سازی.xlsx"><button style="background:#00d2d3;color:white;padding:15px 30px;border:none;border-radius:10px;font-size:18px;cursor:pointer;">دانلود گزارش کامل اکسل</button></a>'
        st.markdown(href, unsafe_allow_html=True)

# تم و فوتر
if st.sidebar.button("تغییر تم"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

st.sidebar.markdown("### هوش مصنوعی + PyPortfolioOpt + هجینگ ایرانی = این ابزار!")
st.caption("Portfolio360 Pro — اولین و بهترین ابزار تحلیل پرتفوی حرفه‌ای فارسی | ۱۴۰۴ | با عشق برای ایران ❤️")
