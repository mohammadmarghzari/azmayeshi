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

# ==================== نصب خودکار PyPortfolioOpt ====================
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, DiscreteAllocation
    from pypfopt.exceptions import OptimizationError
except ImportError:
    st.error("در حال نصب PyPortfolioOpt... چند ثانیه صبر کنید")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPortfolioOpt"])
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
        st.error("حداقل یک نماد وارد کنید!")
        return pd.DataFrame()
    
    with st.spinner("در حال دریافت داده‌ها از Yahoo Finance..."):
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
        data = data.ffill().bfill()
    
    if data.empty or data.shape[1] == 0:
        st.error("داده‌ای برای نماد(های) وارد شده پیدا نشد. نمادها را چک کنید.")
        return pd.DataFrame()
    
    return data

# ==================== تحلیل حرفه‌ای با PyPortfolioOpt ====================
def analyze_with_pypfopt(prices, hedge_type, max_btc=20):
    if prices.shape[1] < 2:
        st.error("برای بهینه‌سازی حداقل ۲ دارایی نیاز است!")
        return {}, (0, 0, 0)

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    asset_names = prices.columns.tolist()

    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

    # محدودیت حداکثر بیت‌کوین
    btc_idx = None
    for i, name in enumerate(asset_names):
        if "BTC" in name.upper():
            btc_idx = i
            break
    if btc_idx is not None:
        ef.add_constraint(lambda w, i=btc_idx: w[i] <= max_btc / 100)

    # هجینگ ایرانی
    gold_idx = dollar_idx = None
    for i, name in enumerate(asset_names):
        if any(x in name.upper() for x in ["GC=", "GOLD", "طلا"]):
            gold_idx = i
        if any(x in name.upper() for x in ["USD", "USDIRR", "TETHER", "USDT", "تتر"]):
            dollar_idx = i

    if hedge_type == "طلا + تتر (ترکیبی)":
        if gold_idx is not None:
            ef.add_constraint(lambda w, i=gold_idx: w[i] >= 0.15)
        if dollar_idx is not None:
            ef.add_constraint(lambda w, i=dollar_idx: w[i] >= 0.10)
    elif hedge_type == "طلا به عنوان هج":
        if gold_idx is not None:
            ef.add_constraint(lambda w, i=gold_idx: w[i] >= 0.15)
    elif hedge_type == "دلار/تتر":
        if dollar_idx is not None:
            ef.add_constraint(lambda w, i=dollar_idx: w[i] >= 0.10)

    try:
        weights = ef.max_sharpe(risk_free_rate=0.02)  # نرخ بدون ریسک ایران ≈ ۲٪
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=0.02)
        return cleaned_weights, perf
    except Exception as e:
        st.warning("بهینه‌سازی شارپ ناموفق — استفاده از وزن برابر")
        equal_weight = 1 / len(asset_names)
        weights = {asset: equal_weight for asset in asset_names}
        returns = prices.pct_change().mean() * 252
        volatility = prices.pct_change().std() * np.sqrt(252)
        sharpe = (returns.mean() - 0.02) / volatility.mean() if volatility.mean() > 0 else 0
        return weights, (returns.mean()*100, volatility.mean()*100, sharpe)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Pro + PyPortfolioOpt", layout="wide")

# هدر زیبا
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gold;'>تحلیل حرفه‌ای وال‌استریت — مخصوص سرمایه‌گذار ایرانی</h3>", unsafe_allow_html=True)

st.sidebar.header("تنظیمات پرتفوی")

tickers = st.sidebar.text_input(
    "نمادهای یاهو فایننس (با کاما جدا کنید)",
    value="BTC-USD, GC=F, USDIRR=X, ^GSPC, AAPL",
    help="مثال: BTC-USD, GC=F (طلا), USDIRR=X (دلار به ریال)"
)

hedge_type = st.sidebar.selectbox(
    "استراتژی هجینگ ایرانی",
    ["طلا + تتر (ترکیبی)", "طلا به عنوان هج", "دلار/تتر", "بدون هجینگ"],
    index=0
)

max_btc = st.sidebar.slider("حداکثر تخصیص به بیت‌کوین (%)", 0, 100, 25, 5)

if st.sidebar.button("تحلیل پرتفوی با PyPortfolioOpt", type="primary"):
    prices = download_data(tickers)
    
    if prices.empty:
        st.stop()
    
    st.session_state.prices = prices

    with st.spinner("در حال بهینه‌سازی حرفه‌ای با PyPortfolioOpt..."):
        weights, (exp_ret, vol, sharpe) = analyze_with_pypfopt(prices, hedge_type, max_btc)

    # نمایش نتایج
    st.success("بهینه‌سازی با موفقیت انجام شد!")

    c1, c2, c3 = st.columns(3)
    c1.metric("بازده مورد انتظار سالیانه", f"{exp_ret:.2f}%")
    c2.metric("ریسک سالیانه (انحراف معیار)", f"{vol:.2f}%")
    c3.metric("نسبت شارپ", f"{sharpe:.3f}")

    # جدول وزن‌ها
    df_weights = pd.DataFrame([
        {"دارایی": k, "وزن (%)": round(v * 100, 2)} for k, v in weights.items()
    ]).sort_values("وزن (%)", ascending=False)

    st.markdown("### تخصیص بهینه دارایی‌ها")
    st.dataframe(df_weights, use_container_width=True, hide_index=True)

    # نمودار دایره‌ای
    fig_pie = px.pie(
        df_weights, values="وزن (%)", names="دارایی",
        title="تخصیص بهینه پرتفوی",
        color_discrete_sequence=px.colors.sequential.Turbo
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

    # نمودار رشد سرمایه
    daily_returns = prices.pct_change().dropna()
    portfolio_daily = daily_returns.dot(df_weights.set_index("دارایی")["وزن (%)"]/100)
    cumulative = (1 + portfolio_daily).cumprod() * 100

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        y=cumulative, name="رشد پرتفوی بهینه", line=dict(width=3, color="#00d2d3")
    ))
    fig_growth.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="سرمایه اولیه")
    fig_growth.update_layout(
        title="رشد سرمایه ۱۰۰ میلیون تومان با پرتفوی بهینه",
        yaxis_title="ارزش پرتفوی (درصد از اولیه)",
        height=550,
        template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white"
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    # دانلود اکسل
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_weights.to_excel(writer, index=False, sheet_name='تخصیص بهینه')
        prices.to_excel(writer, sheet_name='داده قیمت')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'''
    <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Portfolio360_بهینه_ساز_پرتفوی.xlsx">
    <button style="background:#00d2d3;color:white;padding:12px 24px;border:none;border-radius:8px;font-size:16px;cursor:pointer;">
    دانلود گزارش کامل اکسل
    </button></a>
    '''
    st.markdown(href, unsafe_allow_html=True)

# ==================== سایدبار اضافی ====================
st.sidebar.markdown("---")
st.sidebar.subheader("تنظیمات ظاهری")
if st.sidebar.button("تغییر تم 🌙"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### هوش مصنوعی + وال‌استریت + هجینگ ایرانی = این ابزار!")

# فوتر
st.markdown("---")
st.caption("Portfolio360 Pro — اولین و بهترین ابزار تحلیل پرتفوی حرفه‌ای فارسی | ۱۴۰۴ | با عشق برای ایران")

# بالن‌ها برای خوشحالی کاربر
if st.sidebar.button("جشن بگیریم؟"):
    st.balloons()
