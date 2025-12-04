import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import warnings
import io
import base64
from datetime import datetime

# این خط‌ها مستقیم ایمپورت میشن — چون در requirements.txt هستن
from pypfopt import EfficientFrontier, risk_models, expected_returns
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

    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    data = data.ffill().bfill()
    
    if data.empty or data.shape[1] == 0:
        st.error("داده‌ای دریافت نشد. نمادها را بررسی کنید (مثل BTC-USD, GC=F)")
        return pd.DataFrame()
    
    return data

# ==================== تحلیل با PyPortfolioOpt ====================
def analyze_portfolio(prices, hedge_type, max_btc_pct):
    if len(prices.columns) < 2:
        st.error("حداقل ۲ دارایی برای بهینه‌سازی نیاز است!")
        return {}, (0, 0, 0)

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

    # محدودیت بیت‌کوین
    btc_col = next((col for col in prices.columns if "BTC" in col.upper()), None)
    if btc_col is not None:
        idx = prices.columns.get_loc(btc_col)
        ef.add_constraint(lambda w, i=idx: w[i] <= max_btc_pct / 100)

    # هجینگ ایرانی
    gold_col = next((col for col in prices.columns if any(x in col.upper() for x in ["GC=", "GOLD", "طلا"])), None)
    dollar_col = next((col for col in prices.columns if any(x in col.upper() for x in ["USD", "USDIRR", "تتر", "USDT"])), None)

    if hedge_type == "طلا + تتر (ترکیبی)":
        if gold_col: ef.add_constraint(lambda w: w[prices.columns.get_loc(gold_col)] >= 0.15)
        if dollar_col: ef.add_constraint(lambda w: w[prices.columns.get_loc(dollar_col)] >= 0.10)
    elif hedge_type == "طلا به عنوان هج" and gold_col:
        ef.add_constraint(lambda w: w[prices.columns.get_loc(gold_col)] >= 0.15)
    elif hedge_type == "دلار/تتر" and dollar_col:
        ef.add_constraint(lambda w: w[prices.columns.get_loc(dollar_col)] >= 0.10)

    try:
        weights = ef.max_sharpe(risk_free_rate=0.30)  # نرخ بدون ریسک ایران ≈۳۰٪
        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=0.30)
        return cleaned, perf
    except:
        st.warning("بهینه‌سازی ناموفق — وزن برابر استفاده شد")
        w = 1 / len(prices.columns)
        weights = {col: w for col in prices.columns}
        ret = prices.pct_change().mean().mean() * 252
        vol = prices.pct_change().std().mean() * np.sqrt(252)
        sharpe = (ret - 0.30) / vol if vol > 0 else 0
        return weights, (ret*100, vol*100, sharpe)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Pro – ایران", layout="wide")

# هدر
c1, c2, c3 = st.columns([1,3,1])
with c2:
    st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gold;'>بهینه‌ساز پرتفوی حرفه‌ای وال‌استریت — مخصوص ایران</h3>", unsafe_allow_html=True)

st.sidebar.header("تنظیمات پرتفوی")

tickers = st.sidebar.text_input(
    "نمادها (با کاما جدا کنید)",
    value="BTC-USD, GC=F, USDIRR=X, ^GSPC",
    help="مثال: BTC-USD, GC=F (طلا), USDIRR=X (دلار به ریال)"
)

hedge_type = st.sidebar.selectbox(
    "استراتژی هجینگ",
    ["طلا + تتر (ترکیبی)", "طلا به عنوان هج", "دلار/تتر", "بدون هجینگ"]
)

max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, 20, 5)

if st.sidebar.button("🚀 تحلیل پرتفوی با PyPortfolioOpt", type="primary"):
    prices = download_data(tickers)
    if prices.empty:
        st.stop()

    with st.spinner("در حال بهینه‌سازی حرفه‌ای..."):
        weights, (ret, vol, sharpe) = analyze_portfolio(prices, hedge_type, max_btc)

    st.success("بهینه‌سازی با موفقیت انجام شد!")

    # متریک‌ها
    c1, c2, c3 = st.columns(3)
    c1.metric("بازده سالیانه", f"{ret:.2f}%")
    c2.metric("ریسک سالیانه", f"{vol:.2f}%")
    c3.metric("نسبت شارپ", f"{sharpe:.3f}")

    # جدول وزن‌ها
    df_w = pd.DataFrame([
        {"دارایی": k, "وزن (%)": round(v*100, 2)} for k, v in weights.items()
    ]).sort_values("وزن (%)", ascending=False)

    st.markdown("### تخصیص بهینه دارایی‌ها")
    st.dataframe(df_w, use_container_width=True, hide_index=True)

    # نمودار دایره‌ای
    fig_pie = px.pie(df_w, values="وزن (%)", names="دارایی", title="تخصیص پرتفوی")
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

    # رشد سرمایه
    returns = prices.pct_change().dropna()
    port_ret = returns.dot(df_w.set_index("دارایی")["وزن (%)"]/100)
    cumulative = (1 + port_ret).cumprod() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cumulative, name="رشد پرتفوی", line=dict(color="#00d2d3", width=4)))
    fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="سرمایه اولیه")
    fig.update_layout(title="رشد سرمایه با پرتفوی بهینه", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # دانلود اکسل
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_w.to_excel(writer, sheet_name="وزن‌ها", index=False)
        prices.to_excel(writer, sheet_name="داده قیمت")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="Portfolio360_بهینه_سازی.xlsx"><button style="background:#00d2d3;color:white;padding:12px 24px;border:none;border-radius:8px;cursor:pointer;font-size:16px;">دانلود گزارش اکسل</button></a>'
    st.markdown(href, unsafe_allow_html=True)

# تم و فوتر
if st.sidebar.button("تغییر تم"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

st.caption("Portfolio360 Pro — اولین ابزار بهینه‌سازی پرتفوی حرفه‌ای فارسی | ۱۴۰۴ | با عشق برای ایران")
