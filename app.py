import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ==================== تابع دانلود بدون خطا ====================
def download_data_safely(tickers, period="5y"):
    prices_dict = {}
    failed = []
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(period=period)
            price_series = df['Close'] if 'Close' in df.columns else None
            if price_series is not None and len(price_series) > 50:
                prices_dict[ticker] = price_series.dropna()
            else:
                failed.append(ticker)
        except:
            failed.append(ticker)
    if not prices_dict:
        st.error("هیچ داده‌ای دانلود نشد!")
        return None
    prices = pd.DataFrame(prices_dict).ffill().bfill()
    if failed:
        st.warning(f"نمادهای ناموفق: {', '.join(failed)}")
    return prices

# ==================== توابع کمکی ====================
def format_recovery_time(days):
    if days == 0 or np.isnan(days):
        return "بدون افت جدی"
    months = days / 21
    years = int(months // 12)
    months_remain = int(months % 12)
    if years > 0 and months_remain > 0:
        return f"{years} سال و {months_remain} ماه"
    elif years > 0:
        return f"{years} سال"
    elif months_remain > 0:
        return f"{months_remain} ماه"
    else:
        return "کمتر از ۱ ماه"

def calculate_recovery_time(returns_series):
    cum = (1 + returns_series).cumprod()
    peak = cum.cummax()
    drawdown = cum / peak - 1
    in_dd = False
    recoveries = []
    start = None
    for i in range(1, len(cum)):
        if drawdown.iloc[i] < 0:
            if not in_dd:
                in_dd = True
                start = i
        else:
            if in_dd:
                in_dd = False
                recoveries.append(i - start)
    return np.mean(recoveries) if recoveries else 0

def portfolio_var(returns_daily, weights, confidence=0.95):
    port_ret = returns_daily.dot(weights)
    return np.percentile(port_ret, (1-confidence)*100) * np.sqrt(252) * 100

def portfolio_cvar(returns_daily, weights, confidence=0.95):
    port_ret = returns_daily.dot(weights)
    var = np.percentile(port_ret, (1-confidence)*100)
    return port_ret[port_ret <= var].mean() * np.sqrt(252) * 100

def max_drawdown(returns_daily, weights):
    port = (1 + returns_daily.dot(weights)).cumprod()
    peak = port.cummax()
    dd = (port / peak - 1).min() * 100
    return dd

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate + Hedging", layout="wide")
st.title("Portfolio360 Ultimate + Hedging")

# ==================== منبع داده ====================
st.sidebar.header("منبع داده")
source = st.sidebar.radio("داده از کجا؟", ["Yahoo Finance", "آپلود CSV"])

prices = None
asset_names = []

if source == "Yahoo Finance":
    default = "BTC-USD,ETH-USD,GC=F,SI=F,^GSPC,USDIRR=X"
    tickers_input = st.sidebar.text_input("نمادها (با کاما)", default)
    period = st.sidebar.selectbox("بازه", ["1y","3y","5y","max"], index=2)
    if st.sidebar.button("دانلود"):
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        prices = download_data_safely(tickers, period)
        if prices is not None:
            asset_names = list(prices.columns)

else:
    uploaded = st.sidebar.file_uploader("آپلود CSV", type="csv", accept_multiple_files=True)
    if uploaded:
        dfs = []
        for file in uploaded:
            df = pd.read_csv(file)
            name = file.name.replace('.csv', '')
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            dfs.append(df[[col]].rename(columns={col: name}))
        prices = pd.concat(dfs, axis=1).ffill().bfill()
        asset_names = list(prices.columns)

if prices is None:
    st.stop()

returns = prices.pct_change().dropna()
if len(returns) < 100:
    st.error("داده کافی نیست.")
    st.stop()

# ==================== تنظیمات ====================
st.sidebar.header("تنظیمات")
rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 30.0, 18.0)
hedge_active = st.sidebar.checkbox("فعال کردن هجینگ", True)
if hedge_active:
    hedge_types = ["طلا به عنوان هج", "دلار/تتر", "پوزیشن شورت بیت‌کوین", "آپشن Put", "پرتفوی معکوس", "استراتژی Collar", "Tail-Risk Hedge"]
    hedge_type = st.sidebar.selectbox("نوع هجینگ", hedge_types)
    st.sidebar.info(f"پیشنهاد: {hedge_type} برای بیمه پرتفوی در ریزش‌ها.")

# ==================== بهینه‌سازی ====================
n_assets = len(asset_names)
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

bounds = [(0.0, 1.0) for _ in range(n_assets)]
if hedge_active:
    for i, name in enumerate(asset_names):
        if 'BTC' in name.upper():
            bounds[i] = (0.0, st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 50, 20)/100)
        if 'GOLD' in name.upper() or hedge_type == "طلا به عنوان هج":
            bounds[i] = (0.15, 0.50)
        if 'USD' in name.upper() or hedge_type == "دلار/تتر":
            bounds[i] = (0.10, 0.40)
        # برای هجینگ‌های دیگر، محدودیت‌های خاص اضافه کن (مثل شورت: وزن منفی، اما ساده نگه داریم)

constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
x0 = np.ones(n_assets) / n_assets

res = minimize(lambda w: - (port_return(w) - rf_rate) / port_risk(w), x0, method='SLSQP', bounds=bounds, constraints=constraints)

# ==================== نمایش ====================
if res.success:
    w = res.x
    ann_ret, ann_vol, sharpe, var95, cvar95, max_dd, rec_time = portfolio_metrics(returns, w)

    st.success("پرتفوی بهینه آماده است!")

    # نتایج

st.markdown("### توضیح سبک استفاده‌شده")
st.write("سبک استفاده شده: بهینه‌سازی میانگین-واریانس با هجینگ (اگر فعال باشد).")
st.write("دلایل: تعادل ریسک و بازده، بیمه پرتفوی در ریزش‌ها، بر اساس داده‌های تاریخی.")
st.write("سبک‌های دیگر: وزن برابر، حداقل واریانس، بلک-لیترمن، ریسک‌پاریتی، مونت‌کارلو خام.")
st.selectbox("اگر می‌خواهید سبک دیگری استفاده شود", ["این سبک خوبه", "وزن برابر", "حداقل واریانس", "بلک-لیترمن"])

st.balloons()
