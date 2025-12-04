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
@st.cache_data(show_spinner=False)
def download_data_safely(tickers_input, period):
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
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

if 'prices' not in st.session_state:
    st.session_state.prices = None
    st.session_state.asset_names = []

if source == "Yahoo Finance":
    default = "BTC-USD,ETH-USD,GC=F,SI=F,^GSPC,USDIRR=X"
    tickers_input = st.sidebar.text_input("نمادها (با کاما)", value=default, key="tickers")
    period = st.sidebar.selectbox("بازه", ["1y","3y","5y","max"], index=2, key="period")
    if st.sidebar.button("دانلود"):
        prices = download_data_safely(tickers_input, period)
        if prices is not None:
            st.session_state.prices = prices
            st.session_state.asset_names = list(prices.columns)

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
        st.session_state.prices = prices
        st.session_state.asset_names = list(prices.columns)

if st.session_state.prices is None:
    st.stop()

prices = st.session_state.prices
asset_names = st.session_state.asset_names

returns = prices.pct_change().dropna()
if len(returns) < 100:
    st.error("داده کافی نیست.")
    st.stop()

# ==================== تنظیمات ====================
st.sidebar.header("تنظیمات")
if 'rf_rate' not in st.session_state:
    st.session_state.rf_rate = 18.0
rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 30.0, value=st.session_state.rf_rate)
st.session_state.rf_rate = rf_rate

if 'hedge_active' not in st.session_state:
    st.session_state.hedge_active = True
hedge_active = st.sidebar.checkbox("فعال کردن هجینگ هوشمند", value=st.session_state.hedge_active)
st.session_state.hedge_active = hedge_active

if hedge_active:
    hedge_types = ["طلا به عنوان هج", "دلار/تتر", "پوزیشن شورت بیت‌کوین", "آپشن Put", "پرتفوی معکوس", "استراتژی Collar", "Tail-Risk Hedge"]
    if 'hedge_type' not in st.session_state:
        st.session_state.hedge_type = hedge_types[0]
    hedge_type = st.sidebar.selectbox("نوع هجینگ", hedge_types, index=hedge_types.index(st.session_state.hedge_type))
    st.session_state.hedge_type = hedge_type
    st.sidebar.info(f"پیشنهاد: {hedge_type} برای بیمه پرتفوی در ریزش‌ها.")

if 'max_btc' not in st.session_state:
    st.session_state.max_btc = 20
max_btc = st.sidebar.slider("حداکثر وزن بیت کوین (%)", 0, 100, value=st.session_state.max_btc)
st.session_state.max_btc = max_btc

# ==================== بهینه‌سازی ====================
n_assets = len(asset_names)
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

bounds = [(0.0, 1.0) for _ in range(n_assets)]
if hedge_active:
    for i, name in enumerate(asset_names):
        if 'BTC' in name.upper():
            bounds[i] = (0.0, max_btc/100)
        if 'GOLD' in name.upper() or hedge_type == "طلا به عنوان هج":
            bounds[i] = (0.15, 0.50)
        if 'USD' in name.upper() or hedge_type == "دلار/تتر":
            bounds[i] = (0.10, 0.40)
        # اضافه کردن محدودیت‌های دیگر هجینگ اگر لازم باشه

constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
x0 = np.ones(n_assets) / n_assets

def port_return(w): return np.dot(w, mean_returns) * 100
def port_risk(w): return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * 100

if st.sidebar.button("محاسبه پرتفوی"):
    with st.spinner("در حال محاسبه..."):
        res = minimize(lambda w: - (port_return(w) - rf_rate) / port_risk(w), x0, method='SLSQP', bounds=bounds, constraints=constraints)

    # ==================== نمایش ====================
    st.markdown("---")

    if res.success:
        w = res.x
        r = port_return(w)
        risk = port_risk(w)
        sharpe = (r - rf_rate) / risk

        st.success("پرتفوی بهینه آماده است!")

        # نتایج
        c1, c2, c3 = st.columns(3)
        c1.metric("بازده سالیانه", f"{r:.2f}%")
        c2.metric("ریسک سالیانه", f"{risk:.2f}%")
        c3.metric("نسبت شارپ", f"{sharpe:.3f}")

        # جدول وزن
        df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(w*100, 2)})
        df_w = df_w.sort_values("وزن (%)", ascending=False)
        st.markdown("### تخصیص بهینه دارایی‌ها")
        st.dataframe(df_w, use_container_width=True)

        # نمودار دایره‌ای
        fig_pie = px.pie(df_w, values="وزن (%)", names="دارایی", title="توزیع دارایی‌ها")
        st.plotly_chart(fig_pie, use_container_width=True)

        # مرز کارا
        st.subheader("مرز کارا پرتفوی")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mc_risk, y=mc_ret, mode='markers',
                                 marker=dict(color=mc_sharpe, colorscale='Viridis', size=5),
                                 name="پرتفوهای تصادفی"))
        fig.add_trace(go.Scatter(x=[risk], y=[r], mode='markers',
                                 marker=dict(color='red', size=15, symbol='star'),
                                 name="پرتفوی بهینه شما"))
        fig.update_layout(xaxis_title="ریسک (%)", yaxis_title="بازده (%)", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # ریکاوری تایم
        st.markdown("### زمان ریکاوری پرتفوی")
        port_returns = returns.dot(w)
        rec_days = calculate_recovery_time(port_returns)
        st.write(format_recovery_time(rec_days))

        if hedge_active:
            st.success("هجینگ فعال است: " + hedge_type)

    else:
        st.error("بهینه‌سازی ناموفق بود. محدودیت‌ها را تغییر دهید.")

    st.balloons()

# ==================== توضیح سبک ====================
st.markdown("### توضیح سبک استفاده‌شده")
st.write("سبک استفاده شده: بهینه‌سازی میانگین-واریانس با هجینگ (اگر فعال باشد).")
st.write("دلایل: تعادل ریسک و بازده، بیمه پرتفوی در ریزش‌ها، بر اساس داده‌های تاریخی.")
st.write("سبک‌های دیگر: وزن برابر، حداقل واریانس, بلک-لیترمن, ریسک‌پاریتی, مونت‌کارلو خام.")
style_choice = st.selectbox("اگر می‌خواهید سبک دیگری استفاده شود", ["این سبک خوبه", "وزن برابر", "حداقل واریانس", "بلک-لیترمن"])
if style_choice != "این سبک خوبه":
    st.info(f"سبک {style_choice} انتخاب شد. برای اعمال، کد را بروزرسانی کنید.")

# ==================== مونت کارلو ====================
@st.cache_data
def monte_carlo(n=10000):
    results = []
    for _ in range(n):
        w = np.random.random(n_assets)
        w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
        w /= w.sum()
        results.append((port_return(w), port_risk(w), (port_return(w)-rf_rate)/port_risk(w)))
    return zip(*results)

mc_ret, mc_risk, mc_sharpe = monte_carlo()
