import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ==================== تابع دانلود ۱۰۰٪ بدون خطا ====================
def download_data_safely(ticker_list, period="5y"):
    prices_dict = {}
    failed = []
    
    for ticker in ticker_list:
        ticker = ticker.strip()
        if not ticker:
            continue
        try:
            with st.spinner(f"در حال دانلود {ticker}..."):
                df = yf.Ticker(ticker).history(period=period)
                if df.empty or 'Close' not in df.columns:
                    failed.append(ticker)
                    continue
                # استفاده از Close یا Adj Close (هر کدوم بود)
                price_series = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                price_series = price_series.dropna()
                if len(price_series) > 50:
                    prices_dict[ticker] = price_series
                else:
                    failed.append(ticker)
        except:
            failed.append(ticker)
    
    if not prices_dict:
        st.error("هیچ داده‌ای دانلود نشد!")
        return None, failed
    
    prices = pd.DataFrame(prices_dict)
    prices = prices.ffill().bfill()
    st.success(f"دانلود موفق: {len(prices_dict)} دارایی")
    if failed:
        st.warning(f"این نمادها دانلود نشدند: {', '.join(failed)}")
    return prices, list(prices_dict.keys())

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate - بدون خطا", layout="wide")
st.title("Portfolio360 Ultimate")
st.markdown("### بهترین ابزار تحلیل پرتفوی فارسی — ۱۰۰٪ بدون خطا — با هجینگ هوشمند")

# ==================== سایدبار ====================
st.sidebar.header("منبع داده")
source = st.sidebar.radio("داده از کجا؟", ["دانلود از Yahoo Finance", "آپلود فایل CSV"])

prices = None
asset_names = []

if source == "دانلود از Yahoo Finance":
    default_tickers = "BTC-USD, ETH-USD, GC=F, SI=F, ^GSPC, USDIRR=X"
    user_input = st.sidebar.text_input("نمادها (با کاما)", default_tickers)
    period = st.sidebar.selectbox("بازه زمانی", ["1y","2y","3y","5y","10y","max"], index=3)
    
    if st.sidebar.button("دانلود داده‌ها", type="primary"):
        tickers = [t.strip() for t in user_input.split(",")]
        result = download_data_safely(tickers, period)
        if result:
            prices, asset_names = result

else:
    uploaded_files = st.sidebar.file_uploader("آپلود CSV", type="csv", accept_multiple_files=True)
    if uploaded_files:
        dfs = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            name = file.name.replace('.csv','').replace(' ','_')
            if 'Adj Close' in df.columns:
                col = 'Adj Close'
            elif 'Close' in df.columns:
                col = 'Close'
            else:
                st.error(f"ستون قیمت در {file.name} پیدا نشد")
                continue
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df[[col]].rename(columns={col: name})
            dfs.append(df)
        if dfs:
            prices = pd.concat(dfs, axis=1).ffill().bfill()
            asset_names = list(prices.columns)
            st.success(f"آپلود موفق: {len(asset_names)} دارایی")

# اگر هنوز داده نداریم، صبر کن
if prices is None:
    st.info("لطفاً داده‌ها را دانلود یا آپلود کنید.")
    st.stop()

returns = prices.pct_change().dropna()
if len(returns) < 100:
    st.error("داده کافی نیست. حداقل ۱۰۰ روز نیاز است.")
    st.stop()

# ==================== تنظیمات پرتفو ====================
st.sidebar.header("تنظیمات")
rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 30.0, 18.0)
hedge = st.sidebar.checkbox("هجینگ هوشمند (طلا + تتر + محدودیت بیت‌کوین)", True)
max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 50, 20) if hedge else 100

# محدودیت وزن
bounds = []
for name in asset_names:
    if hedge:
        if any(x in name.upper() for x in ['BTC', 'بیت']):
            bounds.append((0.0, max_btc/100))
        elif any(x in name.upper() for x in ['GC=', 'GOLD', 'طلا']):
            bounds.append((0.15, 0.50))  # حداقل ۱۵٪ طلا
        elif any(x in name.upper() for x in ['USD', 'تتر', 'دلار']):
            bounds.append((0.10, 0.40))
        else:
            bounds.append((0.05, 0.70))
    else:
        bounds.append((0.0, 1.0))

constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
x0 = np.ones(len(asset_names)) / len(asset_names)

# ==================== توابع پرتفو ====================
def port_return(w): return (returns @ w).mean() * 252 * 100
def port_risk(w): return (returns @ w).std() * np.sqrt(252) * 100
def neg_sharpe(w):
    r = port_return(w)
    risk = port_risk(w)
    return -(r - rf_rate) / risk if risk > 0 else 999

# ==================== بهینه‌سازی ====================
with st.spinner("در حال محاسبه پرتفوی بهینه..."):
    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)

# ==================== مونت کارلو ====================
@st.cache_data
def monte_carlo(n=10000):
    results = []
    for _ in range(n):
        w = np.random.random(len(asset_names))
        w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
        w /= w.sum()
        results.append((port_return(w), port_risk(w), (port_return(w)-rf_rate)/port_risk(w)))
    return zip(*results)

mc_ret, mc_risk, mc_sharpe = monte_carlo()

# ==================== نمایش نتایج ====================
st.markdown("---")

if res.success:
    w = res.x
    r = port_return(w)
    risk = port_risk(w)
    sharpe = (r - rf_rate) / risk

    st.success("پرتفوی بهینه با موفقیت محاسبه شد!")

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

    if hedge:
        st.success("هجینگ هوشمند فعال است: طلا + تتر + محدودیت بیت‌کوین")

else:
    st.error("بهینه‌سازی ناموفق بود. محدودیت‌ها را تغییر دهید.")

st.balloons()
st.caption("Portfolio360 Ultimate - نسخه نهایی بدون خطا | ۱۴۰۴")
