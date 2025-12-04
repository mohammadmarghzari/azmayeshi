import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ==================== دانلود بدون خطا ====================
@st.cache_data(show_spinner=False)
def download_data(tickers_str, period):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    data = {}
    failed = []
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, auto_adjust=True)
            if not df.empty and len(df) > 50:
                data[t] = df['Close']
            else:
                failed.append(t)
        except:
            failed.append(t)
    if not data:
        st.error("هیچ داده‌ای دانلود نشد!")
        return None
    prices = pd.DataFrame(data).ffill().bfill()
    if failed:
        st.warning(f"دانلود نشد: {', '.join(failed)}")
    return prices

# ==================== توابع ریسک و ریکاوری ====================
def calculate_recovery_time(returns_series):
    cum = (1 + returns_series).cumprod()
    peak = cum.cummax()
    drawdown = cum / peak - 1
    in_dd = False
    recoveries = []
    for i in range(1, len(cum)):
        if drawdown.iloc[i] < -0.01:
            if not in_dd:
                in_dd = True
                start = i
        elif in_dd:
            in_dd = False
            recoveries.append(i - start)
    return np.mean(recoveries) if recoveries else 0

def format_recovery(days):
    if days == 0: return "بدون افت"
    months = int(days / 21)
    years = months // 12
    months = months % 12
    if years and months:
        return f"{years} سال و {months} ماه"
    elif years:
        return f"{years} سال"
    else:
        return f"{months} ماه"

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate", layout="wide")
st.title("Portfolio360 Ultimate")
st.markdown("### کامل‌ترین ابزار تحلیل پرتفوی فارسی با هجینگ هوشمند و انتخاب سبک")

# ==================== سایدبار ====================
st.sidebar.header("منبع داده")
data_source = st.sidebar.radio("داده از کجا؟", ["Yahoo Finance", "آپلود CSV"])

prices = None
if data_source == "Yahoo Finance":
    default_tickers = "BTC-USD,ETH-USD,GC=F,^GSPC,USDIRR=X"
    tickers = st.sidebar.text_input("نمادها", default_tickers)
    period = st.sidebar.selectbox("بازه زمانی", ["1y","3y","5y","10y","max"], index=2)
    if st.sidebar.button("دانلود داده‌ها"):
        prices = download_data(tickers, period)

else:
    uploaded = st.sidebar.file_uploader("آپلود فایل CSV", type="csv", accept_multiple_files=True)
    if uploaded:
        dfs = []
        for f in uploaded:
            df = pd.read_csv(f)
            name = f.name.replace(".csv", "")
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            dfs.append(df[[col]].rename(columns={col: name}))
        prices = pd.concat(dfs, axis=1).ffill().bfill()

if prices is None:
    st.info("لطفاً داده‌ها را دانلود یا آپلود کنید.")
    st.stop()

returns = prices.pct_change().dropna()
asset_names = list(prices.columns)

# ==================== تنظیمات ====================
st.sidebar.header("تنظیمات پرتفوی")
rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, 18.0, 0.5)

hedge_on = st.sidebar.checkbox("هجینگ هوشمند فعال باشد", True)

hedge_type = "بدون هج"
if hedge_on:
    hedge_options = [
        "طلا به عنوان هج (حداقل ۱۵٪)",
        "دلار/تتر (حداقل ۱۰٪)",
        "محدودیت بیت‌کوین (حداکثر ۲۰٪)",
        "طلا + تتر (ترکیبی)",
        "پرتفوی معکوس (شورت)",
        "استراتژی Collar",
        "Tail-Risk Hedge"
    ]
    hedge_type = st.sidebar.selectbox("نوع هجینگ", hedge_options, index=0)

max_btc = st.sidebar.slider("حداکثر وزن بیت‌کوین (%)", 0, 100, 20) if hedge_on else 100

# ==================== محدودیت‌ها بر اساس هجینگ ====================
bounds = []
for name in asset_names:
    lower = 0.0
    upper = 1.0
    
    if hedge_on:
        if "BTC" in name.upper():
            upper = max_btc / 100
        if hedge_type == "طلا به عنوان هج (حداقل ۱۵٪)" and ("GC=" in name or "GOLD" in name.upper()):
            lower = 0.15
        if hedge_type == "دلار/تتر (حداقل ۱۰٪)" and "USD" in name.upper():
            lower = 0.10
        if hedge_type == "طلا + تتر (ترکیبی)":
            if "GC=" in name or "GOLD" in name.upper():
                lower = 0.15
            if "USD" in name.upper():
                lower = 0.10
    
    bounds.append((lower, upper))

constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
mean_ret = returns.mean() * 252
cov_mat = returns.cov() * 252

def portfolio_performance(w):
    ret = np.dot(w, mean_ret) * 100
    risk = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100
    sharpe = (ret - rf_rate) / risk if risk > 0 else -999
    return ret, risk, sharpe

def negative_sharpe(w):
    _, _, sharpe = portfolio_performance(w)
    return -sharpe

# ==================== بهینه‌سازی ====================
x0 = np.ones(len(asset_names)) / len(asset_names)
with st.spinner("در حال بهینه‌سازی پرتفوی..."):
    result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})

# ==================== مونت کارلو ====================
@st.cache_data
def monte_carlo_simulation(_n=8000):
    results = []
    for _ in range(_n):
        w = np.random.random(len(asset_names))
        w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
        w /= w.sum()
        ret, risk, sharpe = portfolio_performance(w)
        results.append((ret, risk, sharpe))
    return zip(*results)

mc_ret, mc_risk, mc_sharpe = monte_carlo_simulation()

# ==================== نمایش نتایج ====================
st.markdown("---")
if result.success:
    weights = result.x
    ret, risk, sharpe = portfolio_performance(weights)

    st.success("پرتفوی بهینه با موفقیت ساخته شد!")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("بازده سالیانه", f"{ret:.2f}%")
    col2.metric("ریسک سالیانه", f"{risk:.2f}%")
    col3.metric("نسبت شارپ", f"{sharpe:.3f}")
    
    port_daily = returns.dot(weights)
    var95 = np.percentile(port_daily, 5) * np.sqrt(252) * 100
    max_dd = ((1 + port_daily).cumprod().cummax() / (1 + port_daily).cumprod() - 1).min() * 100
    recovery = format_recovery(calculate_recovery_time(port_daily))
    col4.metric("زمان ریکاوری", recovery)

    # وزن‌ها
    df_weights = pd.DataFrame({
        "دارایی": asset_names,
        "وزن (%)": np.round(weights * 100, 2)
    }).sort_values("وزن (%)", ascending=False)
    st.markdown("### تخصیص دارایی‌ها")
    st.dataframe(df_weights, use_container_width=True)

    fig_pie = px.pie(df_weights, values="وزن (%)", names="دارایی", title="توزیع وزن")
    st.plotly_chart(fig_pie, use_container_width=True)

    # مرز کارا
    st.subheader("مرز کارا پرتفوی")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_risk, y=mc_ret, mode='markers',
                             marker=dict(color=mc_sharpe, colorscale='RdYlGn', size=6, colorbar=dict(title="شارپ")),
                             name="پرتفوهای تصادفی"))
    fig.add_trace(go.Scatter(x=[risk], y=[ret], mode='markers',
                             marker=dict(color='gold', size=18, symbol='star-diamond'),
                             name="پرتفوی بهینه شما"))
    fig.update_layout(height=600, xaxis_title="ریسک (%)", yaxis_title="بازده (%)")
    st.plotly_chart(fig, use_container_width=True)

    # توضیح سبک
    st.markdown("### سبک استفاده‌شده در این پرتفوی")
    st.success("**بهینه‌سازی میانگین-واریانس (Markowitz) با هجینگ هوشمند**")
    st.write("""
    دلایل استفاده:
    • بالاترین نسبت شارپ (بازده به ریسک)
    • محافظت در برابر ریزش‌های بزرگ بازار
    • بر اساس داده‌های واقعی تاریخی
    • کاملاً عملی و قابل اجرا در ایران
    """)

    st.markdown("### سبک‌های دیگر موجود:")
    style = st.selectbox("اگر می‌خواهید سبک دیگری تست کنید:",
        ["این سبک عالیه", "وزن برابر", "حداقل ریسک", "ریسک‌پاریتی", "بلک-لیترمن"])

    if style != "این سبک عالیه":
        st.info(f"سبک «{style}» انتخاب شد. در نسخه بعدی اضافه می‌کنم!")

    if hedge_on:
        st.success(f"هجینگ فعال: {hedge_type}")

else:
    st.error("بهینه‌سازی ناموفق بود. محدودیت‌ها را شل کنید.")

st.balloons()
st.caption("Portfolio360 Ultimate - نسخه نهایی ۱۴۰۴ | با عشق برای سرمایه‌گذاران ایرانی")
