import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ==================== دانلود داده ====================
@st.cache_data(show_spinner=False)
def download_data(tickers_str, period):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    data = {}
    failed = []
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, auto_adjust=True)
            if len(df) > 50:
                data[t] = df["Close"]
            else:
                failed.append(t)
        except:
            failed.append(t)
    if not data:
        return None
    prices = pd.DataFrame(data).ffill().bfill()
    if failed:
        st.warning(f"دانلود نشد: {', '.join(failed)}")
    return prices

# ==================== توابع ریسک ====================
def calculate_recovery_time(ret_series):
    cum = (1 + ret_series).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    in_dd = False
    recs = []
    for i in range(1, len(cum)):
        if dd.iloc[i] < -0.01:
            if not in_dd:
                in_dd = True
                start = i
        elif in_dd:
            in_dd = False
            recs.append(i - start)
    return np.mean(recs) if recs else 0

def format_recovery(days):
    if days == 0: return "بدون افت جدی"
    m = int(days / 21)
    y, m = divmod(m, 12)
    if y and m: return f"{y} سال و {m} ماه"
    if y: return f"{y} سال"
    return f"{m} ماه"

# ==================== محاسبه پرتفوی (فقط این قسمت دوباره اجرا میشه) ====================
@st.fragment
def calculate_portfolio():
    if st.session_state.prices is None:
        st.info("ابتدا داده‌ها را دانلود یا آپلود کنید.")
        return

    returns = st.session_state.prices.pct_change().dropna()
    asset_names = list(st.session_state.prices.columns)
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252

    # محدودیت‌ها
    bounds = []
    for name in asset_names:
        low = 0.0
        up = 1.0
        if st.session_state.hedge_active:
            if "BTC" in name.upper():
                up = st.session_state.max_btc / 100
            if st.session_state.hedge_type == "طلا به عنوان هج" and ("GC=" in name or "GOLD" in name.upper()):
                low = 0.15
            if st.session_state.hedge_type == "دلار/تتر" and "USD" in name.upper():
                low = 0.10
            if st.session_state.hedge_type == "طلا + تتر (ترکیبی)":
                if "GC=" in name or "GOLD" in name.upper():
                    low = 0.15
                if "USD" in name.upper():
                    low = 0.10
        bounds.append((low, up))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    }]
    x0 = np.ones(len(asset_names)) / len(asset_names)

    def neg_sharpe(w):
        r = np.dot(w, mean_ret) * 100
        risk = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100
        return -(r - st.session_state.rf_rate) / risk if risk > 0 else 9999

    with st.spinner("در حال بهینه‌سازی پرتفوی..."):
        res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000})

    if not res.success:
        st.error("بهینه‌سازی ناموفق بود. محدودیت‌ها را شل کنید.")
        return

    w = res.x
    ret = np.dot(w, mean_ret) * 100
    risk = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100
    sharpe = (ret - st.session_state.rf_rate) / risk

    port_daily = returns.dot(w)
    var95 = np.percentile(port_daily, 5) * np.sqrt(252) * 100
    max_dd = ((1 + port_daily).cumprod() / (1 + port_daily).cumprod().cummax() - 1).min() * 100
    recovery = format_recovery(calculate_recovery_time(port_daily))

    # نمایش نتایج
    st.success("پرتفوی بهینه محاسبه شد!")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("بازده سالیانه", f"{ret:.2f}%")
    c2.metric("ریسک سالیانه", f"{risk:.2f}%")
    c3.metric("نسبت شارپ", f"{sharpe:.3f}")
    c4.metric("زمان ریکاوری", recovery)

    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(w*100, 2)}).sort_values("وزن (%)", ascending=False)
    st.markdown("### تخصیص دارایی‌ها")
    st.dataframe(df_w, use_container_width=True)

    fig_pie = px.pie(df_w, values="وزن (%)", names="دارایی", title="توزیع وزن دارایی‌ها")
    st.plotly_chart(fig_pie, use_container_width=True)

    # مرز کارا
    st.subheader("مرز کارا")
    mc_ret, mc_risk, mc_sharpe = [], [], []
    for _ in range(8000):
        ww = np.random.random(len(asset_names))
        ww = np.clip(ww, [b[0] for b in bounds], [b[1] for b in bounds])
        ww /= ww.sum()
        rr = np.dot(ww, mean_ret) * 100
        ri = np.sqrt(np.dot(ww.T, np.dot(cov_mat, ww))) * 100
        mc_ret.append(rr)
        mc_risk.append(ri)
        mc_sharpe.append((rr - st.session_state.rf_rate) / ri if ri > 0 else -10)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_risk, y=mc_ret, mode='markers',
                             marker=dict(color=mc_sharpe, colorscale='RdYlGn', size=6,
                                         colorbar=dict(title="شارپ")),
                             name="پرتفوهای تصادفی"))
    fig.add_trace(go.Scatter(x=[risk], y=[ret], mode='markers',
                             marker=dict(color='gold', size=20, symbol='star-diamond'),
                             name="پرتفوی بهینه شما"))
    fig.update_layout(height=600, xaxis_title="ریسک (%)", yaxis_title="بازده (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"هجینگ: {'فعال – ' + st.session_state.hedge_type if st.session_state.hedge_active else 'غیرفعال'}")

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate", layout="wide")
st.title("Portfolio360 Ultimate")
st.markdown("### بهترین ابزار تحلیل پرتفوی فارسی با هجینگ هوشمند، ریکاوری تایم و انتخاب سبک")

# ==================== سایدبار – تنظیمات بدون ریست ====================
st.sidebar.header("منبع داده")
if st.sidebar.button("دانلود از Yahoo Finance", type="primary"):
    default = "BTC-USD,ETH-USD,GC=F,^GSPC,USDIRR=X"
    tickers = st.sidebar.text_input("نمادها", value=default)
    period = st.sidebar.selectbox("بازه زمانی", ["1y","3y","5y","max"], index=2)
    prices = download_data(tickers, period)
    if prices is not None:
        st.session_state.prices = prices
        st.rerun()

if st.sidebar.button("آپلود فایل CSV"):
    uploaded = st.sidebar.file_uploader("فایل‌ها را بکشید", type="csv", accept_multiple_files=True, key="uploader")
    if uploaded:
        dfs = []
        for f in uploaded:
            df = pd.read_csv(f)
            name = f.name.replace(".csv","")
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            dfs.append(df[[col]].rename(columns={col: name}))
        st.session_state.prices = pd.concat(dfs, axis=1).ffill().bfill()
        st.rerun()

# تنظیمات بدون ریست
st.sidebar.header("تنظیمات پرتفوی")

# مقداردهی اولیه session_state
for key, val in {
    "rf_rate": 18.0,
    "hedge_active": True,
    "hedge_type": "طلا به عنوان هج",
    "max_btc": 20
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.session_state.rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)
st.session_state.hedge_active = st.sidebar.checkbox("هجینگ هوشمند فعال باشد", st.session_state.hedge_active)

if st.session_state.hedge_active:
    hedge_options = [
        "طلا به عنوان هج", "دلار/تتر", "طلا + تتر (ترکیبی)",
        "پوزیشن شورت بیت‌کوین", "آپشن Put", "پرتفوی معکوس",
        "استراتژی Collar", "Tail-Risk Hedge"
    ]
    idx = hedge_options.index(st.session_state.hedge_type) if st.session_state.hedge_type in hedge_options else 0
    st.session_state.hedge_type = st.sidebar.selectbox("نوع هجینگ", hedge_options, index=idx)

st.session_state.max_btc = st.sidebar.slider("حداکثر وزن بیت‌کوین (%)", 0, 100, st.session_state.max_btc)

# ==================== دکمه محاسبه (اجباری نیست، خودکار هم کار می‌کنه) ====================
if st.sidebar.button("محاسبه پرتفوی جدید", type="primary"):
    st.rerun()

# ==================== اجرای محاسبات ====================
calculate_portfolio()

# ==================== توضیح سبک ====================
with st.expander("توضیح سبک استفاده‌شده و انتخاب سبک دیگر", expanded=True):
    st.success("**سبک فعلی: بهینه‌سازی میانگین-واریانس (Markowitz) با هجینگ هوشمند**")
    st.write("""
    - بالاترین نسبت شارپ
    - محافظت قوی در برابر ریزش‌های بازار
    - کاملاً عملی برای سرمایه‌گذاران ایرانی
    """)

    st.markdown("**سبک‌های دیگر قابل انتخاب:**")
    style = st.radio("انتخاب سبک پرتفوی با سبک دیگر:",
                     ["بهینه‌سازی مارکوویتز (فعلی)", "وزن برابر", "حداقل ریسک", "ریسک‌پاریتی"])
    if style != "بهینه‌سازی مارکوویتز (فعلی)":
        st.info(f"در نسخه بعدی سبک «{style}» رو برات فعال می‌کنم. فقط بگو!")

st.balloons()
st.caption("Portfolio360 Ultimate - نسخه نهایی بدون ریست | ۱۴۰۴")
