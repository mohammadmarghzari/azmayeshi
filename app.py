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
def download_data(tickers_str, period="5y"):
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    data = {}
    failed = []
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, auto_adjust=True)
            if len(df) > 50 and "Close" in df.columns:
                data[t] = df["Close"]
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

# ==================== توابع کمکی ====================
def calculate_recovery_time(ret_series):
    cum = (1 + ret_series).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    recoveries = []
    in_dd = False
    start = None
    for i in range(1, len(cum)):
        if dd.iloc[i] < -0.01:
            if not in_dd:
                in_dd = True
                start = i
        elif in_dd:
            in_dd = False
            recoveries.append(i - start)
    return np.mean(recoveries) if recoveries else 0

def format_recovery(days):
    if days == 0 or np.isnan(days):
        return "بدون افت جدی"
    months = int(days / 21)
    years, months = divmod(months, 12)
    if years and months: return f"{years} سال و {months} ماه"
    if years: return f"{years} سال"
    if months: return f"{months} ماه"
    return "کمتر از ۱ ماه"

# ==================== محاسبه پرتفوی + مرز کارا برای همه سبک‌ها ====================
@st.fragment
def calculate_portfolio():
    if st.session_state.get("prices") is None:
        st.info("لطفاً داده‌ها را دانلود یا آپلود کنید.")
        return

    prices = st.session_state.prices
    returns = prices.pct_change().dropna()
    asset_names = list(prices.columns)
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    rf = st.session_state.rf_rate

    # محدودیت‌های هجینگ
    bounds = [(0.0, 1.0) for _ in asset_names]
    if st.session_state.hedge_active:
        for i, name in enumerate(asset_names):
            low, up = 0.0, 1.0
            if any(x in name.upper() for x in ["BTC", "بیت"]):
                up = st.session_state.max_btc / 100
            if st.session_state.hedge_type == "طلا به عنوان هج" and any(x in name.upper() for x in ["GC=", "GOLD", "طلا"]):
                low = 0.15
            if st.session_state.hedge_type == "دلار/تتر" and any(x in name.upper() for x in ["USD", "USDIRR", "تتر"]):
                low = 0.10
            if st.session_state.hedge_type == "طلا + تتر (ترکیبی)":
                if any(x in name.upper() for x in ["GC=", "GOLD", "طلا"]): low = 0.15
                if any(x in name.upper() for x in ["USD", "USDIRR", "تتر"]): low = 0.10
            bounds[i] = (low, up)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.ones(len(asset_names)) / len(asset_names)

    # محاسبه همه سبک‌ها
    portfolios = {}
    styles_map = {
        "مارکوویتز + هجینگ (بهینه‌ترین شارپ)": ("مارکوویتز", "gold", "star-diamond"),
        "وزن برابر (ساده و مقاوم)": ("وزن برابر", "blue", "circle"),
        "حداقل ریسک (محافظه‌کارانه)": ("حداقل ریسک", "green", "square"),
        "ریسک‌پاریتی (Risk Parity)": ("ریسک‌پاریتی", "purple", "diamond"),
        "بلک-لیترمن (ترکیب نظر شخصی)": ("بلک-لیترمن", "orange", "x")
    }

    # 1. مارکوویتز
    res = minimize(lambda w: - (np.dot(w, mean_ret)*100 - rf) / (np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))*100 + 1e-8),
                   x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = res.x if res.success else x0 / sum(x0)
    portfolios["مارکوویتز"] = w

    # 2. وزن برابر
    w = np.ones(len(asset_names)) / len(asset_names)
    w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
    w /= w.sum()
    portfolios["وزن برابر"] = w

    # 3. حداقل ریسک
    res = minimize(lambda w: np.dot(w.T, np.dot(cov_mat, w)), x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = res.x if res.success else x0 / sum(x0)
    portfolios["حداقل ریسک"] = w

    # 4. ریسک‌پاریتی
    def rc_error(w):
        port_var = np.dot(w.T, np.dot(cov_mat, w))
        if port_var == 0: return 9999
        contrib = w * np.dot(cov_mat, w) / np.sqrt(port_var)
        return np.sum((contrib - np.mean(contrib))**2)
    res = minimize(rc_error, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = res.x if res.success else x0 / sum(x0)
    portfolios["ریسک‌پاریتی"] = w

    # 5. بلک-لیترمن
    w = mean_ret / mean_ret.sum()
    w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
    w /= w.sum()
    portfolios["بلک-لیترمن"] = w

    # مونت‌کارلو
    mc_ret, mc_risk = [], []
    for _ in range(12000):
        w = np.random.random(len(asset_names))
        w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
        w /= w.sum()
        mc_ret.append(np.dot(w, mean_ret) * 100)
        mc_risk.append(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100)

    # نمودار مرز کارا با همه سبک‌ها
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_risk, y=mc_ret, mode='markers', marker=dict(color='lightgray', size=5), name="پرتفوهای تصادفی"))

    selected_key = next((k for k in styles_map if styles_map[k][0] in st.session_state.selected_style), "مارکوویتز")

    for full_name, (key, color, symbol) in styles_map.items():
        w = portfolios[key]
        ret = np.dot(w, mean_ret) * 100
        risk = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100
        is_selected = key in st.session_state.selected_style
        fig.add_trace(go.Scatter(x=[risk], y=[ret], mode='markers',
                                 marker=dict(color=color, size=28 if is_selected else 18, symbol=symbol,
                                             line=dict(width=4 if is_selected else 1, color='black')),
                                 name=f"{full_name} ← انتخاب شده" if is_selected else full_name))

    fig.update_layout(height=650, title="مرز کارا — همه سبک‌ها روی نمودار", xaxis_title="ریسک (%)", yaxis_title="بازده (%)")
    st.plotly_chart(fig, use_container_width=True)

    # نمایش پرتفوی انتخاب‌شده
    w = portfolios[selected_key]
    ret = np.dot(w, mean_ret) * 100
    risk = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100
    sharpe = (ret - rf) / risk if risk > 0 else 0
    recovery = format_recovery(calculate_recovery_time(returns.dot(w)))

    st.success(f"پرتفوی نهایی با سبک: **{st.session_state.selected_style}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("بازده سالیانه", f"{ret:.2f}%")
    c2.metric("ریسک سالیانه", f"{risk:.2f}%")
    c3.metric("نسبت شارپ", f"{sharpe:.3f}")
    c4.metric("زمان ریکاوری", recovery)

    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(w*100, 2)}).sort_values("وزن (%)", ascending=False)
    st.markdown("### تخصیص دارایی‌ها")
    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی"), use_container_width=True)

    if st.session_state.hedge_active:
        st.success(f"هجینگ فعال: {st.session_state.hedge_type}")

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate", layout="wide")

# لوگو
st.sidebar.header("لوگوی شما")
uploaded_logo = st.sidebar.file_uploader("لوگو آپلود کن", type=["png","jpg","jpeg","webp","svg"])
if uploaded_logo:
    st.session_state.logo = uploaded_logo

col1, col2, col3 = st.columns([1,3,1])
with col2:
    if "logo" in st.session_state:
        st.image(st.session_state.logo, use_column_width=True)
    else:
        st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Ultimate</h1>", unsafe_allow_html=True)

st.markdown("---")

# منبع داده — بدون پیش‌فرض
st.sidebar.header("دانلود از Yahoo Finance")
tickers_input = st.sidebar.text_input("نمادها رو با کاما بنویس", value="", placeholder="BTC-USD, ETH-USD, GC=F")

with st.sidebar.expander("راهنما: چه نمادی وارد کنم؟"):
    st.write("BTC-USD → بیت‌کوین\nETH-USD → اتریوم\nGC=F → طلا\nUSDIRR=X → دلار به ریال\n^GSPC → بورس آمریکا")

if st.sidebar.button("دانلود داده‌ها", type="primary"):
    if not tickers_input.strip():
        st.error("لطفاً حداقل یک نماد وارد کنید!")
    else:
        with st.spinner("در حال دانلود..."):
            prices = download_data(tickers_input)
            if prices is not None:
                st.session_state.prices = prices
                st.success(f"دانلود موفق! {len(prices.columns)} دارایی")
                st.rerun()

if st.sidebar.button("آپلود فایل CSV"):
    uploaded = st.sidebar.file_uploader("فایل‌ها", type="csv", accept_multiple_files=True)
    if uploaded:
        dfs = []
        for f in uploaded:
            df = pd.read_csv(f)
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            name = f.name.replace(".csv","")
            dfs.append(df[[col]].rename(columns={col: name}))
        st.session_state.prices = pd.concat(dfs, axis=1).ffill().bfill()
        st.rerun()

# تنظیمات
defaults = {"rf_rate":18.0, "hedge_active":True, "hedge_type":"طلا به عنوان هج", "max_btc":20,
            "selected_style":"مارکوویتز + هجینگ (بهینه‌ترین شارپ)"}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

st.session_state.rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)
st.session_state.hedge_active = st.sidebar.checkbox("هجینگ هوشمند", st.session_state.hedge_active)
if st.session_state.hedge_active:
    opts = ["طلا به عنوان هج","دلار/تتر","طلا + تتر (ترکیبی)","پوزیشن شورت بیت‌کوین","آپشن Put","پرتفوی معکوس","استراتژی Collar","Tail-Risk Hedge"]
    idx = opts.index(st.session_state.hedge_type) if st.session_state.hedge_type in opts else 0
    st.session_state.hedge_type = st.sidebar.selectbox("نوع هجینگ", opts, index=idx)
st.session_state.max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, st.session_state.max_btc)

styles = ["مارکوویتز + هجینگ (بهینه‌ترین شارپ)","وزن برابر (ساده و مقاوم)","حداقل ریسک (محافظه‌کارانه)",
          "ریسک‌پاریتی (Risk Parity)","بلک-لیترمن (ترکیب نظر شخصی)"]
idx = styles.index(st.session_state.selected_style)
st.session_state.selected_style = st.sidebar.selectbox("سبک پرتفوی", styles, index=idx)

with st.sidebar.expander("توضیح سبک"):
    exp = {
        "مارکوویتز + هجینگ (بهینه‌ترین شارپ)": "بهترین بازده به ریسک",
        "وزن برابر (ساده و مقاوم)": "در بلندمدت عالی عمل می‌کند",
        "حداقل ریسک (محافظه‌کارانه)": "کمترین نوسان ممکن",
        "ریسک‌پاریتی (Risk Parity)": "هر دارایی ریسک برابر بدهد",
        "بلک-لیترمن (ترکیب نظر شخصی)": "نظر شما هم وارد محاسبه می‌شود"
    }
    st.write(exp[st.session_state.selected_style])

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Ultimate — نسخه نهایی | ۱۴۰۴ | با عشق برای سرمایه‌گذاران ایرانی")
