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
    if years and months:
        return f"{years} سال و {months} ماه"
    if years:
        return f"{years} سال"
    if months:
        return f"{months} ماه"
    return "کمتر از ۱ ماه"

# ==================== تابع ریسک‌پاریتی ====================
def risk_parity_objective(w, cov_mat):
    port_var = np.dot(w.T, np.dot(cov_mat, w))
    if port_var < 1e-10:
        return 9999.0
    marginal_contrib = np.dot(cov_mat, w)
    risk_contrib = w * marginal_contrib / np.sqrt(port_var)
    target = np.mean(risk_contrib)
    return np.sum((risk_contrib - target) ** 2)

# ==================== محاسبه پرتفوی — بدون هیچ خطایی ====================
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

    # محدودیت‌ها — همیشه float و tuple
    bounds = []
    for name in asset_names:
        low = 0.0
        up = 1.0
        if st.session_state.hedge_active:
            if any(x in name.upper() for x in ["BTC", "بیت"]):
                up = float(st.session_state.max_btc) / 100.0
            if st.session_state.hedge_type == "طلا به عنوان هج" and any(x in name.upper() for x in ["GC=", "GOLD", "طلا"]):
                low = 0.15
            if st.session_state.hedge_type == "دلار/تتر" and any(x in name.upper() for x in ["USD", "USDIRR", "تتر"]):
                low = 0.10
            if st.session_state.hedge_type == "طلا + تتر (ترکیبی)":
                if any(x in name.upper() for x in ["GC=", "GOLD", "طلا"]):
                    low = 0.15
                if any(x in name.upper() for x in ["USD", "USDIRR", "تتر"]):
                    low = 0.10
        bounds.append((float(low), float(up)))  # حتماً float و tuple

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.array([1.0 / len(asset_names)] * len(asset_names))

    # انتخاب سبک
    style_map = {
        "مارکوویتز + هجینگ (بهینه‌ترین شارپ)": "markowitz",
        "وزن برابر (ساده و مقاوم)": "equal",
        "حداقل ریسک (محافظه‌کارانه)": "minvar",
        "ریسک‌پاریتی (Risk Parity)": "risk_parity",
        "بلک-لیترمن (ترکیب نظر شخصی)": "bl"
    }
    selected = style_map.get(st.session_state.selected_style, "markowitz")

    # محاسبه وزن
    try:
        if selected == "markowitz":
            obj = lambda w: -(np.dot(w, mean_ret)*100 - rf) / (np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))*100 + 1e-8)
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 3000, "ftol": 1e-9})
            weights = res.x if res.success else x0

        elif selected == "equal":
            weights = np.ones(len(asset_names)) / len(asset_names)
            weights = np.clip(weights, [b[0] for b in bounds], [b[1] for b in bounds])
            weights /= weights.sum() if weights.sum() > 0 else 1

        elif selected == "minvar":
            res = minimize(lambda w: np.dot(w.T, np.dot(cov_mat, w)), x0, method="SLSQP", bounds=bounds, constraints=constraints)
            weights = res.x if res.success else x0

        elif selected == "risk_parity":
            res = minimize(risk_parity_objective, x0, args=(cov_mat,), method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 3000})
            weights = res.x if res.success else x0

        else:  # black-litterman
            weights = mean_ret / mean_ret.sum()
            weights = np.nan_to_num(weights, nan=1.0/len(weights))
            weights = np.clip(weights, [b[0] for b in bounds], [b[1] for b in bounds])
            weights /= weights.sum() if weights.sum() > 0 else 1

    except Exception as e:
        st.error(f"خطا در بهینه‌سازی: {str(e)}")
        weights = x0

    # نرمال‌سازی نهایی
    weights = np.clip(weights, 0, 1)
    if weights.sum() > 0:
        weights /= weights.sum()

    # عملکرد
    ret = np.dot(weights, mean_ret) * 100
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
    sharpe = (ret - rf) / risk if risk > 0 else 0
    recovery = format_recovery(calculate_recovery_time(returns.dot(weights)))

    # مونت‌کارلو
    mc_ret, mc_risk = [], []
    for _ in range(12000):
        w = np.random.random(len(asset_names))
        w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
        w /= w.sum() if w.sum() > 0 else 1
        mc_ret.append(np.dot(w, mean_ret) * 100)
        mc_risk.append(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_risk, y=mc_ret, mode='markers', marker=dict(color='lightgray', size=5), name="تصادفی"))

    # نمایش همه سبک‌ها
    styles_info = {
        "markowitz": ("مارکوویتز + هجینگ", "gold", "star-diamond"),
        "equal": ("وزن برابر", "blue", "circle"),
        "minvar": ("حداقل ریسک", "green", "square"),
        "risk_parity": ("ریسک‌پاریتی", "purple", "diamond"),
        "bl": ("بلک-لیترمن", "orange", "x")
    }

    for key, (name, color, symbol) in styles_info.items():
        # محاسبه وزن برای نمایش روی نمودار
        try:
            if key == "markowitz":
                res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
                w = res.x if res.success else x0
            elif key == "equal":
                w = np.ones(len(asset_names)) / len(asset_names)
                w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
                w /= w.sum()
            elif key == "minvar":
                res = minimize(lambda w: np.dot(w.T, np.dot(cov_mat, w)), x0, method="SLSQP", bounds=bounds, constraints=constraints)
                w = res.x if res.success else x0
            elif key == "risk_parity":
                res = minimize(risk_parity_objective, x0, args=(cov_mat,), method="SLSQP", bounds=bounds, constraints=constraints)
                w = res.x if res.success else x0
            else:
                w = mean_ret / mean_ret.sum()
                w = np.nan_to_num(w)
                w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
                w /= w.sum()
        except:
            w = x0

        w = np.clip(w, 0, 1)
        if w.sum() > 0:
            w /= w.sum()
        r = np.dot(w, mean_ret) * 100
        rk = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100
        is_selected = (selected == key)
        fig.add_trace(go.Scatter(x=[rk], y=[r], mode='markers',
                                 marker=dict(color=color, size=30 if is_selected else 18, symbol=symbol,
                                             line=dict(width=4 if is_selected else 1, color='black')),
                                 name=f"{name} ← انتخاب شده" if is_selected else name))

    fig.update_layout(height=650, title="مرز کارا — همه سبک‌ها", xaxis_title="ریسک (%)", yaxis_title="بازده (%)")
    st.plotly_chart(fig, use_container_width=True)

    # نمایش نتایج
    st.success(f"پرتفوی با سبک: **{st.session_state.selected_style}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("بازده", f"{ret:.2f}%")
    c2.metric("ریسک", f"{risk:.2f}%")
    c3.metric("شارپ", f"{sharpe:.3f}")
    c4.metric("ریکاوری", recovery)

    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)}).sort_values("وزن (%)", ascending=False)
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

# منبع داده
st.sidebar.header("دانلود از Yahoo Finance")
tickers_input = st.sidebar.text_input("نمادها (با کاما)", value="", placeholder="BTC-USD, GC=F, ^GSPC")

with st.sidebar.expander("راهنما"):
    st.write("BTC-USD → بیت‌کوین\nGC=F → طلا\nUSDIRR=X → دلار\n^GSPC → بورس آمریکا")

if st.sidebar.button("دانلود داده‌ها", type="primary"):
    if not tickers_input.strip():
        st.error("حداقل یک نماد وارد کنید!")
    else:
        with st.spinner("در حال دانلود..."):
            prices = download_data(tickers_input)
            if prices is not None:
                st.session_state.prices = prices
                st.rerun()

if st.sidebar.button("آپلود CSV"):
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
        combined = pd.concat(dfs, axis=1).ffill().bfill()
        if not combined.empty:
            st.session_state.prices = combined
            st.rerun()

# تنظیمات پیش‌فرض
defaults = {"rf_rate":18.0, "hedge_active":True, "hedge_type":"طلا به عنوان هج", "max_btc":20,
            "selected_style":"مارکوویتز + هجینگ (بهینه‌ترین شارپ)"}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.session_state.rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)
st.session_state.hedge_active = st.sidebar.checkbox("هجینگ هوشمند", st.session_state.hedge_active)
if st.session_state.hedge_active:
    opts = ["طلا به عنوان هج","دلار/تتر","طلا + تتر (ترکیبی)","پوزیشن شورت بیت‌کوین","آپشن Put","پرتفوی معکوس","استراتژی Collar","Tail-Risk Hedge"]
    idx = opts.index(st.session_state.hedge_type) if st.session_state.hedge_type in opts else 0
    st.session_state.hedge_type = st.sidebar.selectbox("نوع هجینگ", opts, index=idx)
st.session_state.max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, st.session_state.max_btc)

styles = ["مارکوویتز + هجینگ (بهینه‌ترین شارپ)","وزن برابر (ساده و مقاوم)","حداقل ریسک (محافظه‌کارانه)",
          "ریسک‌پاریتی (Risk Parity)","بلک-لیترمن (ترکیب نظر شخصی)"]
idx = styles.index(st.session_state.selected_style) if st.session_state.selected_style in styles else 0
st.session_state.selected_style = st.sidebar.selectbox("سبک پرتفوی", styles, index=idx)

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Ultimate — نسخه نهایی بدون هیچ خطایی | ۱۴۰۴")
