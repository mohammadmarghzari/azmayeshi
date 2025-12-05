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

# ==================== ریسک پاریتی ====================
def risk_parity_objective(w, cov_mat):
    port_var = np.dot(w.T, np.dot(cov_mat, w))
    if port_var < 1e-10:
        return 9999.0
    marginal_contrib = np.dot(cov_mat, w)
    risk_contrib = w * marginal_contrib / np.sqrt(port_var)
    target = np.mean(risk_contrib)
    return np.sum((risk_contrib - target) ** 2)

# ==================== هوش مصنوعی هجینگ خودکار ====================
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_signals():
    try:
        usd = yf.Ticker("USDIRR=X").history(period="7d")["Close"].iloc[-1]
        usd_change_30d = (usd / yf.Ticker("USDIRR=X").history(period="90d")["Close"].iloc[-31]) - 1
        
        gold = yf.Ticker("GC=F").history(period="7d")["Close"].iloc[-1]
        gold_change_1y = (gold / yf.Ticker("GC=F").history(period="2y")["Close"].iloc[-252]) - 1
        
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        
        return {
            "usd_rising_fast": usd_change_30d > 0.15,
            "gold_rising": gold_change_1y > 0.25,
            "vix_high": vix > 25,
            "vix_extreme": vix > 35,
        }
    except:
        return {
            "usd_rising_fast": True,
            "gold_rising": True,
            "vix_high": True,
            "vix_extreme": False
        }

def recommend_hedge():
    s = get_market_signals()
    if s["vix_extreme"]:
        return "Tail-Risk Hedge", "وحشت جهانی! حداکثر حفاظت لازم است"
    if s["vix_high"] and s["usd_rising_fast"] and s["gold_rising"]:
        return "طلا + تتر (ترکیبی)", "بحران جهانی + سقوط ریال + تورم شدید"
    if s["usd_rising_fast"] and s["gold_rising"]:
        return "طلا + تتر (ترکیبی)", "وضعیت فعلی ایران — بهترین دفاع"
    if s["usd_rising_fast"]:
        return "دلار/تتر", "ریال در سقوط آزاد — دلار اولویت اول"
    if s["gold_rising"]:
        return "طلا به عنوان هج", "تورم و نااطمینانی جهانی"
    return "طلا + تتر (ترکیبی)", "حداقل حفاظت همیشه خوبه!"

# ==================== محاسبه پرتفوی ====================
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
    bounds = []
    for name in asset_names:
        low = 0.0
        up = 1.0
        if st.session_state.hedge_active:
            if any(x in name.upper() for x in ["BTC", "بیت"]):
                up = st.session_state.max_btc / 100.0
            if st.session_state.hedge_type == "طلا به عنوان هج" and any(x in name.upper() for x in ["GC=", "GOLD", "طلا"]):
                low = 0.15
            if st.session_state.hedge_type == "دلار/تتر" and any(x in name.upper() for x in ["USD", "USDIRR", "تتر"]):
                low = 0.10
            if st.session_state.hedge_type == "طلا + تتر (ترکیبی)":
                if any(x in name.upper() for x in ["GC=", "GOLD", "طلا"]):
                    low = 0.15
                if any(x in name.upper() for x in ["USD", "USDIRR", "تتر"]):
                    low = 0.10
        if low > up:
            low, up = 0.0, 1.0
        bounds.append((float(low), float(up)))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.ones(len(asset_names)) / len(asset_names)

    style_map = {
        "مارکوویتز + هجینگ (بهینه‌ترین شارپ)": "markowitz",
        "وزن برابر (ساده و مقاوم)": "equal",
        "حداقل ریسک (محافظه‌کارانه)": "minvar",
        "ریسک‌پاریتی (Risk Parity)": "risk_parity",
        "بلک-لیترمن (ترکیب نظر شخصی)": "bl"
    }
    selected = style_map.get(st.session_state.selected_style, "markowitz")

    try:
        if selected == "markowitz":
            obj = lambda w: -(np.dot(w, mean_ret)*100 - rf) / (np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))*100 + 1e-8)
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 3000})
            weights = res.x if res.success else x0
        elif selected == "equal":
            weights = np.ones(len(asset_names)) / len(asset_names)
            weights = np.clip(weights, [b[0] for b in bounds], [b[1] for b in bounds])
            weights /= weights.sum() if weights.sum() > 0 else 1
        elif selected == "minvar":
            res = minimize(lambda w: np.dot(w.T, np.dot(cov_mat, w)), x0, method="SLSQP", bounds=bounds, constraints=constraints)
            weights = res.x if res.success else x0
        elif selected == "risk_parity":
            res = minimize(risk_parity_objective, x0, args=(cov_mat,), method="SLSQP", bounds=bounds, constraints=constraints)
            weights = res.x if res.success else x0
        else:  # Black-Litterman ساده
            weights = mean_ret / mean_ret.sum()
            weights = np.nan_to_num(weights, nan=1.0/len(asset_names))
            weights = np.clip(weights, [b[0] for b in bounds], [b[1] for b in bounds])
            weights /= weights.sum() if weights.sum() > 0 else 1
    except Exception as e:
        st.error(f"خطا در بهینه‌سازی: {str(e)}")
        weights = x0

    weights = np.clip(weights, 0, 1)
    if weights.sum() > 0:
        weights /= weights.sum()

    ret = np.dot(weights, mean_ret) * 100
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
    sharpe = (ret - rf) / risk if risk > 0 else 0
    recovery = format_recovery(calculate_recovery_time(returns.dot(weights)))

    # مونت‌کارلو برای مرز کارا
    mc_ret, mc_risk = [], []
    for _ in range(12000):
        w = np.random.random(len(asset_names))
        w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
        w /= w.sum() if w.sum() > 0 else 1
        mc_ret.append(np.dot(w, mean_ret) * 100)
        mc_risk.append(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_risk, y=mc_ret, mode='markers', marker=dict(color='lightgray', size=5), name="پرتفوی‌های تصادفی"))

    # رسم همه سبک‌ها روی نمودار
    styles_info = {
        "markowitz": ("مارکوویتز + هجینگ", "#00d2d3", "star-diamond"),
        "equal": ("وزن برابر", "#1f77b4", "circle"),
        "minvar": ("حداقل ریسک", "#2ca02c", "square"),
        "risk_parity": ("ریسک‌پاریتی", "#9467bd", "diamond"),
        "bl": ("بلک-لیترمن", "#ff7f0e", "x")
    }

    for key, (name, color, symbol) in styles_info.items():
        if key == "markowitz":
            w_opt = weights
        elif key == "equal":
            w_opt = np.ones(len(asset_names)) / len(asset_names)
        elif key == "minvar":
            res = minimize(lambda w: np.dot(w.T, np.dot(cov_mat, w)), x0, method="SLSQP", bounds=bounds, constraints=constraints)
            w_opt = res.x if res.success else x0
        elif key == "risk_parity":
            res = minimize(risk_parity_objective, x0, args=(cov_mat,), method="SLSQP", bounds=bounds, constraints=constraints)
            w_opt = res.x if res.success else x0
        else:
            w_opt = mean_ret / mean_ret.sum()
            w_opt = np.nan_to_num(w_opt, nan=1.0/len(asset_names))

        w_opt = np.clip(w_opt, [b[0] for b in bounds], [b[1] for b in bounds])
        if w_opt.sum() > 0:
            w_opt /= w_opt.sum()

        port_ret = np.dot(w_opt, mean_ret) * 100
        port_risk = np.sqrt(np.dot(w_opt.T, np.dot(cov_mat, w_opt))) * 100

        fig.add_trace(go.Scatter(
            x=[port_risk],
            y=[port_ret],
            mode='markers',
            marker=dict(color=color, size=16, symbol=symbol, line=dict(width=3, color='black')),
            name=name,
            text=f"{name}<br>بازده: {port_ret:.2f}%<br>ریسک: {port_risk:.2f}%",
            hoverinfo="text"
        ))

    fig.update_layout(
        height=650,
        title="مرز کارا — مقایسه همه سبک‌های بهینه‌سازی",
        xaxis_title="ریسک سالانه (%)",
        yaxis_title="بازده سالانه (%)",
        template="plotly_dark",
        hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True)

    # نمایش نتایج
    st.success(f"پرتفوی بهینه با سبک: **{st.session_state.selected_style}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("بازده سالانه", f"{ret:.2f}%")
    c2.metric("ریسک سالانه", f"{risk:.2f}%")
    c3.metric("نسبت شارپ", f"{sharpe:.3f}")
    c4.metric("زمان متوسط ریکاوری", recovery)

    df_w = pd.DataFrame({
        "دارایی": asset_names,
        "وزن (%)": np.round(weights*100, 2)
    }).sort_values("وزن (%)", ascending=False)

    st.markdown("### تخصیص دارایی‌ها")
    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی", color_discrete_sequence=px.colors.sequential.Plasma), use_container_width=True)

    if st.session_state.hedge_active:
        st.success(f"هجینگ فعال: {st.session_state.hedge_type}")

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate - هوش مصنوعی هجینگ ایرانی", layout="wide")

# لوگو
st.sidebar.header("لوگوی شما")
uploaded_logo = st.sidebar.file_uploader("آپلود لوگو", type=["png","jpg","jpeg","webp","svg"])
if uploaded_logo:
    st.session_state.logo = uploaded_logo

col1, col2, col3 = st.columns([1,3,1])
with col2:
    if "logo" in st.session_state:
        st.image(st.session_state.logo, use_column_width=True)
    else:
        st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Ultimate</h1>", unsafe_allow_html=True)

st.markdown("---")

# دانلود داده
st.sidebar.header("دانلود از Yahoo Finance")
tickers_input = st.sidebar.text_input("نمادها (با کاما جدا کنید)", value="BTC-USD, GC=F, USDIRR=X, ^GSPC", placeholder="BTC-USD, GC=F, ...")
with st.sidebar.expander("راهنمای نمادها"):
    st.write("• BTC-USD → بیت‌کوین\n• GC=F → طلا\n• USDIRR=X → دلار به ریال\n• ^GSPC → شاخص S&P 500\n• EURUSD=X → یورو به دلار")

if st.sidebar.button("دانلود داده‌ها", type="primary"):
    if not tickers_input.strip():
        st.error("حداقل یک نماد وارد کنید!")
    else:
        with st.spinner("در حال دانلود داده‌ها از Yahoo Finance..."):
            prices = download_data(tickers_input)
            if prices is not None:
                st.session_state.prices = prices
                st.success("داده‌ها با موفقیت دانلود شد!")
                st.rerun()

# تنظیمات پیش‌فرض
defaults = {
    "rf_rate": 18.0,
    "hedge_active": True,
    "hedge_type": "طلا + تتر (ترکیبی)",
    "max_btc": 20,
    "selected_style": "مارکوویتز + هجینگ (بهینه‌ترین شارپ)"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.session_state.rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)
st.session_state.hedge_active = st.sidebar.checkbox("فعال‌سازی هجینگ هوشمند", st.session_state.hedge_active)
st.session_state.max_btc = st.sidebar.slider("حداکثر تخصیص بیت‌کوین (%)", 0, 100, st.session_state.max_btc)

# هوش مصنوعی هجینگ خودکار
auto_rec, auto_reason = recommend_hedge()
with st.sidebar.expander("هوش مصنوعی هجینگ خودکار (Auto-Hedge AI)", expanded=True):
    st.success(f"پیشنهاد هوش مصنوعی: **{auto_rec}**")
    st.info(auto_reason)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("اعمال پیشنهاد AI", type="primary"):
            st.session_state.hedge_type = auto_rec
            st.session_state.hedge_active = True
            st.rerun()
    with col2:
        if st.checkbox("همیشه پیشنهاد AI را اعمال کن", key="auto_apply_hedge"):
            st.session_state.hedge_type = auto_rec

if st.session_state.get("auto_apply_hedge", False):
    st.session_state.hedge_type = auto_rec

# هجینگ دستی
if not st.session_state.get("auto_apply_hedge", False):
    if st.session_state.hedge_active:
        opts = ["طلا به عنوان هج", "دلار/تتر", "طلا + تتر (ترکیبی)", "Tail-Risk Hedge", "بدون هج"]
        idx = opts.index(st.session_state.hedge_type) if st.session_state.hedge_type in opts else 2
        st.session_state.hedge_type = st.sidebar.selectbox("نوع هجینگ", opts, index=idx)

# انتخاب سبک پرتفوی
styles = [
    "مارکوویتز + هجینگ (بهینه‌ترین شارپ)",
    "وزن برابر (ساده و مقاوم)",
    "حداقل ریسک (محافظه‌کارانه)",
    "ریسک‌پاریتی (Risk Parity)",
    "بلک-لیترمن (ترکیب نظر شخصی)"
]
idx = styles.index(st.session_state.selected_style) if st.session_state.selected_style in styles else 0
st.session_state.selected_style = st.sidebar.selectbox("سبک بهینه‌سازی پرتفوی", styles, index=idx)

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Ultimate — اولین اپلیکیشن ایرانی با هوش مصنوعی هجینگ خودکار | ۱۴۰۴ | با عشق برای ایران")
