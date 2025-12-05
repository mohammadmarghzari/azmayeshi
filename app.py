import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ==================== دانلود داده ====================
@st.cache_data(show_spinner=False)
def download_data(tickers_str, period="5y"):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
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
        st.error("هیچ داده‌ای دانلود نشد. نمادها را چک کنید.")
        return None
    prices = pd.DataFrame(data).ffill().bfill()
    if failed:
        st.sidebar.warning(f"دانلود نشد: {', '.join(failed)}")
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

# ==================== استراتژی‌های هجینگ ====================
hedge_strategies = {
    "طلا + تتر (ترکیبی)": {"gold_min": 0.15, "usd_min": 0.10, "btc_max": 0.20},
    "Regime-Switching": {"gold_min": 0.30, "usd_min": 0.20, "btc_max": 0.10},
    "Barbell (نسیم طالب)": {"gold_min": 0.40, "usd_min": 0.45, "btc_max": 0.15},
    "Tail-Risk Active": {"gold_min": 0.35, "usd_min": 0.35, "btc_max": 0.05},
    "حداقل هج": {"gold_min": 0.10, "usd_min": 0.00, "btc_max": 0.40},
    "بدون هجینگ": {"gold_min": 0.00, "usd_min": 0.00, "btc_max": 1.00}
}

# ==================== استراتژی‌های آپشن ====================
option_strategies = {
    "بدون آپشن": {"cost_pct": 0.0},
    "Protective Put": {"cost_pct": 4.8},
    "Collar (تقریباً رایگان)": {"cost_pct": 0.4},
    "Covered Call": {"cost_pct": -3.2},
    "Cash-Secured Put": {"cost_pct": -2.9},
    "Tail-Risk Put": {"cost_pct": 2.1},
    "Iron Condor": {"cost_pct": -5.5},
}

# ==================== محاسبه پرتفوی ====================
@st.fragment
def calculate_portfolio():
    if "prices" not in st.session_state or st.session_state.prices is None:
        st.info("لطفاً داده‌ها را دانلود کنید.")
        return

    prices = st.session_state.prices
    returns = prices.pct_change().dropna()
    asset_names = list(prices.columns)
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    rf = st.session_state.rf_rate / 100

    hedge = st.session_state.hedge_strategy
    opt = st.session_state.option_strategy
    info_hedge = hedge_strategies[hedge]
    info_opt = option_strategies[opt]

    # تشخیص وجود دارایی‌های هجینگ
    has_gold = any(x in name.upper() for name in asset_names for x in ["GC=", "GOLD", "SI="])
    has_usd = any(x in name.upper() for name in asset_names for x in ["USD", "USDIRR", "USDT"])
    has_btc = any(x in name.upper() for name in asset_names for x in ["BTC", "بیت"])

    # هشدار اگر دارایی لازم نباشه
    if hedge != "بدون هجینگ":
        if info_hedge["gold_min"] > 0 and not has_gold:
            st.warning("برای این استراتژی، نماد طلا (مثل GC=F) لازم است!")
        if info_hedge["usd_min"] > 0 and not has_usd:
            st.warning("برای این استراتژی، نماد دلار (مثل USDIRR=X) لازم است!")
        if info_hedge["btc_max"] < 1.0 and not has_btc:
            st.info("محدودیت بیت‌کوین اعمال نمی‌شود (نماد BTC-USD نیست)")

    # محدودیت‌ها
    bounds = []
    for name in asset_names:
        low = 0.0
        up = 1.0
        n = name.upper()
        if any(x in n for x in ["GC=", "GOLD", "SI="]): 
            low = max(low, info_hedge["gold_min"])
        if any(x in n for x in ["USD", "USDIRR", "USDT"]): 
            low = max(low, info_hedge["usd_min"])
        if any(x in n for x in ["BTC", "بیت"]): 
            up = min(up, info_hedge["btc_max"])
        bounds.append((low, up))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.ones(len(asset_names)) / len(asset_names)

    try:
        obj = lambda w: -(np.dot(w, mean_ret) - rf) / (np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) + 1e-8)
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 5000})
        weights = res.x if res.success else x0
    except:
        weights = x0

    weights = np.clip(weights, [b[0] for b in bounds], [b[1] for b in bounds])
    weights /= weights.sum() if weights.sum() > 0 else 1

    ret = np.dot(weights, mean_ret) * 100
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
    sharpe = (ret - st.session_state.rf_rate) / risk if risk > 0 else 0
    recovery = format_recovery(calculate_recovery_time(returns.dot(weights)))

    st.success(f"استراتژی فعال: **{hedge}** + **{opt}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("بازده سالانه", f"{ret:.2f}%")
    c2.metric("ریسک سالانه", f"{risk:.2f}%")
    c3.metric("نسبت شارپ", f"{sharpe:.3f}")
    c4.metric("زمان ریکاوری", recovery)

    if info_opt["cost_pct"] > 0:
        st.warning(f"هزینه آپشن سالانه ≈ {info_opt['cost_pct']:.1f}%")
    elif info_opt["cost_pct"] < 0:
        st.success(f"درآمد سالانه از آپشن ≈ {abs(info_opt['cost_pct']):.1f}%")

    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)}).sort_values("وزن (%)", ascending=False)
    st.markdown("### تخصیص دارایی")
    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی", color_discrete_sequence=px.colors.sequential.Plasma), use_container_width=True)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate – ایران ۱۴۰۴", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Ultimate</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>اپ حرفه‌ای سرمایه‌گذاری ایرانی</h3>", unsafe_allow_html=True)

# پیش‌فرض‌ها
defaults = {"rf_rate": 18.0, "max_btc": 25, "hedge_strategy": "طلا + تتر (ترکیبی)", "option_strategy": "Collar (تقریباً رایگان)"}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# سایدبار
st.sidebar.header("دانلود داده")
tickers = st.sidebar.text_input("نمادها (مثال: BTC-USD, GC=F, USDIRR=X)", value="BTC-USD, GC=F, USDIRR=X, ^GSPC")
if st.sidebar.button("دانلود داده‌ها", type="primary"):
    with st.spinner("در حال دانلود..."):
        data = download_data(tickers)
        if data is not None:
            st.session_state.prices = data
            st.rerun()

st.sidebar.header("تنظیمات")
st.session_state.rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)
st.session_state.max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, st.session_state.max_btc)

st.sidebar.markdown("### استراتژی هجینگ")
for name, info in hedge_strategies.items():
    st.sidebar.markdown(f"**{name}** → طلا: {info['gold_min']*100:.0f}% | دلار: {info['usd_min']*100:.0f}% | بیت‌کوین: ≤{info['btc_max']*100:.0f}%")
st.session_state.hedge_strategy = st.sidebar.selectbox("انتخاب هجینگ", options=list(hedge_strategies.keys()))

st.sidebar.markdown("### استراتژی آپشن")
for name, info in option_strategies.items():
    cost = f"+{info['cost_pct']:.1f}%" if info['cost_pct'] > 0 else f"{info['cost_pct']:.1f}% (درآمد)"
    st.sidebar.markdown(f"**{name}** → هزینه: {cost}")
st.session_state.option_strategy = st.sidebar.selectbox("انتخاب آپشن", options=list(option_strategies.keys()))

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Ultimate – نسخه نهایی | با عشق برای ایران")
