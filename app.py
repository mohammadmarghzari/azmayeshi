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
        st.error("هیچ داده‌ای دانلود نشد.")
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

def max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min() * 100

def max_gain(returns):
    annual = (1 + returns).resample('Y').last().pct_change().dropna() * 100
    return annual.max() if not annual.empty else 0

# ==================== استراتژی‌های هجینگ ====================
hedge_strategies = {
    "Barbell طالب (۹۰/۱۰)": {"gold_min": 0.45, "usd_min": 0.45, "btc_max": 0.10},
    "Tail-Risk طالب": {"gold_min": 0.35, "usd_min": 0.35, "btc_max": 0.05},
    "Antifragile طالب": {"gold_min": 0.40, "usd_min": 0.20, "btc_max": 0.40},
    "طلا + تتر (ترکیبی)": {"gold_min": 0.15, "usd_min": 0.10, "btc_max": 0.20},
    "حداقل هج": {"gold_min": 0.10, "usd_min": 0.00, "btc_max": 0.40},
    "بدون هجینگ (کاملاً آزاد)": {"gold_min": 0.00, "usd_min": 0.00, "btc_max": 1.00},
}

# ==================== جدول هجینگ ====================
hedge_table = pd.DataFrame([
    ["Barbell طالب (۹۰/۱۰)", "۹۰٪ ایمن + ۱۰٪ پرریسک", "همیشه — ایده اصلی نسیم طالب", "۴۵٪", "۴۵٪", "۱۰٪"],
    ["Tail-Risk طالب", "حفاظت در فاجعه + Put دور", "وقتی احتمال بلک سوان بالاست", "۳۵٪", "۳۵٪", "۵٪"],
    ["Antifragile طالب", "هرچه آشوب بیشتر، سود بیشتر!", "ایران ۱۴۰۴ با تورم و تحریم", "۴۰٪", "۲۰٪", "۴۰٪"],
    ["طلا + تتر (ترکیبی)", "تعادل بین حفاظت و رشد", "سرمایه‌گذار معمولی ایرانی", "۱۵٪", "۱۰٪", "۲۰٪"],
    ["حداقل هج", "فقط یک لایه حفاظتی کوچک", "ریسک‌پذیر در بازار صعودی", "۱۰٪", "۰٪", "۴۰٪"],
    ["بدون هجینگ (کاملاً آزاد)", "هیچ محدودیتی ندارد", "فقط برای تحمل ضرر ۵۰–۸۰٪", "۰٪", "۰٪", "۱۰۰٪"],
], columns=["استراتژی", "چرا این استراتژی؟", "مناسب برای", "حداقل طلا", "حداقل تتر/دلار", "حداکثر بیت‌کوین"])

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

# ==================== جدول آپشن ====================
option_table = pd.DataFrame([
    ["بدون آپشن", "ساده و بدون هزینه", "بازار صعودی قوی", "۰٪"],
    ["Protective Put", "بیمه کامل پرتفوی", "خواب راحت در بحران", "+۴.۸٪"],
    ["Collar (تقریباً رایگان)", "هج کم‌هزینه", "بازار متوسط", "+۰.۴٪"],
    ["Covered Call", "درآمد ماهانه", "بازار رنج", "−۳.۲٪ (درآمد)"],
    ["Cash-Secured Put", "خرید ارزان در ریزش", "وقتی به کف اعتقاد دارید", "−۲.۹٪ (درآمد)"],
    ["Tail-Risk Put", "حفاظت در فاجعه", "کرونا، جنگ", "+۲.۱٪"],
    ["Iron Condor", "درآمد از ثبات", "بازار آرام", "−۵.۵٪ (درآمد)"],
], columns=["استراتژی", "چرا این استراتژی؟", "مناسب برای", "هزینه سالانه"])

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

    # تشخیص دارایی‌ها
    has_gold = any(x in name.upper() for name in asset_names for x in ["GC=", "GOLD", "SI="])
    has_usd = any(x in name.upper() for name in asset_names for x in ["USD", "USDIRR", "USDT"])
    has_btc = any(x in name.upper() for name in asset_names for x in ["BTC", "بیت"])

    # محدودیت‌ها
    bounds = []
    for name in asset_names:
        low = 0.0
        up = 1.0
        n = name.upper()
        if any(x in n for x in ["GC=", "GOLD", "SI="]) and has_gold:
            low = max(low, info_hedge["gold_min"])
        if any(x in n for x in ["USD", "USDIRR", "USDT"]) and has_usd:
            low = max(low, info_hedge["usd_min"])
        if any(x in n for x in ["BTC", "بیت"]) and has_btc:
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

    port_returns = returns.dot(weights)
    port_return = np.dot(weights, mean_ret) * 100
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
    sharpe = (port_return - st.session_state.rf_rate) / port_risk if port_risk > 0 else 0
    recovery = format_recovery(calculate_recovery_time(port_returns))
    max_dd = max_drawdown(port_returns)
    max_g = max_gain(port_returns)

    st.success(f"استراتژی فعال: **{hedge}** + **{opt}**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("بازده سالانه", f"{port_return:.2f}%")
    c2.metric("بهترین حالت", f"{max_g:.2f}%")
    c3.metric("ریسک سالانه", f"{port_risk:.2f}%")
    c4.metric("بدترین حالت", f"{max_dd:.2f}%")
    c5.metric("نسبت شارپ", f"{sharpe:.3f}")
    c6.metric("زمان ریکاوری", recovery)

    if info_opt["cost_pct"] > 0:
        st.warning(f"هزینه آپشن سالانه ≈ {info_opt['cost_pct']:.1f}%")
    elif info_opt["cost_pct"] < 0:
        st.success(f"درآمد سالانه از آپشن ≈ {abs(info_opt['cost_pct']):.1f}%")

    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)}).sort_values("وزن (%)", ascending=False)
    df_w = df_w[df_w["وزن (%)"] > 0]
    st.markdown("### تخصیص دارایی")
    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی", color_discrete_sequence=px.colors.sequential.Plasma), use_container_width=True)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate – کامل", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Ultimate</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>اپ حرفه‌ای سرمایه‌گذاری ایرانی</h3>", unsafe_allow_html=True)

# پیش‌فرض‌ها
defaults = {"rf_rate": 18.0, "hedge_strategy": "Barbell طالب (۹۰/۱۰)", "option_strategy": "بدون آپشن"}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# سایدبار
st.sidebar.header("دانلود داده")
tickers = st.sidebar.text_input("نمادها", value="BTC-USD, GC=F, USDIRR=X, ^GSPC")
if st.sidebar.button("دانلود داده‌ها", type="primary"):
    with st.spinner("در حال دانلود..."):
        data = download_data(tickers)
        if data is not None:
            st.session_state.prices = data
            st.rerun()

st.sidebar.header("تنظیمات")
st.session_state.rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)

# جدول هجینگ
st.sidebar.markdown("### راهنمای کامل استراتژی‌های هجینگ")
st.sidebar.dataframe(hedge_table, use_container_width=True, hide_index=True)
st.session_state.hedge_strategy = st.sidebar.selectbox("انتخاب هجینگ", options=hedge_table["استراتژی"])

# جدول آپشن
st.sidebar.markdown("### راهنمای کامل استراتژی‌های آپشن")
st.sidebar.dataframe(option_table, use_container_width=True, hide_index=True)
st.session_state.option_strategy = st.sidebar.selectbox("انتخاب آپشن", options=option_table["استراتژی"])

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Ultimate – نسخه نهایی با ریکاوری تایم + بهترین/بدترین | با عشق برای ایران")
