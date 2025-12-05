import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf
import math
import warnings
warnings.filterWarnings("ignore")

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

# ==================== تخمین قیمت آپشن (بلک-شولز ساده) ====================
def estimate_put_price(volatility, days_to_expiry=30, strike_pct=0.90):
    T = days_to_expiry / 365.0
    r = st.session_state.rf_rate / 100
    sigma = volatility
    if sigma == 0: sigma = 0.01
    d1 = (np.log(1/strike_pct) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put_price = strike_pct*np.exp(-r*T)*norm.cdf(-d2) - norm.cdf(-d1)
    return max(put_price * 100, 0.1)

# ==================== تشخیص رژیم بازار ====================
@st.cache_data(ttl=3600)
def get_market_regime():
    try:
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        usd = yf.Ticker("USDIRR=X").history(period="90d")["Close"]
        usd_change = (usd.iloc[-1] / usd.iloc[-31]) - 1 if len(usd) > 31 else 0.2
        btc = yf.Ticker("BTC-USD").history(period="200d")["Close"]
        ma50 = btc.rolling(50).mean().iloc[-1] if len(btc) >= 50 else btc.iloc[-1]
        ma200 = btc.rolling(200).mean().iloc[-1] if len(btc) >= 200 else btc.iloc[-1]
        btc_bear = ma50 < ma200
        return {
            "crisis": vix > 35 or usd_change > 0.20,
            "high_vol": vix > 25,
            "usd_crisis": usd_change > 0.15,
            "btc_bear": btc_bear,
            "vix": vix
        }
    except:
        return {"crisis": True, "high_vol": True, "usd_crisis": True, "btc_bear": True, "vix": 40}

# ==================== استراتژی‌های هجینگ و آپشن ====================
hedge_strategies = {
    "طلا + تتر (ترکیبی)": {"desc": "۱۵٪ طلا + ۱۰٪ تتر – بهترین دفاع فعلی", "gold_min": 0.15, "usd_min": 0.10, "btc_max": 0.20},
    "Regime-Switching": {"desc": "در بحران → ۵۰٪+ هج | عادی → آزاد", "gold_min": 0.30, "usd_min": 0.20, "btc_max": 0.10},
    "Barbell (نسیم طالب)": {"desc": "۸۵٪ ایمن + ۱۵٪ خیلی پرریسک", "gold_min": 0.40, "usd_min": 0.45, "btc_max": 0.15},
    "Tail-Risk Active": {"desc": "حداکثر حفاظت در وحشت جهانی", "gold_min": 0.35, "usd_min": 0.35, "btc_max": 0.05},
    "حداقل هج": {"desc": "فقط ۱۰٪ طلا – ریسک‌پذیر", "gold_min": 0.10, "usd_min": 0.00, "btc_max": 0.40},
    "بدون هجینگ": {"desc": "کاملاً آزاد – فقط بهینه‌سازی", "gold_min": 0.00, "usd_min": 0.00, "btc_max": 1.00}
}

option_strategies = {
    "بدون آپشن": {"desc": "بدون پوشش آپشن", "cost": 0.0, "protection": 0, "income": 0},
    "Protective Put": {"desc": "خرید Put برای محافظت کامل", "cost": 4.8, "protection": 95, "income": 0},
    "Collar (تقریباً رایگان)": {"desc": "فروش Call + خرید Put → هزینه ≈ ۰", "cost": 0.4, "protection": 80, "income": 0},
    "Covered Call": {"desc": "درآمد ماهانه از فروش Call", "cost": -3.2, "protection": 25, "income": 3.2},
    "Cash-Secured Put": {"desc": "فروش Put → خرید ارزان‌تر در ریزش", "cost": -2.9, "protection": 40, "income": 2.9},
    "Tail-Risk Put": {"desc": "Put خیلی دور برای فاجعه (کرونا/جنگ)", "cost": 2.1, "protection": 99, "income": 0},
    "Iron Condor": {"desc": "درآمد بالا از ثبات بازار", "cost": -5.5, "protection": 15, "income": 5.5},
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
    regime = get_market_regime()

    # استراتژی هج انتخابی
    hedge = st.session_state.hedge_strategy
    opt = st.session_state.option_strategy
    info_hedge = hedge_strategies[hedge]
    info_opt = option_strategies[opt]

    # سوئیچ خودکار در بحران
    if hedge == "Regime-Switching" and regime["crisis"]:
        hedge = "Tail-Risk Active"
        st.error("بحران تشخیص داده شد! استراتژی به **Tail-Risk Active** تغییر کرد!")

    info_hedge = hedge_strategies[hedge]

    # محدودیت‌ها
    bounds = []
    for name in asset_names:
        low = 0.0
        up = 1.0
        n = name.upper()
        if any(x in n for x in ["GC=", "GOLD", "SI="]): low = info_hedge["gold_min"]
        if any(x in n for x in ["USD", "USDIRR", "USDT"]): low = info_hedge["usd_min"]
        if any(x in n for x in ["BTC", "بیت"]): up = min(info_hedge["btc_max"], st.session_state.max_btc / 100)
        bounds.append((low, up))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.ones(len(asset_names)) / len(asset_names)

    # بهینه‌سازی
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

    # نمایش نتایج
    st.success(f"استراتژی هجینگ: **{hedge}** | آپشن: **{opt}**")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("بازده سالانه", f"{ret:.2f}%")
    col2.metric("ریسک سالانه", f"{risk:.2f}%")
    col3.metric("نسبت شارپ", f"{sharpe:.3f}")
    col4.metric("زمان ریکاوری", recovery)

    if info_opt["cost"] > 0:
        st.warning(f"هزینه آپشن سالانه ≈ {info_opt['cost']:.1f}% از پرتفوی")
    elif info_opt["cost"] < 0:
        st.success(f"درآمد سالانه از آپشن ≈ {abs(info_opt['income']):.1f}%")

    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)}).sort_values("وزن (%)", ascending=False)
    st.markdown("### تخصیص دارایی")
    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی", color_discrete_sequence=px.colors.sequential.Plasma), use_container_width=True)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate – هجینگ + آپشن حرفه‌ای", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Ultimate</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>اولین اپلیکیشن ایرانی با هجینگ و آپشن واقعی</h3>", unsafe_allow_html=True)

# پیش‌فرض‌ها
defaults = {"rf_rate": 18.0, "max_btc": 25, "hedge_strategy": "طلا + تتر (ترکیبی)", "option_strategy": "Collar (تقریباً رایگان)"}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# سایدبار
st.sidebar.header("دانلود داده")
tickers = st.sidebar.text_input("نمادها", value="BTC-USD, GC=F, USDIRR=X, ^GSPC, VXX")
if st.sidebar.button("دانلود داده‌ها", type="primary"):
    with st.spinner("در حال دانلود..."):
        data = download_data(tickers)
        if data is not None:
            st.session_state.prices = data
            st.rerun()

st.sidebar.header("تنظیمات")
st.session_state.rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)
st.session_state.max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, st.session_state.max_btc)

# جدول مقایسه استراتژی‌ها
st.sidebar.markdown("### مقایسه استراتژی‌های هجینگ")
hedge_df = pd.DataFrame([
    [k, v["desc"]] for k, v in hedge_strategies.items()
], columns=["استراتژی", "توضیح"])
st.sidebar.dataframe(hedge_df, use_container_width=True)

st.session_state.hedge_strategy = st.sidebar.selectbox("استراتژی هجینگ", options=list(hedge_strategies.keys()))

st.sidebar.markdown("### مقایسه استراتژی‌های آپشن")
opt_df = pd.DataFrame([
    [k, v["desc"], f"{v['cost']:+.1f}%", f"{v['protection']}٪"] for k, v in option_strategies.items()
], columns=["استراتژی", "توضیح", "هزینه/درآمد سالانه", "سطح پوشش"])
st.sidebar.dataframe(opt_df, use_container_width=True)

st.session_state.option_strategy = st.sidebar.selectbox("استراتژی آپشن", options=list(option_strategies.keys()))

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Ultimate – ساخته شده با عشق برای سرمایه‌گذار ایرانی | نسخه نهایی ۱۴۰۴–۲۰۲۵ | هجینگ + آپشن واقعی")
