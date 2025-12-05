import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf
import warnings
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ==================== دریافت قیمت زنده دلار و طلای ۱۸ عیار از tgju.org ====================
@st.cache_data(ttl=300)
def get_iran_prices():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        # دلار آزاد
        r = requests.get("https://www.tgju.org/profile/price_dollar_rl", headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        tag = soup.find("span", class_="info-price")
        dollar_toman = float(tag.text.replace(",", "").strip()) if tag else 119700
        usd_irr = dollar_toman * 10

        # طلای ۱۸ عیار (هر گرم)
        r = requests.get("https://www.tgju.org/profile/geram18", headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        tag = soup.find("span", class_="info-price")
        gold_toman = float(tag.text.replace(",", "").strip()) if tag else 122480
        gold_irr = gold_toman * 10

        return usd_irr, gold_irr

    except:
        try:
            usd_irr = yf.Ticker("USDIRR=X").history(period="1d")["Close"].iloc[-1]
            gold_world = yf.Ticker("GC=F").history(period="1d")["Close"].iloc[-1]
            gold_18k = gold_world * usd_irr * 0.75 / 31.1035 * 10
            return usd_irr, gold_18k
        except:
            return 1197000, 12248000

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

# ==================== جدول کامل استراتژی‌های هجینگ ====================
hedge_data = {
    "استراتژی": ["طلا + تتر (ترکیبی)", "Regime-Switching", "Barbell (نسیم طالب)", "Tail-Risk Active", "حداقل هج", "بدون هجینگ"],
    "مناسب برای": [
        "سرمایه‌گذار معمولی ایرانی که می‌خواهد از تورم و سقوط ریال در امان باشد اما همچنان رشد داشته باشد. دلیل: تعادل کامل بین حفاظت (طلا/تتر) و سود (بیت‌کوین محدود).",
        "بازارهای پرنوسان با شوک‌های ناگهانی (مثل جنگ, تحریم جدید, انتخابات آمریکا). دلیل: خودکار سوئیچ به حفاظت حداکثری در بحران, بدون از دست دادن فرصت در حالت عادی.",
        "وقتی نمی‌دانید دنیا کجا می‌رود: ۸۵٪ در دارایی‌های کاملاً ایمن + ۱۵٪ در دارایی‌های خیلی پرریسک. دلیل: زنده ماندن تضمینی + شانس سود انفجاری (ایده نسیم طالب).",
        "سناریوهای آخرالزمانی: جنگ جهانی, فروپاشی دلار, کرونا جدید. دلیل: حداکثر حفاظت (۷۰٪+ ایمن) حتی اگر بقیه دنیا نابود شود – برای خواب راحت.",
        "سرمایه‌گذار ریسک‌پذیر که فقط می‌خواهد یک لایه حفاظتی کوچک داشته باشد. دلیل: آزادی عمل بالا در بازار صعودی, بدون هزینه زیاد حفاظت.",
        "سرمایه‌گذار حرفه‌ای با اعتماد کامل به بازار صعودی. دلیل: هیچ محدودیتی ندارد – فقط بهینه‌سازی ریاضی خالص برای حداکثر سود."
    ],
    "حداقل طلا (%)": [15, 30, 40, 35, 10, 0],
    "حداقل تتر/دلار (%)": [10, 20, 45, 35, 0, 0],
    "حداکثر بیت‌کوین (%)": [20, 10, 15, 5, 40, 100]
}

hedge_strategies = {
    "طلا + تتر (ترکیبی)": {"gold_min": 0.15, "usd_min": 0.10, "btc_max": 0.20},
    "Regime-Switching": {"gold_min": 0.30, "usd_min": 0.20, "btc_max": 0.10},
    "Barbell (نسیم طالب)": {"gold_min": 0.40, "usd_min": 0.45, "btc_max": 0.15},
    "Tail-Risk Active": {"gold_min": 0.35, "usd_min": 0.35, "btc_max": 0.05},
    "حداقل هج": {"gold_min": 0.10, "usd_min": 0.00, "btc_max": 0.40},
    "بدون هجینگ": {"gold_min": 0.00, "usd_min": 0.00, "btc_max": 1.00}
}

# ==================== جدول کامل استراتژی‌های آپشن ====================
option_data = {
    "استراتژی": [
        "بدون آپشن",
        "Protective Put",
        "Collar (تقریباً رایگان)",
        "Covered Call",
        "Cash-Secured Put",
        "Tail-Risk Put",
        "Iron Condor"
    ],
    "توضیح کامل": [
        "بدون هیچ پوشش آپشنی – ساده و بدون هزینه اضافی. دلیل: مناسب کسانی که ریسک‌پذیر هستند و نیازی به حفاظت اضافی نمی‌بینند.",
        "مثل بیمه کامل پرتفوی: حتی اگر بازار ۵۰٪ ریخت, ضرر شما محدود است. دلیل: خواب راحت در برابر سقوط‌های شدید (مثل ۲۰۲۰), اما هزینه سالانه دارد.",
        "هج تقریباً رایگان: با فروش Call سود بالقوه را محدود می‌کنید تا Put مجانی شود. دلیل: حفاظت خوب بدون هزینه زیاد – تعادل ایده‌آل برای بازارهای متوسط.",
        "درآمد ماهانه ثابت: Call روی دارایی‌های خودتان می‌فروشید و پول می‌گیرید. دلیل: سود اضافی در بازار رنج یا صعودی ملایم.",
        "اگر بازار ریخت, با تخفیف می‌خرید: Put می‌فروشید و پول می‌گیرید. دلیل: فرصت خرید ارزان + درآمد, مناسب وقتی به کف قیمت اعتقاد دارید.",
        "برای فاجعه‌های بزرگ (کرونا, جنگ): Put خیلی دور می‌خرید. دلیل: حفاظت کامل در برابر بلک سوان با هزینه کم – برای بحران‌های نادر.",
        "درآمد بالا از ثبات: وقتی بازار رنج می‌زند, پول زیادی از فروش آپشن می‌گیرید. دلیل: سود حداکثری در بازارهای آرام, اما حفاظت کم."
    ],
    "هزینه سالانه (%)": ["۰٪", "+۴.۸٪", "+۰.۴٪", "−۳.۲٪ (درآمد)", "−۲.۹٪ (درآمد)", "+۲.۱٪", "−۵.۵٪ (درآمد)"],
    "سطح حفاظت از سقوط (%)": ["۰", "۹۵", "۸۰", "۲۵", "۴۰", "۹۹", "۱۵"]
}

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

    st.success(f"استراتژی انتخابی: **{hedge}** + **{opt}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("بازده سالانه", f"{ret:.2f}%")
    c2.metric("ریسک سالانه", f"{risk:.2f}%")
    c3.metric("نسبت شارپ", f"{sharpe:.3f}")
    c4.metric("زمان ریکاوری", recovery)

    if info_opt["cost_pct"] > 0:
        st.warning(f"هزینه آپشن سالانه ≈ {info_opt['cost_pct']:.1f}% از پرتفوی")
    elif info_opt["cost_pct"] < 0:
        st.success(f"درآمد سالانه از آپشن ≈ {abs(info_opt['cost_pct']):.1f}%")

    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)}).sort_values("وزن (%)", ascending=False)
    st.markdown("### تخصیص دارایی")
    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی", color_discrete_sequence=px.colors.sequential.Plasma), use_container_width=True)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate – ایران ۱۴۰۴", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Ultimate</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>اولین اپ حرفه‌ای سرمایه‌گذاری ایرانی با قیمت زنده</h3>", unsafe_allow_html=True)

# قیمت زنده
usd_irr, gold_gram = get_iran_prices()
st.sidebar.success(f"دلار آزاد (زنده): **{usd_irr:,.0f}** ریال")
st.sidebar.success(f"طلای ۱۸ عیار (زنده): **{gold_gram:,.0f}** ریال/گرم")

# پیش‌فرض‌ها
defaults = {"rf_rate": 18.0, "max_btc": 25, "hedge_strategy": "طلا + تتر (ترکیبی)", "option_strategy": "Collar (تقریباً رایگان)"}
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
st.session_state.max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, st.session_state.max_btc)

# جدول هجینگ
st.sidebar.markdown("### راهنمای کامل استراتژی‌های هجینگ")
hedge_df = pd.DataFrame(hedge_data)
st.sidebar.dataframe(hedge_df, use_container_width=True, hide_index=True)
st.session_state.hedge_strategy = st.sidebar.selectbox("انتخاب استراتژی هجینگ", options=hedge_data["استراتژی"])

# جدول آپشن
st.sidebar.markdown("### راهنمای کامل استراتژی‌های آپشن")
option_df = pd.DataFrame(option_data)
st.sidebar.dataframe(option_df, use_container_width=True, hide_index=True)
st.session_state.option_strategy = st.sidebar.selectbox("انتخاب استراتژی آپشن", options=option_data["استراتژی"])

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Ultimate – نسخه نهایی ۱۴۰۴ | با عشق برای ایران")
