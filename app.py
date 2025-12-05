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
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ==================== دریافت قیمت‌های آنلاین ====================
@st.cache_data(ttl=300)  # هر 5 دقیقه بروزرسانی
def get_iran_prices():
    try:
        # قیمت دلار از tgju.org
        url_dollar = "https://www.tgju.org/profile/price_dollar_rl"
        response = requests.get(url_dollar)
        soup = BeautifulSoup(response.text, 'html.parser')
        dollar_text = soup.find('span', class_='info-price').text.replace(',', '').strip() if soup.find('span', class_='info-price') else "1197000"
        usd_irr = float(dollar_text) / 10  # تومان به ریال (تقریبی)

        # قیمت طلا 18 عیار از tgju.org (هر گرم)
        url_gold = "https://www.tgju.org/profile/geram18"
        response_gold = requests.get(url_gold)
        soup_gold = BeautifulSoup(response_gold.text, 'html.parser')
        gold_text = soup_gold.find('span', class_='info-price').text.replace(',', '').strip() if soup_gold.find('span', class_='info-price') else "1224800"
        gold_price = float(gold_text) / 10  # تومان به ریال (تقریبی)

        return usd_irr, gold_price
    except:
        # fallback به yfinance
        usd = yf.Ticker("USDIRR=X").history(period="1d")["Close"].iloc[-1]
        gold = yf.Ticker("GC=F").history(period="1d")["Close"].iloc[-1] * 31.1035 * 0.75  # هر اونس به گرم 18 عیار
        return usd, gold

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

# ==================== تشخیص رژیم بازار ====================
@st.cache_data(ttl=3600)
def get_market_regime(usd_irr):
    try:
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        usd_change = 0.20  # تقریبی
        return {"crisis": vix > 35 or usd_change > 0.20, "high_vol": vix > 25, "vix": vix}
    except:
        return {"crisis": True, "high_vol": True, "vix": 40}

# ==================== استراتژی‌های هجینگ با توضیحات ====================
hedge_strategies = {
    "طلا + تتر (ترکیبی)": {
        "desc": "۱۵٪ طلا + ۱۰٪ تتر/دلار – بهترین دفاع در شرایط فعلی ایران (تورم بالا + سقوط تدریجی ریال). مناسب برای سرمایه‌گذارانی که می‌خواهند تعادل بین حفاظت و بازده داشته باشند.",
        "gold_min": 0.15, "usd_min": 0.10, "btc_max": 0.20
    },
    "Regime-Switching": {
        "desc": "در بحران (VIX>35 یا سقوط شدید ریال) → حداکثر حفاظت با ۵۰٪+ هج | در حالت عادی → پرتفوی آزاد. مناسب برای بازارهای ناپایدار با شوک‌های ناگهانی.",
        "gold_min": 0.30, "usd_min": 0.20, "btc_max": 0.10
    },
    "Barbell (نسیم طالب)": {
        "desc": "۸۵٪ در دارایی‌های ایمن (تتر + طلا) + ۱۵٪ در دارایی‌های خیلی پرریسک (بیت‌کوین). مناسب وقتی نمی‌دانید بازار کجا می‌رود اما می‌خواهید هم زنده بمانید هم شانس سود بزرگ داشته باشید.",
        "gold_min": 0.40, "usd_min": 0.45, "btc_max": 0.15
    },
    "Tail-Risk Active": {
        "desc": "حالت وحشت جهانی با حداقل ۳۵٪ طلا + تتر + پوشش آپشن Put عمیق. مناسب برای سناریوهای فاجعه‌بار مثل جنگ یا رکود جهانی.",
        "gold_min": 0.35, "usd_min": 0.35, "btc_max": 0.05
    },
    "حداقل هج": {
        "desc": "فقط ۱۰٪ طلا – برای کسانی که ریسک‌پذیر هستند اما نمی‌خواهند کاملاً بی‌حفاظت باشند. مناسب بازار صعودی پایدار.",
        "gold_min": 0.10, "usd_min": 0.00, "btc_max": 0.40
    },
    "بدون هجینگ": {
        "desc": "کاملاً آزاد – فقط بر اساس بهینه‌سازی ریاضی. مناسب بازار گاوی قوی با اعتماد کامل به روند صعودی.",
        "gold_min": 0.00, "usd_min": 0.00, "btc_max": 1.00
    }
}

# ==================== استراتژی‌های آپشن با هزینه دلاری ====================
option_strategies = {
    "بدون آپشن": {"desc": "بدون پوشش آپشن – هیچ هزینه‌ای ندارد.", "cost_pct": 0.0, "cost_usd_per_contract": 0, "protection": 0, "income": 0},
    "Protective Put": {"desc": "خرید Put برای محافظت کامل از سقوط شدید (مثل ۲۰٪ ریزش). هزینه تقریبی: ۴.۸٪ از ارزش پرتفوی در سال.", "cost_pct": 4.8, "cost_usd_per_contract": 480, "protection": 95, "income": 0},
    "Collar (تقریباً رایگان)": {"desc": "فروش Call + خرید Put → هزینه نزدیک به صفر با محدود کردن سود بالقوه. مناسب هج کم‌هزینه.", "cost_pct": 0.4, "cost_usd_per_contract": 40, "protection": 80, "income": 0},
    "Covered Call": {"desc": "فروش Call روی دارایی‌هایی که دارید → درآمد ماهانه اما سقف سود. درآمد تقریبی: ۳.۲٪ در سال.", "cost_pct": -3.2, "cost_usd_per_contract": -320, "protection": 25, "income": 3.2},
    "Cash-Secured Put": {"desc": "فروش Put نقدی → اگر بازار ریخت، با تخفیف می‌خرید. درآمد تقریبی: ۲.۹٪ در سال.", "cost_pct": -2.9, "cost_usd_per_contract": -290, "protection": 40, "income": 2.9},
    "Tail-Risk Put": {"desc": "Put خیلی دور (۵۰٪ پایین‌تر) برای فاجعه‌ها مثل کرونا. هزینه کم اما حفاظت بالا.", "cost_pct": 2.1, "cost_usd_per_contract": 210, "protection": 99, "income": 0},
    "Iron Condor": {"desc": "درآمد بالا از ثبات بازار (رنج زدن قیمت). درآمد تقریبی: ۵.۵٪ در سال اما ریسک محدود.", "cost_pct": -5.5, "cost_usd_per_contract": -550, "protection": 15, "income": 5.5},
}

# ==================== محاسبه پرتفوی ====================
@st.fragment
def calculate_portfolio(usd_irr, gold_price):
    if "prices" not in st.session_state or st.session_state.prices is None:
        st.info("لطفاً داده‌ها را دانلود کنید.")
        return

    prices = st.session_state.prices
    returns = prices.pct_change().dropna()
    asset_names = list(prices.columns)
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    rf = st.session_state.rf_rate / 100
    regime = get_market_regime(usd_irr)

    # استراتژی انتخابی
    hedge = st.session_state.hedge_strategy
    opt = st.session_state.option_strategy
    info_hedge = hedge_strategies[hedge]
    info_opt = option_strategies[opt]

    # سوئیچ خودکار در بحران
    if hedge == "Regime-Switching" and regime["crisis"]:
        hedge = "Tail-Risk Active"
        st.error("بحران تشخیص داده شد! استراتژی به Tail-Risk Active تغییر کرد!")

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

    # بهینه‌سازی مارکوویتز
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
    st.success(f"هجینگ: **{hedge}** | آپشن: **{opt}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("بازده سالانه", f"{ret:.2f}%")
    c2.metric("ریسک سالانه", f"{risk:.2f}%")
    c3.metric("نسبت شارپ", f"{sharpe:.3f}")
    c4.metric("زمان ریکاوری", recovery)

    # هزینه آپشن با USD
    if info_opt["cost_pct"] > 0:
        cost_rial = info_opt["cost_pct"] / 100 * 1000000  # فرض 1M تومان پرتفوی
        cost_usd = cost_rial / usd_irr
        st.warning(f"هزینه آپشن سالانه ≈ {info_opt['cost_pct']:.1f}% ({cost_usd:.0f} USD)")
    elif info_opt["cost_pct"] < 0:
        income_rial = abs(info_opt["income"]) / 100 * 1000000
        income_usd = income_rial / usd_irr
        st.success(f"درآمد سالانه از آپشن ≈ {abs(info_opt['income']):.1f}% ({income_usd:.0f} USD)")

    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)}).sort_values("وزن (%)", ascending=False)
    st.markdown("### تخصیص دارایی")
    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی", color_discrete_sequence=px.colors.sequential.Plasma), use_container_width=True)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate – ایران ۱۴۰۴", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Ultimate</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>اولین اپلیکیشن ایرانی با هجینگ + آپشن واقعی</h3>", unsafe_allow_html=True)

# دریافت قیمت‌های آنلاین
usd_irr, gold_price = get_iran_prices()
st.sidebar.metric("قیمت دلار (ریال)", f"{usd_irr:,.0f}")
st.sidebar.metric("قیمت طلا ۱۸ عیار (ریال/گرم)", f"{gold_price:,.0f}")

# پیش‌فرض‌ها
defaults = {
    "rf_rate": 18.0,
    "max_btc": 25,
    "hedge_strategy": "طلا + تتر (ترکیبی)",
    "option_strategy": "Collar (تقریباً رایگان)"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# سایدبار
st.sidebar.header("دانلود داده")
tickers = st.sidebar.text_input("نمادها (با کاما)", value="BTC-USD, GC=F, USDIRR=X, ^GSPC")
if st.sidebar.button("دانلود داده‌ها", type="primary"):
    with st.spinner("در حال دانلود..."):
        data = download_data(tickers)
        if data is not None:
            st.session_state.prices = data
            st.rerun()

st.sidebar.header("تنظیمات")
st.session_state.rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)
st.session_state.max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, st.session_state.max_btc)

# جدول هجینگ با توضیحات
st.sidebar.markdown("### استراتژی هجینگ")
hedge_df = pd.DataFrame([
    [k, v["desc"]] for k, v in hedge_strategies.items()
], columns=["استراتژی", "توضیح کامل"])
st.sidebar.dataframe(hedge_df, use_container_width=True)
st.session_state.hedge_strategy = st.sidebar.selectbox("انتخاب هجینگ", options=list(hedge_strategies.keys()))

# جدول آپشن با توضیحات و هزینه
st.sidebar.markdown("### استراتژی آپشن")
opt_df = pd.DataFrame([
    [k, v["desc"], f"{v['cost_usd_per_contract']:+.0f} USD", f"{v['protection']}٪"] for k, v in option_strategies.items()
], columns=["استراتژی", "توضیح کامل", "هزینه تقریبی (USD/قرارداد)", "سطح پوشش"])
st.sidebar.dataframe(opt_df, use_container_width=True)
st.session_state.option_strategy = st.sidebar.selectbox("انتخاب آپشن", options=list(option_strategies.keys()))

# اجرا
calculate_portfolio(usd_irr, gold_price)

st.balloons()
st.caption("Portfolio360 Ultimate – نسخه نهایی ۱۴۰۴ | با عشق برای ایران")
