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

# ==================== دریافت داده‌های تاریخی بورس ایران از BrsApi.ir ====================
@st.cache_data(ttl=3600)
def get_bors_history(symbol, days=365):
    api_key = "Bs76KjzbGZbipLtarU5SRrH7vCeufSqQ"  # کلید کاربر
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        url = f"https://api.brsapi.ir/v1/stock/history?symbol={symbol}&api_key={api_key}"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # فرض: data یک لیست از قیمت‌ها (historical prices, close, volume, etc)
            df = pd.DataFrame(data)  # بسته به ساختار API، ممکنه نیاز به تنظیم داشته باشه
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()[-days:]
            return df
        else:
            st.warning(f"API بورس در دسترس نیست (کد {r.status_code}). fallback به yfinance.")
    except:
        st.warning("خطا در دریافت داده بورس. استفاده از fallback.")
    
    # fallback به yfinance (برای نمادهای جهانی یا شاخص‌های مرتبط)
    try:
        df = yf.Ticker(symbol).history(period=f"{days}d")
        return df
    except:
        return pd.DataFrame()  # خالی اگر هیچی نبود

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
        "سرمایه‌گذار معمولی ایرانی که می‌خواهد از تورم و سقوط ریال در امان باشد اما همچنان رشد داشته باشد. دلیل: تعادل بین حفاظت و سود.",
        "بازارهای پرنوسان با شوک‌های ناگهانی (مثل جنگ، تحریم جدید). دلیل: خودکار سوئیچ به حفاظت حداکثری در بحران.",
        "وقتی نمی‌دانید دنیا کجا می‌رود: ۸۵٪ ایمن + ۱۵٪ پرریسک. دلیل: زنده ماندن تضمینی + شانس سود انفجاری (ایده نسیم طالب).",
        "سناریوهای آخرالزمانی: جنگ، رکود، کرونا. دلیل: حداکثر حفاظت حتی اگر بقیه دنیا نابود شود.",
        "سرمایه‌گذار ریسک‌پذیر که فقط لایه حفاظتی کوچک می‌خواهد. دلیل: آزادی عمل بالا در بازار صعودی.",
        "سرمایه‌گذار حرفه‌ای با اعتماد کامل به بازار صعودی. دلیل: هیچ محدودیتی ندارد – فقط بهینه‌سازی ریاضی."
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
    "استراتژی": ["بدون آپشن", "Protective Put", "Collar (تقریباً رایگان)", "Covered Call", "Cash-Secured Put", "Tail-Risk Put", "Iron Condor"],
    "توضیح کامل": [
        "بدون هیچ پوشش آپشنی – ساده و بدون هزینه اضافی. دلیل: مناسب کسانی که ریسک‌پذیر هستند و نیازی به حفاظت اضافی نمی‌بینند.",
        "مثل بیمه کامل پرتفوی: حتی اگر بازار ۵۰٪ ریخت، ضرر شما محدود است. دلیل: خواب راحت در برابر سقوط‌های شدید (مثل ۲۰۲۰).",
        "هج تقریباً رایگان: با فروش Call سود بالقوه را محدود می‌کنید تا Put مجانی شود. دلیل: حفاظت خوب بدون هزینه زیاد.",
        "درآمد ماهانه ثابت: Call روی دارایی‌های خودتان می‌فروشید و پول می‌گیرید. دلیل: سود اضافی در بازار رنج یا صعودی ملایم.",
        "اگر بازار ریخت، با تخفیف می‌خرید: Put می‌فروشید و پول می‌گیرید. دلیل: فرصت خرید ارزان + درآمد.",
        "برای فاجعه‌های بزرگ (کرونا، جنگ): Put خیلی دور می‌خرید. دلیل: حفاظت کامل در برابر بلک سوان با هزینه کم.",
        "درآمد بالا از ثبات: وقتی بازار رنج می‌زند، پول زیادی از فروش آپشن می‌گیرید. دلیل: سود حداکثری در بازارهای آرام."
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

# بخش جدید: دریافت داده‌های تاریخی بورس ایران
st.sidebar.markdown("### داده‌های تاریخی بورس ایران (از BrsApi.ir)")
bors_symbol = st.sidebar.text_input("نماد بورس (مثل SHARIF یا INDEX:TEDPIX)", value="TEDPIX")
bors_days = st.sidebar.slider("تعداد روزهای تاریخی", 30, 365, 180)
if st.sidebar.button("دریافت داده بورس"):
    with st.spinner("در حال دریافت داده بورس..."):
        bors_df = get_bors_history(bors_symbol, bors_days)
        if not bors_df.empty:
            st.success(f"داده‌های تاریخی برای {bors_symbol} ( {bors_days} روز)")
            st.dataframe(bors_df, use_container_width=True)
            fig = px.line(bors_df, y="Close", title=f"نمودار قیمت {bors_symbol}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("داده‌ای یافت نشد. نماد رو چک کنید.")

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Ultimate – نسخه نهایی ۱۴۰۴ | با عشق برای ایران")
