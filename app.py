import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import yfinance as yf
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ==================== دانلود داده ====================
@st.cache_data(show_spinner=False)
def download_data(tickers_str, period="max"):
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
    """محاسبه زمان بازیابی از افت سرمایه (Drawdown Recovery Time)"""
    if len(ret_series) == 0:
        return 0
    
    cum = (1 + ret_series).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1  # drawdown درصدی
    
    recoveries = []
    in_dd = False
    start = None
    
    for i in range(1, len(cum)):
        # شروع دوره افت (drawdown)
        if dd.iloc[i] < -0.01:  # افت بیشتر از 1%
            if not in_dd:
                in_dd = True
                start = i
        # خروج از دوره افت و بازیابی
        elif in_dd and dd.iloc[i] >= -0.001:  # بازیابی به نزدیکی قله
            in_dd = False
            if start is not None:
                recoveries.append(i - start)
    
    return np.mean(recoveries) if recoveries else 0

def format_recovery(days):
    """تبدیل روزها به فرمت سال و ماه فارسی"""
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

# ==================== تخصیص سرمایه ====================
def allocate_capital(weights, assets, total_usd):
    """محاسبه تخصیص سرمایه به دلار، تومان و ریال"""
    rate_toman = 200_000_000 / 1200  # نرخ تبدیل دلار به تومان
    
    df = pd.DataFrame({
        "دارایی": assets,
        "وزن (%)": np.round(weights * 100, 2),
        "دلار ($)": np.round(weights * total_usd, 2),
        "تومان": np.round(weights * total_usd * rate_toman, 0),
        "ریال": np.round(weights * total_usd * rate_toman * 10, 0)
    })
    
    return df.sort_values("وزن (%)", ascending=False)

# ==================== پیش‌بینی قیمت (Monte Carlo) ====================
def forecast_price_series(price_series, days=63, sims=400):
    """شبیه‌سازی مسیرهای قیمتی با روش Monte Carlo"""
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    mu = log_ret.mean()  # میانگین بازدهی لگاریتمی
    sigma = log_ret.std()  # انحراف معیار بازدهی
    last_price = price_series.iloc[-1]

    # شبیه‌سازی مسیرهای قیمتی
    paths = np.zeros((days, sims))
    
    for i in range(sims):
        prices = [last_price]
        for _ in range(days):
            # فرمول حرکت براونی هندسی
            prices.append(prices[-1] * np.exp(mu + sigma * np.random.normal()))
        paths[:, i] = prices[1:]
    
    return paths

def plot_forecast(prices, asset):
    """رسم نمودار پیش‌بینی قیمت دارایی"""
    series = prices[asset]
    ma150 = series.rolling(150).mean()  # میانگین متحرک 150 روزه

    # محاسبه شبیه‌سازی‌های قیمتی
    paths = forecast_price_series(series, 63)

    fig = go.Figure()
    
    # قیمت واقعی
    fig.add_trace(go.Scatter(y=series, name="قیمت واقعی", mode="lines"))
    
    # میانگین متحرک
    fig.add_trace(go.Scatter(y=ma150, name="MA 150", 
                            line=dict(dash="dash"), mode="lines"))
    
    # پیش‌بینی نرمال (50 درصدیل)
    fig.add_trace(go.Scatter(
        y=np.percentile(paths, 50, axis=1),
        name="پیش‌بینی نرمال (۳ ماه)",
        line=dict(color="orange"),
        mode="lines"
    ))
    
    # سناریوی خوش‌بینانه (85 درصدیل)
    fig.add_trace(go.Scatter(
        y=np.percentile(paths, 85, axis=1),
        name="سناریوی خوش‌بینانه",
        line=dict(dash="dot", color="green"),
        mode="lines"
    ))
    
    # سناریوی بدبینانه (15 درصدیل)
    fig.add_trace(go.Scatter(
        y=np.percentile(paths, 15, axis=1),
        name="سناریوی بدبینانه",
        line=dict(dash="dot", color="red"),
        mode="lines"
    ))

    fig.update_layout(
        title=f"پیش‌بینی قیمت {asset}",
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# ==================== محاسبه پرتفوی ====================
@st.fragment
def calculate_portfolio():
    """محاسبه و نمایش پرتفوی سرمایه‌گذاری"""
    if "prices" not in st.session_state:
        st.info("لطفاً داده‌ها را دانلود کنید.")
        return
    
    prices = st.session_state.prices
    returns = prices.pct_change().dropna()
    mean_ret = returns.mean() * 252  # بازدهی سالانه
    cov_mat = returns.cov() * 252    # ماتریس کوواریانس سالانه
    rf = st.session_state.rf_rate / 100  # نرخ بدون ریسک
    n = len(mean_ret)

    # وزن‌های برابر (Equal Weight)
    weights = np.ones(n) / n

    # نمایش نتایج
    st.success("✅ پرتفوی محاسبه شد")
    
    # جدول وزن‌ها
    df_w = pd.DataFrame({
        "دارایی": prices.columns,
        "وزن (%)": np.round(weights * 100, 2)
    })
    st.dataframe(df_w, use_container_width=True)
    
    # نمودار دایره‌ای
    st.plotly_chart(
        px.pie(df_w, values="وزن (%)", names="دارایی", 
               title="توزیع سرمایه در پرتفوی"),
        use_container_width=True
    )

    # ==================== تخصیص سرمایه ====================
    st.markdown("### 💰 تخصیص سرمایه")
    capital = st.number_input(
        "کل سرمایه (دلار)",
        min_value=100,
        max_value=10_000_000,
        value=1200,
        step=100
    )
    
    alloc = allocate_capital(weights, prices.columns, capital)
    st.dataframe(alloc, use_container_width=True)

    # ==================== پیش‌بینی ====================
    st.markdown("### 🔮 پیش‌بینی قیمت دارایی‌ها")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        asset = st.selectbox("انتخاب دارایی", prices.columns)
    
    with col2:
        st.write("")  # فاصله
    
    # نمایش نمودار پیش‌بینی
    st.plotly_chart(plot_forecast(prices, asset), use_container_width=True)
    
    # آمار اضافی
    st.markdown("#### 📊 آمار دارایی")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = prices[asset].iloc[-1]
        st.metric("قیمت فعلی", f"${current_price:.2f}")
    
    with col2:
        annual_return = returns[asset].mean() * 252 * 100
        st.metric("بازدهی سالانه", f"{annual_return:.2f}%")
    
    with col3:
        annual_volatility = returns[asset].std() * np.sqrt(252) * 100
        st.metric("نوسان‌پذیری سالانه", f"{annual_volatility:.2f}%")
    
    with col4:
        sharpe_ratio = (annual_return/100 - rf) / (annual_volatility/100)
        st.metric("نسبت شارپ", f"{sharpe_ratio:.2f}")


# ==================== UI ====================
st.set_page_config(
    page_title="Portfolio360 Ultimate Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# عنوان اصلی
st.markdown(
    "<h1 style='text-align:center;color:#00d2d3;font-size:3em'>💼 Portfolio360 Ultimate Pro</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:#999;font-size:1.1em'>سیستم تحلیل و پیش‌بینی پرتفوی سرمایه‌گذاری</p>",
    unsafe_allow_html=True
)

# درج خط جداکننده
st.divider()

# منوی کناری
with st.sidebar:
    st.header("📥 دانلود داده")
    tickers = st.text_input(
        "نمادهای دارایی (با کاما جدا کنید)",
        "BTC-USD, GC=F, ETH-USD",
        help="مثال: BTC-USD (بیتکوین), GC=F (طلا), ETH-USD (اتریوم)"
    )
    
    if st.button("🔄 دانلود داده", use_container_width=True):
        with st.spinner("در حال دانلود..."):
            st.session_state.prices = download_data(tickers)
            st.rerun()

    st.markdown("---")
    
    st.header("⚙️ تنظیمات")
    st.session_state.rf_rate = st.number_input(
        "نرخ بدون ریسک (%) - سالانه",
        min_value=0.0,
        max_value=50.0,
        value=18.0,
        step=0.1,
        help="نرخ بهره تضمین‌شده بدون ریسک"
    )

    st.markdown("---")
    
    # اطلاعات درباره
    with st.expander("ℹ️ درباره برنامه"):
        st.write("""
        **Portfolio360 Ultimate Pro** یک ابزار قدرتمند برای:
        - 📊 تحلیل پرتفوی سرمایه‌گذاری
        - 🔮 پیش‌بینی قیمت‌های دارایی‌ها
        - 💰 تخصیص بهینه سرمایه
        - 📈 محاسبه شاخص‌های مالی
        """)

# محاسبه و نمایش پرتفوی
calculate_portfolio()

# پاورقی
st.divider()
st.caption(
    "🔐 Portfolio360 Ultimate Pro | Powered by Streamlit & Plotly | "
    f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
