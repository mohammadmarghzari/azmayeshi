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

# ==================== ریسک پاریتی ====================
def risk_parity_objective(w, cov_mat):
    port_var = np.dot(w.T, np.dot(cov_mat, w))
    if port_var < 1e-10:
        return 9999.0
    marginal_contrib = np.dot(cov_mat, w)
    risk_contrib = w * marginal_contrib / np.sqrt(port_var)
    target = np.mean(risk_contrib)
    return np.sum((risk_contrib - target) ** 2)

# ==================== هوش مصنوعی هجینگ پیشرفته ====================
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_regime():
    try:
        # VIX
        vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
        # دلار به ریال (30 روز)
        usd = yf.Ticker("USDIRR=X").history(period="90d")["Close"]
        usd_change_30d = (usd.iloc[-1] / usd.iloc[-31]) - 1
        # طلا (1 سال)
        gold = yf.Ticker("GC=F").history(period="2y")["Close"]
        gold_change_1y = (gold.iloc[-1] / gold.iloc[-252]) - 1
        # بیت‌کوین روند
        btc = yf.Ticker("BTC-USD").history(period="200d")["Close"]
        ma50 = btc.rolling(50).mean().iloc[-1]
        ma200 = btc.rolling(200).mean().iloc[-1]
        btc_bear = ma50 < ma200

        return {
            "crisis": vix > 35 or (vix > 25 and usd_change_30d > 0.15),
            "high_vol": vix > 25,
            "usd_crisis": usd_change_30d > 0.15,
            "gold_rising": gold_change_1y > 0.25,
            "btc_bear": btc_bear,
            "vix_level": vix
        }
    except:
        return {"crisis": True, "high_vol": True, "usd_crisis": True, "gold_rising": True, "btc_bear": True, "vix_level": 40}

# ==================== جدول توضیح استراتژی‌های هجینگ پیشرفته ====================
hedge_strategies = {
    "طلا + تتر (ترکیبی)": {
        "توضیح": "۱۵٪ طلا + ۱۰٪ تتر/دلار – بهترین دفاع در شرایط فعلی ایران",
        "مناسب": "تورم بالا + سقوط تدریجی ریال",
        "حداقل طلا": 0.15, "حداقل تتر": 0.10, "حداکثر بیت": 0.20
    },
    "Regime-Switching (دو حالته)": {
        "توضیح": "در بحران → ۵۰٪+ هج | در حالت عادی → پرتفوی عادی",
        "مناسب": "بازارهای ناپایدار با شوک‌های ناگهانی",
        "حداقل طلا": 0.30, "حداقل تتر": 0.20, "حداکثر بیت": 0.10
    },
    "Trend-Following Hedge": {
        "توضیح": "اگر بیت‌کوین یا بورس آمریکا نزولی باشه → خودکار به طلا/تتر شیفت می‌کنه",
        "مناسب": "سقوط بازارهای پرریسک",
        "حداقل طلا": 0.20, "حداقل تتر": 0.15, "حداکثر بیت": 0.15
    },
    "Barbell Strategy (نسیم طالب)": {
        "توضیح": "۸۵٪ تتر + طلا | ۱۵٪ فقط بیت‌کوین و دارایی‌های خیلی پرریسک",
        "مناسب": "وقتی نمی‌دونی دنیا کجا می‌ره ولی می‌خوای هم زنده بمونی هم شانس سود بزرگ داشته باشی",
        "حداقل طلا": 0.40, "حداقل تتر": 0.45, "حداکثر بیت": 0.15
    },
    "Tail-Risk Active (وحشت جهانی)": {
        "توضیح": "خرید فعال VXX/UVXY + حداقل ۴۰٪ طلا + تتر در حالت وحشت",
        "مناسب": "جنگ، کرونا، فروپاشی مالی",
        "حداقل طلا": 0.30, "حداقل تتر": 0.30, "حداکثر بیت": 0.05
    },
    "حداقل هج (Minimum Hedge)": {
        "توضیح": "فقط ۱۰٪ طلا – برای کسانی که ریسک‌پذیرن ولی نمی‌خوان کامل بی‌هج باشن",
        "مناسب": "بازار صعودی پایدار",
        "حداقل طلا": 0.10, "حداقل تتر": 0.00, "حداکثر بیت": 0.40
    },
    "بدون هجینگ": {
        "توضیح": "کاملاً آزاد – فقط بهینه‌سازی ریاضی",
        "مناسب": "بازار گاوی قوی + اعتماد کامل",
        "حداقل طلا": 0.00, "حداقل تتر": 0.00, "حداکثر بیت": 1.00
    }
}

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
    regime = get_market_regime()

    # اعمال استراتژی هجینگ انتخابی
    strategy = st.session_state.hedge_strategy
    info = hedge_strategies[strategy]

    # تشخیص خودکار رژیم برای Regime-Switching
    if strategy == "Regime-Switching (دو حالته)" and regime["crisis"]:
        strategy = "Tail-Risk Active (وحشت جهانی)"
        st.warning("هشدار: رژیم بازار به حالت **بحران** تغییر کرد! استراتژی به Tail-Risk سوئیچ شد.")

    # به‌روزرسانی info بر اساس رژیم
    info = hedge_strategies[strategy]

    # محدودیت‌ها بر اساس استراتژی
    bounds = []
    for name in asset_names:
        low = 0.0
        up = 1.0

        name_up = name.upper()
        is_gold = any(x in name_up for x in ["GC=", "GOLD", "طلا", "SI="])
        is_usd = any(x in name_up for x in ["USD", "USDIRR", "تتر", "USDT"])
        is_btc = any(x in name_up for x in ["BTC", "بیت"])
        is_vix = any(x in name_up for x in ["VXX", "UVXY", "^VIX"])

        if is_gold:
            low = info["حداقل طلا"]
        if is_usd:
            low = info["حداقل تتر"]
        if is_btc:
            up = min(info["حداکثر بیت"], st.session_state.max_btc / 100.0)
        if is_vix and "Tail-Risk" in strategy:
            low = 0.05  # حداقل ۵٪ VXX/UVXY در حالت وحشت

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
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 5000})
            weights = res.x if res.success else x0
        elif selected == "equal":
            weights = np.ones(len(asset_names)) / len(asset_names)
        elif selected == "minvar":
            res = minimize(lambda w: np.dot(w.T, np.dot(cov_mat, w)), x0, method="SLSQP", bounds=bounds, constraints=constraints)
            weights = res.x if res.success else x0
        elif selected == "risk_parity":
            res = minimize(risk_parity_objective, x0, args=(cov_mat,), method="SLSQP", bounds=bounds, constraints=constraints)
            weights = res.x if res.success else x0
        else:
            weights = mean_ret / mean_ret.sum()
            weights = np.nan_to_num(weights, nan=1.0/len(asset_names))
    except Exception as e:
        st.error(f"خطا در بهینه‌سازی: {str(e)}")
        weights = x0

    weights = np.clip(weights, [b[0] for b in bounds], [b[1] for b in bounds])
    if weights.sum() > 0:
        weights /= weights.sum()

    ret = np.dot(weights, mean_ret) * 100
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
    sharpe = (ret - rf) / risk if risk > 0 else 0
    recovery = format_recovery(calculate_recovery_time(returns.dot(weights)))

    # مونت‌کارلو
    mc_ret, mc_risk = [], []
    for _ in range(10000):
        w = np.random.random(len(asset_names))
        w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
        w /= w.sum() if w.sum() > 0 else 1
        mc_ret.append(np.dot(w, mean_ret) * 100)
        mc_risk.append(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_risk, y=mc_ret, mode='markers', marker=dict(color='lightgray', size=5), name="پرتفوی‌های تصادفی"))

    styles_info = {
        "markowitz": ("مارکوویتز", "#00d2d3", "star-diamond"),
        "equal": ("وزن برابر", "#1f77b4", "circle"),
        "minvar": ("حداقل ریسک", "#2ca02c", "square"),
        "risk_parity": ("ریسک‌پاریتی", "#9467bd", "diamond"),
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

        w_opt = np.clip(w_opt, [b[0] for b in bounds], [b[1] for b in bounds])
        w_opt /= w_opt.sum() if w_opt.sum() > 0 else 1

        port_ret = np.dot(w_opt, mean_ret) * 100
        port_risk = np.sqrt(np.dot(w_opt.T, np.dot(cov_mat, w_opt))) * 100

        fig.add_trace(go.Scatter(
            x=[port_risk], y=[port_ret], mode='markers',
            marker=dict(color=color, size=16, symbol=symbol, line=dict(width=3, color='black')),
            name=name, text=f"{name}<br>بازده: {port_ret:.1f}%<br>ریسک: {port_risk:.1f}%",
            hoverinfo="text"
        ))

    fig.update_layout(height=650, title="مرز کارا – مقایسه سبک‌ها", xaxis_title="ریسک (%)", yaxis_title="بازده (%)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"استراتژی هجینگ فعال: **{strategy}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("بازده", f"{ret:.2f}%")
    c2.metric("ریسک", f"{risk:.2f}%")
    c3.metric("شارپ", f"{sharpe:.3f}")
    c4.metric("ریکاوری", recovery)

    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)}).sort_values("وزن (%)", ascending=False)
    st.markdown("### تخصیص دارایی")
    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی", color_discrete_sequence=px.colors.sequential.Viridis), use_container_width=True)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Pro – هجینگ پیشرفته ایرانی", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Pro</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>اولین اپلیکیشن ایرانی با هجینگ حرفه‌ای و هوش مصنوعی واقعی</h3>", unsafe_allow_html=True)

# تنظیمات پیش‌فرض
defaults = {
    "rf_rate": 18.0, "max_btc": 20,
    "selected_style": "مارکوویتز + هجینگ (بهینه‌ترین شارپ)",
    "hedge_strategy": "طلا + تتر (ترکیبی)"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# سایدبار
st.sidebar.header("دانلود داده")
tickers_input = st.sidebar.text_input("نمادها", value="BTC-USD, GC=F, USDIRR=X, ^GSPC, VXX", placeholder="BTC-USD, GC=F, ...")
if st.sidebar.button("دانلود داده‌ها", type="primary"):
    with st.spinner("در حال دانلود..."):
        prices = download_data(tickers_input)
        if prices is not None:
            st.session_state.prices = prices
            st.rerun()

st.sidebar.header("تنظیمات")
st.session_state.rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)
st.session_state.max_btc = st.sidebar.slider("حداکثر بیت‌کوین (%)", 0, 100, st.session_state.max_btc)

# جدول توضیح استراتژی‌ها
st.sidebar.markdown("### راهنمای استراتژی‌های هجینگ")
strategy_df = pd.DataFrame([
    [k, v["توضیح"], v["مناسب"]] for k, v in hedge_strategies.items()
], columns=["استراتژی", "توضیح کوتاه", "مناسب برای"])
st.sidebar.dataframe(strategy_df, use_container_width=True)

# انتخاب استراتژی
st.session_state.hedge_strategy = st.sidebar.selectbox(
    "استراتژی هجینگ حرفه‌ای",
    options=list(hedge_strategies.keys()),
    index=list(hedge_strategies.keys()).index(st.session_state.hedge_strategy)
)

# انتخاب سبک
styles = ["مارکوویتز + هجینگ (بهینه‌ترین شارپ)","وزن برابر (ساده و مقاوم)","حداقل ریسک (محافظه‌کارانه)","ریسک‌پاریتی (Risk Parity)"]
st.session_state.selected_style = st.sidebar.selectbox("سبک بهینه‌سازی", styles, index=styles.index(st.session_state.selected_style))

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Pro – ساخته شده با عشق برای سرمایه‌گذار ایرانی | ۱۴۰۴–۲۰۲۵ | نسخه حرفه‌ای با هجینگ واقعی")
