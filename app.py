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
        st.error("هیچ داده‌ای دانلود نشد.")
        return None
    prices = pd.DataFrame(data).ffill().bfill()
    if failed:
        st.sidebar.warning(f"دانلود نشد: {', '.join(failed)}")
    return prices

# ==================== توابع کمکی ====================
def portfolio_performance(weights, mean_returns, cov_matrix, rf_rate):
    returns = np.dot(weights, mean_returns) * 252 * 100
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) * 100
    sharpe = (returns - rf_rate) / std if std > 0 else 0
    return returns, std, sharpe

def negative_sharpe(weights, mean_returns, cov_matrix, rf_rate):
    returns, std, _ = portfolio_performance(weights, mean_returns, cov_matrix, rf_rate)
    return - (returns - rf_rate) / std

def max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min() * 100

def max_gain(returns):
    if len(returns) == 0:
        return 0
    annual = (1 + returns).resample('Y').last().pct_change().dropna() * 100
    return annual.max() if not annual.empty else 0

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
    "Barbell طالب (۹۰/۱۰)": {"gold_min": 0.45, "usd_min": 0.45, "btc_max": 0.10},
    "Tail-Risk طالب": {"gold_min": 0.35, "usd_min": 0.35, "btc_max": 0.05},
    "Antifragile طالب": {"gold_min": 0.40, "usd_min": 0.20, "btc_max": 0.40},
    "طلا + تتر (ترکیبی)": {"gold_min": 0.15, "usd_min": 0.10, "btc_max": 0.20},
    "حداقل هج": {"gold_min": 0.10, "usd_min": 0.00, "btc_max": 0.40},
    "بدون هجینگ (کاملاً آزاد)": {"gold_min": 0.00, "usd_min": 0.00, "btc_max": 1.00},
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

# ==================== محاسبه پرتفوی + نمودار مرز کارا ====================
@st.fragment
def calculate_portfolio():
    if "prices" not in st.session_state or st.session_state.prices is None:
        st.info("لطفاً داده‌ها را دانلود کنید.")
        return

    prices = st.session_state.prices
    returns = prices.pct_change().dropna()
    asset_names = list(prices.columns)
    mean_ret = returns.mean()
    cov_mat = returns.cov()
    rf = st.session_state.rf_rate

    # محاسبه مرز کارا
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(len(asset_names))
        weights /= np.sum(weights)
        ret, risk, sharpe = portfolio_performance(weights, mean_ret, cov_mat, rf)
        results[0,i] = risk
        results[1,i] = ret
        results[2,i] = sharpe

    # نمودار مرز کارا
    fig_ef = go.Figure()

    fig_ef.add_trace(go.Scatter(
        x=results[0,:], y=results[1,:],
        mode='markers',
        marker=dict(color=results[2,:], colorscale='Viridis', size=5, opacity=0.6),
        name='پرتفوی‌های ممکن',
        hoverinfo='skip'
    ))

    max_sharpe_idx = np.argmax(results[2])
    fig_ef.add_trace(go.Scatter(
        x=[results[0, max_sharpe_idx]], y=[results[1, max_sharpe_idx]],
        mode='markers',
        marker=dict(color='red', size=18, symbol='star'),
        name=f'بهینه‌ترین پرتفوی (شارپ: {results[2, max_sharpe_idx]:.2f})'
    ))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5']
    for idx, (name, info) in enumerate(hedge_strategies.items()):
        bounds = []
        for asset in asset_names:
            low = 0.0
            up = 1.0
            n = asset.upper()
            if any(x in n for x in ["GC=", "GOLD", "SI="]):
                low = max(low, info["gold_min"])
            if any(x in n for x in ["USD", "USDIRR", "USDT"]):
                low = max(low, info["usd_min"])
            if any(x in n for x in ["BTC", "بیت"]):
                up = min(up, info["btc_max"])
            bounds.append((low, up))

        try:
            res = minimize(negative_sharpe, x0=np.ones(len(asset_names))/len(asset_names),
                          args=(mean_ret, cov_mat, rf),
                          method='SLSQP', bounds=bounds,
                          constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}])
            if res.success:
                w = res.x
                w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
                w /= w.sum()
                ret, risk, sharpe_ratio = portfolio_performance(w, mean_ret, cov_mat, rf)
                fig_ef.add_trace(go.Scatter(
                    x=[risk], y=[ret],
                    mode='markers+text',
                    marker=dict(color=colors[idx], size=16, symbol='diamond'),
                    name=name,
                    text=name,
                    textposition="top center",
                    textfont=dict(size=14, color="white")
                ))
        except:
            pass

    fig_ef.update_layout(
        title="مرز کارا (Efficient Frontier) — موقعیت هر استراتژی",
        xaxis_title="ریسک سالانه (%)",
        yaxis_title="بازده سالانه (%)",
        template="plotly_dark",
        width=1000, height=600
    )

    # پرتفوی نهایی کاربر
    hedge = st.session_state.hedge_strategy
    opt = st.session_state.option_strategy
    info_hedge = hedge_strategies[hedge]
    info_opt = option_strategies[opt]

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
        res = minimize(negative_sharpe, x0, args=(mean_ret, cov_mat, rf),
                      method='SLSQP', bounds=bounds, constraints=constraints)
        weights = res.x if res.success else x0
    except:
        weights = x0

    weights = np.clip(weights, [b[0] for b in bounds], [b[1] for b in bounds])
    weights /= weights.sum()

    port_returns = returns.dot(weights)
    port_return = np.dot(weights, mean_ret) * 252 * 100
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252) * 100
    sharpe = (port_return - rf) / port_risk if port_risk > 0 else 0
    recovery = format_recovery(calculate_recovery_time(port_returns))
    max_dd = max_drawdown(port_returns)
    max_g = max_gain(port_returns)

    # نمایش نتایج
    st.success(f"استراتژی فعال: **{hedge}** + **{opt}**")
    cols = st.columns(6)
    cols[0].metric("بازده سالانه", f"{port_return:.2f}%")
    cols[1].metric("بهترین حالت", f"{max_g:.2f}%")
    cols[2].metric("ریسک سالانه", f"{port_risk:.2f}%")
    cols[3].metric("بدترین حالت", f"{max_dd:.2f}%")
    cols[4].metric("نسبت شارپ", f"{sharpe:.3f}")
    cols[5].metric("زمان ریکاوری", recovery)

    if info_opt["cost_pct"] > 0:
        st.warning(f"هزینه آپشن سالانه ≈ {info_opt['cost_pct']:.1f}%")
    elif info_opt["cost_pct"] < 0:
        st.success(f"درآمد سالانه از آپشن ≈ {abs(info_opt['cost_pct']):.1f}%")

    # نمودار مرز کارا
    st.plotly_chart(fig_ef, use_container_width=True)

    # نمودار تخصیص دارایی
    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)})
    df_w = df_w[df_w["وزن (%)"] > 0].sort_values("وزن (%)", ascending=False)
    st.markdown("### تخصیص دارایی")
    col1, col2 = st.columns([2,1])
    with col1:
        st.dataframe(df_w, use_container_width=True)
    with col2:
        fig_pie = px.pie(df_w, values="وزن (%)", names="دارایی", color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig_pie, use_container_width=True)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate – کامل", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Ultimate</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>اپ حرفه‌ای سرمایه‌گذاری ایرانی – با جدول‌های کامل و مرز کارا</h3>", unsafe_allow_html=True)

# پیش‌فرض‌ها
defaults = {"rf_rate": 18.0, "hedge_strategy": "Barbell طالب (۹۰/۱۰)", "option_strategy": "بدون آپشن"}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# سایدبار اصلی
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

# سایدبار جدا برای توضیحات
with st.sidebar:
    st.markdown("### راهنمای کامل استراتژی‌ها")
    
    # جدول هجینگ
    st.markdown("#### استراتژی‌های هجینگ")
    hedge_df = pd.DataFrame([
        ["Barbell طالب (۹۰/۱۰)", "۹۰٪ ایمن + ۱۰٪ پرریسک", "همیشه — ایده اصلی نسیم طالب", "۴۵٪", "۴۵٪", "۱۰٪"],
        ["Tail-Risk طالب", "حفاظت در فاجعه + Put دور", "وقتی احتمال بلک سوان بالاست", "۳۵٪", "۳۵٪", "۵٪"],
        ["Antifragile طالب", "هرچه آشوب بیشتر، سود بیشتر!", "ایران ۱۴۰۴ با تورم و تحریم", "۴۰٪", "۲۰٪", "۴۰٪"],
        ["طلا + تتر (ترکیبی)", "تعادل بین حفاظت و رشد", "سرمایه‌گذار معمولی ایرانی", "۱۵٪", "۱۰٪", "۲۰٪"],
        ["حداقل هج", "فقط یک لایه حفاظتی کوچک", "ریسک‌پذیر در بازار صعودی", "۱۰٪", "۰٪", "۴۰٪"],
        ["بدون هجینگ (کاملاً آزاد)", "هیچ محدودیتی ندارد", "فقط برای تحمل ضرر ۵۰–۸۰٪", "۰٪", "۰٪", "۱۰۰٪"],
    ], columns=["استراتژی", "چرا این استراتژی؟", "مناسب برای", "حداقل طلا", "حداقل تتر/دلار", "حداکثر بیت‌کوین"])
    st.dataframe(hedge_df, use_container_width=True, hide_index=True)
    st.session_state.hedge_strategy = st.selectbox("انتخاب هجینگ", options=hedge_df["استراتژی"])

    # جدول آپشن
    st.markdown("#### استراتژی‌های آپشن")
    option_df = pd.DataFrame([
        ["بدون آپشن", "ساده و بدون هزینه", "بازار صعودی قوی", "۰٪"],
        ["Protective Put", "بیمه کامل پرتفوی", "خواب راحت در بحران", "+۴.۸٪"],
        ["Collar (تقریباً رایگان)", "هج کم‌هزینه", "بازار متوسط", "+۰.۴٪"],
        ["Covered Call", "درآمد ماهانه", "بازار رنج", "−۳.۲٪ (درآمد)"],
        ["Cash-Secured Put", "خرید ارزان در ریزش", "وقتی به کف اعتقاد دارید", "−۲.۹٪ (درآمد)"],
        ["Tail-Risk Put", "حفاظت در فاجعه", "کرونا، جنگ", "+۲.۱٪"],
        ["Iron Condor", "درآمد از ثبات", "بازار آرام", "−۵.۵٪ (درآمد)"],
    ], columns=["استراتژی", "چرا این استراتژی؟", "مناسب برای", "هزینه سالانه"])
    st.dataframe(option_df, use_container_width=True, hide_index=True)
    st.session_state.option_strategy = st.selectbox("انتخاب آپشن", options=option_df["استراتژی"])

    # توضیح ریاضی مرز کارا
    st.markdown("### توضیح ریاضی مرز کارا")
    st.markdown("""
    **مرز کارا** مجموعه‌ای از پرتفوی‌هاست که:
    - حداکثر بازده را برای یک سطح ریسک مشخص می‌دهند  
    - یا حداقل ریسک را برای یک سطح بازده مشخص می‌دهند

    **فرمول ریاضی**:
    - بازده پرتفوی:  
      $$ R_p = \\sum_{i=1}^{n} w_i R_i $$
    - ریسک پرتفوی:  
      $$ \\sigma_p = \\sqrt{w^T \\Sigma w} $$

    هر نقطه روی این منحنی، **بهترین ترکیب ممکن** از دارایی‌هاست.

    **نقطه قرمز ستاره**: پرتفویی با **بیشترین نسبت شارپ**  
    **نقاط رنگی**: موقعیت استراتژی‌های طالب و دیگران روی مرز کارا

    نسیم طالب میگه:  
    «من نمی‌خوام روی مرز کارا باشم — من می‌خوام در سمت چپ خیلی پایین باشم (۹۰٪ ایمن) و با ۱۰٪ بقیه، شانس سود انفجاری داشته باشم.»
    """)

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Ultimate – نسخه نهایی با جدول‌های کامل و توضیح ریاضی | با عشق برای ایران")
