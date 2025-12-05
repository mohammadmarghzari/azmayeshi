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
def portfolio_performance(weights, mean_returns, cov_matrix, rf_rate):
    returns = np.dot(weights, mean_returns) * 252 * 100
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) * 100
    sharpe = (returns - rf_rate) / std if std > 0 else 0
    return returns, std, sharpe

def negative_sharpe(weights, mean_returns, cov_matrix, rf_rate):
    returns, std, _ = portfolio_performance(weights, mean_returns, cov_matrix, rf_rate)
    return - (returns - rf_rate) / std

def max_drawdown(returns):
    if len(returns) == 0:
        return 0
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min() * 100

def calculate_recovery_time(ret_series):
    if len(ret_series) == 0:
        return 0
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

# ==================== استراتژی‌ها ====================
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

# ==================== جدول‌های کامل ====================
hedge_table = pd.DataFrame([
    ["Barbell طالب (۹۰/۱۰)", "۹۰٪ ایمن + ۱۰٪ پرریسک — ایده اصلی نسیم طالب", "همیشه — مخصوصاً وقتی نمی‌دانید آینده چه می‌شود", "۴۵٪", "۴۵٪", "۱۰٪"],
    ["Tail-Risk طالب", "حفاظت در فاجعه + خرید Put دور — طالب در ۲۰۲۰ +۴۰۰۰٪ سود کرد", "وقتی احتمال جنگ/کرونا/سقوط دلار بالاست", "۳۵٪", "۳۵٪", "۵٪"],
    ["Antifragile طالب", "هرچه آشوب بیشتر، سود بیشتر! طلا + بیت‌کوین + Long Volatility", "ایران ۱۴۰۴ با تورم و تحریم", "۴۰٪", "۲۰٪", "۴۰٪"],
    ["طلا + تتر (ترکیبی)", "بهترین دفاع در شرایط فعلی ایران — تعادل عالی", "سرمایه‌گذار معمولی ایرانی", "۱۵٪", "۱۰٪", "۲۰٪"],
    ["حداقل هج", "فقط یک لایه حفاظتی کوچک", "ریسک‌پذیر در بازار صعودی", "۱۰٪", "۰٪", "۴۰٪"],
    ["بدون هجینگ (کاملاً آزاد)", "هیچ محدودیتی ندارد — فقط برای تحمل ضرر ۵۰–۸۰٪", "افق بلندمدت + بخش زیادی از دارایی جداگانه در طلا/دلار", "۰٪", "۰٪", "۱۰۰٪"],
], columns=["استراتژی", "چرا این استراتژی؟", "مناسب برای", "حداقل طلا", "حداقل تتر/دلار", "حداکثر بیت‌کوین"])

option_table = pd.DataFrame([
    ["بدون آپشن", "ساده و بدون هزینه", "بازار صعودی قوی", "۰٪"],
    ["Protective Put", "بیمه کامل پرتفوی", "خواب راحت در بحران", "+۴.۸٪"],
    ["Collar (تقریباً رایگان)", "هج کم‌هزینه با فروش Call", "بازار متوسط", "+۰.۴٪"],
    ["Covered Call", "درآمد ماهانه از فروش Call", "بازار رنج", "−۳.۲٪ (درآمد)"],
    ["Cash-Secured Put", "خرید ارزان در ریزش", "وقتی به کف اعتقاد دارید", "−۲.۹٪ (درآمد)"],
    ["Tail-Risk Put", "حفاظت در فاجعه با هزینه کم", "کرونا، جنگ", "+۲.۱٪"],
    ["Iron Condor", "درآمد بالا از ثبات", "بازار آرام", "−۵.۵٪ (درآمد)"],
], 
], columns=["استراتژی", "چرا این استراتژی؟", "مناسب برای", "هزینه سالانه"])

# ==================== محاسبه پرتفوی + همه نمودارها ====================
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

    # ———————— مرز کارا ————————
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        w = np.random.random(len(asset_names))
        w /= w.sum()
        ret, risk, sharpe = portfolio_performance(w, mean_ret, cov_mat, rf)
        results[0,i] = risk
        results[1,i] = ret
        results[2,i] = sharpe

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
        x=[results[0, max_sharpe_idx]], y=results[1, max_sharpe_idx]],
        mode='markers',
        marker=dict(color='red', size=18, symbol='star'),
        name=f'بهینه‌ترین پرتفوی (شارپ: {results[2, max_sharpe_idx]:.2f})'
    ))

    # نقاط استراتژی‌ها
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5']
    for idx, (name, info) in enumerate(hedge_strategies.items()):
        bounds = [(info.get("gold_min",0), 1.0) if "GOLD" in a or "GC=" in a else (0,1) for a in asset_names]
        # ساده‌سازی برای سرعت (محدودیت‌ها رو فقط در صورتی اعمال می‌کنیم که دارایی موجود باشه)
        try:
            res = minimize(negative_sharpe, x0=np.ones(len(asset_names))/len(asset_names),
                          args=(mean_ret, cov_mat, rf),
                          method='SLSQP', bounds=[(0,1)]*len(asset_names),
                          constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x)-1}])
            if res.success:
                w = res.x
                w /= w.sum()
                ret, risk, sh = portfolio_performance(w, mean_ret, cov_mat, rf)
                fig_ef.add_trace(go.Scatter(
                    x=[risk], y=[ret],
                    mode='markers+text',
                    marker=dict(color=colors[idx], size=16, symbol='diamond'),
                    name=name,
                    text=name,
                    textposition="top center"
                ))
        except:
            pass

    fig_ef.update_layout(
        title="مرز کارا (Efficient Frontier)",
        xaxis_title="ریسک سالانه (%)",
        yaxis_title="بازده سالانه (%)",
        template="plotly_dark",
        height=600
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # ———————— توضیح ریاضی مرز کارا ————————
    with st.expander("مرز کارا چیست؟ — توضیح ریاضی کامل", expanded=False):
        st.markdown("""
        **مرز کارا** مجموعه پرتفوی‌هایی است که:
        - برای یک سطح ریسک مشخص، **بیشترین بازده** را می‌دهند  
        - یا برای یک سطح بازده مشخص، **کمترین ریسک** را دارند

        **فرمول‌ها**:
        - بازده پرتفوی:  
          $$ R_p = \\sum_{i} w_i R_i $$
        - ریسک پرتفوی:  
          $$ \\sigma_p = \\sqrt{\\mathbf{w}^T \\Sigma \\mathbf{w}} $$

        هر نقطه روی این منحنی **بهترین ترکیب ممکن** از دارایی‌هاست.
        نقطه قرمز ستاره = پرتفویی با **بیشترین نسبت شارپ** (بهترین تعادل بازده/ریسک).

        نسیم طالب می‌گوید:  
        «من نمی‌خواهم روی مرز کارا باشم — می‌خواهم در سمت چپ خیلی پایین باشم (۹۰٪ ایمن) و با ۱۰٪ بقیه، شانس سود انفجاری داشته باشم.»
        """, unsafe_allow_html=True)

    # ———————— پرتفوی نهایی کاربر ————————
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
    sharpe_ratio = (port_return - rf) / port_risk if port_risk > 0 else 0
    current_dd = max_drawdown(port_returns)
    recovery = format_recovery(calculate_recovery_time(port_returns))

    st.success(f"استراتژی فعال: **{hedge}** + **{opt}**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("بازده سالانه", f"{port_return:.2f}%")
    c2.metric("ریسک سالانه", f"{port_risk:.2f}%")
    c3.metric("نسبت شارپ", f"{sharpe_ratio:.3f}")
    c4.metric("بدترین افت", f"{current_dd:.1f}%")
    c5.metric("وضعیت فعلی", "خطرناک!" if current_dd < -20 else "هشدار" if current_dd < -10 else "عالی")
    c6.metric("زمان ریکاوری", recovery)

    # هشدار سقوط
    if current_dd < -20:
        st.error(f"هشدار سقوط شدید: پرتفوی شما {abs(current_dd):.1f}% از قله افت کرده!")
    elif current_dd < -10:
        st.warning(f"افت {abs(current_dd):.1f}% از قله — مراقب باشید!")

    # نمودار تخصیص
    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)})
    df_w = df_w[df_w["وزن (%)"] > 0].sort_values("وزن (%)", ascending=False)
    st.markdown("### تخصیص دارایی")
    col1, col2 = st.columns([2,1])
    with col1:
        st.dataframe(df_w, use_container_width=True)
    with col2:
        fig_pie = px.pie(df_w, values="وزن (%)", names="دارایی", color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig_pie, use_container_width=True)

    # حالت چه می‌شد اگر
    st.markdown("### چه می‌شد اگر؟")
    col1, col2, col3 = st.columns(3)
    initial = col1.number_input("سرمایه اولیه (میلیون تومان)", 10, 10000, 100)
    years = col2.selectbox("چند سال پیش شروع کرده بودید؟", [1, 3, 5, 10])
    monthly = col3.number_input("سرمایه‌گذاری ماهانه (میلیون)", 0, 100, 5)

    port_daily = returns.dot(weights)
    backtest = (1 + port_daily).cumprod()

    if len(backtest) > years * 252:
        backtest = backtest.tail(years * 252)

    value = initial
    for i, r in enumerate(backtest):
        value *= (1 + r)
        if i % 21 == 0 and i > 0:  # هر ماه
            value += monthly

    total_invested = initial + (monthly * years * 12)
    profit = value - total_invested
    profit_pct = (profit / total_invested) * 100 if total_invested > 0 else 0

    st.metric("اگر از اون موقع شروع کرده بودید، الان داشتید:", f"{value:,.0f} میلیون تومان")
    st.metric("سود خالص", f"{profit:,.0f} میلیون تومان", delta=f"{profit_pct:.1f}%")

    fig_back = px.line(backtest * initial, title=f"رشد سرمایه از {years} سال پیش")
    st.plotly_chart(fig_back, use_container_width=True)

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate – نهایی", layout="wide")
st.markdown("<h1 style='text-align: center; color: #00d2d3;'>Portfolio360 Ultimate</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>اپ حرفه‌ای سرمایه‌گذاری ایرانی – با مرز کارا، جدول کامل و «چه می‌شد اگر»</h3>", unsafe_allow_html=True)

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

# جدول‌های کامل در سایدبار
st.sidebar.markdown("### راهنمای کامل استراتژی‌های هجینگ")
st.sidebar.dataframe(hedge_table, use_container_width=True, hide_index=True)
st.session_state.hedge_strategy = st.sidebar.selectbox("انتخاب هجینگ", options=hedge_table["استراتژی"])

st.sidebar.markdown("### راهنمای کامل استراتژی‌های آپشن")
st.sidebar.dataframe(option_table, use_container_width=True, hide_index=True)
st.session_state.option_strategy = st.sidebar.selectbox("انتخاب آپشن", options=option_table["استراتژی"])

# اجرا
calculate_portfolio()

st.balloons()
st.caption("Portfolio360 Ultimate – نسخه نهایی | با عشق برای ایران")
