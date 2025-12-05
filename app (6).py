import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# ------------------ تابع تبدیل دوره به سال و ماه ------------------
def format_recovery_time(periods, period_type):
    if periods == 0 or pd.isna(periods):
        return "بدون افت جدی"
    
    months_per_period = {'ماهانه': 1, 'سه‌ماهه': 3, 'شش‌ماهه': 6}[period_type]
    total_months = periods * months_per_period
    
    years = int(total_months // 12)
    months = int(total_months % 12)
    decimal_month = (total_months % 12) - months
    
    parts = []
    if years > 0:
        parts.append(f"{years} سال")
    if months > 0 or decimal_month > 0.3:  # حداقل نیم ماه نشون بده
        month_display = months + (1 if decimal_month >= 0.5 else 0)
        if month_display > 0:
            parts.append(f"{month_display} ماه")
    
    return " و ".join(parts) if parts else f"{total_months:.1f} ماه"

# ------------------ محاسبه زمان ریکاوری ------------------
def calculate_recovery_time(returns_series):
    if len(returns_series) < 2:
        return 0
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    in_drawdown = False
    recovery_times = []
    start_idx = None
    for i in range(1, len(cum_returns)):
        if drawdown.iloc[i] < 0:
            if not in_drawdown:
                in_drawdown = True
                start_idx = i
        else:
            if in_drawdown:
                in_drawdown = False
                recovery_times.append(i - start_idx)
    return np.mean(recovery_times) if recovery_times else 0

# ------------------ صفحه ------------------
st.set_page_config(page_title="تحلیل پرتفوی هوشمند + زمان دقیق ریکاوری", layout="wide")
st.title("ابزار تحلیل پرتفوی هوشمند + زمان دقیق بازگشت به قله")

# ------------------ تست ریسک ------------------
st.sidebar.header("تست پروفایل ریسک رفتاری")
with st.sidebar.expander("انجام تست ریسک (اجباری)", expanded=True):
    q1 = st.radio("اگر پرتفوی شما ۱۵٪ افت کند چه می‌کنید؟", 
                  ["سریع می‌فروشم", "نگه می‌دارم", "خرید می‌کنم"], key="q1")
    q2 = st.radio("سرمایه‌گذاری پرریسک با بازده بالا؟", 
                  ["نگران", "بی‌تفاوت", "هیجان‌زده"], key="q2")
    q3 = st.radio("کدام جمله به شما نزدیک‌تر است؟", [
        "ترجیح می‌دهم سود کم ولی قطعی",
        "سود متوسط با کمی ریسک",
        "پتانسیل سود بالا مهم‌تر از ریسک است"
    ], key="q3")
    q4 = st.radio("در گذشته بعد از ضرر سنگین چه کردید؟", [
        "کاملاً عقب‌نشینی کردم",
        "تحمل کردم و صبر کردم",
        "با تحلیل دوباره وارد شدم"
    ], key="q4")

    map1 = {"سریع می‌فروشم": 1, "نگه می‌دارم": 2, "خرید می‌کنم": 3}
    map2 = {"نگران": 1, "بی‌تفاوت": 2, "هیجان‌زده": 3}
    map3 = {"ترجیح می‌دهم سود کم ولی قطعی": 1, "سود متوسط با کمی ریسک": 2, "پتانسیل سود بالا مهم‌تر از ریسک است": 3}
    map4 = {"کاملاً عقب‌نشینی کردم": 1, "تحمل کردم و صبر کردم": 2, "با تحلیل دوباره وارد شدم": 3}

    if st.button("محاسبه پروفایل ریسک من"):
        score = map1[q1] + map2[q2] + map3[q3] + map4[q4]
        if score <= 6:
            profile, risk = "محافظه‌کار", 0.12
        elif score <= 9:
            profile, risk = "متعادل", 0.25
        else:
            profile, risk = "تهاجمی", 0.45
        st.session_state.risk_profile = profile
        st.session_state.risk_target = risk
        st.success(f"پروفایل ریسک شما: {profile}")

if "risk_target" not in st.session_state:
    st.warning("لطفاً ابتدا تست ریسک رفتاری را انجام دهید.")
    st.stop()

# ------------------ تنظیمات ------------------
with st.sidebar.expander("تنظیمات تحلیل", expanded=True):
    period = st.selectbox("بازه تحلیل بازده", ["ماهانه", "سه‌ماهه", "شش‌ماهه"], index=0)
    rf_rate = st.number_input("نرخ بدون ریسک سالانه (%)", 0.0, 30.0, 5.0, 0.5)
    seed = st.number_input("Seed تصادفی", 0, 99999, 42)

resample_map = {"ماهانه": "M", "سه‌ماهه": "Q", "شش‌ماهه": "6M"}
annual_factor = {"ماهانه": 12, "سه‌ماهه": 4, "شش‌ماهه": 2}

resample_rule = resample_map[period]

# ------------------ دریافت داده ------------------
all_assets = []

# آپلود CSV
uploaded = st.sidebar.file_uploader("آپلود فایل CSV دارایی‌ها", type="csv", accept_multiple_files=True)
if uploaded:
    for file in uploaded:
        try:
            df = pd.read_csv(file)
            if 'date' in df.columns.str.lower() and ('price' in df.columns.str.lower() or 'close' in df.columns.str.lower()):
                date_col = [c for c in df.columns if 'date' in c.lower()][0]
                price_col = [c for c in df.columns if c.lower() in ['price', 'close']][0]
                df = df[[date_col, price_col]].copy()
                df.columns = ['Date', 'Price']
                df['Date'] = pd.to_datetime(df['Date'])
                name = file.name.replace('.csv', '').replace('.CSV', '')
                all_assets.append((name, df))
        except:
            st.error(f"خطا در خواندن {file.name}")

# دانلود از یاهو
with st.sidebar.expander("دانلود آنلاین از Yahoo Finance"):
    symbols = st.text_input("نمادها (مثال: BTC-USD,AAPL,TSLA)", "BTC-USD,AAPL,ETH-USD,GOOGL")
    if st.button("دانلود داده‌ها"):
        tickers = [t.strip() for t in symbols.split(",") if t.strip()]
        with st.spinner("در حال دانلود..."):
            data = yf.download(tickers, period="max", progress=False)['Close']
        for col in data.columns:
            df = data[col].dropna().reset_index()
            df.columns = ['Date', 'Price']
            all_assets.append((col, df))

if not all_assets:
    st.info("لطفاً داده‌ها را آپلود یا دانلود کنید.")
    st.stop()

# ------------------ پردازش قیمت‌ها ------------------
prices = pd.DataFrame()
for name, df in all_assets:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')['Price'].to_frame(name)
    prices = prices.join(df, how='outer') if not prices.empty else df

prices = prices.resample(resample_rule).last().ffill().dropna()
returns = prices.pct_change().dropna()

if len(returns) < 10:
    st.error("داده کافی برای تحلیل وجود ندارد.")
    st.stop()

mean_returns = returns.mean() * annual_factor[period]
cov_matrix = returns.cov() * annual_factor[period]
asset_names = list(returns.columns)

# ------------------ مونت‌کارلو ------------------
np.random.seed(seed)
n_ports = 12000
results = np.zeros((3, n_ports))
weights_record = np.zeros((n_ports, len(asset_names)))

for i in range(n_ports):
    w = np.random.random(len(asset_names))
    w /= w.sum()
    w = np.clip(w, 0.02, 0.5)
    w /= w.sum()
    
    ret = np.dot(w, mean_returns)
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    results[0,i] = ret
    results[1,i] = vol
    results[2,i] = (ret - rf_rate/100) / vol if vol > 0 else -99
    weights_record[i] = w

# بهترین پرتفو نزدیک به ریسک هدف
target = st.session_state.risk_target
diff = np.abs(results[1] - target)
best_idx = np.argmin(diff)
mc_weights = weights_record[best_idx]

# ------------------ سبک‌ها ------------------
styles = {
    "مونت‌کارلو (هدف ریسک)": mc_weights,
    "وزن مساوی": np.ones(len(asset_names)) / len(asset_names),
    "حداقل واریانس": np.linalg.solve(cov_matrix, np.ones(len(asset_names))),
}
styles["حداقل واریانس"] = styles["حداقل واریانس"] / styles["حداقل واریانس"].sum()

# ------------------ نمایش نتایج با تب ------------------
st.markdown("---")
st.subheader("نتایج پرتفوهای بهینه")

tabs = st.tabs(list(styles.keys()))

for tab, (name, w) in zip(tabs, styles.items()):
    with tab:
        port_ret = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe = (port_ret - rf_rate/100) / port_vol if port_vol > 0 else 0
        recovery_periods = calculate_recovery_time(returns.dot(w))
        recovery_text = format_recovery_time(recovery_periods, period)

        col1, col2 = st.columns([2, 3])

        with col1:
            fig = px.pie(
                values=w*100,
                names=asset_names,
                title=f"توزیع وزنی — {name}",
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=420, showlegend=True, legend_title="دارایی")
            st.plotly_chart(fig, use_container_width=True, key=f"pie_{name}_{hash(name)}")

        with col2:
            st.metric("بازده سالانه پیش‌بینی", f"{port_ret*100:.2f}%")
            st.metric("ریسک سالانه", f"{port_vol*100:.2f}%")
            st.metric("نسبت شارپ", f"{sharpe:.3f}")
            st.metric("زمان ریکاوری میانگین", recovery_text, delta=None)

# ------------------ جدول ریکاوری هر دارایی ------------------
st.markdown("---")
st.subheader("زمان بازگشت به قله (Recovery Time) هر دارایی")

rec_list = []
for asset in asset_names:
    rt = calculate_recovery_time(returns[asset])
    rec_list.append({
        "دارایی": asset,
        "زمان ریکاوری": format_recovery_time(rt, period)
    })

st.dataframe(rec_list, use_container_width=True)

st.success("تحلیل با موفقیت انجام شد! حالا بهترین پرتفو را انتخاب کنید")
st.balloons()
