import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# ------------------ تابع محاسبه زمان ریکاوری ------------------
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

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="تحلیل پرتفوی + زمان ریکاوری", layout="wide")
st.title("ابزار تحلیل پرتفوی هوشمند + زمان بازگشت به قله")

# ------------------ تست ریسک رفتاری ------------------
st.sidebar.header("تست پروفایل ریسک رفتاری")
with st.sidebar.expander("انجام تست ریسک (اجباری)", expanded=True):
    q1 = st.radio("اگر پرتفوی شما ۱۵٪ افت کند چه می‌کنید؟", 
                  ["سریع می‌فروشم", "نگه می‌دارم", "خرید می‌کنم"], key="q1")
    q2 = st.radio("سرمایه‌گذاری پرریسک با بازده بالا برایتان؟", 
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

    # نقشه امتیازات
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
        st.success(f"پروفایل ریسک شما: **{profile}**")

# اگر تست نداده باشد، جلوی اجرا رو بگیر
if "risk_target" not in st.session_state:
    st.warning("لطفاً ابتدا تست ریسک رفتاری را انجام دهید.")
    st.stop()

# ------------------ تنظیمات ------------------
with st.sidebar.expander("تنظیمات تحلیل", expanded=True):
    period = st.selectbox("بازه تحلیل", ["ماهانه", "سه‌ماهه", "شش‌ماهه"])
    rf_rate = st.number_input("نرخ بدون ریسک سالانه (%)", 0.0, 30.0, 5.0, 0.5)
    seed = st.number_input("Seed تصادفی", 0, 99999, 42)

resample_map = {"ماهانه": "M", "سه‌ماهه": "Q", "شش‌ماهه": "6M"}
period_name_map = {"M": "ماه", "Q": "سه‌ماه", "6M": "شش‌ماه"}
annual_factor = {"ماهانه": 12, "سه‌ماهه": 4, "شش‌ماهه": 2}

resample_rule = resample_map[period]
period_name = period_name_map[resample_rule]

# ------------------ دریافت داده ------------------
all_assets = []

# آپلود فایل
uploaded = st.sidebar.file_uploader("آپلود CSV (هر فایل یک دارایی)", type="csv", accept_multiple_files=True)
if uploaded:
    for file in uploaded:
        df = pd.read_csv(file)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'date' in df.columns and 'price' in df.columns:
            df = df[['date', 'price']].copy()
            df.columns = ['Date', 'Price']
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df.columns = [file.name.replace('.csv', '')]
            all_assets.append((file.name.replace('.csv', ''), df))

# دانلود از یاهو
with st.sidebar.expander("دانلود از Yahoo Finance"):
    symbols = st.text_input("نمادها (با کاما)", "BTC-USD,AAPL,ETH-USD,TSLA")
    if st.button("دانلود داده"):
        tickers = [t.strip() for t in symbols.split(",") if t.strip()]
        data = yf.download(tickers, period="5y", progress=False)['Close']
        for col in data.columns:
            df = data[col].dropna().reset_index()
            df.columns = ['Date', 'Price']
            all_assets.append((col, df))

if not all_assets:
    st.info("لطفاً داده دارایی‌ها را آپلود یا دانلود کنید.")
    st.stop()

# ------------------ ساخت دیتافریم قیمت ------------------
prices = pd.DataFrame()
for name, df in all_assets:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')['Price'].to_frame(name)
    prices = prices.join(df, how='outer') if not prices.empty else df

prices = prices.resample(resample_rule).last().dropna()
returns = prices.pct_change().dropna()

if len(returns) < 8:
    st.error("داده کافی نیست.")
    st.stop()

mean_returns = returns.mean() * annual_factor[period]
cov_matrix = returns.cov() * annual_factor[period]
asset_names = list(returns.columns)

# ------------------ مونت‌کارلو ساده برای هدف ریسک ------------------
np.random.seed(seed)
n_ports = 10000
results = np.zeros((3, n_ports))
weights_record = np.zeros((n_ports, len(asset_names)))

for i in range(n_ports):
    w = np.random.random(len(asset_names))
    w /= w.sum()
    w = np.clip(w, 0.05, 0.5)
    w /= w.sum()
    
    ret = np.dot(w, mean_returns)
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    results[0,i] = ret
    results[1,i] = vol
    weights_record[i] = w

# بهترین پرتفو نزدیک به ریسک هدف کاربر
target_risk = st.session_state.risk_target
diff = np.abs(results[1] - target_risk)
best_idx = np.argmin(diff)
mc_weights = weights_record[best_idx]

# ------------------ سبک‌های مختلف ------------------
styles = {
    "مونت‌کارلو (هدف ریسک)": mc_weights,
    "وزن مساوی": np.ones(len(asset_names)) / len(asset_names),
    "حداقل ریسک": np.linalg.solve(cov_matrix, np.ones(len(asset_names)))
}
styles["حداقل ریسک"] /= styles["حداقل ریسک"].sum()  # نرمال‌سازی

# ------------------ نمایش با تب‌های زیبا (بدون خطا!) ------------------
st.markdown("---")
st.subheader("نتایج پرتفوهای مختلف")

tabs = st.tabs(list(styles.keys()))

for tab, (name, w) in zip(tabs, styles.items()):
    with tab:
        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe = (port_return - rf_rate/100) / port_vol if port_vol > 0 else 0
        
        port_series = returns.dot(w)
        recovery = calculate_recovery_time(port_series)

        col1, col2 = st.columns([2, 3])

        with col1:
            # اینجا کلید منحصر به فرد داریم → خطای Duplicate نمی‌ده
            fig = px.pie(
                values=w*100, 
                names=asset_names,
                title=f"توزیع وزنی — {name}",
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"pie_{name}")

        with col2:
            st.metric("بازده سالانه", f"{port_return*100:.2f}%")
            st.metric("ریسک سالانه", f"{port_vol*100:.2f}%")
            st.metric("نسبت شارپ", f"{sharpe:.3f}")
            st.metric("زمان ریکاوری میانگین", f"{recovery:.1f} {period_name}")

# ------------------ جدول زمان ریکاوری هر دارایی ------------------
st.markdown("---")
st.subheader("زمان بازگشت به قله (Recovery Time) هر دارایی")
rec_list = []
for asset in asset_names:
    rt = calculate_recovery_time(returns[asset])
    rec_list.append({"دارایی": asset, "زمان ریکاوری": f"{rt:.1f} {period_name}" if rt > 0 else "بدون افت جدی"})
st.dataframe(rec_list, use_container_width=True)

st.success("تحلیل با موفقیت انجام شد!")
