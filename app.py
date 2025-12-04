import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import scipy.optimize as sco

# ------------------ Utils -------------------
def calculate_recovery_time(returns_series):
    """محاسبه میانگین تعداد دوره‌های لازم برای بازگشت به قله قبلی"""
    if len(returns_series) < 2:
        return 0
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = cum_returns / running_max - 1
    in_drawdown = False
    recovery_times = []
    start_idx = None
    
    for i in range(1, len(cum_returns)):
        if drawdowns.iloc[i] < 0:
            if not in_drawdown:
                in_drawdown = True
                start_idx = i
        else:
            if in_drawdown:
                in_drawdown = False
                recovery_time = i - start_idx
                recovery_times.append(recovery_time)
    
    return np.mean(recovery_times) if recovery_times else 0

def get_price_dataframe_from_yf(data, ticker):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            price_series = data[ticker]['Close']
        else:
            price_series = data['Close']
        df = price_series.reset_index()
        df.columns = ['Date', 'Price']
        return df, None
    except Exception as e:
        return None, f"خطا در پردازش داده {ticker}: {e}"

def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace('%', '')
        df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
        if 'Date' not in df.columns or 'Price' not in df.columns:
            return None, "فایل باید ستون‌های 'Date' و 'Price' داشته باشد."
        return df, None
    except Exception as e:
        return None, f"خطا در خواندن فایل {file.name}: {e}"

def msg(msg, level="warning"):
    if level == "warning": st.warning(msg)
    elif level == "error": st.error(msg)
    elif level == "info": st.info(msg)
    else: st.success(msg)

def compact_pie_weights(asset_names, weights, min_percent=0.5):
    weights_percent = 100 * np.array(weights)
    shown_assets, shown_weights = [], []
    other_weight = 0
    for name, w in zip(asset_names, weights_percent):
        if w >= min_percent:
            shown_assets.append(name)
            shown_weights.append(w)
        else:
            other_weight += w
    if other_weight > 0:
        shown_assets.append('سایر')
        shown_weights.append(other_weight)
    return shown_assets, shown_weights

def portfolio_stats(weights, mean_returns, cov_matrix, returns, rf, annual_factor):
    port_ann_return = np.dot(weights, mean_returns)
    port_ann_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    downrets = returns.copy(); downrets[downrets > 0] = 0
    port_ann_downstd = np.sqrt(np.dot(weights.T, np.dot(downrets.cov() * annual_factor, weights)))
    sharpe = (port_ann_return - rf/100) / (port_ann_vol if port_ann_vol else np.nan)
    sortino = (port_ann_return - rf/100) / (port_ann_downstd if port_ann_downstd else np.nan)

    stats = {}
    for label, period in [('سالانه', annual_factor), ('سه‌ماهه', 3), ('دوماهه', 2), ('یک‌ماهه', 1)]:
        mu = np.dot(weights, mean_returns) / annual_factor
        sigma = port_ann_vol / np.sqrt(annual_factor)
        stats[label] = {"return": mu * period, "vol": sigma * np.sqrt(period)}
    stats['sharpe'] = sharpe
    stats['sortino'] = sortino
    return stats

# ---------------- Streamlit App -----------------
st.set_page_config(page_title="تحلیل پرتفو + زمان ریکاوری", layout="wide")
st.title("ابزار تحلیل پرتفوی هوشمند + زمان بازگشت سرمایه (Recovery Time)")

# ---------------- Sidebar: تست ریسک ----------------
st.sidebar.markdown("## تست پروفایل ریسک رفتاری")
with st.sidebar.expander("انجام تست ریسک (اجباری)", expanded=True):
    q1 = st.radio("اگر پرتفوی شما ۱۵٪ افت کند چه می‌کنید؟", ["می‌فروشم", "نگه می‌دارم", "می‌خرم"], key="q1")
    q2 = st.radio("سرمایه‌گذاری پرریسک با بازده بالا برایتان چطور است؟", ["نگرانم", "بی‌تفاوت", "هیجان‌زده"], key="q2")
    q3 = st.radio("کدام به شما نزدیک‌تر است؟", [
        "سود کم ولی مطمئن", "سود متوسط با کمی ریسک", "سود بالا حتی با ریسک زیاد"
    ], key="q3")
    q4 = st.radio("در گذشته بعد از ضرر سنگین چه کردید؟", [
        "عقب‌نشینی کامل", "صبر کردم", "دوباره با تحلیل وارد شدم"
    ], key="q4")

    score = {"می‌فروشم":1,"نگه می‌دارم":2,"می‌خرم":3}
    = {"نگرانم":1,"بی‌تفاوت":2,"هیجان‌زده":3}
    = {"سود کم ولی مطمئن":1,"سود متوسط با کمی ریسک":2,"سود بالا حتی با ریسک زیاد":3}
    = {"عقب‌نشینی کامل":1,"صبر کردم":2,"دوباره با تحلیل وارد شدم":3}

    if st.button("محاسبه پروفایل ریسک من", key="calc_risk"):
        score = [q1] + [q2] + [q3] + [q4]
        if score <= 6:
            profile, risk_val = "محافظه‌کار", 0.12
        elif score <= 9:
            profile, risk_val = "متعادل", 0.25
        else:
            profile, risk_val = "تهاجمی", 0.40
        st.session_state.risk_profile = profile
        st.session_state.risk_value = risk_val
        st.success(f"پروفایل شما: **{profile}**")

if "risk_value" not in st.session_state:
    st.warning("لطفاً ابتدا تست ریسک را انجام دهید.")
    st.stop()

# ---------------- Sidebar: تنظیمات ----------------
with st.sidebar.expander("تنظیمات تحلیل", expanded=True):
    period = st.selectbox("بازه زمانی بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'], index=0)
    rf = st.number_input("نرخ بدون ریسک سالانه (%)", 0.0, 50.0, 5.0, 0.5)
    total_capital = st.number_input("سرمایه کل ($)", value=100_000.0)
    n_portfolios = st.slider("تعداد پرتفوهای مونت‌کارلو", 1000, 20000, 8000, 500)
    seed_value = st.number_input("Seed تصادفی", 0, 99999, 42)

resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
period_name = {"M": "ماه", "Q": "سه‌ماه", "2Q": "شش‌ماه"}[resample_rule]

user_risk_target = st.sidebar.slider(
    "ریسک هدف پرتفوی (انحراف معیار سالانه)",
    0.05, 0.60, float(st.session_state.risk_value), 0.01
)

# ---------------- دریافت داده ----------------
all_assets = []

# آپلود فایل
uploaded_files = st.sidebar.file_uploader(
    "آپلود فایل CSV (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True
)
if uploaded_files:
    for file in uploaded_files:
        df, err = read_csv_file(file)
        if df is not None:
            all_assets.append((file.name.removesuffix('.csv'), df))

# دانلود از یاهو
with st.sidebar.expander("دریافت آنلاین از Yahoo Finance"):
    tickers_input = st.text_input("نمادها (با کاما)", "BTC-USD,AAPL,TSLA,ETH-USD")
    col_start, col_end = st.columns(2)
    start = col_start.date_input("از", pd.to_datetime("2020-01-01"))
    end = col_end.date_input("تا", pd.to_datetime("today"))
    if st.button("دانلود داده‌ها"):
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        with st.spinner("در حال دانلود..."):
            data = yf.download(tickers, start=start, end=end, progress=False)['Close']
        for t in tickers:
            if t in data.columns:
                df = data[t].dropna().reset_index()
                df.columns = ['Date', 'Price']
                all_assets.append((t, df))
                st.success(f"{t} دانلود شد")

if not all_assets:
    st.info("لطفاً داده دارایی‌ها را آپلود یا دانلود کنید.")
    st.stop()

# ---------------- پردازش داده ----------------
prices_df = pd.DataFrame()
for name, df in all_assets:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna().set_index('Date')['Price'].to_frame(name)
    prices_df = prices_df.join(df, how='outer') if not prices_df.empty else df

prices_df = prices_df.dropna(axis=1, thresh=int(len(prices_df)*0.7))  # حداقل 70% داده
resampled = prices_df.resample(resample_rule).last()
returns = resampled.pct_change().dropna()
if len(returns) < 10:
    st.error("داده کافی برای تحلیل وجود ندارد.")
    st.stop()

mean_returns = returns.mean() * annual_factor
cov_matrix = returns.cov() * annual_factor
asset_names = list(returns.columns)

# ---------------- مونت‌کارلو با هدف ریسک ----------------
np.random.seed(seed_value)
results = np.zeros((3 + len(asset_names), n_portfolios))
weights_record = []

for i in range(n_portfolios):
    w = np.random.random(len(asset_names))
    w /= w.sum()
    # احترام به حداقل/حداکثر وزن (ساده)
    w = np.clip(w, 0.0, 0.50)
    w /= w.sum()
    
    ret = np.dot(w, mean_returns)
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe = (ret - rf/100) / vol if vol > 0 else -999
    
    results[0,i] = ret
    results[1,i] = vol
    results[2,i] = sharpe
    weights_record.append(w)

# بهترین پرتفو با نزدیک‌ترین ریسک به هدف کاربر
diff = np.abs(results[1] - user_risk_target)
best_idx = np.argmin(diff)
best_weights_mc = weights_record[best_idx]

# ---------------- سبک‌های مختلف ----------------
styles = {
    "مونت‌کارلو (هدف ریسک)": best_weights_mc,
    "وزن مساوی": np.ones(len(asset_names)) / len(asset_names),
    "حداقل واریانس": sco.minimize(
        lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
        np.ones(len(asset_names))/len(asset_names),
        constraints={'type':'eq', 'fun':lambda x: np.sum(x)-1},
        bounds=[(0,1)]*len(asset_names),
        method='SLSQP'
    ).x
}

# ---------------- نمایش نتایج با تب‌های زیبا ----------------
st.markdown("---")
st.subheader("نتایج پرتفوهای مختلف")

tabs = st.tabs(list(styles.keys()))

for tab, (name, weights) in zip(tabs, styles.items()):
    with tab:
        stats = portfolio_stats(weights, mean_returns, cov_matrix, returns, rf, annual_factor)
        port_ret_series = returns.dot(weights)
        recovery_periods = calculate_recovery_time(port_ret_series)

        col1, col2 = st.columns([1.8, 2.2])

        with col1:
            names_small, values_small = compact_pie_weights(asset_names, weights, min_percent=1.0)
            fig_pie = px.pie(
                values=values_small, names=names_small,
                title=f"توزیع وزن — {name}",
                color_discrete_sequence=px.colors.sequential.Greens
            )
            fig_p.update_traces(textposition='inside', textinfo='percent+label')
            fig_p.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)

        with col2:
            st.metric("بازده سالانه پیش‌بینی‌شده", f"{stats['سالانه']['return']*100:.2f}%")
            st.metric("ریسک سالانه", f"{stats['سالانه']['vol']*100:.2f}%")
            st.metric("نسبت شارپ", f"{stats['sharpe']:.3f}")
            st.metric("نسبت سورتینو", f"{stats['sortino']:.3f}")
            st.metric("زمان ریکاوری میانگین", f"{recovery_periods:.1f} {period_name}")

# ---------------- زمان ریکاوری هر دارایی ----------------
st.markdown("---")
st.subheader("زمان بازگشت به قله (Recovery Time) تاریخی هر دارایی")

recovery_data = []
for col in returns.columns:
    rt = calculate_recovery_time(returns[col])
    recovery_data.append({"دارایی": col, "زمان ریکاوری (دوره)": rt, "تقریباً": f"{rt:.1f} {period_name}" if rt > 0 else "بدون افت جدی"})

rec_df = pd.DataFrame(recovery_data)
st.dataframe(rec_df.sort_values("زمان ریکاوری (دوره)", ascending=False), use_container_width=True)

st.success("تحلیل کامل شد! حالا می‌توانید پرتفوی بهینه خود را انتخاب کنید.")
