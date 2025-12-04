import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import scipy.optimize as sco

# ------------------ Utils -------------------
def calculate_recovery_time(returns_series):
    """محاسبه میانگین تعداد دوره‌های لازم برای بازگشت به قله قبلی (Recovery Time)"""
    if len(returns_series) < 2:
        return 0
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = cum_returns / running_max - 1
    in_drawdown = False
    recovery_times = []
    start_idx = None
    
    for i in range(1, len(cum_returns)):
        if drawdowns.iloc[i] < 0:  # در حال ضرر هستیم
            if not in_drawdown:
                in_drawdown = True
                start_idx = i
        else:
            if in_drawdown:
                in_drawdown = False
                recovery_time = i - start_idx
                recovery_times.append(recovery_time)
    
    # اگر هنوز در دراودان باشیم، آن را نادیده بگیر
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

def validate_weights(min_weights, max_weights, asset_names):
    min_total = np.sum([min_weights.get(name, 0)/100 for name in asset_names])
    max_total = np.sum([max_weights.get(name, 100)/100 for name in asset_names])
    if min_total > 1.0:
        return False, "مجموع حداقل وزن دارایی‌ها بیشتر از ۱۰۰٪ است!"
    if max_total < 0.99:
        return False, "مجموع حداکثر وزن دارایی‌ها کمتر از ۱۰۰٪ است! ممکن است به خطا منتهی شود."
    return True, ""

def is_all_assets_valid(all_assets):
    valid_names = [
        name for name, df in all_assets
        if df is not None
        and 'Date' in df.columns
        and 'Price' in df.columns
        and (~df['Price'].isna()).sum() > 0
    ]
    return len(valid_names) > 0

def msg(msg, level="warning"):
    if level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    elif level == "info":
        st.info(msg)
    else:
        st.success(msg)

def compact_pie_weights(asset_names, weights, min_percent=0.1):
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

def opt_min_variance(mean_returns, cov_matrix, bounds):
    n = len(mean_returns)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    init_guess = np.ones(n)/n
    result = sco.minimize(
        lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=cons
    )
    return result.x if result.success else None

def opt_max_sharpe(mean_returns, cov_matrix, rf, bounds):
    n = len(mean_returns)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    def neg_sharpe(w):
        port_ret = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -((port_ret - rf) / port_vol) if port_vol != 0 else 0
    init_guess = np.ones(n)/n
    result = sco.minimize(
        neg_sharpe,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=cons
    )
    return result.x if result.success else None

def equally_weighted_weights(n):
    return np.ones(n) / n

def portfolio_stats(weights, mean_returns, cov_matrix, returns, rf, annual_factor):
    mean_m = mean_returns / annual_factor
    cov_m = cov_matrix / annual_factor

    port_ann_return = np.dot(weights, mean_returns)
    port_ann_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    downrets = returns.copy(); downrets[downrets > 0] = 0
    port_ann_downstd = np.sqrt(np.dot(weights.T, np.dot(downrets.cov()*annual_factor, weights)))
    sharpe = (port_ann_return - rf/100) / (port_ann_vol if port_ann_vol else np.nan)
    sortino = (port_ann_return - rf/100) / (port_ann_downstd if port_ann_downstd else np.nan)

    stats = {}
    for label, period in [('سالانه', annual_factor), ('سه‌ماهه', 3), ('دوماهه', 2), ('یک‌ماهه', 1)]:
        mu = np.dot(weights, mean_m)
        sigma = np.sqrt(np.dot(weights, np.dot(cov_m, weights)))
        port_return = mu * period
        port_vol = sigma * np.sqrt(period)
        stats[label] = {"return": port_return, "vol": port_vol}
    stats['sharpe'] = sharpe; stats['sortino'] = sortino
    return stats

# ---------------- Streamlit App -----------------
st.set_page_config(page_title="تحلیل پرتفو با سبک‌های مختلف", layout="wide")
st.sidebar.markdown("## تست پروفایل ریسک رفتاری")
with st.sidebar.expander("انجام تست ریسک رفتاری", expanded=True):
    q1 = st.radio("اگر ارزش پرتفو شما به طور موقت ۱۵٪ کاهش یابد، چه کار می‌کنید؟", ["سریع می‌فروشم", "نگه می‌دارم", "خرید می‌کنم"], key="risk_q1")
    q2 = st.radio("در یک سرمایه‌گذاری پرریسک با بازده بالا، چه احساسی دارید؟", ["نگران", "بی‌تفاوت", "هیجان‌زده"], key="risk_q2")
    q3 = st.radio("کدام جمله به شما نزدیک‌تر است؟", [
        "ترجیح می‌دهم سود کم ولی قطعی داشته باشم",
        "سود متوسط ولی با کمی ریسک را می‌پذیرم",
        "پتانسیل سود بالا مهم‌تر از ریسک است"
    ], key="risk_q3")
    q4 = st.radio("در گذشته اگر ضرر قابل توجهی کردید، چه واکنشی داشتید؟", [
        "کاملاً عقب نشینی کردم",
        "تحمل کردم و صبر کردم",
        "با تحلیل دوباره وارد شدم"
    ], key="risk_q4")
    q1_map = {"سریع می‌فروشم": 1, "نگه می‌دارم": 2, "خرید می‌کنم": 3}
    q2_map = {"نگران": 1, "بی‌تفاوت": 2, "هیجان‌زده": 3}
    q3_map = {
        "ترجیح می‌دهم سود کم ولی قطعی داشته باشم": 1,
        "سود متوسط ولی با کمی ریسک را می‌پذیرم": 2,
        "پتانسیل سود بالا مهم‌تر از ریسک است": 3
    }
    q4_map = {
        "کاملاً عقب نشینی کردم": 1,
        "تحمل کردم و صبر کردم": 2,
        "با تحلیل دوباره وارد شدم": 3
    }
    if st.button("ثبت نتیجه تست ریسک رفتاری", key="submit_risk_test"):
        risk_score = q1_map[q1] + q2_map[q2] + q3_map[q3] + q4_map[q4]
        if risk_score <= 6:
            risk_profile = "محافظه‌کار (Conservative)"
            risk_value = 0.10
        elif risk_score <= 9:
            risk_profile = "متعادل (Moderate)"
            risk_value = 0.25
        else:
            risk_profile = "تهاجمی (Aggressive)"
            risk_value = 0.40
        msg(f"پروفایل ریسک شما: **{risk_profile}**", 'success')
        st.session_state["risk_profile"] = risk_profile
        st.session_state["risk_value"] = risk_value

if "risk_profile" not in st.session_state or "risk_value" not in st.session_state:
    st.warning("تست ریسک را کامل کنید.")
    st.stop()

st.title("ابزار تحلیل پرتفو با سبک‌های مختلف + زمان ریکاوری")
with st.sidebar.expander("تنظیمات کلی", expanded=True):
    period = st.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
    rf = st.number_input("نرخ بدون ریسک سالانه (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    st.markdown("---")
    total_capital = st.number_input("سرمایه کل (دلار)", min_value=0.0, value=100000.0, step=100.0)
    capital_for_gain = st.number_input("سرمایه برای نمایش بازده ($)", min_value=0.0, value=total_capital, step=100.0)
    n_portfolios = st.slider("تعداد پرتفو برای مونت‌کارلو", 500, 30000, 7500, 500)
    n_mc = st.slider("تعداد شبیه‌سازی در MC", 250, 4000, 800, 100)
    seed_value = st.number_input("ثابت تصادفی (seed)", 0, 99999, 42)

with st.sidebar.expander("محدودیت وزن دارایی‌ها", expanded=True):
    uploaded_files = st.file_uploader("چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True, key="uploader")
    all_assets = []
    asset_read_errors = []
    if uploaded_files:
        for file in uploaded_files:
            df, err = read_csv_file(file)
            if df is not None:
                all_assets.append((file.name.split('.')[0], df))
            else:
                asset_read_errors.append(f"{file.name}: {err}")
    if "downloaded_dfs" not in st.session_state:
        st.session_state["downloaded_dfs"] = []
    with st.expander("دریافت داده آنلاین"):
        st.markdown("نمادها را با کاما و بدون فاصله وارد کنید (مثال: BTC-USD,AAPL,ETH-USD)")
        tickers_input = st.text_input("نماد دارایی‌ها")
        start = st.date_input("تاریخ شروع", value=pd.to_datetime("2023-01-01"))
        end = st.date_input("تاریخ پایان", value=pd.to_datetime("today"))
        download_btn = st.button("دریافت داده")
    if download_btn and tickers_input.strip():
        tickers = [t.strip() for t in tickers_input.strip().split(",") if t.strip()]
        try:
            data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
            if not data.empty:
                for t in tickers:
                    df, err = get_price_dataframe_from_yf(data, t)
                    if df is not None and not df.empty:
                        df['Date'] = pd.to_datetime(df['Date'])
                        all_assets.append((t, df))
                        st.session_state["downloaded_dfs"].append((t, df))
                        msg(f"داده {t} با موفقیت دانلود شد.", "success")
        except Exception as ex:
            msg(f"خطا در دریافت داده: {ex}", "error")

    asset_min_weights = {}
    asset_max_weights = {}
    for name, df in all_assets:
        if df is None: continue
        asset_min_weights[name] = st.number_input(f"حداقل وزن {name}", 0.0, 100.0, 0.0, step=1.0, key=f"min_{name}")
        asset_max_weights[name] = st.number_input(f"حداکثر وزن {name}", 0.0, 100.0, 100.0, step=1.0, key=f"max_{name}")

resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, float(st.session_state.get("risk_value", 0.25)), 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

if is_all_assets_valid(all_assets):
    prices_df = pd.DataFrame()
    for name, df in all_assets:
        if df is None: continue
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce')
        df = df.dropna(subset=['Date', 'Price']).set_index('Date')[['Price']]
        df.columns = [name]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
    asset_names = list(prices_df.columns)

    if prices_df.empty:
        st.error("داده معتبری یافت نشد.")
        st.stop()

    st.subheader("پیش‌نمایش داده‌ها")
    st.dataframe(prices_df.tail())

    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_devs = np.sqrt(np.diag(cov_matrix))

    # محاسبه Recovery Time برای هر دارایی (بر اساس بازه انتخابی کاربر)
    recovery_times = {}
    period_name = {"M": "ماه", "Q": "سه‌ماه", "2Q": "شش‌ماه"}[resample_rule]
    for name in asset_names:
        asset_ret = returns[name]
        rt = calculate_recovery_time(asset_ret)
        recovery_times[name] = rt

    # ادامه کد مونت‌کارلو و بهینه‌سازی (بدون تغییر)
    preference_weights = np.array([1 / max(std_devs[i], 1e-4) for i in range(len(asset_names))])
    preference_weights /= preference_weights.sum()

    np.random.seed(seed_value)
    results = np.zeros((5 + len(asset_names), n_portfolios))
    min_weights_arr = np.array([asset_min_weights.get(n, 0)/100 for n in asset_names])
    max_weights_arr = np.array([asset_max_weights.get(n, 100)/100 for n in asset_names])

    for i in range(n_portfolios):
        w = np.random.random(len(asset_names))
        w = w * preference_weights
        w /= w.sum()
        w = min_weights_arr + (max_weights_arr - min_weights_arr) * w
        w /= w.sum()

        ret = np.dot(w, mean_returns)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe = (ret - rf/100) / vol if vol > 0 else 0

        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe
        results[5:, i] = w

    best_idx = np.argmin(np.abs(results[1] - user_risk))
    best_weights = results[5:, best_idx]

    style_dict = {
        'مونت‌کارلو (هدف ریسک)': best_weights,
        'وزن برابر': np.ones(len(asset_names)) / len(asset_names)
    }
    color_map = {'مونت‌کارلو (هدف ریسک)': '#03a678', 'وزن برابر': '#7ed6a5'}

    # نمایش نتایج هر سبک + زمان ریکاوری پرتفو
    st.subheader("نتایج پرتفوهای بهینه + زمان ریکاوری")
    for style, weights in style_dict.items():
        stats = portfolio_stats(weights, mean_returns, cov_matrix, returns, rf, annual_factor)
        port_returns = returns.dot(weights)
        port_recovery = calculate_recovery_time(port_returns)

        st.markdown(f"#### <span style='color:{color_map[style]}'>{style}</span>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(values=weights*100, names=asset_names, title="توزیع وزنی")
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.write(f"**بازده سالانه:** {stats['سالانه']['return']*100:.2f}%")
            st.write(f"**ریسک سالانه:** {stats['سالانه']['vol']*100:.2f}%")
            st.write(f"**نسبت شارپ:** {stats['sharpe']:.2f}")
            st.write(f"**زمان ریکاوری پرتفو:** {port_recovery:.1f} {period_name}")

    # نمایش زمان ریکاوری هر دارایی به صورت جداگانه
    st.subheader("زمان ریکاوری تاریخی هر دارایی")
    recovery_df = pd.DataFrame({
        "دارایی": asset_names,
        "میانگین زمان ریکاوری (دوره)": [recovery_times[name] for name in asset_names]
    })
    recovery_df["دوره"] = period_name
    recovery_df["زمان تقریبی"] = recovery_df["میانگین زمان ریکاوری (دوره)"].apply(
        lambda x: f"{x:.1f} {period_name}" if x > 0 else "بدون دراودان قابل توجه"
    )
    st.dataframe(recovery_df[["دارایی", "زمان تقریبی"]], use_container_width=True)

else:
    st.warning("لطفاً داده‌های معتبر وارد کنید.")
