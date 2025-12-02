# ابزار تحلیل پرتفو با حذف دارایی‌ها، بیمه دارایی، نمایش Drawdown/RecoveryTime باواحد صحیح و کامنت‌گذاری کامل
# نویسنده: mohammadmarghzari + Copilot کامل

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import scipy.optimize as sco

# ======= استایل فونت + دکمه حذف
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: "Vazirmatn", "IRANYekan", "Tahoma", sans-serif !important;
    }
    .asset-delete-btn {
        color: #fff !important;
        background: #d35400 !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        margin-bottom: 7px !important;
        border: none !important;
        padding: 5px 25px !important;
        transition: background 0.2s;
    }
    .asset-delete-btn:hover {
        background: #ea7832 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Utils -------------------
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
        return False, "💡 مجموع حداقل وزن دارایی‌ها بیشتر از ۱۰۰٪ است!"
    if max_total < 0.99:
        return False, "💡 مجموع حداکثر وزن دارایی‌ها کمتر از ۱۰۰٪ است! ممکن است به خطا منتهی شود."
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

# خلاصه واحد زمانی
def get_time_unit_and_format(period, freq):
    if period == "ماهانه" or freq == "M":
        return "ماه", "%Y-%m"
    elif period == "سه‌ماهه" or freq == "Q":
        return "سه‌ماهه", "%Y-%m"
    elif period == "شش‌ماهه" or freq == "2Q":
        return "شش‌ماهه", "%Y-%m"
    else:
        return "روز", "%Y-%m-%d"

# تابع محاسبه Drawdown و Recovery (همراه تاریخ شروع و پایان recovery)
def calculate_drawdown_recovery(df, period_unit):
    df = df.sort_values("Date").reset_index(drop=True)
    prices = df['Price'].values
    dates = pd.to_datetime(df['Date']).values
    peak = prices[0]
    recovery_infos = [] # هر ریکاوری: (index_start, index_min, index_end)
    max_drawdown_info = None
    i = 0
    while i < len(prices):
        if prices[i] >= peak:
            peak = prices[i]
            peak_idx = i
            i += 1
            continue
        drawdown_start_idx = i - 1
        drawdown_start_date = dates[drawdown_start_idx]
        min_price = prices[i]
        min_idx = i
        while i < len(prices) and prices[i] < peak:
            if prices[i] < min_price:
                min_price = prices[i]
                min_idx = i
            i += 1
        # حالا i اولین جایی است که قیمت ≥ peak مجدد (یا انتهای دیتا)
        if i < len(prices):  # یعنی ریکاوری کامل شد
            recovery_end_idx = i
            duration = recovery_end_idx - drawdown_start_idx
            drawdown = (peak - min_price) / peak
            recovery_infos.append({
                "start_idx": drawdown_start_idx,
                "start_date": dates[drawdown_start_idx],
                "min_idx": min_idx,
                "min_date": dates[min_idx],
                "end_idx": recovery_end_idx,
                "end_date": dates[recovery_end_idx],
                "duration": duration,
                "drawdown": drawdown
            })
            if max_drawdown_info is None or drawdown > max_drawdown_info['drawdown']:
                max_drawdown_info = recovery_infos[-1]
    return recovery_infos, max_drawdown_info

# تبدیل بازه تاریخ به متن فارسی با واحد دوره
def pretty_time_period(start, end, duration, unit):
    return f"""<b>{duration} {unit}</b> (<span style='color:#0097e6'>{start}</span> تا <span style='color:#0097e6'>{end}</span>)"""

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

# ================== Streamlit SECTION ==================
st.set_page_config(page_title="تحلیل پرتفو با سبک‌های مختلف", layout="wide")
st.markdown("<h1 style='font-family:Vazirmatn; color: #2980b9;'>ابزار تحلیل پرتفو، بازیابی، بیمه و مدیریت حذف دارایی</h1>", unsafe_allow_html=True)

# -- رییسک رفتاری
st.sidebar.markdown("## 🎯 تست رفتار ریسک")
with st.sidebar.expander("تست سنجش رفتار ریسک"):
    q1 = st.radio("اگر ارزش پرتفو شما موقتاً ۱۵٪ کاهش یابد…", ["سریع می‌فروشم", "نگه می‌دارم", "خرید می‌کنم"], key="risk_q1")
    q2 = st.radio("در سرمایه‌گذاری پرریسک با بازده بالا چه احساسی دارید؟", ["نگران", "بی‌تفاوت", "هیجان‌زده"], key="risk_q2")
    q3 = st.radio("کدام جمله به شما نزدیک‌تر است؟", [
        "سود کم ولی قطعی داشته باشم",
        "سود متوسط با کمی ریسک را می‌پذیرم",
        "پتانسیل سود بالا مهم‌تر است"
    ], key="risk_q3")
    q4 = st.radio("در گذشته اگر ضرر قابل توجه داشتی…", [
        "عقب‌نشینی کردم",
        "تحمل و صبر کردم",
        "دوباره ورود کردم"
    ], key="risk_q4")
    q1_map = {"سریع می‌فروشم": 1, "نگه می‌دارم": 2, "خرید می‌کنم": 3}
    q2_map = {"نگران": 1, "بی‌تفاوت": 2, "هیجان‌زده": 3}
    q3_map = {
        "سود کم ولی قطعی داشته باشم": 1,
        "سود متوسط با کمی ریسک را می‌پذیرم": 2,
        "پتانسیل سود بالا مهم‌تر است": 3
    }
    q4_map = {
        "عقب‌نشینی کردم": 1,
        "تحمل و صبر کردم": 2,
        "دوباره ورود کردم": 3
    }
    if st.button("ثبت نتیجه تست", key="submit_risk_test"):
        risk_score = q1_map[q1] + q2_map[q2] + q3_map[q3] + q4_map[q4]
        if risk_score <= 6:
            risk_profile = "محافظه‌کار"
            risk_value = 0.10
        elif risk_score <= 9:
            risk_profile = "متعادل"
            risk_value = 0.25
        else:
            risk_profile = "تهاجمی"
            risk_value = 0.40
        msg(f"پروفایل ریسک شما: **{risk_profile}**", 'success')
        st.session_state["risk_profile"] = risk_profile
        st.session_state["risk_value"] = risk_value

if "risk_profile" not in st.session_state or "risk_value" not in st.session_state:
    st.warning("⚠️ تست ریسک را کامل کنید.")
    st.stop()

with st.sidebar.expander("⚙️ تنظیمات کلی"):
    period = st.selectbox("بازه تحلیل", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
    rf = st.number_input("نرخ بدون ریسک سالانه (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    st.markdown("---")
    total_capital = st.number_input("سرمایه کل (دلار)", 0.0, value=100000.0, step=100.0)
    capital_for_gain = st.number_input("سرمایه برای سود محاسبات (اختیاری)", 0.0, value=total_capital, step=100.0)
    n_portfolios = st.slider("تعداد پرتفو برای مونت‌کارلو", 500, 30000, 5000, 500)
    n_mc = st.slider("تعداد سیمولیشن مونت‌کارلو", 200, 4000, 800, 100)
    seed_value = st.number_input("ثابت تصادفی (seed)", 0, 99999, 42)

# ---- مدیریت دارایی (آپلود و دانلود و حذف با ظاهر زیبا)
with st.sidebar.expander("🗃️ مدیریت دارایی‌ها"):
    uploaded_files = st.file_uploader("آپلود دارایی‌ها (CSV)", type=['csv'], accept_multiple_files=True, key="uploader")
    if "deleted_assets" not in st.session_state:
        st.session_state["deleted_assets"] = set()
    deleted_assets = st.session_state["deleted_assets"]
    all_assets = []
    asset_read_errors = []
    if uploaded_files:
        for file in uploaded_files:
            asset_name = file.name.split('.')[0]
            if asset_name in deleted_assets: continue
            df, err = read_csv_file(file)
            if df is not None:
                all_assets.append((asset_name, df))
            else:
                asset_read_errors.append(f"{file.name}: {err}")
    if "downloaded_dfs" not in st.session_state:
        st.session_state["downloaded_dfs"] = []
    with st.expander("دریافت آنلاین"):
        st.markdown("مثال: BTC-USD,AAPL,ETH-USD ")
        tickers_input = st.text_input("نماد دارایی‌ها")
        start = st.date_input("تاریخ شروع", value=pd.to_datetime("2023-01-01"))
        end = st.date_input("تاریخ پایان", value=pd.to_datetime("today"))
        download_btn = st.button("دریافت داده")
    if download_btn and tickers_input.strip():
        tickers = [t.strip() for t in tickers_input.strip().split(",") if t.strip()]
        try:
            data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
            if data.empty:
                msg("داده‌ای دریافت نشد!", "error")
            else:
                new_downloaded = []
                for t in tickers:
                    if t in deleted_assets: continue
                    df, err = get_price_dataframe_from_yf(data, t)
                    if df is not None and not df.empty and not df["Price"].isna().all():
                        df['Date'] = pd.to_datetime(df['Date'])
                        new_downloaded.append((t, df))
                        msg(f"داده {t} با موفقیت دانلود شد.", "success")
                    else:
                        asset_read_errors.append(f"{t}: داده دریافتی معتبر نیست یا پر از NaN است.")
                st.session_state["downloaded_dfs"].extend(new_downloaded)
        except Exception as ex:
            msg(f"خطا در دریافت داده: {ex}", "error")
    if st.session_state.get("downloaded_dfs"):
        for t, df in st.session_state["downloaded_dfs"]:
            if t not in deleted_assets:
                all_assets.append((t, df))
    st.markdown("#### <span style='color:#6091b3;font-weight:bold'>لیست دارایی و حذف:</span>", unsafe_allow_html=True)
    assets_to_remove = []
    for idx, (name, df) in enumerate(all_assets):
        col1, col2 = st.columns([6,1])
        with col1:
            st.markdown(f"<div style='font-size:15px'>{idx+1}. <b>{name}</b></div>", unsafe_allow_html=True)
        with col2:
            rm_btn = st.button(f"🗑️ حذف", key=f"remove_asset_{name}", help="حذف این دارایی", type="secondary")
            if rm_btn:
                assets_to_remove.append(name)
    if assets_to_remove:
        for name in assets_to_remove:
            deleted_assets.add(name)
        st.experimental_rerun()
    for err in asset_read_errors: msg(f"⚠️ {err}", "warning")

    # محدودیت وزن دارایی‌ها
    asset_min_weights = {}
    asset_max_weights = {}
    asset_names_show = [name for name, df in all_assets if df is not None]
    for name, df in all_assets:
        if df is None: continue
        asset_min_weights[name] = st.number_input(f"حداقل وزن {name}", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"min_weight_{name}")
        asset_max_weights[name] = st.number_input(f"حداکثر وزن {name}", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key=f"max_weight_{name}")
    if len(all_assets) > 0:
        is_valid, weights_msg = validate_weights(asset_min_weights, asset_max_weights, asset_names_show)
        if not is_valid:
            st.warning(weights_msg)

# --- رزولوشن بازه زمانی برای تحلیل و واحد زمانی
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, float(st.session_state.get("risk_value", 0.25)), 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

# =================== تحلیل پرتفو فقط اگر داده کافی باشد
if is_all_assets_valid(all_assets):
    prices_df = pd.DataFrame()
    for name, df in all_assets:
        if df is None or 'Date' not in df.columns or 'Price' not in df.columns:
            continue
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
    asset_names = list(prices_df.columns)
    if prices_df.empty or len(asset_names) == 0:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد - لطفاً داده‌های معتبر وارد کنید.")
        st.stop()
    st.subheader("🧪 پیش‌نمایش داده‌ها")
    st.dataframe(prices_df.tail(), use_container_width=True)

    # ---------------- بیمه دارایی‌ها + کاهش ریسک آنها در پرتفوی
    insured_assets = {}
    for name in asset_names:
        st.sidebar.markdown(f"---\n### ⚙️ بیمه `{name}`")
        insured = st.sidebar.checkbox(f"فعالسازی بیمه برای {name}", key=f"insured_{name}")
        if insured:
            loss_percent = st.sidebar.number_input(f"📉 درصد کاهش (حد ضرر بیمه)", 0.0, 100.0, 30.0, step=0.01, key=f"loss_{name}")
            insured_assets[name] = {'loss_percent': loss_percent}
    st.sidebar.markdown("**نکته:** بیمه ریسک هر دارایی را به نسبت درصد بیمه کاهش می‌دهد و اثر آن در ریسک نهایی محاسبه می‌شود.")

    # ---- محاسبات پرتفو و وزن‌ها با اعمال بیمه
    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_devs = np.sqrt(np.diag(cov_matrix))

    adjusted_cov = cov_matrix.copy()
    for i, name in enumerate(asset_names):
        if name in insured_assets:
            risk_scale = 1 - insured_assets[name]['loss_percent'] / 100
            adjusted_cov.iloc[i, :] *= risk_scale
            adjusted_cov.iloc[:, i] *= risk_scale
    # محاسبه وزن ترجیحی (وزن‌دهی بیشتر به دارایی با بیمه بیشتر ریسک‌کاسته)
    preference_weights = []
    for i, name in enumerate(asset_names):
        risk_scale = 1.0
        if name in insured_assets:
            risk_scale = 1 - insured_assets[name]['loss_percent'] / 100
        preference_weights.append(1 / max(std_devs[i] * risk_scale**0.7, 1e-4))
    preference_weights = np.array(preference_weights)
    preference_weights /= np.sum(preference_weights)

    np.random.seed(seed_value)
    results = np.zeros((5 + len(asset_names), n_portfolios))
    downside = returns.copy(); downside[downside > 0] = 0

    min_weights_arr = np.array([asset_min_weights.get(name, 0)/100 for name in asset_names])
    max_weights_arr = np.array([asset_max_weights.get(name, 100)/100 for name in asset_names])
    valid_minmax, _ = validate_weights(asset_min_weights, asset_max_weights, asset_names)
    if not valid_minmax:
        st.error("محدودیت‌های وزن دارایی‌ها اشتباه تعریف شده است.")
        st.stop()

    for i in range(n_portfolios):
        weights = np.random.random(len(asset_names)) * preference_weights
        weights /= np.sum(weights)
        weights = min_weights_arr + (max_weights_arr - min_weights_arr) * weights
        weights /= np.sum(weights)
        if np.sum(min_weights_arr) > 1:
            weights = min_weights_arr / np.sum(min_weights_arr)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(adjusted_cov, weights)))
        downside_risk = np.sqrt(np.dot(weights.T, np.dot(downside.cov() * annual_factor, weights)))
        sharpe_ratio = (port_return - rf/100) / (port_std if port_std!=0 else np.nan)
        sortino_ratio = (port_return - rf/100) / (downside_risk if downside_risk>0 else np.nan)

        mc_sims = np.random.multivariate_normal(mean_returns/annual_factor, adjusted_cov/annual_factor, n_mc)
        port_mc_returns = np.dot(mc_sims, weights)
        VaR = np.percentile(port_mc_returns, (1 - cvar_alpha) * 100)
        CVaR = port_mc_returns[port_mc_returns <= VaR].mean() if np.any(port_mc_returns <= VaR) else VaR

        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe_ratio
        results[3, i] = sortino_ratio
        results[4, i] = -CVaR
        results[5:, i] = weights

    best_idx = np.argmin(np.abs(results[1] - user_risk))
    best_weights = results[5:, best_idx]
    cvar_idx = np.nanargmin(results[4])
    cvar_weights = results[5:, cvar_idx]
    bounds = [(asset_min_weights.get(name,0)/100, asset_max_weights.get(name,100)/100) for name in asset_names]
    w_mvp = opt_min_variance(mean_returns, cov_matrix, bounds)
    w_sharpe = opt_max_sharpe(mean_returns, cov_matrix, rf/100, bounds)
    w_eq = equally_weighted_weights(len(asset_names))

    style_dict = {
        'مونت‌کارلو': best_weights,
        f'CVaR {int(cvar_alpha*100)}%': cvar_weights,
        'مینیمم واریانس': w_mvp if w_mvp is not None else np.zeros(len(asset_names)),
        'ماکزیمم شارپ': w_sharpe if w_sharpe is not None else np.zeros(len(asset_names)),
        'وزن برابر': w_eq
    }
    style_keys = list(style_dict.keys())
    color_map = {
        'مونت‌کارلو': '#03a678',
        f'CVaR {int(cvar_alpha*100)}%': '#d35400',
        'مینیمم واریانس': '#8e44ad',
        'ماکزیمم شارپ': '#34495e',
        'وزن برابر': "#7ed6a5"
    }

    # ----------- نمایش اطلاعات هر دارایی و ریکاوری‌تایم
    st.subheader("🔮 پیش‌بینی قیمت و ریکاوری تایم بر اساس داده و اعمال بیمه")
    unit_time, dt_format = get_time_unit_and_format(period, resample_rule)
    for i, name in enumerate(asset_names):
        last_price = prices_df[name].iloc[-1]
        st.markdown(f"<span style='font-family:Vazirmatn; font-size:20px; color:#34495e'><b>{name}</b></span>", unsafe_allow_html=True)
        # ---------- drawdown & recovery
        this_prices = prices_df[[name]].reset_index()
        this_prices = this_prices.rename(columns={name: "Price"})
        recovery_infos, max_drawdown_info = calculate_drawdown_recovery(this_prices, unit_time)
        # نمایش اطلاعات recovery
        if max_drawdown_info:
            start_cal = pd.to_datetime(max_drawdown_info['start_date']).strftime(dt_format)
            end_cal = pd.to_datetime(max_drawdown_info['end_date']).strftime(dt_format)
            st.markdown(
                f"<span style='color:#ff6f00; font-weight:500'>⏳ طولانی‌ترین بازیابی:</span> "
                f"{pretty_time_period(start_cal, end_cal, max_drawdown_info['duration'], unit_time)}<br>"
                f"💧 <b>افت:</b> <span style='color:#b71c1c'>{max_drawdown_info['drawdown']:.2%}</span>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown("<span style='color:#2d3436;font-size:15px'>هیچ ریکاوری در این بازه نیاز نبود.</span>", unsafe_allow_html=True)
        # میانگین
        if recovery_infos:
            mean_duration = np.mean([r["duration"] for r in recovery_infos])
            st.markdown(
                f"<span style='color:#009432'>🧮 <b>میانگین بازیابی:</b> {mean_duration:.1f} {unit_time}</span>", unsafe_allow_html=True
            )
        else:
            st.markdown(f"<span style='color:#00b894'>این دارایی دوره ریکاوری نداشته است.</span>", unsafe_allow_html=True)
        # توضیح بیمه:
        if name in insured_assets:
            st.markdown(f"<span style='color:#273c75'><b>این دارایی بیمه شده 👒</b> (ریسک: ×{(1-insured_assets[name]['loss_percent']/100):.2f})</span>", unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)
else:
    st.warning("⚠️ ابتدا فایل یا داده معتبر بارگذاری کنید.")
