# ابزار تحلیلی پرتفوی با حذف دارایی، نمایش drawdown/recovery و کامنت کامل + توضیحات کاربر
# نویسنده: mohammadmarghzari و Copilot

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import scipy.optimize as sco

# ======= تنظیمات فونت کلی Streamlit با CSS (Vazirmatn اگر روی سیستم یا هاست هست) =======
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: "Vazirmatn", "IranYekan", "Tahoma", sans-serif !important;
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

# ============================
# [بخش ۱] توابع کمکی (Utils) (عین قبل + یک تابع helper جدید برای توصیف بازه زمانی)
# ============================
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

# ============================
# [بخش ۲] محاسبه بازه‌های ریکاوری و Drawdown همراه تاریخ دقیق شروع و پایان
# ============================
def calculate_drawdown_recovery(df):
    df = df.sort_values("Date").reset_index(drop=True)
    prices = df['Price'].values
    dates = df['Date'].values
    peak = prices[0]
    recovery_infos = [] # هر عضو: (تاریخ شروع, تاریخ پایان, مدت زمان, مقدار drawdown)
    max_drawdown_info = None
    i = 0
    while i < len(prices):
        if prices[i] >= peak:
            peak = prices[i]
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
        recovery_end_idx = i-1
        if i < len(prices):  # ریکاوری انجام شده
            recovery_time = i - drawdown_start_idx - 1
            drawdown = (peak - min_price) / peak
            recovery_infos.append({
                "start_idx": drawdown_start_idx,
                "end_idx": i-1,
                "start_date": str(pd.to_datetime(drawdown_start_date).date()),
                "end_date": str(pd.to_datetime(dates[i-1]).date()),
                "duration": recovery_time,
                "drawdown": drawdown,
                "min_idx": min_idx,
            })
            if max_drawdown_info is None or drawdown > max_drawdown_info["drawdown"]:
                max_drawdown_info = recovery_infos[-1]
    return recovery_infos, max_drawdown_info

# فرمت بازه زمانی به متن فارسی زیبا
def pretty_time_period(start, end, duration, unit):
    return f"""<span style="font-weight:bold">{duration} {unit}</span> &nbsp;از <span style='color:#0097e6'>{start}</span> تا <span style='color:#0097e6'>{end}</span>"""

# ============================
# [بخش ۳] رابط کاربری Streamlit (کامنت/راهنمای کامل)
# ============================
st.set_page_config(page_title="تحلیل پرتفو با سبک‌های مختلف", layout="wide")
st.markdown("<h1 style='font-family:Vazirmatn; color: #2980b9;'>ابزار تحلیل پرتفو و مدیریت دارایی</h1>", unsafe_allow_html=True)

# ---------- تست ریسک رفتاری
st.sidebar.markdown("## 🎯 تست رفتار ریسک کاربر")
st.sidebar.info("با انجام تست ریسک، میزان تمایل شما به ریسک مشخص شده و سبک بهینه برای پرتفوی پیشنهاد می‌شود.")
with st.sidebar.expander("تست ریسک رفتاری"):
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

# ---------- تنظیمات کلی ابزار
with st.sidebar.expander("⚙️ تنظیمات کلی"):
    st.markdown("نوع بازه زمانی تحلیل (ماهانه/سه‌ماهه/شش‌ماهه) و سرمایه را انتخاب نمایید.")
    period = st.selectbox("بازه تحلیل", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
    rf = st.number_input("نرخ بدون ریسک سالانه (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    st.markdown("---")
    total_capital = st.number_input("سرمایه کل (دلار)", 0.0, value=100000.0, step=100.0)
    capital_for_gain = st.number_input("سرمایه برای سود محاسبات (اختیاری)", 0.0, value=total_capital, step=100.0)
    n_portfolios = st.slider("تعداد پرتفو برای مونت‌کارلو", 500, 30000, 5000, 500)
    n_mc = st.slider("تعداد سیمولیشن مونت‌کارلو", 200, 4000, 800, 100)
    seed_value = st.number_input("ثابت تصادفی (seed)", 0, 99999, 42)

# ---------- مدیریت دارایی‌ها (آپلود و دانلود و حذف)
with st.sidebar.expander("🗃️ مدیریت دارایی‌ها"):
    st.markdown("آپلود فایل CSV هر دارایی (ستون date, price) یا بارگذاری آنلاین از یاهوفایننس.")
    uploaded_files = st.file_uploader("آپلود دارایی‌ها", type=['csv'], accept_multiple_files=True, key="uploader")
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
    with st.expander("دریافت داده آنلاین"):
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
    # --- زیباتر کردن حذف دارایی با المان HTML ---
    st.markdown("#### <span style='color:#6091b3;font-weight:bold'>🔎 لیست دارایی‌ها و حذف هرکدام:</span>", unsafe_allow_html=True)
    assets_to_remove = []
    for idx, (name, df) in enumerate(all_assets):
        col1, col2 = st.columns([5,1])
        with col1:
            st.markdown(f"<div style='font-size:15px'>{idx+1}. <b>{name}</b></div>", unsafe_allow_html=True)
        with col2:
            # دکمه حذف با سبک زیبا و ایموجی سطل آشغال 
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

# --- سایر پارامترهای تحلیلی
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, float(st.session_state.get("risk_value", 0.25)), 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

# =================== تحلیل نهایی اگر داده معتبر وجود داشته باشد ===================
if is_all_assets_valid(all_assets):
    st.markdown("<h3 style='color:#0a3d62;'>🧪 پیش‌نمایش داده‌ها</h3>", unsafe_allow_html=True)
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
    st.dataframe(prices_df.tail(), use_container_width=True)

    # ... (ادامه همه تحلیل‌ها مثل قبل) ...

    # تحلیل شبیه‌سازی پرتفو و ... (تا قبل از نمایش drawdown/recovery مانند قبل)
    # ...
    # نمایش پیش‌بینی و بازیابی برای هر دارایی:
    st.subheader("🔮 پیش‌بینی قیمت و جزئیات ریکاوری تایم/افت برای هر دارایی")
    st.markdown("**در هر دارایی، بازه زمانی و مدت دقیق طولانی‌ترین بازیابی پس از افت و میانگین ریکاوری‌ها به واحد زمانی خود داده نمایش داده می‌شود.**")
    prediction_periods = [("سه‌ماهه (۳ ماه)", 3), ("دو ماهه", 2), ("یک ماهه", 1)]
    time_unit, dt_format = get_time_unit_and_format(period, resample_rule)
    for i, name in enumerate(asset_names):
        last_price = prices_df[name].iloc[-1]
        mu = (prices_df[name].pct_change().dropna().mean() * annual_factor)
        sigma = (prices_df[name].pct_change().dropna().std() * np.sqrt(annual_factor))
        st.markdown(f"<span style='font-family:Vazirmatn; font-size:20px; color:#34495e'><b>{name}</b></span>", unsafe_allow_html=True)
        # بخش پیش‌بینی معمول
        cols = st.columns(len(prediction_periods))
        for j, (label, future_months) in enumerate(prediction_periods):
            sim_prices = []
            n_sim = 400
            for _ in range(n_sim):
                sim = last_price * np.exp(np.cumsum(np.random.normal(mu/annual_factor, sigma/np.sqrt(annual_factor), future_months)))
                sim_prices.append(sim[-1])
            sim_prices = np.array(sim_prices)
            future_price_mean = np.mean(sim_prices)
            future_return = (future_price_mean - last_price) / last_price
            with cols[j]:
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Histogram(x=sim_prices, nbinsx=20, name="پیش‌بینی قیمت", marker_color='purple'))
                fig_pred.add_vline(x=future_price_mean, line_dash="dash", line_color="green")
                fig_pred.update_layout(
                    title=f"{label}",
                    xaxis_title="قیمت انتهایی", 
                    yaxis_title="تعداد شبیه‌سازی", 
                    font_family="Vazirmatn", 
                    title_font_size=15, height=220
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                st.markdown(f"<span style='color:#148f77;font-weight:bold;'>میانگین:</span> <span style='font-size:16px'>{future_price_mean:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#884ea0;font-weight:bold;'>بازده:</span> <span style='font-size:16px'>{future_return:.2%}</span>", unsafe_allow_html=True)
        # نمایش details ریکاوری تایم و بیشترین افت
        this_prices = prices_df[[name]].reset_index()
        this_prices = this_prices.rename(columns={name: "Price"})
        recovery_infos, max_drawdown_info = calculate_drawdown_recovery(this_prices)
        # بزرگترین drawdown (و بازیابی) با جزییات تاریخ/مدت
        if max_drawdown_info:
            st.markdown(
                f"""<div style='margin-top:10px'>
                    <span style='color:#ff6f00; font-weight:500'>⏳ طولانی‌ترین زمان بازیابی قیمت :</span>
                    {pretty_time_period(
                        pd.to_datetime(max_drawdown_info['start_date']).strftime(dt_format),
                        pd.to_datetime(max_drawdown_info['end_date']).strftime(dt_format),
                        max_drawdown_info['duration'],
                        time_unit
                    )}
                </div>
                <div style='margin-bottom:3px'><span style='color:#7f8c8d'>از افت <b>{max_drawdown_info['drawdown']:.2%}</b> (سقف تا کف) طی این بازه</span></div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown("<div style='color:#2d3436;margin:5px 0;font-size:15px'>در این بازه داده نیاز به ریکاوری مشاهده نشد.</div>", unsafe_allow_html=True)
        # میانگین همه ریکاوری‌ها
        if recovery_infos:
            mean_duration = np.mean([r["duration"] for r in recovery_infos])
            st.markdown(
                f"<div style='color:#00b894'>🧮 میانگین زمان ریکاوری: <b>{mean_duration:.1f} {time_unit}</b></div>",
                unsafe_allow_html=True
            )
        st.markdown("---", unsafe_allow_html=True)
else:
    st.warning("⚠️ ابتدا فایل یا داده معتبر بارگذاری کنید.")
