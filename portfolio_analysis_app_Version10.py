import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import scipy.optimize as sco

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
st.sidebar.markdown("## 🧠 تست پروفایل ریسک رفتاری")
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
    st.warning("⚠️ تست ریسک را کامل کنید.")
    st.stop()

st.title("📊 ابزار تحلیل پرتفو با سبک‌های مختلف")
with st.sidebar.expander("تنظیمات کلی", expanded=True):
    period = st.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
    rf = st.number_input("نرخ بدون ریسک سالانه (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    st.markdown("---")
    st.markdown("#### :money_with_wings: سرمایه کل (دلار)")
    total_capital = st.number_input("سرمایه کل (دلار)", min_value=0.0, value=100000.0, step=100.0)
    st.markdown("#### مبلغ دلاری محاسبه بازده (اختیاری)")
    capital_for_gain = st.number_input("سرمایه برای نمایش بازده ($)", min_value=0.0, value=total_capital, step=100.0)
    st.markdown("#### تعداد پرتفوهای شبیه‌سازی")
    n_portfolios = st.slider("تعداد پرتفو برای مونت‌کارلو", 500, 30000, 7500, 500)
    st.markdown("#### تعداد سیمولیشن مونت‌کارلو")
    n_mc = st.slider("تعداد شبیه‌سازی در MC", 250, 4000, 800, 100)
    seed_value = st.number_input("ثابت تصادفی (seed)", 0, 99999, 42)

with st.sidebar.expander("محدودیت وزن دارایی‌ها :lock:", expanded=True):
    st.markdown("##### محدودیت وزن هر دارایی")
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
    with st.expander("دریافت داده آنلاین 📥"):
        st.markdown("""
        <div dir="rtl" style="text-align: right;">
        <b>راهنما:</b>
        <br>نمادها را با کاما و بدون فاصله وارد کنید (مثال: <span style="direction:ltr;display:inline-block">BTC-USD,AAPL,ETH-USD</span>)
        </div>
        """, unsafe_allow_html=True)
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
        all_assets.extend(st.session_state["downloaded_dfs"])

    for err in asset_read_errors:
        msg(f"⚠️ {err}", "warning")

    asset_min_weights = {}
    asset_max_weights = {}
    asset_names_show = [name for name, df in all_assets if df is not None]
    for name, df in all_assets:
        if df is None: continue
        asset_min_weights[name] = st.number_input(
            f"حداقل وزن {name}", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"min_weight_{name}"
        )
        asset_max_weights[name] = st.number_input(
            f"حداکثر وزن {name}", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key=f"max_weight_{name}"
        )
    if len(all_assets) > 0:
        is_valid, weights_msg = validate_weights(asset_min_weights, asset_max_weights, asset_names_show)
        if not is_valid:
            st.warning(weights_msg)

resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, float(st.session_state.get("risk_value", 0.25)), 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

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
    st.dataframe(prices_df.tail())

    insured_assets = {}
    for name in asset_names:
        st.sidebar.markdown(f"---\n### ⚙️ تنظیمات بیمه برای دارایی: `{name}`")
        insured = st.sidebar.checkbox(f"📌 فعال‌سازی بیمه برای {name}", key=f"insured_{name}")
        if insured:
            loss_percent = st.sidebar.number_input(f"📉 درصد ضرر معامله پوت برای {name}", 0.0, 100.0, 30.0, step=0.01, key=f"loss_{name}")
            strike = st.sidebar.number_input(f"🎯 قیمت اعمال پوت برای {name}", 0.0, 1e6, 100.0, step=0.01, key=f"strike_{name}")
            premium = st.sidebar.number_input(f"💰 قیمت قرارداد پوت برای {name}", 0.0, 1e6, 5.0, step=0.01, key=f"premium_{name}")
            amount = st.sidebar.number_input(f"📦 مقدار قرارداد برای {name}", 0.0, 1e6, 1.0, step=0.01, key=f"amount_{name}")
            spot_price = st.sidebar.number_input(f"📌 قیمت فعلی دارایی پایه {name}", 0.0, 1e6, 100.0, step=0.01, key=f"spot_{name}")
            asset_amount = st.sidebar.number_input(f"📦 مقدار دارایی پایه {name}", 0.0, 1e6, 1.0, step=0.01, key=f"base_{name}")
            insured_assets[name] = {
                'loss_percent': loss_percent,
                'strike': strike,
                'premium': premium,
                'amount': amount,
                'spot': spot_price,
                'base': asset_amount
            }

    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_devs = np.sqrt(np.diag(cov_matrix))

    adjusted_cov = cov_matrix.copy()
    preference_weights = []
    for i, name in enumerate(asset_names):
        if name in insured_assets:
            risk_scale = 1 - insured_assets[name]['loss_percent'] / 100
            adjusted_cov.iloc[i, :] *= risk_scale
            adjusted_cov.iloc[:, i] *= risk_scale
            preference_weights.append(1 / max(std_devs[i] * risk_scale**0.7, 1e-4))
        else:
            preference_weights.append(1 / max(std_devs[i], 1e-4))
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

    # سبک‌های پرتفو و آمار آنها
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
    min_percent_for_pie = 0.1

    # نمایش اطلاعات و pie هر سبک + سود دلاری هر سبک و هر بازه
    st.subheader(":rocket: اطلاعات سبد و نمودار دایره‌ای سبک‌ها")
    gains_table = {}
    periods = [('سالانه', 1), ('سه‌ماهه', 3/12), ('دوماهه', 2/12), ('یک‌ماهه', 1/12)]
    for style, weights in style_dict.items():
        stats = portfolio_stats(weights, mean_returns, cov_matrix, returns, rf, annual_factor)
        shown_names, shown_weights = compact_pie_weights(asset_names, weights, min_percent=min_percent_for_pie)
        fig_pie = px.pie(
            names=shown_names,
            values=shown_weights,
            title=f"توزیع وزنی دارایی‌ها ({style})",
            hole=0.3,
            color=shown_names,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_traces(textinfo='percent+label+value',
                              pull=[0.08 if n in insured_assets else 0 for n in shown_names],
                              marker=dict(line=dict(color='#222', width=2)))
        fig_pie.update_layout(font_family="Vazirmatn", title_font_size=20, height=340)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(f"##### <span style='color:{color_map[style]}'>سبک: {style}</span>", unsafe_allow_html=True)
        gain_row = []
        for (p_label, p_mult) in periods:
            exp_ret = stats[p_label]['return']
            exp_gain = exp_ret * capital_for_gain
            gain_row.append(exp_gain)
            st.write(f"📈 سود مورد انتظار {p_label}: **{exp_gain:,.0f} $**  (سود نسبی: {exp_ret*100:.2f}%)")
            st.write(f"⚠️ ریسک/نوسان {p_label}: **{stats[p_label]['vol']*100:.2f}%**")
            if p_label == 'سالانه':
                st.write(f"🔹 نسبت شارپ: {stats['sharpe']:.2f} | نسبت سورتینو: {stats['sortino']:.2f}")
        gains_table[style] = gain_row
        st.markdown("---")

    # جدول مقایسه سود دلاری سبک‌ها
    col_gains = ['سالانه', 'سه‌ماهه', 'دوماهه', 'یک‌ماهه']
    st.subheader("📋 جدول مقایسه سود دلاری سبک‌ها")
    gains_df = pd.DataFrame(gains_table, index=col_gains)
    st.dataframe(gains_df.T, use_container_width=True)

    # نمودار میله‌ای سود برای هر بازه (هر نمودار یک بازه؛ سبک‌ها کنار هم)
    st.subheader("📈 مقایسه دلاری سبک‌ها در هر بازه (Bar Chart)")
    for i, period in enumerate(col_gains):
        fig_bar = go.Figure()
        for style in style_keys:
            fig_bar.add_trace(go.Bar(
                x=[style], y=[gains_df[style][i]], name=style, marker_color=color_map[style]
            ))
        # ستونی:
        fig_bar.update_layout(
            title=f"سود دلاری {period} (برای سرمایه {capital_for_gain:,.0f} دلار)",
            yaxis_title="سود تخمینی ($)",
            showlegend=False,
            font_family="Vazirmatn",
            title_font_size=22
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # جدول مقایسه وزن‌های سبک‌ها
    st.subheader("📋 جدول مقایسه وزن دارایی‌ها")
    compare_dict = {"دارایی": asset_names}
    for style, weights in style_dict.items():
        compare_dict[style] = [w*100 for w in weights]
    df_compare = pd.DataFrame(compare_dict)
    st.dataframe(df_compare.set_index("دارایی"), use_container_width=True)

    # مرز کارا برای هر سبک: فقط جایی که معنی‌دار است (مونت‌کارلو و CVaR، و سبک‌های نقطه‌ای)
    st.subheader("🌈 مرز کارا پرتفوها (سبک‌ها)")
    for style in style_keys:
        st.markdown(f"#### مرز کارا: {style}")
        if style in ['مونت‌کارلو', f'CVaR {int(cvar_alpha*100)}%']:
            idx = best_idx if style == 'مونت‌کارلو' else cvar_idx
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results[1]*100, y=results[0]*100,
                mode='markers', marker=dict(
                    size=6, color=results[2] if style == 'مونت‌کارلو' else -results[4],
                    colorscale='Viridis' if style == 'مونت‌کارلو' else 'Blues',
                    colorbar=dict(title='Sharpe' if style=='مونت‌کارلو' else '-CVaR')
                ),
                name="پرتفوهای شبیه‌سازی‌شده"
            ))
            w = (best_weights if style=='مونت‌کارلو' else cvar_weights)
            fig.add_trace(go.Scatter(
                x=[np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))*100],
                y=[np.dot(w, mean_returns)*100], mode='markers+text',
                marker=dict(size=18, color='red' if style=='مونت‌کارلو' else 'orange', symbol='star'),
                name=f"بهینه {style}", text=["⭐"], textposition="top center"
            ))
            fig.update_layout(
                title=f"مرز کارا - {style}",
                xaxis_title="ریسک سالانه (%)", yaxis_title="بازده سالانه (%)",
                font_family="Vazirmatn", title_font_size=22
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # مینیمم واریانس - ماکزیمم شارپ - وزن برابر، نقطه‌ای نمایش بده
            w = style_dict[style]
            port_return = np.dot(w, mean_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[port_vol*100], y=[port_return*100],
                mode='markers+text',
                marker=dict(size=18, color=color_map[style], symbol='star'),
                name=f'بهینه {style}',
                text=["⭐"], textposition="top center"
            ))
            fig.update_layout(
                title=f"مرز کارا - {style}",
                xaxis_title="ریسک سالانه (%)", yaxis_title="بازده سالانه (%)",
                font_family="Vazirmatn", title_font_size=22
            )
            st.plotly_chart(fig, use_container_width=True)

    # پیش‌بینی قیمت برای 3/2/1 ماهه برای هر دارایی
    st.subheader("🔮 پیش‌بینی قیمت و بازده آتی هر دارایی")
    prediction_periods = [("سه‌ماهه (۳ ماه)", 3), ("دو ماهه", 2), ("یک ماهه", 1)]
    for i, name in enumerate(asset_names):
        last_price = resampled_prices[name].iloc[-1]
        mu = mean_returns[i] / annual_factor
        sigma = std_devs[i] / np.sqrt(annual_factor)
        if sigma < 1e-4:
            st.info(f"برای {name} واریانس داده‌ها بسیار کم است و پیش‌بینی معناداری نمی‌توان ارائه داد.")
            continue
        st.markdown(f"#### {name}")
        cols = st.columns(len(prediction_periods))
        for j, (label, future_months) in enumerate(prediction_periods):
            sim_prices = []
            n_sim = 500
            for _ in range(n_sim):
                sim = last_price * np.exp(np.cumsum(np.random.normal(mu, sigma, future_months)))
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
                    title_font_size=16,
                    height=270
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                st.markdown(f"📈 **میانگین:** `{future_price_mean:.2f}`")
                st.markdown(f"📊 **بازده:** `{future_return:.2%}`")
        st.markdown("---")

else:
    st.warning("⚠️ لطفاً فایل‌های CSV معتبر شامل ستون‌های Date و Price را آپلود کنید یا داده آنلاین وارد نمایید.")