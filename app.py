import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# --------- تابع رفع خطا و تبدیل داده yfinance به DataFrame مناسب ---------
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
# -------------------------------------------------------------------------

# ------------- بخش تست پروفایل ریسک رفتاری (Behavioral Risk Profile) -------------
st.sidebar.markdown("## 🧠 تست پروفایل ریسک رفتاری")
with st.sidebar.expander("انجام تست ریسک رفتاری", expanded=False):
    st.write("به چند سؤال رفتاری پاسخ دهید تا پروفایل ریسک شما مشخص شود:")

    q1 = st.radio(
        "اگر ارزش پرتفو شما به طور موقت ۱۵٪ کاهش یابد، چه کار می‌کنید؟",
        ["سریع می‌فروشم", "نگه می‌دارم", "خرید می‌کنم"], key="risk_q1"
    )
    q2 = st.radio(
        "در یک سرمایه‌گذاری پرریسک با بازده بالا، چه احساسی دارید؟",
        ["نگران", "بی‌تفاوت", "هیجان‌زده"], key="risk_q2"
    )
    q3 = st.radio(
        "کدام جمله به شما نزدیک‌تر است؟",
        [
            "ترجیح می‌دهم سود کم ولی قطعی داشته باشم", 
            "سود متوسط ولی با کمی ریسک را می‌پذیرم", 
            "پتانسیل سود بالا مهم‌تر از ریسک است"
        ], key="risk_q3"
    )
    q4 = st.radio(
        "در گذشته اگر ضرر قابل توجهی کردید، چه واکنشی داشتید؟",
        [
            "کاملاً عقب نشینی کردم", 
            "تحمل کردم و صبر کردم", 
            "با تحلیل دوباره وارد شدم"
        ], key="risk_q4"
    )

    risk_score = 0
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

    if st.button("ثبت نتیجه تست ریسک رفتاری"):
        risk_score = q1_map[q1] + q2_map[q2] + q3_map[q3] + q4_map[q4]
        if risk_score <= 6:
            risk_profile = "محافظه‌کار (Conservative)"
            risk_desc = "شما ریسک‌گریز هستید، سرمایه‌گذاری کم‌ریسک برای شما مناسب‌تر است."
            risk_value = 0.10
        elif risk_score <= 9:
            risk_profile = "متعادل (Moderate)"
            risk_desc = "تحمل ریسک شما متوسط است. ترکیبی از دارایی‌های با ریسک متوسط و کم توصیه می‌شود."
            risk_value = 0.25
        else:
            risk_profile = "تهاجمی (Aggressive)"
            risk_desc = "شما پذیرای ریسک بالا هستید و می‌توانید سراغ دارایی‌های پرنوسان‌تر بروید."
            risk_value = 0.40

        st.success(f"پروفایل ریسک شما: **{risk_profile}**")
        st.info(risk_desc)
        st.session_state["risk_profile"] = risk_profile
        st.session_state["risk_value"] = risk_value
    else:
        # مقدار پیش‌فرض برای زمانی که هنوز تست کامل نشده
        if "risk_profile" not in st.session_state:
            st.session_state["risk_profile"] = "متعادل (Moderate)"
            st.session_state["risk_value"] = 0.25
# -------------------------------------------------------------------------

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو، CVaR و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو، CVaR و استراتژی Married Put")

def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace('%', '')
        df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

st.sidebar.markdown("## تنظیمات کلی 	:gear:")
with st.sidebar.expander("تنظیمات کلی", expanded=True):
    period = st.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
    rf = st.number_input("نرخ بدون ریسک سالانه (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    st.markdown("---")
    st.markdown("#### :money_with_wings: سرمایه کل (دلار)")
    total_capital = st.number_input("سرمایه کل (دلار)", min_value=0.0, value=100000.0, step=100.0)
    register_btn = st.button("ثبت")

with st.sidebar.expander("محدودیت وزن دارایی‌ها :lock:", expanded=True):
    st.markdown("##### محدودیت وزن هر دارایی")
    uploaded_files = st.file_uploader(
        "چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True, key="uploader"
    )
    # داده‌های آپلود شده و دانلود شده را با هم ترکیب کن
    all_assets = []
    if uploaded_files:
        for file in uploaded_files:
            all_assets.append((file.name.split('.')[0], read_csv_file(file)))

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
                st.error("داده‌ای دریافت نشد!")
            else:
                new_downloaded = []
                for t in tickers:
                    df, err = get_price_dataframe_from_yf(data, t)
                    if df is not None:
                        df['Date'] = pd.to_datetime(df['Date'])
                        new_downloaded.append((t, df))
                        st.success(f"داده {t} با موفقیت دانلود شد.")
                    else:
                        st.error(f"{err}")
                st.session_state["downloaded_dfs"].extend(new_downloaded)
        except Exception as ex:
            st.error(f"خطا در دریافت داده: {ex}")

    if st.session_state.get("downloaded_dfs"):
        all_assets.extend(st.session_state["downloaded_dfs"])

    # حداقل و حداکثر وزن هر دارایی (درصدی)
    asset_min_weights = {}
    asset_max_weights = {}
    for name, df in all_assets:
        if df is None:
            continue
        asset_min_weights[name] = st.number_input(
            f"حداقل وزن {name}", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"min_weight_{name}"
        )
        asset_max_weights[name] = st.number_input(
            f"حداکثر وزن {name}", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key=f"max_weight_{name}"
        )

# استفاده از نتیجه تست ریسک رفتاری برای مقدار پیش‌فرض ریسک هدف پرتفو
default_risk = st.session_state.get("risk_value", 0.25)
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, float(default_risk), 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

if all_assets:
    prices_df = pd.DataFrame()
    asset_names = []
    insured_assets = {}

    for name, df in all_assets:
        if df is None:
            continue

        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"فایل {name} باید دارای ستون‌های 'Date' و 'Price' باشد.")
            continue

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]

        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

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

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    st.subheader("🧪 پیش‌نمایش داده‌ها")
    st.dataframe(prices_df.tail())

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
            preference_weights.append(1 / (std_devs[i] * risk_scale**0.7))
        else:
            preference_weights.append(1 / std_devs[i])
    preference_weights = np.array(preference_weights)
    preference_weights /= np.sum(preference_weights)

    # شبیه‌سازی مونت‌کارلو با CVaR با رعایت محدودیت‌های وزن
    n_portfolios = 10000
    n_mc = 1000
    results = np.zeros((5 + len(asset_names), n_portfolios))
    np.random.seed(42)

    downside = returns.copy()
    downside[downside > 0] = 0

    min_weights_arr = np.array([asset_min_weights.get(name, 0)/100 for name in asset_names])
    max_weights_arr = np.array([asset_max_weights.get(name, 100)/100 for name in asset_names])

    for i in range(n_portfolios):
        # تولید وزن‌های تصادفی در بازه min/max تعریف‌شده
        weights = np.random.random(len(asset_names)) * preference_weights
        weights /= np.sum(weights)
        weights = min_weights_arr + (max_weights_arr - min_weights_arr) * weights
        weights /= np.sum(weights)  # مجموع ۱ شود
        # چک: اگر مجموع minهای وارد شده >۱، نرمالایز فقط بر مبنای min نباشد
        if np.sum(min_weights_arr) > 1:
            weights = min_weights_arr / np.sum(min_weights_arr)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(adjusted_cov, weights)))
        downside_risk = np.sqrt(np.dot(weights.T, np.dot(downside.cov() * annual_factor, weights)))
        sharpe_ratio = (port_return - rf/100) / port_std
        sortino_ratio = (port_return - rf/100) / downside_risk if downside_risk > 0 else np.nan

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
    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_sortino = results[3, best_idx]
    best_weights = results[5:, best_idx]

    best_cvar_idx = np.argmin(results[4])
    best_cvar_return = results[0, best_cvar_idx]
    best_cvar_risk = results[1, best_cvar_idx]
    best_cvar_cvar = results[4, best_cvar_idx]
    best_cvar_weights = results[5:, best_cvar_idx]

    st.subheader("📈 پرتفو بهینه (مونت‌کارلو)")
    st.markdown(f"""
    - ✅ بازده سالانه: **{best_return:.2%}**
    - ⚠️ ریسک سالانه (انحراف معیار): **{best_risk:.2%}**
    - 🧠 نسبت شارپ: **{best_sharpe:.2f}**
    - 📉 نسبت سورتینو: **{best_sortino:.2f}**
    """)
    for i, name in enumerate(asset_names):
        st.markdown(f"🔹 وزن {name}: {best_weights[i]*100:.2f}%")

    # Pie Chart سبک مونت‌کارلو
    st.subheader("🥧 نمودار توزیع دارایی‌ها در پرتفو بهینه (مونت‌کارلو)")
    fig_pie_mc = px.pie(
        names=asset_names,
        values=best_weights * 100,
        title="توزیع وزنی دارایی‌ها در پرتفو بهینه (مونت‌کارلو)",
        hole=0.3
    )
    fig_pie_mc.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie_mc, use_container_width=True)

    st.subheader(f"🟢 پرتفو بهینه بر اساس CVaR ({int(cvar_alpha*100)}%)")
    st.markdown(f"""
    - ✅ بازده سالانه: **{best_cvar_return:.2%}**
    - ⚠️ ریسک سالانه (انحراف معیار): **{best_cvar_risk:.2%}**
    - 🟠 CVaR ({int(cvar_alpha*100)}%): **{best_cvar_cvar:.2%}**
    """)
    for i, name in enumerate(asset_names):
        st.markdown(f"🔸 وزن {name}: {best_cvar_weights[i]*100:.2f}%")
    # Pie Chart سبک CVaR
    st.subheader(f"🥧 نمودار توزیع دارایی‌ها در پرتفو بهینه (CVaR {int(cvar_alpha*100)}%)")
    fig_pie_cvar = px.pie(
        names=asset_names,
        values=best_cvar_weights * 100,
        title=f"توزیع وزنی دارایی‌ها در پرتفو بهینه (CVaR {int(cvar_alpha*100)}%)",
        hole=0.3
    )
    fig_pie_cvar.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie_cvar, use_container_width=True)

    st.subheader("📋 جدول مقایسه وزن دارایی‌ها (مونت‌کارلو و CVaR)")
    compare_df = pd.DataFrame({
        'دارایی': asset_names,
        'وزن مونت‌کارلو (%)': best_weights * 100,
        f'وزن CVaR ({int(cvar_alpha*100)}%) (%)': best_cvar_weights * 100
    })
    compare_df['اختلاف وزن (%)'] = compare_df[f'وزن CVaR ({int(cvar_alpha*100)}%) (%)'] - compare_df['وزن مونت‌کارلو (%)']
    st.dataframe(compare_df.set_index('دارایی'), use_container_width=True, height=300)

    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(x=asset_names, y=best_weights*100, name='مونت‌کارلو'))
    fig_w.add_trace(go.Bar(x=asset_names, y=best_cvar_weights*100, name=f'CVaR {int(cvar_alpha*100)}%'))
    fig_w.update_layout(barmode='group', title="مقایسه وزن دارایی‌ها در دو سبک")
    st.plotly_chart(fig_w, use_container_width=True)

    st.subheader("🌈 نمودار مرز کارا")
    fig = px.scatter(
        x=results[1]*100,
        y=results[0]*100,
        color=results[2],
        labels={'x': 'ریسک (%)', 'y': 'بازده (%)'},
        title='پرتفوهای شبیه‌سازی‌شده (مونت‌کارلو) و مرز CVaR',
        color_continuous_scale='Viridis'
    )
    fig.add_trace(go.Scatter(x=[best_risk*100], y=[best_return*100],
                             mode='markers', marker=dict(size=12, color='red', symbol='star'),
                             name='پرتفوی بهینه مونت‌کارلو'))
    fig.add_trace(go.Scatter(x=[best_cvar_risk*100], y=[best_cvar_return*100],
                             mode='markers', marker=dict(size=12, color='orange', symbol='star'),
                             name='پرتفوی بهینه CVaR'))
    cvar_sorted_idx = np.argsort(results[4])
    fig.add_trace(go.Scatter(
        x=results[1, cvar_sorted_idx]*100,
        y=results[0, cvar_sorted_idx]*100,
        mode='lines',
        line=dict(color='orange', dash='dot'),
        name='مرز کارا (CVaR)'
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔵 نمودار بازده- CVaR برای پرتفوها")
    fig_cvar = px.scatter(
        x=results[4], y=results[0],
        labels={'x': f'CVaR ({int(cvar_alpha*100)}%)', 'y': 'بازده'},
        title='پرتفوها بر اساس بازده و CVaR',
        color=results[1], color_continuous_scale='Blues'
    )
    fig_cvar.add_trace(go.Scatter(x=[best_cvar_cvar], y=[best_cvar_return],
                                  mode='markers', marker=dict(size=12, color='red', symbol='star'),
                                  name='پرتفوی بهینه CVaR'))
    st.plotly_chart(fig_cvar, use_container_width=True)

    st.subheader("💡 دارایی‌های پیشنهادی بر اساس نسبت بازده به ریسک")
    asset_scores = {}
    for i, name in enumerate(asset_names):
        insured_factor = 1 - insured_assets.get(name, {}).get('loss_percent', 0)/100 if name in insured_assets else 1
        score = mean_returns[i] / (std_devs[i]*insured_factor)
        asset_scores[name] = score

    sorted_assets = sorted(asset_scores.items(), key=lambda x: x[1], reverse=True)
    st.markdown("**به ترتیب اولویت:**")
    for name, score in sorted_assets:
        insured_str = " (بیمه شده)" if name in insured_assets else ""
        st.markdown(f"🔸 **{name}{insured_str}** | نسبت بازده به ریسک: {score:.2f}")

    # Married Put charts
    for name, info in insured_assets.items():
        st.subheader(f"📉 نمودار سود و زیان استراتژی Married Put - {name}")
        x = np.linspace(info['spot'] * 0.5, info['spot'] * 1.5, 200)
        asset_pnl = (x - info['spot']) * info['base']
        put_pnl = np.where(x < info['strike'], (info['strike'] - x) * info['amount'], 0) - info['premium'] * info['amount']
        total_pnl = asset_pnl + put_pnl

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x[total_pnl>=0], y=total_pnl[total_pnl>=0], mode='lines', name='سود', line=dict(color='green', width=3)
        ))
        fig2.add_trace(go.Scatter(
            x=x[total_pnl<0], y=total_pnl[total_pnl<0], mode='lines', name='زیان', line=dict(color='red', width=3)
        ))
        fig2.add_trace(go.Scatter(
            x=x, y=asset_pnl, mode='lines', name='دارایی پایه', line=dict(dash='dot', color='gray')
        ))
        fig2.add_trace(go.Scatter(
            x=x, y=put_pnl, mode='lines', name='پوت', line=dict(dash='dot', color='blue')
        ))
        zero_crossings = np.where(np.diff(np.sign(total_pnl)))[0]
        if len(zero_crossings):
            breakeven_x = x[zero_crossings[0]]
            fig2.add_trace(go.Scatter(x=[breakeven_x], y=[0], mode='markers+text', marker=dict(color='orange', size=10),
                                      text=["سر به سر"], textposition="bottom center", name='سر به سر'))
        max_pnl = np.max(total_pnl)
        max_x = x[np.argmax(total_pnl)]
        fig2.add_trace(go.Scatter(x=[max_x], y=[max_pnl], mode='markers+text', marker=dict(color='green', size=10),
                                  text=[f"{(max_pnl/(info['spot']*info['base'])*100):.1f}% سود"], textposition="top right",
                                  showlegend=False))
        fig2.update_layout(title='نمودار سود و زیان', xaxis_title='قیمت دارایی در سررسید', yaxis_title='سود/زیان')
        st.plotly_chart(fig2, use_container_width=True)

        if st.button(f"📷 ذخیره نمودار Married Put برای {name}"):
            try:
                img_bytes = fig2.to_image(format="png")
                st.download_button("دانلود تصویر", img_bytes, file_name=f"married_put_{name}.png")
            except Exception as e:
                st.error(f"❌ خطا در ذخیره تصویر: {e}")

    st.subheader("🔮 پیش‌بینی قیمت و بازده آتی هر دارایی")
    future_months = 6 if period == 'شش‌ماهه' else (3 if period == 'سه‌ماهه' else 1)
    for i, name in enumerate(asset_names):
        last_price = resampled_prices[name].iloc[-1]
        mu = mean_returns[i] / annual_factor
        sigma = std_devs[i] / np.sqrt(annual_factor)
        sim_prices = []
        n_sim = 500
        for _ in range(n_sim):
            sim = last_price * np.exp(np.cumsum(np.random.normal(mu, sigma, future_months)))
            sim_prices.append(sim[-1])
        sim_prices = np.array(sim_prices)
        future_price_mean = np.mean(sim_prices)
        future_return = (future_price_mean - last_price) / last_price

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=sim_prices, nbinsx=20, name="پیش‌بینی قیمت", marker_color='purple'))
        fig3.add_vline(x=future_price_mean, line_dash="dash", line_color="green", annotation_text=f"میانگین: {future_price_mean:.2f}")
        fig3.update_layout(title=f"پیش‌بینی قیمت {name} در {future_months} {'ماه' if future_months>1 else 'ماه'} آینده",
            xaxis_title="قیمت انتهایی", yaxis_title="تعداد شبیه‌سازی")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f"📈 **میانگین قیمت آینده:** {future_price_mean:.2f} | 📊 **درصد بازده آتی:** {future_return:.2%}")

else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید.")
