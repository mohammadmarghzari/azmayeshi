import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# ==== Behavioral Risk Profile Test ====
st.sidebar.markdown("## 🧠 تست پروفایل ریسک رفتاری")
with st.sidebar.expander("انجام تست ریسک رفتاری", expanded=True):
    st.write("به چند سؤال رفتاری پاسخ دهید تا پروفایل ریسک شما مشخص شود:")
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

if "risk_profile" not in st.session_state or "risk_value" not in st.session_state:
    st.warning("⚠️ لطفاً ابتدا تست ریسک رفتاری را کامل کنید تا دیگر ابزارها در دسترس قرار گیرند.")
    st.stop()

st.set_page_config(page_title="تحلیل پرتفو با بیمه، مونت‌کارلو، CVaR و پیش‌بینی حرفه‌ای", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش‌های بهینه‌سازی، بیمه و پیش‌بینی حرفه‌ای")

st.sidebar.markdown("## تنظیمات کلی 	:gear:")
with st.sidebar.expander("تنظیمات کلی", expanded=True):
    period = st.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
    rf = st.number_input("نرخ بدون ریسک سالانه (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
    st.markdown("---")
    st.markdown("#### :money_with_wings: سرمایه کل (دلار)")
    total_capital = st.number_input("سرمایه کل (دلار)", min_value=0.0, value=100000.0, step=100.0)
    register_btn = st.button("ثبت")

with st.sidebar.expander("ورود داده دارایی‌ها", expanded=True):
    uploaded_files = st.file_uploader(
        "چند فایل CSV آپلود کنید (هر دارایی یک فایل، دو ستون Date و Price)", type=['csv'], accept_multiple_files=True, key="uploader"
    )
    all_assets = []
    if uploaded_files:
        for file in uploaded_files:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip().str.lower()
            if set(['date', 'price']).issubset(df.columns):
                all_assets.append((file.name.split('.')[0], df.rename(columns={'date':'Date','price':'Price'})))

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
                for t in tickers:
                    if isinstance(data.columns, pd.MultiIndex):
                        price_series = data[t]['Close']
                    else:
                        price_series = data['Close']
                    df = price_series.reset_index()
                    df.columns = ['Date', 'Price']
                    all_assets.append((t, df))
                    st.success(f"داده {t} با موفقیت دانلود شد.")
        except Exception as ex:
            st.error(f"خطا در دریافت داده: {ex}")

if all_assets:
    show_methods = st.multiselect(
        "کدام سبک بهینه‌سازی پرتفو نمایش داده شود؟",
        ["MPT (مارکوویتز کلاسیک)", "مونت‌کارلو/ CVaR"],
        default=["MPT (مارکوویتز کلاسیک)"]
    )

    prices_df = pd.DataFrame()
    asset_names = []
    insured_assets = {}
    for name, df in all_assets:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

        # ---------- تنظیمات بیمه برای هر دارایی ----------
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

    resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
    annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_devs = np.sqrt(np.diag(cov_matrix))
    downside = returns.copy()
    downside[downside > 0] = 0

    user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, float(st.session_state.get("risk_value", 0.25)), 0.01)
    cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

    # ------------- MPT -------------
    if "MPT (مارکوویتز کلاسیک)" in show_methods:
        n = len(asset_names)
        bounds = tuple((0, 1) for _ in asset_names)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        x0 = np.ones(n) / n

        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
            port_return = np.dot(weights, mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - rf/100) / port_std

        from scipy.optimize import minimize
        opt = minimize(
            neg_sharpe_ratio, x0=x0, args=(mean_returns, cov_matrix, rf),
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        mpt_weights = opt.x
        mpt_return = np.dot(mpt_weights, mean_returns)
        mpt_risk = np.sqrt(np.dot(mpt_weights.T, np.dot(cov_matrix, mpt_weights)))
        mpt_sharpe = (mpt_return - rf/100) / mpt_risk

        st.subheader("📈 پرتفو بهینه به سبک MPT (مارکوویتز)")
        st.markdown(f"""
        - ✅ بازده سالانه: **{mpt_return:.2%}**
        - ⚠️ ریسک سالانه (انحراف معیار): **{mpt_risk:.2%}**
        - 🧠 نسبت شارپ: **{mpt_sharpe:.2f}**
        """)
        for i, name in enumerate(asset_names):
            st.markdown(f"🔹 وزن {name}: {mpt_weights[i]*100:.2f}%")
        fig_pie_mpt = go.Figure(data=[go.Pie(labels=asset_names, values=mpt_weights * 100, hole=.3)])
        fig_pie_mpt.update_traces(textinfo='percent+label')
        fig_pie_mpt.update_layout(title="توزیع وزنی دارایی‌ها در پرتفو بهینه (MPT)")
        st.plotly_chart(fig_pie_mpt, use_container_width=True)

    # ------------- Monte Carlo/CVaR -------------
    if "مونت‌کارلو/ CVaR" in show_methods:
        n_portfolios = 2000
        n_mc = 200
        results = np.zeros((5 + len(asset_names), n_portfolios))
        np.random.seed(42)
        for i in range(n_portfolios):
            weights = np.random.random(len(asset_names))
            weights /= np.sum(weights)
            port_return = np.dot(weights, mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            downside_risk = np.sqrt(np.dot(weights.T, np.dot(downside.cov() * annual_factor, weights)))
            sharpe_ratio = (port_return - rf/100) / port_std
            sortino_ratio = (port_return - rf/100) / downside_risk if downside_risk > 0 else np.nan
            mc_sims = np.random.multivariate_normal(mean_returns/annual_factor, cov_matrix/annual_factor, n_mc)
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
        fig_pie_mc = go.Figure(data=[go.Pie(labels=asset_names, values=best_weights * 100, hole=.3)])
        fig_pie_mc.update_traces(textinfo='percent+label')
        fig_pie_mc.update_layout(title="توزیع وزنی دارایی‌ها در پرتفو بهینه (مونت‌کارلو)")
        st.plotly_chart(fig_pie_mc, use_container_width=True)
        st.subheader(f"🟢 پرتفو بهینه بر اساس CVaR ({int(cvar_alpha*100)}%)")
        st.markdown(f"""
        - ✅ بازده سالانه: **{best_cvar_return:.2%}**
        - ⚠️ ریسک سالانه (انحراف معیار): **{best_cvar_risk:.2%}**
        - 🟠 CVaR ({int(cvar_alpha*100)}%): **{best_cvar_cvar:.2%}**
        """)
        for i, name in enumerate(asset_names):
            st.markdown(f"🔸 وزن {name}: {best_cvar_weights[i]*100:.2f}%")
        fig_pie_cvar = go.Figure(data=[go.Pie(labels=asset_names, values=best_cvar_weights * 100, hole=.3)])
        fig_pie_cvar.update_traces(textinfo='percent+label')
        fig_pie_cvar.update_layout(title=f"توزیع وزنی دارایی‌ها (CVaR {int(cvar_alpha*100)}%)")
        st.plotly_chart(fig_pie_cvar, use_container_width=True)

    # ----------- Married Put (بیمه) برای دارایی‌های انتخاب‌شده -----------
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
        fig2.update_layout(title='نمودار سود و زیان Married Put', xaxis_title='قیمت دارایی در سررسید', yaxis_title='سود/زیان')
        st.plotly_chart(fig2, use_container_width=True)

    # ----------- پیش‌بینی حرفه‌ای قیمت با ARIMA و GARCH -----------
    st.subheader("🔮 پیش‌بینی حرفه‌ای قیمت و بازده آتی هر دارایی (ARIMA و GARCH)")
    future_periods = 30
    for name in asset_names:
        st.markdown(f"**{name}**")
        price_series = prices_df[name].dropna()
        if price_series.shape[0] < 30:
            st.warning("داده کافی برای پیش‌بینی حرفه‌ای وجود ندارد (حداقل 30 نقطه لازم است).")
            continue
        # ARIMA
        try:
            model_arima = ARIMA(price_series, order=(1,1,1))
            model_arima_fit = model_arima.fit()
            forecast_arima = model_arima_fit.forecast(steps=future_periods)
            forecast_index = pd.date_range(price_series.index[-1], periods=future_periods+1, freq='D')[1:]
            forecast_arima.index = forecast_index
            st.success("پیش‌بینی ARIMA با موفقیت انجام شد.")
        except Exception as ex:
            st.error(f"خطا در مدل ARIMA: {ex}")
            continue
        # GARCH
        try:
            returns = price_series.pct_change().dropna() * 100
            garch_model = arch_model(returns, p=1, q=1)
            garch_fit = garch_model.fit(disp="off")
            garch_forecast = garch_fit.forecast(horizon=future_periods)
            forecast_vol = np.sqrt(garch_forecast.variance.values[-1, :])
            arima_last = price_series.iloc[-1]
            garch_sim = arima_last * (1 + np.cumsum(np.random.normal(0, forecast_vol/100, future_periods)))
            st.success("پیش‌بینی GARCH با موفقیت انجام شد.")
        except Exception as ex:
            st.error(f"خطا در مدل GARCH: {ex}")
            forecast_vol = None
        # نمودار
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_series.index, y=price_series, mode='lines', name='قیمت تاریخی'))
        fig.add_trace(go.Scatter(x=forecast_arima.index, y=forecast_arima, mode='lines', name='پیش‌بینی ARIMA', line=dict(color='blue')))
        if forecast_vol is not None:
            fig.add_trace(go.Scatter(x=forecast_arima.index, y=garch_sim, mode='lines', name='سناریو GARCH', line=dict(color='orange', dash='dot')))
        fig.update_layout(title=f"پیش‌بینی قیمت {name} (ARIMA و GARCH)", xaxis_title='تاریخ', yaxis_title='قیمت')
        st.plotly_chart(fig, use_container_width=True)
        # نتایج عددی
        last_actual = price_series.iloc[-1]
        arima_last_pred = forecast_arima.iloc[-1]
        arima_return = (arima_last_pred - last_actual) / last_actual
        st.markdown(f"📈 **قیمت فعلی:** {last_actual:.2f} | **قیمت پیش‌بینی ARIMA ({future_periods} روز):** {arima_last_pred:.2f}")
        st.markdown(f"📊 **بازده پیش‌بینی‌شده ARIMA ({future_periods} روز):** {arima_return:.2%}")
        if forecast_vol is not None:
            garch_last_pred = garch_sim[-1]
            garch_return = (garch_last_pred - last_actual) / last_actual
            st.markdown(f"🌪️ **قیمت سناریوی GARCH ({future_periods} روز):** {garch_last_pred:.2f}")
            st.markdown(f"🌪️ **بازده سناریوی GARCH:** {garch_return:.2%}")

else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید یا داده آنلاین وارد نمایید.")
