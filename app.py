import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import yfinance as yf
import warnings
from datetime import datetime, timedelta

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
    if years and months:
        return f"{years} سال و {months} ماه"
    if years:
        return f"{years} سال"
    if months:
        return f"{months} ماه"
    return "کمتر از ۱ ماه"

def max_drawdown(returns):
    if len(returns) == 0:
        return 0
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min() * 100

# ==================== پیش‌بینی قیمت (Monte Carlo) ====================
def forecast_price_series(price_series, days=63, sims=500):
    """شبیه‌سازی قیمت‌های آینده با روش Monte Carlo"""
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    mu = log_ret.mean()
    sigma = log_ret.std()
    last_price = price_series.iloc[-1]

    paths = np.zeros((days, sims))
    
    for i in range(sims):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * np.exp(mu + sigma * np.random.normal()))
        paths[:, i] = prices[1:]
    
    return paths

def plot_forecast_single(price_series, asset_name):
    """رسم نمودار پیش‌بینی برای یک دارایی"""
    ma150 = price_series.rolling(150).mean()
    paths = forecast_price_series(price_series, 90)

    fig = go.Figure()
    
    # قیمت واقعی
    fig.add_trace(go.Scatter(
        y=price_series,
        name="قیمت واقعی",
        mode="lines",
        line=dict(color="blue", width=2)
    ))
    
    # میانگین متحرک
    fig.add_trace(go.Scatter(
        y=ma150,
        name="MA 150",
        mode="lines",
        line=dict(dash="dash", color="gray")
    ))
    
    # پیش‌بینی نرمال (50 درصدیل)
    median_forecast = np.percentile(paths, 50, axis=1)
    fig.add_trace(go.Scatter(
        y=median_forecast,
        name="پیش‌بینی نرمال (۳ ماه)",
        mode="lines",
        line=dict(color="orange", width=2)
    ))
    
    # سناریوی خوش‌بینانه
    optimistic = np.percentile(paths, 85, axis=1)
    fig.add_trace(go.Scatter(
        y=optimistic,
        name="سناریو خوش‌بینانه (85%)",
        mode="lines",
        line=dict(dash="dot", color="green")
    ))
    
    # سناریوی بدبینانه
    pessimistic = np.percentile(paths, 15, axis=1)
    fig.add_trace(go.Scatter(
        y=pessimistic,
        name="سناریو بدبینانه (15%)",
        mode="lines",
        line=dict(dash="dot", color="red")
    ))
    
    # منطقه عدم قطعیت (75% تا 25%)
    upper_bound = np.percentile(paths, 75, axis=1)
    lower_bound = np.percentile(paths, 25, axis=1)
    
    fig.add_trace(go.Scatter(
        y=upper_bound,
        fill=None,
        mode="lines",
        line_color="rgba(0,0,0,0)",
        showlegend=False,
        name="منطقه احتمالی"
    ))
    
    fig.add_trace(go.Scatter(
        y=lower_bound,
        fill='tonexty',
        mode="lines",
        line_color="rgba(0,0,0,0)",
        fillcolor='rgba(0,100,255,0.2)',
        name="منطقه احتمالی (50%)"
    ))

    fig.update_layout(
        title=f"🔮 پیش‌بینی قیمت {asset_name} (۹۰ روز آینده)",
        xaxis_title="روز",
        yaxis_title="قیمت",
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# ==================== ماشین حساب تخصیص دارایی ====================
def capital_allocator_calculator(weights, asset_names, total_usd, exchange_rate):
    """محاسبه جزئیات خریداری برای هر دارایی"""
    
    allocation_data = []
    
    for i, asset in enumerate(asset_names):
        weight = weights[i]
        amount_usd = weight * total_usd
        amount_toman = amount_usd * (exchange_rate / 1_000_000)
        amount_rial = amount_toman * 10
        
        allocation_data.append({
            "دارایی": asset,
            "درصد وزن": f"{weight*100:.2f}%",
            "دلار ($)": f"${amount_usd:,.2f}",
            "تومان (تومان)": f"{amount_toman:,.0f}",
            "ریال (ریال)": f"{amount_rial:,.0f}",
            "بدون فرمت_USD": amount_usd,
            "بدون فرمت_Toman": amount_toman,
            "بدون فرمت_Rial": amount_rial,
        })
    
    df_allocation = pd.DataFrame(allocation_data)
    return df_allocation.sort_values("بدون فرمت_USD", ascending=False)

# ==================== استراتژی‌های هجینگ و آپشن ====================
hedge_strategies = {
    "Barbell طالب (۹۰/۱۰)": {"gold_min": 0.45, "usd_min": 0.45, "btc_max": 0.10},
    "Tail-Risk طالب": {"gold_min": 0.35, "usd_min": 0.35, "btc_max": 0.05},
    "Antifragile طالب": {"gold_min": 0.40, "usd_min": 0.20, "btc_max": 0.40},
    "طلا + تتر (ترکیبی)": {"gold_min": 0.15, "usd_min": 0.10, "btc_max": 0.20},
    "حداقل هج": {"gold_min": 0.10, "usd_min": 0.00, "btc_max": 0.40},
    "بدون هجینگ": {"gold_min": 0.00, "usd_min": 0.00, "btc_max": 1.00},
}

option_strategies = {
    "بدون آپشن": {"cost_pct": 0.0, "name": "بدون تغییر"},
    "Protective Put": {"cost_pct": 4.8, "name": "بیمه کامل"},
    "Collar": {"cost_pct": 0.4, "name": "هج کم‌هزینه"},
    "Covered Call": {"cost_pct": -3.2, "name": "درآمد ماهانه"},
    "Tail-Risk Put": {"cost_pct": 2.1, "name": "محافظت در سقوط"},
}

# ==================== تمام ۱۴ سبک حرفه‌ای ====================
def get_portfolio_weights(style, returns, mean_ret, cov_mat, rf, bounds):
    n = len(mean_ret)
    x0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    try:
        # 1. مارکوویتز + هجینگ
        if style == "مارکوویتز + هجینگ (بهینه‌ترین شارپ)":
            def obj(w):
                port_ret = np.dot(mean_ret, w)
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                if port_vol < 1e-8:
                    return 9999
                return -(port_ret - rf) / port_vol
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 5000})
            return res.x if res.success else x0

        # 2. وزن برابر
        elif style == "وزن برابر (ساده و مقاوم)":
            w = np.ones(n) / n
            w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
            w /= w.sum()
            return w

        # 3. حداقل ریسک
        elif style == "حداقل ریسک (محافظه‌کارانه)":
            def obj(w):
                return np.dot(w.T, np.dot(cov_mat, w))
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return res.x if res.success else x0

        # 4. ریسک‌پاریتی
        elif style == "ریسک‌پاریتی (Risk Parity)":
            def rp_obj(w):
                port_var = np.dot(w.T, np.dot(cov_mat, w))
                if port_var < 1e-10:
                    return 9999
                contrib = w * np.dot(cov_mat, w) / np.sqrt(port_var)
                return np.sum((contrib - np.mean(contrib))**2)
            res = minimize(rp_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return res.x if res.success else x0

        # 5. مونت‌کارلو مقاوم
        elif style == "مونت‌کارلو مقاوم (Resampled Frontier)":
            best_sharpe = -9999
            best_w = x0
            for _ in range(10000):
                w = np.random.random(n)
                w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
                w /= w.sum()
                ret = np.dot(mean_ret, w)
                risk = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                sharpe = (ret - rf) / risk if risk > 0 else -9999
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_w = w
            return best_w

        # 6. HRP (سلسله‌مراتبی)
        elif style == "HRP (سلسله‌مراتبی)":
            corr = returns.corr()
            dist = np.sqrt((1 - corr) / 2)
            link = linkage(squareform(dist), 'single')
            order = np.array([link[-i, 0] for i in range(1, n)][::-1] + [link[-i, 1] for i in range(1, n)][::-1])
            w = np.zeros(n)
            for i in order.astype(int)[:n]:
                if i < len(returns.columns):
                    w[i] = 1 / (np.var(returns.iloc[:, i]) + 1e-8)
            w /= (w.sum() + 1e-8)
            return w

        # 7. Maximum Diversification
        elif style == "Maximum Diversification":
            vol = np.sqrt(np.diag(cov_mat))
            def obj(w):
                numerator = np.dot(w, vol)
                denominator = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                return -numerator / (denominator + 1e-8)
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return res.x if res.success else x0

        # 8. Inverse Volatility
        elif style == "Inverse Volatility":
            vol = np.sqrt(np.diag(cov_mat))
            w = 1 / (vol + 1e-8)
            w /= w.sum()
            return w

        # 9. Barbell طالب
        elif style == "Barbell طالب (۹۰/۱۰)":
            w = np.zeros(n)
            safe = [i for i, name in enumerate(returns.columns) if any(x in name.upper() for x in ["GC=", "GOLD", "USD", "USDIRR", "USDT"])]
            risky = [i for i in range(n) if i not in safe]
            if safe:
                w[safe] = 0.9 / len(safe)
            if risky:
                w[risky] = 0.1 / len(risky)
            return w

        # 10. Antifragile طالب
        elif style == "Antifragile طالب":
            w = np.zeros(n)
            gold = [i for i, name in enumerate(returns.columns) if "GC=" in name.upper() or "GOLD" in name.upper()]
            btc = [i for i, name in enumerate(returns.columns) if "BTC" in name.upper()]
            if gold:
                w[gold] = 0.4 / len(gold)
            if btc:
                w[btc] = 0.4 / len(btc)
            rest = [i for i in range(n) if i not in gold + btc]
            if rest:
                w[rest] = 0.2 / len(rest)
            return w

        # 11. Kelly Criterion
        elif style == "Kelly Criterion (حداکثر رشد)":
            diag_cov = np.diag(cov_mat)
            w = mean_ret / (diag_cov + 1e-8)
            w = np.clip(w, 0, None)
            w /= (w.sum() + 1e-8)
            return w

        # 12. Most Diversified Portfolio
        elif style == "Most Diversified Portfolio":
            vol = np.sqrt(np.diag(cov_mat))
            def obj(w):
                numerator = np.dot(w, vol)
                denominator = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                return -numerator / (denominator + 1e-8)
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return res.x if res.success else x0

        # 13. Equal Risk Bounding
        elif style == "Equal Risk Bounding":
            target = 1.0 / n
            def erb_obj(w):
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                contrib = w * np.dot(cov_mat, w) / (port_vol + 1e-8)
                return np.sum((contrib - target)**2)
            res = minimize(erb_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return res.x if res.success else x0

        # 14. بلک-لیترمن
        elif style == "بلک-لیترمن (ترکیب نظر شخصی)":
            w = mean_ret / (mean_ret.sum() + 1e-8)
            w = np.nan_to_num(w)
            w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
            w /= (w.sum() + 1e-8)
            return w

    except Exception as e:
        st.warning(f"خطا در {style}: {str(e)[:50]} — وزن برابر استفاده شد")
        return x0

# ==================== محاسبه ریسک پرتفو با Protective Put ====================
def calculate_portfolio_with_protective_put(returns, weights, cov_mat, asset_names, 
                                           btc_premium_pct=0.0, eth_premium_pct=0.0,
                                           btc_strike=None, eth_strike=None):
    """
    محاسبه ریسک پرتفوی با احساب تاثیر Protective Put
    """
    
    # محاسبه ریسک بدون بیمه
    original_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
    
    # کپی وزن‌ها برای محاسبه تاثیر بیمه
    adjusted_weights = weights.copy()
    
    # محاسبه تاثیر بیمه برای BTC و ETH
    btc_idx = None
    eth_idx = None
    
    for i, name in enumerate(asset_names):
        if "BTC" in name.upper():
            btc_idx = i
        if "ETH" in name.upper():
            eth_idx = i
    
    # تنظیم ریسک بر اساس بیمه
    # Protective Put کاهش volatility را شبیه‌سازی می‌کند
    if btc_idx is not None and btc_premium_pct > 0:
        # هرچه premium بیشتر، محافظت بیشتر
        protection_factor_btc = 1.0 - (btc_premium_pct / 100.0) * 0.5
        adjusted_weights[btc_idx] *= protection_factor_btc
    
    if eth_idx is not None and eth_premium_pct > 0:
        protection_factor_eth = 1.0 - (eth_premium_pct / 100.0) * 0.5
        adjusted_weights[eth_idx] *= protection_factor_eth
    
    # نرمال‌سازی وزن‌ها
    if adjusted_weights.sum() > 0:
        adjusted_weights /= adjusted_weights.sum()
    
    # محاسبه ریسک جدید
    new_risk = np.sqrt(np.dot(adjusted_weights.T, np.dot(cov_mat, adjusted_weights))) * 100
    
    # محاسبه کاهش ریسک
    risk_reduction = original_risk - new_risk
    risk_reduction_pct = (risk_reduction / original_risk * 100) if original_risk > 0 else 0
    
    return {
        "original_risk": original_risk,
        "new_risk": new_risk,
        "risk_reduction": risk_reduction,
        "risk_reduction_pct": risk_reduction_pct,
        "adjusted_weights": adjusted_weights
    }

# ==================== محاسبه پرتفوی ====================
@st.fragment
def calculate_portfolio():
    if "prices" not in st.session_state or st.session_state.prices is None:
        st.info("📊 لطفاً ابتدا داده‌ها را دانلود کنید.")
        return

    prices = st.session_state.prices
    returns = prices.pct_change().dropna()
    asset_names = list(prices.columns)
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    rf = st.session_state.rf_rate / 100

    # محدودیت‌های هجینگ
    bounds = []
    hedge = hedge_strategies[st.session_state.hedge_strategy]
    for name in asset_names:
        low = 0.0
        up = 1.0
        n = name.upper()
        if any(x in n for x in ["GC=", "GOLD", "SI="]):
            low = max(low, hedge["gold_min"])
        if any(x in n for x in ["USD", "USDIRR", "USDT"]):
            low = max(low, hedge["usd_min"])
        if any(x in n for x in ["BTC", "بیت"]):
            up = min(up, hedge["btc_max"])
        if low > up:
            low, up = 0.0, 1.0
        bounds.append((float(low), float(up)))

    # وزن‌ها
    weights = get_portfolio_weights(st.session_state.selected_style, returns, mean_ret, cov_mat, rf, bounds)
    
    # اعمال آپشن
    opt = option_strategies[st.session_state.option_strategy]
    option_cost = opt["cost_pct"]
    adjusted_return = np.dot(mean_ret, weights) * 100 - option_cost
    adjusted_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
    
    if "Put" in st.session_state.option_strategy:
        adjusted_risk *= 0.7
    elif "Call" in st.session_state.option_strategy:
        adjusted_risk *= 1.1

    sharpe = (adjusted_return/100 - rf) / (adjusted_risk/100) if adjusted_risk > 0 else 0
    recovery = format_recovery(calculate_recovery_time(returns.dot(weights)))

    # ==================== تب‌ها ====================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 خلاصه", "💰 تخصیص دارایی", "🔮 پیش‌بینی قیمت", "📈 بک‌تست", "🛡️ Protective Put"])

    with tab1:
        st.markdown("### 📋 خلاصه پرتفوی")
        is_option_active = st.session_state.option_strategy != "بدون آپشن"
        st.success(f"**سبک:** {st.session_state.selected_style} | **آپشن:** {opt['name']}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📈 بازده" + (" (با آپشن)" if is_option_active else ""), f"{adjusted_return:.2f}%")
        c2.metric("⚠️ ریسک" + (" (با آپشن)" if is_option_active else ""), f"{adjusted_risk:.2f}%")
        c3.metric("⭐ شارپ" + (" (با آپشن)" if is_option_active else ""), f"{sharpe:.3f}")
        c4.metric("⏱️ زمان ریکاوری", recovery)

        st.markdown("---")
        
        df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)}).sort_values("وزن (%)", ascending=False)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### وزن‌های دارایی‌ها")
            st.dataframe(df_w, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("### نمودار توزیع")
            st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی"), use_container_width=True)

    with tab2:
        st.markdown("### 💰 ماشین حساب تخصیص دارایی")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_usd = st.number_input("💵 کل سرمایه (دلار)", min_value=100, max_value=10_000_000, value=1200, step=100)
        with col2:
            exchange_rate = st.number_input("💱 نرخ تبدیل (دلار به تومان)", min_value=100_000, max_value=500_000_000, value=200_000_000, step=1_000_000)
        with col3:
            st.write("")
            st.write("")

        # محاسبه تخصیص
        df_alloc = capital_allocator_calculator(weights, asset_names, total_usd, exchange_rate)
        
        st.markdown("#### جزئیات خریداری:")
        st.dataframe(
            df_alloc[["دارایی", "درصد وزن", "دلار ($)", "تومان (تومان)", "ریال (ریال)"]],
            use_container_width=True,
            hide_index=True
        )
        
        # خلاصه کل
        st.markdown("---")
        st.markdown("#### 📊 خلاصه کل:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💵 کل دلار", f"${total_usd:,.2f}")
        with col2:
            total_toman = total_usd * (exchange_rate / 1_000_000)
            st.metric("💴 کل تومان", f"{total_toman:,.0f}")
        with col3:
            total_rial = total_toman * 10
            st.metric("💳 کل ریال", f"{total_rial:,.0f}")

        # دانلود فایل
        csv = df_alloc[["دارایی", "درصد وزن", "دلار ($)", "تومان (تومان)", "ریال (ریال)"]].to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 دانلود تخصیص (CSV)",
            data=csv,
            file_name=f"portfolio_allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with tab3:
        st.markdown("### 🔮 پیش‌بینی قیمت (Monte Carlo)")
        
        selected_assets = st.multiselect(
            "انتخاب دارایی‌هایی برای پیش‌بینی:",
            asset_names,
            default=asset_names[:min(2, len(asset_names))]
        )
        
        if selected_assets:
            forecast_days = st.slider("روزهای پیش‌بینی:", 30, 365, 90)
            
            for asset in selected_assets:
                st.markdown(f"#### {asset}")
                
                price_series = prices[asset]
                
                # محاسبه آمار
                current_price = price_series.iloc[-1]
                ma_50 = price_series.rolling(50).mean().iloc[-1]
                ma_200 = price_series.rolling(200).mean().iloc[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💲 قیمت فعلی", f"${current_price:.2f}")
                col2.metric("📊 MA 50", f"${ma_50:.2f}")
                col3.metric("📈 MA 200", f"${ma_200:.2f}")
                col4.metric("📍 وضعیت", "✅ بالای MA200" if current_price > ma_200 else "⚠️ پایین MA200")
                
                # نمودار
                fig = plot_forecast_single(price_series, asset)
                st.plotly_chart(fig, use_container_width=True)
                
                # آمار پیش‌بینی
                paths = forecast_price_series(price_series, forecast_days, sims=500)
                
                # محاسبه آمار درست
                percentile_50 = np.percentile(paths, 50, axis=1)[-1]
                percentile_85 = np.percentile(paths, 85, axis=1)[-1]
                percentile_15 = np.percentile(paths, 15, axis=1)[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🎯 قیمت پیش‌بینی شده", f"${percentile_50:.2f}")
                col2.metric("📈 سناریو خوش‌بینانه", f"${percentile_85:.2f}")
                col3.metric("📉 سناریو بدبینانه", f"${percentile_15:.2f}")
                col4.metric("📊 صعود احتمالی", f"{((percentile_50 / current_price - 1) * 100):.1f}%")
                
                st.markdown("---")

        else:
            st.info("🔍 برای مشاهده پیش‌بینی، حداقل یک دارایی را انتخاب کنید.")

    with tab4:
        st.markdown("### 📈 بک‌تست واقعی (چه می‌شد اگر؟)")
        
        col1, col2, col3 = st.columns(3)
        initial = col1.number_input("💰 سرمایه اولیه (میلیون تومان)", 10, 10000, 100)
        years = col2.selectbox("📅 چند سال پیش شروع کرده بودید؟", [1, 3, 5, 10], index=2)
        monthly = col3.number_input("📊 سرمایه‌گذاری ماهانه (میلیون)", 0, 100, 10)

        full_returns = prices.pct_change().dropna()
        port_daily = full_returns.dot(weights)
        backtest_days = years * 252
        if len(port_daily) > backtest_days:
            port_daily = port_daily.tail(backtest_days)

        value = initial
        values = [initial]
        for i in range(len(port_daily)):
            value *= (1 + port_daily.iloc[i])
            if i % 21 == 0 and i > 0:
                value += monthly
            values.append(value)

        total_invested = initial + (monthly * years * 12)
        profit = value - total_invested
        profit_pct = (profit / total_invested) * 100 if total_invested > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("💎 سرمایه نهایی", f"{value:,.0f} میلیون", delta=f"{profit_pct:.1f}%")
        col2.metric("💵 سود خالص", f"{profit:,.0f} میلیون")
        col3.metric("📊 نسبت سود/سرمایه", f"{(profit/total_invested):.1%}" if total_invested > 0 else "0%")

        fig_back = go.Figure()
        fig_back.add_trace(go.Scatter(
            y=values,
            name="رشد سرمایه شما",
            mode="lines",
            fill="tozeroy",
            line=dict(color="green", width=2)
        ))
        fig_back.add_hline(y=initial, line_dash="dash", annotation_text="سرمایه اولیه", line_color="red")
        fig_back.update_layout(
            title=f"📈 رشد سرمایه از {years} سال پیش تا امروز",
            xaxis_title="روز",
            yaxis_title="میلیون تومان",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig_back, use_container_width=True)

    with tab5:
        st.markdown("### 🛡️ Protective Put برای کاهش ریسک")
        st.info("""
        📌 **Protective Put** یک استراتژی بیمه است که:
        - تعداد قراردادهای Long Put برای محافظت خریداری می‌کنید
        - اگر قیمت دارایی سقوط کند، Put سود می‌دهد
        - اگر قیمت بالا برود، فقط premium از دست می‌رود
        - نتیجه: محافظت از سقوط‌های شدید با هزینه معقول
        """)
        
        st.markdown("---")
        
        # پیدا کردن BTC و ETH
        btc_col = None
        eth_col = None
        
        for col in asset_names:
            if "BTC" in col.upper():
                btc_col = col
            if "ETH" in col.upper():
                eth_col = col
        
        if btc_col is None or eth_col is None:
            st.error("❌ برای استراتژی Protective Put، نیاز به BTC-USD و ETH-USD دارید!")
            st.info(f"📊 دارایی‌های موجود: {', '.join(asset_names)}")
            return
        
        st.markdown("#### 📝 مشخصات قرارداد Protective Put")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔵 BTC-USD")
            btc_price = prices[btc_col].iloc[-1]
            st.write(f"**قیمت فعلی:** ${btc_price:,.2f}")
            
            btc_strike = st.number_input(
                "Strike Price (ضربه) برای BTC ($)",
                min_value=btc_price * 0.70,
                max_value=btc_price * 0.99,
                value=btc_price * 0.90,
                step=100.0,
                key="btc_strike"
            )
            
            btc_premium = st.number_input(
                "Premium (حق‌العمل) برای هر قرارداد BTC ($)",
                min_value=0.0,
                max_value=btc_price * 0.20,
                value=btc_price * 0.04,
                step=100.0,
                key="btc_premium"
            )
            
            btc_contracts = st.number_input(
                "تعداد قراردادهای Put برای BTC",
                min_value=1,
                max_value=100,
                value=1,
                key="btc_contracts"
            )
            
            btc_contract_size = st.number_input(
                "تعداد BTC در هر قرارداد",
                min_value=0.1,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key="btc_size"
            )
            
            btc_expiry = st.date_input(
                "تاریخ انقضا قرارداد BTC",
                value=(datetime.now() + timedelta(days=45)).date(),
                key="btc_expiry"
            )
        
        with col2:
            st.markdown("##### 🟢 ETH-USD")
            eth_price = prices[eth_col].iloc[-1]
            st.write(f"**قیمت فعلی:** ${eth_price:,.2f}")
            
            eth_strike = st.number_input(
                "Strike Price (ضربه) برای ETH ($)",
                min_value=eth_price * 0.70,
                max_value=eth_price * 0.99,
                value=eth_price * 0.90,
                step=10.0,
                key="eth_strike"
            )
            
            eth_premium = st.number_input(
                "Premium (حق‌العمل) برای هر قرارداد ETH ($)",
                min_value=0.0,
                max_value=eth_price * 0.20,
                value=eth_price * 0.04,
                step=10.0,
                key="eth_premium"
            )
            
            eth_contracts = st.number_input(
                "تعداد قراردادهای Put برای ETH",
                min_value=1,
                max_value=100,
                value=1,
                key="eth_contracts"
            )
            
            eth_contract_size = st.number_input(
                "تعداد ETH در هر قرارداد",
                min_value=0.1,
                max_value=1000.0,
                value=1.0,
                step=0.1,
                key="eth_size"
            )
            
            eth_expiry = st.date_input(
                "تاریخ انقضا قرارداد ETH",
                value=(datetime.now() + timedelta(days=45)).date(),
                key="eth_expiry"
            )
        
        st.markdown("---")
        
        # محاسبات
        # BTC
        btc_total_premium = btc_premium * btc_contracts * btc_contract_size
        btc_total_premium_pct = (btc_total_premium / (btc_price * btc_contract_size * btc_contracts)) * 100 if (btc_price * btc_contract_size * btc_contracts) > 0 else 0
        btc_max_loss = (btc_price - btc_strike) * btc_contract_size * btc_contracts
        btc_max_loss_pct = (btc_max_loss / (btc_price * btc_contract_size * btc_contracts)) * 100 if (btc_price * btc_contract_size * btc_contracts) > 0 else 0
        
        # ETH
        eth_total_premium = eth_premium * eth_contracts * eth_contract_size
        eth_total_premium_pct = (eth_total_premium / (eth_price * eth_contract_size * eth_contracts)) * 100 if (eth_price * eth_contract_size * eth_contracts) > 0 else 0
        eth_max_loss = (eth_price - eth_strike) * eth_contract_size * eth_contracts
        eth_max_loss_pct = (eth_max_loss / (eth_price * eth_contract_size * eth_contracts)) * 100 if (eth_price * eth_contract_size * eth_contracts) > 0 else 0
        
        st.markdown("#### 📊 تحلیل تفصیلی")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📈 تجزیه BTC")
            st.metric("💰 کل Premium پرداختی", f"${btc_total_premium:,.2f}")
            st.metric("📊 Premium (% قیمت)", f"{btc_total_premium_pct:.3f}%")
            st.metric("🛡️ محافظت تا", f"${btc_strike:,.2f}")
            st.metric("📉 حداکثر ضرر (سقوط)", f"${btc_max_loss:,.2f}")
            st.metric("📉 حداکثر ضرر (%)", f"{btc_max_loss_pct:.3f}%")
            
            st.markdown("**💡 نتیجه:**")
            if btc_max_loss_pct <= 2.0:
                st.success(f"✅ ریسک BTC کاهش یافته است: {btc_max_loss_pct:.3f}% < 2%")
            else:
                st.warning(f"⚠️ ریسک BTC هنوز بالاتر از 2% است: {btc_max_loss_pct:.3f}%")
        
        with col2:
            st.markdown("##### 📈 تجزیه ETH")
            st.metric("💰 کل Premium پرداختی", f"${eth_total_premium:,.2f}")
            st.metric("📊 Premium (% قیمت)", f"{eth_total_premium_pct:.3f}%")
            st.metric("🛡️ محافظت تا", f"${eth_strike:,.2f}")
            st.metric("📉 حداکثر ضرر (سقوط)", f"${eth_max_loss:,.2f}")
            st.metric("📉 حداکثر ضرر (%)", f"{eth_max_loss_pct:.3f}%")
            
            st.markdown("**💡 نتیجه:**")
            if eth_max_loss_pct <= 2.0:
                st.success(f"✅ ریسک ETH کاهش یافته است: {eth_max_loss_pct:.3f}% < 2%")
            else:
                st.warning(f"⚠️ ریسک ETH هنوز بالاتر از 2% است: {eth_max_loss_pct:.3f}%")
        
        st.markdown("---")
        
        # محاسبه ریسک پرتفوی با Protective Put
        st.markdown("#### 🎯 تاثیر بیمه بر ریسک کل پرتفوی")
        
        # ریسک بدون بیمه
        original_portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
        
        # محاسبه ریسک با بیمه
        result = calculate_portfolio_with_protective_put(
            returns, weights, cov_mat, asset_names,
            btc_premium_pct=btc_total_premium_pct,
            eth_premium_pct=eth_total_premium_pct,
            btc_strike=btc_strike,
            eth_strike=eth_strike
        )
        
        new_portfolio_risk = result['new_risk']
        risk_reduction = result['risk_reduction']
        risk_reduction_pct = result['risk_reduction_pct']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 ریسک پرتفوی (بدون بیمه)", f"{original_portfolio_risk:.2f}%")
        col2.metric("🛡️ ریسک پرتفوی (با بیمه)", f"{new_portfolio_risk:.2f}%")
        col3.metric("📉 کاهش ریسک", f"{risk_reduction:.2f}% ({risk_reduction_pct:.2f}%)")
        
        # نمودار مقایسه
        fig_risk = go.Figure()
        
        fig_risk.add_trace(go.Bar(
            x=['بدون Protective Put', 'با Protective Put'],
            y=[original_portfolio_risk, new_portfolio_risk],
            name='ریسک پرتفوی',
            marker=dict(color=['#ff6b6b', '#51cf66'])
        ))
        
        fig_risk.update_layout(
            title="📊 مقایسه ریسک پرتفوی",
            yaxis_title="ریسک (%)",
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # خلاصه نهایی
        st.markdown("---")
        st.markdown("#### 📋 خلاصه نهایی")
        
        total_premium = btc_total_premium + eth_total_premium
        
        summary_data = {
            "دارایی": ["BTC-USD", "ETH-USD", "کل"],
            "قیمت فعلی": [f"${btc_price:,.2f}", f"${eth_price:,.2f}", "-"],
            "Strike": [f"${btc_strike:,.2f}", f"${eth_strike:,.2f}", "-"],
            "تعداد قراردادها": [btc_contracts, eth_contracts, btc_contracts + eth_contracts],
            "Premium کل": [f"${btc_total_premium:,.2f}", f"${eth_total_premium:,.2f}", f"${total_premium:,.2f}"],
            "حداکثر ضرر": [f"${btc_max_loss:,.2f}", f"${eth_max_loss:,.2f}", f"${btc_max_loss + eth_max_loss:,.2f}"],
            "ریسک (%)": [f"{btc_max_loss_pct:.3f}%", f"{eth_max_loss_pct:.3f}%", "-"],
            "تاریخ انقضا": [str(btc_expiry), str(eth_expiry), "-"]
        }
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        # نمودار Payoff
        st.markdown("---")
        st.markdown("#### 📈 نمودار سود/ضرر Protective Put")
        
        # BTC Payoff
        btc_price_range = np.linspace(btc_strike * 0.8, btc_price * 1.2, 100)
        btc_payoff = []
        for p in btc_price_range:
            put_payoff = max(btc_strike - p, 0) * btc_contract_size * btc_contracts - btc_total_premium
            btc_payoff.append(put_payoff)
        
        # ETH Payoff
        eth_price_range = np.linspace(eth_strike * 0.8, eth_price * 1.2, 100)
        eth_payoff = []
        for p in eth_price_range:
            put_payoff = max(eth_strike - p, 0) * eth_contract_size * eth_contracts - eth_total_premium
            eth_payoff.append(put_payoff)
        
        fig_payoff = go.Figure()
        
        fig_payoff.add_trace(go.Scatter(
            x=btc_price_range,
            y=btc_payoff,
            name="BTC Protective Put",
            mode="lines",
            line=dict(color="orange", width=2)
        ))
        
        fig_payoff.add_trace(go.Scatter(
            x=eth_price_range,
            y=eth_payoff,
            name="ETH Protective Put",
            mode="lines",
            line=dict(color="blue", width=2)
        ))
        
        fig_payoff.add_hline(y=0, line_dash="dash", line_color="red")
        fig_payoff.update_layout(
            title="📊 نمودار سود/ضرر Protective Put",
            xaxis_title="قیمت دارایی ($)",
            yaxis_title="سود/ضرر ($)",
            height=500,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_payoff, use_container_width=True)
        
        # دانلود
        csv_summary = df_summary.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 دانلود استراتژی Protective Put (CSV)",
            data=csv_summary,
            file_name=f"protective_put_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ==================== صفحه اصلی + سایدبار ====================
st.set_page_config(page_title="Portfolio360 Ultimate Pro", layout="wide")

# Header
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0;'>💼 Portfolio360 Ultimate Pro</h1>
    <p style='color: #ddd; margin: 5px 0;'>سیستم تحلیل و مدیریت پرتفوی سرمایه‌گذاری حرفه‌ای</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔧 تنظیمات سیستم")
    
    st.markdown("---")
    
    st.header("📥 دانلود داده")
    tickers = st.text_input(
        "نمادهای دارایی (با کاما جدا کنید)",
        "BTC-USD, GC=F, USDIRR=X, ^GSPC, ETH-USD"
    )
    if st.button("🔄 دانلود داده", type="primary", use_container_width=True):
        with st.spinner("درحال دانلود..."):
            data = download_data(tickers)
            if data is not None:
                st.session_state.prices = data
                st.success(f"✅ {len(data.columns)} دارایی بارگذاری شد!")
                st.rerun()

    st.markdown("---")
    
    st.header("🛡️ هجینگ")
    if "hedge_strategy" not in st.session_state:
        st.session_state.hedge_strategy = "طلا + تتر (ترکیبی)"
    st.session_state.hedge_strategy = st.selectbox(
        "استراتژی هجینگ:",
        list(hedge_strategies.keys()),
        index=3
    )

    st.markdown("---")
    
    st.header("📊 آپشن")
    if "option_strategy" not in st.session_state:
        st.session_state.option_strategy = "بدون آپشن"
    st.session_state.option_strategy = st.selectbox(
        "استراتژی آپشن:",
        list(option_strategies.keys())
    )

    st.markdown("---")
    
    st.header("🎯 سبک پرتفوی")
    styles = [
        "مارکوویتز + هجینگ (بهینه‌ترین شارپ)",
        "وزن برابر (ساده و مقاوم)",
        "حداقل ریسک (محافظه‌کارانه)",
        "ریسک‌پاریتی (Risk Parity)",
        "مونت‌کارلو مقاوم (Resampled Frontier)",
        "HRP (سلسله‌مراتبی)",
        "Maximum Diversification",
        "Inverse Volatility",
        "Barbell طالب (۹۰/۱۰)",
        "Antifragile طالب",
        "Kelly Criterion (حداکثر رشد)",
        "Most Diversified Portfolio",
        "Equal Risk Bounding",
        "بلک-لیترمن (ترکیب نظر شخصی)"
    ]
    if "selected_style" not in st.session_state:
        st.session_state.selected_style = styles[0]
    st.session_state.selected_style = st.selectbox(
        "انتخاب سبک:",
        styles,
        index=styles.index(st.session_state.selected_style)
    )

    st.markdown("---")
    
    st.header("⚙️ تنظیمات عمومی")
    if "rf_rate" not in st.session_state:
        st.session_state.rf_rate = 18.0
    st.session_state.rf_rate = st.number_input(
        "نرخ بدون ریسک (%) سالانه:",
        0.0,
        50.0,
        st.session_state.rf_rate,
        0.5
    )

    st.markdown("---")
    
    with st.expander("ℹ️ درباره سیستم"):
        st.write("""
        **Portfolio360 Ultimate Pro** یک ابزار جامع برای:
        
        ✅ **14 سبک حرفه‌ای** - از Markowitz تا Black-Litterman
        
        🛡️ **6 استراتژی هجینگ** - محافظت از ریسک سقوط
        
        📊 **5 استراتژی آپشن** - بیمه و درآمد اضافی
        
        💰 **ماشین حساب تخصیص** - خریداری دقیق هر دارایی
        
        🔮 **پیش‌بینی قیمت** - Monte Carlo برای تمام دارایی‌ها
        
        📈 **بک‌تست واقعی** - اگر از قبل شروع کرده بودید چی می‌شد؟
        
        🛡️ **Protective Put** - بیمه‌گذاری برای BTC و ETH با تاثیر بر ریسک کل
        """)

# اجرا
calculate_portfolio()

st.balloons()
st.caption("✨ Portfolio360 Ultimate Pro v4.0 — تمام ۱۴ سبک + تخصیص دقیق + پیش‌بینی + بک‌تست + Protective Put با تاثیر ریسک | ۱۴۰۴ | ❤️ با عشق برای ایران")
