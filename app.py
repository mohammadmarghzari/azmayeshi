# Full app.py with Married Put (Protective Put) that updates portfolio risk and shows payoff chart + suggestion to reach target risk.
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

def plot_forecast_single(price_series, asset_name, days_default=90):
    ma150 = price_series.rolling(150).mean()
    paths = forecast_price_series(price_series, days_default)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=price_series, name="قیمت واقعی", mode="lines", line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(y=ma150, name="MA 150", mode="lines", line=dict(dash="dash", color="gray")))
    median_forecast = np.percentile(paths, 50, axis=1)
    fig.add_trace(go.Scatter(y=median_forecast, name="پیش‌بینی نرمال", mode="lines", line=dict(color="orange", width=2)))
    fig.add_trace(go.Scatter(y=np.percentile(paths, 85, axis=1), name="سناریو خوش‌بینانه", mode="lines", line=dict(dash="dot", color="green")))
    fig.add_trace(go.Scatter(y=np.percentile(paths, 15, axis=1), name="سناریو بدبینانه", mode="lines", line=dict(dash="dot", color="red")))
    upper = np.percentile(paths, 75, axis=1); lower = np.percentile(paths, 25, axis=1)
    fig.add_trace(go.Scatter(y=upper, fill=None, mode="lines", line_color="rgba(0,0,0,0)", showlegend=False))
    fig.add_trace(go.Scatter(y=lower, fill='tonexty', mode="lines", line_color="rgba(0,0,0,0)", fillcolor='rgba(0,100,255,0.12)', name="منطقه احتمالی"))
    fig.update_layout(title=f"🔮 پیش‌بینی قیمت {asset_name} ({days_default} روز)", hovermode='x unified', template='plotly_white', height=480)
    return fig

# ==================== ماشین حساب تخصیص دارایی ====================
def capital_allocator_calculator(weights, asset_names, total_usd, exchange_rate):
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
        if style == "مارکوویتز + هجینگ (بهینه‌ترین شارپ)":
            def obj(w):
                port_ret = np.dot(mean_ret, w)
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                if port_vol < 1e-8:
                    return 9999
                return -(port_ret - rf) / port_vol
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 3000})
            return res.x if res.success else x0
        elif style == "وزن برابر (ساده و مقاوم)":
            w = np.ones(n) / n
            w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
            w /= w.sum()
            return w
        elif style == "حداقل ریسک (محافظه‌کارانه)":
            res = minimize(lambda w: np.dot(w.T, np.dot(cov_mat, w)), x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return res.x if res.success else x0
        elif style == "ریسک‌پاریتی (Risk Parity)":
            def rp_obj(w):
                port_var = np.dot(w.T, np.dot(cov_mat, w))
                if port_var < 1e-10:
                    return 9999
                contrib = w * np.dot(cov_mat, w) / np.sqrt(port_var)
                return np.sum((contrib - np.mean(contrib))**2)
            res = minimize(rp_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return res.x if res.success else x0
        elif style == "مونت‌کارلو مقاوم (Resampled Frontier)":
            best_sharpe = -9999; best_w = x0
            for _ in range(8000):
                w = np.random.random(n)
                w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
                w /= w.sum()
                ret = np.dot(mean_ret, w); risk = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                sharpe = (ret - rf) / risk if risk > 0 else -9999
                if sharpe > best_sharpe:
                    best_sharpe = sharpe; best_w = w
            return best_w
        elif style == "HRP (سلسله‌مراتبی)":
            corr = returns.corr(); dist = np.sqrt((1 - corr)/2)
            link = linkage(squareform(dist), 'single')
            # simple greedy ordering fallback
            w = np.ones(n); w /= w.sum(); return w
        elif style == "Maximum Diversification" or style == "Most Diversified Portfolio":
            vol = np.sqrt(np.diag(cov_mat))
            def obj(w):
                numerator = np.dot(w, vol)
                denominator = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                return -numerator / (denominator + 1e-8)
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return res.x if res.success else x0
        elif style == "Inverse Volatility":
            vol = np.sqrt(np.diag(cov_mat))
            w = 1/(vol + 1e-8); w /= w.sum(); return w
        elif style == "Barbell طالب (۹۰/۱۰)":
            w = np.zeros(n)
            safe = [i for i, name in enumerate(returns.columns) if any(x in name.upper() for x in ["GC=", "GOLD", "USD", "USDIRR", "USDT"])]
            risky = [i for i in range(n) if i not in safe]
            if safe: w[safe] = 0.9/len(safe)
            if risky: w[risky] = 0.1/len(risky)
            return w
        elif style == "Antifragile طالب":
            w = np.zeros(n); gold = [i for i, name in enumerate(returns.columns) if "GC=" in name.upper() or "GOLD" in name.upper()]
            btc = [i for i, name in enumerate(returns.columns) if "BTC" in name.upper()]
            if gold: w[gold] = 0.4/len(gold)
            if btc: w[btc] = 0.4/len(btc)
            rest = [i for i in range(n) if i not in gold + btc]
            if rest: w[rest] = 0.2/len(rest)
            return w
        elif style == "Kelly Criterion (حداکثر رشد)":
            diag_cov = np.diag(cov_mat)
            w = mean_ret/(diag_cov + 1e-8); w = np.clip(w, 0, None); w /= (w.sum()+1e-8); return w
        elif style == "Equal Risk Bounding":
            target = 1.0/n
            def erb_obj(w):
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                contrib = w * np.dot(cov_mat, w) / (port_vol +1e-8)
                return np.sum((contrib-target)**2)
            res = minimize(erb_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
            return res.x if res.success else x0
        elif style == "بلک-لیترمن (ترکیب نظر شخصی)":
            w = mean_ret/(mean_ret.sum()+1e-8); w = np.nan_to_num(w)
            w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds]); w /= (w.sum()+1e-8); return w
    except Exception as e:
        st.warning(f"خطا در محاسبه وزن‌ها ({style}): {e}")
        return x0

# ==================== Protective Put (Married Put) impact on portfolio risk ====================
def apply_protective_put_to_cov(cov_mat, asset_names, btc_idx, eth_idx, btc_reduction, eth_reduction):
    """
    Returns adjusted covariance matrix after applying 'protective' effect.
    btc_reduction and eth_reduction are fractions in [0,1] representing how much of that asset's volatility is removed.
    We approximate by scaling rows and columns corresponding to that asset.
    """
    cov_adj = cov_mat.copy().astype(float)
    n = cov_adj.shape[0]
    scale = np.ones(n)
    if btc_idx is not None:
        scale[btc_idx] = max(0.0, 1.0 - btc_reduction)
    if eth_idx is not None:
        scale[eth_idx] = max(0.0, 1.0 - eth_reduction)
    # scale covariance: cov_ij -> scale_i * scale_j * cov_ij
    for i in range(n):
        for j in range(n):
            cov_adj.iloc[i, j] = cov_mat.iloc[i, j] * scale[i] * scale[j]
    return cov_adj

# ==================== Suggest contracts to reach target risk (simple search) ====================
def suggest_contracts_for_target_risk(prices, returns, asset_names, weights, cov_mat, total_usd, btc_idx, eth_idx, btc_contract_size, eth_contract_size, max_contracts=20, target_risk_pct=2.0):
    """
    Brute-force search over number of contracts (0..max_contracts) for BTC and ETH to find combination 
    that reduces portfolio risk to <= target_risk_pct. Returns best solution (min premium) or None.
    Model: premium reduces volatility proportionally to premium/exposure (same model used elsewhere).
    """
    best = None
    # exposures
    exposures = {name: weights[i]*total_usd for i, name in enumerate(asset_names)}
    btc_name = asset_names[btc_idx] if btc_idx is not None else None
    eth_name = asset_names[eth_idx] if eth_idx is not None else None

    for b in range(0, max_contracts+1):
        for e in range(0, max_contracts+1):
            # compute total premiums (user will supply per-contract premium estimates; here we approximate)
            # For suggestion we need per-contract premium estimate -> we ask user to supply "est_premium_per_contract" outside.
            # But to run this function we expect user-provided estimates in st.session_state keys (see usage below).
            est_btc_prem = st.session_state.get("btc_premium_est", None)
            est_eth_prem = st.session_state.get("eth_premium_est", None)
            if est_btc_prem is None or est_eth_prem is None:
                return None  # can't suggest without per-contract premium estimates
            btc_total_premium = b * est_btc_prem * btc_contract_size if btc_idx is not None else 0.0
            eth_total_premium = e * est_eth_prem * eth_contract_size if eth_idx is not None else 0.0

            # premium_pct relative to exposure
            btc_premium_pct = (btc_total_premium / (exposures[btc_name] + 1e-8))*100 if btc_name else 0.0
            eth_premium_pct = (eth_total_premium / (exposures[eth_name] + 1e-8))*100 if eth_name else 0.0

            # map to reduction factor (same model as apply): reduction = min(0.95, premium_pct * 0.5)
            btc_reduction = min(0.95, btc_premium_pct * 0.5 / 100.0)
            eth_reduction = min(0.95, eth_premium_pct * 0.5 / 100.0)

            cov_adj = apply_protective_put_to_cov(cov_mat, asset_names, btc_idx, eth_idx, btc_reduction, eth_reduction)
            new_risk = np.sqrt(np.dot(weights.T, np.dot(cov_adj, weights))) * 100
            total_premium = btc_total_premium + eth_total_premium

            if new_risk <= target_risk_pct:
                # choose best by minimal premium then minimal (b+e)
                if best is None or (total_premium < best["total_premium"]) or (total_premium == best["total_premium"] and (b+e) < (best["b"]+best["e"])):
                    best = {"b": b, "e": e, "new_risk": new_risk, "btc_reduction": btc_reduction, "eth_reduction": eth_reduction, "total_premium": total_premium, "btc_total_premium": btc_total_premium, "eth_total_premium": eth_total_premium}
    return best

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

    # bounds based on hedge strategy
    bounds = []
    hedge = hedge_strategies[st.session_state.hedge_strategy]
    for name in asset_names:
        low = 0.0; up = 1.0; n = name.upper()
        if any(x in n for x in ["GC=", "GOLD", "SI="]): low = max(low, hedge["gold_min"])
        if any(x in n for x in ["USD", "USDIRR", "USDT"]): low = max(low, hedge["usd_min"])
        if any(x in n for x in ["BTC", "بیت"]): up = min(up, hedge["btc_max"])
        if low > up: low, up = 0.0, 1.0
        bounds.append((float(low), float(up)))

    weights = get_portfolio_weights(st.session_state.selected_style, returns, mean_ret, cov_mat, rf, bounds)

    opt = option_strategies[st.session_state.option_strategy]
    option_cost = opt["cost_pct"]
    adjusted_return = np.dot(mean_ret, weights) * 100 - option_cost
    adjusted_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
    if "Put" in st.session_state.option_strategy:
        adjusted_risk *= 0.7
    elif "Call" in st.session_state.option_strategy:
        adjusted_risk *= 1.1

    sharpe = (adjusted_return/100 - rf) / (adjusted_risk/100) if adjusted_risk>0 else 0
    recovery = format_recovery(calculate_recovery_time(returns.dot(weights)))

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 خلاصه", "💰 تخصیص دارایی", "🔮 پیش‌بینی قیمت", "📈 بک‌تست", "🛡️ Protective (Married) Put"])

    with tab1:
        st.markdown("### 📋 خلاصه پرتفوی")
        is_option_active = st.session_state.option_strategy != "بدون آپشن"
        st.success(f"**سبک:** {st.session_state.selected_style} | **آپشن:** {opt['name']}")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📈 بازده" + (" (با آپشن)" if is_option_active else ""), f"{adjusted_return:.2f}%")
        c2.metric("⚠️ ریسک" + (" (با آپشن)" if is_option_active else ""), f"{adjusted_risk:.2f}%")
        c3.metric("⭐ شارپ" + (" (با آپشن)" if is_option_active else ""), f"{sharpe:.3f}")
        c4.metric("⏱️ زمان ریکاوری", recovery)
        st.markdown("---")
        df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100,2)}).sort_values("وزن (%)", ascending=False)
        col1, col2 = st.columns([2,1])
        with col1: st.markdown("### وزن‌های دارایی"); st.dataframe(df_w, use_container_width=True, hide_index=True)
        with col2: st.markdown("### نمودار توزیع"); st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی"), use_container_width=True)

    with tab2:
        st.markdown("### 💰 ماشین حساب تخصیص دارایی")
        col1, col2 = st.columns([2,1])
        with col1:
            total_usd = st.number_input("💵 کل سرمایه (دلار) — برای محاسبات exposure", min_value=100, max_value=20_000_000, value=1200, step=100)
            df_alloc = capital_allocator_calculator(weights, asset_names, total_usd, (st.session_state.get("exchange_rate",200_000_000)))
            st.dataframe(df_alloc[["دارایی","درصد وزن","دلار ($)"]], use_container_width=True, hide_index=True)
        with col2:
            exchange_rate = st.number_input("💱 نرخ تبدیل (دل��ر -> تومان)", min_value=100_000, max_value=500_000_000, value=200_000_000, step=1_000_000)
            st.session_state["exchange_rate"] = exchange_rate
            st.metric("💵 کل سرمایه (دلار)", f"${total_usd:,.2f}")
            st.metric("💴 معادل تومان (تقریبی)", f"{total_usd * (exchange_rate/1_000_000):,.0f} تومان")
        st.markdown("---")
        csv = df_alloc[["دارایی","درصد وزن","دلار ($)"]].to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 دانلود تخصیص (CSV)", csv, file_name=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    with tab3:
        st.markdown("### 🔮 پیش‌بینی قیمت (Monte Carlo)")
        selected_assets = st.multiselect("انتخاب دارایی‌ها برای پیش‌بینی", asset_names, default=asset_names[:min(2,len(asset_names))])
        if selected_assets:
            days = st.slider("روزهای پیش‌بینی", 30, 365, 90)
            for asset in selected_assets:
                st.markdown(f"#### {asset}")
                fig = plot_forecast_single(prices[asset], asset, days_default=days)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("یک یا چند دارایی انتخاب کنید.")

    with tab4:
        st.markdown("### 📈 بک‌تست (چه می‌شد اگر؟)")
        col1,col2,col3 = st.columns(3)
        initial = col1.number_input("💰 سرمایه اولیه (میلیون تومان)", 10, 10000, 100)
        years = col2.selectbox("📅 چند سال پیش شروع کرده بودید؟", [1,3,5,10], index=2)
        monthly = col3.number_input("📊 سرمایه ماهیانه (میلیون)", 0, 100, 10)
        full_returns = prices.pct_change().dropna()
        port_daily = full_returns.dot(weights)
        backtest_days = years * 252
        if len(port_daily) > backtest_days: port_daily = port_daily.tail(backtest_days)
        value = initial; values=[initial]
        for i in range(len(port_daily)):
            value *= (1 + port_daily.iloc[i])
            if i % 21 == 0 and i>0: value += monthly
            values.append(value)
        total_invested = initial + monthly * years * 12
        profit = value - total_invested
        profit_pct = (profit/total_invested)*100 if total_invested>0 else 0
        col1,col2,col3 = st.columns(3)
        col1.metric("💎 سرمایه نهایی", f"{value:,.0f} میلیون", delta=f"{profit_pct:.1f}%"); col2.metric("💵 سود خالص", f"{profit:,.0f} میلیون"); col3.metric("📊 نسبت سود/سرمایه", f"{(profit/total_invested):.1%}" if total_invested>0 else "0%")
        fig_back = go.Figure(); fig_back.add_trace(go.Scatter(y=values, name="رشد سرمایه", mode="lines", fill="tozeroy", line=dict(color="green"))); fig_back.add_hline(y=initial, line_dash="dash", annotation_text="سرمایه اولیه", line_color="red"); fig_back.update_layout(title=f"رشد سرمایه از {years} سال پیش", xaxis_title="روز", yaxis_title="میلیون تومان", template="plotly_white"); st.plotly_chart(fig_back, use_container_width=True)

    with tab5:
        st.markdown("### 🛡️ Protective (Married) Put — بیمهٔ مستقیم دارایی (Long Put)")
        st.info("در این بخش می‌توانید برای BTC-USD و ETH-USD قرارداد Put بخرید (Married Put). هزینه‌ها و تعداد قراردادها را وارد کنید؛ سیستم تأثیر بیمه را روی ریسک کل پرتفوی محاسبه و نمایش می‌دهد.")
        # find BTC and ETH columns
        btc_idx = None; eth_idx = None
        for i, name in enumerate(asset_names):
            if "BTC" in name.upper(): btc_idx = i
            if "ETH" in name.upper(): eth_idx = i
        if btc_idx is None and eth_idx is None:
            st.error("برای Protective Put نیاز به BTC و/یا ETH در پرتفوی است.")
            return

        col1, col2 = st.columns(2)
        # get total_usd from earlier tab (or default)
        total_usd = st.session_state.get("last_total_usd", 1200)
        # allow user to enter (or use previously entered) total_usd for accurate exposure
        total_usd = col1.number_input("💵 (برای محاسبه exposure) کل سرمایه پرتفوی (دلار)", min_value=100, max_value=20_000_000, value=total_usd, step=100)
        st.session_state["last_total_usd"] = total_usd

        # Per-contract premium estimates for suggestion algorithm
        st.markdown("#### 🔧 برآورد حق‌الزحمه (برای پیشنهاد تعداد قراردادها)")
        est_col1, est_col2 = st.columns(2)
        est_btc_prem = None; est_eth_prem = None
        if btc_idx is not None:
            est_btc_prem = est_col1.number_input("برآورد Premium هر قرارداد BTC ($) — برای پیشنهاد", min_value=0.0, value=float(prices.iloc[-1, btc_idx]*0.04), step=1.0, key="btc_premium_est_input")
            st.session_state["btc_premium_est"] = est_btc_prem
        if eth_idx is not None:
            est_eth_prem = est_col2.number_input("برآورد Premium هر قرارداد ETH ($) — برای پیشنهاد", min_value=0.0, value=float(prices.iloc[-1, eth_idx]*0.04), step=0.5, key="eth_premium_est_input")
            st.session_state["eth_premium_est"] = est_eth_prem

        st.markdown("---")
        st.markdown("#### 📝 مشخصات Protective Put (Married Put) — ورودی‌ها")
        c1,c2 = st.columns(2)
        # BTC inputs
        if btc_idx is not None:
            with c1:
                st.markdown("##### 🔵 BTC-USD")
                btc_price = prices.iloc[-1, btc_idx]
                st.write(f"قیمت فع��ی BTC: ${btc_price:,.2f}")
                btc_strike = st.number_input("Strike برای BTC ($)", min_value=btc_price*0.5, max_value=btc_price*0.999, value=btc_price*0.90, step=50.0, key="btc_strike_input")
                btc_premium = st.number_input("Premium برای هر قرارداد BTC ($)", min_value=0.0, value=max(1.0, btc_price*0.04), step=1.0, key="btc_premium_input")
                btc_contracts = st.number_input("تعداد قراردادهای BTC", min_value=0, max_value=500, value=0, step=1, key="btc_contracts_input")
                btc_contract_size = st.number_input("اندازه قرارداد (BTC در هر قرارداد)", min_value=0.01, max_value=100.0, value=1.0, step=0.01, key="btc_contract_size_input")
        else:
            btc_strike=btc_premium=btc_contracts=btc_contract_size=None

        # ETH inputs
        if eth_idx is not None:
            with c2:
                st.markdown("##### 🟢 ETH-USD")
                eth_price = prices.iloc[-1, eth_idx]
                st.write(f"قیمت فعلی ETH: ${eth_price:,.2f}")
                eth_strike = st.number_input("Strike برای ETH ($)", min_value=eth_price*0.5, max_value=eth_price*0.999, value=eth_price*0.90, step=5.0, key="eth_strike_input")
                eth_premium = st.number_input("Premium برای هر قرارداد ETH ($)", min_value=0.0, value=max(0.5, eth_price*0.04), step=0.5, key="eth_premium_input")
                eth_contracts = st.number_input("تعداد قراردادهای ETH", min_value=0, max_value=500, value=0, step=1, key="eth_contracts_input")
                eth_contract_size = st.number_input("اندازه قرارداد (ETH در هر قرارداد)", min_value=0.01, max_value=1000.0, value=1.0, step=0.01, key="eth_contract_size_input")
        else:
            eth_strike=eth_premium=eth_contracts=eth_contract_size=None

        st.markdown("---")
        # compute premiums and premium_pct relative to exposure
        exposures = {asset_names[i]: weights[i]*total_usd for i in range(len(asset_names))}
        # BTC totals
        btc_total_premium = 0.0
        btc_total_premium_pct = 0.0
        if btc_idx is not None and btc_contracts > 0:
            btc_total_premium = btc_premium * btc_contracts * btc_contract_size
            exposure_btc = exposures.get(asset_names[btc_idx], 1e-8)
            btc_total_premium_pct = (btc_total_premium / (exposure_btc + 1e-8)) * 100
        # ETH totals
        eth_total_premium = 0.0
        eth_total_premium_pct = 0.0
        if eth_idx is not None and eth_contracts > 0:
            eth_total_premium = eth_premium * eth_contracts * eth_contract_size
            exposure_eth = exposures.get(asset_names[eth_idx], 1e-8)
            eth_total_premium_pct = (eth_total_premium / (exposure_eth + 1e-8)) * 100

        # map premium_pct -> volatility reduction (model)
        # reduction = min(0.95, premium_pct * 0.5 / 100)
        btc_reduction = min(0.95, btc_total_premium_pct * 0.5 / 100.0)
        eth_reduction = min(0.95, eth_total_premium_pct * 0.5 / 100.0)

        # adjusted covariance after applying "protective" effect
        cov_adj = apply_protective_put_to_cov(cov_mat, asset_names, btc_idx, eth_idx, btc_reduction, eth_reduction)
        original_portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
        new_portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_adj, weights))) * 100
        risk_reduction_abs = original_portfolio_risk - new_portfolio_risk
        risk_reduction_pct_total = (risk_reduction_abs / original_portfolio_risk * 100) if original_portfolio_risk>0 else 0.0

        # display metrics
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("📊 ریسک پرتفوی (بدون بیمه)", f"{original_portfolio_risk:.2f}%")
        col2.metric("🛡️ ریسک پرتفوی (با بیمه)", f"{new_portfolio_risk:.2f}%")
        col3.metric("📉 کاهش ریسک (مطلق)", f"{risk_reduction_abs:.3f}%")
        col4.metric("📉 کاهش ریسک (%)", f"{risk_reduction_pct_total:.2f}%")

        st.markdown("---")
        # Payoff chart prepared similarly to screenshot: combined payoff of BTC and ETH protective puts (USD)
        st.markdown("#### 📈 نمودار Payoff (تاثیر بیمه بر BTC و ETH)")
        # choose slider price ranges based on current prices
        price_min_mult = st.slider("زوم قیمت: کوچک‌تر از (نسبت به قیمت فعلی)", 50, 90, 80)
        price_max_mult = st.slider("زوم قیمت: بزرگ‌تر از (نسبت به قیمت فعلی)", 110, 150, 120)
        # build ranges
        btc_price_min = (btc_strike if btc_strike is not None else (prices.iloc[-1, btc_idx] if btc_idx is not None else 0)) * (price_min_mult/100.0) if btc_idx is not None else 0
        btc_price_max = (btc_price if btc_idx is not None else 1) * (price_max_mult/100.0) if btc_idx is not None else 1
        eth_price_min = (eth_strike if eth_strike is not None else (prices.iloc[-1, eth_idx] if eth_idx is not None else 0)) * (price_min_mult/100.0) if eth_idx is not None else 0
        eth_price_max = (eth_price if eth_idx is not None else 1) * (price_max_mult/100.0) if eth_idx is not None else 1

        # create combined price axis (we will plot both on their own scales side-by-side by overlaying ranges)
        n_points = 200
        fig = go.Figure()
        # BTC payoff
        if btc_idx is not None:
            price_range_btc = np.linspace(btc_price_min, btc_price_max, n_points)
            btc_payoff = [max(btc_strike - p, 0) * btc_contract_size * btc_contracts - btc_total_premium for p in price_range_btc]
            fig.add_trace(go.Scatter(x=price_range_btc, y=btc_payoff, name="BTC Protective Put (USD)", mode="lines", line=dict(color="orange", width=2), fill='tozeroy', fillcolor='rgba(255,165,0,0.15)'))
            # BE vertical for BTC
            btc_be = btc_strike - (btc_total_premium / (btc_contract_size*btc_contracts+1e-8)) if btc_contracts>0 else None
            if btc_be:
                fig.add_vline(x=btc_be, line_dash="dot", line_color="orange", annotation_text=f"BTC BE: ${btc_be:.2f}", annotation_position="top left")
        # ETH payoff
        if eth_idx is not None:
            price_range_eth = np.linspace(eth_price_min, eth_price_max, n_points)
            eth_payoff = [max(eth_strike - p, 0) * eth_contract_size * eth_contracts - eth_total_premium for p in price_range_eth]
            fig.add_trace(go.Scatter(x=price_range_eth, y=eth_payoff, name="ETH Protective Put (USD)", mode="lines", line=dict(color="blue", width=2), fill='tozeroy', fillcolor='rgba(0,123,255,0.12)'))
            eth_be = eth_strike - (eth_total_premium / (eth_contract_size*eth_contracts+1e-8)) if eth_contracts>0 else None
            if eth_be:
                fig.add_vline(x=eth_be, line_dash="dot", line_color="blue", annotation_text=f"ETH BE: ${eth_be:.2f}", annotation_position="top right")
        # combined payoff: sample aligned price grid by taking union and interpolating
        all_prices = np.unique(np.concatenate([
            (price_range_btc if btc_idx is not None else np.array([])),
            (price_range_eth if eth_idx is not None else np.array([]))
        ]))
        if len(all_prices) > 0:
            combined_payoff = np.zeros_like(all_prices)
            if btc_idx is not None:
                from numpy import interp
                combined_payoff += interp(all_prices, price_range_btc, btc_payoff)
            if eth_idx is not None:
                combined_payoff += interp(all_prices, price_range_eth, eth_payoff)
            fig.add_trace(go.Scatter(x=all_prices, y=combined_payoff, name="Total Protective Payoff (USD)", mode="lines", line=dict(color="green", width=3)))
            # mark total BE approx where combined crosses zero (last crossing)
            sign = np.sign(combined_payoff)
            cross_idx = np.where(np.diff(sign) != 0)[0]
            if cross_idx.size > 0:
                be_price = all_prices[cross_idx[-1]]
                fig.add_vline(x=be_price, line_dash="dash", line_color="black", annotation_text=f"Total BE ~ ${be_price:.2f}", annotation_position="bottom right")

        fig.update_layout(title="📈 Payoff Protective Put (BTC & ETH)", xaxis_title="Price ($)", yaxis_title="Payoff (USD)", template='plotly_white', height=480)
        st.plotly_chart(fig, use_container_width=True)

        # Suggest minimal contracts to reach target risk (2%) if user wants
        st.markdown("---")
        st.markdown("#### 🤖 پیشنهاد تعداد قراردادها برای رسیدن به هدف ریسک")
        target_risk = st.number_input("هدف ریسک کل پرتفوی (%)", min_value=0.5, max_value=20.0, value=2.0, step=0.1, key="target_risk_input")
        max_search = st.number_input("حداکثر قرارداد برای جستجو در هر دارایی (برای پیشنهاد)", min_value=1, max_value=200, value=30, step=1, key="max_search_input")
        if st.button("🔎 پیشنهاد بده"):
            # need est premiums per contract to search - we earlier saved them as est_btc_prem and est_eth_prem
            st.session_state["btc_premium_est"] = est_btc_prem if btc_idx is not None else None
            st.session_state["eth_premium_est"] = est_eth_prem if eth_idx is not None else None
            suggestion = suggest_contracts_for_target_risk(prices, returns, asset_names, weights, cov_mat, total_usd, btc_idx, eth_idx, btc_contract_size if btc_idx is not None else 1.0, eth_contract_size if eth_idx is not None else 1.0, max_contracts=int(max_search), target_risk_pct=float(target_risk))
            if suggestion is None:
                st.error("برای ارائه پیشنهاد نیاز به برآورد premium به ازای هر قرارداد دارید (در بالا وارد کنید).")
            elif suggestion is False:
                st.info("نتیجه‌ای یافت نشد.")
            elif suggestion is None:
                st.info("پیشنهادی وجود ندارد.")
            else:
                st.success("پیشنهاد یافت شد (کمترین هزینه برای رسیدن به هدف ریسک)")
                st.markdown(f"- تعداد قرارداد BTC پیشنهادی: {suggestion['b']}\n- تعداد قرارداد ETH پیشنهادی: {suggestion['e']}\n- ریسک جدید پرتفوی: {suggestion['new_risk']:.3f}%\n- هزینه کل Premium (USD): ${suggestion['total_premium']:.2f}")
                # show detailed
                st.json({
                    "btc_total_premium": suggestion['btc_total_premium'],
                    "eth_total_premium": suggestion['eth_total_premium'],
                    "btc_reduction": suggestion['btc_reduction'],
                    "eth_reduction": suggestion['eth_reduction']
                })

# ==================== UI اولیه و سایدبار ====================
st.set_page_config(page_title="Portfolio360 Ultimate Pro", layout="wide")
st.markdown("""
<div style='text-align:center;padding:12px;background:linear-gradient(90deg,#00b4d8,#0077b6);border-radius:6px;color:white'>
  <h2 style='margin:0'>💼 Portfolio360 Ultimate Pro — Protective Put (Married Put) Integrated</h2>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 دانلود داده")
    tickers = st.text_input("نمادها (با کاما جدا کنید)", "BTC-USD, GC=F, USDIRR=X, ^GSPC, ETH-USD")
    if st.button("🔄 دانلود داده"):
        with st.spinner("در حال دانلود..."):
            data = download_data(tickers)
            if data is not None:
                st.session_state.prices = data
                st.success(f"✅ {len(data.columns)} دارایی بارگذاری شد!")
                st.experimental_rerun()

    st.markdown("---")
    st.header("⚙️ تنظیمات")
    if "hedge_strategy" not in st.session_state: st.session_state.hedge_strategy = "طلا + تتر (ترکیبی)"
    st.session_state.hedge_strategy = st.selectbox("استراتژی هجینگ", list(hedge_strategies.keys()), index=list(hedge_strategies.keys()).index(st.session_state.hedge_strategy))
    if "option_strategy" not in st.session_state: st.session_state.option_strategy = "بدون آپشن"
    st.session_state.option_strategy = st.selectbox("استراتژی آپشن", list(option_strategies.keys()))
    styles = [
        "مارکوویتز + هجینگ (بهینه‌ترین شارپ)","وزن برابر (ساده و مقاوم)","حداقل ریسک (محافظه‌کارانه)","ریسک‌پاریتی (Risk Parity)",
        "مونت‌کارلو مقاوم (Resampled Frontier)","HRP (سلسله‌مراتبی)","Maximum Diversification","Inverse Volatility",
        "Barbell طالب (۹۰/۱۰)","Antifragile طالب","Kelly Criterion (حداکثر رشد)","Most Diversified Portfolio","Equal Risk Bounding","بلک-لیترمن (ترکیب نظر شخصی)"
    ]
    if "selected_style" not in st.session_state: st.session_state.selected_style = styles[0]
    st.session_state.selected_style = st.selectbox("انتخاب سبک", styles, index=styles.index(st.session_state.selected_style))
    if "rf_rate" not in st.session_state: st.session_state.rf_rate = 18.0
    st.session_state.rf_rate = st.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)

# compute and render
calculate_portfolio()

st.caption(f"Portfolio360 Ultimate Pro — updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
