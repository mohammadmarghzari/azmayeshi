# Portfolio360 Ultimate Pro — Full app with correct Married (Protective) Put calculations and example scenarios
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

# --------------------- Utility functions ---------------------
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
        except Exception:
            failed.append(t)
    if not data:
        st.error("هیچ داده‌ای دانلود نشد.")
        return None
    prices = pd.DataFrame(data).ffill().bfill()
    if failed:
        st.sidebar.warning(f"دانلود نشد: {', '.join(failed)}")
    return prices

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

def forecast_price_series(price_series, days=63, sims=500):
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_ret) < 2:
        # fallback to geometric Brownian with small sigma
        mu = 0.0
        sigma = 0.01
    else:
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

def apply_protective_put_to_cov(cov_mat, asset_names, btc_idx, eth_idx, btc_reduction, eth_reduction):
    cov_adj = cov_mat.copy().astype(float)
    n = cov_adj.shape[0]
    scale = np.ones(n)
    if btc_idx is not None:
        scale[btc_idx] = max(0.0, 1.0 - btc_reduction)
    if eth_idx is not None:
        scale[eth_idx] = max(0.0, 1.0 - eth_reduction)
    for i in range(n):
        for j in range(n):
            cov_adj.iloc[i, j] = cov_mat.iloc[i, j] * scale[i] * scale[j]
    return cov_adj

# --------------------- Portfolio strategies (simplified) ---------------------
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

def get_portfolio_weights(style, returns, mean_ret, cov_mat, rf, bounds):
    # For simplicity we use equal weight fallback; full implementations from previous versions can be plugged in.
    n = len(mean_ret)
    w = np.ones(n) / n
    return w

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

# --------------------- Suggestion helper ---------------------
def suggest_contracts_for_target_risk(prices, returns, asset_names, weights, cov_mat, total_usd, btc_idx, eth_idx, btc_contract_size, eth_contract_size, est_btc_prem, est_eth_prem, max_contracts=20, target_risk_pct=2.0):
    best = None
    exposures = {name: weights[i]*total_usd for i, name in enumerate(asset_names)}
    btc_name = asset_names[btc_idx] if btc_idx is not None else None
    eth_name = asset_names[eth_idx] if eth_idx is not None else None
    for b in range(0, max_contracts+1):
        for e in range(0, max_contracts+1):
            btc_total_premium = b * est_btc_prem * btc_contract_size if btc_idx is not None else 0.0
            eth_total_premium = e * est_eth_prem * eth_contract_size if eth_idx is not None else 0.0
            btc_premium_pct = (btc_total_premium / (exposures.get(btc_name,1e-8))) * 100 if btc_name else 0.0
            eth_premium_pct = (eth_total_premium / (exposures.get(eth_name,1e-8))) * 100 if eth_name else 0.0
            btc_reduction = min(0.95, btc_premium_pct * 0.5 / 100.0)
            eth_reduction = min(0.95, eth_premium_pct * 0.5 / 100.0)
            cov_adj = apply_protective_put_to_cov(cov_mat, asset_names, btc_idx, eth_idx, btc_reduction, eth_reduction)
            new_risk = np.sqrt(np.dot(weights.T, np.dot(cov_adj, weights))) * 100
            total_premium = btc_total_premium + eth_total_premium
            if new_risk <= target_risk_pct:
                if best is None or total_premium < best["total_premium"] or (total_premium==best["total_premium"] and (b+e)<(best["b"]+best["e"])):
                    best = {"b":b,"e":e,"new_risk":new_risk,"btc_total_premium":btc_total_premium,"eth_total_premium":eth_total_premium,"btc_reduction":btc_reduction,"eth_reduction":eth_reduction,"total_premium":total_premium}
    return best

# --------------------- Plot helpers ---------------------
def plot_married_payoff_for_asset(S0, strike, premium, units_held, contracts, contract_size, name, color):
    """
    Returns (fig_traces, grid, married_pnl) for this asset.
    married_pnl computed per price grid (USD).
    """
    # choose sensible grid (80% .. 120%)
    p_min = max(0.01, S0 * 0.5)
    p_max = S0 * 1.5
    grid = np.linspace(p_min, p_max, 400)
    underlying_pnl = (grid - S0) * units_held  # per price
    put_payout = np.maximum(strike - grid, 0.0) * (contracts * contract_size)
    total_premium = premium * contracts * contract_size
    married_pnl = underlying_pnl + put_payout - total_premium
    # create traces: total married pnl line, fill positive green, negative red
    trace_line = go.Scatter(x=grid, y=married_pnl, name=f"{name} Married Put (USD)", mode="lines", line=dict(color=color, width=2))
    trace_positive = go.Scatter(x=grid, y=np.where(married_pnl>=0, married_pnl, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(50,205,50,0.12)', showlegend=False)
    trace_negative = go.Scatter(x=grid, y=np.where(married_pnl<0, married_pnl, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(255,99,71,0.12)', showlegend=False)
    return (trace_line, trace_positive, trace_negative, grid, married_pnl, total_premium)

# --------------------- Main calculation and UI ---------------------
@st.fragment
def calculate_portfolio():
    if "prices" not in st.session_state or st.session_state.prices is None:
        st.info("ابتدا داده‌ها را دانلود کنید.")
        return

    prices = st.session_state.prices
    returns = prices.pct_change().dropna()
    asset_names = list(prices.columns)
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    rf = st.session_state.get("rf_rate", 18.0) / 100.0

    # weights (fallback equal)
    weights = get_portfolio_weights(None, returns, mean_ret, cov_mat, rf, None)

    # UI tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 خلاصه", "💰 تخصیص", "🔮 پیش‌بینی", "📈 بک‌تست", "🛡️ Protective (Married) Put"])

    with tab1:
        st.markdown("### خلاصه پرتفوی")
        df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100,2)})
        st.dataframe(df_w, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### ماشین حساب تخصیص")
        total_usd = st.number_input("کل سرمایه برای تخصیص (دلار)", min_value=100, value=1200, step=100)
        exchange_rate = st.number_input("نرخ تبدیل (دلار -> تومان)", min_value=100000, value=200000000, step=1000000)
        df_alloc = capital_allocator_calculator(weights, asset_names, total_usd, exchange_rate)
        st.dataframe(df_alloc[["دارایی","درصد وزن","دلار ($)"]], use_container_width=True, hide_index=True)
        csv = df_alloc[["دارای��","درصد وزن","دلار ($)"]].to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 دانلود تخصیص CSV", csv, file_name=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    with tab3:
        st.markdown("### پیش‌بینی قیمتها (Monte Carlo)")
        assets_sel = st.multiselect("دارایی‌ها", asset_names, default=asset_names[:min(2,len(asset_names))])
        if assets_sel:
            days = st.slider("روزهای پیش‌بینی", 30, 365, 90)
            for a in assets_sel:
                fig = go.Figure()
                fig = plot_forecast_single(prices[a], a, days_default=days)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("یک یا چند دارایی انتخاب کنید.")

    with tab4:
        st.markdown("### بک‌تست (خلاصه)")
        st.write("بک‌تست خلاصه — بخش قابل گسترش")

    with tab5:
        st.markdown("### Protective (Married) Put — محاسبه و نمودار صحیح")
        st.info("در Married (Protective) Put شما دارایی پایه را نگه می‌دارید و به ازای واحدهای نگهداری، قرارداد Put می‌خرید (تعداد قرارداد برابر واحدهای نگهداری یا کسری از آن). PnL نهایی = PnL underlying + put payoff - total premium.")

        # find BTC and ETH indices (if present)
        btc_idx = None; eth_idx = None
        for i, name in enumerate(asset_names):
            if "BTC" in name.upper(): btc_idx = i
            if "ETH" in name.upper(): eth_idx = i

        st.markdown("#### تنظیمات و ورودی‌ها")
        colA, colB = st.columns(2)
        total_usd = colA.number_input("کل سرمایه مرجع (دلار) — برای محاسبه exposure", min_value=100, value=1200, step=100)
        exchange_rate = colB.number_input("نرخ تبدیل (دلار->تومان)", min_value=100000, value=200000000, step=1000000)

        exposures = {asset_names[i]: weights[i]*total_usd for i in range(len(asset_names))}

        st.markdown("#### مشخصات قرارداد (Married Put) — ورودی کاربر")
        c1, c2 = st.columns(2)

        # default example parameters (for the example you provided)
        ex_S0 = 50.0
        ex_premium = 2.0
        ex_strike = 50.0

        # BTC inputs
        if btc_idx is not None:
            with c1:
                st.markdown("🔵 BTC-USD (اگر در پرتفوی دارید)")
                btc_price = float(prices.iloc[-1, btc_idx])
                st.write(f"قیمت فعلی BTC: ${btc_price:,.2f}")
                btc_strike = st.number_input("Strike BTC ($)", value=btc_price*0.90, step=10.0, key="btc_strike")
                btc_premium = st.number_input("Premium هر قرارداد BTC ($)", value=max(0.0, btc_price*0.04), step=1.0, key="btc_premium")
                btc_contracts = st.number_input("تعداد قرارداد BTC (برای Married Put)", min_value=0, max_value=1000, value=0, step=1, key="btc_contracts")
                btc_contract_size = st.number_input("حجم در هر قرارداد (BTC)", min_value=0.01, max_value=100.0, value=1.0, step=0.01, key="btc_size")
        else:
            btc_price = btc_strike = btc_premium = btc_contracts = btc_contract_size = None

        # ETH inputs
        if eth_idx is not None:
            with c2:
                st.markdown("🟢 ETH-USD (اگر در پرتفوی دارید)")
                eth_price = float(prices.iloc[-1, eth_idx])
                st.write(f"قیمت فعلی ETH: ${eth_price:,.2f}")
                eth_strike = st.number_input("Strike ETH ($)", value=eth_price*0.90, step=5.0, key="eth_strike")
                eth_premium = st.number_input("Premium هر قرارداد ETH ($)", value=max(0.0, eth_price*0.04), step=0.5, key="eth_premium")
                eth_contracts = st.number_input("تعداد قرارداد ETH (برای Married Put)", min_value=0, max_value=1000, value=0, step=1, key="eth_contracts")
                eth_contract_size = st.number_input("حجم در هر قرارداد (ETH)", min_value=0.01, max_value=1000.0, value=1.0, step=0.01, key="eth_size")
        else:
            eth_price = eth_strike = eth_premium = eth_contracts = eth_contract_size = None

        st.markdown("---")

        # compute units held from exposure: units = exposure_usd / current_price
        units_held_btc = 0.0
        if btc_idx is not None:
            exposure_btc = exposures.get(asset_names[btc_idx], 0.0)
            units_held_btc = exposure_btc / (btc_price + 1e-8)

        units_held_eth = 0.0
        if eth_idx is not None:
            exposure_eth = exposures.get(asset_names[eth_idx], 0.0)
            units_held_eth = exposure_eth / (eth_price + 1e-8)

        # total premiums
        btc_total_premium = (btc_premium * btc_contracts * btc_contract_size) if btc_idx is not None else 0.0
        eth_total_premium = (eth_premium * eth_contracts * eth_contract_size) if eth_idx is not None else 0.0

        # reductions for portfolio risk (approx model)
        btc_premium_pct = (btc_total_premium / (exposures.get(asset_names[btc_idx],1e-8))) * 100 if btc_idx is not None else 0.0
        eth_premium_pct = (eth_total_premium / (exposures.get(asset_names[eth_idx],1e-8))) * 100 if eth_idx is not None else 0.0
        btc_reduction = min(0.95, btc_premium_pct * 0.5 / 100.0)
        eth_reduction = min(0.95, eth_premium_pct * 0.5 / 100.0)
        cov_adj = apply_protective_put_to_cov(cov_mat, asset_names, btc_idx, eth_idx, btc_reduction, eth_reduction)
        original_portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
        new_portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_adj, weights))) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ریسک پرتفوی (بدون بیمه)", f"{original_portfolio_risk:.2f}%")
        col2.metric("ریسک پرتفوی (با Married Put)", f"{new_portfolio_risk:.2f}%")
        col3.metric("کاهش ریسک (نسبت)", f"{(original_portfolio_risk - new_portfolio_risk):.3f}%")
        total_prem_display = btc_total_premium + eth_total_premium
        col4.metric("کل Premium پرداختی (USD)", f"${total_prem_display:,.2f}")

        st.markdown("---")
        st.markdown("### نمودار Payoff — Married Put (Underlying + Long Put)")
        # Build traces
        fig = go.Figure()
        all_prices = np.array([])
        # BTC
        if btc_idx is not None and (btc_contracts>0 or btc_total_premium>0):
            t_line, t_pos, t_neg, grid_btc, married_pnl_btc, btc_prem_paid = plot_married_payoff_for_asset(
                S0=btc_price,
                strike=btc_strike,
                premium=btc_premium,
                units_held=units_held_btc,
                contracts=int(btc_contracts),
                contract_size=float(btc_contract_size),
                name="BTC",
                color="orange"
            )
            fig.add_trace(t_line); fig.add_trace(t_pos); fig.add_trace(t_neg)
            all_prices = np.concatenate([all_prices, grid_btc])
        # ETH
        if eth_idx is not None and (eth_contracts>0 or eth_total_premium>0):
            e_line, e_pos, e_neg, grid_eth, married_pnl_eth, eth_prem_paid = plot_married_payoff_for_asset(
                S0=eth_price,
                strike=eth_strike,
                premium=eth_premium,
                units_held=units_held_eth,
                contracts=int(eth_contracts),
                contract_size=float(eth_contract_size),
                name="ETH",
                color="blue"
            )
            fig.add_trace(e_line); fig.add_trace(e_pos); fig.add_trace(e_neg)
            all_prices = np.concatenate([all_prices, grid_eth])

        # Combined
        if all_prices.size > 0:
            common_min = float(np.nanmin(all_prices))
            common_max = float(np.nanmax(all_prices))
            common_grid = np.linspace(common_min, common_max, 600)
            total_payoff = np.zeros_like(common_grid)
            if btc_idx is not None and (btc_contracts>0 or btc_total_premium>0):
                from numpy import interp
                total_payoff += interp(common_grid, grid_btc, married_pnl_btc)
            if eth_idx is not None and (eth_contracts>0 or eth_total_premium>0):
                from numpy import interp
                total_payoff += interp(common_grid, grid_eth, married_pnl_eth)
            fig.add_trace(go.Scatter(x=common_grid, y=total_payoff, name="Total Married Put Payoff (USD)", mode="lines", line=dict(color="green", width=3)))
            # shade positive/negative for total
            fig.add_trace(go.Scatter(x=common_grid, y=np.where(total_payoff>=0, total_payoff, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(50,205,50,0.12)', showlegend=False))
            fig.add_trace(go.Scatter(x=common_grid, y=np.where(total_payoff<0, total_payoff, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(255,99,71,0.12)', showlegend=False))
            # BE total
            sign_t = np.sign(total_payoff)
            cross_t = np.where(np.diff(sign_t) != 0)[0]
            if cross_t.size > 0:
                be_total = common_grid[cross_t[-1]]
                fig.add_vline(x=be_total, line_dash="dash", line_color="black", annotation_text=f"Total BE ~ ${be_total:.2f}", annotation_position="bottom right")

        # Breakeven per asset (S0 + premium_per_share)
        if btc_idx is not None:
            be_btc = btc_price + (btc_premium if btc_premium is not None else 0.0)
            fig.add_vline(x=be_btc, line_dash="dot", line_color="orange", annotation_text=f"BTC BE = {be_btc:.2f}", annotation_position="top left")
        if eth_idx is not None:
            be_eth = eth_price + (eth_premium if eth_premium is not None else 0.0)
            fig.add_vline(x=be_eth, line_dash="dot", line_color="blue", annotation_text=f"ETH BE = {be_eth:.2f}", annotation_position="top right")

        fig.update_layout(title="Payoff — Married Put (Underlying + Long Put)", xaxis_title="Price ($)", yaxis_title="PnL (USD)", template='plotly_white', height=540)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### مثال عددی و سه سناریو (مطابق توضیح شما)")
        st.write("مثال پایه: فرض کنید سهم را در قیمت S0 = 50 خریدید و premium = 2 به ازای هر سهم پرداختید (قرارداد PUT با Strike = S0 خریدید — حالت ATM).")
        S0 = ex_S0
        p = ex_premium
        K = ex_strike

        scen = {
            "صعودی": 57.0,
            "ثابت": 50.0,
            "نزولی": 47.0
        }
        st.markdown(f"- قیمت خرید (S0): ${S0:.2f} — Strike مثال: ${K:.2f} — Premium: ${p:.2f} per share")
        st.markdown(f"- Breakeven = S0 + premium = ${S0 + p:.2f}")

        rows = []
        for name_s, ST in scen.items():
            underlying_pnl = ST - S0
            put_payoff = max(K - ST, 0)
            net = underlying_pnl + put_payoff - p
            rows.append({"سناریو": name_s, "S_T ($)": ST, "PnL underlying": underlying_pnl, "Put payoff": put_payoff, "Premium": -p, "Net PnL": net})
        df_scen = pd.DataFrame(rows)
        df_scen = df_scen[["سناریو","S_T ($)","PnL underlying","Put payoff","Premium","Net PnL"]]
        st.dataframe(df_scen, use_container_width=True)

        st.markdown("توضیح موارد:")
        st.write("1) اگر S_T = 57 → PnL underlying = 7 ، put payoff = 0 ، premium = 2 → Net = 7 - 2 = 5")
        st.write("2) اگر S_T = 50 → PnL underlying = 0 ، put payoff = 0 ، premium = 2 → Net = -2")
        st.write("3) اگر S_T = 47 → PnL underlying = -3 ، put payoff = 3 ، premium = 2 → Net = -2")

        # Max Loss (for the example)
        # If K == S0 (ATM), max loss = premium (p) as shown in example
        max_loss_per_share = K - S0 - p  # this will be negative (loss), but better to show absolute
        # More intuitive: worst-case net PnL when ST -> 0: married_pnl = 0 - S0 + K - p = K - S0 - p
        st.markdown("---")
        st.markdown("حداکثر ریسک و نقطه سر به سر")
        st.write(f"- نقطه سربه‌سر (Breakeven) در سررسید = S0 + premium = ${S0 + p:.2f}")
        if K == S0:
            st.write(f"- مثال ATM: اگر K = S0، حداکثر زیان برابر premium است: ${p:.2f} per share")
        else:
            worst = K - S0 - p
            st.write(f"- حداکثر زیان نظری: {worst:.2f} (اگر ST <= K) — در مثال واقعی معمولاً این مقدار منفی است و بیانگر زیان هر سهم است.")

        st.markdown("---")
        st.markdown("اگر خواستید من همین الان برای پرتفوی شما پیشنهاد تعداد قرارداد بدهم تا ریسک کل را به هدفی مثل 2% برسانیم، مقدار Premium تخمینی برای هر قرارداد BTC/ETH را وارد کنید و دکمه پیشنهاد را بزنید.")
        st.write("پیشنهاد مبتنی بر مدل تقریبی premium->کاهش volatility است؛ برای مدل دقیق‌تر باید implied vol / delta / قراردادهای واقعی استفاده شود.")

        st.markdown("### پیشنهاد خودکار (اختیاری)")
        est_btc_prem = st.number_input("برآورد Premium هر قرارداد BTC ($) — برای پیشنهاد", value=float(btc_premium if btc_premium is not None else 0.0), step=1.0)
        est_eth_prem = st.number_input("برآورد Premium هر قرارداد ETH ($) — برای پیشنهاد", value=float(eth_premium if eth_premium is not None else 0.0), step=0.5)
        target_risk = st.number_input("هدف ریسک کل پرتفوی (%)", min_value=0.5, max_value=20.0, value=2.0, step=0.1)
        max_search = st.number_input("حداکثر قرارداد برای جستجو (هر دارایی)", min_value=1, max_value=200, value=30, step=1)
        if st.button("پیشنهاد بده برای رسیدن به هدف ریسک"):
            suggestion = suggest_contracts_for_target_risk(prices, returns, asset_names, weights, cov_mat, total_usd, btc_idx, eth_idx, float(btc_contract_size if btc_contract_size else 1.0), float(eth_contract_size if eth_contract_size else 1.0), float(est_btc_prem), float(est_eth_prem), max_contracts=int(max_search), target_risk_pct=float(target_risk))
            if suggestion:
                st.success(f"پیشنهاد: BTC contracts={suggestion['b']} — ETH contracts={suggestion['e']} — هزینه کل ${suggestion['total_premium']:.2f} — ریسک جدید {suggestion['new_risk']:.3f}%")
            else:
                st.info("پیشنهادی یافت نشد یا اطلاعات پریمیوم کافی نبود.")

# --------------------- UI Sidebar & boot ---------------------
st.set_page_config(page_title="Portfolio360 Ultimate Pro", layout="wide")
st.markdown("<h2 style='text-align:center;color:#00a3cc'>Portfolio360 Ultimate Pro — Married Put (Protective) — Corrected</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 دانلود داده")
    tickers = st.text_input("نمادها (با کاما جدا کنید)", "BTC-USD, ETH-USD, GC=F, USDIRR=X, ^GSPC")
    if st.button("🔄 دانلود داده"):
        with st.spinner("در حال دانلود..."):
            data = download_data(tickers)
            if data is not None:
                st.session_state.prices = data
                st.success(f"✅ {len(data.columns)} دارایی بارگذاری شد!")
                st.experimental_rerun()

    st.markdown("---")
    st.header("⚙️ تنظیمات")
    if "option_strategy" not in st.session_state: st.session_state.option_strategy = "بدون آپشن"
    st.session_state.option_strategy = st.selectbox("استراتژی آپشن", list(option_strategies.keys()))
    if "rf_rate" not in st.session_state: st.session_state.rf_rate = 18.0
    st.session_state.rf_rate = st.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)

# run
calculate_portfolio()
st.caption(f"Portfolio360 Ultimate Pro — updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
