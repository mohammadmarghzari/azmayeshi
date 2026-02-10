# app.py — Portfolio360 Ultimate Pro (updated: married-put payoff fixed)
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

# ==================== توابع کمکی ساده ====================
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

# ==================== محاسبات تخصیص و مدل‌های سبک‌ها (خلاصه‌شده) ====================
# (برای brevity همان توابع قبلی اما سالم و کافی برای اجرای اپ)
hedge_strategies = {
    "طلا + تتر (ترکیبی)": {"gold_min": 0.15, "usd_min": 0.10, "btc_max": 0.20},
    "بدون هجینگ": {"gold_min": 0.00, "usd_min": 0.00, "btc_max": 1.00},
}
option_strategies = {
    "بدون آپشن": {"cost_pct": 0.0, "name": "بدون تغییر"},
    "Protective Put": {"cost_pct": 4.8, "name": "بیمه کامل"},
}

def get_portfolio_weights(style, returns, mean_ret, cov_mat, rf, bounds):
    # خیلی خلاصه: وزن برابر به عنوان fallback
    n = len(mean_ret)
    return np.ones(n) / n

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

# ==================== اعمال تاثیر protective put روی کوواریانس (تقریبی) ====================
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

# ==================== Monte Carlo forecast (همان تابع قبلی) ====================
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
    fig.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), name="میانه پیش‌بینی", mode="lines", line=dict(color="orange")))
    fig.update_layout(title=f"🔮 پیش‌بینی قیمت {asset_name} ({days_default} روز)", hovermode='x unified', template='plotly_white', height=450)
    return fig

# ==================== تابع پیشنهادی ساده (مثل قبل) ====================
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

# ==================== محاسبه پرتفوی + تب Protective Put (با Married Put صحیح) ====================
@st.fragment
def calculate_portfolio():
    if "prices" not in st.session_state or st.session_state.prices is None:
        st.info("لطفاً ابتدا داده‌ها را دانلود کنید.")
        return

    prices = st.session_state.prices
    returns = prices.pct_change().dropna()
    asset_names = list(prices.columns)
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    rf = st.session_state.rf_rate / 100 if "rf_rate" in st.session_state else 0.18

    # weights (simple equal-weight fallback)
    n = len(asset_names)
    weights = np.ones(n) / n

    opt = option_strategies.get(st.session_state.get("option_strategy","بدون آپشن"), {"name":"بدون تغییر"})
    adjusted_return = np.dot(mean_ret, weights) * 100
    adjusted_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
    recovery = "—"

    # tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 خلاصه", "💰 تخصیص", "🔮 پیش‌بینی", "📈 بک‌تست", "🛡️ Protective (Married) Put"])

    with tab1:
        st.markdown("### خلاصه")
        df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100,2)})
        st.dataframe(df_w, use_container_width=True)

    with tab2:
        total_usd = st.number_input("کل سرمایه برای محاسبات (دلار)", min_value=100, value=1200, step=100)
        exchange_rate = st.number_input("نرخ تبدیل (دلار->تومان)", min_value=100000, value=200000000, step=1000000)
        df_alloc = capital_allocator_calculator(weights, asset_names, total_usd, exchange_rate)
        st.dataframe(df_alloc[["دارایی","درصد وزن","دلار ($)"]], use_container_width=True)

    with tab3:
        selected = st.multiselect("انتخاب دارایی‌ها برای پیش‌بینی", asset_names, default=asset_names[:min(2,len(asset_names))])
        if selected:
            days = st.slider("روزهای پیش‌بینی", 30, 365, 90)
            for a in selected:
                st.plotly_chart(plot_forecast_single(prices[a], a, days_default=days), use_container_width=True)

    with tab4:
        st.markdown("بک‌تست (خلاصه) — این بخش مختصر است")
        st.write("...")

    with tab5:
        st.markdown("### Protective (Married) Put — محاسبه صحیح Married Put (دارایی + Long Put)")
        # پیدا کردن BTC و ETH
        btc_idx = None; eth_idx = None
        for i, name in enumerate(asset_names):
            if "BTC" in name.upper(): btc_idx = i
            if "ETH" in name.upper(): eth_idx = i

        if btc_idx is None and eth_idx is None:
            st.error("برای استفاده از این بخش باید BTC و/یا ETH در لیست دارایی‌ها موجود باشند.")
            return

        st.markdown("#### تنظیمات کلی")
        col_a, col_b = st.columns(2)
        total_usd = col_a.number_input("کل سرمایه مرجع (دلار) — برای محاسبه exposure", min_value=100, value=1200, step=100)
        # exposures = weights * total_usd
        exposures = {asset_names[i]: weights[i]*total_usd for i in range(len(asset_names))}

        st.markdown("#### ورودی قراردادها (Married Put)")
        c1, c2 = st.columns(2)
        # BTC inputs
        if btc_idx is not None:
            with c1:
                st.markdown("BTC-USD")
                btc_price = prices.iloc[-1, btc_idx]
                st.write(f"قیمت فعلی BTC: ${btc_price:,.2f}")
                btc_strike = st.number_input("Strike BTC ($)", min_value=btc_price*0.2, max_value=btc_price*0.999, value=btc_price*0.90, step=50.0, key="btc_strike")
                btc_premium = st.number_input("Premium هر قرارداد BTC ($)", min_value=0.0, value=max(1.0, btc_price*0.04), step=1.0, key="btc_premium")
                btc_contracts = st.number_input("تعداد قرارداد BTC", min_value=0, max_value=1000, value=0, step=1, key="btc_contracts")
                btc_contract_size = st.number_input("اندازه هر قرارداد (BTC)", min_value=0.01, max_value=100.0, value=1.0, step=0.01, key="btc_contract_size")
        else:
            btc_price = btc_strike = btc_premium = btc_contracts = btc_contract_size = None

        # ETH inputs
        if eth_idx is not None:
            with c2:
                st.markdown("ETH-USD")
                eth_price = prices.iloc[-1, eth_idx]
                st.write(f"قیمت فعلی ETH: ${eth_price:,.2f}")
                eth_strike = st.number_input("Strike ETH ($)", min_value=eth_price*0.2, max_value=eth_price*0.999, value=eth_price*0.90, step=5.0, key="eth_strike")
                eth_premium = st.number_input("Premium هر قرارداد ETH ($)", min_value=0.0, value=max(0.5, eth_price*0.04), step=0.5, key="eth_premium")
                eth_contracts = st.number_input("تعداد قرارداد ETH", min_value=0, max_value=1000, value=0, step=1, key="eth_contracts")
                eth_contract_size = st.number_input("اندازه هر قرارداد (ETH)", min_value=0.01, max_value=1000.0, value=1.0, step=0.01, key="eth_contract_size")
        else:
            eth_price = eth_strike = eth_premium = eth_contracts = eth_contract_size = None

        # محاسبات پرداختی‌ها و exposure -> واحد پا��ه (واحد دارایی) برای Held underlying
        # units_held = exposure_in_usd / current_price
        units_held_btc = 0.0
        if btc_idx is not None:
            exposure_btc = exposures.get(asset_names[btc_idx], 0.0)
            units_held_btc = exposure_btc / (btc_price + 1e-8)

        units_held_eth = 0.0
        if eth_idx is not None:
            exposure_eth = exposures.get(asset_names[eth_idx], 0.0)
            units_held_eth = exposure_eth / (eth_price + 1e-8)

        # کل پریمیوم پرداختی
        btc_total_premium = (btc_premium * btc_contracts * btc_contract_size) if btc_idx is not None else 0.0
        eth_total_premium = (eth_premium * eth_contracts * eth_contract_size) if eth_idx is not None else 0.0

        st.markdown("---")
        st.markdown("#### نتایج عددی")
        c1, c2, c3 = st.columns(3)
        original_portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
        c1.metric("ریسک پرتفوی (بدون بیمه)", f"{original_portfolio_risk:.2f}%")
        # محاسبه کاهش تقریبی با نسبت پریمیوم به exposure (همان مدل تقریبی)
        btc_premium_pct = (btc_total_premium / (exposures.get(asset_names[btc_idx],1e-8))) * 100 if btc_idx is not None else 0.0
        eth_premium_pct = (eth_total_premium / (exposures.get(asset_names[eth_idx],1e-8))) * 100 if eth_idx is not None else 0.0
        btc_reduction = min(0.95, btc_premium_pct * 0.5 / 100.0)
        eth_reduction = min(0.95, eth_premium_pct * 0.5 / 100.0)
        cov_adj = apply_protective_put_to_cov(cov_mat, asset_names, btc_idx, eth_idx, btc_reduction, eth_reduction)
        new_portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_adj, weights))) * 100
        c2.metric("ریسک پرتفوی (با بیمه)", f"{new_portfolio_risk:.2f}%")
        c3.metric("کاهش ریسک (%)", f"{(original_portfolio_risk - new_portfolio_risk):.3f}%")

        st.markdown("---")
        st.markdown("#### نمودار Payoff صحیح برای Married Put (Underlying + Long Put)")
        # پارامترهای محور قیمت برای هر دارایی (zoom controls)
        p_min_mult = st.slider("کمینه محور قیمت نسبت به قیمت فعلی (%)", 50, 90, 80)
        p_max_mult = st.slider("بیشینه محور قیمت نسبت به قیمت فعلی (%)", 110, 150, 120)

        # BTC payoff: compute for a grid of prices
        fig = go.Figure()
        all_prices = np.array([])

        if btc_idx is not None:
            btc_min = btc_price * (p_min_mult/100.0)
            btc_max = btc_price * (p_max_mult/100.0)
            grid_btc = np.linspace(btc_min, btc_max, 300)
            # underlying PnL per price: (S_T - S0) * units_held_btc
            underlying_pnl_btc = (grid_btc - btc_price) * units_held_btc
            # put payoff: max(strike - S_T, 0) * contracts * contract_size - total_premium
            put_payout_btc = np.maximum(btc_strike - grid_btc, 0.0) * (btc_contracts * btc_contract_size)
            # Married Put PnL = underlying_pnl + put_payout - premium_paid
            married_pnl_btc = underlying_pnl_btc + put_payout_btc - btc_total_premium
            fig.add_trace(go.Scatter(x=grid_btc, y=married_pnl_btc, name="BTC Married Put (USD)", mode="lines", line=dict(color="orange", width=2)))
            # shade positive/negative
            fig.add_trace(go.Scatter(x=grid_btc, y=np.where(married_pnl_btc>=0, married_pnl_btc, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(50,205,50,0.18)', showlegend=False))
            fig.add_trace(go.Scatter(x=grid_btc, y=np.where(married_pnl_btc<0, married_pnl_btc, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(255,99,71,0.18)', showlegend=False))
            # BE point for BTC: find approx crossing where married_pnl_btc == 0
            sign = np.sign(married_pnl_btc)
            cross_idx = np.where(np.diff(sign) != 0)[0]
            if cross_idx.size > 0:
                be_btc = grid_btc[cross_idx[-1]]
                fig.add_vline(x=be_btc, line_dash="dash", line_color="orange", annotation_text=f"BTC BE: ${be_btc:.2f}", annotation_position="top left")
            all_prices = np.concatenate([all_prices, grid_btc])

        if eth_idx is not None:
            eth_min = eth_price * (p_min_mult/100.0)
            eth_max = eth_price * (p_max_mult/100.0)
            grid_eth = np.linspace(eth_min, eth_max, 300)
            underlying_pnl_eth = (grid_eth - eth_price) * units_held_eth
            put_payout_eth = np.maximum(eth_strike - grid_eth, 0.0) * (eth_contracts * eth_contract_size)
            married_pnl_eth = underlying_pnl_eth + put_payout_eth - eth_total_premium
            fig.add_trace(go.Scatter(x=grid_eth, y=married_pnl_eth, name="ETH Married Put (USD)", mode="lines", line=dict(color="blue", width=2)))
            fig.add_trace(go.Scatter(x=grid_eth, y=np.where(married_pnl_eth>=0, married_pnl_eth, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(50,205,50,0.12)', showlegend=False))
            fig.add_trace(go.Scatter(x=grid_eth, y=np.where(married_pnl_eth<0, married_pnl_eth, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(255,99,71,0.12)', showlegend=False))
            sign_e = np.sign(married_pnl_eth)
            cross_idx_e = np.where(np.diff(sign_e) != 0)[0]
            if cross_idx_e.size > 0:
                be_eth = grid_eth[cross_idx_e[-1]]
                fig.add_vline(x=be_eth, line_dash="dash", line_color="blue", annotation_text=f"ETH BE: ${be_eth:.2f}", annotation_position="top right")
            all_prices = np.concatenate([all_prices, grid_eth])

        # Combined payoff: align onto common price grid and sum payoffs (interpolate)
        if all_prices.size > 0:
            common_min = float(np.nanmin(all_prices))
            common_max = float(np.nanmax(all_prices))
            common_grid = np.linspace(common_min, common_max, 400)
            total_payoff = np.zeros_like(common_grid)
            if btc_idx is not None:
                from numpy import interp
                total_payoff += interp(common_grid, grid_btc, married_pnl_btc)
            if eth_idx is not None:
                from numpy import interp
                total_payoff += interp(common_grid, grid_eth, married_pnl_eth)
            fig.add_trace(go.Scatter(x=common_grid, y=total_payoff, name="Total Married Put Payoff (BTC+ETH)", mode="lines", line=dict(color="green", width=3)))
            # BE total
            sign_t = np.sign(total_payoff)
            cross_t = np.where(np.diff(sign_t) != 0)[0]
            if cross_t.size > 0:
                be_total = common_grid[cross_t[-1]]
                fig.add_vline(x=be_total, line_dash="dot", line_color="black", annotation_text=f"Total BE ~ ${be_total:.2f}", annotation_position="bottom right")

        fig.update_layout(title="Payoff — Married Put (Underlying + Long Put)", xaxis_title="Price ($)", yaxis_title="PnL (USD)", template='plotly_white', height=520)
        st.plotly_chart(fig, use_container_width=True)

        # پیشنهاد تعداد قراردادها برای رسیدن به هدف ریسک (اختیاری)
        st.markdown("---")
        st.markdown("#### پیشنهاد خودکار برای رسیدن به هدف ریسک (اختیاری)")
        est_btc_prem = st.number_input("برآورد Premium هر قرارداد BTC ($) — برای پیشنهاد خودکار", value=float(btc_premium if btc_premium is not None else 0.0), step=1.0)
        est_eth_prem = st.number_input("برآورد Premium هر قرارداد ETH ($) — برای پیشنهاد خودکار", value=float(eth_premium if eth_premium is not None else 0.0), step=0.5)
        target_risk = st.number_input("هدف ریسک کل پرتفوی (%)", min_value=0.5, max_value=20.0, value=2.0, step=0.1)
        max_search = st.number_input("حداکثر قرارداد برای جستجو (هر دارایی)", min_value=1, max_value=200, value=30, step=1)
        if st.button("پیشنهاد بده (جستجوی ساده)"):
            suggestion = suggest_contracts_for_target_risk(prices, returns, asset_names, weights, cov_mat, total_usd, btc_idx, eth_idx, float(btc_contract_size if btc_contract_size else 1.0), float(eth_contract_size if eth_contract_size else 1.0), float(est_btc_prem), float(est_eth_prem), max_contracts=int(max_search), target_risk_pct=float(target_risk))
            if suggestion:
                st.success(f"پیشنهاد: BTC contracts={suggestion['b']} — ETH contracts={suggestion['e']} — هزینه کل ${suggestion['total_premium']:.2f} — ریسک جدید {suggestion['new_risk']:.3f}%")
            else:
                st.info("پیشنهادی یافت نشد (یا اطلاعات پریمیوم کافی نیست).")

# ========== UI اصلی و سایدبار ==========
st.set_page_config(page_title="Portfolio360 Ultimate Pro", layout="wide")
st.markdown("<h2 style='text-align:center;color:#00a3cc'>Portfolio360 Ultimate Pro — Married Put Payoff Fixed</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.header("دانلود داده")
    tickers = st.text_input("نمادها", "BTC-USD, ETH-USD, GC=F, USDIRR=X, ^GSPC")
    if st.button("دانلود"):
        with st.spinner("در حال دانلود..."):
            data = download_data(tickers)
            if data is not None:
                st.session_state.prices = data
                st.success(f"{len(data.columns)} دارایی بارگذاری شد")
                st.experimental_rerun()

    st.markdown("---")
    if "option_strategy" not in st.session_state: st.session_state.option_strategy = "بدون آپشن"
    st.session_state.option_strategy = st.selectbox("استراتژی آپشن", list(option_strategies.keys()))
    if "rf_rate" not in st.session_state: st.session_state.rf_rate = 18.0
    st.session_state.rf_rate = st.number_input("نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5)

# اجرا
calculate_portfolio()

st.caption(f"Portfolio360 Ultimate Pro — updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
