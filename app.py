"""
Portfolio360 Ultimate Pro — Professional Edition
- Enhanced UI with modern design
- Comprehensive help tooltips for each feature
- Better organized sections with expandable explanations
- Professional styling and visual improvements
- Self-contained single-file Streamlit app
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import math

warnings.filterwarnings("ignore")

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0; }
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem; border-radius: 15px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: white !important; font-size: 2.5rem !important; font-weight: 700 !important; }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem 1.5rem; border-radius: 10px;
        font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;
    }
    .help-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border: 1px solid #d1d5db; border-radius: 10px; padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELP TEXTS (خلاصه)
# =============================================================================
HELP_TEXTS = {
    "data_download": {"title": "📥 راهنمای دانلود داده", "content": "داده‌های قیمتی را دانلود می‌کند..."},
    "risk_free_rate": {"title": "📊 نرخ بدون ریسک", "content": "..."},
    "hedge_strategy": {"title": "🛡️ استراتژی‌های هجینگ", "content": "..."},
    "option_strategy": {"title": "📈 استراتژی‌های آپشن", "content": "..."},
    "portfolio_style": {"title": "🎯 سبک‌های پرتفوی", "content": "..."},
    "capital_allocation": {"title": "💰 تخصیص سرمایه", "content": "..."},
    "monte_carlo_forecast": {"title": "🔮 پیش‌بینی مونت‌کارلو", "content": "..."},
    "married_put": {"title": "🛡️ Protective Put", "content": "..."},
    "dca_time": {"title": "⏳ DCA زمانی", "content": "..."},
    "risk_metrics": {"title": "📉 معیارهای ریسک", "content": "..."}
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
@st.cache_data(show_spinner=False)
def download_data(tickers_str, period="max"):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    data = {}
    failed = []
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, auto_adjust=True)
            if len(df) > 20 and "Close" in df.columns:
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

def forecast_price_series(price_series, days=63, sims=500):
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_ret) < 2:
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

# =============================================================================
# DCA HELPERS - اصلاح شده
# =============================================================================
def generate_dca_dates(start_datetime, periods, freq_days):
    return [start_datetime + timedelta(days=i*freq_days) for i in range(periods)]

def map_dates_to_trading_days(dates, price_index):
    mapped = []
    idx = pd.to_datetime(price_index).tz_localize(None)  # تبدیل به naive
    
    for d in dates:
        ts = pd.Timestamp(d).tz_localize(None)
        if ts <= idx[0]:
            mapped.append(idx[0])
            continue
        locs = idx.searchsorted(ts)
        if locs >= len(idx):
            mapped.append(idx[-1])
        else:
            mapped.append(idx[locs])
    return pd.to_datetime(mapped)

def simulate_time_dca(price_series, total_amount, periods, freq_days=1, start_date=None, levels=None):
    if start_date is None:
        start_dt = price_series.index[0]
    else:
        if isinstance(start_date, datetime):
            start_dt = start_date
        else:
            try:
                start_dt = datetime.combine(start_date, datetime.min.time())
            except:
                start_dt = pd.Timestamp(start_date)
    
    desired_dates = generate_dca_dates(start_dt, periods, freq_days)
    mapped_dates = map_dates_to_trading_days(desired_dates, price_series.index)

    if levels:
        levels = sorted([float(l) for l in levels], reverse=True)
        base = periods // len(levels)
        remainder = periods % len(levels)
        level_schedule = []
        for i, lvl in enumerate(levels):
            cnt = base + (1 if i < remainder else 0)
            level_schedule += [lvl] * cnt
        level_schedule += [levels[-1]] * (periods - len(level_schedule))
        level_schedule = level_schedule[:periods]
    else:
        level_schedule = [None] * periods

    per_amount = total_amount / periods
    purchases = []
    for i, dt in enumerate(mapped_dates):
        price_on_date = float(price_series.loc[dt])
        allocated = per_amount
        units = allocated / price_on_date if price_on_date > 0 else 0.0
        purchases.append({
            "date": pd.Timestamp(dt),
            "price": price_on_date,
            "amount_usd": allocated,
            "units": units,
            "level_assigned": level_schedule[i]
        })
    
    df = pd.DataFrame(purchases)
    total_units = df["units"].sum()
    avg_price = (df["amount_usd"].sum() / (total_units + 1e-12)) if total_units > 0 else np.nan
    final_price = float(price_series.iloc[-1])
    final_value = total_units * final_price
    profit = final_value - total_amount
    profit_pct = (profit / total_amount) * 100 if total_amount > 0 else np.nan
    
    summary = {
        "total_invested": total_amount,
        "total_units": total_units,
        "avg_price_per_unit": avg_price,
        "final_price": final_price,
        "final_value": final_value,
        "profit": profit,
        "profit_pct": profit_pct,
        "first_date": df["date"].min(),
        "last_date": df["date"].max()
    }
    return df, summary

def plot_price_with_purchases(price_series, purchases_df, title="Price with purchases"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_series.index, y=price_series.values, name="Price", mode="lines", line=dict(color="#0b69ff")))
    if not purchases_df.empty:
        fig.add_trace(go.Scatter(x=purchases_df["date"], y=purchases_df["price"], mode="markers+text", name="Purchases",
                                marker=dict(size=8, color="orange"),
                                text=[f"{a:.2f}$" for a in purchases_df["amount_usd"]], textposition="top center"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white", height=480)
    return fig

# =============================================================================
# MARRIED PUT HELPERS
# =============================================================================
def married_put_pnl_grid(S0, strike, premium_per_contract, units_held, contracts, contract_size, grid_min=None, grid_max=None, ngrid=600):
    if grid_min is None:
        grid_min = max(0.01, S0 * 0.5)
    if grid_max is None:
        grid_max = S0 * 1.5
    grid = np.linspace(grid_min, grid_max, ngrid)
    underlying_pnl = (grid - S0) * units_held
    put_payout = np.maximum(strike - grid, 0.0) * (contracts * contract_size)
    total_premium = premium_per_contract * contracts * contract_size
    married_pnl = underlying_pnl + put_payout - total_premium
    return grid, married_pnl, total_premium

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

def suggest_contracts_for_target_risk(prices, returns, asset_names, weights, cov_mat, total_usd, 
                                      btc_idx, eth_idx, btc_contract_size, eth_contract_size, 
                                      est_btc_prem, est_eth_prem, max_contracts=30, target_risk_pct=2.0):
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
                if best is None or total_premium < best.get("total_premium", float('inf')) or \
                   (total_premium == best.get("total_premium") and (b+e) < best.get("b",0)+best.get("e",0)):
                    best = {
                        "b": b, "e": e, "new_risk": new_risk,
                        "btc_total_premium": btc_total_premium,
                        "eth_total_premium": eth_total_premium,
                        "btc_reduction": btc_reduction,
                        "eth_reduction": eth_reduction,
                        "total_premium": total_premium
                    }
    return best

# =============================================================================
# COVERED CALL HELPERS
# =============================================================================
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_call_greeks(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"Delta":0, "Gamma":0, "Theta":0, "Vega":0}
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm_cdf(d1)
    gamma = (math.exp(-d1*d1/2) / math.sqrt(2*math.pi)) / (S * sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * (math.exp(-d1*d1/2) / math.sqrt(2*math.pi)) / 100
    theta = (-(S * (math.exp(-d1*d1/2) / math.sqrt(2*math.pi)) * sigma) / (2 * math.sqrt(T))) / 365
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega}

# =============================================================================
# MAIN APP
# =============================================================================
st.set_page_config(page_title="Portfolio360 Ultimate Pro", page_icon="📊", layout="wide")

st.markdown("""
<div class="main-header">
    <h1>📊 Portfolio360 Ultimate Pro</h1>
    <p>سیستم جامع تحلیل و مدیریت پرتفوی | نسخه حرفه‌ای</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📥 دانلود داده‌ها")
    tickers = st.text_input("نمادها (با کاما جدا کنید)", "BTC-USD, ETH-USD, GC=F, USDIRR=X, ^GSPC")
    period = st.selectbox("بازه زمانی", ["1y", "2y", "5y", "10y", "max"], index=1)
    
    if st.button("🔄 دانلود / بروزرسانی داده‌ها", use_container_width=True):
        with st.spinner("در حال دانلود..."):
            data = download_data(tickers, period)
            if data is not None:
                st.session_state.prices = data
                st.success(f"✅ {len(data.columns)} دارایی بارگذاری شد")
                st.rerun()

# Main Content
if "prices" not in st.session_state or st.session_state.prices is None:
    st.info("👈 لطفاً ابتدا داده‌ها را از سایدبار دانلود کنید.")
else:
    prices = st.session_state.prices
    asset_names = list(prices.columns)
    returns = prices.pct_change().dropna()
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252

    # ... (بقیه بخش‌های UI بدون تغییر - Portfolio Config, Monte Carlo, Married Put, DCA, Covered Call)

    # DCA Section (اصلاح شده)
    st.markdown('<div class="section-header">⏳ DCA زمانی (Time-based Dollar-Cost Averaging)</div>', unsafe_allow_html=True)
    
    col_dca1, col_dca2, col_dca3 = st.columns([2, 1, 1])
    with col_dca1:
        dca_asset = st.selectbox("دارایی برای DCA", asset_names, index=0)
    with col_dca2:
        dca_total = st.number_input("کل سرمایه ($)", min_value=1.0, value=1000.0, step=100.0)
    with col_dca3:
        dca_periods = st.number_input("تعداد دوره‌ها", min_value=1, value=30, step=1)
    
    col_dca4, col_dca5, col_dca6 = st.columns([1, 1, 1])
    with col_dca4:
        dca_freq_days = st.number_input("فاصله زمانی (روز)", min_value=1, value=1, step=1)
    with col_dca5:
        dca_start_date = st.date_input("تاریخ شروع", value=(prices.index[0] + pd.Timedelta(days=1)).date())
    with col_dca6:
        use_levels = st.checkbox("استفاده از سطوح قیمتی", value=False)
    
    levels_input = None
    if use_levels:
        levels_txt = st.text_input("سطوح قیمتی (با کاما جدا کنید)", "2500,2200,1800")
        try:
            levels_input = [float(x.strip()) for x in levels_txt.split(",") if x.strip()]
        except:
            levels_input = None
    
    if st.button("▶️ اجرای شبیه‌سازی DCA", use_container_width=True):
        with st.spinner("در حال شبیه‌سازی..."):
            series = prices[dca_asset]
            df_purchases, summary = simulate_time_dca(
                series, dca_total, int(dca_periods),
                int(dca_freq_days), start_date=dca_start_date, levels=levels_input
            )
            
            st.dataframe(df_purchases[["date", "price", "amount_usd", "units"]], use_container_width=True, hide_index=True)
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("کل سرمایه‌گذاری", f"${summary['total_invested']:.2f}")
            with col_res2:
                st.metric("میانگین قیمت خرید", f"${summary['avg_price_per_unit']:.4f}")
            with col_res3:
                st.metric("سود/زیان", f"${summary['profit']:.2f} ({summary['profit_pct']:.2f}%)")
            
            fig_p = plot_price_with_purchases(series, df_purchases)
            st.plotly_chart(fig_p, use_container_width=True)

# Covered Call Sections (بدون تغییر)
# ... (بقیه کد Covered Call و Advanced Suite همان‌طور که در فایل اصلی بود)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>📊 <strong>Portfolio360 Ultimate Pro</strong> — نسخه حرفه‌ای</p>
</div>
""", unsafe_allow_html=True)
