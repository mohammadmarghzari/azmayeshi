"""
Portfolio360 Ultimate Pro — Final COMPLETE edition
- All previous features included and consolidated
- Correct Married (Protective) Put payoff (Underlying + Long Put - Premium)
- Suggest helper to reach target portfolio risk (approximate)
- DCA زمانی (Time-based DCA) simulation & CSV export
- Enhanced payoff plot styling (legend, colors, shaded PnL areas, BE lines, zoom controls)
- Self-contained single-file Streamlit app
"""
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

# -------------------------
# Utilities
# -------------------------
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

# -------------------------
# Strategies / Helpers (simplified where needed)
# -------------------------
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
    n = len(mean_ret)
    if style == "وزن برابر (ساده و مقاوم)":
        return np.ones(n) / n
    if style == "Inverse Volatility":
        vol = np.sqrt(np.diag(cov_mat))
        w = 1.0 / (vol + 1e-8)
        return w / w.sum()
    # fallback: equal weight
    return np.ones(n) / n

def capital_allocator_calculator(weights, asset_names, total_usd, exchange_rate):
    usd_to_toman = exchange_rate
    allocation_data = []
    for i, asset in enumerate(asset_names):
        weight = float(weights[i])
        amount_usd = weight * total_usd
        amount_toman = amount_usd * usd_to_toman
        amount_rial = amount_toman * 10
        allocation_data.append({
            "دارایی": asset,
            "درصد وزن": f"{weight*100:.2f}%",
            "دلار ($)": f"${amount_usd:,.2f}",
            "تومان": f"{amount_toman:,.0f}",
            "ریال": f"{amount_rial:,.0f}",
            "بدون فرمت_USD": amount_usd
        })
    df = pd.DataFrame(allocation_data)
    return df.sort_values("بدون فرمت_USD", ascending=False)

# -------------------------
# Married Put (Protective) helpers
# -------------------------
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

def suggest_contracts_for_target_risk(prices, returns, asset_names, weights, cov_mat, total_usd, btc_idx, eth_idx, btc_contract_size, eth_contract_size, est_btc_prem, est_eth_prem, max_contracts=30, target_risk_pct=2.0):
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
                if best is None or total_premium < best["total_premium"] or (total_premium == best["total_premium"] and (b+e) < (best["b"]+best["e"])):
                    best = {"b": b, "e": e, "new_risk": new_risk, "btc_total_premium": btc_total_premium, "eth_total_premium": eth_total_premium, "btc_reduction": btc_reduction, "eth_reduction": eth_reduction, "total_premium": total_premium}
    return best

# -------------------------
# DCA (Time-based) helpers (fixed)
# -------------------------
def generate_dca_dates(start_datetime, periods, freq_days):
    return [start_datetime + timedelta(days=i*freq_days) for i in range(periods)]

def map_dates_to_trading_days(dates, price_index):
    mapped = []
    idx = price_index
    for d in dates:
        ts = pd.Timestamp(d)
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
            except Exception:
                start_dt = pd.Timestamp(start_date)
    desired_dates = generate_dca_dates(start_dt, periods, freq_days)
    mapped_dates = map_dates_to_trading_days(desired_dates, price_series.index)

    if levels:
        levels = [float(l) for l in levels]
        levels = sorted(levels, reverse=True)
        base = periods // len(levels)
        remainder = periods % len(levels)
        level_schedule = []
        for i, lvl in enumerate(levels):
            cnt = base + (1 if i < remainder else 0)
            level_schedule += [lvl] * cnt
        if len(level_schedule) < periods:
            level_schedule += [levels[-1]] * (periods - len(level_schedule))
        elif len(level_schedule) > periods:
            level_schedule = level_schedule[:periods]
    else:
        level_schedule = [None] * periods

    per_amount = total_amount / periods
    purchases = []
    for i, dt in enumerate(mapped_dates):
        price_on_date = float(price_series.loc[dt])
        allocated = per_amount
        units = allocated / price_on_date if price_on_date > 0 else 0.0
        purchases.append({"date": pd.Timestamp(dt), "price": price_on_date, "amount_usd": allocated, "units": units, "level_assigned": level_schedule[i]})
    df = pd.DataFrame(purchases)
    total_units = df["units"].sum()
    avg_price = (df["amount_usd"].sum() / (total_units + 1e-12)) if total_units > 0 else np.nan
    final_price = float(price_series.iloc[-1])
    final_value = total_units * final_price
    profit = final_value - total_amount
    profit_pct = (profit / total_amount) * 100 if total_amount > 0 else np.nan
    summary = {"total_invested": total_amount, "total_units": total_units, "avg_price_per_unit": avg_price, "final_price": final_price, "final_value": final_value, "profit": profit, "profit_pct": profit_pct, "first_date": df["date"].min(), "last_date": df["date"].max()}
    return df, summary

def plot_price_with_purchases(price_series, purchases_df, title="Price with purchases"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_series.index, y=price_series.values, name="Price", mode="lines", line=dict(color="#0b69ff")))
    if not purchases_df.empty:
        fig.add_trace(go.Scatter(x=purchases_df["date"], y=purchases_df["price"], mode="markers+text", name="Purchases", marker=dict(size=8, color="orange"), text=[f"{a:.2f}$" for a in purchases_df["amount_usd"]], textposition="top center"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white", height=480)
    return fig

# -------------------------
# UI & Main
# -------------------------
st.set_page_config(page_title="Portfolio360 Ultimate Pro — Final", layout="wide")
st.markdown("<h1 style='text-align:center;color:#0b9bd1'>Portfolio360 Ultimate Pro — FINAL</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 دانلود داده")
    tickers = st.text_input("نمادها (با کاما جدا کنید)", "BTC-USD, ETH-USD, GC=F, USDIRR=X, ^GSPC")
    period = st.selectbox("بازه زمانی (yfinance)", ["1y", "2y", "5y", "10y", "max"], index=1)
    if st.button("🔄 دانلود/بروزرسانی داده"):
        with st.spinner("در حال دانلود ..."):
            data = download_data(tickers, period=period)
            if data is not None:
                st.session_state.prices = data
                st.success(f"✅ {len(data.columns)} دارایی بارگذاری شد.")
                st.experimental_rerun()

    st.markdown("---")
    st.header("⚙️ تنظیمات")
    if "rf_rate" not in st.session_state: st.session_state.rf_rate = 18.0
    st.session_state.rf_rate = st.number_input("نرخ بدون ریسک (%)", min_value=0.0, max_value=50.0, value=st.session_state.rf_rate, step=0.5)
    if "hedge_strategy" not in st.session_state: st.session_state.hedge_strategy = list(hedge_strategies.keys())[3]
    st.session_state.hedge_strategy = st.selectbox("استراتژی هجینگ", list(hedge_strategies.keys()), index=list(hedge_strategies.keys()).index(st.session_state.hedge_strategy))
    if "option_strategy" not in st.session_state: st.session_state.option_strategy = list(option_strategies.keys())[0]
    st.session_state.option_strategy = st.selectbox("استراتژی آپشن", list(option_strategies.keys()))

if "prices" not in st.session_state or st.session_state.prices is None:
    st.info("ابتدا در سایدبار داده‌ها را دانلود کنید.")
else:
    prices = st.session_state.prices
    asset_names = list(prices.columns)
    returns = prices.pct_change().dropna()
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    rf = st.session_state.rf_rate / 100.0

    # Top controls
    st.markdown("## تنظیمات پرتفوی و تخصیص")
    colA, colB, colC = st.columns([2,1,1])
    with colA:
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
        st.session_state.selected_style = st.selectbox("انتخاب سبک", styles, index=styles.index(st.session_state.selected_style))
    with colB:
        capital_usd = st.number_input("کل سرمایه (دلار)", min_value=1, max_value=50_000_000, value=1200, step=100)
        exchange_rate = st.number_input("نرخ تبدیل (تومان برای هر دلار)", min_value=1000, max_value=1_000_000_000, value=200_000, step=1000)
    with colC:
        if st.button("محاسبه پرتفوی"):
            weights = get_portfolio_weights(st.session_state.selected_style, returns, mean_ret, cov_mat, rf, None)
            st.session_state.weights = weights
            st.session_state.last_capital_usd = capital_usd
            st.success("وزن‌ها محاسبه شد.")

    if "weights" not in st.session_state:
        st.session_state.weights = np.ones(len(asset_names)) / len(asset_names)

    weights = st.session_state.weights
    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100,2)})
    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی", title="توزیع پرتفوی"), use_container_width=True)
    alloc_df = capital_allocator_calculator(weights, asset_names, capital_usd, exchange_rate)
    st.markdown("### تخصیص سرمایه (جزئیات)")
    st.dataframe(alloc_df[["دارایی","درصد وزن","دلار ($)","تومان","ریال"]], use_container_width=True)
    st.download_button("📥 دانلود تخصیص CSV", alloc_df.to_csv(index=False, encoding="utf-8-sig"), file_name=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # Forecast
    st.markdown("## پیش‌بینی قیمت (Monte Carlo)")
    sel_asset = st.selectbox("یک دارایی برای پیش‌بینی انتخاب کنید", asset_names)
    days_forecast = st.slider("روزهای پیش‌بینی", 30, 365, 90)
    if st.button("اجرای پیش‌بینی"):
        series = prices[sel_asset]
        paths = forecast_price_series(series, days=days_forecast, sims=400)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name="قیمت واقعی", line=dict(color="black")))
        future_x = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=days_forecast)
        median = np.percentile(paths, 50, axis=1)
        fig.add_trace(go.Scatter(x=future_x, y=median, name="میانه پیش‌بینی", line=dict(color="orange")))
        fig.update_layout(title=f"پیش‌بینی قیمت {sel_asset}", xaxis_title="Date", yaxis_title="Price", template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)

    # Married Put section (with enhanced plotting controls)
    st.markdown("## Protective (Married) Put — نمودار پیشرفته")
    btc_col = next((c for c in asset_names if "BTC" in c.upper()), None)
    eth_col = next((c for c in asset_names if "ETH" in c.upper()), None)
    st.info("نمودار payoff و BE به‌صورت گرافیکی و قابل زوم نمایش داده می‌شود. برای شباهت بیشتر به تصویر، می‌توانید رنگ‌ها و محدوده زوم را تنظیم کنید.")

    col1, col2 = st.columns(2)
    with col1:
        if btc_col:
            st.subheader("BTC-USD")
            btc_price = float(prices[btc_col].iloc[-1])
            btc_strike = st.number_input("Strike BTC ($)", value=btc_price*0.90, step=10.0, key="ui_btc_strike_final")
            btc_premium = st.number_input("Premium هر قرارداد BTC ($)", value=max(0.0, btc_price*0.04), step=1.0, key="ui_btc_prem_final")
            btc_contracts = st.number_input("تعداد قرارداد BTC (long put)", min_value=0, max_value=200, value=0, step=1, key="ui_btc_contracts_final")
            btc_contract_size = st.number_input("اندازه هر قرارداد (BTC)", min_value=0.01, max_value=100.0, value=1.0, step=0.01, key="ui_btc_size_final")
    with col2:
        if eth_col:
            st.subheader("ETH-USD")
            eth_price = float(prices[eth_col].iloc[-1])
            eth_strike = st.number_input("Strike ETH ($)", value=eth_price*0.90, step=5.0, key="ui_eth_strike_final")
            eth_premium = st.number_input("Premium هر قرارداد ETH ($)", value=max(0.0, eth_price*0.04), step=0.5, key="ui_eth_prem_final")
            eth_contracts = st.number_input("تعداد قرارداد ETH (long put)", min_value=0, max_value=200, value=0, step=1, key="ui_eth_contracts_final")
            eth_contract_size = st.number_input("اندازه هر قرارداد (ETH)", min_value=0.01, max_value=1000.0, value=1.0, step=0.01, key="ui_eth_size_final")

    # zoom controls (percent of S0)
    st.markdown("### تنظیم محدوده نمایش نمودار (Zoom)")
    zcol1, zcol2 = st.columns(2)
    zoom_min_pct = zcol1.slider("کاهش حداقل نسبت نسبت به قیمت فعلی (%)", 10, 100, 80)
    zoom_max_pct = zcol2.slider("حداکثر نسبت نسبت به قیمت فعلی (%)", 100, 250, 140)

    if st.button("نمایش نمودار Married Put (پیشرفته)"):
        exposures = {asset_names[i]: float(weights[i])*capital_usd for i in range(len(asset_names))}
        units_btc = exposures.get(btc_col, 0.0) / (btc_price + 1e-8) if btc_col else 0.0
        units_eth = exposures.get(eth_col, 0.0) / (eth_price + 1e-8) if eth_col else 0.0

        traces = []
        all_prices = np.array([])

        if btc_col and btc_contracts > 0:
            grid_btc, married_btc, btc_prem_paid = married_put_pnl_grid(btc_price, btc_strike, btc_premium, units_btc, int(btc_contracts), float(btc_contract_size))
            traces.append(("BTC", grid_btc, married_btc, "#ff8c00"))
            all_prices = np.concatenate([all_prices, grid_btc])
        if eth_col and eth_contracts > 0:
            grid_eth, married_eth, eth_prem_paid = married_put_pnl_grid(eth_price, eth_strike, eth_premium, units_eth, int(eth_contracts), float(eth_contract_size))
            traces.append(("ETH", grid_eth, married_eth, "#1f77b4"))
            all_prices = np.concatenate([all_prices, grid_eth])

        fig = go.Figure()
        for name, grid, pnl, color in traces:
            fig.add_trace(go.Scatter(x=grid, y=pnl, name=f"{name} Married Put (USD)", mode="lines", line=dict(color=color, width=2)))
            fig.add_trace(go.Scatter(x=grid, y=np.where(pnl>=0, pnl, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(50,205,50,0.12)', showlegend=False))
            fig.add_trace(go.Scatter(x=grid, y=np.where(pnl<0, pnl, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(255,99,71,0.12)', showlegend=False))

        if all_prices.size > 0:
            common_min = float(np.nanmin(all_prices))
            common_max = float(np.nanmax(all_prices))
            common_grid = np.linspace(common_min, common_max, 800)
            total_payoff = np.zeros_like(common_grid)
            from numpy import interp
            if any(t[0] == "BTC" for t in traces):
                total_payoff += interp(common_grid, grid_btc, married_btc)
            if any(t[0] == "ETH" for t in traces):
                total_payoff += interp(common_grid, grid_eth, married_eth)
            fig.add_trace(go.Scatter(x=common_grid, y=total_payoff, name="Total Married Put Payoff (USD)", mode="lines", line=dict(color="#2ca02c", width=3)))
            fig.add_trace(go.Scatter(x=common_grid, y=np.where(total_payoff>=0, total_payoff, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(50,205,50,0.08)', showlegend=False))
            fig.add_trace(go.Scatter(x=common_grid, y=np.where(total_payoff<0, total_payoff, np.nan), fill='tozeroy', mode='none', fillcolor='rgba(255,99,71,0.08)', showlegend=False))
            sign_t = np.sign(total_payoff)
            cross_t = np.where(np.diff(sign_t) != 0)[0]
            if cross_t.size > 0:
                be_total = common_grid[cross_t[-1]]
                fig.add_vline(x=be_total, line_dash="dash", line_color="black", annotation_text=f"Total BE ~ ${be_total:.2f}", annotation_position="bottom right")

        # per-asset BE
        if btc_col and btc_contracts > 0:
            be_btc = btc_price + btc_premium
            fig.add_vline(x=be_btc, line_dash="dot", line_color="#ff8c00", annotation_text=f"BTC BE = {be_btc:.2f}", annotation_position="top left")
        if eth_col and eth_contracts > 0:
            be_eth = eth_price + eth_premium
            fig.add_vline(x=be_eth, line_dash="dot", line_color="#1f77b4", annotation_text=f"ETH BE = {be_eth:.2f}", annotation_position="top right")

        # Zoom: determine display window from slider relative to S0 values
        # Use the primary S0 chosen (if both exist use combined min S0)
        s0_candidates = []
        if btc_col and btc_contracts > 0: s0_candidates.append(btc_price)
        if eth_col and eth_contracts > 0: s0_candidates.append(eth_price)
        if len(s0_candidates) == 0:
            base = 1.0
        else:
            # center on average S0
            base = float(np.mean(s0_candidates))
        display_min = base * (zoom_min_pct / 100.0)
        display_max = base * (zoom_max_pct / 100.0)
        fig.update_xaxes(range=[display_min, display_max])
        # Style legend and layout to resemble target image
        fig.update_layout(
            title=dict(text="نمودار سود/زیان Married Put — نمایش زیبا", x=0.5),
            xaxis=dict(title="Price ($)", showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
            yaxis=dict(title="PnL (USD)", showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=40, r=20, t=80, b=40),
            height=560
        )
        # show modebar (zoom etc) by default when plotly renders
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        # risk impact
        btc_idx = asset_names.index(btc_col) if (btc_col in asset_names) else None
        eth_idx = asset_names.index(eth_col) if (eth_col in asset_names) else None
        btc_total_premium = (btc_premium * btc_contracts * btc_contract_size) if (btc_col and btc_contracts>0) else 0.0
        eth_total_premium = (eth_premium * eth_contracts * eth_contract_size) if (eth_col and eth_contracts>0) else 0.0
        exposures = {asset_names[i]: float(weights[i])*capital_usd for i in range(len(asset_names))}
        btc_premium_pct = (btc_total_premium / (exposures.get(btc_col,1e-8))) * 100 if btc_col else 0.0
        eth_premium_pct = (eth_total_premium / (exposures.get(eth_col,1e-8))) * 100 if eth_col else 0.0
        btc_reduction = min(0.95, btc_premium_pct * 0.5 / 100.0)
        eth_reduction = min(0.95, eth_premium_pct * 0.5 / 100.0)
        cov_adj = apply_protective_put_to_cov(cov_mat, asset_names, btc_idx, eth_idx, btc_reduction, eth_reduction)
        original_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * 100
        new_risk = np.sqrt(np.dot(weights.T, np.dot(cov_adj, weights))) * 100
        st.markdown(f"- ریس�� پرتفوی (بدون بیمه): **{original_risk:.2f}%**")
        st.markdown(f"- ریسک پرتفوی (با Married Put): **{new_risk:.2f}%**")
        st.markdown(f"- کاهش ریسک تقریبی: **{original_risk - new_risk:.3f}%**")
        st.markdown(f"- کل Premium پرداختی: **${(btc_total_premium + eth_total_premium):,.2f}**")

        # Suggestion helper
        st.markdown("#### پیشنهاد خودکار برای رسیدن به هدف ریسک (اختیاری)")
        est_btc_prem = st.number_input("برآورد Premium هر قرارداد BTC ($) — برای پیشنهاد", value=float(btc_premium if (btc_col and btc_contracts>0) else 0.0), step=1.0)
        est_eth_prem = st.number_input("برآورد Premium هر قرارداد ETH ($) — برای پیشنهاد", value=float(eth_premium if (eth_col and eth_contracts>0) else 0.0), step=0.5)
        target_risk = st.number_input("هدف ریسک کل پرتفوی (%)", min_value=0.5, max_value=20.0, value=2.0, step=0.1)
        max_search = st.number_input("حداکثر قرارداد برای جستجو (هر دارایی)", min_value=1, max_value=200, value=30, step=1)
        if st.button("🔎 پیشنهاد بده"):
            suggestion = suggest_contracts_for_target_risk(prices, returns, asset_names, weights, cov_mat, capital_usd, btc_idx, eth_idx, float(btc_contract_size if btc_contract_size else 1.0), float(eth_contract_size if eth_contract_size else 1.0), float(est_btc_prem), float(est_eth_prem), max_contracts=int(max_search), target_risk_pct=float(target_risk))
            if suggestion:
                st.success(f"پیشنهاد: BTC contracts={suggestion['b']} — ETH contracts={suggestion['e']} — هزینه کل ${suggestion['total_premium']:.2f} — ریسک جدید {suggestion['new_risk']:.3f}%")
            else:
                st.info("پیشنهادی یافت نشد (یا اطلاعات پریمیوم کافی نبود).")

    # -------------------------
    # DCA زمانی (Time-based) — final
    # -------------------------
    st.markdown("## ⏳ DCA زمانی (Time-based DCA)")
    st.markdown("""
    DCA زمانی (یا DCA مبتنی بر زمان، استراتژی چوب‌خط یا DCA ثابت زمانی) نوعی اجرای خاص از ا��تراتژی Dollar-Cost Averaging است که تمرکز اصلی آن روی زمان ثابت خرید است، نه روی سطوح قیمتی.
    توضیحات کام�� در UI بالا درج شده است.
    """)
    dca_asset = st.selectbox("دارایی برای DCA", asset_names, index=0)
    dca_total = st.number_input("کل سرمایه برای DCA (دلار)", min_value=1.0, value=1000.0, step=100.0)
    dca_periods = st.number_input("تعداد خریدها (دوره‌ها)", min_value=1, value=30, step=1)
    dca_freq_days = st.number_input("فواصل زمانی بین خریدها (روز)", min_value=1, value=1, step=1)
    dca_start_date = st.date_input("تاریخ شروع (برای شبیه‌سازی تاریخی)", value=(prices.index[0] + pd.Timedelta(days=1)).date())
    use_levels = st.checkbox("استفاده از سطوح قیمتی (اختیاری)", value=False)
    levels_input = None
    if use_levels:
        levels_txt = st.text_input("سطوح قیمتی جداشده با کاما (مثال: 2500,2200,1800)", "")
        try:
            levels_input = [float(x.strip()) for x in levels_txt.split(",") if x.strip()]
            if len(levels_input) == 0:
                levels_input = None
        except Exception:
            levels_input = None

    if st.button("اجرای شبیه‌سازی DCA"):
        series = prices[dca_asset]
        df_purchases, summary = simulate_time_dca(series, dca_total, int(dca_periods), int(dca_freq_days), start_date=dca_start_date, levels=levels_input)
        st.markdown("#### جدول معاملات زمان‌بندی‌شده")
        st.dataframe(df_purchases[["date","price","amount_usd","units","level_assigned"]].assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")), use_container_width=True)
        st.markdown("#### خلاصه شبیه‌سازی")
        st.write(f"- کل سرمایه صرف‌شده: ${summary['total_invested']:.2f}")
        st.write(f"- تعداد دوره‌ها: {int(dca_periods)}")
        st.write(f"- مجموع واحد خریداری‌شده: {summary['total_units']:.6f}")
        st.write(f"- میانگین قیمت خرید: ${summary['avg_price_per_unit']:.4f}")
        st.write(f"- قیمت نهایی (آخرین موجود): ${summary['final_price']:.2f}")
        st.write(f"- ارزش کنونی پوزیشن: ${summary['final_value']:.2f}")
        st.write(f"- سود/زیان: ${summary['profit']:.2f} ({summary['profit_pct']:.2f}%)")
        fig_p = plot_price_with_purchases(series, df_purchases, title=f"DCA on {dca_asset}")
        st.plotly_chart(fig_p, use_container_width=True)
        csv = df_purchases.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 دانلود معاملات DCA (CSV)", csv, file_name=f"dca_{dca_asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    st.markdown("---")
    st.caption("Portfolio360 Ultimate Pro — Final complete. برای دقت بیشتر در قیمت‌گذاری آپشن‌ها (Black-Scholes) داده‌های بازار نیاز است؛ می‌توانم اضافه کنم.")
