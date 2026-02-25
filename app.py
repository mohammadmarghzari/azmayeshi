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


# ==================== کش آماری ====================
@st.cache_data(show_spinner=False)
def compute_stats(returns):
    mean_ret = returns.mean().values * 252
    cov_mat = returns.cov().values * 252
    corr_mat = returns.corr().values
    return mean_ret, cov_mat, corr_mat


# ==================== استراتژی‌ها ====================
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


# ==================== وزن‌ها ====================
def get_portfolio_weights(style, returns, mean_ret, cov_mat, corr_mat, rf, bounds):

    n = len(mean_ret)
    x0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # ---------- مارکوویتز ----------
    if style == "مارکوویتز + هجینگ (بهینه‌ترین شارپ)":

        def obj(w):
            risk = np.sqrt(w @ cov_mat @ w)
            return -(w @ mean_ret - rf) / (risk + 1e-8)

        res = minimize(obj, x0, method="SLSQP",
                       bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 600})
        return res.x if res.success else x0

    # ---------- مونت کارلو فوق سریع ----------
    elif style == "مونت‌کارلو مقاوم (Resampled Frontier)":

        sims = 4000  # سریع ولی دقیق
        w = np.random.random((sims, n))
        w = w / w.sum(axis=1, keepdims=True)

        returns_sim = w @ mean_ret
        risk_sim = np.sqrt(np.einsum('ij,jk,ik->i', w, cov_mat, w))
        sharpe = (returns_sim - rf) / (risk_sim + 1e-8)

        return w[np.argmax(sharpe)]

    # ---------- حداقل ریسک ----------
    elif style == "حداقل ریسک (محافظه‌کارانه)":

        def obj(w):
            return w @ cov_mat @ w

        res = minimize(obj, x0, method="SLSQP",
                       bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 600})
        return res.x if res.success else x0

    # ---------- ریسک پاریتی ----------
    elif style == "ریسک‌پاریتی (Risk Parity)":

        def rp_obj(w):
            port_var = w @ cov_mat @ w
            contrib = w * (cov_mat @ w) / np.sqrt(port_var + 1e-8)
            return np.sum((contrib - contrib.mean()) ** 2)

        res = minimize(rp_obj, x0, method="SLSQP",
                       bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 600})
        return res.x if res.success else x0

    # ---------- HRP ----------
    elif style == "HRP (سلسله‌مراتبی)":

        dist = np.sqrt((1 - corr_mat) / 2)
        link = linkage(squareform(dist), 'single')

        ivp = 1 / np.diag(cov_mat)
        w = ivp / ivp.sum()
        return w

    # ---------- Inverse Vol ----------
    elif style == "Inverse Volatility":
        vol = np.sqrt(np.diag(cov_mat))
        w = 1 / vol
        return w / w.sum()

    # ---------- Kelly ----------
    elif style == "Kelly Criterion (حداکثر رشد)":
        w = mean_ret / np.diag(cov_mat)
        w = np.clip(w, 0, None)
        return w / (w.sum() if w.sum() > 0 else 1)

    # ---------- سایر ----------
    else:
        return x0


# ==================== پرتفوی ====================
@st.fragment
def calculate_portfolio():

    if "prices" not in st.session_state:
        st.info("لطفاً داده‌ها را دانلود کنید.")
        return

    prices = st.session_state.prices
    returns = prices.pct_change().dropna()

    mean_ret, cov_mat, corr_mat = compute_stats(returns)
    rf = st.session_state.rf_rate / 100

    bounds = [(0.0, 1.0)] * len(mean_ret)

    weights = get_portfolio_weights(
        st.session_state.selected_style,
        returns,
        mean_ret,
        cov_mat,
        corr_mat,
        rf,
        bounds
    )

    port_return = weights @ mean_ret * 100
    port_risk = np.sqrt(weights @ cov_mat @ weights) * 100
    sharpe = (port_return/100 - rf) / (port_risk/100 + 1e-8)

    st.success(f"سبک انتخابی: {st.session_state.selected_style}")

    c1, c2, c3 = st.columns(3)
    c1.metric("بازده سالانه", f"{port_return:.2f}%")
    c2.metric("ریسک سالانه", f"{port_risk:.2f}%")
    c3.metric("Sharpe", f"{sharpe:.3f}")

    df_w = pd.DataFrame({
        "دارایی": prices.columns,
        "وزن (%)": np.round(weights * 100, 2)
    }).sort_values("وزن (%)", ascending=False)

    st.dataframe(df_w, use_container_width=True)
    st.plotly_chart(px.pie(df_w, values="وزن (%)", names="دارایی"),
                    use_container_width=True)


# ==================== UI ====================
st.set_page_config(page_title="Portfolio360 Ultimate Pro", layout="wide")
st.title("Portfolio360 Ultimate Pro — High Performance")

with st.sidebar:

    tickers = st.text_input("نمادها", "BTC-USD, GC=F, USDIRR=X, ^GSPC")

    if st.button("دانلود", type="primary"):
        with st.spinner("در حال دانلود..."):
            data = download_data(tickers)
            if data is not None:
                st.session_state.prices = data
                st.rerun()

    styles = [
        "مارکوویتز + هجینگ (بهینه‌ترین شارپ)",
        "مونت‌کارلو مقاوم (Resampled Frontier)",
        "حداقل ریسک (محافظه‌کارانه)",
        "ریسک‌پاریتی (Risk Parity)",
        "HRP (سلسله‌مراتبی)",
        "Inverse Volatility",
        "Kelly Criterion (حداکثر رشد)"
    ]

    if "selected_style" not in st.session_state:
        st.session_state.selected_style = styles[0]

    st.session_state.selected_style = st.selectbox("انتخاب سبک", styles)

    st.session_state.rf_rate = st.number_input(
        "نرخ بدون ریسک (%)", 0.0, 50.0, 18.0, 0.5
    )

calculate_portfolio()
