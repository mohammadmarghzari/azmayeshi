"""
Portfolio360 — Blueprint Edition
سبک‌های پرتفوی + نمادهای جهانی
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import yfinance as yf
import warnings
import math
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Portfolio360 | Blueprint",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# BLUEPRINT CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg:      #08111f;
    --panel:   #0c1a2e;
    --card:    #0e2040;
    --border:  rgba(255,255,255,0.12);
    --accent:  #4fc3f7;
    --accent2: #81d4fa;
    --gold:    #ffd54f;
    --green:   #64ffda;
    --red:     #ff6b6b;
    --white:   #e8f4fd;
    --dim:     rgba(232,244,253,0.5);
    --mono:    'Courier New', 'Consolas', monospace;
}

/* ── زمینه با شبکه blueprint ── */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    background-image:
        linear-gradient(rgba(79,195,247,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79,195,247,0.05) 1px, transparent 1px),
        linear-gradient(rgba(79,195,247,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79,195,247,0.02) 1px, transparent 1px) !important;
    background-size: 100px 100px, 100px 100px, 20px 20px, 20px 20px !important;
}

/* ── سایدبار ── */
[data-testid="stSidebar"] > div:first-child {
    background-color: #060f1c !important;
    border-right: 1px solid var(--border) !important;
    background-image:
        linear-gradient(rgba(79,195,247,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79,195,247,0.04) 1px, transparent 1px) !important;
    background-size: 20px 20px !important;
}

/* ── تمام متن ── */
*, .stApp * { color: var(--white) !important; font-family: var(--mono) !important; }

/* ── لیبل‌ها ── */
label, [data-testid="stWidgetLabel"] p {
    color: var(--accent) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── هدر اصلی ── */
.bp-header {
    border: 1px solid var(--accent);
    border-top: 3px solid var(--accent);
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    background: rgba(79,195,247,0.03);
    box-shadow: 0 0 40px rgba(79,195,247,0.07);
}
.bp-header::before {
    content:''; position:absolute; top:-3px; left:-3px;
    width:18px; height:18px;
    border-top:3px solid var(--gold); border-left:3px solid var(--gold);
}
.bp-header::after {
    content:''; position:absolute; bottom:-3px; right:-3px;
    width:18px; height:18px;
    border-bottom:3px solid var(--gold); border-right:3px solid var(--gold);
}
.bp-header h1 {
    font-size:1.7rem !important; font-weight:700 !important;
    letter-spacing:0.12em !important; text-transform:uppercase !important;
    color:var(--white) !important;
    text-shadow: 0 0 25px rgba(79,195,247,0.4) !important;
    margin:0 !important;
}
.bp-header p { color:var(--accent) !important; font-size:0.75rem !important;
    letter-spacing:0.18em !important; text-transform:uppercase !important;
    margin-top:0.3rem !important; }

/* ── section header ── */
.bp-section {
    border-left:3px solid var(--accent);
    border-bottom:1px solid rgba(79,195,247,0.2);
    padding:0.55rem 1rem;
    margin:1.5rem 0 1rem 0;
    font-size:0.78rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase;
    color:var(--accent) !important;
    background: rgba(79,195,247,0.03);
}

/* ── کارت metric ── */
.bp-metric {
    background:var(--card);
    border:1px solid var(--border);
    border-top:2px solid var(--accent);
    padding:0.9rem 1rem;
    margin-bottom:0.5rem;
}
.bp-metric .val { font-size:1.5rem; font-weight:700; color:var(--accent) !important; }
.bp-metric .lbl { font-size:0.65rem; color:var(--dim) !important;
    letter-spacing:0.1em; text-transform:uppercase; margin-top:0.2rem; }
.bp-metric .delta-pos { color:var(--green) !important; font-size:0.8rem; }
.bp-metric .delta-neg { color:var(--red) !important; font-size:0.8rem; }

/* ── دکمه‌ها ── */
.stButton > button {
    background:transparent !important;
    color:var(--accent) !important;
    border:1.5px solid var(--accent) !important;
    border-radius:0 !important;
    font-size:0.72rem !important; font-weight:700 !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important;
    font-family:var(--mono) !important;
    transition:all 0.15s !important;
}
.stButton > button:hover {
    background:rgba(79,195,247,0.1) !important;
    box-shadow:0 0 18px rgba(79,195,247,0.25) !important;
}

/* ── ورودی‌ها ── */
.stTextInput input, .stNumberInput input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background:rgba(8,17,31,0.9) !important;
    border:1px solid var(--border) !important;
    border-radius:0 !important;
    color:var(--white) !important;
    font-family:var(--mono) !important;
    font-size:0.82rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color:var(--accent) !important;
    box-shadow:0 0 12px rgba(79,195,247,0.2) !important;
}

/* ── selectbox ── */
[data-testid="stSelectbox"] > div > div,
.stSelectbox > div > div {
    background:rgba(8,17,31,0.9) !important;
    border:1px solid var(--border) !important;
    border-radius:0 !important;
}

/* ── multiselect ── */
[data-testid="stMultiSelect"] > div > div {
    background:rgba(8,17,31,0.9) !important;
    border:1px solid var(--border) !important;
    border-radius:0 !important;
}
[data-testid="stMultiSelect"] span {
    background:rgba(79,195,247,0.15) !important;
    border:1px solid var(--accent) !important;
    border-radius:0 !important;
    font-size:0.72rem !important;
}

/* ── metric native ── */
[data-testid="stMetric"] {
    background:var(--card) !important;
    border:1px solid var(--border) !important;
    border-top:2px solid var(--accent) !important;
    border-radius:0 !important;
    padding:0.8rem !important;
}
[data-testid="stMetricValue"] {
    color:var(--accent) !important;
    font-family:var(--mono) !important;
    font-size:1.4rem !important;
}
[data-testid="stMetricLabel"] p {
    color:var(--dim) !important;
    font-size:0.65rem !important;
    letter-spacing:0.08em !important;
    text-transform:uppercase !important;
}
[data-testid="stMetricDelta"] { font-size:0.78rem !important; }

/* ── اکسپندر ── */
[data-testid="stExpander"] summary {
    background:rgba(79,195,247,0.05) !important;
    border:1px solid var(--border) !important;
    border-left:3px solid var(--accent) !important;
    border-radius:0 !important;
    font-size:0.75rem !important; font-weight:700 !important;
    letter-spacing:0.08em !important; text-transform:uppercase !important;
}
[data-testid="stExpander"] { border:1px solid var(--border) !important; border-radius:0 !important; }

/* ── جدول ── */
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:0 !important; }

/* ── info / success / warning ── */
.stAlert { background:rgba(79,195,247,0.05) !important;
    border:1px solid rgba(79,195,247,0.2) !important; border-radius:0 !important; }

/* ── checkbox ── */
.stCheckbox label { text-transform:none !important; font-size:0.78rem !important; }

/* ── tab ── */
[data-testid="stTabs"] [role="tab"] {
    background:transparent !important;
    border-bottom:2px solid transparent !important;
    border-radius:0 !important;
    font-size:0.72rem !important; font-weight:700 !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important;
    color:var(--dim) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:var(--accent) !important;
    border-bottom:2px solid var(--accent) !important;
    background:rgba(79,195,247,0.05) !important;
}

hr { border:none !important; border-top:1px solid rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# نمادها
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS = {
    "💰 ارزهای دیجیتال": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "BNB-USD": "BNB",
        "SOL-USD": "Solana",
        "XRP-USD": "XRP",
        "ADA-USD": "Cardano",
        "AVAX-USD": "Avalanche",
        "DOGE-USD": "Dogecoin",
        "DOT-USD": "Polkadot",
        "MATIC-USD": "Polygon",
        "LINK-USD": "Chainlink",
        "LTC-USD": "Litecoin",
        "UNI7083-USD": "Uniswap",
        "ATOM-USD": "Cosmos",
        "XLM-USD": "Stellar",
    },
    "📈 سهام آمریکا": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet",
        "AMZN": "Amazon",
        "NVDA": "NVIDIA",
        "META": "Meta",
        "TSLA": "Tesla",
        "BERKB": "Berkshire B",
        "JPM": "JPMorgan",
        "V": "Visa",
        "JNJ": "J&J",
        "WMT": "Walmart",
        "XOM": "Exxon",
        "BAC": "Bank of America",
        "MA": "Mastercard",
        "PG": "P&G",
        "HD": "Home Depot",
        "CVX": "Chevron",
        "ABBV": "AbbVie",
        "KO": "Coca-Cola",
        "PEP": "PepsiCo",
        "LLY": "Eli Lilly",
        "MRK": "Merck",
        "CRM": "Salesforce",
        "AMD": "AMD",
        "INTC": "Intel",
        "NFLX": "Netflix",
        "DIS": "Disney",
        "PYPL": "PayPal",
        "UBER": "Uber",
    },
    "🌍 سهام جهانی": {
        "TSM": "TSMC (Taiwan)",
        "ASML": "ASML (Netherlands)",
        "SAP": "SAP (Germany)",
        "TM": "Toyota (Japan)",
        "NVO": "Novo Nordisk (Denmark)",
        "HSBC": "HSBC (UK)",
        "BP": "BP (UK)",
        "SHEL": "Shell (UK)",
        "UL": "Unilever (UK)",
        "RIO": "Rio Tinto (UK)",
        "BABA": "Alibaba (China)",
        "JD": "JD.com (China)",
        "SONY": "Sony (Japan)",
        "HMC": "Honda (Japan)",
        "BCS": "Barclays (UK)",
    },
    "🏦 ETF و شاخص": {
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq 100 ETF",
        "DIA": "Dow Jones ETF",
        "IWM": "Russell 2000 ETF",
        "VTI": "Total Market ETF",
        "EEM": "Emerging Markets ETF",
        "VEA": "Developed Markets ETF",
        "AGG": "Bond Aggregate ETF",
        "TLT": "20Y Treasury ETF",
        "HYG": "High Yield Bond ETF",
        "GLD": "Gold ETF",
        "SLV": "Silver ETF",
        "USO": "Oil ETF",
        "XLE": "Energy ETF",
        "XLF": "Financials ETF",
        "XLK": "Technology ETF",
        "XLV": "Healthcare ETF",
        "ARKK": "ARK Innovation ETF",
        "VNQ": "Real Estate ETF",
        "PDBC": "Commodity ETF",
    },
    "🥇 کامودیتی و فارکس": {
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "CL=F": "Crude Oil (WTI)",
        "BZ=F": "Brent Oil",
        "NG=F": "Natural Gas",
        "HG=F": "Copper",
        "PL=F": "Platinum",
        "ZW=F": "Wheat",
        "ZC=F": "Corn",
        "ZS=F": "Soybeans",
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY",
        "USDCHF=X": "USD/CHF",
        "AUDUSD=X": "AUD/USD",
        "USDCAD=X": "USD/CAD",
        "USDIRR=X": "USD/IRR",
    },
}

PERIODS = {"۶ ماه": "6mo", "۱ سال": "1y", "۲ سال": "2y", "۵ سال": "5y", "۱۰ سال": "10y", "حداکثر": "max"}

STYLES = {
    "📐 بیشترین شارپ (Markowitz)": "max_sharpe",
    "🛡️ کمترین واریانس (Min Variance)": "min_var",
    "🎲 مونت‌کارلو (CVaR)": "monte_carlo",
    "🔮 وزن برابر (Equal Weight)": "equal_weight",
    "⚖️ ریسک پاریتی (Risk Parity)": "risk_parity",
    "🦢 ۹۰/۱۰ طالب (Taleb Barbell)": "taleb_barbell",
}

TALEB_SAFE = {"GC=F", "GLD", "TLT", "AGG", "EURUSD=X", "GBPUSD=X", "USDCHF=X"}
TALEB_RISKY = {"BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "NVDA", "TSLA", "ARKK"}

# ─────────────────────────────────────────────────────────────────────────────
# توابع
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(tickers: tuple, period: str):
    data = {}
    failed = []
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, auto_adjust=True)
            if len(df) > 20 and "Close" in df.columns:
                s = df["Close"].copy()
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                data[t] = s
            else:
                failed.append(t)
        except Exception:
            failed.append(t)
    if not data:
        return None, failed
    prices = pd.DataFrame(data).ffill().bfill().dropna()
    return prices, failed


def calc_weights(style: str, returns: pd.DataFrame, rf: float, selected: list):
    n = len(selected)
    eq = np.ones(n) / n
    mean_r = returns.mean() * 252
    cov = returns.cov() * 252
    bnds = [(0.01, 0.6)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    if style == "equal_weight":
        return eq, cov

    if style == "max_sharpe":
        def neg_sharpe(w):
            r = w @ mean_r.values
            v = np.sqrt(w @ cov.values @ w)
            return -(r - rf) / (v + 1e-9)
        res = minimize(neg_sharpe, eq, method="SLSQP", bounds=bnds, constraints=cons)
        return (res.x / res.x.sum() if res.success else eq), cov

    if style == "min_var":
        def port_var(w): return w @ cov.values @ w
        res = minimize(port_var, eq, method="SLSQP", bounds=bnds, constraints=cons)
        return (res.x / res.x.sum() if res.success else eq), cov

    if style == "risk_parity":
        def rp_obj(w):
            sig = np.sqrt(w @ cov.values @ w)
            mrc = cov.values @ w / (sig + 1e-9)
            rc = w * mrc
            target = sig / n
            return np.sum((rc - target) ** 2)
        res = minimize(rp_obj, eq, method="SLSQP", bounds=[(0.01, 1)] * n, constraints=cons)
        w = res.x / res.x.sum() if res.success else eq
        return w, cov

    if style == "monte_carlo":
        # Resampled Efficient Frontier — minimise CVaR
        best_w, best_cvar = eq, np.inf
        returns_arr = returns.values
        for _ in range(200):
            idx = np.random.choice(len(returns_arr), size=len(returns_arr), replace=True)
            samp = returns_arr[idx]
            def neg_sharpe_s(w):
                r = samp @ w
                port_cvar = -np.percentile(r, 5)
                return port_cvar
            res = minimize(neg_sharpe_s, eq, method="SLSQP", bounds=bnds, constraints=cons)
            if res.success and res.fun < best_cvar:
                best_cvar = res.fun
                best_w = res.x / res.x.sum()
        return best_w, cov

    if style == "taleb_barbell":
        w = np.ones(n) * (0.10 / n)
        safe_idx = [i for i, s in enumerate(selected) if s in TALEB_SAFE]
        risky_idx = [i for i, s in enumerate(selected) if s in TALEB_RISKY]
        other_idx = [i for i in range(n) if i not in safe_idx and i not in risky_idx]

        safe_count = len(safe_idx) or 1
        risky_count = len(risky_idx) or 1
        other_count = len(other_idx) or 1

        if safe_idx and risky_idx:
            for i in safe_idx:   w[i] = 0.90 / safe_count
            for i in risky_idx:  w[i] = 0.10 / risky_count
            for i in other_idx:  w[i] = 0.0
        elif safe_idx:
            for i in safe_idx:   w[i] = 0.90 / safe_count
            for i in other_idx:  w[i] = 0.10 / other_count
        else:
            w = eq

        w = np.clip(w, 0, 1)
        w /= w.sum()
        return w, cov

    return eq, cov


def portfolio_metrics(weights, returns, rf):
    port_ret = returns.values @ weights
    ann_ret = (1 + port_ret).prod() ** (252 / len(port_ret)) - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)

    # Max Drawdown
    cum = (1 + port_ret).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = dd.min()

    # Recovery Time
    dd_series = pd.Series(dd)
    in_dd = dd_series < -0.01
    max_recovery = 0
    streak = 0
    for v in in_dd:
        streak = streak + 1 if v else 0
        max_recovery = max(max_recovery, streak)

    # CVaR 95%
    cvar = -np.percentile(port_ret, 5)

    # Calmar
    calmar = ann_ret / (abs(max_dd) + 1e-9)

    return {
        "بازده سالانه": ann_ret,
        "نوسان سالانه": ann_vol,
        "نسبت شارپ": sharpe,
        "حداکثر افت (Max Drawdown)": max_dd,
        "CVaR 95%": cvar,
        "نسبت کالمار": calmar,
        "ریکاوری تایم (روز)": max_recovery,
    }


def bp_layout(title="", xt="", yt="", h=450):
    af = dict(color="#4fc3f7", size=10, family="Courier New")
    return dict(
        title=dict(text=title, font=dict(color="#4fc3f7", size=12, family="Courier New"), x=0.5),
        paper_bgcolor="#08111f",
        plot_bgcolor="#08111f",
        font=dict(color="#e8f4fd", family="Courier New", size=10),
        xaxis=dict(
            title=dict(text=xt, font=af),
            gridcolor="rgba(79,195,247,0.08)",
            linecolor="rgba(255,255,255,0.15)",
            tickfont=dict(color="#4fc3f7", size=9),
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=yt, font=af),
            gridcolor="rgba(79,195,247,0.08)",
            linecolor="rgba(255,255,255,0.15)",
            tickfont=dict(color="#4fc3f7", size=9),
            zeroline=True,
            zerolinecolor="rgba(79,195,247,0.25)",
            zerolinewidth=1,
        ),
        legend=dict(
            bgcolor="rgba(8,17,31,0.85)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(color="#e8f4fd", size=9),
        ),
        margin=dict(l=55, r=20, t=55, b=45),
        height=h,
    )


def metric_card(label, value, delta=None, delta_pos=True):
    delta_html = ""
    if delta is not None:
        cls = "delta-pos" if delta_pos else "delta-neg"
        arrow = "▲" if delta_pos else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    return f"""
    <div class="bp-metric">
        <div class="lbl">{label}</div>
        <div class="val">{value}</div>
        {delta_html}
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="bp-header">
    <h1>📐 Portfolio360</h1>
    <p>سیستم تحلیل و بهینه‌سازی پرتفوی — Blueprint Edition</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — انتخاب نمادها
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="bp-section">⚙️ تنظیمات</div>', unsafe_allow_html=True)

    period_label = st.selectbox("بازه زمانی", list(PERIODS.keys()), index=2)
    period = PERIODS[period_label]

    rf_pct = st.number_input("نرخ بدون ریسک (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    rf = rf_pct / 100

    st.markdown('<div class="bp-section">📋 انتخاب نمادها</div>', unsafe_allow_html=True)

    selected_tickers = []
    for cat, syms in SYMBOLS.items():
        with st.expander(cat, expanded=False):
            chosen = st.multiselect(
                f"انتخاب از {cat}",
                options=list(syms.keys()),
                format_func=lambda x, s=syms: f"{x} — {s[x]}",
                key=f"ms_{cat}",
                label_visibility="collapsed"
            )
            selected_tickers.extend(chosen)

    st.markdown(f"**{len(selected_tickers)} نماد انتخاب شده**")

    fetch_btn = st.button("📡 دانلود داده‌ها", use_container_width=True)

    st.markdown('<div class="bp-section">🎯 سبک پرتفوی</div>', unsafe_allow_html=True)
    style_label = st.selectbox("استراتژی", list(STYLES.keys()))
    style = STYLES[style_label]

    calc_btn = st.button("🧮 محاسبه پرتفوی", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# دانلود داده‌ها
# ─────────────────────────────────────────────────────────────────────────────
if fetch_btn:
    if len(selected_tickers) < 2:
        st.warning("⚠️ حداقل ۲ نماد انتخاب کنید.")
    else:
        with st.spinner("در حال دانلود از Yahoo Finance..."):
            prices, failed = fetch_data(tuple(selected_tickers), period)
        if prices is not None:
            st.session_state["prices"] = prices
            st.session_state["weights"] = None
            st.session_state["metrics"] = None
            if failed:
                st.warning(f"دانلود نشد: {', '.join(failed)}")
            st.success(f"✅ {len(prices.columns)} نماد بارگذاری شد — {len(prices)} روز معاملاتی")
        else:
            st.error("❌ دانلود ناموفق. نمادها را بررسی کنید.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
if "prices" not in st.session_state or st.session_state["prices"] is None:
    st.markdown("""
    <div style="border:1px solid rgba(79,195,247,0.3); padding:2rem; text-align:center; margin-top:3rem; background:rgba(79,195,247,0.03);">
        <div style="color:#4fc3f7; font-size:2rem;">📐</div>
        <div style="color:#4fc3f7; font-size:0.85rem; letter-spacing:0.15em; text-transform:uppercase; margin-top:0.5rem;">
            از سایدبار نمادها را انتخاب کنید و داده دانلود کنید
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

prices = st.session_state["prices"]
asset_names = list(prices.columns)
returns = prices.pct_change().dropna()

# محاسبه وزن‌ها
if calc_btn:
    if len(asset_names) < 2:
        st.warning("⚠️ حداقل ۲ نماد نیاز است.")
    else:
        with st.spinner("در حال محاسبه..."):
            w, cov = calc_weights(style, returns[asset_names], rf, asset_names)
            m = portfolio_metrics(w, returns[asset_names], rf)
        st.session_state["weights"] = w
        st.session_state["cov"] = cov
        st.session_state["metrics"] = m
        st.session_state["style_label"] = style_label

if st.session_state.get("weights") is None:
    st.info("⬅️ سبک پرتفوی را انتخاب و «محاسبه پرتفوی» را بزنید.")

# ── تب‌ها ──
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  تخصیص پرتفوی",
    "📉  ریسک و بازده",
    "🔮  نمودار قیمت",
    "📈  مقایسه سبک‌ها",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — تخصیص
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="bp-section">📊 توزیع وزن‌ها</div>', unsafe_allow_html=True)

    w = st.session_state.get("weights")
    if w is None:
        st.info("ابتدا پرتفوی را محاسبه کنید.")
    else:
        df_w = pd.DataFrame({
            "نماد": asset_names,
            "وزن (%)": np.round(w * 100, 2)
        }).sort_values("وزن (%)", ascending=False)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(df_w, use_container_width=True, hide_index=True)

            total_usd = st.number_input("💵 کل سرمایه (دلار)", min_value=100, value=10000, step=500)
            df_alloc = df_w.copy()
            df_alloc["مبلغ ($)"] = (df_alloc["وزن (%)"] / 100 * total_usd).round(2)
            st.dataframe(df_alloc[["نماد", "وزن (%)", "مبلغ ($)"]], use_container_width=True, hide_index=True)

            st.download_button("📥 دانلود CSV", df_alloc.to_csv(index=False),
                               file_name="portfolio.csv", use_container_width=True)

        with col2:
            bp_colors = ["#4fc3f7","#ffd54f","#64ffda","#ff8a65","#ce93d8",
                         "#80cbc4","#ffcc02","#ef9a9a","#b0bec5","#90caf9",
                         "#a5d6a7","#ffab91","#80deea","#ffe082","#b39ddb"]
            fig_pie = go.Figure(go.Pie(
                labels=df_w["نماد"], values=df_w["وزن (%)"],
                hole=0.42,
                marker=dict(colors=bp_colors[:len(df_w)],
                            line=dict(color="#08111f", width=2)),
                textfont=dict(color="#08111f", size=10, family="Courier New"),
                textinfo="percent+label",
            ))
            fig_pie.update_layout(**bp_layout(
                title=f"ALLOCATION — {st.session_state.get('style_label','')}", h=400
            ))
            st.plotly_chart(fig_pie, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — ریسک و بازده
# ════════════════════════════════════════════════════════════════════
with tab2:
    m = st.session_state.get("metrics")
    w = st.session_state.get("weights")
    if m is None or w is None:
        st.info("ابتدا پرتفوی را محاسبه کنید.")
    else:
        st.markdown('<div class="bp-section">📉 معیارهای کلیدی ریسک و بازده</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("بازده سالانه", f"{m['بازده سالانه']*100:.2f}%")
            st.metric("نسبت شارپ", f"{m['نسبت شارپ']:.3f}")
        with c2:
            st.metric("نوسان سالانه", f"{m['نوسان سالانه']*100:.2f}%")
            st.metric("نسبت کالمار", f"{m['نسبت کالمار']:.3f}")
        with c3:
            st.metric("حداکثر افت (MDD)", f"{m['حداکثر افت (Max Drawdown)']*100:.2f}%")
            st.metric("CVaR 95%", f"{m['CVaR 95%']*100:.2f}%")
        with c4:
            rec = m["ریکاوری تایم (روز)"]
            rec_months = int(rec / 21)
            rec_str = f"{rec_months} ماه" if rec_months else f"{rec} روز"
            st.metric("ریکاوری تایم", rec_str)
            st.metric("نمادها", str(len(asset_names)))

        st.markdown('<div class="bp-section">📉 نمودار Drawdown</div>', unsafe_allow_html=True)

        port_ret = returns[asset_names].values @ w
        cum = (1 + port_ret).cumprod()
        roll_max = pd.Series(cum).cummax()
        dd = (pd.Series(cum) - roll_max) / roll_max

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=prices.index[-len(dd):], y=dd.values * 100,
            mode="lines", name="Drawdown",
            line=dict(color="#ff6b6b", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,107,107,0.1)"
        ))
        fig_dd.update_layout(**bp_layout("DRAWDOWN (%)", "DATE", "DRAWDOWN %", 350))
        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown('<div class="bp-section">📈 منحنی رشد پرتفوی</div>', unsafe_allow_html=True)

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=prices.index[-len(cum):], y=cum,
            mode="lines", name="Portfolio",
            line=dict(color="#4fc3f7", width=2)
        ))
        # خط مبنا
        fig_cum.add_hline(y=1.0, line_dash="dash",
                          line_color="rgba(255,255,255,0.2)", line_width=1)
        fig_cum.update_layout(**bp_layout("PORTFOLIO GROWTH (BASE=1)", "DATE", "CUMULATIVE RETURN", 380))
        st.plotly_chart(fig_cum, use_container_width=True)

        st.markdown('<div class="bp-section">📊 توزیع بازده‌های روزانه</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=port_ret * 100, nbinsx=60,
            marker_color="rgba(79,195,247,0.6)",
            marker_line=dict(color="#08111f", width=0.5),
            name="بازده روزانه"
        ))
        cvar_line = np.percentile(port_ret, 5) * 100
        fig_hist.add_vline(x=cvar_line, line_dash="dash",
                           line_color="#ff6b6b", line_width=1.5,
                           annotation_text=f"CVaR 95%: {cvar_line:.2f}%",
                           annotation_font_color="#ff6b6b",
                           annotation_font_size=10)
        fig_hist.update_layout(**bp_layout("DAILY RETURN DISTRIBUTION", "RETURN %", "FREQUENCY", 360))
        st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — نمودار قیمت
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="bp-section">🔮 نمودار قیمت نمادها</div>', unsafe_allow_html=True)

    view_mode = st.radio("نمایش", ["نرمال‌شده (base=100)", "قیمت خام"], horizontal=True)

    bp_colors2 = ["#4fc3f7","#ffd54f","#64ffda","#ff8a65","#ce93d8",
                  "#80cbc4","#ffcc02","#ef9a9a","#b0bec5","#90caf9",
                  "#a5d6a7","#ffab91","#80deea","#ffe082","#b39ddb"]

    fig_price = go.Figure()
    for i, col in enumerate(asset_names):
        s = prices[col]
        y = (s / s.iloc[0] * 100).values if view_mode.startswith("نرمال") else s.values
        fig_price.add_trace(go.Scatter(
            x=prices.index, y=y,
            mode="lines", name=col,
            line=dict(color=bp_colors2[i % len(bp_colors2)], width=1.5)
        ))
    yt = "قیمت نرمال‌شده (base=100)" if view_mode.startswith("نرمال") else "قیمت ($)"
    fig_price.update_layout(**bp_layout("PRICE CHART", "DATE", yt, 520))
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown('<div class="bp-section">🔥 ماتریس همبستگی</div>', unsafe_allow_html=True)
    if len(asset_names) >= 2:
        corr = returns[asset_names].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[
                [0.0, "#ff6b6b"], [0.5, "#08111f"], [1.0, "#4fc3f7"]
            ],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9, color="white"),
            showscale=True,
        ))
        fig_corr.update_layout(**bp_layout("CORRELATION MATRIX", h=max(350, len(asset_names)*35)))
        fig_corr.update_layout(
            xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#4fc3f7")),
            yaxis=dict(tickfont=dict(size=9, color="#4fc3f7")),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — مقایسه سبک‌ها
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="bp-section">📈 مقایسه تمام سبک‌های پرتفوی</div>', unsafe_allow_html=True)

    if len(asset_names) < 2:
        st.info("حداقل ۲ نماد نیاز است.")
    else:
        run_compare = st.button("▶️ اجرای مقایسه همه سبک‌ها", use_container_width=True)
        if run_compare or st.session_state.get("compare_done"):
            if run_compare:
                with st.spinner("در حال محاسبه تمام سبک‌ها..."):
                    compare_results = {}
                    for lbl, sty in STYLES.items():
                        try:
                            ww, _ = calc_weights(sty, returns[asset_names], rf, asset_names)
                            mm = portfolio_metrics(ww, returns[asset_names], rf)
                            compare_results[lbl] = {**mm, "_weights": ww}
                        except Exception:
                            pass
                st.session_state["compare_results"] = compare_results
                st.session_state["compare_done"] = True

            compare_results = st.session_state.get("compare_results", {})
            if compare_results:
                rows = []
                for lbl, mm in compare_results.items():
                    rec = mm["ریکاوری تایم (روز)"]
                    rec_m = int(rec / 21)
                    rows.append({
                        "سبک": lbl,
                        "بازده سالانه (%)": round(mm["بازده سالانه"] * 100, 2),
                        "نوسان (%)": round(mm["نوسان سالانه"] * 100, 2),
                        "شارپ": round(mm["نسبت شارپ"], 3),
                        "MDD (%)": round(mm["حداکثر افت (Max Drawdown)"] * 100, 2),
                        "CVaR 95% (%)": round(mm["CVaR 95%"] * 100, 2),
                        "ریکاوری تایم": f"{rec_m} ماه" if rec_m else f"{rec} روز",
                        "کالمار": round(mm["نسبت کالمار"], 3),
                    })
                df_cmp = pd.DataFrame(rows)
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)

                # نمودار مقایسه‌ای
                fig_cmp = go.Figure()
                metrics_to_plot = ["بازده سالانه (%)", "نوسان (%)", "شارپ", "MDD (%)"]
                for i, metric in enumerate(metrics_to_plot):
                    vals = [r[metric] for r in rows]
                    fig_cmp.add_trace(go.Bar(
                        name=metric,
                        x=[r["سبک"].split("(")[0].strip() for r in rows],
                        y=vals,
                        marker_color=bp_colors2[i],
                        marker_line=dict(color="#08111f", width=0.5),
                    ))
                fig_cmp.update_layout(**bp_layout("STRATEGY COMPARISON", "", "VALUE", 420))
                fig_cmp.update_layout(barmode="group")
                st.plotly_chart(fig_cmp, use_container_width=True)

                # منحنی رشد همه سبک‌ها
                st.markdown('<div class="bp-section">📈 منحنی رشد همه سبک‌ها</div>', unsafe_allow_html=True)
                fig_growth = go.Figure()
                for i, (lbl, mm) in enumerate(compare_results.items()):
                    ww = mm["_weights"]
                    pr = returns[asset_names].values @ ww
                    cum_c = (1 + pr).cumprod()
                    fig_growth.add_trace(go.Scatter(
                        x=prices.index[-len(cum_c):], y=cum_c,
                        mode="lines",
                        name=lbl.split("(")[0].strip(),
                        line=dict(color=bp_colors2[i % len(bp_colors2)], width=1.8)
                    ))
                fig_growth.update_layout(**bp_layout("GROWTH CURVES — ALL STRATEGIES", "DATE", "CUMULATIVE RETURN (BASE=1)", 480))
                st.plotly_chart(fig_growth, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:rgba(79,195,247,0.4); font-size:0.65rem;
    letter-spacing:0.15em; text-transform:uppercase; padding:0.5rem;">
    Portfolio360 Blueprint Edition — Data via Yahoo Finance
</div>
""", unsafe_allow_html=True)
