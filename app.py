"""
Portfolio360 — Blueprint Edition
سبک‌های پرتفوی + نمادهای جهانی
با سایدبار بازطراحی‌شده + دو تم روشن/تاریک
+ بازده مورد انتظار و ریسک‌های سه‌گانه
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
# THEME STATE
# ─────────────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

theme = st.session_state["theme"]
is_dark = theme == "dark"

# ─────────────────────────────────────────────────────────────────────────────
# THEME CSS
# ─────────────────────────────────────────────────────────────────────────────

LIGHT_CSS = """
:root {
    --bg:         #f0f0ed;
    --bg2:        #e8e8e4;
    --panel:      #ddddd8;
    --card:       #e4e4e0;
    --border:     rgba(60,60,55,0.15);
    --border2:    rgba(60,60,55,0.08);
    --accent:     #1a1a18;
    --accent2:    #555550;
    --gold:       #8a6a1a;
    --green:      #1a6640;
    --red:        #8a2020;
    --white:      #111110;
    --silver:     #444440;
    --muted:      #888882;
    --sb-bg:      #e0e0db;
    --sb-border:  rgba(0,0,0,0.08);
    --card-top:   #333330;
    --input-bg:   #d8d8d3;
    --btn-color:  #1a1a18;
    --btn-border: rgba(26,26,24,0.4);
    --btn-hover:  rgba(26,26,24,0.06);
    --tag-bg:     rgba(26,26,24,0.08);
    --tag-border: rgba(26,26,24,0.25);
    --tag-color:  #1a1a18;
    --plot-bg:    #e8e8e4;
    --plot-grid:  rgba(60,60,55,0.1);
    --plot-line:  rgba(0,0,0,0.15);
    --plot-tick:  #444440;
    --plot-text:  #222220;
    --risk-geo:   #7a3a00;
    --risk-mon:   #1a4a7a;
    --risk-sys:   #4a1a6a;
    --risk-bg:    rgba(26,26,24,0.04);
}
"""

DARK_CSS = """
:root {
    --bg:         #111111;
    --bg2:        #181818;
    --panel:      #1e1e1e;
    --card:       #222222;
    --border:     rgba(180,180,180,0.1);
    --border2:    rgba(255,255,255,0.05);
    --accent:     #b0b0b0;
    --accent2:    #888888;
    --gold:       #c8a84b;
    --green:      #5aaa78;
    --red:        #cc5555;
    --white:      #d4d4d4;
    --silver:     #909090;
    --muted:      #555555;
    --sb-bg:      #0e0e0e;
    --sb-border:  rgba(255,255,255,0.06);
    --card-top:   #888888;
    --input-bg:   #161616;
    --btn-color:  #b0b0b0;
    --btn-border: rgba(180,180,180,0.3);
    --btn-hover:  rgba(180,180,180,0.07);
    --tag-bg:     rgba(180,180,180,0.07);
    --tag-border: rgba(180,180,180,0.2);
    --tag-color:  #b0b0b0;
    --plot-bg:    #161616;
    --plot-grid:  rgba(255,255,255,0.05);
    --plot-line:  rgba(255,255,255,0.1);
    --plot-tick:  #888888;
    --plot-text:  #c0c0c0;
    --risk-geo:   #e8945a;
    --risk-mon:   #5a9be8;
    --risk-sys:   #b07ad4;
    --risk-bg:    rgba(255,255,255,0.03);
}
"""

SHARED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body { background: var(--bg) !important; }

.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
}

*, .stApp *, [data-testid="stSidebar"] * {
    font-family: 'Inter', system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ── GLOBAL TEXT ── */
p, li, div {
    color: var(--white) !important;
    font-size: 0.93rem !important;
    line-height: 1.65 !important;
}

label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--silver) !important;
    margin-bottom: 4px !important;
}

/* ════════════════════════════
   SIDEBAR — REDESIGN
════════════════════════════ */
[data-testid="stSidebar"] > div:first-child {
    background: var(--sb-bg) !important;
    border-right: 1px solid var(--sb-border) !important;
    padding: 0 !important;
}

/* ── SIDEBAR HEADER BAR ── */
.sb-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.1rem 0.9rem;
    border-bottom: 1px solid var(--border2);
}
.sb-wordmark {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase;
    color: var(--accent) !important;
    line-height: 1 !important;
}
.sb-version {
    font-size: 0.6rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    color: var(--muted) !important;
    margin-top: 3px;
    line-height: 1 !important;
}

/* ── SIDEBAR SECTION LABEL ── */
.sb-section-label {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0.85rem 1.1rem 0.4rem;
}
.sb-section-label-text {
    font-size: 0.58rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    line-height: 1 !important;
}
.sb-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border2);
}

/* ── ASSET COUNTER PILL ── */
.sb-counter {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0.2rem 1.1rem 0.6rem;
    padding: 0.5rem 0.75rem;
    background: var(--card);
    border: 1px solid var(--border2);
    border-radius: 4px;
}
.sb-counter-label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    line-height: 1 !important;
}
.sb-counter-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    line-height: 1 !important;
}
.sb-counter-value.warn {
    color: var(--red) !important;
}

/* ── RISK PANEL ── */
.risk-panel {
    margin: 0.3rem 0.8rem 0.5rem;
    padding: 0.7rem 0.85rem;
    background: var(--risk-bg);
    border: 1px solid var(--border2);
    border-radius: 4px;
}
.risk-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.18rem 0;
}
.risk-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    flex-shrink: 0;
}
.risk-label {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    color: var(--silver) !important;
    flex: 1;
    line-height: 1 !important;
}
.risk-bar-wrap {
    width: 52px;
    height: 4px;
    background: var(--border2);
    border-radius: 2px;
    overflow: hidden;
    margin: 0 8px;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 2px;
}
.risk-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    min-width: 36px;
    text-align: right;
    line-height: 1 !important;
}
.risk-impact-note {
    margin-top: 0.5rem;
    padding-top: 0.4rem;
    border-top: 1px solid var(--border2);
    font-size: 0.6rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.06em !important;
    line-height: 1.5 !important;
}
.expected-ret-display {
    margin: 0 0.8rem 0.3rem;
    padding: 0.5rem 0.85rem;
    background: var(--card);
    border: 1px solid var(--border2);
    border-left: 3px solid var(--gold);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.expected-ret-label {
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    line-height: 1 !important;
}
.expected-ret-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: var(--gold) !important;
    line-height: 1 !important;
}

/* ── INPUTS ── */
.stTextInput input, .stNumberInput input {
    background: var(--input-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--white) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.44rem 0.7rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent2) !important;
    outline: none !important;
}

/* ── SELECT / MULTISELECT ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--input-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    color: var(--white) !important;
}
[data-testid="stMultiSelect"] > div > div {
    background: var(--input-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: var(--tag-bg) !important;
    border: 1px solid var(--tag-border) !important;
    border-radius: 3px !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    color: var(--tag-color) !important;
    padding: 2px 6px !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    color: var(--btn-color) !important;
    border: 1px solid var(--btn-border) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 0.9rem !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--btn-hover) !important;
    border-color: var(--accent) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    border: none !important;
    border-top: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    background: transparent !important;
    margin: 0 !important;
}
[data-testid="stExpander"] summary {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 1.1rem !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--silver) !important;
    transition: color 0.12s, background 0.12s !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--accent) !important;
    background: var(--border2) !important;
}
[data-testid="stExpander"] > div > div {
    padding: 0.2rem 0.5rem 0.5rem !important;
}

/* ════════════════════════════
   MAIN AREA
════════════════════════════ */
.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2rem !important;
}

/* ── PAGE HEADER ── */
.bp-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 2px solid var(--accent);
    padding: 0 0 1rem 0;
    margin-bottom: 1.8rem;
}
.bp-header-left h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    color: var(--accent) !important;
    margin: 0 0 0.2rem 0 !important;
    line-height: 1.15 !important;
}
.bp-header-left p {
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}
.bp-header-right {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.14em !important;
    text-align: right;
    line-height: 1.6;
}

/* ── SECTION DIVIDERS ── */
.bp-section {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 2rem 0 1.1rem;
}
.bp-section-text {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--accent2) !important;
    line-height: 1 !important;
    white-space: nowrap;
}
.bp-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border2);
}

/* ── RISK BADGE in main area ── */
.risk-badge-row {
    display: flex;
    gap: 10px;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.risk-badge {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 0.4rem 0.75rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--card);
}
.risk-badge-icon {
    font-size: 0.8rem;
}
.risk-badge-label {
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    line-height: 1 !important;
}
.risk-badge-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
}

/* ── METRIC CARDS ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border2) !important;
    border-top: 2px solid var(--card-top) !important;
    border-radius: 4px !important;
    padding: 1rem 1.1rem !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 0.25rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border2) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 0.7rem 1rem !important;
    transition: all 0.12s !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--silver) !important;
    background: var(--border2) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border2) !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}

/* ── ALERTS ── */
.stAlert {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent2) !important;
    border-radius: 4px !important;
    font-size: 0.88rem !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--border2) !important;
    margin: 1.6rem 0 !important;
}

/* ── THEME TOGGLE BUTTON ── */
.theme-toggle-wrap .stButton > button {
    border-radius: 20px !important;
    padding: 0.3rem 0.8rem !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.15em !important;
    width: auto !important;
}

/* ── RADIO ── */
[data-testid="stRadio"] label {
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em !important;
    text-transform: none !important;
    color: var(--silver) !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] {
    padding: 0 0.3rem !important;
}
"""

# inject CSS
theme_vars = DARK_CSS if is_dark else LIGHT_CSS
st.markdown(f"<style>{theme_vars}{SHARED_CSS}</style>", unsafe_allow_html=True)

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
    "بیشترین شارپ (Markowitz)": "max_sharpe",
    "کمترین واریانس (Min Variance)": "min_var",
    "مونت‌کارلو (CVaR)": "monte_carlo",
    "وزن برابر (Equal Weight)": "equal_weight",
    "ریسک پاریتی (Risk Parity)": "risk_parity",
    "۹۰/۱۰ طالب (Taleb Barbell)": "taleb_barbell",
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


def calc_risk_penalty(risk_geo: float, risk_mon: float, risk_sys: float) -> float:
    """
    محاسبه ضریب تنزل بازده بر اساس ریسک‌های وارد‌شده.
    منطق: هر ریسک با وزن متفاوتی روی بازده نهایی تأثیر می‌ذاره.
    - ریسک ژئوپولیتیک: تأثیر نوسانی بیشتر (وزن ۴۰٪)
    - ریسک سیاست پولی: تأثیر روی نرخ تنزیل (وزن ۳۵٪)
    - ریسک سیستماتیک: تأثیر کلی بازار (وزن ۲۵٪)
    مجموع ریسک‌ها تا حداکثر ۵۰٪ بازده مورد انتظار رو کاهش می‌ده.
    """
    weighted_risk = (risk_geo * 0.40 + risk_mon * 0.35 + risk_sys * 0.25) / 100.0
    # penalty: حداکثر ۵۰٪ کاهش بازده
    penalty = min(weighted_risk * 0.5, 0.50)
    return penalty


def calc_weights(style: str, returns: pd.DataFrame, rf: float, selected: list,
                 expected_return: float = 0.0, risk_geo: float = 0.0,
                 risk_mon: float = 0.0, risk_sys: float = 0.0):
    n = len(selected)
    eq = np.ones(n) / n
    mean_r = returns.mean() * 252
    cov = returns.cov() * 252
    bnds = [(0.01, 0.6)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    # تأثیر ریسک‌ها روی نرخ بدون‌ریسک مؤثر (افزایش هزینه ریسک)
    risk_penalty = calc_risk_penalty(risk_geo, risk_mon, risk_sys)
    effective_rf = rf + risk_penalty

    # اگه بازده مورد انتظار وارد شده، قید اضافه می‌کنیم
    expected_ret_ann = expected_return / 100.0
    use_expected_ret = expected_ret_ann > 0.001

    if style == "equal_weight":
        return eq, cov

    if style == "max_sharpe":
        def neg_sharpe(w):
            r = w @ mean_r.values
            v = np.sqrt(w @ cov.values @ w)
            return -(r - effective_rf) / (v + 1e-9)

        active_cons = list(cons)
        if use_expected_ret:
            # قید: بازده پرتفو حداقل باید برابر انتظار باشه
            active_cons.append({
                "type": "ineq",
                "fun": lambda w: (w @ mean_r.values) - expected_ret_ann
            })

        res = minimize(neg_sharpe, eq, method="SLSQP", bounds=bnds, constraints=active_cons)
        if not res.success:
            # اگه با قید بازده جواب نداد، بدون قید ولی با شارپ بهتر
            res = minimize(neg_sharpe, eq, method="SLSQP", bounds=bnds, constraints=cons)
        return (res.x / res.x.sum() if res.success else eq), cov

    if style == "min_var":
        def port_var(w): return w @ cov.values @ w

        active_cons = list(cons)
        if use_expected_ret:
            active_cons.append({
                "type": "ineq",
                "fun": lambda w: (w @ mean_r.values) - expected_ret_ann
            })

        res = minimize(port_var, eq, method="SLSQP", bounds=bnds, constraints=active_cons)
        if not res.success:
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


def portfolio_metrics(weights, returns, rf, expected_return=0.0,
                      risk_geo=0.0, risk_mon=0.0, risk_sys=0.0):
    port_ret = returns.values @ weights
    ann_ret = (1 + port_ret).prod() ** (252 / len(port_ret)) - 1
    ann_vol = port_ret.std() * np.sqrt(252)

    # اعمال تأثیر ریسک‌ها روی بازده تعدیل‌شده
    risk_penalty = calc_risk_penalty(risk_geo, risk_mon, risk_sys)
    risk_adjusted_ret = ann_ret * (1 - risk_penalty)

    sharpe = (risk_adjusted_ret - rf) / (ann_vol + 1e-9)

    cum = pd.Series((1 + port_ret).cumprod())
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = float(dd.min())

    max_recovery = 0
    streak = 0
    for v in (dd < -0.01):
        streak = streak + 1 if v else 0
        max_recovery = max(max_recovery, streak)

    cvar = float(-np.percentile(port_ret, 5))
    calmar = risk_adjusted_ret / (abs(max_dd) + 1e-9)

    # واگرایی از بازده مورد انتظار
    expected_ret_ann = expected_return / 100.0
    ret_gap = risk_adjusted_ret - expected_ret_ann if expected_ret_ann > 0 else None

    return {
        "بازده سالانه":                    float(ann_ret),
        "بازده تعدیل‌شده ریسک":            float(risk_adjusted_ret),
        "نوسان سالانه":                    float(ann_vol),
        "نسبت شارپ":                       float(sharpe),
        "حداکثر افت (Max Drawdown)":       max_dd,
        "CVaR 95%":                        cvar,
        "نسبت کالمار":                     float(calmar),
        "ریکاوری تایم (روز)":              int(max_recovery),
        "واگرایی از هدف":                  ret_gap,
        "تنزل ریسک (%)":                   float(risk_penalty * 100),
    }


def get_plot_layout(title="", xt="", yt="", h=450):
    bg   = "#161616" if is_dark else "#e8e8e4"
    grid = "rgba(255,255,255,0.05)" if is_dark else "rgba(60,60,55,0.1)"
    line = "rgba(255,255,255,0.1)"  if is_dark else "rgba(0,0,0,0.12)"
    tick = "#888888" if is_dark else "#555550"
    txt  = "#c0c0c0" if is_dark else "#222220"
    af   = dict(color=tick, size=10, family="JetBrains Mono, Courier New")
    return dict(
        title=dict(text=title, font=dict(color=txt, size=11, family="JetBrains Mono, Courier New"), x=0.5),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=txt, family="JetBrains Mono, Courier New", size=9),
        xaxis=dict(
            title=dict(text=xt, font=af),
            gridcolor=grid, linecolor=line,
            tickfont=dict(color=tick, size=9),
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=yt, font=af),
            gridcolor=grid, linecolor=line,
            tickfont=dict(color=tick, size=9),
            zeroline=True,
            zerolinecolor=grid, zerolinewidth=1,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)" if is_dark else "rgba(240,240,237,0.85)",
            bordercolor=line, borderwidth=1,
            font=dict(color=txt, size=9),
        ),
        margin=dict(l=55, r=20, t=50, b=45),
        height=h,
    )


COLORS = [
    "#5b9bd5","#e8a838","#3db87a","#e05c5c","#9b72c8",
    "#48b8c0","#d47f3a","#c45b8e","#7eb35a","#5a8fc4",
    "#d4a855","#4db88c","#c46060","#8062b8","#40a8a8",
]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    col_brand, col_toggle = st.columns([3, 2])
    with col_brand:
        st.markdown("""
        <div style="padding:0.9rem 0 0.7rem 0.2rem;">
            <div class="sb-wordmark">Portfolio360</div>
            <div class="sb-version">Blueprint · v3.1</div>
        </div>
        """, unsafe_allow_html=True)
    with col_toggle:
        st.markdown("<div style='padding-top:0.85rem'>", unsafe_allow_html=True)
        toggle_label = "☀ Light" if is_dark else "● Dark"
        if st.button(toggle_label, key="theme_btn"):
            st.session_state["theme"] = "light" if is_dark else "dark"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:0 0 0.5rem 0'>", unsafe_allow_html=True)

    # ══ SECTION: CONFIG ══
    st.markdown('<div class="sb-section-label"><span class="sb-section-label-text">تنظیمات</span></div>', unsafe_allow_html=True)

    period_label = st.selectbox("بازه زمانی", list(PERIODS.keys()), index=2, label_visibility="visible")
    period = PERIODS[period_label]

    rf_pct = st.number_input(
        "نرخ بدون ریسک (%)",
        min_value=0.0, max_value=50.0,
        value=5.0, step=0.5,
        help="برای محاسبه نسبت شارپ"
    )
    rf = rf_pct / 100

    # ══ SECTION: انتظارات و ریسک ══
    st.markdown('<div class="sb-section-label" style="margin-top:0.5rem"><span class="sb-section-label-text">انتظارات و ریسک</span></div>', unsafe_allow_html=True)

    expected_return = st.number_input(
        "بازده مورد انتظار (%)",
        min_value=0.0, max_value=1000.0,
        value=0.0, step=5.0,
        help="بازده سالانه‌ای که انتظار دارید. صفر یعنی اعمال نشود."
    )

    # نمایش بازده مورد انتظار به صورت بصری
    if expected_return > 0:
        st.markdown(f"""
        <div class="expected-ret-display">
            <span class="expected-ret-label">هدف بازده</span>
            <span class="expected-ret-value">{expected_return:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ریسک ژئوپولیتیک
    risk_geo = st.slider(
        "🌐 ریسک ژئوپولیتیک (%)",
        min_value=0, max_value=100, value=0, step=5,
        help="تأثیر: کاهش بازده تعدیل‌شده از طریق افزایش هزینه ریسک (وزن ۴۰٪)"
    )

    # ریسک سیاست پولی
    risk_mon = st.slider(
        "🏦 ریسک سیاست پولی (%)",
        min_value=0, max_value=100, value=0, step=5,
        help="تأثیر: تغییر نرخ تنزیل مؤثر و کاهش بازده تعدیل‌شده (وزن ۳۵٪)"
    )

    # ریسک سیستماتیک
    risk_sys = st.slider(
        "📉 ریسک سیستماتیک (%)",
        min_value=0, max_value=100, value=0, step=5,
        help="تأثیر: ریسک کل بازار که قابل تنوع‌بخشی نیست (وزن ۲۵٪)"
    )

    # نمایش بصری ریسک‌ها در سایدبار
    if risk_geo > 0 or risk_mon > 0 or risk_sys > 0:
        total_penalty = calc_risk_penalty(risk_geo, risk_mon, risk_sys)

        geo_color  = "#e8945a" if is_dark else "#7a3a00"
        mon_color  = "#5a9be8" if is_dark else "#1a4a7a"
        sys_color  = "#b07ad4" if is_dark else "#4a1a6a"

        def bar_html(val, color):
            pct = min(val, 100)
            return f'<div class="risk-bar-wrap"><div class="risk-bar-fill" style="width:{pct}%;background:{color}"></div></div>'

        st.markdown(f"""
        <div class="risk-panel">
            <div class="risk-row">
                <span class="risk-dot" style="background:{geo_color}"></span>
                <span class="risk-label">ژئوپولیتیک</span>
                {bar_html(risk_geo, geo_color)}
                <span class="risk-value" style="color:{geo_color}">{risk_geo}%</span>
            </div>
            <div class="risk-row">
                <span class="risk-dot" style="background:{mon_color}"></span>
                <span class="risk-label">سیاست پولی</span>
                {bar_html(risk_mon, mon_color)}
                <span class="risk-value" style="color:{mon_color}">{risk_mon}%</span>
            </div>
            <div class="risk-row">
                <span class="risk-dot" style="background:{sys_color}"></span>
                <span class="risk-label">سیستماتیک</span>
                {bar_html(risk_sys, sys_color)}
                <span class="risk-value" style="color:{sys_color}">{risk_sys}%</span>
            </div>
            <div class="risk-impact-note">
                تنزل بازده تعدیل‌شده: <strong style="color:{'#cc5555' if is_dark else '#8a2020'}">{total_penalty*100:.1f}%</strong>
                &nbsp;·&nbsp; نرخ بدون‌ریسک مؤثر: <strong>{(rf + total_penalty)*100:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ══ SECTION: ASSETS ══
    st.markdown('<div class="sb-section-label" style="margin-top:0.5rem"><span class="sb-section-label-text">انتخاب دارایی</span></div>', unsafe_allow_html=True)

    selected_tickers = []
    for cat, syms in SYMBOLS.items():
        with st.expander(cat, expanded=False):
            chosen = st.multiselect(
                cat,
                options=list(syms.keys()),
                format_func=lambda x, s=syms: f"{x}  ·  {s[x]}",
                key=f"ms_{cat}",
                label_visibility="collapsed"
            )
            selected_tickers.extend(chosen)

    n_sel = len(selected_tickers)
    val_cls = "" if n_sel >= 2 else "warn"
    st.markdown(f"""
    <div class="sb-counter">
        <span class="sb-counter-label">دارایی انتخاب‌شده</span>
        <span class="sb-counter-value {val_cls}">{n_sel}</span>
    </div>
    """, unsafe_allow_html=True)

    fetch_btn = st.button("↓  دریافت داده از Yahoo Finance", use_container_width=True)

    # ══ SECTION: STRATEGY ══
    st.markdown('<div class="sb-section-label" style="margin-top:0.6rem"><span class="sb-section-label-text">استراتژی</span></div>', unsafe_allow_html=True)

    style_label = st.selectbox("روش بهینه‌سازی", list(STYLES.keys()))
    style = STYLES[style_label]

    calc_btn = st.button("▶  محاسبه پرتفوی", use_container_width=True)

    # ══ SECTION: GUIDE ══
    st.markdown('<div class="sb-section-label" style="margin-top:0.4rem"><span class="sb-section-label-text">راهنما</span></div>', unsafe_allow_html=True)
    with st.expander("مراحل استفاده", expanded=False):
        steps = [
            ("۱", "نمادها را از گروه‌های بالا انتخاب کنید"),
            ("۲", "دکمه دریافت داده را بزنید"),
            ("۳", "بازده مورد انتظار و ریسک‌ها را تنظیم کنید"),
            ("۴", "روش بهینه‌سازی را انتخاب کنید"),
            ("۵", "دکمه محاسبه پرتفوی را بزنید"),
        ]
        for num, txt in steps:
            st.markdown(f"""
            <div style="display:flex;gap:10px;align-items:flex-start;padding:0.22rem 0;">
                <span style="font-family:monospace;font-size:0.7rem;font-weight:700;
                    color:var(--accent2);min-width:14px;line-height:1.6">{num}</span>
                <span style="font-size:0.78rem;color:var(--silver);line-height:1.6">{txt}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top:0.6rem;padding:0.5rem 0.6rem;background:var(--border2);
            border-radius:3px;font-size:0.7rem;color:var(--muted);line-height:1.7">
            🌐 ریسک ژئوپولیتیک: جنگ، تحریم، بی‌ثباتی<br>
            🏦 ریسک سیاست پولی: نرخ بهره، QE/QT<br>
            📉 ریسک سیستماتیک: ریزش کل بازار
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top:0.4rem;padding:0.4rem 0.6rem;background:var(--border2);
            border-radius:3px;font-size:0.7rem;color:var(--muted)">
            ⚠ حداقل ۲ نماد برای محاسبه لازم است
        </div>
        """, unsafe_allow_html=True)


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
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%Y·%m·%d")
st.markdown(f"""
<div class="bp-header">
    <div class="bp-header-left">
        <h1>Portfolio360</h1>
        <p>Portfolio Analysis &amp; Optimization System · Blueprint Edition</p>
    </div>
    <div class="bp-header-right">
        {now_str}<br>
        {'■ Dark' if is_dark else '□ Light'}
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
if "prices" not in st.session_state or st.session_state["prices"] is None:
    empty_border = "rgba(180,180,180,0.15)" if is_dark else "rgba(60,60,55,0.15)"
    empty_color  = "#888888" if is_dark else "#555550"
    st.markdown(f"""
    <div style="border:1px solid {empty_border}; padding:3rem 2rem; text-align:center;
        margin-top:3rem; border-radius:6px;">
        <div style="font-size:2rem;margin-bottom:0.8rem">📐</div>
        <div style="color:{empty_color}; font-size:0.75rem; letter-spacing:0.18em;
            text-transform:uppercase; font-family:'JetBrains Mono',monospace;">
            از سایدبار نمادها را انتخاب کنید و داده دانلود کنید
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

prices = st.session_state["prices"]
asset_names = list(prices.columns)
returns = prices.pct_change().dropna()

if calc_btn:
    if len(asset_names) < 2:
        st.warning("⚠️ حداقل ۲ نماد نیاز است.")
    else:
        with st.spinner("در حال محاسبه..."):
            w, cov = calc_weights(
                style, returns[asset_names], rf, asset_names,
                expected_return=expected_return,
                risk_geo=risk_geo, risk_mon=risk_mon, risk_sys=risk_sys
            )
            m = portfolio_metrics(
                w, returns[asset_names], rf,
                expected_return=expected_return,
                risk_geo=risk_geo, risk_mon=risk_mon, risk_sys=risk_sys
            )
        st.session_state["weights"] = w
        st.session_state["cov"] = cov
        st.session_state["metrics"] = m
        st.session_state["style_label"] = style_label
        st.session_state["saved_risks"] = {
            "expected_return": expected_return,
            "risk_geo": risk_geo,
            "risk_mon": risk_mon,
            "risk_sys": risk_sys,
        }

if st.session_state.get("weights") is None:
    st.info("⬅️ سبک پرتفوی را انتخاب و «محاسبه پرتفوی» را بزنید.")

# ── نمایش نوار ریسک در بالای صفحه اصلی (اگه ریسک تنظیم شده)
saved = st.session_state.get("saved_risks", {})
if saved and (saved.get("risk_geo",0) > 0 or saved.get("risk_mon",0) > 0
              or saved.get("risk_sys",0) > 0 or saved.get("expected_return",0) > 0):

    geo_color_main  = "#e8945a" if is_dark else "#7a3a00"
    mon_color_main  = "#5a9be8" if is_dark else "#1a4a7a"
    sys_color_main  = "#b07ad4" if is_dark else "#4a1a6a"
    gold_color_main = "#c8a84b" if is_dark else "#8a6a1a"

    penalty_pct = calc_risk_penalty(
        saved.get("risk_geo",0), saved.get("risk_mon",0), saved.get("risk_sys",0)
    ) * 100

    badges = []
    if saved.get("expected_return",0) > 0:
        badges.append(f"""<div class="risk-badge">
            <span class="risk-badge-icon">🎯</span>
            <span class="risk-badge-label">هدف بازده</span>
            <span class="risk-badge-value" style="color:{gold_color_main}">{saved['expected_return']:.0f}%</span>
        </div>""")
    if saved.get("risk_geo",0) > 0:
        badges.append(f"""<div class="risk-badge">
            <span class="risk-badge-icon">🌐</span>
            <span class="risk-badge-label">ژئوپولیتیک</span>
            <span class="risk-badge-value" style="color:{geo_color_main}">{saved['risk_geo']}%</span>
        </div>""")
    if saved.get("risk_mon",0) > 0:
        badges.append(f"""<div class="risk-badge">
            <span class="risk-badge-icon">🏦</span>
            <span class="risk-badge-label">سیاست پولی</span>
            <span class="risk-badge-value" style="color:{mon_color_main}">{saved['risk_mon']}%</span>
        </div>""")
    if saved.get("risk_sys",0) > 0:
        badges.append(f"""<div class="risk-badge">
            <span class="risk-badge-icon">📉</span>
            <span class="risk-badge-label">سیستماتیک</span>
            <span class="risk-badge-value" style="color:{sys_color_main}">{saved['risk_sys']}%</span>
        </div>""")
    if penalty_pct > 0:
        red_c = "#cc5555" if is_dark else "#8a2020"
        badges.append(f"""<div class="risk-badge" style="border-color:{'rgba(204,85,85,0.3)' if is_dark else 'rgba(138,32,32,0.3)'}">
            <span class="risk-badge-icon">⚠</span>
            <span class="risk-badge-label">تنزل بازده</span>
            <span class="risk-badge-value" style="color:{red_c}">−{penalty_pct:.1f}%</span>
        </div>""")

    st.markdown(f'<div class="risk-badge-row">{"".join(badges)}</div>', unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs([
    "تخصیص پرتفوی",
    "ریسک و بازده",
    "نمودار قیمت",
    "مقایسه سبک‌ها",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — تخصیص
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="bp-section"><span class="bp-section-text">Portfolio Allocation</span></div>', unsafe_allow_html=True)

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

            total_usd = st.number_input("کل سرمایه ($)", min_value=100, value=10000, step=500)
            df_alloc = df_w.copy()
            df_alloc["مبلغ ($)"] = (df_alloc["وزن (%)"] / 100 * total_usd).round(2)
            st.dataframe(df_alloc[["نماد", "وزن (%)", "مبلغ ($)"]], use_container_width=True, hide_index=True)

            st.download_button("↓ دانلود CSV", df_alloc.to_csv(index=False),
                               file_name="portfolio.csv", use_container_width=True)

        with col2:
            fig_pie = go.Figure(go.Pie(
                labels=df_w["نماد"], values=df_w["وزن (%)"],
                hole=0.44,
                marker=dict(
                    colors=COLORS[:len(df_w)],
                    line=dict(color="#161616" if is_dark else "#e8e8e4", width=2)
                ),
                textfont=dict(size=9, family="JetBrains Mono"),
                textinfo="percent+label",
            ))
            fig_pie.update_layout(**get_plot_layout(
                title=f"ALLOCATION — {st.session_state.get('style_label','')}", h=400
            ))
            st.plotly_chart(fig_pie, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — ریسک و بازده
# ════════════════════════════════════════════════════════════════════
with tab2:
    m = st.session_state.get("metrics")
    w = st.session_state.get("weights")
    saved = st.session_state.get("saved_risks", {})

    if m is None or w is None:
        st.info("ابتدا پرتفوی را محاسبه کنید.")
    else:
        st.markdown('<div class="bp-section"><span class="bp-section-text">Risk &amp; Return Metrics</span></div>', unsafe_allow_html=True)

        # ── سطر اول متریک‌ها ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("بازده سالانه (خام)", f"{m['بازده سالانه']*100:.2f}%")
            st.metric("نسبت شارپ", f"{m['نسبت شارپ']:.3f}")
        with c2:
            adj_ret = m["بازده تعدیل‌شده ریسک"]
            raw_ret = m["بازده سالانه"]
            delta_adj = f"−{(raw_ret - adj_ret)*100:.1f}% تنزل" if raw_ret != adj_ret else "بدون تنزل"
            st.metric("بازده تعدیل‌شده ریسک", f"{adj_ret*100:.2f}%", delta=delta_adj,
                      delta_color="inverse")
            st.metric("نسبت کالمار", f"{m['نسبت کالمار']:.3f}")
        with c3:
            st.metric("نوسان سالانه", f"{m['نوسان سالانه']*100:.2f}%")
            st.metric("CVaR 95%", f"{m['CVaR 95%']*100:.2f}%")
        with c4:
            rec = m["ریکاوری تایم (روز)"]
            rec_months = int(rec / 21)
            rec_str = f"{rec_months} ماه" if rec_months else f"{rec} روز"
            st.metric("ریکاوری تایم", rec_str)
            st.metric("حداکثر افت (MDD)", f"{m['حداکثر افت (Max Drawdown)']*100:.2f}%")

        # ── سطر دوم: واگرایی از هدف و تنزل ریسک ──
        if saved.get("expected_return", 0) > 0 or saved.get("risk_geo", 0) > 0 \
                or saved.get("risk_mon", 0) > 0 or saved.get("risk_sys", 0) > 0:

            st.markdown('<div class="bp-section"><span class="bp-section-text">Risk-Adjusted Analysis</span></div>', unsafe_allow_html=True)

            ca, cb, cc, cd = st.columns(4)
            with ca:
                penalty = m["تنزل ریسک (%)"]
                st.metric("تنزل کل ریسک‌ها", f"{penalty:.1f}%",
                          help="مجموع اثر ریسک‌های وارد‌شده روی بازده")
            with cb:
                gap = m["واگرایی از هدف"]
                if gap is not None:
                    gap_str = f"{gap*100:+.2f}%"
                    delta_color = "normal" if gap >= 0 else "inverse"
                    st.metric("واگرایی از هدف بازده", gap_str,
                              delta="بالاتر از هدف" if gap >= 0 else "پایین‌تر از هدف",
                              delta_color=delta_color)
                else:
                    st.metric("واگرایی از هدف بازده", "—")
            with cc:
                eff_rf = (rf + m["تنزل ریسک (%)"]/100) * 100
                st.metric("نرخ بدون‌ریسک مؤثر", f"{eff_rf:.2f}%",
                          help="نرخ بدون‌ریسک پایه + تنزل ریسک")
            with cd:
                st.metric("تعداد نمادها", str(len(asset_names)))

            # ── نمودار رادار ریسک‌ها ──
            if saved.get("risk_geo", 0) > 0 or saved.get("risk_mon", 0) > 0 or saved.get("risk_sys", 0) > 0:
                st.markdown('<div class="bp-section"><span class="bp-section-text">Risk Radar</span></div>', unsafe_allow_html=True)

                categories = ["ریسک ژئوپولیتیک", "ریسک سیاست پولی", "ریسک سیستماتیک"]
                values = [
                    saved.get("risk_geo", 0),
                    saved.get("risk_mon", 0),
                    saved.get("risk_sys", 0),
                ]
                # بستن راداره
                values_closed = values + [values[0]]
                categories_closed = categories + [categories[0]]

                bg = "#161616" if is_dark else "#e8e8e4"
                txt_c = "#c0c0c0" if is_dark else "#222220"
                grid_c = "rgba(255,255,255,0.08)" if is_dark else "rgba(60,60,55,0.1)"

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill='toself',
                    fillcolor="rgba(204,85,85,0.15)" if is_dark else "rgba(138,32,32,0.1)",
                    line=dict(color="#cc5555" if is_dark else "#8a2020", width=2),
                    marker=dict(size=6, color="#cc5555" if is_dark else "#8a2020"),
                    name="ریسک‌ها"
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor=bg,
                        angularaxis=dict(
                            tickfont=dict(size=10, color=txt_c, family="JetBrains Mono"),
                            linecolor=grid_c,
                            gridcolor=grid_c,
                        ),
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            tickfont=dict(size=8, color=txt_c),
                            gridcolor=grid_c,
                            linecolor=grid_c,
                            tickvals=[25, 50, 75, 100],
                            ticktext=["25%", "50%", "75%", "100%"],
                        )
                    ),
                    paper_bgcolor=bg,
                    plot_bgcolor=bg,
                    font=dict(color=txt_c, family="JetBrains Mono"),
                    showlegend=False,
                    height=320,
                    margin=dict(l=60, r=60, t=40, b=40),
                    title=dict(text="RISK RADAR", font=dict(color=txt_c, size=11, family="JetBrains Mono"), x=0.5)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        # ── نمودار Drawdown ──
        st.markdown('<div class="bp-section"><span class="bp-section-text">Drawdown Chart</span></div>', unsafe_allow_html=True)
        port_ret_arr = returns[asset_names].values @ w
        cum = (1 + port_ret_arr).cumprod()
        roll_max = pd.Series(cum).cummax()
        dd = (pd.Series(cum) - roll_max) / roll_max

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=prices.index[-len(dd):], y=dd.values * 100,
            mode="lines", name="Drawdown",
            line=dict(color="#cc5555" if is_dark else "#8a2020", width=1.5),
            fill="tozeroy", fillcolor="rgba(204,85,85,0.12)" if is_dark else "rgba(138,32,32,0.08)"
        ))
        fig_dd.update_layout(**get_plot_layout("DRAWDOWN (%)", "DATE", "DRAWDOWN %", 340))
        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown('<div class="bp-section"><span class="bp-section-text">Portfolio Growth Curve</span></div>', unsafe_allow_html=True)
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=prices.index[-len(cum):], y=cum,
            mode="lines", name="Portfolio",
            line=dict(color="#5b9bd5", width=2)
        ))
        # خط هدف بازده
        if saved.get("expected_return", 0) > 0:
            n_days = len(cum)
            daily_target = (1 + saved["expected_return"] / 100) ** (1/252)
            target_curve = [daily_target ** i for i in range(n_days)]
            fig_cum.add_trace(go.Scatter(
                x=prices.index[-n_days:], y=target_curve,
                mode="lines", name=f"هدف {saved['expected_return']:.0f}%",
                line=dict(color="#c8a84b" if is_dark else "#8a6a1a", width=1.5, dash="dash")
            ))
        fig_cum.add_hline(y=1.0, line_dash="dash",
                          line_color="rgba(128,128,128,0.3)", line_width=1)
        fig_cum.update_layout(**get_plot_layout("PORTFOLIO GROWTH (BASE=1)", "DATE", "CUMULATIVE RETURN", 370))
        st.plotly_chart(fig_cum, use_container_width=True)

        st.markdown('<div class="bp-section"><span class="bp-section-text">Daily Return Distribution</span></div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=port_ret_arr * 100, nbinsx=60,
            marker_color="rgba(91,155,213,0.55)" if is_dark else "rgba(60,100,170,0.4)",
            marker_line=dict(color="rgba(91,155,213,0.8)", width=0.5),
            name="بازده روزانه"
        ))
        cvar_line = np.percentile(port_ret_arr, 5) * 100
        fig_hist.add_vline(
            x=cvar_line, line_dash="dash",
            line_color="#cc5555" if is_dark else "#8a2020", line_width=1.5,
            annotation_text=f"CVaR 95%: {cvar_line:.2f}%",
            annotation_font_color="#cc5555" if is_dark else "#8a2020",
            annotation_font_size=9
        )
        fig_hist.update_layout(**get_plot_layout("DAILY RETURN DISTRIBUTION", "RETURN %", "FREQUENCY", 350))
        st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — نمودار قیمت
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="bp-section"><span class="bp-section-text">Price Chart</span></div>', unsafe_allow_html=True)

    view_mode = st.radio("نمایش", ["نرمال‌شده (base=100)", "قیمت خام"], horizontal=True)

    fig_price = go.Figure()
    for i, col in enumerate(asset_names):
        s = prices[col]
        y = (s / s.iloc[0] * 100).values if view_mode.startswith("نرمال") else s.values
        fig_price.add_trace(go.Scatter(
            x=prices.index, y=y,
            mode="lines", name=col,
            line=dict(color=COLORS[i % len(COLORS)], width=1.5)
        ))
    yt = "قیمت نرمال‌شده (base=100)" if view_mode.startswith("نرمال") else "قیمت ($)"
    fig_price.update_layout(**get_plot_layout("PRICE CHART", "DATE", yt, 500))
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown('<div class="bp-section"><span class="bp-section-text">Correlation Matrix</span></div>', unsafe_allow_html=True)
    if len(asset_names) >= 2:
        corr = returns[asset_names].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[
                [0.0, "#cc5555"], [0.5, "#161616" if is_dark else "#e0e0db"], [1.0, "#5b9bd5"]
            ],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9),
            showscale=True,
        ))
        fig_corr.update_layout(**get_plot_layout("CORRELATION MATRIX", h=max(340, len(asset_names)*34)))
        fig_corr.update_layout(
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — مقایسه سبک‌ها
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="bp-section"><span class="bp-section-text">Strategy Comparison</span></div>', unsafe_allow_html=True)

    if len(asset_names) < 2:
        st.info("حداقل ۲ نماد نیاز است.")
    else:
        run_compare = st.button("▶ اجرای مقایسه همه سبک‌ها", use_container_width=True)
        if run_compare or st.session_state.get("compare_done"):
            if run_compare:
                with st.spinner("در حال محاسبه تمام سبک‌ها..."):
                    compare_results = {}
                    for lbl, sty in STYLES.items():
                        try:
                            ww, _ = calc_weights(
                                sty, returns[asset_names], rf, asset_names,
                                expected_return=expected_return,
                                risk_geo=risk_geo, risk_mon=risk_mon, risk_sys=risk_sys
                            )
                            mm = portfolio_metrics(
                                ww, returns[asset_names], rf,
                                expected_return=expected_return,
                                risk_geo=risk_geo, risk_mon=risk_mon, risk_sys=risk_sys
                            )
                            compare_results[lbl] = {**mm, "_weights": ww}
                        except Exception:
                            pass
                st.session_state["compare_results"] = compare_results
                st.session_state["compare_done"] = True
                st.session_state["compare_asset_names"] = asset_names[:]

            compare_results = st.session_state.get("compare_results", {})
            stored_assets = st.session_state.get("compare_asset_names", [])
            if stored_assets != asset_names:
                compare_results = {}
                st.session_state["compare_done"] = False

            if compare_results:
                rows = []
                for lbl, mm in compare_results.items():
                    rec = mm["ریکاوری تایم (روز)"]
                    rec_m = int(rec / 21)
                    gap = mm["واگرایی از هدف"]
                    rows.append({
                        "سبک": lbl,
                        "بازده خام (%)": round(mm["بازده سالانه"] * 100, 2),
                        "بازده تعدیل‌شده (%)": round(mm["بازده تعدیل‌شده ریسک"] * 100, 2),
                        "نوسان (%)": round(mm["نوسان سالانه"] * 100, 2),
                        "شارپ": round(mm["نسبت شارپ"], 3),
                        "MDD (%)": round(mm["حداکثر افت (Max Drawdown)"] * 100, 2),
                        "CVaR 95% (%)": round(mm["CVaR 95%"] * 100, 2),
                        "واگرایی هدف (%)": f"{gap*100:+.1f}%" if gap is not None else "—",
                        "تنزل ریسک (%)": round(mm["تنزل ریسک (%)"], 1),
                        "ریکاوری": f"{rec_m} ماه" if rec_m else f"{rec} روز",
                        "کالمار": round(mm["نسبت کالمار"], 3),
                    })
                df_cmp = pd.DataFrame(rows)
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)

                # ── نمودار مقایسه‌ای با بازده تعدیل‌شده ──
                fig_cmp = go.Figure()
                metrics_to_plot = ["بازده تعدیل‌شده (%)", "نوسان (%)", "شارپ", "MDD (%)"]
                data_keys = ["بازده تعدیل‌شده (%)", "نوسان (%)", "شارپ", "MDD (%)"]
                for i, (metric, key) in enumerate(zip(metrics_to_plot, data_keys)):
                    vals = [r[key] for r in rows]
                    fig_cmp.add_trace(go.Bar(
                        name=metric,
                        x=[r["سبک"] for r in rows],
                        y=vals,
                        marker_color=COLORS[i],
                        marker_line=dict(color="rgba(0,0,0,0.15)", width=0.5),
                    ))
                fig_cmp.update_layout(**get_plot_layout("STRATEGY COMPARISON (RISK-ADJUSTED)", "", "VALUE", 400))
                fig_cmp.update_layout(barmode="group")
                st.plotly_chart(fig_cmp, use_container_width=True)

                # ── نمودار رشد سبک‌ها ──
                st.markdown('<div class="bp-section"><span class="bp-section-text">Growth Curves — All Strategies</span></div>', unsafe_allow_html=True)
                fig_growth = go.Figure()
                ret_vals = returns[asset_names].values
                for i, (lbl, mm) in enumerate(compare_results.items()):
                    ww = np.array(mm["_weights"], dtype=float)
                    if ww.shape[0] != ret_vals.shape[1]:
                        continue
                    pr = ret_vals @ ww
                    cum_c = (1 + pr).cumprod()
                    fig_growth.add_trace(go.Scatter(
                        x=prices.index[-len(cum_c):], y=cum_c,
                        mode="lines", name=lbl,
                        line=dict(color=COLORS[i % len(COLORS)], width=1.8)
                    ))

                # خط هدف بازده در نمودار رشد
                if saved.get("expected_return", 0) > 0:
                    n_d = len(ret_vals)
                    dly_t = (1 + saved["expected_return"] / 100) ** (1/252)
                    tgt_c = [dly_t ** i for i in range(n_d)]
                    fig_growth.add_trace(go.Scatter(
                        x=prices.index[-n_d:], y=tgt_c,
                        mode="lines",
                        name=f"هدف {saved['expected_return']:.0f}%",
                        line=dict(color="#c8a84b" if is_dark else "#8a6a1a", width=2, dash="dot")
                    ))

                fig_growth.update_layout(**get_plot_layout(
                    "GROWTH CURVES — ALL STRATEGIES", "DATE", "CUMULATIVE RETURN", 460
                ))
                st.plotly_chart(fig_growth, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
risk_summary = ""
saved_f = st.session_state.get("saved_risks", {})
if saved_f:
    parts = []
    if saved_f.get("expected_return", 0) > 0: parts.append(f"Target {saved_f['expected_return']:.0f}%")
    if saved_f.get("risk_geo", 0) > 0:         parts.append(f"Geo {saved_f['risk_geo']}%")
    if saved_f.get("risk_mon", 0) > 0:         parts.append(f"Mon {saved_f['risk_mon']}%")
    if saved_f.get("risk_sys", 0) > 0:         parts.append(f"Sys {saved_f['risk_sys']}%")
    if parts:
        risk_summary = " · " + " · ".join(parts)

st.markdown(f"""
<div style="text-align:center; color:var(--muted); font-size:0.6rem;
    letter-spacing:0.15em; text-transform:uppercase; padding:0.4rem;
    font-family:'JetBrains Mono',monospace;">
    Portfolio360 Blueprint · Data via Yahoo Finance · {'Dark' if is_dark else 'Light'} Mode{risk_summary}
</div>
""", unsafe_allow_html=True)
