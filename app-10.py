"""
Portfolio360 — Blueprint Edition
سبک‌های پرتفوی + نمادهای جهانی
با سایدبار بازطراحی‌شده + دو تم روشن/تاریک
+ بازده مورد انتظار و ریسک‌های سه‌گانه
+ استراتژی کاورد کال
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

[data-testid="stSidebar"] > div:first-child {
    background: var(--sb-bg) !important;
    border-right: 1px solid var(--sb-border) !important;
    padding: 0 !important;
}

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

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2rem !important;
}

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

[data-testid="stDataFrame"] {
    border: 1px solid var(--border2) !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}

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

.theme-toggle-wrap .stButton > button {
    border-radius: 20px !important;
    padding: 0.3rem 0.8rem !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.15em !important;
    width: auto !important;
}

[data-testid="stRadio"] label {
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em !important;
    text-transform: none !important;
    color: var(--silver) !important;
}

[data-testid="stSlider"] {
    padding: 0 0.3rem !important;
}

/* ══ COVERED CALL CARD ══ */
.cc-verdict-card {
    margin: 0.8rem 0;
    padding: 1rem 1.1rem;
    border-radius: 6px;
    border-left: 4px solid;
    background: var(--card);
    border-color: var(--border);
}
.cc-verdict-card.positive {
    border-left-color: var(--green) !important;
    background: rgba(26,102,64,0.07) !important;
}
.cc-verdict-card.negative {
    border-left-color: var(--red) !important;
    background: rgba(138,32,32,0.07) !important;
}
.cc-verdict-card.neutral {
    border-left-color: var(--gold) !important;
    background: rgba(138,106,26,0.07) !important;
}
.cc-verdict-title {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.4rem !important;
}
.cc-verdict-body {
    font-size: 0.82rem !important;
    color: var(--silver) !important;
    line-height: 1.6 !important;
}
.cc-metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.28rem 0;
    border-bottom: 1px solid var(--border2);
}
.cc-metric-row:last-child { border-bottom: none; }
.cc-metric-label {
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
}
.cc-metric-val {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
}
.cc-help-note {
    font-size: 0.65rem !important;
    color: var(--muted) !important;
    line-height: 1.55 !important;
    padding: 0.3rem 0 0 0 !important;
    border-top: 1px solid var(--border2);
    margin-top: 0.3rem !important;
}
"""

theme_vars = DARK_CSS if is_dark else LIGHT_CSS
st.markdown(f"<style>{theme_vars}{SHARED_CSS}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# نمادها
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS = {
    "💰 ارزهای دیجیتال": {
        "BTC-USD": "Bitcoin","ETH-USD": "Ethereum","BNB-USD": "BNB","SOL-USD": "Solana",
        "XRP-USD": "XRP","ADA-USD": "Cardano","AVAX-USD": "Avalanche","DOGE-USD": "Dogecoin",
        "DOT-USD": "Polkadot","MATIC-USD": "Polygon","LINK-USD": "Chainlink","LTC-USD": "Litecoin",
        "UNI7083-USD": "Uniswap","ATOM-USD": "Cosmos","XLM-USD": "Stellar",
    },
    "📈 سهام آمریکا": {
        "AAPL": "Apple","MSFT": "Microsoft","GOOGL": "Alphabet","AMZN": "Amazon",
        "NVDA": "NVIDIA","META": "Meta","TSLA": "Tesla","BERKB": "Berkshire B",
        "JPM": "JPMorgan","V": "Visa","JNJ": "J&J","WMT": "Walmart","XOM": "Exxon",
        "BAC": "Bank of America","MA": "Mastercard","PG": "P&G","HD": "Home Depot",
        "CVX": "Chevron","ABBV": "AbbVie","KO": "Coca-Cola","PEP": "PepsiCo",
        "LLY": "Eli Lilly","MRK": "Merck","CRM": "Salesforce","AMD": "AMD",
        "INTC": "Intel","NFLX": "Netflix","DIS": "Disney","PYPL": "PayPal","UBER": "Uber",
    },
    "🌍 سهام جهانی": {
        "TSM": "TSMC (Taiwan)","ASML": "ASML (Netherlands)","SAP": "SAP (Germany)",
        "TM": "Toyota (Japan)","NVO": "Novo Nordisk (Denmark)","HSBC": "HSBC (UK)",
        "BP": "BP (UK)","SHEL": "Shell (UK)","UL": "Unilever (UK)","RIO": "Rio Tinto (UK)",
        "BABA": "Alibaba (China)","JD": "JD.com (China)","SONY": "Sony (Japan)",
        "HMC": "Honda (Japan)","BCS": "Barclays (UK)",
    },
    "🏦 ETF و شاخص": {
        "SPY": "S&P 500 ETF","QQQ": "Nasdaq 100 ETF","DIA": "Dow Jones ETF",
        "IWM": "Russell 2000 ETF","VTI": "Total Market ETF","EEM": "Emerging Markets ETF",
        "VEA": "Developed Markets ETF","AGG": "Bond Aggregate ETF","TLT": "20Y Treasury ETF",
        "HYG": "High Yield Bond ETF","GLD": "Gold ETF","SLV": "Silver ETF","USO": "Oil ETF",
        "XLE": "Energy ETF","XLF": "Financials ETF","XLK": "Technology ETF",
        "XLV": "Healthcare ETF","ARKK": "ARK Innovation ETF","VNQ": "Real Estate ETF",
        "PDBC": "Commodity ETF",
    },
    "🥇 کامودیتی و فارکس": {
        "GC=F": "Gold Futures","SI=F": "Silver Futures","CL=F": "Crude Oil (WTI)",
        "BZ=F": "Brent Oil","NG=F": "Natural Gas","HG=F": "Copper","PL=F": "Platinum",
        "ZW=F": "Wheat","ZC=F": "Corn","ZS=F": "Soybeans","EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD","USDJPY=X": "USD/JPY","USDCHF=X": "USD/CHF",
        "AUDUSD=X": "AUD/USD","USDCAD=X": "USD/CAD","USDIRR=X": "USD/IRR",
    },
}

PERIODS = {"۶ ماه": "6mo","۱ سال": "1y","۲ سال": "2y","۵ سال": "5y","۱۰ سال": "10y","حداکثر": "max"}

STYLES = {
    "بیشترین شارپ (Markowitz)": "max_sharpe",
    "کمترین واریانس (Min Variance)": "min_var",
    "مونت‌کارلو (CVaR)": "monte_carlo",
    "وزن برابر (Equal Weight)": "equal_weight",
    "ریسک پاریتی (Risk Parity)": "risk_parity",
    "۹۰/۱۰ طالب (Taleb Barbell)": "taleb_barbell",
}

TALEB_SAFE  = {"GC=F","GLD","TLT","AGG","EURUSD=X","GBPUSD=X","USDCHF=X"}
TALEB_RISKY = {"BTC-USD","ETH-USD","SOL-USD","AVAX-USD","NVDA","TSLA","ARKK"}

# ─────────────────────────────────────────────────────────────────────────────
# توابع اصلی
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(tickers: tuple, period: str):
    data = {}; failed = []
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


def calc_risk_penalty(risk_geo, risk_mon, risk_sys):
    weighted_risk = (risk_geo * 0.40 + risk_mon * 0.35 + risk_sys * 0.25) / 100.0
    return min(weighted_risk * 0.5, 0.50)


def calc_weights(style, returns, rf, selected, expected_return=0.0,
                 risk_geo=0.0, risk_mon=0.0, risk_sys=0.0):
    n = len(selected)
    eq = np.ones(n) / n
    mean_r = returns.mean() * 252
    cov = returns.cov() * 252
    bnds = [(0.01, 0.6)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    risk_penalty = calc_risk_penalty(risk_geo, risk_mon, risk_sys)
    effective_rf = rf + risk_penalty
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
            active_cons.append({"type": "ineq","fun": lambda w: (w @ mean_r.values) - expected_ret_ann})
        res = minimize(neg_sharpe, eq, method="SLSQP", bounds=bnds, constraints=active_cons)
        if not res.success:
            res = minimize(neg_sharpe, eq, method="SLSQP", bounds=bnds, constraints=cons)
        return (res.x / res.x.sum() if res.success else eq), cov
    if style == "min_var":
        def port_var(w): return w @ cov.values @ w
        active_cons = list(cons)
        if use_expected_ret:
            active_cons.append({"type": "ineq","fun": lambda w: (w @ mean_r.values) - expected_ret_ann})
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
        res = minimize(rp_obj, eq, method="SLSQP", bounds=[(0.01,1)]*n, constraints=cons)
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
                return -np.percentile(r, 5)
            res = minimize(neg_sharpe_s, eq, method="SLSQP", bounds=bnds, constraints=cons)
            if res.success and res.fun < best_cvar:
                best_cvar = res.fun
                best_w = res.x / res.x.sum()
        return best_w, cov
    if style == "taleb_barbell":
        w = np.ones(n) * (0.10 / n)
        safe_idx  = [i for i,s in enumerate(selected) if s in TALEB_SAFE]
        risky_idx = [i for i,s in enumerate(selected) if s in TALEB_RISKY]
        other_idx = [i for i in range(n) if i not in safe_idx and i not in risky_idx]
        safe_count = len(safe_idx) or 1; risky_count = len(risky_idx) or 1; other_count = len(other_idx) or 1
        if safe_idx and risky_idx:
            for i in safe_idx:  w[i] = 0.90 / safe_count
            for i in risky_idx: w[i] = 0.10 / risky_count
            for i in other_idx: w[i] = 0.0
        elif safe_idx:
            for i in safe_idx:  w[i] = 0.90 / safe_count
            for i in other_idx: w[i] = 0.10 / other_count
        else:
            w = eq
        w = np.clip(w, 0, 1); w /= w.sum()
        return w, cov
    return eq, cov


def portfolio_metrics(weights, returns, rf, expected_return=0.0,
                      risk_geo=0.0, risk_mon=0.0, risk_sys=0.0):
    port_ret = returns.values @ weights
    ann_ret  = (1 + port_ret).prod() ** (252 / len(port_ret)) - 1
    ann_vol  = port_ret.std() * np.sqrt(252)
    risk_penalty = calc_risk_penalty(risk_geo, risk_mon, risk_sys)
    risk_adjusted_ret = ann_ret * (1 - risk_penalty)
    sharpe = (risk_adjusted_ret - rf) / (ann_vol + 1e-9)
    cum = pd.Series((1 + port_ret).cumprod())
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = float(dd.min())
    max_recovery = 0; streak = 0
    for v in (dd < -0.01):
        streak = streak + 1 if v else 0
        max_recovery = max(max_recovery, streak)
    cvar = float(-np.percentile(port_ret, 5))
    calmar = risk_adjusted_ret / (abs(max_dd) + 1e-9)
    expected_ret_ann = expected_return / 100.0
    ret_gap = risk_adjusted_ret - expected_ret_ann if expected_ret_ann > 0 else None
    return {
        "بازده سالانه": float(ann_ret),
        "بازده تعدیل‌شده ریسک": float(risk_adjusted_ret),
        "نوسان سالانه": float(ann_vol),
        "نسبت شارپ": float(sharpe),
        "حداکثر افت (Max Drawdown)": max_dd,
        "CVaR 95%": cvar,
        "نسبت کالمار": float(calmar),
        "ریکاوری تایم (روز)": int(max_recovery),
        "واگرایی از هدف": ret_gap,
        "تنزل ریسک (%)": float(risk_penalty * 100),
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
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=txt, family="JetBrains Mono, Courier New", size=9),
        xaxis=dict(title=dict(text=xt, font=af), gridcolor=grid, linecolor=line,
                   tickfont=dict(color=tick, size=9), zeroline=False),
        yaxis=dict(title=dict(text=yt, font=af), gridcolor=grid, linecolor=line,
                   tickfont=dict(color=tick, size=9), zeroline=True,
                   zerolinecolor=grid, zerolinewidth=1),
        legend=dict(bgcolor="rgba(0,0,0,0.3)" if is_dark else "rgba(240,240,237,0.85)",
                    bordercolor=line, borderwidth=1, font=dict(color=txt, size=9)),
        margin=dict(l=55, r=20, t=50, b=45),
        height=h,
    )

COLORS = [
    "#5b9bd5","#e8a838","#3db87a","#e05c5c","#9b72c8",
    "#48b8c0","#d47f3a","#c45b8e","#7eb35a","#5a8fc4",
    "#d4a855","#4db88c","#c46060","#8062b8","#40a8a8",
]

# ─────────────────────────────────────────────────────────────────────────────
# تابع محاسبه کاورد کال (Black-Scholes)
# ─────────────────────────────────────────────────────────────────────────────
def black_scholes_call(S, K, T, r, sigma):
    """قیمت‌گذاری اختیار خرید با مدل بلک-شولز"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0), 0, 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    from scipy.stats import norm
    N = norm.cdf
    call_price = S * N(d1) - K * np.exp(-r * T) * N(d2)
    delta = N(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T) + 1e-9)
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N(d2)) / 365
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    return call_price, delta, gamma, theta, vega


def analyze_covered_call(S, K, T_days, r, sigma, premium, contracts,
                          expected_ret_pct, risk_geo, risk_mon, risk_sys):
    """
    تحلیل کاورد کال:
    - S: قیمت فعلی دارایی پایه
    - K: قیمت اعمال (Strike)
    - T_days: روز تا انقضا
    - r: نرخ بدون ریسک
    - sigma: نوسان ضمنی (IV)
    - premium: پرمیوم دریافتی واقعی (اگه صفر بزنی از BS محاسبه می‌شه)
    - contracts: تعداد قرارداد (هر قرارداد ۱۰۰ سهم)
    - expected_ret_pct: بازده مورد انتظار سالانه از پرتفو
    - risk_*: ریسک‌ها
    """
    T = T_days / 365.0
    bs_price, delta, gamma, theta, vega = black_scholes_call(S, K, T, r, sigma)

    if premium <= 0:
        premium = bs_price

    shares = contracts * 100
    total_premium = premium * shares
    cost_basis = S * shares

    # سناریوها
    # ۱) قیمت زیر استرایک می‌مونه → اختیار منقضی می‌شه، پرمیوم سود خالص
    profit_below = total_premium
    ret_below = profit_below / cost_basis

    # ۲) قیمت بالای استرایک می‌ره → سهام call می‌شه، سود محدود به K - S + premium
    capped_gain = (K - S) * shares + total_premium
    ret_capped = capped_gain / cost_basis

    # ۳) بدترین سناریو: قیمت دارایی به صفر می‌رسه (نظری)
    max_loss = cost_basis - total_premium
    ret_max_loss = -max_loss / cost_basis

    # بازده سالانه‌شده پرمیوم
    ann_premium_yield = (premium / S) * (365 / T_days)

    # نقطه سر به سر
    breakeven = S - premium

    # اثر ریسک‌ها روی تحلیل
    risk_penalty = calc_risk_penalty(risk_geo, risk_mon, risk_sys)
    # بازده موردانتظار سالانه تعدیل‌شده
    expected_ann_adj = (expected_ret_pct / 100) * (1 - risk_penalty)
    # بازده کاورد کال برای دوره مورد نظر
    cc_period_ret = (premium / S)
    cc_ann_ret = cc_period_ret * (365 / T_days)
    cc_adj_ret = cc_ann_ret * (1 - risk_penalty * 0.5)  # ریسک کمتر تاثیر می‌ذاره چون income strategy

    # مقایسه با بازده موردانتظار
    # اگه بازده تعدیل‌شده CC بالای بازده انتظاری باشه → به‌صرفه
    worthwhile_score = cc_adj_ret - expected_ann_adj

    # Moneyness
    moneyness = (K - S) / S * 100  # مثبت = OTM، منفی = ITM

    # اگه IV بالا باشه (بالای ۳۰٪) و OTM باشه → شرایط ایده‌آل
    iv_ok = sigma >= 0.20
    otm_ok = K >= S * 1.02  # حداقل ۲٪ OTM
    time_ok = T_days >= 21  # حداقل ۳ هفته

    return {
        "bs_price": bs_price,
        "premium": premium,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "total_premium": total_premium,
        "cost_basis": cost_basis,
        "profit_below": profit_below,
        "ret_below": ret_below,
        "capped_gain": capped_gain,
        "ret_capped": ret_capped,
        "max_loss": max_loss,
        "ret_max_loss": ret_max_loss,
        "ann_premium_yield": ann_premium_yield,
        "breakeven": breakeven,
        "moneyness": moneyness,
        "cc_period_ret": cc_period_ret,
        "cc_ann_ret": cc_ann_ret,
        "cc_adj_ret": cc_adj_ret,
        "expected_ann_adj": expected_ann_adj,
        "worthwhile_score": worthwhile_score,
        "risk_penalty": risk_penalty,
        "iv_ok": iv_ok,
        "otm_ok": otm_ok,
        "time_ok": time_ok,
        "T": T,
        "shares": shares,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ① توابع اختیار معامله پیشرفته — Protective Put / Iron Condor / Rolling CC
# ─────────────────────────────────────────────────────────────────────────────
from scipy.stats import norm as _norm

def bs_price(S, K, T, r, sigma, opt="call"):
    """Black-Scholes برای call و put با تمام Greeks"""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S-K,0) if opt=="call" else max(K-S,0)
        return intrinsic, 0, 0, 0, 0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == "call":
        price = S*_norm.cdf(d1) - K*np.exp(-r*T)*_norm.cdf(d2)
        delta = _norm.cdf(d1)
    else:
        price = K*np.exp(-r*T)*_norm.cdf(-d2) - S*_norm.cdf(-d1)
        delta = _norm.cdf(d1) - 1
    gamma = _norm.pdf(d1) / (S*sigma*np.sqrt(T) + 1e-9)
    theta = (-(S*_norm.pdf(d1)*sigma)/(2*np.sqrt(T+1e-9))
             - r*K*np.exp(-r*T)*(_norm.cdf(d2) if opt=="call" else _norm.cdf(-d2))) / 365
    vega  = S*_norm.pdf(d1)*np.sqrt(T)/100
    return price, delta, gamma, theta, vega


def analyze_protective_put(S, K_put, T_days, r, sigma_put, premium_put, shares_owned,
                            expected_ret_pct, risk_geo, risk_mon, risk_sys):
    T = T_days/365.0
    pp, delta, gamma, theta, vega = bs_price(S, K_put, T, r, sigma_put, "put")
    if premium_put <= 0:
        premium_put = pp
    total_cost   = premium_put * shares_owned
    cost_pct     = premium_put / S
    max_loss_ins = (S - K_put + premium_put) * shares_owned
    breakeven    = S + premium_put
    risk_penalty = calc_risk_penalty(risk_geo, risk_mon, risk_sys)
    expected_adj = (expected_ret_pct/100)*(1-risk_penalty)
    cost_ann     = cost_pct*(365/T_days)
    net_ret      = expected_adj - cost_ann
    return dict(
        bs_price=pp, premium=premium_put, delta=delta, gamma=gamma, theta=theta, vega=vega,
        total_cost=total_cost, cost_pct=cost_pct*100, cost_ann=cost_ann*100,
        max_loss_insured=max_loss_ins, breakeven=breakeven,
        net_ret_after_insurance=net_ret*100,
        expected_adj=expected_adj*100, risk_penalty=risk_penalty*100,
        worthwhile=(net_ret > 0)
    )


def analyze_iron_condor(S, K_put_buy, K_put_sell, K_call_sell, K_call_buy,
                         T_days, r, sigma, contracts,
                         expected_ret_pct, risk_geo, risk_mon, risk_sys):
    T = T_days/365.0
    pb_p, *_ = bs_price(S, K_put_buy,   T, r, sigma, "put")
    ps_p, *_ = bs_price(S, K_put_sell,  T, r, sigma, "put")
    cs_p, *_ = bs_price(S, K_call_sell, T, r, sigma, "call")
    cb_p, *_ = bs_price(S, K_call_buy,  T, r, sigma, "call")
    net_credit    = ps_p - pb_p + cs_p - cb_p
    total_credit  = net_credit * contracts * 100
    spread_put    = K_put_sell  - K_put_buy
    spread_call   = K_call_buy  - K_call_sell
    max_loss      = (max(spread_put, spread_call) - net_credit) * contracts * 100
    be_lower      = K_put_sell  - net_credit
    be_upper      = K_call_sell + net_credit
    profit_zone_pct = (be_upper - be_lower)/S*100
    ret_on_risk   = net_credit / (max(spread_put, spread_call) - net_credit + 1e-9)
    ann_ret       = ret_on_risk*(365/T_days)
    risk_penalty  = calc_risk_penalty(risk_geo, risk_mon, risk_sys)
    adj_ann_ret   = ann_ret*(1-risk_penalty*0.3)
    expected_adj  = (expected_ret_pct/100)*(1-risk_penalty)
    worthwhile_score = adj_ann_ret - expected_adj
    return dict(
        net_credit=net_credit, total_credit=total_credit, max_loss=max_loss,
        be_lower=be_lower, be_upper=be_upper,
        profit_zone_pct=profit_zone_pct,
        ret_on_risk=ret_on_risk*100, ann_ret=ann_ret*100, adj_ann_ret=adj_ann_ret*100,
        expected_adj=expected_adj*100, worthwhile_score=worthwhile_score*100,
        risk_penalty=risk_penalty*100,
        pb_price=pb_p, ps_price=ps_p, cs_price=cs_p, cb_price=cb_p,
    )


def simulate_rolling_cc(S_series, K_offset_pct, dte, r, sigma, contracts):
    """شبیه‌سازی Rolling CC — فروش ماهانه اختیار و تمدید"""
    step = max(dte, 5)
    n = len(S_series)
    results = []
    for i in range(0, n - step, step):
        S = float(S_series.iloc[i])
        K = S*(1 + K_offset_pct/100)
        T = dte/365.0
        p, delta, *_ = bs_price(S, K, T, r, sigma, "call")
        premium_earned = p*contracts*100
        S_exp = float(S_series.iloc[min(i+step, n-1)])
        exercised = S_exp >= K
        stock_pnl  = (S_exp - S)*contracts*100
        option_pnl = (-(S_exp-K)*contracts*100 + premium_earned) if exercised else premium_earned
        results.append(dict(
            date=S_series.index[i], S=S, K=round(K,2), premium=round(p,3),
            premium_earned=round(premium_earned,2), S_exp=round(S_exp,2),
            exercised=exercised, stock_pnl=round(stock_pnl,2),
            option_pnl=round(option_pnl,2), total_pnl=round(stock_pnl+option_pnl,2),
            delta=round(delta,3),
        ))
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# ② Black-Litterman + Factor Exposure
# ─────────────────────────────────────────────────────────────────────────────
def black_litterman(weights_mkt, cov, mean_returns, views_dict, tau=0.05):
    """
    Black-Litterman:
    views_dict = {"AAPL": 0.20}  → پیش‌بینی بازده سالانه برای هر نماد
    """
    n = len(weights_mkt)
    Pi = tau * cov.values @ weights_mkt
    if not views_dict:
        return weights_mkt, mean_returns.values * 252

    assets = list(mean_returns.index)
    view_assets = [a for a in views_dict if a in assets]
    if not view_assets:
        return weights_mkt, mean_returns.values * 252

    k = len(view_assets)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    for i, a in enumerate(view_assets):
        j = assets.index(a)
        P[i, j] = 1.0
        Q[i]    = views_dict[a]

    Omega = np.diag(np.diag(P @ (tau*cov.values) @ P.T))
    try:
        M_inv = np.linalg.inv(
            np.linalg.inv(tau*cov.values) + P.T @ np.linalg.inv(Omega) @ P
        )
        mu_bl = M_inv @ (
            np.linalg.inv(tau*cov.values) @ Pi + P.T @ np.linalg.inv(Omega) @ Q
        )
        w_bl = np.linalg.inv(cov.values) @ mu_bl
        w_bl = np.clip(w_bl, 0, None)
        w_bl = w_bl / w_bl.sum() if w_bl.sum() > 0 else weights_mkt
    except Exception:
        w_bl, mu_bl = weights_mkt, Pi
    return w_bl, mu_bl


def compute_factor_exposure(returns_df):
    """Factor Analysis: Momentum، Volatility، Beta، Sharpe"""
    market = returns_df.mean(axis=1)
    rows = []
    for col in returns_df.columns:
        r = returns_df[col]
        momentum  = float((1+r.tail(126)).prod() - 1)*100
        vol       = float(r.std()*np.sqrt(252))*100
        cov_m     = np.cov(r.values, market.values)[0,1]
        beta      = cov_m/(np.var(market.values)+1e-9)
        sharpe    = float(r.mean()/(r.std()+1e-9)*np.sqrt(252))
        rows.append(dict(نماد=col, مومنتوم_6ماه=round(momentum,2),
                         نوسان_سالانه=round(vol,2), بتا=round(beta,3), شارپ=round(sharpe,3)))
    return pd.DataFrame(rows).set_index("نماد")


# ─────────────────────────────────────────────────────────────────────────────
# ③ Stress Test + Monte Carlo آینده
# ─────────────────────────────────────────────────────────────────────────────
CRISIS_PERIODS = {
    "بحران مالی ۲۰۰۸":  ("2008-09-01","2009-03-31"),
    "Flash Crash 2010":  ("2010-04-23","2010-07-02"),
    "بحران اروپا ۲۰۱۱": ("2011-07-01","2011-10-03"),
    "افت چین ۲۰۱۵":     ("2015-06-12","2015-09-29"),
    "کرونا ۲۰۲۰":       ("2020-02-19","2020-03-23"),
    "افت تورم ۲۰۲۲":    ("2022-01-01","2022-10-13"),
}

def run_stress_tests(prices_df, weights_series):
    results = []
    for name, (start, end) in CRISIS_PERIODS.items():
        try:
            mask = (prices_df.index >= start) & (prices_df.index <= end)
            sub  = prices_df[mask]
            if len(sub) < 5:
                continue
            sub_ret = sub.pct_change().dropna()
            cols = [c for c in weights_series.index if c in sub_ret.columns]
            if not cols:
                continue
            w_sub = weights_series[cols].values
            w_sub = w_sub/w_sub.sum()
            pr    = sub_ret[cols].values @ w_sub
            cum   = float((1+pr).prod()-1)
            vol   = float(pr.std()*np.sqrt(252))
            dd    = float(((pd.Series((1+pr).cumprod())/pd.Series((1+pr).cumprod()).cummax())-1).min())
            results.append(dict(بحران=name, بازده_کل=round(cum*100,2),
                                نوسان_سالانه=round(vol*100,2),
                                حداکثر_افت=round(dd*100,2), روز=len(pr)))
        except Exception:
            pass
    return pd.DataFrame(results)


def monte_carlo_future(weights, returns_df, n_sims=400, horizon_years=3):
    port_ret = returns_df.values @ weights
    mu    = port_ret.mean()
    sigma = port_ret.std()
    n_days = int(horizon_years*252)
    paths = np.zeros((n_sims, n_days))
    for i in range(n_sims):
        daily = np.random.normal(mu, sigma, n_days)
        paths[i] = (1+daily).cumprod()
    final = paths[:,-1]
    return dict(
        pct5=np.percentile(paths,5,axis=0),
        pct25=np.percentile(paths,25,axis=0),
        pct50=np.percentile(paths,50,axis=0),
        pct75=np.percentile(paths,75,axis=0),
        pct95=np.percentile(paths,95,axis=0),
        final=final,
        prob_profit=float((final>1.0).mean()*100),
        prob_2x=float((final>2.0).mean()*100),
        median=float(np.median(final)),
        worst5=float(np.percentile(final,5)),
        best5=float(np.percentile(final,95)),
        n_days=n_days,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ④ Rebalancing + Correlation Regime Detection
# ─────────────────────────────────────────────────────────────────────────────
def calc_rebalancing(current_prices_series, target_weights, asset_names, total_capital, threshold=0.05):
    vals = {a: float(current_prices_series.get(a, 0)) for a in asset_names if a in current_prices_series}
    total = sum(vals.values())
    if total == 0:
        return pd.DataFrame()
    rows = []
    for i, a in enumerate(asset_names):
        tw   = float(target_weights[i])
        cv   = vals.get(a, 0)
        cw   = cv/total
        drift = cw - tw
        trade = (tw - cw)*total_capital
        rows.append(dict(نماد=a, وزن_هدف=round(tw*100,2), وزن_فعلی=round(cw*100,2),
                         انحراف=round(drift*100,2), معامله_دلار=round(trade,2),
                         وضعیت=("⚠ ری‌بالانس" if abs(drift)>threshold else "✓ در محدوده")))
    return pd.DataFrame(rows)


def detect_correlation_regime(returns_df, window_short=30, window_long=126):
    if returns_df.shape[1] < 2:
        return pd.DataFrame(columns=["date","corr_short","corr_long","signal"]), "⚠ حداقل ۲ دارایی برای تشخیص رژیم لازم است"
    avg_s, avg_l, dates = [], [], []
    for i in range(window_long, len(returns_df)):
        cs = returns_df.iloc[i-window_short:i].corr().values.copy()
        cl = returns_df.iloc[i-window_long:i].corr().values.copy()
        np.fill_diagonal(cs, np.nan); np.fill_diagonal(cl, np.nan)
        avg_s.append(float(np.nanmean(cs)))
        avg_l.append(float(np.nanmean(cl)))
        dates.append(returns_df.index[i])
    df = pd.DataFrame(dict(date=dates, corr_short=avg_s, corr_long=avg_l))
    df["signal"] = df["corr_short"] > df["corr_long"] + 0.10
    regime = "🔴 بحران — همبستگی‌ها بالا رفته (هشدار تنوع‌بخشی)" if (len(df)>0 and df["signal"].iloc[-1]) else "🟢 عادی — همبستگی در محدوده نرمال"
    return df, regime


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ مقایسه با Benchmark
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_benchmark(ticker, period):
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if len(df) < 10:
            return None
        s = df["Close"].copy()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s
    except Exception:
        return None


def compare_to_benchmark(port_ret_series, bench_series):
    common = port_ret_series.index.intersection(bench_series.pct_change().dropna().index)
    if len(common) < 20:
        return None
    p = port_ret_series.loc[common]
    b = bench_series.pct_change().dropna().loc[common]
    port_ann  = (1+p).prod()**(252/len(p))-1
    bench_ann = (1+b).prod()**(252/len(b))-1
    port_vol  = p.std()*np.sqrt(252)
    bench_vol = b.std()*np.sqrt(252)
    cov_m     = np.cov(p.values, b.values)
    beta      = cov_m[0,1]/(cov_m[1,1]+1e-9)
    alpha     = port_ann - beta*bench_ann
    te        = (p-b).std()*np.sqrt(252)
    ir        = alpha/(te+1e-9)
    return dict(
        port_ann=port_ann*100, bench_ann=bench_ann*100,
        port_vol=port_vol*100, bench_vol=bench_vol*100,
        alpha=alpha*100, beta=beta, te=te*100, ir=ir,
        port_cum=(1+p).cumprod(), bench_cum=(1+b).cumprod(),
        dates=common, outperform=(port_ann>bench_ann),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    col_brand, col_toggle = st.columns([3, 2])
    with col_brand:
        st.markdown("""
        <div style="padding:0.9rem 0 0.7rem 0.2rem;">
            <div class="sb-wordmark">Portfolio360</div>
            <div class="sb-version">Blueprint · v3.2</div>
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

    # ══ CONFIG ══
    st.markdown('<div class="sb-section-label"><span class="sb-section-label-text">تنظیمات</span></div>', unsafe_allow_html=True)
    period_label = st.selectbox("بازه زمانی", list(PERIODS.keys()), index=2)
    period = PERIODS[period_label]
    rf_pct = st.number_input("نرخ بدون ریسک (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    rf = rf_pct / 100

    # ══ انتظارات و ریسک ══
    st.markdown('<div class="sb-section-label" style="margin-top:0.5rem"><span class="sb-section-label-text">انتظارات و ریسک</span></div>', unsafe_allow_html=True)
    expected_return = st.number_input("بازده مورد انتظار (%)", min_value=0.0, max_value=1000.0, value=0.0, step=5.0,
                                       help="بازده سالانه‌ای که انتظار دارید. صفر یعنی اعمال نشود.")
    if expected_return > 0:
        st.markdown(f"""
        <div class="expected-ret-display">
            <span class="expected-ret-label">هدف بازده</span>
            <span class="expected-ret-value">{expected_return:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    risk_geo = st.slider("🌐 ریسک ژئوپولیتیک (%)", 0, 100, 0, 5, help="تأثیر: کاهش بازده تعدیل‌شده (وزن ۴۰٪)")
    risk_mon = st.slider("🏦 ریسک سیاست پولی (%)", 0, 100, 0, 5, help="تأثیر: تغییر نرخ تنزیل مؤثر (وزن ۳۵٪)")
    risk_sys = st.slider("📉 ریسک سیستماتیک (%)", 0, 100, 0, 5, help="تأثیر: ریسک کل بازار (وزن ۲۵٪)")

    if risk_geo > 0 or risk_mon > 0 or risk_sys > 0:
        total_penalty = calc_risk_penalty(risk_geo, risk_mon, risk_sys)
        geo_color = "#e8945a" if is_dark else "#7a3a00"
        mon_color = "#5a9be8" if is_dark else "#1a4a7a"
        sys_color = "#b07ad4" if is_dark else "#4a1a6a"
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
                تنزل بازده: <strong style="color:{'#cc5555' if is_dark else '#8a2020'}">{total_penalty*100:.1f}%</strong>
                &nbsp;·&nbsp; نرخ مؤثر: <strong>{(rf + total_penalty)*100:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ══ ASSETS ══
    st.markdown('<div class="sb-section-label" style="margin-top:0.5rem"><span class="sb-section-label-text">انتخاب دارایی</span></div>', unsafe_allow_html=True)
    selected_tickers = []
    for cat, syms in SYMBOLS.items():
        with st.expander(cat, expanded=False):
            chosen = st.multiselect(cat, options=list(syms.keys()),
                                    format_func=lambda x, s=syms: f"{x}  ·  {s[x]}",
                                    key=f"ms_{cat}", label_visibility="collapsed")
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

    # ══ STRATEGY ══
    st.markdown('<div class="sb-section-label" style="margin-top:0.6rem"><span class="sb-section-label-text">استراتژی پرتفوی</span></div>', unsafe_allow_html=True)
    style_label = st.selectbox("روش بهینه‌سازی", list(STYLES.keys()))
    style = STYLES[style_label]
    calc_btn = st.button("▶  محاسبه پرتفوی", use_container_width=True)

    # ══════════════════════════════════════════════
    # ══ COVERED CALL — جدید ══
    # ══════════════════════════════════════════════
    st.markdown('<div class="sb-section-label" style="margin-top:0.6rem"><span class="sb-section-label-text">📊 Covered Call</span></div>', unsafe_allow_html=True)

    with st.expander("پارامترهای قرارداد", expanded=False):

        cc_spot = st.number_input(
            "قیمت فعلی دارایی ($)",
            min_value=0.01, max_value=1_000_000.0, value=100.0, step=1.0,
            help="قیمت لحظه‌ای سهام یا دارایی پایه‌ای که می‌خواهید call بفروشید."
        )

        cc_strike = st.number_input(
            "قیمت اعمال Strike ($)",
            min_value=0.01, max_value=1_000_000.0, value=105.0, step=1.0,
            help="قیمتی که اگر دارایی به آن برسد، اختیار اعمال می‌شود. بالاتر از spot یعنی OTM."
        )

        cc_days = st.number_input(
            "روز تا انقضا (DTE)",
            min_value=1, max_value=730, value=30, step=1,
            help="تعداد روز تا تاریخ انقضای قرارداد. معمولاً ۲۱ تا ۴۵ روز برای استراتژی ماهانه."
        )

        cc_iv = st.slider(
            "نوسان ضمنی IV (%)",
            min_value=5, max_value=200, value=30, step=1,
            help="نوسان ضمنی (Implied Volatility) که از بازار اختیار دریافت می‌شود. بالاتر = پرمیوم بیشتر."
        )

        cc_premium_manual = st.number_input(
            "پرمیوم دریافتی ($) — اختیاری",
            min_value=0.0, max_value=10_000.0, value=0.0, step=0.1,
            help="اگر قیمت واقعی پرمیوم را می‌دانید اینجا وارد کنید؛ در غیر این صورت از Black-Scholes محاسبه می‌شود."
        )

        cc_contracts = st.number_input(
            "تعداد قرارداد",
            min_value=1, max_value=10_000, value=1, step=1,
            help="هر قرارداد اختیار معامله معادل ۱۰۰ سهم است."
        )

    cc_analyze_btn = st.button("📊 تحلیل Covered Call", use_container_width=True)

    # ══ GUIDE ══
    st.markdown('<div class="sb-section-label" style="margin-top:0.4rem"><span class="sb-section-label-text">راهنما</span></div>', unsafe_allow_html=True)
    with st.expander("مراحل استفاده", expanded=False):
        steps = [
            ("۱","نمادها را از گروه‌های بالا انتخاب کنید"),
            ("۲","دکمه دریافت داده را بزنید"),
            ("۳","بازده مورد انتظار و ریسک‌ها را تنظیم کنید"),
            ("۴","روش بهینه‌سازی را انتخاب کنید"),
            ("۵","دکمه محاسبه پرتفوی را بزنید"),
            ("۶","برای تحلیل CC پارامترهای قرارداد را وارد کنید"),
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
        <div style="margin-top:0.4rem;padding:0.4rem 0.6rem;background:var(--border2);
            border-radius:3px;font-size:0.7rem;color:var(--muted)">
            ⚠ حداقل ۲ نماد برای محاسبه پرتفوی لازم است
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
# COVERED CALL — اگه دکمه زده شده، قبل از tabs نمایش بده
# ─────────────────────────────────────────────────────────────────────────────
if cc_analyze_btn:
    cc_result = analyze_covered_call(
        S=cc_spot, K=cc_strike, T_days=cc_days, r=rf,
        sigma=cc_iv / 100.0, premium=cc_premium_manual,
        contracts=cc_contracts,
        expected_ret_pct=expected_return,
        risk_geo=risk_geo, risk_mon=risk_mon, risk_sys=risk_sys
    )
    st.session_state["cc_result"] = cc_result
    st.session_state["cc_params"] = {
        "spot": cc_spot, "strike": cc_strike, "days": cc_days,
        "iv": cc_iv, "contracts": cc_contracts
    }

if st.session_state.get("cc_result"):
    cc = st.session_state["cc_result"]
    cp = st.session_state.get("cc_params", {})

    st.markdown('<div class="bp-section"><span class="bp-section-text">📊 Covered Call Analysis</span></div>', unsafe_allow_html=True)

    # ── VERDICT ──
    score = cc["worthwhile_score"]
    has_expected = expected_return > 0
    if not has_expected:
        verdict_class = "neutral"
        verdict_icon  = "⚖"
        verdict_title = "بازده مورد انتظار وارد نشده — تحلیل مقایسه‌ای موجود نیست"
        verdict_body  = (f"بازده سالانه‌شده پرمیوم: <strong>{cc['cc_ann_ret']*100:.2f}%</strong> "
                         f"(تعدیل‌شده ریسک: <strong>{cc['cc_adj_ret']*100:.2f}%</strong>). "
                         f"برای دیدن اینکه این CC نسبت به هدف بازده شما به‌صرفه هست یا نه، "
                         f"بازده مورد انتظار را در سایدبار وارد کنید.")
    elif score >= 0.01:
        verdict_class = "positive"
        verdict_icon  = "✅"
        verdict_title = "فروش این قرارداد به‌صرفه است"
        verdict_body  = (f"بازده تعدیل‌شده CC (<strong>{cc['cc_adj_ret']*100:.2f}%</strong>) "
                         f"از بازده انتظاری تعدیل‌شده شما (<strong>{cc['expected_ann_adj']*100:.2f}%</strong>) "
                         f"<strong>{score*100:.2f}%</strong> بیشتر است. "
                         f"با توجه به ریسک‌های وارد‌شده، این CC می‌تواند درآمد مکمل مناسبی باشد.")
    elif score >= -0.02:
        verdict_class = "neutral"
        verdict_icon  = "⚠"
        verdict_title = "مرزی — فروش قرارداد ارزش بررسی بیشتر دارد"
        verdict_body  = (f"بازده تعدیل‌شده CC ({cc['cc_adj_ret']*100:.2f}%) تقریباً "
                         f"برابر بازده انتظاری تعدیل‌شده شما ({cc['expected_ann_adj']*100:.2f}%) است. "
                         f"تفاوت ({score*100:.2f}%) در محدوده خطاست. "
                         f"IV بالاتر یا Strike نزدیک‌تر می‌تواند وضعیت را بهتر کند.")
    else:
        verdict_class = "negative"
        verdict_icon  = "❌"
        verdict_title = "فروش این قرارداد به‌صرفه نیست"
        verdict_body  = (f"بازده تعدیل‌شده CC ({cc['cc_adj_ret']*100:.2f}%) "
                         f"از بازده انتظاری تعدیل‌شده شما ({cc['expected_ann_adj']*100:.2f}%) "
                         f"<strong>{abs(score)*100:.2f}%</strong> کمتر است. "
                         f"پرمیوم این قرارداد ریسک محدودیت سود (cap) را توجیه نمی‌کند.")

    green_c = "#5aaa78" if is_dark else "#1a6640"
    gold_c  = "#c8a84b" if is_dark else "#8a6a1a"
    red_c   = "#cc5555" if is_dark else "#8a2020"
    blue_c  = "#5b9bd5"
    silver_c = "#909090" if is_dark else "#444440"

    verdict_color = green_c if verdict_class=="positive" else (red_c if verdict_class=="negative" else gold_c)

    st.markdown(f"""
    <div class="cc-verdict-card {verdict_class}">
        <div class="cc-verdict-title" style="color:{verdict_color}">{verdict_icon} {verdict_title}</div>
        <div class="cc-verdict-body">{verdict_body}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── GREEKS + METRICS ──
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border2);border-radius:4px;padding:0.9rem 1rem;margin-bottom:0.6rem">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.15em;color:var(--muted);text-transform:uppercase;margin-bottom:0.6rem">قیمت‌گذاری</div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">قیمت BS</span>
                <span class="cc-metric-val" style="color:{blue_c}">${cc['bs_price']:.3f}</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">پرمیوم مورد استفاده</span>
                <span class="cc-metric-val" style="color:{green_c}">${cc['premium']:.3f}</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">کل پرمیوم دریافتی</span>
                <span class="cc-metric-val" style="color:{green_c}">${cc['total_premium']:,.2f}</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">نقطه سر به سر</span>
                <span class="cc-metric-val">${cc['breakeven']:.2f}</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">Moneyness</span>
                <span class="cc-metric-val" style="color:{green_c if cc['moneyness']>0 else red_c}">{cc['moneyness']:+.2f}%</span>
            </div>
            <div class="cc-help-note">
                📌 <strong>قیمت BS</strong>: برآورد بلک-شولز از ارزش منصفانه اختیار.<br>
                📌 <strong>نقطه سربه‌سر</strong>: اگر قیمت دارایی به این عدد برسد، معامله نه سود دارد نه زیان.<br>
                📌 <strong>Moneyness</strong>: مثبت = اختیار OTM (خارج از پول) — ایده‌آل برای CC.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border2);border-radius:4px;padding:0.9rem 1rem;margin-bottom:0.6rem">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.15em;color:var(--muted);text-transform:uppercase;margin-bottom:0.6rem">Greeks</div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">Delta (Δ)</span>
                <span class="cc-metric-val">{cc['delta']:.4f}</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">Gamma (Γ)</span>
                <span class="cc-metric-val">{cc['gamma']:.6f}</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">Theta (Θ) روزانه</span>
                <span class="cc-metric-val" style="color:{green_c}">${cc['theta']:.4f}</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">Vega (ν) per 1%IV</span>
                <span class="cc-metric-val">${cc['vega']:.4f}</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">IV استفاده‌شده</span>
                <span class="cc-metric-val">{cc_iv}%</span>
            </div>
            <div class="cc-help-note">
                📌 <strong>Delta</strong>: احتمال اینکه اختیار در سود (ITM) منقضی شود. CC مناسب: Delta زیر ۰.۴<br>
                📌 <strong>Theta</strong>: سود روزانه از گذر زمان — فروشنده CC هر روز این مقدار کسب می‌کند.<br>
                📌 <strong>Vega</strong>: حساسیت پرمیوم به تغییر IV. افت IV = افت ارزش اختیار فروخته‌شده (خوب برای فروشنده).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border2);border-radius:4px;padding:0.9rem 1rem;margin-bottom:0.6rem">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.15em;color:var(--muted);text-transform:uppercase;margin-bottom:0.6rem">بازده و ریسک تعدیل‌شده</div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">بازده پرمیوم (دوره)</span>
                <span class="cc-metric-val" style="color:{green_c}">{cc['cc_period_ret']*100:.3f}%</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">بازده سالانه‌شده CC</span>
                <span class="cc-metric-val" style="color:{green_c}">{cc['cc_ann_ret']*100:.2f}%</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">CC تعدیل‌شده (ریسک)</span>
                <span class="cc-metric-val" style="color:{gold_c}">{cc['cc_adj_ret']*100:.2f}%</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">هدف بازده تعدیل‌شده</span>
                <span class="cc-metric-val" style="color:{blue_c}">{cc['expected_ann_adj']*100:.2f}%</span>
            </div>
            <div class="cc-metric-row">
                <span class="cc-metric-label">واگرایی از هدف</span>
                <span class="cc-metric-val" style="color:{green_c if score>=0 else red_c}">{score*100:+.2f}%</span>
            </div>
            <div class="cc-help-note">
                📌 <strong>بازده سالانه‌شده CC</strong>: پرمیوم را روی ۳۶۵ روز نرمال می‌کند تا با بازده سالانه پرتفو قابل مقایسه باشد.<br>
                📌 <strong>تعدیل‌شده ریسک</strong>: اثر ریسک‌های ژئوپولیتیک، پولی و سیستماتیک وارد‌شده را اعمال می‌کند.<br>
                📌 <strong>واگرایی</strong>: مثبت = CC از هدف شما بیشتر می‌دهد.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── سناریوها ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">سناریوهای P&L</span></div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric(
            "✅ اختیار منقضی می‌شه (زیر استرایک)",
            f"${cc['profit_below']:,.2f}",
            delta=f"{cc['ret_below']*100:.2f}% بازده",
            delta_color="normal"
        )
        st.caption("قیمت زیر Strike می‌ماند، اختیار بی‌ارزش منقضی می‌شود. کل پرمیوم سود خالص است. بهترین سناریو برای فروشنده CC.")

    with col_s2:
        st.metric(
            "⚡ اختیار اعمال می‌شه (بالای استرایک)",
            f"${cc['capped_gain']:,.2f}",
            delta=f"{cc['ret_capped']*100:.2f}% بازده",
            delta_color="normal"
        )
        st.caption(f"قیمت از Strike ({cp.get('strike','—')}$) بالا می‌رود. سهام call می‌شود. سود محدود به Strike - Spot + Premium. سقف سود وجود دارد.")

    with col_s3:
        st.metric(
            "💥 بدترین سناریو (قیمت صفر می‌شه)",
            f"-${cc['max_loss']:,.2f}",
            delta=f"{cc['ret_max_loss']*100:.2f}%",
            delta_color="inverse"
        )
        st.caption("ریسک نظری کاورد کال: افت شدید قیمت دارایی پایه. پرمیوم دریافتی این ضرر را کمی جبران می‌کند.")

    # ── نمودار P&L ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">نمودار سود و زیان در انقضا</span></div>', unsafe_allow_html=True)

    S_range = np.linspace(cp.get("spot", 100) * 0.5, cp.get("spot", 100) * 1.5, 300)
    K_val   = cp.get("strike", 105)
    prem    = cc["premium"]
    sh      = cc["shares"]

    # P&L استراتژی CC = سود/زیان سهام + پرمیوم - ارزش اختیار در انقضا
    pnl_stock = (S_range - cp.get("spot", 100)) * sh
    pnl_short_call = np.where(S_range > K_val, -(S_range - K_val) * sh, 0) + prem * sh
    pnl_cc = pnl_stock + pnl_short_call

    # فقط سهام (بدون CC)
    pnl_stock_only = (S_range - cp.get("spot", 100)) * sh

    bg_plot = "#161616" if is_dark else "#e8e8e4"
    txt_c   = "#c0c0c0" if is_dark else "#222220"
    grid_c  = "rgba(255,255,255,0.05)" if is_dark else "rgba(60,60,55,0.1)"

    fig_pnl = go.Figure()

    fig_pnl.add_trace(go.Scatter(
        x=S_range, y=pnl_cc, mode="lines", name="Covered Call",
        line=dict(color="#5b9bd5", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(91,155,213,0.08)"
    ))
    fig_pnl.add_trace(go.Scatter(
        x=S_range, y=pnl_stock_only, mode="lines", name="فقط سهام",
        line=dict(color="#888888", width=1.5, dash="dash"),
    ))
    # خط Strike
    fig_pnl.add_vline(x=K_val, line_dash="dot", line_color=gold_c, line_width=1.5,
                       annotation_text=f"Strike ${K_val}", annotation_font_color=gold_c, annotation_font_size=9)
    # خط Spot
    fig_pnl.add_vline(x=cp.get("spot",100), line_dash="dot",
                       line_color=silver_c, line_width=1,
                       annotation_text=f"Spot ${cp.get('spot',100)}", annotation_font_color=silver_c, annotation_font_size=9)
    # خط breakeven
    fig_pnl.add_vline(x=cc["breakeven"], line_dash="dot", line_color=red_c, line_width=1,
                       annotation_text=f"BE ${cc['breakeven']:.2f}", annotation_font_color=red_c, annotation_font_size=9)
    fig_pnl.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)

    fig_pnl.update_layout(**get_plot_layout("P&L AT EXPIRATION", "قیمت دارایی در انقضا ($)", "سود / زیان ($)", 420))
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ── نمودار Theta Decay ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">Theta Decay — کاهش ارزش اختیار با گذر زمان</span></div>', unsafe_allow_html=True)

    days_range = np.arange(cc_days, 0, -1)
    theta_vals = []
    for d in days_range:
        p, _, _, _, _ = black_scholes_call(cc_spot, cc_strike, d/365.0, rf, cc_iv/100.0)
        theta_vals.append(p)

    fig_theta = go.Figure()
    fig_theta.add_trace(go.Scatter(
        x=days_range, y=theta_vals,
        mode="lines", name="ارزش اختیار",
        line=dict(color="#e8a838", width=2),
        fill="tozeroy",
        fillcolor="rgba(232,168,56,0.1)"
    ))
    fig_theta.add_hline(y=cc["premium"], line_dash="dash",
                        line_color=green_c, line_width=1.2,
                        annotation_text=f"پرمیوم: ${cc['premium']:.3f}",
                        annotation_font_color=green_c, annotation_font_size=9)
    fig_theta.update_layout(**get_plot_layout("THETA DECAY — ارزش اختیار با گذر زمان", "روز تا انقضا", "ارزش اختیار ($)", 340))
    fig_theta.update_xaxes(autorange="reversed")
    st.plotly_chart(fig_theta, use_container_width=True)

    # ── شرایط ایده‌آل ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">چک‌لیست شرایط ایده‌آل CC</span></div>', unsafe_allow_html=True)

    checks = [
        (cc["otm_ok"],  f"OTM بودن Strike (فعلاً {'✓' if cc['otm_ok'] else f'✗ — Strike باید حداقل ۲٪ بالاتر از Spot باشد ({cc_strike:.1f} vs {cc_spot*1.02:.1f})'})"),
        (cc["iv_ok"],   f"IV کافی (IV = {cc_iv}% — {'✓ بالای ۲۰٪' if cc['iv_ok'] else '✗ — IV زیر ۲۰٪ پرمیوم کمی می‌دهد'})"),
        (cc["time_ok"], f"زمان کافی تا انقضا ({cc_days} روز — {'✓ بالای ۲۱ روز' if cc['time_ok'] else '✗ — زیر ۲۱ روز Theta کم است'})"),
        (cc["delta"] < 0.40, f"Delta مناسب (Δ={cc['delta']:.3f} — {'✓ زیر ۰.۴۰' if cc['delta'] < 0.40 else '✗ — Delta بالای ۰.۴ ریسک اعمال بالاست'})"),
        (cc["moneyness"] > 0, f"Moneyness مناسب ({cc['moneyness']:+.1f}% — {'✓ OTM است' if cc['moneyness']>0 else '✗ — اختیار ITM است، ریسک اعمال بالاست'})"),
    ]

    passed = sum(1 for c, _ in checks if c)
    score_color = green_c if passed >= 4 else (gold_c if passed >= 2 else red_c)

    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border2);border-radius:4px;padding:0.9rem 1.1rem">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.7rem">
            <span style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;color:var(--muted);text-transform:uppercase">شرایط کلی</span>
            <span style="font-family:'JetBrains Mono',monospace;font-weight:700;font-size:0.9rem;color:{score_color}">{passed}/{len(checks)} ✓</span>
        </div>
        {"".join(f'''<div style="display:flex;align-items:flex-start;gap:8px;padding:0.25rem 0;border-bottom:1px solid var(--border2)">
            <span style="color:{green_c if ok else red_c};font-size:0.85rem;min-width:16px;margin-top:1px">{"✓" if ok else "✗"}</span>
            <span style="font-size:0.75rem;color:var(--{'silver' if ok else 'muted'});line-height:1.5">{txt}</span>
        </div>''' for ok, txt in checks)}
        <div style="margin-top:0.6rem;font-size:0.65rem;color:var(--muted);line-height:1.6">
            💡 CC ایده‌آل: OTM · IV بالا · Delta ۰.۲–۰.۳۵ · ۲۱–۴۵ روز تا انقضا · روند سهام خنثی یا کمی صعودی
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT — پرتفوی
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
            w, cov = calc_weights(style, returns[asset_names], rf, asset_names,
                                   expected_return=expected_return,
                                   risk_geo=risk_geo, risk_mon=risk_mon, risk_sys=risk_sys)
            m = portfolio_metrics(w, returns[asset_names], rf,
                                   expected_return=expected_return,
                                   risk_geo=risk_geo, risk_mon=risk_mon, risk_sys=risk_sys)
        st.session_state["weights"] = w
        st.session_state["cov"] = cov
        st.session_state["metrics"] = m
        st.session_state["style_label"] = style_label
        st.session_state["saved_risks"] = {
            "expected_return": expected_return, "risk_geo": risk_geo,
            "risk_mon": risk_mon, "risk_sys": risk_sys,
        }

if st.session_state.get("weights") is None:
    st.info("⬅️ سبک پرتفوی را انتخاب و «محاسبه پرتفوی» را بزنید.")

saved = st.session_state.get("saved_risks", {})
if saved and (saved.get("risk_geo",0)>0 or saved.get("risk_mon",0)>0
              or saved.get("risk_sys",0)>0 or saved.get("expected_return",0)>0):
    geo_color_main = "#e8945a" if is_dark else "#7a3a00"
    mon_color_main = "#5a9be8" if is_dark else "#1a4a7a"
    sys_color_main = "#b07ad4" if is_dark else "#4a1a6a"
    gold_color_main= "#c8a84b" if is_dark else "#8a6a1a"
    red_c_main     = "#cc5555" if is_dark else "#8a2020"

    penalty_pct = calc_risk_penalty(
        saved.get("risk_geo",0), saved.get("risk_mon",0), saved.get("risk_sys",0)
    ) * 100

    badges = []
    if saved.get("expected_return",0)>0:
        badges.append(f"""<div class="risk-badge"><span class="risk-badge-icon">🎯</span>
            <span class="risk-badge-label">هدف بازده</span>
            <span class="risk-badge-value" style="color:{gold_color_main}">{saved['expected_return']:.0f}%</span></div>""")
    if saved.get("risk_geo",0)>0:
        badges.append(f"""<div class="risk-badge"><span class="risk-badge-icon">🌐</span>
            <span class="risk-badge-label">ژئوپولیتیک</span>
            <span class="risk-badge-value" style="color:{geo_color_main}">{saved['risk_geo']}%</span></div>""")
    if saved.get("risk_mon",0)>0:
        badges.append(f"""<div class="risk-badge"><span class="risk-badge-icon">🏦</span>
            <span class="risk-badge-label">سیاست پولی</span>
            <span class="risk-badge-value" style="color:{mon_color_main}">{saved['risk_mon']}%</span></div>""")
    if saved.get("risk_sys",0)>0:
        badges.append(f"""<div class="risk-badge"><span class="risk-badge-icon">📉</span>
            <span class="risk-badge-label">سیستماتیک</span>
            <span class="risk-badge-value" style="color:{sys_color_main}">{saved['risk_sys']}%</span></div>""")
    if penalty_pct>0:
        badges.append(f"""<div class="risk-badge" style="border-color:{'rgba(204,85,85,0.3)' if is_dark else 'rgba(138,32,32,0.3)'}">
            <span class="risk-badge-icon">⚠</span>
            <span class="risk-badge-label">تنزل بازده</span>
            <span class="risk-badge-value" style="color:{red_c_main}">−{penalty_pct:.1f}%</span></div>""")

    st.markdown(f'<div class="risk-badge-row">{"".join(badges)}</div>', unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab_ef, tab5, tab6, tab7, tab8, tab9, tab_live = st.tabs([
    "تخصیص پرتفوی",
    "ریسک و بازده",
    "نمودار قیمت",
    "مقایسه سبک‌ها",
    "📈 Efficient Frontier",
    "🎯 اختیار پیشرفته",
    "🧠 Black-Litterman",
    "🔥 Stress Test & MC",
    "⚖ ری‌بالانس",
    "📊 Benchmark",
    "🌐 داده زنده",
])

# ═══ TAB 1 ═══
with tab1:
    st.markdown('<div class="bp-section"><span class="bp-section-text">Portfolio Allocation</span></div>', unsafe_allow_html=True)
    w = st.session_state.get("weights")
    if w is None:
        st.info("ابتدا پرتفوی را محاسبه کنید.")
    else:
        df_w = pd.DataFrame({"نماد": asset_names,"وزن (%)": np.round(w*100,2)}).sort_values("وزن (%)", ascending=False)
        col1, col2 = st.columns([1,1])
        with col1:
            st.dataframe(df_w, use_container_width=True, hide_index=True)
            total_usd = st.number_input("کل سرمایه ($)", min_value=100, value=10000, step=500)
            df_alloc = df_w.copy()
            df_alloc["مبلغ ($)"] = (df_alloc["وزن (%)"]/100*total_usd).round(2)
            st.dataframe(df_alloc[["نماد","وزن (%)","مبلغ ($)"]], use_container_width=True, hide_index=True)
            st.download_button("↓ دانلود CSV", df_alloc.to_csv(index=False), file_name="portfolio.csv", use_container_width=True)
        with col2:
            fig_pie = go.Figure(go.Pie(
                labels=df_w["نماد"], values=df_w["وزن (%)"], hole=0.44,
                marker=dict(colors=COLORS[:len(df_w)],
                            line=dict(color="#161616" if is_dark else "#e8e8e4", width=2)),
                textfont=dict(size=9, family="JetBrains Mono"), textinfo="percent+label",
            ))
            fig_pie.update_layout(**get_plot_layout(title=f"ALLOCATION — {st.session_state.get('style_label','')}", h=400))
            st.plotly_chart(fig_pie, use_container_width=True)

# ═══ TAB 2 ═══
with tab2:
    m = st.session_state.get("metrics")
    w = st.session_state.get("weights")
    saved = st.session_state.get("saved_risks", {})
    if m is None or w is None:
        st.info("ابتدا پرتفوی را محاسبه کنید.")
    else:
        st.markdown('<div class="bp-section"><span class="bp-section-text">Risk &amp; Return Metrics</span></div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.metric("بازده سالانه (خام)", f"{m['بازده سالانه']*100:.2f}%")
            st.metric("نسبت شارپ", f"{m['نسبت شارپ']:.3f}")
        with c2:
            adj_ret = m["بازده تعدیل‌شده ریسک"]
            raw_ret = m["بازده سالانه"]
            delta_adj = f"−{(raw_ret-adj_ret)*100:.1f}% تنزل" if raw_ret!=adj_ret else "بدون تنزل"
            st.metric("بازده تعدیل‌شده ریسک", f"{adj_ret*100:.2f}%", delta=delta_adj, delta_color="inverse")
            st.metric("نسبت کالمار", f"{m['نسبت کالمار']:.3f}")
        with c3:
            st.metric("نوسان سالانه", f"{m['نوسان سالانه']*100:.2f}%")
            st.metric("CVaR 95%", f"{m['CVaR 95%']*100:.2f}%")
        with c4:
            rec = m["ریکاوری تایم (روز)"]
            rec_months = int(rec/21)
            rec_str = f"{rec_months} ماه" if rec_months else f"{rec} روز"
            st.metric("ریکاوری تایم", rec_str)
            st.metric("حداکثر افت (MDD)", f"{m['حداکثر افت (Max Drawdown)']*100:.2f}%")

        if saved.get("expected_return",0)>0 or saved.get("risk_geo",0)>0 or saved.get("risk_mon",0)>0 or saved.get("risk_sys",0)>0:
            st.markdown('<div class="bp-section"><span class="bp-section-text">Risk-Adjusted Analysis</span></div>', unsafe_allow_html=True)
            ca,cb,cc_col,cd = st.columns(4)
            with ca:
                st.metric("تنزل کل ریسک‌ها", f"{m['تنزل ریسک (%):']:.1f}%" if 'تنزل ریسک (%):' in m else f"{m['تنزل ریسک (%)']:.1f}%")
            with cb:
                gap = m["واگرایی از هدف"]
                if gap is not None:
                    st.metric("واگرایی از هدف بازده", f"{gap*100:+.2f}%",
                              delta="بالاتر از هدف" if gap>=0 else "پایین‌تر از هدف",
                              delta_color="normal" if gap>=0 else "inverse")
                else:
                    st.metric("واگرایی از هدف بازده","—")
            with cc_col:
                eff_rf = (rf + m["تنزل ریسک (%)"]/100)*100
                st.metric("نرخ بدون‌ریسک مؤثر", f"{eff_rf:.2f}%")
            with cd:
                st.metric("تعداد نمادها", str(len(asset_names)))

            if saved.get("risk_geo",0)>0 or saved.get("risk_mon",0)>0 or saved.get("risk_sys",0)>0:
                st.markdown('<div class="bp-section"><span class="bp-section-text">Risk Radar</span></div>', unsafe_allow_html=True)
                categories = ["ریسک ژئوپولیتیک","ریسک سیاست پولی","ریسک سیستماتیک"]
                values = [saved.get("risk_geo",0), saved.get("risk_mon",0), saved.get("risk_sys",0)]
                values_closed = values + [values[0]]
                categories_closed = categories + [categories[0]]
                bg = "#161616" if is_dark else "#e8e8e4"
                txt_c = "#c0c0c0" if is_dark else "#222220"
                grid_c = "rgba(255,255,255,0.08)" if is_dark else "rgba(60,60,55,0.1)"
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values_closed, theta=categories_closed, fill='toself',
                    fillcolor="rgba(204,85,85,0.15)" if is_dark else "rgba(138,32,32,0.1)",
                    line=dict(color="#cc5555" if is_dark else "#8a2020", width=2),
                    marker=dict(size=6, color="#cc5555" if is_dark else "#8a2020"),
                    name="ریسک‌ها"
                ))
                fig_radar.update_layout(
                    polar=dict(bgcolor=bg,
                               angularaxis=dict(tickfont=dict(size=10,color=txt_c,family="JetBrains Mono"),linecolor=grid_c,gridcolor=grid_c),
                               radialaxis=dict(visible=True,range=[0,100],tickfont=dict(size=8,color=txt_c),gridcolor=grid_c,linecolor=grid_c,
                                               tickvals=[25,50,75,100],ticktext=["25%","50%","75%","100%"])),
                    paper_bgcolor=bg, plot_bgcolor=bg,
                    font=dict(color=txt_c,family="JetBrains Mono"),
                    showlegend=False, height=320,
                    margin=dict(l=60,r=60,t=40,b=40),
                    title=dict(text="RISK RADAR",font=dict(color=txt_c,size=11,family="JetBrains Mono"),x=0.5)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<div class="bp-section"><span class="bp-section-text">Drawdown Chart</span></div>', unsafe_allow_html=True)
        port_ret_arr = returns[asset_names].values @ w
        cum = (1+port_ret_arr).cumprod()
        roll_max = pd.Series(cum).cummax()
        dd = (pd.Series(cum)-roll_max)/roll_max
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=prices.index[-len(dd):],y=dd.values*100,mode="lines",name="Drawdown",
            line=dict(color="#cc5555" if is_dark else "#8a2020",width=1.5),
            fill="tozeroy",fillcolor="rgba(204,85,85,0.12)" if is_dark else "rgba(138,32,32,0.08)"))
        fig_dd.update_layout(**get_plot_layout("DRAWDOWN (%)","DATE","DRAWDOWN %",340))
        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown('<div class="bp-section"><span class="bp-section-text">Portfolio Growth Curve</span></div>', unsafe_allow_html=True)
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=prices.index[-len(cum):],y=cum,mode="lines",name="Portfolio",line=dict(color="#5b9bd5",width=2)))
        if saved.get("expected_return",0)>0:
            n_days = len(cum)
            daily_target = (1+saved["expected_return"]/100)**(1/252)
            target_curve = [daily_target**i for i in range(n_days)]
            fig_cum.add_trace(go.Scatter(x=prices.index[-n_days:],y=target_curve,mode="lines",
                name=f"هدف {saved['expected_return']:.0f}%",
                line=dict(color="#c8a84b" if is_dark else "#8a6a1a",width=1.5,dash="dash")))
        fig_cum.add_hline(y=1.0,line_dash="dash",line_color="rgba(128,128,128,0.3)",line_width=1)
        fig_cum.update_layout(**get_plot_layout("PORTFOLIO GROWTH (BASE=1)","DATE","CUMULATIVE RETURN",370))
        st.plotly_chart(fig_cum, use_container_width=True)

        st.markdown('<div class="bp-section"><span class="bp-section-text">Daily Return Distribution</span></div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=port_ret_arr*100,nbinsx=60,
            marker_color="rgba(91,155,213,0.55)" if is_dark else "rgba(60,100,170,0.4)",
            marker_line=dict(color="rgba(91,155,213,0.8)",width=0.5),name="بازده روزانه"))
        cvar_line = np.percentile(port_ret_arr,5)*100
        fig_hist.add_vline(x=cvar_line,line_dash="dash",line_color="#cc5555" if is_dark else "#8a2020",line_width=1.5,
                            annotation_text=f"CVaR 95%: {cvar_line:.2f}%",
                            annotation_font_color="#cc5555" if is_dark else "#8a2020",annotation_font_size=9)
        fig_hist.update_layout(**get_plot_layout("DAILY RETURN DISTRIBUTION","RETURN %","FREQUENCY",350))
        st.plotly_chart(fig_hist, use_container_width=True)

# ═══ TAB 3 ═══
with tab3:
    st.markdown('<div class="bp-section"><span class="bp-section-text">Price Chart</span></div>', unsafe_allow_html=True)
    view_mode = st.radio("نمایش",["نرمال‌شده (base=100)","قیمت خام"],horizontal=True)
    fig_price = go.Figure()
    for i, col in enumerate(asset_names):
        s = prices[col]
        y = (s/s.iloc[0]*100).values if view_mode.startswith("نرمال") else s.values
        fig_price.add_trace(go.Scatter(x=prices.index,y=y,mode="lines",name=col,
                                        line=dict(color=COLORS[i%len(COLORS)],width=1.5)))
    yt = "قیمت نرمال‌شده (base=100)" if view_mode.startswith("نرمال") else "قیمت ($)"
    fig_price.update_layout(**get_plot_layout("PRICE CHART","DATE",yt,500))
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown('<div class="bp-section"><span class="bp-section-text">Correlation Matrix</span></div>', unsafe_allow_html=True)
    if len(asset_names) >= 2:
        corr = returns[asset_names].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,x=corr.columns,y=corr.index,
            colorscale=[[0.0,"#cc5555"],[0.5,"#161616" if is_dark else "#e0e0db"],[1.0,"#5b9bd5"]],
            zmid=0,zmin=-1,zmax=1,
            text=np.round(corr.values,2),texttemplate="%{text}",
            textfont=dict(size=9),showscale=True,
        ))
        fig_corr.update_layout(**get_plot_layout("CORRELATION MATRIX",h=max(340,len(asset_names)*34)))
        fig_corr.update_layout(xaxis=dict(tickangle=-45),yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_corr, use_container_width=True)

# ═══ TAB 4 ═══
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
                            ww,_ = calc_weights(sty,returns[asset_names],rf,asset_names,
                                                expected_return=expected_return,
                                                risk_geo=risk_geo,risk_mon=risk_mon,risk_sys=risk_sys)
                            mm = portfolio_metrics(ww,returns[asset_names],rf,
                                                   expected_return=expected_return,
                                                   risk_geo=risk_geo,risk_mon=risk_mon,risk_sys=risk_sys)
                            compare_results[lbl] = {**mm,"_weights":ww}
                        except Exception:
                            pass
                st.session_state["compare_results"] = compare_results
                st.session_state["compare_done"] = True
                st.session_state["compare_asset_names"] = asset_names[:]

            compare_results = st.session_state.get("compare_results",{})
            stored_assets = st.session_state.get("compare_asset_names",[])
            if stored_assets != asset_names:
                compare_results = {}
                st.session_state["compare_done"] = False

            if compare_results:
                rows = []
                for lbl, mm in compare_results.items():
                    rec = mm["ریکاوری تایم (روز)"]
                    rec_m = int(rec/21)
                    gap = mm["واگرایی از هدف"]
                    rows.append({
                        "سبک": lbl,
                        "بازده خام (%)": round(mm["بازده سالانه"]*100,2),
                        "بازده تعدیل‌شده (%)": round(mm["بازده تعدیل‌شده ریسک"]*100,2),
                        "نوسان (%)": round(mm["نوسان سالانه"]*100,2),
                        "شارپ": round(mm["نسبت شارپ"],3),
                        "MDD (%)": round(mm["حداکثر افت (Max Drawdown)"]*100,2),
                        "CVaR 95% (%)": round(mm["CVaR 95%"]*100,2),
                        "واگرایی هدف (%)": f"{gap*100:+.1f}%" if gap is not None else "—",
                        "تنزل ریسک (%)": round(mm["تنزل ریسک (%)"],1),
                        "ریکاوری": f"{rec_m} ماه" if rec_m else f"{rec} روز",
                        "کالمار": round(mm["نسبت کالمار"],3),
                    })
                df_cmp = pd.DataFrame(rows)
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)

                fig_cmp = go.Figure()
                metrics_to_plot = ["بازده تعدیل‌شده (%)","نوسان (%)","شارپ","MDD (%)"]
                for i, metric in enumerate(metrics_to_plot):
                    vals = [r[metric] for r in rows]
                    fig_cmp.add_trace(go.Bar(name=metric,x=[r["سبک"] for r in rows],y=vals,
                                              marker_color=COLORS[i],marker_line=dict(color="rgba(0,0,0,0.15)",width=0.5)))
                fig_cmp.update_layout(**get_plot_layout("STRATEGY COMPARISON (RISK-ADJUSTED)","","VALUE",400))
                fig_cmp.update_layout(barmode="group")
                st.plotly_chart(fig_cmp, use_container_width=True)

                st.markdown('<div class="bp-section"><span class="bp-section-text">Growth Curves — All Strategies</span></div>', unsafe_allow_html=True)
                fig_growth = go.Figure()
                ret_vals = returns[asset_names].values
                for i,(lbl,mm) in enumerate(compare_results.items()):
                    ww = np.array(mm["_weights"],dtype=float)
                    if ww.shape[0]!=ret_vals.shape[1]: continue
                    pr = ret_vals @ ww
                    cum_c = (1+pr).cumprod()
                    fig_growth.add_trace(go.Scatter(x=prices.index[-len(cum_c):],y=cum_c,
                                                     mode="lines",name=lbl,
                                                     line=dict(color=COLORS[i%len(COLORS)],width=1.8)))
                if saved.get("expected_return",0)>0:
                    n_d = len(ret_vals)
                    dly_t = (1+saved["expected_return"]/100)**(1/252)
                    tgt_c = [dly_t**i for i in range(n_d)]
                    fig_growth.add_trace(go.Scatter(x=prices.index[-n_d:],y=tgt_c,mode="lines",
                        name=f"هدف {saved['expected_return']:.0f}%",
                        line=dict(color="#c8a84b" if is_dark else "#8a6a1a",width=2,dash="dot")))
                fig_growth.update_layout(**get_plot_layout("GROWTH CURVES — ALL STRATEGIES","DATE","CUMULATIVE RETURN",460))
                st.plotly_chart(fig_growth, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB EF — Efficient Frontier & Rolling Metrics
# ════════════════════════════════════════════════════════════════════
with tab_ef:
    w_ef = st.session_state.get("weights")
    if w_ef is None or "prices" not in st.session_state:
        st.info("ابتدا پرتفوی را محاسبه کنید.")
    else:
        green_ef = "#5aaa78" if is_dark else "#1a6640"
        gold_ef  = "#c8a84b" if is_dark else "#8a6a1a"
        red_ef   = "#cc5555" if is_dark else "#8a2020"
        blue_ef  = "#5b9bd5"

        # ── Efficient Frontier ──
        st.markdown('<div class="bp-section"><span class="bp-section-text">Efficient Frontier — مرز کارایی پرتفوی</span></div>', unsafe_allow_html=True)
        st.caption("هر نقطه یک ترکیب ممکن از دارایی‌هاست. مرز کارایی بالاترین بازده به ازای هر سطح ریسک را نشان می‌دهد.")

        ef_btn = st.button("▶ رسم Efficient Frontier", use_container_width=True, key="ef_run")
        if ef_btn:
            with st.spinner("در حال شبیه‌سازی ۱۵۰۰ پرتفوی تصادفی..."):
                ret_ef   = returns[asset_names]
                mean_ef  = ret_ef.mean() * 252
                cov_ef   = ret_ef.cov() * 252
                n_ef     = len(asset_names)
                n_sims   = 1500
                sim_vols, sim_rets, sim_sharpes, sim_weights = [], [], [], []
                for _ in range(n_sims):
                    rw = np.random.dirichlet(np.ones(n_ef))
                    r  = float(rw @ mean_ef.values)
                    v  = float(np.sqrt(rw @ cov_ef.values @ rw))
                    sh = (r - rf) / (v + 1e-9)
                    sim_vols.append(v * 100)
                    sim_rets.append(r * 100)
                    sim_sharpes.append(sh)
                    sim_weights.append(rw)

                # نقاط مرز کارایی واقعی — از min_var تا max_sharpe
                frontier_vols, frontier_rets = [], []
                target_rets = np.linspace(min(sim_rets), max(sim_rets), 60)
                bnds_ef = [(0.01, 0.6)] * n_ef
                cons_ef = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
                eq_ef   = np.ones(n_ef) / n_ef
                for tr in target_rets:
                    cons_tr = cons_ef + [{"type": "eq", "fun": lambda w, t=tr: (w @ mean_ef.values) * 100 - t}]
                    try:
                        res = minimize(lambda w: w @ cov_ef.values @ w,
                                       eq_ef, method="SLSQP", bounds=bnds_ef, constraints=cons_tr)
                        if res.success:
                            frontier_vols.append(np.sqrt(res.fun) * 100)
                            frontier_rets.append(tr)
                    except Exception:
                        pass

                # موقعیت پرتفوی فعلی
                cur_ret = float(w_ef @ mean_ef.values) * 100
                cur_vol = float(np.sqrt(w_ef @ cov_ef.values @ w_ef)) * 100
                cur_sh  = (cur_ret/100 - rf) / (cur_vol/100 + 1e-9)

                st.session_state["ef_data"] = dict(
                    sim_vols=sim_vols, sim_rets=sim_rets, sim_sharpes=sim_sharpes,
                    frontier_vols=frontier_vols, frontier_rets=frontier_rets,
                    cur_vol=cur_vol, cur_ret=cur_ret, cur_sh=cur_sh,
                    sim_weights=sim_weights,
                )

        green_ef = "#5aaa78" if is_dark else "#1a6640"
        gold_ef  = "#c8a84b" if is_dark else "#8a6a1a"
        red_ef   = "#cc5555" if is_dark else "#8a2020"
        blue_ef  = "#5b9bd5"

        ef_data = st.session_state.get("ef_data")
        if ef_data:
            fig_ef = go.Figure()

            # ابر نقاط تصادفی رنگ‌بندی بر اساس شارپ
            fig_ef.add_trace(go.Scatter(
                x=ef_data["sim_vols"], y=ef_data["sim_rets"],
                mode="markers",
                marker=dict(
                    size=4, opacity=0.55,
                    color=ef_data["sim_sharpes"],
                    colorscale=[[0,"#cc5555"],[0.5,"#c8a84b"],[1,"#5aaa78"]],
                    showscale=True,
                    colorbar=dict(title=dict(text="Sharpe", font=dict(size=9, color="#888")),
                                  tickfont=dict(size=8, color="#888"), thickness=10, len=0.6),
                ),
                name="پرتفوی‌های تصادفی",
                hovertemplate="ریسک: %{x:.2f}%<br>بازده: %{y:.2f}%<extra></extra>",
            ))

            # مرز کارایی
            if ef_data["frontier_vols"]:
                fig_ef.add_trace(go.Scatter(
                    x=ef_data["frontier_vols"], y=ef_data["frontier_rets"],
                    mode="lines", name="مرز کارایی",
                    line=dict(color=blue_ef, width=2.5),
                ))

            # پرتفوی فعلی
            fig_ef.add_trace(go.Scatter(
                x=[ef_data["cur_vol"]], y=[ef_data["cur_ret"]],
                mode="markers+text",
                marker=dict(size=14, color=gold_ef, symbol="star",
                            line=dict(color="#fff", width=1.5)),
                text=["پرتفوی شما"], textposition="top center",
                textfont=dict(color=gold_ef, size=10),
                name=f"پرتفوی فعلی (Sharpe={ef_data['cur_sh']:.2f})",
            ))

            # بهترین شارپ از نقاط تصادفی
            best_idx = int(np.argmax(ef_data["sim_sharpes"]))
            fig_ef.add_trace(go.Scatter(
                x=[ef_data["sim_vols"][best_idx]], y=[ef_data["sim_rets"][best_idx]],
                mode="markers+text",
                marker=dict(size=12, color=green_ef, symbol="diamond",
                            line=dict(color="#fff", width=1.5)),
                text=["بیشترین شارپ"], textposition="top center",
                textfont=dict(color=green_ef, size=10),
                name=f"بهترین شارپ ({max(ef_data['sim_sharpes']):.2f})",
            ))

            fig_ef.update_layout(**get_plot_layout(
                "EFFICIENT FRONTIER — مرز کارایی پرتفوی",
                "نوسان سالانه (%)", "بازده سالانه (%)", 520))
            st.plotly_chart(fig_ef, use_container_width=True)

            # اطلاعات تکمیلی
            ca, cb, cc_col = st.columns(3)
            with ca:
                st.metric("بازده پرتفوی فعلی", f"{ef_data['cur_ret']:.2f}%")
                st.caption("بازده سالانه‌شده بر اساس داده تاریخی")
            with cb:
                st.metric("نوسان پرتفوی فعلی", f"{ef_data['cur_vol']:.2f}%")
                st.caption("ریسک سالانه‌شده پرتفوی انتخابی")
            with cc_col:
                st.metric("نسبت شارپ فعلی", f"{ef_data['cur_sh']:.3f}")
                st.caption("بازده اضافه تقسیم بر ریسک")

        # ── Rolling Metrics ──
        st.markdown('<div class="bp-section"><span class="bp-section-text">Rolling Metrics — تغییرات شارپ و نوسان در طول زمان</span></div>', unsafe_allow_html=True)
        st.caption("عملکرد پرتفوی در پنجره های زمانی متحرک - برای شناسایی دوره های ضعف یا قدرت.")

        roll_window = st.slider("پنجره Rolling (روز)", 21, 252, 63, 21, key="roll_win")

        port_ret_ef    = returns[asset_names].values @ w_ef
        port_series_ef = pd.Series(port_ret_ef, index=returns.index)

        roll_vol    = port_series_ef.rolling(roll_window).std() * np.sqrt(252) * 100
        roll_mean   = port_series_ef.rolling(roll_window).mean() * 252 * 100
        roll_sharpe = (port_series_ef.rolling(roll_window).mean() * 252 - rf) / \
                      (port_series_ef.rolling(roll_window).std() * np.sqrt(252) + 1e-9)

        _green = green_ef
        _gold  = gold_ef
        _red   = red_ef
        _blue  = blue_ef

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=returns.index, y=roll_sharpe.values,
            mode="lines", name="Rolling Sharpe",
            line=dict(color=_blue, width=2),
            fill="tozeroy", fillcolor="rgba(91,155,213,0.08)",
        ))
        fig_roll.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
        fig_roll.add_hline(y=1, line_dash="dot",
                           line_color=_green, line_width=1,
                           annotation_text="Sharpe=1", annotation_font_color=_green, annotation_font_size=9)
        fig_roll.update_layout(**get_plot_layout(
            f"ROLLING SHARPE — پنجره {roll_window} روزه", "DATE", "Sharpe", 320))
        st.plotly_chart(fig_roll, use_container_width=True)

        fig_roll2 = go.Figure()
        fig_roll2.add_trace(go.Scatter(
            x=returns.index, y=roll_vol.values,
            mode="lines", name="Rolling Volatility",
            line=dict(color=_red, width=1.8),
            fill="tozeroy", fillcolor="rgba(204,85,85,0.08)",
        ))
        fig_roll2.add_trace(go.Scatter(
            x=returns.index, y=roll_mean.values,
            mode="lines", name="Rolling Return (ann.)",
            line=dict(color=_green, width=1.8),
        ))
        fig_roll2.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
        if st.session_state.get("saved_risks", {}).get("expected_return", 0) > 0:
            fig_roll2.add_hline(
                y=st.session_state["saved_risks"]["expected_return"],
                line_dash="dot", line_color=_gold, line_width=1.2,
                annotation_text=f"هدف {st.session_state['saved_risks']['expected_return']:.0f}%",
                annotation_font_color=_gold, annotation_font_size=9,
            )
        fig_roll2.update_layout(**get_plot_layout(
            f"ROLLING VOLATILITY & RETURN — پنجره {roll_window} روزه", "DATE", "%", 340))
        st.plotly_chart(fig_roll2, use_container_width=True)

        # Rolling Sharpe توزیع
        st.markdown('<div class="bp-section"><span class="bp-section-text">توزیع Rolling Sharpe</span></div>', unsafe_allow_html=True)
        valid_sharpe = roll_sharpe.dropna().values
        if len(valid_sharpe) > 10:
            pct_positive = (valid_sharpe > 0).mean() * 100
            pct_above1   = (valid_sharpe > 1).mean() * 100
            fig_sh_hist = go.Figure()
            fig_sh_hist.add_trace(go.Histogram(
                x=valid_sharpe, nbinsx=50,
                marker_color="rgba(91,155,213,0.55)",
                marker_line=dict(color="rgba(91,155,213,0.8)", width=0.5),
                name="Rolling Sharpe",
            ))
            fig_sh_hist.add_vline(x=0, line_color=_red, line_width=1.5, line_dash="dash",
                                   annotation_text="Sharpe=0", annotation_font_color=_red, annotation_font_size=9)
            fig_sh_hist.add_vline(x=1, line_color=_green, line_width=1.5, line_dash="dash",
                                   annotation_text="Sharpe=1", annotation_font_color=_green, annotation_font_size=9)
            fig_sh_hist.update_layout(**get_plot_layout("ROLLING SHARPE DISTRIBUTION", "Sharpe", "فراوانی", 300))
            st.plotly_chart(fig_sh_hist, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("درصد دوره‌های Sharpe > 0", f"{pct_positive:.1f}%")
                st.caption("چه درصد از پنجره‌های زمانی بازده مثبت نسبت به ریسک داشتند")
            with c2:
                st.metric("درصد دوره‌های Sharpe > 1", f"{pct_above1:.1f}%")
                st.caption("چه درصد از پنجره‌ها عملکرد عالی (Sharpe بالای ۱) داشتند")


# ════════════════════════════════════════════════════════════════════
# TAB 5 — اختیار معامله پیشرفته
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="bp-section"><span class="bp-section-text">Advanced Options Strategies</span></div>', unsafe_allow_html=True)

    green_t = "#5aaa78" if is_dark else "#1a6640"
    gold_t  = "#c8a84b" if is_dark else "#8a6a1a"
    red_t   = "#cc5555" if is_dark else "#8a2020"
    blue_t  = "#5b9bd5"

    opt_tab = st.selectbox("استراتژی", ["Protective Put", "Iron Condor", "Rolling Covered Call"])

    if opt_tab == "Protective Put":
        st.markdown('<div class="bp-section"><span class="bp-section-text">Protective Put — بیمه پرتفوی</span></div>', unsafe_allow_html=True)
        st.caption("با خرید Put، ریسک سقوط قیمت را محدود می‌کنید. مثل بیمه عمر برای دارایی.")

        co1, co2, co3 = st.columns(3)
        with co1:
            pp_spot    = st.number_input("قیمت فعلی دارایی ($)", value=100.0, step=1.0, key="pp_spot")
            pp_strike  = st.number_input("Strike Put ($)", value=95.0, step=1.0, key="pp_strike",
                                          help="قیمت اعمال Put — معمولاً ۵–۱۰٪ زیر قیمت فعلی")
        with co2:
            pp_days    = st.number_input("روز تا انقضا", value=30, step=1, key="pp_days")
            pp_iv      = st.slider("IV (%)", 5, 200, 30, key="pp_iv",
                                    help="نوسان ضمنی Put. بالاتر = بیمه گرانتر")
        with co3:
            pp_shares  = st.number_input("تعداد سهام", value=100, step=100, key="pp_shares",
                                          help="تعداد سهامی که می‌خواهید بیمه کنید")
            pp_prem    = st.number_input("پرمیوم واقعی ($) — اختیاری", value=0.0, step=0.1, key="pp_prem")

        if st.button("محاسبه Protective Put", use_container_width=True, key="pp_calc"):
            pp = analyze_protective_put(pp_spot, pp_strike, pp_days, rf, pp_iv/100,
                                         pp_prem, pp_shares, expected_return,
                                         risk_geo, risk_mon, risk_sys)
            st.session_state["pp_result"] = pp

        pp_r = st.session_state.get("pp_result")
        if pp_r:
            verdict_c = "positive" if pp_r["worthwhile"] else "negative"
            verdict_icon = "✅" if pp_r["worthwhile"] else "❌"
            verdict_txt = (
                f"هزینه سالانه‌شده بیمه ({pp_r['cost_ann']:.2f}%) از بازده انتظاری تعدیل‌شده ({pp_r['expected_adj']:.2f}%) "
                f"{'کمتر است — بیمه منطقی است' if pp_r['worthwhile'] else 'بیشتر است — هزینه بیمه زیاد است'}"
            ) if expected_return > 0 else f"هزینه سالانه‌شده بیمه: {pp_r['cost_ann']:.2f}% — بازده مورد انتظار وارد نشده."

            vc = green_t if pp_r["worthwhile"] else red_t
            st.markdown(f"""
            <div class="cc-verdict-card {verdict_c}">
                <div class="cc-verdict-title" style="color:{vc}">{verdict_icon} Protective Put</div>
                <div class="cc-verdict-body">{verdict_txt}</div>
            </div>""", unsafe_allow_html=True)

            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.metric("قیمت Put (BS)", f"${pp_r['bs_price']:.3f}")
                st.caption("ارزش منصفانه بلک-شولز برای Put خریداری‌شده")
            with c2:
                st.metric("کل هزینه بیمه", f"${pp_r['total_cost']:,.2f}")
                st.caption("پرمیوم کل = پرمیوم × تعداد سهام")
            with c3:
                st.metric("حداکثر زیان بیمه‌شده", f"${pp_r['max_loss_insured']:,.2f}")
                st.caption("بدترین حالت با وجود Put: اگر قیمت زیر Strike برود")
            with c4:
                st.metric("نقطه سربه‌سر", f"${pp_r['breakeven']:.2f}")
                st.caption("قیمتی که باید دارایی به آن برسد تا هزینه Put جبران شود")

            c5,c6,c7 = st.columns(3)
            with c5:
                st.metric("Delta", f"{pp_r['delta']:.4f}")
                st.caption("حساسیت قیمت Put به تغییر دارایی. برای Put منفی است (محافظ)")
            with c6:
                st.metric("Theta روزانه", f"${pp_r['theta']:.4f}")
                st.caption("هر روز این مقدار از ارزش Put کم می‌شود — هزینه نگهداری")
            with c7:
                st.metric("هزینه سالانه‌شده", f"{pp_r['cost_ann']:.2f}%")
                st.caption("هزینه Put روی ۳۶۵ روز نرمال‌شده — برای مقایسه با بازده پرتفو")

            # P&L Chart
            S_r = np.linspace(pp_spot*0.5, pp_spot*1.5, 300)
            pnl_naked = (S_r - pp_spot)*pp_shares
            pnl_pp    = pnl_naked + np.maximum(pp_strike - S_r, 0)*pp_shares - pp_r["total_cost"]
            fig_pp = go.Figure()
            fig_pp.add_trace(go.Scatter(x=S_r, y=pnl_pp, mode="lines", name="با Protective Put",
                                         line=dict(color=blue_t, width=2.5), fill="tozeroy", fillcolor="rgba(91,155,213,0.08)"))
            fig_pp.add_trace(go.Scatter(x=S_r, y=pnl_naked, mode="lines", name="بدون بیمه",
                                         line=dict(color="#888888", width=1.5, dash="dash")))
            fig_pp.add_vline(x=pp_strike, line_dash="dot", line_color=red_t, line_width=1.5,
                              annotation_text=f"Strike ${pp_strike}", annotation_font_color=red_t, annotation_font_size=9)
            fig_pp.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
            fig_pp.update_layout(**get_plot_layout("PROTECTIVE PUT — P&L AT EXPIRATION", "قیمت دارایی ($)", "سود / زیان ($)", 380))
            st.plotly_chart(fig_pp, use_container_width=True)

    elif opt_tab == "Iron Condor":
        st.markdown('<div class="bp-section"><span class="bp-section-text">Iron Condor — درآمد از بازار خنثی</span></div>', unsafe_allow_html=True)
        st.caption("Iron Condor با فروش همزمان یک Put Spread و یک Call Spread، از بازار رنجبند درآمد کسب می‌کند.")

        co1, co2 = st.columns(2)
        with co1:
            ic_spot = st.number_input("قیمت فعلی ($)", value=100.0, step=1.0, key="ic_spot")
            ic_kpb  = st.number_input("Strike Put Buy ($)", value=88.0, step=1.0, key="ic_kpb",
                                       help="پایین‌ترین Strike — حداکثر زیان پایین را محدود می‌کند")
            ic_kps  = st.number_input("Strike Put Sell ($)", value=92.0, step=1.0, key="ic_kps",
                                       help="Strike Put فروخته‌شده — زیر قیمت فعلی")
        with co2:
            ic_kcs  = st.number_input("Strike Call Sell ($)", value=108.0, step=1.0, key="ic_kcs",
                                       help="Strike Call فروخته‌شده — بالای قیمت فعلی")
            ic_kcb  = st.number_input("Strike Call Buy ($)", value=112.0, step=1.0, key="ic_kcb",
                                       help="بالاترین Strike — حداکثر زیان بالا را محدود می‌کند")
            ic_days = st.number_input("روز تا انقضا", value=30, step=1, key="ic_days")
            ic_iv   = st.slider("IV (%)", 5, 200, 30, key="ic_iv")
            ic_cont = st.number_input("تعداد قرارداد", value=1, step=1, key="ic_cont")

        if st.button("محاسبه Iron Condor", use_container_width=True, key="ic_calc"):
            ic = analyze_iron_condor(ic_spot, ic_kpb, ic_kps, ic_kcs, ic_kcb,
                                      ic_days, rf, ic_iv/100, ic_cont,
                                      expected_return, risk_geo, risk_mon, risk_sys)
            st.session_state["ic_result"] = ic
            st.session_state["ic_params"] = dict(spot=ic_spot, kpb=ic_kpb, kps=ic_kps, kcs=ic_kcs, kcb=ic_kcb, cont=ic_cont)

        ic_r = st.session_state.get("ic_result")
        if ic_r:
            icp = st.session_state.get("ic_params", {})
            ws  = ic_r["worthwhile_score"]
            vc  = green_t if ws >= 0 else red_t
            vi  = "✅" if ws >= 1 else ("⚠" if ws >= -2 else "❌")
            vt  = ("Iron Condor به‌صرفه است" if ws>=1 else ("مرزی — بررسی بیشتر لازم است" if ws>=-2 else "Iron Condor به‌صرفه نیست"))

            st.markdown(f"""
            <div class="cc-verdict-card {'positive' if ws>=1 else ('neutral' if ws>=-2 else 'negative')}">
                <div class="cc-verdict-title" style="color:{vc}">{vi} {vt}</div>
                <div class="cc-verdict-body">
                    بازده تعدیل‌شده: <strong>{ic_r['adj_ann_ret']:.2f}%</strong> vs هدف: <strong>{ic_r['expected_adj']:.2f}%</strong> —
                    واگرایی: <strong style="color:{vc}">{ws:+.2f}%</strong><br>
                    محدوده سود: <strong>${ic_r['be_lower']:.2f}</strong> تا <strong>${ic_r['be_upper']:.2f}</strong>
                    ({ic_r['profit_zone_pct']:.1f}% از قیمت فعلی)
                </div>
            </div>""", unsafe_allow_html=True)

            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.metric("اعتبار خالص دریافتی", f"${ic_r['net_credit']:.3f}/سهم")
                st.caption("مجموع پرمیوم دریافتی منهای پرمیوم پرداختی — سود اگر در محدوده بمانیم")
            with c2:
                st.metric("کل اعتبار", f"${ic_r['total_credit']:,.2f}")
                st.caption("اعتبار خالص × تعداد قرارداد × ۱۰۰ سهم")
            with c3:
                st.metric("حداکثر زیان", f"${ic_r['max_loss']:,.2f}")
                st.caption("اگر قیمت از محدوده خارج شود و به بالاترین یا پایین‌ترین Strike برسد")
            with c4:
                st.metric("بازده روی ریسک", f"{ic_r['ret_on_risk']:.2f}%")
                st.caption("نسبت اعتبار دریافتی به حداکثر زیان — معیار اصلی کارایی IC")

            # P&L Chart
            S_r  = np.linspace(icp.get("kpb",88)*0.9, icp.get("kcb",112)*1.1, 300)
            kpb, kps = icp.get("kpb",88), icp.get("kps",92)
            kcs, kcb = icp.get("kcs",108), icp.get("kcb",112)
            nc   = ic_r["net_credit"]
            cont = icp.get("cont",1)
            pnl  = np.where(S_r <= kpb,  -(kps-kpb-nc)*cont*100,
                   np.where(S_r <= kps,  (S_r-kpb-nc)*cont*100 - (kps-kpb)*cont*100 + nc*cont*100,  # approximate
                   np.where(S_r <= kcs,   nc*cont*100,
                   np.where(S_r <= kcb,  (nc-(S_r-kcs))*cont*100,
                                          -(kcb-kcs-nc)*cont*100))))
            # simpler accurate calc
            pnl2 = []
            for s in S_r:
                put_spread  = -max(kps-s,0) + max(kpb-s,0)
                call_spread = -max(s-kcs,0) + max(s-kcb,0)
                pnl2.append((put_spread + call_spread + nc)*cont*100)

            fig_ic = go.Figure()
            fig_ic.add_trace(go.Scatter(x=S_r, y=pnl2, mode="lines", name="Iron Condor P&L",
                                         line=dict(color=blue_t, width=2.5),
                                         fill="tozeroy", fillcolor="rgba(91,155,213,0.08)"))
            for xv, lbl, c in [(kpb,"Put Buy",red_t),(kps,"Put Sell",gold_t),
                                 (kcs,"Call Sell",gold_t),(kcb,"Call Buy",red_t)]:
                fig_ic.add_vline(x=xv, line_dash="dot", line_color=c, line_width=1.2,
                                  annotation_text=lbl, annotation_font_color=c, annotation_font_size=8)
            fig_ic.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
            fig_ic.update_layout(**get_plot_layout("IRON CONDOR — P&L AT EXPIRATION","قیمت دارایی ($)","سود / زیان ($)",380))
            st.plotly_chart(fig_ic, use_container_width=True)

    elif opt_tab == "Rolling Covered Call":
        st.markdown('<div class="bp-section"><span class="bp-section-text">Rolling Covered Call — شبیه‌سازی تاریخی</span></div>', unsafe_allow_html=True)
        st.caption("شبیه‌سازی می‌کند اگه هر دوره یک CC می‌فروختید و roll می‌کردید، چقدر درآمد کسب می‌کردید.")

        if "prices" not in st.session_state or st.session_state["prices"] is None:
            st.info("ابتدا داده دانلود کنید.")
        else:
            all_tickers = list(st.session_state["prices"].columns)
            rc_ticker = st.selectbox("نماد پایه", all_tickers, key="rc_ticker")
            co1, co2, co3 = st.columns(3)
            with co1:
                rc_offset = st.slider("Strike OTM (%)", 1, 20, 5, key="rc_off",
                                       help="درصد بالاتر از قیمت فعلی برای Strike هر دوره")
            with co2:
                rc_dte = st.number_input("DTE هر دوره (روز)", value=30, step=5, key="rc_dte",
                                          help="تعداد روز هر CC قبل از Roll")
            with co3:
                rc_iv = st.slider("IV فرضی (%)", 10, 100, 30, key="rc_iv",
                                   help="نوسان ضمنی ثابت فرضی برای شبیه‌سازی")
                rc_cont = st.number_input("تعداد قرارداد", value=1, step=1, key="rc_cont")

            if st.button("اجرای شبیه‌سازی Rolling CC", use_container_width=True, key="rc_run"):
                prices_sel = st.session_state["prices"][rc_ticker]
                df_roll = simulate_rolling_cc(prices_sel, rc_offset, rc_dte, rf, rc_iv/100, rc_cont)
                st.session_state["roll_result"] = df_roll

            roll_df = st.session_state.get("roll_result")
            if roll_df is not None and len(roll_df) > 0:
                total_prem  = roll_df["premium_earned"].sum()
                total_opt   = roll_df["option_pnl"].sum()
                total_stock = roll_df["stock_pnl"].sum()
                n_cycles    = len(roll_df)
                n_exercised = roll_df["exercised"].sum()

                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("کل پرمیوم دریافتی", f"${total_prem:,.2f}")
                    st.caption("مجموع همه پرمیوم‌های دریافتی در طول دوره شبیه‌سازی")
                with c2:
                    st.metric("سود/زیان اختیارات", f"${total_opt:,.2f}")
                    st.caption("سود خالص از اختیارات پس از کسر زیان دوره‌های اعمال‌شده")
                with c3:
                    st.metric("تعداد سیکل‌ها", str(n_cycles))
                    st.caption("تعداد دفعاتی که CC فروخته و Roll شد")
                with c4:
                    st.metric("دفعات اعمال", f"{n_exercised} ({n_exercised/max(n_cycles,1)*100:.0f}%)")
                    st.caption("تعداد دوره‌هایی که اختیار اعمال شد (سهام call شد)")

                fig_roll = go.Figure()
                fig_roll.add_trace(go.Bar(x=roll_df["date"], y=roll_df["option_pnl"],
                                           name="سود اختیار", marker_color=green_t, opacity=0.8))
                fig_roll.add_trace(go.Scatter(x=roll_df["date"],
                                               y=roll_df["option_pnl"].cumsum(),
                                               name="تجمعی پرمیوم", mode="lines",
                                               line=dict(color=gold_t, width=2.5)))
                fig_roll.update_layout(**get_plot_layout("ROLLING CC — سود دوره‌ای و تجمعی","","سود ($)",380))
                st.plotly_chart(fig_roll, use_container_width=True)

                st.dataframe(roll_df.rename(columns={
                    "date":"تاریخ","S":"قیمت ورود","K":"Strike","premium":"پرمیوم",
                    "premium_earned":"پرمیوم کل","S_exp":"قیمت انقضا",
                    "exercised":"اعمال شد؟","option_pnl":"سود اختیار","delta":"Delta"
                })[["تاریخ","قیمت ورود","Strike","پرمیوم","قیمت انقضا","اعمال شد؟","سود اختیار","Delta"]],
                use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# TAB 6 — Black-Litterman + Factor
# ════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="bp-section"><span class="bp-section-text">Black-Litterman — دیدگاه‌های شخصی روی بازار</span></div>', unsafe_allow_html=True)
    st.caption("Black-Litterman دیدگاه‌های شما را با داده تاریخی ترکیب می‌کند تا وزن‌های بهتری بسازد.")

    if "prices" not in st.session_state or st.session_state["prices"] is None:
        st.info("ابتدا داده دانلود کنید.")
    else:
        prices_bl = st.session_state["prices"]
        asset_bl  = list(prices_bl.columns)
        ret_bl    = prices_bl.pct_change().dropna()
        cov_bl    = ret_bl.cov()*252
        mean_bl   = ret_bl.mean()
        eq_w      = np.ones(len(asset_bl))/len(asset_bl)

        st.markdown('<div class="bp-section"><span class="bp-section-text">دیدگاه‌های شما (Views)</span></div>', unsafe_allow_html=True)
        st.caption("برای هر نمادی که دیدگاه دارید، بازده سالانه مورد انتظار وارد کنید (مثلاً ۰.۲۰ یعنی ۲۰٪+).")

        views = {}
        n_views = st.number_input("تعداد دیدگاه", 0, min(len(asset_bl),10), 0, step=1, key="bl_nv")
        for i in range(int(n_views)):
            cv1, cv2 = st.columns([2,1])
            with cv1:
                v_asset = st.selectbox(f"نماد {i+1}", asset_bl, key=f"bl_a{i}")
            with cv2:
                v_ret = st.number_input(f"بازده انتظاری (%)", value=10.0, step=5.0, key=f"bl_r{i}")
            views[v_asset] = v_ret/100

        tau = st.slider("Tau (اطمینان به پرایور)", 0.01, 0.20, 0.05, 0.01, key="bl_tau",
                         help="عدد کوچکتر = اطمینان بیشتر به داده بازار، بزرگتر = اطمینان بیشتر به دیدگاه شما")

        if st.button("محاسبه وزن‌های Black-Litterman", use_container_width=True, key="bl_calc"):
            w_bl, mu_bl = black_litterman(eq_w, cov_bl, mean_bl, views, tau)
            st.session_state["bl_weights"] = w_bl
            st.session_state["bl_mu"] = mu_bl

        w_bl = st.session_state.get("bl_weights")
        if w_bl is not None:
            df_bl = pd.DataFrame({
                "نماد": asset_bl,
                "وزن BL (%)": np.round(w_bl*100,2),
                "وزن برابر (%)": np.round(eq_w*100,2),
                "تفاوت (%)": np.round((w_bl-eq_w)*100,2),
            }).sort_values("وزن BL (%)", ascending=False)

            col1, col2 = st.columns([1,1])
            with col1:
                st.dataframe(df_bl, use_container_width=True, hide_index=True)
            with col2:
                fig_bl = go.Figure()
                fig_bl.add_trace(go.Bar(name="BL", x=df_bl["نماد"], y=df_bl["وزن BL (%)"],
                                         marker_color=blue_t, opacity=0.85))
                fig_bl.add_trace(go.Bar(name="برابر", x=df_bl["نماد"], y=df_bl["وزن برابر (%)"],
                                         marker_color="#888888", opacity=0.5))
                fig_bl.update_layout(**get_plot_layout("BL vs EQUAL WEIGHT","","وزن (%)",380))
                fig_bl.update_layout(barmode="group")
                st.plotly_chart(fig_bl, use_container_width=True)

        st.markdown('<div class="bp-section"><span class="bp-section-text">Factor Exposure — عوامل موثر بر دارایی‌ها</span></div>', unsafe_allow_html=True)
        st.caption("تحلیل فاکتورهای کلیدی هر دارایی: Momentum (شتاب ۶ ماهه)، Volatility، Beta (حساسیت به بازار)، Sharpe.")

        if st.button("محاسبه Factor Exposure", use_container_width=True, key="factor_calc"):
            df_factor = compute_factor_exposure(ret_bl[asset_bl])
            st.session_state["factor_df"] = df_factor

        factor_df = st.session_state.get("factor_df")
        if factor_df is not None:
            st.dataframe(factor_df, use_container_width=True)

            fig_fac = go.Figure()
            metrics_f = ["مومنتوم_6ماه","نوسان_سالانه","بتا","شارپ"]
            colors_f  = [blue_t, red_t, gold_t, "#3db87a"]
            for m, c in zip(metrics_f, colors_f):
                fig_fac.add_trace(go.Bar(name=m, x=factor_df.index.tolist(), y=factor_df[m].values,
                                          marker_color=c, opacity=0.8))
            fig_fac.update_layout(**get_plot_layout("FACTOR EXPOSURE","","مقدار",400))
            fig_fac.update_layout(barmode="group")
            st.plotly_chart(fig_fac, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 7 — Stress Test + Monte Carlo
# ════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="bp-section"><span class="bp-section-text">Stress Test — مقاومت در برابر بحران‌های تاریخی</span></div>', unsafe_allow_html=True)
    st.caption("شبیه‌سازی می‌کند اگه پرتفوی فعلی در هر بحران تاریخی بود، چه اتفاقی می‌افتاد.")

    w_st = st.session_state.get("weights")
    if w_st is None or "prices" not in st.session_state:
        st.info("ابتدا پرتفوی را محاسبه کنید.")
    else:
        prices_st = st.session_state["prices"]
        asset_st  = list(prices_st.columns)
        w_series  = pd.Series(w_st, index=asset_st)

        if st.button("اجرای Stress Test", use_container_width=True, key="st_run"):
            df_stress = run_stress_tests(prices_st, w_series)
            st.session_state["stress_df"] = df_stress

        df_stress = st.session_state.get("stress_df")
        if df_stress is not None and len(df_stress) > 0:
            st.dataframe(df_stress, use_container_width=True, hide_index=True)

            fig_stress = go.Figure()
            fig_stress.add_trace(go.Bar(x=df_stress["بحران"], y=df_stress["بازده_کل"],
                                         name="بازده کل",
                                         marker_color=[("#cc5555" if v<0 else "#3db87a") for v in df_stress["بازده_کل"]]))
            fig_stress.add_trace(go.Bar(x=df_stress["بحران"], y=df_stress["حداکثر_افت"],
                                         name="حداکثر افت", marker_color=red_t, opacity=0.6))
            fig_stress.update_layout(**get_plot_layout("STRESS TEST — عملکرد در بحران‌ها","","بازده (%)",400))
            fig_stress.update_layout(barmode="group")
            st.plotly_chart(fig_stress, use_container_width=True)

        st.markdown('<div class="bp-section"><span class="bp-section-text">Monte Carlo — شبیه‌سازی آینده</span></div>', unsafe_allow_html=True)
        st.caption("۴۰۰ مسیر تصادفی احتمالی برای آینده پرتفو با توزیع بازده تاریخی.")

        mc_years = st.slider("افق زمانی (سال)", 1, 10, 3, key="mc_years")
        mc_capital = st.number_input("سرمایه اولیه ($)", value=10000, step=1000, key="mc_cap")

        if st.button("اجرای Monte Carlo", use_container_width=True, key="mc_run"):
            returns_mc = prices_st.pct_change().dropna()[asset_st]
            mc = monte_carlo_future(w_st, returns_mc, n_sims=400, horizon_years=mc_years)
            st.session_state["mc_result"] = mc
            st.session_state["mc_capital"] = mc_capital

        mc = st.session_state.get("mc_result")
        if mc:
            cap = st.session_state.get("mc_capital", 10000)
            days_arr = np.arange(mc["n_days"])

            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.metric("احتمال سود", f"{mc['prob_profit']:.1f}%")
                st.caption("درصد مسیرهایی که پرتفو در پایان افق زمانی در سود است")
            with c2:
                st.metric("احتمال دو برابر شدن", f"{mc['prob_2x']:.1f}%")
                st.caption("درصد مسیرهایی که سرمایه دو برابر می‌شود")
            with c3:
                st.metric("میانه سرمایه نهایی", f"${mc['median']*cap:,.0f}")
                st.caption("مقدار میانه سرمایه در پایان دوره در بین تمام سناریوها")
            with c4:
                st.metric("بدترین ۵٪", f"${mc['worst5']*cap:,.0f}")
                st.caption("در ۵٪ بدترین سناریوها، سرمایه به این عدد می‌رسد")

            fig_mc = go.Figure()
            x_dates = days_arr
            fig_mc.add_trace(go.Scatter(x=x_dates, y=mc["pct95"]*cap, mode="lines", name="سقف ۹۵٪",
                                         line=dict(color=green_t, width=0.5, dash="dot")))
            fig_mc.add_trace(go.Scatter(x=x_dates, y=mc["pct75"]*cap, mode="lines", name="۷۵ درصدیل",
                                         line=dict(color=green_t, width=1),
                                         fill="tonexty", fillcolor="rgba(90,170,120,0.06)"))
            fig_mc.add_trace(go.Scatter(x=x_dates, y=mc["pct50"]*cap, mode="lines", name="میانه",
                                         line=dict(color=blue_t, width=2.5),
                                         fill="tonexty", fillcolor="rgba(91,155,213,0.08)"))
            fig_mc.add_trace(go.Scatter(x=x_dates, y=mc["pct25"]*cap, mode="lines", name="۲۵ درصدیل",
                                         line=dict(color=red_t, width=1),
                                         fill="tonexty", fillcolor="rgba(204,85,85,0.06)"))
            fig_mc.add_trace(go.Scatter(x=x_dates, y=mc["pct5"]*cap, mode="lines", name="کف ۵٪",
                                         line=dict(color=red_t, width=0.5, dash="dot")))
            fig_mc.add_hline(y=cap, line_dash="dash", line_color="rgba(128,128,128,0.4)", line_width=1,
                              annotation_text="سرمایه اولیه", annotation_font_size=9)
            fig_mc.update_layout(**get_plot_layout(f"MONTE CARLO — {mc_years} سال آینده ({400} مسیر)","روز","ارزش پرتفو ($)",440))
            st.plotly_chart(fig_mc, use_container_width=True)

            # توزیع نهایی
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=mc["final"]*cap, nbinsx=50,
                                             marker_color="rgba(91,155,213,0.6)",
                                             marker_line=dict(color="rgba(91,155,213,0.9)", width=0.5),
                                             name="توزیع ارزش نهایی"))
            fig_dist.add_vline(x=cap, line_dash="dash", line_color=gold_t, line_width=1.5,
                                annotation_text="سرمایه اولیه", annotation_font_color=gold_t, annotation_font_size=9)
            fig_dist.add_vline(x=mc["median"]*cap, line_dash="dash", line_color=blue_t, line_width=1.5,
                                annotation_text=f"میانه ${mc['median']*cap:,.0f}", annotation_font_color=blue_t, annotation_font_size=9)
            fig_dist.update_layout(**get_plot_layout("توزیع سرمایه نهایی","ارزش ($)","تعداد مسیر",320))
            st.plotly_chart(fig_dist, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 8 — Rebalancing + Correlation Regime
# ════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown('<div class="bp-section"><span class="bp-section-text">Rebalancing Alert — نیاز به ری‌بالانس</span></div>', unsafe_allow_html=True)
    st.caption("مقایسه وزن هدف با وزن فعلی (بر اساس آخرین قیمت) و محاسبه معاملات لازم.")

    w_rb = st.session_state.get("weights")
    if w_rb is None or "prices" not in st.session_state:
        st.info("ابتدا پرتفوی را محاسبه کنید.")
    else:
        prices_rb  = st.session_state["prices"]
        asset_rb   = list(prices_rb.columns)
        last_prices = prices_rb.iloc[-1]
        rb_capital  = st.number_input("ارزش فعلی پرتفو ($)", value=10000, step=1000, key="rb_cap")
        rb_threshold = st.slider("آستانه ری‌بالانس (%)", 1, 20, 5, key="rb_thresh",
                                  help="اگه انحراف وزن از این عدد بیشتر بشه، ری‌بالانس لازمه")

        df_rb = calc_rebalancing(last_prices, w_rb, asset_rb, rb_capital, rb_threshold/100)
        if len(df_rb) > 0:
            needs = (df_rb["وضعیت"].str.contains("ری‌بالانس")).sum()
            if needs > 0:
                st.warning(f"⚠ {needs} نماد نیاز به ری‌بالانس دارد (انحراف بیش از {rb_threshold}٪)")
            else:
                st.success(f"✅ همه نمادها در محدوده هدف هستند (انحراف زیر {rb_threshold}٪)")
            st.dataframe(df_rb, use_container_width=True, hide_index=True)

            fig_rb = go.Figure()
            fig_rb.add_trace(go.Bar(name="وزن هدف", x=df_rb["نماد"], y=df_rb["وزن_هدف"],
                                     marker_color=blue_t, opacity=0.8))
            fig_rb.add_trace(go.Bar(name="وزن فعلی", x=df_rb["نماد"], y=df_rb["وزن_فعلی"],
                                     marker_color=[red_t if abs(v)>rb_threshold else green_t for v in df_rb["انحراف"]],
                                     opacity=0.7))
            fig_rb.update_layout(**get_plot_layout("REBALANCING — وزن هدف vs فعلی","","وزن (%)",360))
            fig_rb.update_layout(barmode="group")
            st.plotly_chart(fig_rb, use_container_width=True)

    st.markdown('<div class="bp-section"><span class="bp-section-text">Correlation Regime — تشخیص رژیم بازار</span></div>', unsafe_allow_html=True)
    st.caption("اگه همبستگی‌ها ناگهان افزایش یابد، معمولاً نشانه استرس بازار یا بحران است.")

    if "prices" not in st.session_state or st.session_state["prices"] is None:
        st.info("ابتدا داده دانلود کنید.")
    elif len(st.session_state["prices"].columns) >= 2:
        prices_cr = st.session_state["prices"]
        asset_cr  = list(prices_cr.columns)
        ret_cr    = prices_cr.pct_change().dropna()[asset_cr]

        if st.button("تشخیص رژیم همبستگی", use_container_width=True, key="cr_run"):
            df_regime, regime_label = detect_correlation_regime(ret_cr)
            st.session_state["regime_df"] = df_regime
            st.session_state["regime_label"] = regime_label

        regime_label = st.session_state.get("regime_label")
        df_regime    = st.session_state.get("regime_df")

        if regime_label:
            is_crisis = "🔴" in regime_label
            rc = red_t if is_crisis else green_t
            st.markdown(f"""
            <div class="cc-verdict-card {'negative' if is_crisis else 'positive'}">
                <div class="cc-verdict-title" style="color:{rc}">رژیم فعلی بازار</div>
                <div class="cc-verdict-body">{regime_label}</div>
            </div>""", unsafe_allow_html=True)

        if df_regime is not None and len(df_regime) > 0:
            fig_regime = go.Figure()
            fig_regime.add_trace(go.Scatter(x=df_regime["date"], y=df_regime["corr_short"],
                                             mode="lines", name="همبستگی ۳۰ روزه",
                                             line=dict(color=red_t, width=1.8)))
            fig_regime.add_trace(go.Scatter(x=df_regime["date"], y=df_regime["corr_long"],
                                             mode="lines", name="همبستگی ۱۲۶ روزه",
                                             line=dict(color=blue_t, width=1.8, dash="dash")))
            # ناحیه‌های بحران
            crisis_dates = df_regime[df_regime["signal"]]["date"]
            if len(crisis_dates) > 0:
                fig_regime.add_trace(go.Scatter(
                    x=df_regime["date"],
                    y=np.where(df_regime["signal"], df_regime["corr_short"], np.nan),
                    mode="lines", name="هشدار بحران",
                    line=dict(color=red_t, width=0), fill="tozeroy",
                    fillcolor="rgba(204,85,85,0.12)"
                ))
            fig_regime.update_layout(**get_plot_layout("CORRELATION REGIME DETECTION","","میانگین همبستگی",380))
            st.plotly_chart(fig_regime, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 9 — Benchmark Comparison
# ════════════════════════════════════════════════════════════════════
with tab9:
    st.markdown('<div class="bp-section"><span class="bp-section-text">Benchmark Comparison — مقایسه با شاخص</span></div>', unsafe_allow_html=True)
    st.caption("مقایسه عملکرد پرتفوی با یک شاخص مرجع: Alpha، Beta، Tracking Error، Information Ratio.")

    w_bm = st.session_state.get("weights")
    if w_bm is None or "prices" not in st.session_state:
        st.info("ابتدا پرتفوی را محاسبه کنید.")
    else:
        prices_bm = st.session_state["prices"]
        asset_bm  = list(prices_bm.columns)
        ret_bm    = prices_bm.pct_change().dropna()[asset_bm]

        bm_ticker  = st.selectbox("بنچمارک",["SPY","QQQ","DIA","IWM","BTC-USD","GLD","TLT"], key="bm_tick")
        bm_period  = st.selectbox("بازه", list(PERIODS.keys()), index=2, key="bm_per")

        if st.button("مقایسه با بنچمارک", use_container_width=True, key="bm_run"):
            bench_prices = fetch_benchmark(bm_ticker, PERIODS[bm_period])
            if bench_prices is not None:
                port_r_arr  = ret_bm.values @ w_bm
                port_series = pd.Series(port_r_arr, index=prices_bm.index[-len(port_r_arr):])
                bm_result   = compare_to_benchmark(port_series, bench_prices)
                st.session_state["bm_result"] = bm_result
                st.session_state["bm_ticker"] = bm_ticker
            else:
                st.error(f"دانلود داده بنچمارک {bm_ticker} ناموفق بود.")

        bm_r = st.session_state.get("bm_result")
        bm_n = st.session_state.get("bm_ticker","")
        if bm_r:
            vc = green_t if bm_r["outperform"] else red_t
            vi = "✅" if bm_r["outperform"] else "❌"
            vt = f"{'بهتر' if bm_r['outperform'] else 'ضعیف‌تر'} از {bm_n}"
            gap = bm_r["port_ann"] - bm_r["bench_ann"]

            st.markdown(f"""
            <div class="cc-verdict-card {'positive' if bm_r['outperform'] else 'negative'}">
                <div class="cc-verdict-title" style="color:{vc}">{vi} پرتفوی {vt}</div>
                <div class="cc-verdict-body">
                    بازده پرتفو: <strong>{bm_r['port_ann']:.2f}%</strong> vs
                    بازده {bm_n}: <strong>{bm_r['bench_ann']:.2f}%</strong> —
                    واگرایی: <strong style="color:{vc}">{gap:+.2f}%</strong>
                </div>
            </div>""", unsafe_allow_html=True)

            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.metric("Alpha (α)", f"{bm_r['alpha']:.2f}%")
                st.caption("بازده اضافه‌ای که پرتفو نسبت به ریسک سیستماتیک کسب کرده. مثبت = مهارت مدیر")
            with c2:
                st.metric("Beta (β)", f"{bm_r['beta']:.3f}")
                st.caption("حساسیت پرتفو به بازار. ۱.۰ = برابر بازار، کمتر = محافظه‌کارانه‌تر")
            with c3:
                st.metric("Tracking Error", f"{bm_r['te']:.2f}%")
                st.caption("انحراف معیار تفاوت بازده پرتفو و بنچمارک. کمتر = نزدیک‌تر به شاخص")
            with c4:
                st.metric("Information Ratio", f"{bm_r['ir']:.3f}")
                st.caption("Alpha تقسیم بر Tracking Error. بالاتر از ۰.۵ = عملکرد فعال خوب")

            c5,c6 = st.columns(2)
            with c5:
                st.metric("بازده پرتفو", f"{bm_r['port_ann']:.2f}%")
                st.caption("بازده سالانه‌شده پرتفوی بهینه‌شده")
            with c6:
                st.metric(f"بازده {bm_n}", f"{bm_r['bench_ann']:.2f}%")
                st.caption(f"بازده سالانه‌شده بنچمارک {bm_n} در همان دوره")

            fig_bm = go.Figure()
            fig_bm.add_trace(go.Scatter(x=bm_r["dates"], y=bm_r["port_cum"],
                                         mode="lines", name="پرتفوی",
                                         line=dict(color=blue_t, width=2.5)))
            fig_bm.add_trace(go.Scatter(x=bm_r["dates"], y=bm_r["bench_cum"],
                                         mode="lines", name=bm_n,
                                         line=dict(color=gold_t, width=1.8, dash="dash")))
            fig_bm.add_hline(y=1.0, line_dash="dash", line_color="rgba(128,128,128,0.3)", line_width=1)
            fig_bm.update_layout(**get_plot_layout(f"PORTFOLIO vs {bm_n} — رشد تجمعی","","بازده تجمعی (base=1)",420))
            st.plotly_chart(fig_bm, use_container_width=True)

            # نمودار Alpha تجمعی
            alpha_cum = bm_r["port_cum"] - bm_r["bench_cum"]
            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Scatter(x=bm_r["dates"], y=alpha_cum.values,
                                            mode="lines", name="Alpha تجمعی",
                                            line=dict(color=green_t if bm_r["outperform"] else red_t, width=2),
                                            fill="tozeroy",
                                            fillcolor=f"rgba({'90,170,120' if bm_r['outperform'] else '204,85,85'},0.1)"))
            fig_alpha.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
            fig_alpha.update_layout(**get_plot_layout("CUMULATIVE ALPHA — پرتفو منهای بنچمارک","","Alpha تجمعی",320))
            st.plotly_chart(fig_alpha, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
risk_summary = ""
# ════════════════════════════════════════════════════════════════════
# TAB LIVE — داده زنده
# ════════════════════════════════════════════════════════════════════
with tab_live:
    import urllib.request
    import xml.etree.ElementTree as ET
    import json as _json

    green_lv = "#5aaa78" if is_dark else "#1a6640"
    gold_lv  = "#c8a84b" if is_dark else "#8a6a1a"
    red_lv   = "#cc5555" if is_dark else "#8a2020"
    blue_lv  = "#5b9bd5"
    muted_lv = "#888888" if is_dark else "#555550"
    card_bg  = "#222222" if is_dark else "#e4e4e0"
    border_c = "rgba(180,180,180,0.1)" if is_dark else "rgba(60,60,55,0.15)"

    st.markdown('<div class="bp-section"><span class="bp-section-text">Fear & Greed Index</span></div>', unsafe_allow_html=True)
    st.caption("شاخص ترس و طمع CNN — بدون API، مستقیم از سایت.")

    @st.cache_data(show_spinner=False, ttl=1800)
    def fetch_fear_greed():
        try:
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = _json.loads(r.read().decode())
            score = float(data["fear_and_greed"]["score"])
            rating = data["fear_and_greed"]["rating"]
            prev_close = float(data["fear_and_greed"]["previous_close"])
            prev_1w    = float(data["fear_and_greed"]["previous_1_week"])
            prev_1m    = float(data["fear_and_greed"]["previous_1_month"])
            prev_1y    = float(data["fear_and_greed"]["previous_1_year"])
            return dict(score=score, rating=rating, prev_close=prev_close,
                        prev_1w=prev_1w, prev_1m=prev_1m, prev_1y=prev_1y, ok=True)
        except Exception as e:
            return dict(ok=False, error=str(e))

    fg = fetch_fear_greed()
    if fg["ok"]:
        score = fg["score"]
        if score <= 25:
            fg_color = red_lv
            fg_label = "ترس شدید"
            fg_emoji = "😱"
        elif score <= 45:
            fg_color = "#e8945a" if is_dark else "#7a3a00"
            fg_label = "ترس"
            fg_emoji = "😨"
        elif score <= 55:
            fg_color = gold_lv
            fg_label = "خنثی"
            fg_emoji = "😐"
        elif score <= 75:
            fg_color = "#7eb35a"
            fg_label = "طمع"
            fg_emoji = "😏"
        else:
            fg_color = green_lv
            fg_label = "طمع شدید"
            fg_emoji = "🤑"

        # gauge
        fig_fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(font=dict(size=36, color=fg_color, family="JetBrains Mono"), suffix=""),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=muted_lv,
                          tickfont=dict(size=9, color=muted_lv)),
                bar=dict(color=fg_color, thickness=0.25),
                bgcolor=card_bg,
                borderwidth=0,
                steps=[
                    dict(range=[0,  25], color="rgba(204,85,85,0.18)"),
                    dict(range=[25, 45], color="rgba(232,148,90,0.14)"),
                    dict(range=[45, 55], color="rgba(200,168,75,0.12)"),
                    dict(range=[55, 75], color="rgba(126,179,90,0.14)"),
                    dict(range=[75,100], color="rgba(90,170,120,0.18)"),
                ],
                threshold=dict(line=dict(color=fg_color, width=3), thickness=0.75, value=score),
            ),
            title=dict(text=f"{fg_emoji} {fg_label}", font=dict(size=14, color=fg_color, family="JetBrains Mono")),
        ))
        fig_fg.update_layout(
            paper_bgcolor="#161616" if is_dark else "#e8e8e4",
            font=dict(color=muted_lv, family="JetBrains Mono"),
            height=280, margin=dict(l=30, r=30, t=40, b=10),
        )
        st.plotly_chart(fig_fg, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        def delta_str(val):
            d = score - val
            return f"{d:+.1f}"
        with c1:
            st.metric("دیروز", f"{fg['prev_close']:.0f}", delta=delta_str(fg["prev_close"]))
        with c2:
            st.metric("هفته پیش", f"{fg['prev_1w']:.0f}", delta=delta_str(fg["prev_1w"]))
        with c3:
            st.metric("ماه پیش", f"{fg['prev_1m']:.0f}", delta=delta_str(fg["prev_1m"]))
        with c4:
            st.metric("سال پیش", f"{fg['prev_1y']:.0f}", delta=delta_str(fg["prev_1y"]))

        st.markdown(f"""
        <div style="font-size:0.65rem;color:{muted_lv};padding:0.4rem 0;line-height:1.7">
        📌 <b>0–25</b> ترس شدید — فرصت خرید احتمالی &nbsp;|&nbsp;
        <b>26–45</b> ترس &nbsp;|&nbsp;
        <b>46–55</b> خنثی &nbsp;|&nbsp;
        <b>56–75</b> طمع &nbsp;|&nbsp;
        <b>76–100</b> طمع شدید — احتیاط در خرید
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"دریافت Fear & Greed ناموفق بود: {fg.get('error','')}")

    # ── اخبار نمادهای انتخابی ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">اخبار نمادهای پرتفوی</span></div>', unsafe_allow_html=True)
    st.caption("آخرین اخبار از Yahoo Finance RSS — بدون نیاز به API.")

    @st.cache_data(show_spinner=False, ttl=900)
    def fetch_news_rss(ticker: str, n: int = 5):
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                root = ET.fromstring(r.read())
            items = root.findall(".//item")[:n]
            news = []
            for item in items:
                title = item.findtext("title", "").strip()
                link  = item.findtext("link", "").strip()
                pub   = item.findtext("pubDate", "").strip()
                desc  = item.findtext("description", "").strip()
                if title:
                    news.append(dict(title=title, link=link, pub=pub, desc=desc))
            return news
        except Exception:
            return []

    if "prices" in st.session_state and st.session_state["prices"] is not None:
        tickers_for_news = list(st.session_state["prices"].columns)
        news_ticker = st.selectbox("نماد", tickers_for_news, key="news_sel")
        news_items = fetch_news_rss(news_ticker, n=7)
        if news_items:
            for item in news_items:
                pub_short = item["pub"][:22] if item["pub"] else ""
                st.markdown(f"""
                <div style="padding:0.65rem 0.9rem;margin-bottom:0.5rem;
                    background:{card_bg};border:1px solid {border_c};
                    border-left:3px solid {blue_lv};border-radius:4px;">
                    <a href="{item['link']}" target="_blank"
                       style="font-size:0.82rem;font-weight:600;color:{blue_lv};
                              text-decoration:none;line-height:1.5">
                        {item['title']}
                    </a>
                    <div style="font-size:0.65rem;color:{muted_lv};margin-top:0.25rem">
                        {pub_short}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"خبری برای {news_ticker} یافت نشد.")
    else:
        st.info("ابتدا داده پرتفوی را دانلود کنید.")

    # ── Seasonality ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">Seasonality — بازده ماهانه تاریخی</span></div>', unsafe_allow_html=True)
    st.caption("میانگین بازده هر ماه در سال‌های گذشته — از داده yfinance محاسبه می‌شود.")

    if "prices" in st.session_state and st.session_state["prices"] is not None:
        seas_ticker = st.selectbox("نماد برای Seasonality", list(st.session_state["prices"].columns), key="seas_sel")
        seas_prices = st.session_state["prices"][seas_ticker]
        seas_ret = seas_prices.resample("ME").last().pct_change().dropna()
        seas_ret.index = pd.to_datetime(seas_ret.index)
        seas_df = pd.DataFrame({"month": seas_ret.index.month, "year": seas_ret.index.year, "ret": seas_ret.values})

        month_names = {1:"فروردین",2:"اردیبهشت",3:"خرداد",4:"تیر",5:"مرداد",6:"شهریور",
                       7:"مهر",8:"آبان",9:"آذر",10:"دی",11:"بهمن",12:"اسفند"}
        month_names_en = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

        avg_by_month = seas_df.groupby("month")["ret"].mean() * 100
        std_by_month = seas_df.groupby("month")["ret"].std() * 100
        pos_by_month = seas_df.groupby("month")["ret"].apply(lambda x: (x > 0).mean() * 100)

        colors_seas = [green_lv if v >= 0 else red_lv for v in avg_by_month.values]

        fig_seas = go.Figure()
        fig_seas.add_trace(go.Bar(
            x=[month_names_en[m] for m in avg_by_month.index],
            y=avg_by_month.values,
            marker_color=colors_seas,
            marker_line=dict(color="rgba(0,0,0,0.15)", width=0.5),
            error_y=dict(type="data", array=std_by_month.values, visible=True,
                         color=muted_lv, thickness=1.2, width=4),
            name="میانگین بازده ماهانه",
            hovertemplate="%{x}<br>میانگین: %{y:.2f}%<extra></extra>",
        ))
        fig_seas.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
        fig_seas.update_layout(**get_plot_layout(
            f"SEASONALITY — {seas_ticker}", "ماه", "میانگین بازده (%)", 360))
        st.plotly_chart(fig_seas, use_container_width=True)

        # جدول ماه × سال (heatmap)
        st.markdown('<div class="bp-section"><span class="bp-section-text">Heatmap بازده ماهانه</span></div>', unsafe_allow_html=True)
        pivot = seas_df.pivot_table(index="year", columns="month", values="ret", aggfunc="mean") * 100
        pivot.columns = [month_names_en[c] for c in pivot.columns]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=[str(y) for y in pivot.index],
            colorscale=[[0.0, red_lv], [0.5, card_bg], [1.0, green_lv]],
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            textfont=dict(size=8),
            showscale=True,
            colorbar=dict(thickness=10, len=0.8,
                          tickfont=dict(size=8, color=muted_lv),
                          title=dict(text="%", font=dict(size=9, color=muted_lv))),
        ))
        fig_heat.update_layout(**get_plot_layout(
            f"MONTHLY RETURN HEATMAP — {seas_ticker}", "", "سال",
            max(320, len(pivot) * 28)))
        fig_heat.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_heat, use_container_width=True)

        # بهترین و بدترین ماه‌ها
        ca, cb = st.columns(2)
        with ca:
            best_m = int(avg_by_month.idxmax())
            st.metric(f"بهترین ماه تاریخی", month_names_en[best_m],
                      delta=f"+{avg_by_month[best_m]:.2f}%")
            st.caption(f"در {pos_by_month[best_m]:.0f}% سال‌ها بازده مثبت داشته")
        with cb:
            worst_m = int(avg_by_month.idxmin())
            st.metric(f"بدترین ماه تاریخی", month_names_en[worst_m],
                      delta=f"{avg_by_month[worst_m]:.2f}%", delta_color="inverse")
            st.caption(f"در {pos_by_month[worst_m]:.0f}% سال‌ها بازده مثبت داشته")
    else:
        st.info("ابتدا داده پرتفوی را دانلود کنید.")



if saved_f:
    parts = []
    if saved_f.get("expected_return",0)>0: parts.append(f"Target {saved_f['expected_return']:.0f}%")
    if saved_f.get("risk_geo",0)>0:        parts.append(f"Geo {saved_f['risk_geo']}%")
    if saved_f.get("risk_mon",0)>0:        parts.append(f"Mon {saved_f['risk_mon']}%")
    if saved_f.get("risk_sys",0)>0:        parts.append(f"Sys {saved_f['risk_sys']}%")
    if parts:
        risk_summary = " · " + " · ".join(parts)

st.markdown(f"""
<div style="text-align:center; color:var(--muted); font-size:0.6rem;
    letter-spacing:0.15em; text-transform:uppercase; padding:0.4rem;
    font-family:'JetBrains Mono',monospace;">
    Portfolio360 Blueprint · Data via Yahoo Finance · {'Dark' if is_dark else 'Light'} Mode{risk_summary}
</div>
""", unsafe_allow_html=True)
