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


tab1, tab2, tab3, tab4, tab_ef, tab5, tab6, tab7, tab8, tab9 = st.tabs([
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
        st.caption("عملکرد پرتفوی در پنجره‌های زمانی متحرک — برای شناسایی دوره‌های ضعف یا قدرت.")

        roll_window = st.slider("پنجر