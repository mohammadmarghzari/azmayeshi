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


tab1, tab2, tab3, tab4, tab_ef, tab5, tab6, tab7, tab8, tab9, tab_live, tab_save, tab_alert, tab_iran, tab_bourse, tab_ime = st.tabs([
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
    "💾 ذخیره پرتفوی",
    "🔔 هشدار",
    "🇮🇷 ابزار ایران",
    "🏛 اختیار بورس کالا",
    "📡 IME Live — گواهی کالا",
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

        # Treemap
        st.markdown('<div class="bp-section"><span class="bp-section-text">Treemap — نمایش بصری وزن‌ها</span></div>', unsafe_allow_html=True)
        fig_tree = go.Figure(go.Treemap(
            labels=df_w["نماد"].tolist(),
            parents=["پرتفوی"] * len(df_w),
            values=df_w["وزن (%)"].tolist(),
            marker=dict(
                colors=COLORS[:len(df_w)],
                line=dict(width=2, color="#161616" if is_dark else "#e8e8e4"),
            ),
            textinfo="label+value+percent root",
            textfont=dict(size=12, family="JetBrains Mono"),
            hovertemplate="<b>%{label}</b><br>وزن: %{value:.2f}%<extra></extra>",
        ))
        fig_tree.update_layout(**get_plot_layout(title="TREEMAP — PORTFOLIO WEIGHTS", h=380))
        st.plotly_chart(fig_tree, use_container_width=True)

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

        # Underwater Chart
        st.markdown('<div class="bp-section"><span class="bp-section-text">Underwater Chart — مدت زمان زیر Peak</span></div>', unsafe_allow_html=True)
        st.caption("هر بار که پرتفوی زیر بالاترین نقطه خودش بوده را نشان می‌دهد. هرچه طولانی‌تر، ریکاوری سخت‌تر.")
        dd_pct = dd.values * 100
        dates_uw = prices.index[-len(dd_pct):]
        fig_uw = go.Figure()
        fig_uw.add_trace(go.Scatter(
            x=dates_uw, y=dd_pct,
            mode="lines", name="Underwater",
            line=dict(color="#9b72c8", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(155,114,200,0.12)" if is_dark else "rgba(100,60,160,0.08)",
        ))
        # خط -10% و -20%
        for lvl, clr in [(-10, "#e8a838"), (-20, "#cc5555")]:
            if dd_pct.min() < lvl:
                fig_uw.add_hline(y=lvl, line_dash="dot", line_color=clr, line_width=1,
                                  annotation_text=f"{lvl}%", annotation_font_color=clr, annotation_font_size=9)
        fig_uw.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
        # طولانی‌ترین دوره زیر peak
        in_dd = dd_pct < -1
        max_streak = cur_streak = 0
        for v in in_dd:
            cur_streak = cur_streak + 1 if v else 0
            max_streak = max(max_streak, cur_streak)
        fig_uw.update_layout(**get_plot_layout("UNDERWATER CHART — زیر Peak", "DATE", "افت از Peak (%)", 320))
        st.plotly_chart(fig_uw, use_container_width=True)
        ca, cb = st.columns(2)
        with ca:
            st.metric("بدترین افت", f"{dd_pct.min():.2f}%")
        with cb:
            st.metric("طولانی‌ترین دوره زیر Peak", f"{max_streak} روز")


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



# ════════════════════════════════════════════════════════════════════
# TAB SAVE — ذخیره و مقایسه پرتفوی‌ها
# ════════════════════════════════════════════════════════════════════
with tab_save:
    green_sv = "#5aaa78" if is_dark else "#1a6640"
    gold_sv  = "#c8a84b" if is_dark else "#8a6a1a"
    red_sv   = "#cc5555" if is_dark else "#8a2020"
    blue_sv  = "#5b9bd5"
    muted_sv = "#888888" if is_dark else "#555550"
    card_sv  = "#222222" if is_dark else "#e4e4e0"
    border_sv= "rgba(180,180,180,0.1)" if is_dark else "rgba(60,60,55,0.15)"

    st.markdown('<div class="bp-section"><span class="bp-section-text">ذخیره پرتفوی فعلی</span></div>', unsafe_allow_html=True)

    if "saved_portfolios" not in st.session_state:
        st.session_state["saved_portfolios"] = {}

    w_sv = st.session_state.get("weights")
    m_sv = st.session_state.get("metrics")

    if w_sv is not None and m_sv is not None:
        col_name, col_btn = st.columns([3, 1])
        with col_name:
            port_name = st.text_input("نام پرتفوی", placeholder="مثلاً: پرتفوی محافظه‌کار ۱۴۰۳", key="port_name_inp")
        with col_btn:
            st.markdown("<div style='padding-top:1.7rem'>", unsafe_allow_html=True)
            save_btn = st.button("💾 ذخیره", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if save_btn:
            if not port_name.strip():
                st.warning("یک نام برای پرتفوی وارد کنید.")
            else:
                st.session_state["saved_portfolios"][port_name.strip()] = {
                    "weights": w_sv.copy(),
                    "metrics": dict(m_sv),
                    "assets": list(asset_names),
                    "style": st.session_state.get("style_label", ""),
                    "risks": dict(st.session_state.get("saved_risks", {})),
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.success(f"پرتفوی «{port_name.strip()}» ذخیره شد.")
    else:
        st.info("ابتدا پرتفوی را محاسبه کنید تا بتوانید آن را ذخیره کنید.")

    # لیست پرتفوی‌های ذخیره‌شده
    saved_ports = st.session_state.get("saved_portfolios", {})
    if saved_ports:
        st.markdown('<div class="bp-section"><span class="bp-section-text">پرتفوی‌های ذخیره‌شده</span></div>', unsafe_allow_html=True)

        for pname, pdata in list(saved_ports.items()):
            pm = pdata["metrics"]
            pr = pdata.get("risks", {})
            with st.expander(f"📁 {pname}  ·  {pdata['date']}  ·  {pdata['style']}", expanded=False):
                ca, cb, cc, cd = st.columns(4)
                with ca:
                    st.metric("بازده تعدیل‌شده", f"{pm.get('بازده تعدیل‌شده ریسک',0)*100:.2f}%")
                with cb:
                    st.metric("نوسان", f"{pm.get('نوسان سالانه',0)*100:.2f}%")
                with cc:
                    st.metric("شارپ", f"{pm.get('نسبت شارپ',0):.3f}")
                with cd:
                    st.metric("MDD", f"{pm.get('حداکثر افت (Max Drawdown)',0)*100:.2f}%")

                # وزن‌ها
                w_list = pdata["weights"]
                a_list = pdata["assets"]
                df_sv = pd.DataFrame({"نماد": a_list, "وزن (%)": np.round(w_list*100, 2)})
                df_sv = df_sv.sort_values("وزن (%)", ascending=False)
                fig_sv = go.Figure(go.Pie(
                    labels=df_sv["نماد"], values=df_sv["وزن (%)"], hole=0.4,
                    marker=dict(colors=COLORS[:len(df_sv)],
                                line=dict(color="#161616" if is_dark else "#e8e8e4", width=2)),
                    textfont=dict(size=8, family="JetBrains Mono"), textinfo="percent+label",
                ))
                fig_sv.update_layout(**get_plot_layout(title=pname, h=280))
                st.plotly_chart(fig_sv, use_container_width=True)

                if st.button(f"🗑 حذف «{pname}»", key=f"del_{pname}"):
                    del st.session_state["saved_portfolios"][pname]
                    st.rerun()

        # مقایسه همه پرتفوی‌های ذخیره‌شده
        if len(saved_ports) >= 2:
            st.markdown('<div class="bp-section"><span class="bp-section-text">مقایسه پرتفوی‌های ذخیره‌شده</span></div>', unsafe_allow_html=True)
            rows_cmp = []
            for pname, pdata in saved_ports.items():
                pm = pdata["metrics"]
                rows_cmp.append({
                    "نام": pname,
                    "سبک": pdata.get("style",""),
                    "بازده تعدیل‌شده (%)": round(pm.get("بازده تعدیل‌شده ریسک",0)*100, 2),
                    "نوسان (%)": round(pm.get("نوسان سالانه",0)*100, 2),
                    "شارپ": round(pm.get("نسبت شارپ",0), 3),
                    "MDD (%)": round(pm.get("حداکثر افت (Max Drawdown)",0)*100, 2),
                    "CVaR (%)": round(pm.get("CVaR 95%",0)*100, 2),
                    "کالمار": round(pm.get("نسبت کالمار",0), 3),
                })
            df_cmp_sv = pd.DataFrame(rows_cmp)
            st.dataframe(df_cmp_sv, use_container_width=True, hide_index=True)

            # نمودار مقایسه
            fig_cmp_sv = go.Figure()
            metrics_cmp = ["بازده تعدیل‌شده (%)", "نوسان (%)", "شارپ", "MDD (%)"]
            for i, metric in enumerate(metrics_cmp):
                fig_cmp_sv.add_trace(go.Bar(
                    name=metric,
                    x=[r["نام"] for r in rows_cmp],
                    y=[r[metric] for r in rows_cmp],
                    marker_color=COLORS[i],
                ))
            fig_cmp_sv.update_layout(**get_plot_layout("PORTFOLIO COMPARISON", "", "مقدار", 400))
            fig_cmp_sv.update_layout(barmode="group")
            st.plotly_chart(fig_cmp_sv, use_container_width=True)
    else:
        st.info("هنوز پرتفویی ذخیره نشده. پرتفوی را محاسبه کنید و سپس ذخیره کنید.")


# ════════════════════════════════════════════════════════════════════
# TAB ALERT — هشدار قیمت و Fear & Greed
# ════════════════════════════════════════════════════════════════════
with tab_alert:
    green_al = "#5aaa78" if is_dark else "#1a6640"
    gold_al  = "#c8a84b" if is_dark else "#8a6a1a"
    red_al   = "#cc5555" if is_dark else "#8a2020"
    blue_al  = "#5b9bd5"
    muted_al = "#888888" if is_dark else "#555550"
    card_al  = "#222222" if is_dark else "#e4e4e0"
    border_al= "rgba(180,180,180,0.1)" if is_dark else "rgba(60,60,55,0.15)"

    if "alerts" not in st.session_state:
        st.session_state["alerts"] = []

    # ── هشدار قیمت ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">هشدار قیمت</span></div>', unsafe_allow_html=True)
    st.caption("قیمت فعلی نمادها را با آستانه‌های تعریف‌شده مقایسه می‌کند.")

    all_syms = [s for cat in SYMBOLS.values() for s in cat.keys()]
    ca, cb, cc, cd = st.columns([2, 1, 1, 1])
    with ca:
        alert_sym = st.selectbox("نماد", all_syms, key="al_sym")
    with cb:
        alert_type = st.selectbox("نوع", ["بالاتر از", "پایین‌تر از"], key="al_type")
    with cc:
        alert_price = st.number_input("قیمت ($)", min_value=0.0, value=100.0, step=1.0, key="al_price")
    with cd:
        st.markdown("<div style='padding-top:1.7rem'>", unsafe_allow_html=True)
        add_alert_btn = st.button("➕ افزودن هشدار", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if add_alert_btn:
        st.session_state["alerts"].append({
            "sym": alert_sym, "type": alert_type,
            "price": alert_price, "triggered": False,
        })
        st.success(f"هشدار برای {alert_sym} {alert_type} ${alert_price:.2f} اضافه شد.")

    # بررسی هشدارها
    alerts = st.session_state.get("alerts", [])
    if alerts:
        st.markdown('<div class="bp-section"><span class="bp-section-text">بررسی هشدارها</span></div>', unsafe_allow_html=True)
        check_btn = st.button("🔄 بررسی قیمت‌های فعلی", use_container_width=True)

        if check_btn:
            with st.spinner("در حال دریافت قیمت‌ها..."):
                triggered_any = False
                for al in st.session_state["alerts"]:
                    try:
                        ticker_obj = yf.Ticker(al["sym"])
                        hist = ticker_obj.history(period="1d", interval="1m")
                        if len(hist) > 0:
                            cur_price = float(hist["Close"].iloc[-1])
                        else:
                            hist2 = ticker_obj.history(period="5d")
                            cur_price = float(hist2["Close"].iloc[-1])
                        al["current"] = cur_price
                        if al["type"] == "بالاتر از":
                            al["triggered"] = cur_price > al["price"]
                        else:
                            al["triggered"] = cur_price < al["price"]
                        if al["triggered"]:
                            triggered_any = True
                    except Exception:
                        al["current"] = None
                if triggered_any:
                    st.warning("⚠ برخی هشدارها فعال شدند!")
                else:
                    st.success("✓ هیچ هشداری فعال نشده.")

        for i, al in enumerate(st.session_state["alerts"]):
            cur = al.get("current")
            triggered = al.get("triggered", False)
            border_color = red_al if triggered else (green_al if cur is not None else muted_al)
            cur_str = f"${cur:.4f}" if cur is not None else "—"
            diff_str = ""
            if cur is not None:
                diff = ((cur - al["price"]) / al["price"]) * 100
                diff_str = f"  ({diff:+.2f}%)"

            st.markdown(f"""
            <div style="padding:0.6rem 1rem;margin-bottom:0.4rem;
                background:{card_al};border:1px solid {border_color};
                border-left:4px solid {border_color};border-radius:4px;
                display:flex;align-items:center;justify-content:space-between;">
                <div>
                    <span style="font-family:'JetBrains Mono',monospace;font-weight:700;
                        font-size:0.85rem;color:{'#c0c0c0' if is_dark else '#222'}">
                        {al['sym']}
                    </span>
                    <span style="font-size:0.75rem;color:{muted_al};margin-right:8px">
                        {al['type']} ${al['price']:.2f}
                    </span>
                </div>
                <div style="text-align:right">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                        font-weight:700;color:{border_color}">
                        {'🔔 فعال' if triggered else ('✓' if cur is not None else '—')}
                        &nbsp;{cur_str}{diff_str}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"🗑 حذف", key=f"del_al_{i}"):
                st.session_state["alerts"].pop(i)
                st.rerun()

    else:
        st.info("هنوز هشداری تعریف نشده. نماد و قیمت آستانه را وارد کنید.")

    # ── هشدار Fear & Greed ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">هشدار Fear & Greed</span></div>', unsafe_allow_html=True)
    st.caption("اگر شاخص ترس و طمع از آستانه‌های تعریف‌شده رد شود، هشدار نمایش می‌دهد.")

    if "fg_alerts" not in st.session_state:
        st.session_state["fg_alerts"] = {"lower": 20, "upper": 80, "active": False}

    fg_al = st.session_state["fg_alerts"]
    c1, c2 = st.columns(2)
    with c1:
        fg_lower = st.slider("هشدار ترس شدید (زیر این مقدار)", 0, 50, fg_al["lower"], key="fg_lower")
    with c2:
        fg_upper = st.slider("هشدار طمع شدید (بالای این مقدار)", 50, 100, fg_al["upper"], key="fg_upper")

    st.session_state["fg_alerts"]["lower"] = fg_lower
    st.session_state["fg_alerts"]["upper"] = fg_upper

    if st.button("🔄 بررسی Fear & Greed اکنون", use_container_width=True):
        fg_now = fetch_fear_greed()
        if fg_now["ok"]:
            score_now = fg_now["score"]
            st.session_state["fg_alerts"]["last_score"] = score_now
            if score_now <= fg_lower:
                st.error(f"😱 هشدار ترس شدید! شاخص = {score_now:.0f} (زیر آستانه {fg_lower})")
                st.caption("تاریخاً این سطح فرصت خرید بوده — با احتیاط تصمیم بگیرید.")
            elif score_now >= fg_upper:
                st.warning(f"🤑 هشدار طمع شدید! شاخص = {score_now:.0f} (بالای آستانه {fg_upper})")
                st.caption("تاریخاً این سطح نشانه اشباع خرید بوده — احتیاط در ورود جدید.")
            else:
                st.success(f"✓ شاخص در محدوده عادی: {score_now:.0f}")
        else:
            st.error("دریافت شاخص ناموفق بود.")

    last_score = st.session_state["fg_alerts"].get("last_score")
    if last_score is not None:
        st.markdown(f"""
        <div style="margin-top:0.5rem;padding:0.5rem 0.9rem;background:{card_al};
            border:1px solid {border_al};border-radius:4px;
            font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:{muted_al}">
            آخرین بررسی: <strong style="color:{'#c0c0c0' if is_dark else '#222'}">{last_score:.0f}</strong>
            &nbsp;·&nbsp; آستانه ترس: {fg_lower} &nbsp;·&nbsp; آستانه طمع: {fg_upper}
        </div>
        """, unsafe_allow_html=True)



risk_summary = ""
# ════════════════════════════════════════════════════════════════════
# TAB IRAN — ابزار ایران
# ════════════════════════════════════════════════════════════════════
with tab_iran:
    green_ir = "#5aaa78" if is_dark else "#1a6640"
    gold_ir  = "#c8a84b" if is_dark else "#8a6a1a"
    red_ir   = "#cc5555" if is_dark else "#8a2020"
    blue_ir  = "#5b9bd5"
    muted_ir = "#888888" if is_dark else "#555550"
    card_ir  = "#222222" if is_dark else "#e4e4e0"
    border_ir= "rgba(180,180,180,0.1)" if is_dark else "rgba(60,60,55,0.15)"

    def fmt_toman(v):
        """فرمت عدد به تومان با جداکننده هزار"""
        return f"{v:,.0f} تومان"

    st.markdown('<div class="bp-section"><span class="bp-section-text">قیمت روز دلار بازار</span></div>', unsafe_allow_html=True)
    st.caption("قیمت دلار را به صورت دستی وارد کنید تا با قیمت واقعی مقایسه شود.")

    market_dollar = st.number_input(
        "قیمت دلار بازار آزاد (تومان)",
        min_value=1000, max_value=500000,
        value=90000, step=500,
        help="قیمتی که الان در بازار آزاد معامله می‌شود",
        key="ir_market_dollar"
    )

    st.markdown("---")

    # ════ روش ۱ — تفاضل تورم ════
    st.markdown('<div class="bp-section"><span class="bp-section-text">روش ۱ — محاسبه به کمک تورم</span></div>', unsafe_allow_html=True)
    st.caption("قیمت پایه دلار × (۱ + اختلاف تورم ایران و آمریکا) = قیمت واقعی تخمینی")

    c1, c2, c3 = st.columns(3)
    with c1:
        base_dollar_inf = st.number_input(
            "قیمت پایه دلار (تومان)",
            min_value=1000, max_value=500000,
            value=31000, step=500,
            help="قیمت دلار در نقطه شروع (مثلاً ابتدای سال یا نیمه اول)",
            key="ir_base_inf"
        )
    with c2:
        iran_inf = st.number_input(
            "تورم ایران (%)",
            min_value=0.0, max_value=500.0,
            value=48.0, step=1.0,
            key="ir_iran_inf"
        )
    with c3:
        us_inf = st.number_input(
            "تورم آمریکا (%)",
            min_value=0.0, max_value=50.0,
            value=7.0, step=0.5,
            key="ir_us_inf"
        )

    inf_diff = iran_inf - us_inf
    real_dollar_inf = base_dollar_inf * (1 + inf_diff / 100)
    gap_inf_toman = real_dollar_inf - market_dollar
    gap_inf_pct   = (gap_inf_toman / market_dollar) * 100

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div style="background:{card_ir};border:1px solid {border_ir};
            border-top:3px solid {gold_ir};border-radius:4px;padding:1rem 1.1rem;">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;
                color:{muted_ir};text-transform:uppercase;margin-bottom:0.4rem">
                اختلاف تورم
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                font-weight:700;color:{gold_ir}">
                {inf_diff:.1f}%
            </div>
            <div style="font-size:0.65rem;color:{muted_ir};margin-top:0.3rem">
                {iran_inf:.0f}% - {us_inf:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div style="background:{card_ir};border:1px solid {border_ir};
            border-top:3px solid {blue_ir};border-radius:4px;padding:1rem 1.1rem;">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;
                color:{muted_ir};text-transform:uppercase;margin-bottom:0.4rem">
                قیمت واقعی تخمینی
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                font-weight:700;color:{blue_ir}">
                {fmt_toman(real_dollar_inf)}
            </div>
            <div style="font-size:0.65rem;color:{muted_ir};margin-top:0.3rem">
                بر اساس تفاضل تورم
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        gap_color = green_ir if gap_inf_toman > 0 else red_ir
        gap_label = "دلار ارزان‌تر از واقعی است" if gap_inf_toman > 0 else "دلار گران‌تر از واقعی است"
        st.markdown(f"""
        <div style="background:{card_ir};border:1px solid {border_ir};
            border-top:3px solid {gap_color};border-radius:4px;padding:1rem 1.1rem;">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;
                color:{muted_ir};text-transform:uppercase;margin-bottom:0.4rem">
                فاصله قیمتی
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                font-weight:700;color:{gap_color}">
                {fmt_toman(abs(gap_inf_toman))}
            </div>
            <div style="font-size:0.65rem;color:{gap_color};margin-top:0.3rem;font-weight:600">
                {gap_inf_pct:+.1f}% &nbsp;·&nbsp; {gap_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # نمودار مقایسه روش ۱
    fig_inf = go.Figure()
    cats_inf = ["دلار بازار", "قیمت واقعی (تورم)"]
    vals_inf = [market_dollar, real_dollar_inf]
    colors_inf = [muted_ir, blue_ir]
    fig_inf.add_trace(go.Bar(
        x=cats_inf, y=vals_inf,
        marker_color=colors_inf,
        marker_line=dict(color="rgba(0,0,0,0.15)", width=0.5),
        text=[fmt_toman(v) for v in vals_inf],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono"),
    ))
    fig_inf.add_hline(y=market_dollar, line_dash="dot",
                      line_color=muted_ir, line_width=1.2,
                      annotation_text="قیمت بازار", annotation_font_color=muted_ir, annotation_font_size=9)
    fig_inf.update_layout(**get_plot_layout("مقایسه قیمت بازار با قیمت واقعی (روش تورم)", "", "تومان", 320))
    fig_inf.update_layout(showlegend=False, yaxis=dict(tickformat=",.0f"))
    st.plotly_chart(fig_inf, use_container_width=True)

    st.markdown("---")

    # ════ روش ۲ — قیمت طلا ════
    st.markdown('<div class="bp-section"><span class="bp-section-text">روش ۲ — محاسبه به کمک قیمت طلا</span></div>', unsafe_allow_html=True)
    st.caption("قیمت سکه بهار آزادی ÷ (انس جهانی × ۴) = قیمت واقعی دلار")

    # دریافت خودکار قیمت انس طلا از yfinance
    @st.cache_data(show_spinner=False, ttl=1800)
    def fetch_gold_price():
        try:
            df = yf.Ticker("GC=F").history(period="2d", auto_adjust=True)
            if len(df) > 0:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass
        return None

    gold_auto = fetch_gold_price()

    c1, c2 = st.columns(2)
    with c1:
        sekke_price = st.number_input(
            "قیمت سکه تمام بهار آزادی (تومان)",
            min_value=100000, max_value=100000000,
            value=20400000, step=100000,
            help="قیمت لحظه‌ای سکه تمام بهار آزادی در بازار",
            key="ir_sekke"
        )
    with c2:
        gold_default = int(gold_auto) if gold_auto else 1866
        gold_oz = st.number_input(
            "قیمت انس جهانی طلا (دلار)",
            min_value=100, max_value=10000,
            value=gold_default, step=1,
            help="قیمت هر اونس طلا در بازار جهانی — از yfinance دریافت شده",
            key="ir_gold_oz"
        )
        if gold_auto:
            st.caption(f"✓ قیمت خودکار از yfinance: ${gold_auto:,.0f}")

    real_dollar_gold = sekke_price / (gold_oz * 4)
    gap_gold_toman = real_dollar_gold - market_dollar
    gap_gold_pct   = (gap_gold_toman / market_dollar) * 100

    col_a2, col_b2, col_c2 = st.columns(3)
    with col_a2:
        st.markdown(f"""
        <div style="background:{card_ir};border:1px solid {border_ir};
            border-top:3px solid {gold_ir};border-radius:4px;padding:1rem 1.1rem;">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;
                color:{muted_ir};text-transform:uppercase;margin-bottom:0.4rem">
                فرمول
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;
                font-weight:600;color:{gold_ir};line-height:1.7">
                {fmt_toman(sekke_price)}<br>
                ÷ ({gold_oz:,} × ۴)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b2:
        st.markdown(f"""
        <div style="background:{card_ir};border:1px solid {border_ir};
            border-top:3px solid {blue_ir};border-radius:4px;padding:1rem 1.1rem;">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;
                color:{muted_ir};text-transform:uppercase;margin-bottom:0.4rem">
                قیمت واقعی تخمینی
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                font-weight:700;color:{blue_ir}">
                {fmt_toman(real_dollar_gold)}
            </div>
            <div style="font-size:0.65rem;color:{muted_ir};margin-top:0.3rem">
                بر اساس قیمت طلا
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_c2:
        gap_color2 = green_ir if gap_gold_toman > 0 else red_ir
        gap_label2 = "دلار ارزان‌تر از واقعی است" if gap_gold_toman > 0 else "دلار گران‌تر از واقعی است"
        st.markdown(f"""
        <div style="background:{card_ir};border:1px solid {border_ir};
            border-top:3px solid {gap_color2};border-radius:4px;padding:1rem 1.1rem;">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;
                color:{muted_ir};text-transform:uppercase;margin-bottom:0.4rem">
                فاصله قیمتی
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                font-weight:700;color:{gap_color2}">
                {fmt_toman(abs(gap_gold_toman))}
            </div>
            <div style="font-size:0.65rem;color:{gap_color2};margin-top:0.3rem;font-weight:600">
                {gap_gold_pct:+.1f}% &nbsp;·&nbsp; {gap_label2}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # نمودار مقایسه هر دو روش با بازار
    st.markdown('<div class="bp-section"><span class="bp-section-text">مقایسه هر دو روش با قیمت بازار</span></div>', unsafe_allow_html=True)

    fig_cmp_ir = go.Figure()
    cats_all = ["دلار بازار آزاد", "روش تورم", "روش طلا", "میانگین دو روش"]
    avg_real = (real_dollar_inf + real_dollar_gold) / 2
    vals_all = [market_dollar, real_dollar_inf, real_dollar_gold, avg_real]
    clrs_all = [muted_ir, blue_ir, gold_ir, green_ir]

    fig_cmp_ir.add_trace(go.Bar(
        x=cats_all, y=vals_all,
        marker_color=clrs_all,
        marker_line=dict(color="rgba(0,0,0,0.15)", width=0.5),
        text=[fmt_toman(v) for v in vals_all],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono"),
    ))
    fig_cmp_ir.add_hline(y=market_dollar, line_dash="dot",
                          line_color=muted_ir, line_width=1.2,
                          annotation_text="قیمت بازار", annotation_font_color=muted_ir, annotation_font_size=9)
    fig_cmp_ir.update_layout(**get_plot_layout("مقایسه قیمت بازار با قیمت‌های واقعی تخمینی", "", "تومان", 380))
    fig_cmp_ir.update_layout(showlegend=False, yaxis=dict(tickformat=",.0f"))
    st.plotly_chart(fig_cmp_ir, use_container_width=True)

    # جدول خلاصه
    st.markdown('<div class="bp-section"><span class="bp-section-text">خلاصه مقایسه</span></div>', unsafe_allow_html=True)
    avg_gap_toman = avg_real - market_dollar
    avg_gap_pct   = (avg_gap_toman / market_dollar) * 100

    rows_ir = [
        {"روش": "قیمت بازار آزاد",    "قیمت (تومان)": fmt_toman(market_dollar),   "فاصله (تومان)": "—",                              "فاصله (%)": "—"},
        {"روش": "روش تورم",            "قیمت (تومان)": fmt_toman(real_dollar_inf),  "فاصله (تومان)": fmt_toman(abs(gap_inf_toman)),    "فاصله (%)": f"{gap_inf_pct:+.1f}%"},
        {"روش": "روش طلا",             "قیمت (تومان)": fmt_toman(real_dollar_gold), "فاصله (تومان)": fmt_toman(abs(gap_gold_toman)),   "فاصله (%)": f"{gap_gold_pct:+.1f}%"},
        {"روش": "میانگین دو روش",      "قیمت (تومان)": fmt_toman(avg_real),         "فاصله (تومان)": fmt_toman(abs(avg_gap_toman)),    "فاصله (%)": f"{avg_gap_pct:+.1f}%"},
    ]
    st.dataframe(pd.DataFrame(rows_ir), use_container_width=True, hide_index=True)

    # نتیجه‌گیری
    avg_color = green_ir if avg_gap_toman > 0 else red_ir
    avg_verdict = "دلار در بازار ارزان‌تر از ارزش واقعی تخمینی معامله می‌شود." if avg_gap_toman > 0 \
                  else "دلار در بازار گران‌تر از ارزش واقعی تخمینی معامله می‌شود."
    st.markdown(f"""
    <div style="margin-top:0.8rem;padding:0.9rem 1.1rem;background:{card_ir};
        border:1px solid {avg_color};border-left:4px solid {avg_color};border-radius:4px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
            font-weight:700;color:{avg_color};letter-spacing:0.1em;margin-bottom:0.35rem">
            نتیجه میانگین دو روش
        </div>
        <div style="font-size:0.82rem;color:{'#c0c0c0' if is_dark else '#333'};line-height:1.7">
            میانگین قیمت واقعی تخمینی: <strong style="color:{avg_color}">{fmt_toman(avg_real)}</strong><br>
            فاصله با بازار: <strong style="color:{avg_color}">{fmt_toman(abs(avg_gap_toman))} ({avg_gap_pct:+.1f}%)</strong><br>
            {avg_verdict}
        </div>
        <div style="margin-top:0.5rem;font-size:0.62rem;color:{muted_ir};border-top:1px solid rgba(128,128,128,0.2);padding-top:0.4rem">
            ⚠ این محاسبات تخمینی هستند و برای تصمیم‌گیری مالی باید با سایر عوامل ترکیب شوند.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # بخش بورس کالای ایران + محاسبه حباب
    # ════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="bp-section"><span class="bp-section-text">🏭 گواهی سپرده کالایی — بورس کالای ایران</span></div>', unsafe_allow_html=True)
    st.caption("قیمت گواهی را از کارگزاری وارد کنید — قیمت جهانی از yfinance دریافت می‌شود — حباب به صورت خودکار محاسبه می‌گردد.")

    # ── دریافت قیمت‌های جهانی از yfinance ──
    @st.cache_data(show_spinner=False, ttl=1800)
    def fetch_world_prices():
        tickers = {"GC=F": "طلا", "SI=F": "نقره", "HG=F": "مس", "ZN=F": "روی"}
        out = {}
        for sym, name in tickers.items():
            try:
                df = yf.Ticker(sym).history(period="2d", auto_adjust=True)
                if len(df) > 0:
                    out[sym] = float(df["Close"].iloc[-1])
            except Exception:
                pass
        return out

    world_px = fetch_world_prices()
    gold_oz   = world_px.get("GC=F")   # دلار / اونس
    silver_oz = world_px.get("SI=F")   # دلار / اونس
    copper_lb = world_px.get("HG=F")   # دلار / پوند  (HG=F بر پوند است)
    zinc_usd  = world_px.get("ZN=F")   # دلار / تن (LME)

    # نمایش قیمت‌های جهانی
    st.markdown('<div class="bp-section"><span class="bp-section-text">قیمت‌های جهانی (yfinance — زنده)</span></div>', unsafe_allow_html=True)
    wc1, wc2, wc3, wc4 = st.columns(4)
    def _world_card(label, val, unit, color, key_manual, default_manual):
        if val:
            st.markdown(f"""
            <div style="background:{card_ir};border:1px solid {border_ir};
                border-top:3px solid {color};border-radius:4px;padding:0.75rem 0.9rem;margin-bottom:0.4rem">
                <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.12em;
                    color:{muted_ir};text-transform:uppercase;margin-bottom:0.3rem">{label}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                    font-weight:700;color:{color}">${val:,.2f}</div>
                <div style="font-size:0.6rem;color:{muted_ir};margin-top:0.2rem">{unit} · از yfinance</div>
            </div>
            """, unsafe_allow_html=True)
            return val
        else:
            return st.number_input(f"{label} ({unit}) — دستی", min_value=0.0,
                                   value=float(default_manual), step=1.0, key=key_manual)

    with wc1:
        gold_oz_use   = _world_card("طلا (Gold)", gold_oz, "$/اونس", gold_ir, "ir_gold_manual", 2350)
    with wc2:
        silver_oz_use = _world_card("نقره (Silver)", silver_oz, "$/اونس", "#aaaaaa", "ir_silver_manual", 30)
    with wc3:
        # HG=F بر پوند است → تبدیل به کیلو: ×2.20462
        copper_kg_use_raw = _world_card("مس (Copper)", copper_lb, "$/پوند", "#cd7f32", "ir_copper_manual", 4.5)
        copper_kg_use = copper_kg_use_raw * 2.20462  # تبدیل پوند → کیلو
    with wc4:
        # ZN=F بر تن متریک (LME) — تبدیل به کیلو: ÷1000
        zinc_t_use_raw = _world_card("روی (Zinc)", zinc_usd, "$/تن", "#8899aa", "ir_zinc_manual", 2800)
        zinc_kg_use = zinc_t_use_raw / 1000  # تبدیل تن → کیلو

    # ── فرمول‌های محاسبه قیمت واقعی (بدون حباب) ──
    # GoldBar:  اونس × دلار ÷ 31.1035  = ریال/گرم
    # GoldCoin: اونس × دلار ÷ 31.1035 × 8.133  = ریال/سکه  (وزن خالص سکه بهار)
    # SilverBar: اونس نقره × دلار ÷ 31.1035 = ریال/گرم
    # CopperCthd: قیمت مس $/kg × دلار = ریال/کیلو
    # ZincIngot: قیمت روی $/kg × دلار = ریال/کیلو
    # SteelRebar / Bitumen: قیمت جهانی پایدار ندارند → حباب محاسبه نمی‌شود
    TROY_OZ = 31.1035
    SEKKE_WEIGHT = 8.133  # گرم طلای خالص در سکه بهار آزادی طرح جدید

    def fair_price(kala_key, dollar):
        """قیمت منصفانه هر گواهی بر اساس قیمت جهانی × نرخ دلار"""
        if kala_key == "GoldBar":
            return (gold_oz_use * dollar) / TROY_OZ if gold_oz_use else None
        elif kala_key == "GoldCoin":
            return (gold_oz_use * dollar / TROY_OZ) * SEKKE_WEIGHT if gold_oz_use else None
        elif kala_key == "SilverBar":
            return (silver_oz_use * dollar) / TROY_OZ if silver_oz_use else None
        elif kala_key == "CopperCthd":
            return copper_kg_use * dollar if copper_kg_use else None
        elif kala_key == "ZincIngot":
            return zinc_kg_use * dollar if zinc_kg_use else None
        else:
            return None  # SteelRebar, Bitumen — حباب ندارد

    # داده‌های گواهی سپرده
    IRAN_COMMODITIES = {
        "GoldBar":    {"label": "GoldBar — شمش طلا 995+",    "unit": "گرم",    "default_price": 20964990,   "category": "فلزات گرانبها", "icon": "🥇"},
        "GoldCoin":   {"label": "GoldCoin — سکه بهار آزادی", "unit": "سکه",    "default_price": 1612393490, "category": "فلزات گرانبها", "icon": "🪙"},
        "SilverBar":  {"label": "SilverBar — شمش نقره 999.9","unit": "گرم",    "default_price": 3748900,    "category": "فلزات گرانبها", "icon": "🥈"},
        "CopperCthd": {"label": "CopperCthd — مس کاتد",      "unit": "کیلوگرم","default_price": 22200000,   "category": "فلزات صنعتی",  "icon": "🟠"},
        "ZincIngot":  {"label": "ZincIngot — شمش روی",       "unit": "کیلوگرم","default_price": 4790100,    "category": "فلزات صنعتی",  "icon": "🔘"},
        "SteelRebar": {"label": "SteelRebar — میلگرد",       "unit": "کیلوگرم","default_price": 580000,     "category": "فلزات صنعتی",  "icon": "⚙"},
        "Bitumen":    {"label": "Bitumen — قیر",              "unit": "کیلوگرم","default_price": 751150,     "category": "نفت و گاز",    "icon": "🛢"},
    }

    # ── ورود قیمت بازار هر گواهی ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">قیمت بازار گواهی‌ها (از کارگزاری)</span></div>', unsafe_allow_html=True)
    st.caption("قیمت لحظه‌ای هر گواهی را از سایت کارگزاری وارد کنید.")

    kala_prices = {}
    cols_kala = st.columns(2)
    for i, (kala_key, kala_info) in enumerate(IRAN_COMMODITIES.items()):
        with cols_kala[i % 2]:
            kala_prices[kala_key] = st.number_input(
                f"{kala_info['icon']} {kala_info['label']} ({kala_info['unit']})",
                min_value=0,
                value=kala_info["default_price"],
                step=max(1, kala_info["default_price"] // 200),
                key=f"ir_kala_cur_{kala_key}",
            )

    # ── محاسبه حباب ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">محاسبه حباب گواهی‌ها</span></div>', unsafe_allow_html=True)
    st.caption("حباب = قیمت بازار گواهی − قیمت منصفانه بر اساس قیمت جهانی × نرخ دلار")

    rows_bubble = []
    bubble_names, bubble_pcts, bubble_colors = [], [], []

    for kala_key, kala_info in IRAN_COMMODITIES.items():
        market_p = kala_prices[kala_key]
        fair_p   = fair_price(kala_key, market_dollar)

        if fair_p and fair_p > 0:
            bubble_toman = market_p - fair_p
            bubble_pct   = (bubble_toman / fair_p) * 100
            bubble_label = "حباب مثبت (گران‌تر از ارزش ذاتی)" if bubble_pct > 0 else "حباب منفی (ارزان‌تر از ارزش ذاتی)"
            b_color      = red_ir if bubble_pct > 5 else (green_ir if bubble_pct < -5 else gold_ir)
            rows_bubble.append({
                "گواهی":              f"{kala_info['icon']} {kala_key}",
                "قیمت بازار":        fmt_toman(market_p),
                "قیمت منصفانه":      fmt_toman(fair_p),
                "حباب (تومان)":      fmt_toman(abs(bubble_toman)),
                "حباب (%)":          f"{bubble_pct:+.1f}%",
                "وضعیت":             bubble_label,
            })
            bubble_names.append(f"{kala_info['icon']} {kala_key}")
            bubble_pcts.append(bubble_pct)
            bubble_colors.append(b_color)
        else:
            rows_bubble.append({
                "گواهی":              f"{kala_info['icon']} {kala_key}",
                "قیمت بازار":        fmt_toman(market_p),
                "قیمت منصفانه":      "—",
                "حباب (تومان)":      "—",
                "حباب (%)":          "—",
                "وضعیت":             "قیمت جهانی مرجع ندارد",
            })

    st.dataframe(pd.DataFrame(rows_bubble), use_container_width=True, hide_index=True)

    # ── کارت‌های حباب برای هر گواهی ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">جزئیات حباب هر گواهی</span></div>', unsafe_allow_html=True)
    bubble_cols = st.columns(3)
    bubble_idx  = 0
    for kala_key, kala_info in IRAN_COMMODITIES.items():
        market_p = kala_prices[kala_key]
        fair_p   = fair_price(kala_key, market_dollar)
        if not fair_p:
            continue
        bubble_toman = market_p - fair_p
        bubble_pct   = (bubble_toman / fair_p) * 100
        b_color = red_ir if bubble_pct > 5 else (green_ir if bubble_pct < -5 else gold_ir)
        verdict = "گران‌تر از ارزش ذاتی" if bubble_pct > 5 else ("ارزان‌تر از ارزش ذاتی" if bubble_pct < -5 else "در محدوده منطقی")

        # فرمول نمایشی
        if kala_key == "GoldBar":
            formula = f"${gold_oz_use:,.0f} × {market_dollar:,} ÷ 31.10"
        elif kala_key == "GoldCoin":
            formula = f"${gold_oz_use:,.0f} × {market_dollar:,} ÷ 31.10 × 8.13"
        elif kala_key == "SilverBar":
            formula = f"${silver_oz_use:,.2f} × {market_dollar:,} ÷ 31.10"
        elif kala_key == "CopperCthd":
            formula = f"${copper_lb:.3f}/lb × 2.205 × {market_dollar:,}"
        elif kala_key == "ZincIngot":
            formula = f"${zinc_t_use_raw:,.0f}/t ÷ 1000 × {market_dollar:,}"
        else:
            formula = "—"

        with bubble_cols[bubble_idx % 3]:
            st.markdown(f"""
            <div style="background:{card_ir};border:1px solid {border_ir};
                border-top:3px solid {b_color};border-radius:4px;
                padding:0.9rem 1rem;margin-bottom:0.7rem">
                <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;
                    color:{muted_ir};text-transform:uppercase;margin-bottom:0.5rem">
                    {kala_info['icon']} {kala_key}
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">
                    <span style="font-size:0.68rem;color:{muted_ir}">قیمت بازار</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                        font-weight:700;color:{'#c0c0c0' if is_dark else '#222'}">{fmt_toman(market_p)}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">
                    <span style="font-size:0.68rem;color:{muted_ir}">قیمت منصفانه</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                        font-weight:700;color:{blue_ir}">{fmt_toman(fair_p)}</span>
                </div>
                <div style="border-top:1px solid {border_ir};margin:0.4rem 0"></div>
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:0.68rem;color:{muted_ir}">حباب</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                        font-weight:700;color:{b_color}">{bubble_pct:+.1f}%</span>
                </div>
                <div style="font-size:0.6rem;color:{b_color};font-weight:600;
                    margin-top:0.3rem;text-align:left">{verdict}</div>
                <div style="font-size:0.55rem;color:{muted_ir};margin-top:0.4rem;
                    font-family:'JetBrains Mono',monospace;border-top:1px solid {border_ir};
                    padding-top:0.3rem;direction:ltr;text-align:left">{formula}</div>
            </div>
            """, unsafe_allow_html=True)
        bubble_idx += 1

    # ── نمودار حباب ──
    if bubble_names:
        st.markdown('<div class="bp-section"><span class="bp-section-text">نمودار حباب گواهی‌ها</span></div>', unsafe_allow_html=True)
        fig_bubble = go.Figure()
        fig_bubble.add_trace(go.Bar(
            x=bubble_names,
            y=bubble_pcts,
            marker_color=bubble_colors,
            marker_line=dict(color="rgba(0,0,0,0.15)", width=0.5),
            text=[f"{p:+.1f}%" for p in bubble_pcts],
            textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono"),
        ))
        fig_bubble.add_hline(y=0, line_dash="solid", line_color=muted_ir, line_width=1.5,
                             annotation_text="قیمت منصفانه", annotation_font_color=muted_ir,
                             annotation_font_size=9)
        fig_bubble.add_hrect(y0=-5, y1=5, fillcolor="rgba(200,168,75,0.07)",
                             line_width=0, annotation_text="محدوده منطقی (±۵٪)",
                             annotation_font_size=8, annotation_font_color=gold_ir)
        fig_bubble.update_layout(**get_plot_layout("حباب گواهی‌های سپرده بورس کالا", "", "حباب (%)", 380))
        fig_bubble.update_layout(showlegend=False, xaxis=dict(tickangle=-20))
        st.plotly_chart(fig_bubble, use_container_width=True)

        # ── نمودار مقایسه قیمت بازار vs منصفانه ──
        st.markdown('<div class="bp-section"><span class="bp-section-text">مقایسه قیمت بازار با قیمت منصفانه</span></div>', unsafe_allow_html=True)
        fair_vals_chart   = []
        market_vals_chart = []
        chart_names       = []
        for kala_key, kala_info in IRAN_COMMODITIES.items():
            fp = fair_price(kala_key, market_dollar)
            if fp:
                chart_names.append(f"{kala_info['icon']} {kala_key}")
                market_vals_chart.append(kala_prices[kala_key])
                fair_vals_chart.append(fp)

        fig_cmp_bubble = go.Figure()
        fig_cmp_bubble.add_trace(go.Bar(
            name="قیمت بازار",
            x=chart_names, y=market_vals_chart,
            marker_color=blue_ir,
            marker_line=dict(color="rgba(0,0,0,0.1)", width=0.5),
        ))
        fig_cmp_bubble.add_trace(go.Bar(
            name="قیمت منصفانه (جهانی × دلار)",
            x=chart_names, y=fair_vals_chart,
            marker_color=gold_ir,
            marker_line=dict(color="rgba(0,0,0,0.1)", width=0.5),
        ))
        fig_cmp_bubble.update_layout(
            **get_plot_layout("قیمت بازار در مقابل قیمت منصفانه", "", "تومان", 400),
            barmode="group"
        )
        fig_cmp_bubble.update_layout(yaxis=dict(tickformat=",.0f"), xaxis=dict(tickangle=-20))
        st.plotly_chart(fig_cmp_bubble, use_container_width=True)

    st.markdown(f"""
    <div style="margin-top:0.5rem;padding:0.8rem 1.1rem;background:{card_ir};
        border:1px solid {border_ir};border-left:4px solid {gold_ir};border-radius:4px;">
        <div style="font-size:0.62rem;color:{muted_ir};line-height:1.9">
            📌 <strong>GoldBar / SilverBar:</strong> اونس جهانی × نرخ دلار ÷ ۳۱.۱۰۳۵ = قیمت هر گرم<br>
            📌 <strong>GoldCoin:</strong> قیمت هر گرم طلا × ۸.۱۳۳ (وزن خالص سکه بهار آزادی طرح جدید)<br>
            📌 <strong>CopperCthd:</strong> قیمت مس ($/پوند) × ۲.۲۰۵ × نرخ دلار = قیمت هر کیلو<br>
            📌 <strong>ZincIngot:</strong> قیمت روی LME ($/تن) ÷ ۱۰۰۰ × نرخ دلار = قیمت هر کیلو<br>
            ⚠ SteelRebar و Bitumen قیمت جهانی مستقیم ندارند — حباب محاسبه نمی‌شود.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # نمودار سود و زیان با قیمت تارگت
    # ════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="bp-section"><span class="bp-section-text">📊 نمودار سود و زیان — قیمت تارگت</span></div>', unsafe_allow_html=True)
    st.caption("برای هر گواهی قیمت خرید، تعداد، و تارگت را وارد کنید — نمودار P&L به صورت خودکار رسم می‌شود.")

    # ── ورود اطلاعات هر گواهی ──
    pnl_data = {}
    for kala_key, kala_info in IRAN_COMMODITIES.items():
        cur_p = kala_prices[kala_key]
        with st.expander(f"{kala_info['icon']} {kala_key} — {kala_info['label'].split('—')[1].strip()}", expanded=False):
            pa, pb, pc, pd_ = st.columns(4)
            with pa:
                buy_p = st.number_input(
                    "قیمت خرید (تومان)",
                    min_value=0,
                    value=int(cur_p * 0.95),
                    step=max(1, cur_p // 200),
                    key=f"pnl_buy_{kala_key}",
                )
            with pb:
                qty = st.number_input(
                    f"تعداد ({kala_info['unit']})",
                    min_value=0,
                    value=10,
                    step=1,
                    key=f"pnl_qty_{kala_key}",
                )
            with pc:
                target_p = st.number_input(
                    "قیمت تارگت (تومان)",
                    min_value=0,
                    value=int(cur_p * 1.20),
                    step=max(1, cur_p // 200),
                    key=f"pnl_target_{kala_key}",
                )
            with pd_:
                stop_p = st.number_input(
                    "حد ضرر (تومان)",
                    min_value=0,
                    value=int(cur_p * 0.85),
                    step=max(1, cur_p // 200),
                    key=f"pnl_stop_{kala_key}",
                )
            pnl_data[kala_key] = {
                "buy": buy_p, "qty": qty,
                "target": target_p, "stop": stop_p,
                "current": cur_p,
                "icon": kala_info["icon"],
                "unit": kala_info["unit"],
            }

    # ── محاسبه P&L و نمایش کارت‌ها ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">خلاصه سود و زیان</span></div>', unsafe_allow_html=True)

    pnl_rows = []
    for kala_key, d in pnl_data.items():
        if d["buy"] <= 0 or d["qty"] <= 0:
            continue
        cost        = d["buy"] * d["qty"]
        cur_val     = d["current"] * d["qty"]
        target_val  = d["target"] * d["qty"]
        stop_val    = d["stop"] * d["qty"]
        pnl_cur     = cur_val - cost
        pnl_target  = target_val - cost
        pnl_stop    = stop_val - cost
        pct_cur     = pnl_cur / cost * 100 if cost > 0 else 0
        pct_target  = pnl_target / cost * 100 if cost > 0 else 0
        pct_stop    = pnl_stop / cost * 100 if cost > 0 else 0
        rr_ratio    = abs(pnl_target / pnl_stop) if pnl_stop != 0 else 0

        pnl_rows.append({
            "گواهی":            f"{d['icon']} {kala_key}",
            "قیمت خرید":       fmt_toman(d["buy"]),
            "قیمت فعلی":       fmt_toman(d["current"]),
            "قیمت تارگت":      fmt_toman(d["target"]),
            "حد ضرر":          fmt_toman(d["stop"]),
            "P&L فعلی":        f"{pct_cur:+.1f}%",
            "سود در تارگت":    f"{pct_target:+.1f}%",
            "زیان در استاپ":   f"{pct_stop:+.1f}%",
            "R/R":             f"{rr_ratio:.2f}",
        })

    if pnl_rows:
        st.dataframe(pd.DataFrame(pnl_rows), use_container_width=True, hide_index=True)

    # ── کارت‌های P&L ──
    pnl_card_cols = st.columns(3)
    card_i = 0
    for kala_key, d in pnl_data.items():
        if d["buy"] <= 0 or d["qty"] <= 0:
            continue
        cost       = d["buy"] * d["qty"]
        pnl_cur    = (d["current"] - d["buy"]) * d["qty"]
        pnl_target = (d["target"]  - d["buy"]) * d["qty"]
        pnl_stop   = (d["stop"]    - d["buy"]) * d["qty"]
        pct_cur    = pnl_cur / cost * 100 if cost > 0 else 0
        pct_target = pnl_target / cost * 100 if cost > 0 else 0
        pct_stop   = pnl_stop / cost * 100 if cost > 0 else 0
        rr         = abs(pnl_target / pnl_stop) if pnl_stop != 0 else 0
        cur_color  = green_ir if pnl_cur >= 0 else red_ir
        rr_color   = green_ir if rr >= 2 else (gold_ir if rr >= 1 else red_ir)

        with pnl_card_cols[card_i % 3]:
            st.markdown(f"""
            <div style="background:{card_ir};border:1px solid {border_ir};
                border-top:3px solid {cur_color};border-radius:4px;
                padding:0.9rem 1rem;margin-bottom:0.7rem">
                <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
                    color:{muted_ir};text-transform:uppercase;margin-bottom:0.55rem">
                    {d['icon']} {kala_key}
                </div>
                <div style="display:flex;justify-content:space-between;padding:0.15rem 0;border-bottom:1px solid {border_ir}">
                    <span style="font-size:0.65rem;color:{muted_ir}">قیمت خرید</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:600;color:{'#c0c0c0' if is_dark else '#333'}">{fmt_toman(d['buy'])}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:0.15rem 0;border-bottom:1px solid {border_ir}">
                    <span style="font-size:0.65rem;color:{muted_ir}">قیمت فعلی</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;color:{cur_color}">{fmt_toman(d['current'])} ({pct_cur:+.1f}%)</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:0.15rem 0;border-bottom:1px solid {border_ir}">
                    <span style="font-size:0.65rem;color:{muted_ir}">🎯 تارگت</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;color:{green_ir}">{fmt_toman(d['target'])} ({pct_target:+.1f}%)</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:0.15rem 0;border-bottom:1px solid {border_ir}">
                    <span style="font-size:0.65rem;color:{muted_ir}">🛑 حد ضرر</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;color:{red_ir}">{fmt_toman(d['stop'])} ({pct_stop:+.1f}%)</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:0.2rem 0;margin-top:0.1rem">
                    <span style="font-size:0.65rem;color:{muted_ir}">نسبت R/R</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:700;color:{rr_color}">{rr:.2f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:0.1rem 0">
                    <span style="font-size:0.65rem;color:{muted_ir}">سود در تارگت</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;color:{green_ir}">{fmt_toman(pnl_target)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        card_i += 1

    # ── نمودار P&L — Waterfall ──
    active_keys = [k for k, d in pnl_data.items() if d["buy"] > 0 and d["qty"] > 0]
    if active_keys:
        st.markdown('<div class="bp-section"><span class="bp-section-text">نمودار P&L — قیمت‌های کلیدی</span></div>', unsafe_allow_html=True)

        # یک نمودار برای هر گواهی — scatter + خطوط افقی
        selected_pnl_key = st.selectbox(
            "گواهی مورد نظر برای نمودار",
            active_keys,
            format_func=lambda k: f"{IRAN_COMMODITIES[k]['icon']} {k}",
            key="pnl_chart_select"
        )

        d = pnl_data[selected_pnl_key]
        buy_p    = d["buy"]
        cur_p_v  = d["current"]
        target_v = d["target"]
        stop_v   = d["stop"]
        qty_v    = d["qty"]

        # محور X: بازه قیمتی از ۷۰٪ خرید تا ۱۳۰٪ تارگت
        p_min = min(stop_v, buy_p) * 0.90
        p_max = target_v * 1.10
        price_range = np.linspace(p_min, p_max, 300)
        pnl_range   = [(p - buy_p) * qty_v for p in price_range]

        fig_pnl = go.Figure()

        # خط P&L
        fig_pnl.add_trace(go.Scatter(
            x=list(price_range),
            y=pnl_range,
            mode="lines",
            line=dict(color=blue_ir, width=2.5),
            name="P&L",
            hovertemplate="قیمت: %{x:,.0f}<br>P&L: %{y:,.0f} تومان<extra></extra>",
        ))

        # رنگ‌آمیزی ناحیه سود (بالای صفر)
        fig_pnl.add_trace(go.Scatter(
            x=list(price_range),
            y=[max(v, 0) for v in pnl_range],
            fill="tozeroy",
            fillcolor="rgba(90,170,120,0.12)" if is_dark else "rgba(26,102,64,0.08)",
            line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))

        # رنگ‌آمیزی ناحیه ضرر (زیر صفر)
        fig_pnl.add_trace(go.Scatter(
            x=list(price_range),
            y=[min(v, 0) for v in pnl_range],
            fill="tozeroy",
            fillcolor="rgba(204,85,85,0.12)" if is_dark else "rgba(138,32,32,0.08)",
            line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))

        # خط صفر
        fig_pnl.add_hline(y=0, line_dash="solid",
                          line_color=muted_ir, line_width=1)

        # خط قیمت خرید
        fig_pnl.add_vline(x=buy_p, line_dash="dash",
                          line_color=gold_ir, line_width=1.5,
                          annotation_text=f"خرید\n{fmt_toman(buy_p)}",
                          annotation_font_color=gold_ir, annotation_font_size=9,
                          annotation_position="top right")

        # خط قیمت فعلی
        fig_pnl.add_vline(x=cur_p_v, line_dash="dot",
                          line_color=blue_ir, line_width=1.5,
                          annotation_text=f"فعلی\n{fmt_toman(cur_p_v)}",
                          annotation_font_color=blue_ir, annotation_font_size=9,
                          annotation_position="top left")

        # خط تارگت
        pnl_at_target = (target_v - buy_p) * qty_v
        fig_pnl.add_vline(x=target_v, line_dash="dash",
                          line_color=green_ir, line_width=1.5,
                          annotation_text=f"🎯 تارگت\n{fmt_toman(target_v)}\n+{fmt_toman(pnl_at_target)}",
                          annotation_font_color=green_ir, annotation_font_size=9,
                          annotation_position="top right")

        # خط حد ضرر
        pnl_at_stop = (stop_v - buy_p) * qty_v
        fig_pnl.add_vline(x=stop_v, line_dash="dash",
                          line_color=red_ir, line_width=1.5,
                          annotation_text=f"🛑 استاپ\n{fmt_toman(stop_v)}\n{fmt_toman(pnl_at_stop)}",
                          annotation_font_color=red_ir, annotation_font_size=9,
                          annotation_position="top left")

        # نقاط کلیدی روی نمودار
        key_prices = [stop_v, buy_p, cur_p_v, target_v]
        key_pnls   = [(p - buy_p) * qty_v for p in key_prices]
        key_colors = [red_ir, gold_ir, blue_ir, green_ir]
        key_labels = ["استاپ", "خرید", "فعلی", "تارگت"]
        fig_pnl.add_trace(go.Scatter(
            x=key_prices, y=key_pnls,
            mode="markers+text",
            marker=dict(size=10, color=key_colors,
                        line=dict(width=1.5, color="rgba(0,0,0,0.3)")),
            text=[f"{l}<br>{fmt_toman(p)}" for l, p in zip(key_labels, key_pnls)],
            textposition=["bottom right","top right","top left","top right"],
            textfont=dict(size=8, family="JetBrains Mono"),
            showlegend=False, hoverinfo="skip",
        ))

        fig_pnl.update_layout(
            **get_plot_layout(
                f"نمودار P&L — {IRAN_COMMODITIES[selected_pnl_key]['icon']} {selected_pnl_key} (تعداد: {qty_v} {d['unit']})",
                "قیمت (تومان)", "سود / زیان (تومان)", 460
            )
        )
        fig_pnl.update_layout(
            showlegend=False,
            xaxis=dict(tickformat=",.0f"),
            yaxis=dict(tickformat=",.0f", zeroline=True,
                       zerolinecolor=muted_ir, zerolinewidth=1.5),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        # ── نمودار مقایسه P&L همه گواهی‌ها در تارگت و استاپ ──
        st.markdown('<div class="bp-section"><span class="bp-section-text">مقایسه سود/زیان همه گواهی‌ها</span></div>', unsafe_allow_html=True)

        all_labels  = []
        all_target_pnl = []
        all_stop_pnl   = []
        all_cur_pnl    = []
        for k, d2 in pnl_data.items():
            if d2["buy"] <= 0 or d2["qty"] <= 0:
                continue
            cost2 = d2["buy"] * d2["qty"]
            all_labels.append(f"{d2['icon']} {k}")
            all_target_pnl.append((d2["target"] - d2["buy"]) / d2["buy"] * 100 if d2["buy"] > 0 else 0)
            all_stop_pnl.append((d2["stop"]   - d2["buy"]) / d2["buy"] * 100 if d2["buy"] > 0 else 0)
            all_cur_pnl.append((d2["current"] - d2["buy"]) / d2["buy"] * 100 if d2["buy"] > 0 else 0)

        if all_labels:
            fig_all_pnl = go.Figure()
            fig_all_pnl.add_trace(go.Bar(
                name="سود در تارگت (%)",
                x=all_labels, y=all_target_pnl,
                marker_color=green_ir,
                marker_line=dict(color="rgba(0,0,0,0.1)", width=0.5),
                text=[f"{v:+.1f}%" for v in all_target_pnl],
                textposition="outside",
                textfont=dict(size=9, family="JetBrains Mono"),
            ))
            fig_all_pnl.add_trace(go.Bar(
                name="P&L فعلی (%)",
                x=all_labels, y=all_cur_pnl,
                marker_color=blue_ir,
                marker_line=dict(color="rgba(0,0,0,0.1)", width=0.5),
                text=[f"{v:+.1f}%" for v in all_cur_pnl],
                textposition="outside",
                textfont=dict(size=9, family="JetBrains Mono"),
            ))
            fig_all_pnl.add_trace(go.Bar(
                name="زیان در استاپ (%)",
                x=all_labels, y=all_stop_pnl,
                marker_color=red_ir,
                marker_line=dict(color="rgba(0,0,0,0.1)", width=0.5),
                text=[f"{v:+.1f}%" for v in all_stop_pnl],
                textposition="outside",
                textfont=dict(size=9, family="JetBrains Mono"),
            ))
            fig_all_pnl.add_hline(y=0, line_dash="solid", line_color=muted_ir, line_width=1)
            fig_all_pnl.update_layout(
                **get_plot_layout("مقایسه P&L همه گواهی‌ها — تارگت / فعلی / استاپ", "", "%", 420),
                barmode="group"
            )
            fig_all_pnl.update_layout(xaxis=dict(tickangle=-20))
            st.plotly_chart(fig_all_pnl, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB BOURSE — اختیار معامله بورس کالای ایران
# ════════════════════════════════════════════════════════════════════
with tab_bourse:
    from scipy.stats import norm as _norm_bc
    import math as _math

    green_bc = "#5aaa78" if is_dark else "#1a6640"
    gold_bc  = "#c8a84b" if is_dark else "#8a6a1a"
    red_bc   = "#cc5555" if is_dark else "#8a2020"
    blue_bc  = "#5b9bd5"
    muted_bc = "#888888" if is_dark else "#555550"
    card_bc  = "#222222" if is_dark else "#e4e4e0"
    border_bc= "rgba(180,180,180,0.1)" if is_dark else "rgba(60,60,55,0.15)"

    # ─── دارایی‌های پایه قابل معامله در بورس کالا ───
    BOURSE_UNDERLYINGS = {
        "TLDY": {
            "label": "صندوق طلای لوتوس",
            "symbol": "TLDY",
            "unit": "ریال",
            "lot_size": 5000,
            "contract_value_multiplier": 1,
            "description": "واحدهای سرمایه‌گذاری صندوق طلای لوتوس — سررسید ماهانه",
            "icon": "🪙",
        },
        "GOLD": {
            "label": "شمش طلا",
            "symbol": "GOLD",
            "unit": "ریال/گرم",
            "lot_size": 100,
            "contract_value_multiplier": 1,
            "description": "شمش طلای ۹۹۵+ گرم",
            "icon": "🥇",
        },
        "KHRB": {
            "label": "صندوق کهربا ۱۰",
            "symbol": "KHRB",
            "unit": "ریال",
            "lot_size": 5000,
            "contract_value_multiplier": 1,
            "description": "واحدهای سرمایه‌گذاری صندوق کهربا ۱۰",
            "icon": "🟡",
        },
        "ZNAY": {
            "label": "صندوق طلای دنای زاگرس",
            "symbol": "ZNAY",
            "unit": "ریال",
            "lot_size": 5000,
            "contract_value_multiplier": 1,
            "description": "واحدهای سرمایه‌گذاری صندوق طلای دنای زاگرس",
            "icon": "🔶",
        },
        "SILV": {
            "label": "شمش نقره",
            "symbol": "SILV",
            "unit": "ریال/گرم",
            "lot_size": 1000,
            "contract_value_multiplier": 1,
            "description": "شمش نقره ۹۹۹.۹ گرم",
            "icon": "🥈",
        },
        "DRSH": {
            "label": "صندوق طلای درخشان",
            "symbol": "DRSH",
            "unit": "ریال",
            "lot_size": 5000,
            "contract_value_multiplier": 1,
            "description": "واحدهای سرمایه‌گذاری صندوق طلای درخشان",
            "icon": "✨",
        },
    }

    # ─── Black-Scholes برای بورس کالا ───
    def bc_bs_price(S, K, T, r, sigma, opt_type="call"):
        """قیمت‌گذاری اختیار معامله — بلک‌شولز — برای بورس کالای ایران"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            intrinsic = max(S-K, 0) if opt_type == "call" else max(K-S, 0)
            return intrinsic, (1.0 if opt_type=="call" and S>K else 0.0), 0.0, 0.0, 0.0
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if opt_type == "call":
            price = S*_norm_bc.cdf(d1) - K*np.exp(-r*T)*_norm_bc.cdf(d2)
            delta = float(_norm_bc.cdf(d1))
        else:
            price = K*np.exp(-r*T)*_norm_bc.cdf(-d2) - S*_norm_bc.cdf(-d1)
            delta = float(_norm_bc.cdf(d1) - 1)
        gamma = float(_norm_bc.pdf(d1) / (S*sigma*np.sqrt(T) + 1e-9))
        theta = float((-(S*_norm_bc.pdf(d1)*sigma)/(2*np.sqrt(T+1e-9))
                 - r*K*np.exp(-r*T)*(_norm_bc.cdf(d2) if opt_type=="call" else _norm_bc.cdf(-d2))) / 365)
        vega  = float(S*_norm_bc.pdf(d1)*np.sqrt(T)/100)
        return float(price), delta, gamma, theta, vega

    def bc_implied_vol(market_price, S, K, T, r, opt_type="call", tol=1e-5, max_iter=200):
        """محاسبه نوسان ضمنی با روش نیوتن-رافسون"""
        if market_price <= 0 or T <= 0:
            return None
        intrinsic = max(S-K, 0) if opt_type=="call" else max(K-S, 0)
        if market_price < intrinsic:
            return None
        sigma = 0.30
        for _ in range(max_iter):
            price, _, _, _, vega_v = bc_bs_price(S, K, T, r, sigma, opt_type)
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            if abs(vega_v) < 1e-10:
                break
            sigma -= diff / (vega_v * 100 + 1e-9)
            sigma = max(0.001, min(sigma, 10.0))
        return sigma

    def bc_pnl_at_expiry(S_range, K, premium_paid, lot_size, n_contracts, opt_type="call", position="long"):
        """محاسبه P&L استراتژی در سررسید"""
        pnl = []
        for s in S_range:
            intrinsic = max(s-K, 0) if opt_type=="call" else max(K-s, 0)
            if position == "long":
                p = (intrinsic - premium_paid) * lot_size * n_contracts
            else:
                p = (premium_paid - intrinsic) * lot_size * n_contracts
            pnl.append(p)
        return np.array(pnl)

    def bc_format_rial(v):
        if abs(v) >= 1e9:
            return f"{v/1e9:.2f} میلیارد"
        elif abs(v) >= 1e6:
            return f"{v/1e6:.1f} میلیون"
        else:
            return f"{v:,.0f}"

    def bc_shamsi_to_days(shamsi_date_str):
        """تبدیل تاریخ شمسی مثل ۱۴۰۵/۱۰/۲۷ به روزهای باقیمانده"""
        try:
            parts = shamsi_date_str.strip().replace("-","/").split("/")
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            # تبدیل تقریبی شمسی به میلادی
            # سال شمسی = سال میلادی - 621/622
            gy = y + 621
            # تقریب ماه
            month_map = {1:3,2:4,3:5,4:6,5:7,6:8,7:9,8:10,9:11,10:12,11:1,12:2}
            gm = month_map[m]
            gd = d
            if m >= 11:
                gy += 1
            from datetime import date
            exp_date = date(gy, gm, gd)
            today = date.today()
            delta = (exp_date - today).days
            return max(delta, 0)
        except Exception:
            return 30

    # ─── رابط اصلی ───
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem">
        <span style="font-size:1.4rem">🏛</span>
        <div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;font-weight:700;
                letter-spacing:0.1em;color:var(--accent)">اختیار معامله — بورس کالای ایران</div>
            <div style="font-size:0.65rem;color:var(--muted);letter-spacing:0.08em">
                قیمت‌گذاری · Greeks · P&L · استراتژی‌های ترکیبی · مقایسه سررسید
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══ انتخاب دارایی پایه ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">① دارایی پایه و پارامترهای قرارداد</span></div>', unsafe_allow_html=True)

    bc_col1, bc_col2 = st.columns([1, 2])
    with bc_col1:
        underlying_key = st.selectbox(
            "دارایی پایه",
            list(BOURSE_UNDERLYINGS.keys()),
            format_func=lambda k: f"{BOURSE_UNDERLYINGS[k]['icon']} {BOURSE_UNDERLYINGS[k]['label']}",
            key="bc_underlying"
        )
        ul = BOURSE_UNDERLYINGS[underlying_key]
        st.markdown(f"""
        <div style="background:{card_bc};border:1px solid {border_bc};border-radius:4px;
            padding:0.6rem 0.8rem;margin-top:0.3rem;font-size:0.65rem;color:{muted_bc};line-height:1.8">
            {ul['icon']} <strong style="color:{'#c0c0c0' if is_dark else '#333'}">{ul['label']}</strong><br>
            📦 اندازه قرارداد: <strong>{ul['lot_size']:,}</strong> واحد<br>
            💱 واحد: {ul['unit']}<br>
            📝 {ul['description']}
        </div>
        """, unsafe_allow_html=True)

    with bc_col2:
        r1, r2, r3 = st.columns(3)
        with r1:
            bc_spot = st.number_input(
                "قیمت لحظه‌ای (ریال)",
                min_value=1, max_value=100_000_000,
                value=700_000, step=5_000,
                key="bc_spot",
                help="قیمت فعلی دارایی پایه در بازار"
            )
            bc_strike = st.number_input(
                "قیمت اعمال Strike (ریال)",
                min_value=1, max_value=100_000_000,
                value=700_000, step=5_000,
                key="bc_strike",
                help="قیمت اعمال قرارداد اختیار"
            )
        with r2:
            bc_exp_method = st.radio(
                "روش ورود سررسید",
                ["تعداد روز", "تاریخ شمسی"],
                key="bc_exp_method",
                horizontal=True,
            )
            if bc_exp_method == "تعداد روز":
                bc_dte = st.number_input("روز تا سررسید", min_value=1, max_value=365,
                                          value=30, step=1, key="bc_dte")
            else:
                bc_shamsi = st.text_input("تاریخ سررسید (شمسی)", value="1405/10/27", key="bc_shamsi")
                bc_dte = bc_shamsi_to_days(bc_shamsi)
                st.caption(f"≈ {bc_dte} روز تا سررسید")

            bc_iv = st.slider("نوسان ضمنی IV (%)", 5, 150, 35, 1, key="bc_iv",
                               help="Implied Volatility — از تابلوی معاملات یا محاسبه معکوس BS")
        with r3:
            bc_rf = st.number_input("نرخ بدون ریسک (%)", min_value=0.0, max_value=50.0,
                                     value=25.0, step=0.5, key="bc_rf",
                                     help="نرخ اوراق خزانه یا سود بانکی ایران — معمولاً ۲۲–۲۸٪")
            bc_n_contracts = st.number_input("تعداد قرارداد", min_value=1, max_value=10000,
                                               value=1, step=1, key="bc_n_cont")
            bc_opt_type = st.radio("نوع اختیار", ["اختیار خرید (Call)", "اختیار فروش (Put)"],
                                    key="bc_opt_type", horizontal=False)

    bc_is_call = "Call" in bc_opt_type
    bc_T = bc_dte / 365.0
    bc_r_dec = bc_rf / 100.0
    bc_sig = bc_iv / 100.0
    lot = ul["lot_size"]

    # ══ محاسبه قیمت‌گذاری ══
    bc_price, bc_delta, bc_gamma, bc_theta, bc_vega = bc_bs_price(
        bc_spot, bc_strike, bc_T, bc_r_dec, bc_sig,
        "call" if bc_is_call else "put"
    )

    moneyness_pct = (bc_strike - bc_spot) / bc_spot * 100
    if bc_is_call:
        if bc_strike > bc_spot * 1.01:
            money_label, money_color = "OTM — خارج از پول", gold_bc
        elif bc_strike < bc_spot * 0.99:
            money_label, money_color = "ITM — داخل پول", red_bc
        else:
            money_label, money_color = "ATM — روی پول", blue_bc
    else:
        if bc_strike < bc_spot * 0.99:
            money_label, money_color = "OTM — خارج از پول", gold_bc
        elif bc_strike > bc_spot * 1.01:
            money_label, money_color = "ITM — داخل پول", red_bc
        else:
            money_label, money_color = "ATM — روی پول", blue_bc

    total_premium_bc = bc_price * lot * bc_n_contracts
    breakeven_bc = bc_spot + bc_price if bc_is_call else bc_spot - bc_price

    # ── نمایش نتایج قیمت‌گذاری ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">② قیمت‌گذاری و Greeks</span></div>', unsafe_allow_html=True)

    # Moneyness badge
    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:8px;padding:0.4rem 0.9rem;
        background:{card_bc};border:1px solid {money_color};border-radius:20px;margin-bottom:0.8rem">
        <span style="width:8px;height:8px;border-radius:50%;background:{money_color};display:inline-block"></span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;font-weight:700;
            color:{money_color}">{money_label}</span>
        <span style="font-size:0.65rem;color:{muted_bc}">|</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:{muted_bc}">
            {'Call' if bc_is_call else 'Put'} · Strike={bc_strike:,} · Spot={bc_spot:,} · {bc_dte}d
        </span>
    </div>
    """, unsafe_allow_html=True)

    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    metric_style = f"background:{card_bc};border:1px solid {border_bc};border-top:3px solid {{c}};border-radius:4px;padding:0.85rem 0.9rem"

    def bc_metric_card(col, title, value, subtitle, color, note=""):
        col.markdown(f"""
        <div style="{metric_style.format(c=color)}">
            <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.12em;
                color:{muted_bc};text-transform:uppercase;margin-bottom:0.3rem">{title}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.15rem;
                font-weight:700;color:{color}">{value}</div>
            <div style="font-size:0.6rem;color:{muted_bc};margin-top:0.25rem">{subtitle}</div>
            {"<div style='font-size:0.58rem;color:"+muted_bc+";margin-top:0.15rem;border-top:1px solid "+border_bc+";padding-top:0.2rem'>"+note+"</div>" if note else ""}
        </div>
        """, unsafe_allow_html=True)

    bc_metric_card(gc1, "قیمت منصفانه (BS)",
                   f"{bc_price:,.0f} ریال",
                   f"کل: {bc_format_rial(total_premium_bc)} ریال",
                   blue_bc, f"هر واحد اختیار = {lot:,} سهم")

    bc_metric_card(gc2, "Delta (Δ)",
                   f"{bc_delta:.4f}",
                   "احتمال ITM تقریبی",
                   green_bc if (bc_is_call and bc_delta>0.5) or (not bc_is_call and bc_delta<-0.5) else gold_bc,
                   "بین ۰.۲–۰.۴ ایده‌آل برای خریدار")

    bc_metric_card(gc3, "Theta (Θ) روزانه",
                   f"{bc_theta*bc_price:,.0f} ریال",
                   "کاهش ارزش روزانه",
                   red_bc,
                   f"در {bc_dte} روز: {bc_format_rial(bc_theta*bc_price*bc_dte)} ریال")

    bc_metric_card(gc4, "Vega (ν) per 1%IV",
                   f"{bc_vega*bc_spot:.4f}",
                   "حساسیت به نوسان",
                   gold_bc,
                   "IV↑1% → قیمت به این مقدار تغییر می‌کند")

    bc_metric_card(gc5, "نقطه سربه‌سر",
                   f"{bc_format_rial(breakeven_bc)} ریال",
                   f"{'Spot + Premium' if bc_is_call else 'Spot − Premium'}",
                   blue_bc,
                   f"فاصله از Spot: {abs(breakeven_bc-bc_spot)/bc_spot*100:.2f}%")

    # ── Greeks جدول کامل ──
    with st.expander("جدول کامل Greeks و محاسبات", expanded=False):
        greeks_col1, greeks_col2 = st.columns(2)
        with greeks_col1:
            st.markdown(f"""
            <div style="background:{card_bc};border:1px solid {border_bc};border-radius:4px;padding:0.8rem 1rem">
                <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.12em;color:{muted_bc};
                    text-transform:uppercase;margin-bottom:0.6rem">Greeks کامل</div>
                {''.join([f"""
                <div style="display:flex;justify-content:space-between;padding:0.22rem 0;
                    border-bottom:1px solid {border_bc}">
                    <span style="font-size:0.68rem;color:{muted_bc};font-weight:600">{lbl}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                        font-weight:700;color:{color}">{val}</span>
                </div>""" for lbl, val, color in [
                    ("Delta (Δ)", f"{bc_delta:.6f}", blue_bc),
                    ("Gamma (Γ)", f"{bc_gamma:.8f}", gold_bc),
                    ("Theta (Θ) روزانه", f"{bc_theta:.6f}", red_bc),
                    ("Vega (ν) per 1%", f"{bc_vega:.6f}", green_bc),
                    ("IV استفاده‌شده", f"{bc_iv}%", blue_bc),
                    ("T (سال)", f"{bc_T:.4f}", muted_bc),
                    ("نرخ بدون‌ریسک", f"{bc_rf}%", muted_bc),
                ]])}
            </div>
            """, unsafe_allow_html=True)
        with greeks_col2:
            st.markdown(f"""
            <div style="background:{card_bc};border:1px solid {border_bc};border-radius:4px;padding:0.8rem 1rem">
                <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.12em;color:{muted_bc};
                    text-transform:uppercase;margin-bottom:0.6rem">مشخصات قرارداد</div>
                {''.join([f"""
                <div style="display:flex;justify-content:space-between;padding:0.22rem 0;
                    border-bottom:1px solid {border_bc}">
                    <span style="font-size:0.68rem;color:{muted_bc};font-weight:600">{lbl}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                        font-weight:700;color:{color}">{val}</span>
                </div>""" for lbl, val, color in [
                    ("دارایی پایه", ul['label'], blue_bc),
                    ("نوع", "Call اختیار خرید" if bc_is_call else "Put اختیار فروش", green_bc if bc_is_call else red_bc),
                    ("قیمت فعلی Spot", f"{bc_spot:,} ریال", blue_bc),
                    ("Strike", f"{bc_strike:,} ریال", gold_bc),
                    ("اندازه قرارداد", f"{lot:,}", muted_bc),
                    ("تعداد قرارداد", f"{bc_n_contracts}", muted_bc),
                    ("ارزش کل قراردادها", f"{bc_format_rial(bc_spot*lot*bc_n_contracts)} ریال", gold_bc),
                    ("وثیقه اولیه (تخمین)", f"{bc_format_rial(bc_price*lot*bc_n_contracts*1.2)} ریال", red_bc),
                ]])}
            </div>
            """, unsafe_allow_html=True)

    # ══ ورود قیمت معامله‌شده و محاسبه IV ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">③ محاسبه نوسان ضمنی (IV) از قیمت بازار</span></div>', unsafe_allow_html=True)
    st.caption("اگه قیمت اختیار در تابلوی معاملات داری، IV واقعی را محاسبه می‌کند.")

    iv_col1, iv_col2, iv_col3 = st.columns([2, 2, 3])
    with iv_col1:
        bc_market_price = st.number_input(
            "قیمت معامله‌شده در تابلو (ریال/واحد)",
            min_value=0, max_value=10_000_000,
            value=0, step=1_000,
            key="bc_mkt_price",
            help="آخرین قیمت معامله یا میانه bid/ask از تابلوی بورس کالا"
        )
    with iv_col2:
        if st.button("محاسبه IV از قیمت بازار", use_container_width=True, key="bc_calc_iv"):
            if bc_market_price > 0:
                iv_calc = bc_implied_vol(
                    bc_market_price, bc_spot, bc_strike, bc_T, bc_r_dec,
                    "call" if bc_is_call else "put"
                )
                st.session_state["bc_iv_calc"] = iv_calc
                st.session_state["bc_iv_market_price"] = bc_market_price
            else:
                st.warning("قیمت بازار را وارد کنید.")

    with iv_col3:
        iv_calc_result = st.session_state.get("bc_iv_calc")
        if iv_calc_result is not None:
            iv_diff = iv_calc_result * 100 - bc_iv
            iv_color = red_bc if iv_diff > 5 else (green_bc if iv_diff < -5 else gold_bc)
            st.markdown(f"""
            <div style="background:{card_bc};border:1px solid {iv_color};border-radius:4px;
                padding:0.6rem 0.9rem;display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:0.65rem;color:{muted_bc}">IV محاسبه‌شده</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                    font-weight:700;color:{iv_color}">{iv_calc_result*100:.2f}%</span>
                <span style="font-size:0.65rem;color:{iv_color};font-weight:600">
                    {'↑ گران‌تر از BS' if iv_diff>5 else ('↓ ارزان‌تر از BS' if iv_diff<-5 else '≈ منصفانه')}
                    ({iv_diff:+.1f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ══ نمودار P&L ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">④ نمودار سود و زیان در سررسید</span></div>', unsafe_allow_html=True)

    pnl_pos = st.radio("موضع معامله‌گر", ["خریدار (Long)", "فروشنده (Short)"],
                        key="bc_position", horizontal=True)
    bc_is_long = "Long" in pnl_pos

    custom_premium_on = st.checkbox("استفاده از قیمت سفارشی به جای BS", key="bc_custom_prem")
    if custom_premium_on:
        bc_prem_used = st.number_input("پرمیوم سفارشی (ریال/واحد)",
                                        min_value=0, max_value=10_000_000,
                                        value=int(bc_price), step=1_000, key="bc_custom_prem_val")
    else:
        bc_prem_used = bc_price

    # محدوده قیمت
    s_lo = bc_spot * 0.60
    s_hi = bc_spot * 1.40
    S_range_bc = np.linspace(s_lo, s_hi, 400)

    pnl_vals = bc_pnl_at_expiry(
        S_range_bc, bc_strike, bc_prem_used, lot, bc_n_contracts,
        "call" if bc_is_call else "put",
        "long" if bc_is_long else "short"
    )

    max_profit = float(np.max(pnl_vals))
    max_loss   = float(np.min(pnl_vals))
    be_price   = breakeven_bc

    fig_bc_pnl = go.Figure()

    # ناحیه سود
    profit_y = np.where(pnl_vals >= 0, pnl_vals, 0)
    loss_y   = np.where(pnl_vals < 0, pnl_vals, 0)

    fig_bc_pnl.add_trace(go.Scatter(
        x=S_range_bc, y=profit_y, fill="tozeroy",
        fillcolor="rgba(90,170,120,0.15)" if is_dark else "rgba(26,102,64,0.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_bc_pnl.add_trace(go.Scatter(
        x=S_range_bc, y=loss_y, fill="tozeroy",
        fillcolor="rgba(204,85,85,0.15)" if is_dark else "rgba(138,32,32,0.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_bc_pnl.add_trace(go.Scatter(
        x=S_range_bc, y=pnl_vals, mode="lines",
        line=dict(color=blue_bc, width=2.5),
        name=f"{'Long' if bc_is_long else 'Short'} {'Call' if bc_is_call else 'Put'}",
        hovertemplate="قیمت: %{x:,.0f}<br>P&L: %{y:,.0f} ریال<extra></extra>",
    ))

    # خطوط کلیدی
    fig_bc_pnl.add_vline(x=bc_spot, line_dash="dot", line_color=blue_bc, line_width=1.5,
                          annotation_text=f"Spot\n{bc_spot:,}", annotation_font_color=blue_bc, annotation_font_size=9)
    fig_bc_pnl.add_vline(x=bc_strike, line_dash="dash", line_color=gold_bc, line_width=1.8,
                          annotation_text=f"Strike\n{bc_strike:,}", annotation_font_color=gold_bc, annotation_font_size=9)
    fig_bc_pnl.add_vline(x=be_price, line_dash="dot", line_color=green_bc if bc_is_long else red_bc,
                          line_width=1.2,
                          annotation_text=f"BE\n{be_price:,.0f}", annotation_font_color=green_bc, annotation_font_size=9)
    fig_bc_pnl.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)

    # حداکثر سود/زیان annotation
    if bc_is_long and bc_is_call:
        fig_bc_pnl.add_annotation(x=s_hi*0.97, y=max_profit,
                                   text=f"Max Profit: {bc_format_rial(max_profit)}",
                                   font=dict(size=9, color=green_bc, family="JetBrains Mono"),
                                   showarrow=False, bgcolor=card_bc, bordercolor=green_bc, borderwidth=1)
        max_loss_y = -bc_prem_used * lot * bc_n_contracts
        fig_bc_pnl.add_annotation(x=bc_spot*0.75, y=max_loss_y,
                                   text=f"Max Loss: {bc_format_rial(max_loss_y)}",
                                   font=dict(size=9, color=red_bc, family="JetBrains Mono"),
                                   showarrow=False, bgcolor=card_bc, bordercolor=red_bc, borderwidth=1)

    fig_bc_pnl.update_layout(**get_plot_layout(
        f"P&L در سررسید — {'Long' if bc_is_long else 'Short'} {'Call' if bc_is_call else 'Put'} | {ul['icon']} {ul['label']}",
        "قیمت دارایی پایه در سررسید (ریال)", "سود / زیان (ریال)", 460
    ))
    fig_bc_pnl.update_layout(
        xaxis=dict(tickformat=",.0f"),
        yaxis=dict(tickformat=",.0f"),
    )
    st.plotly_chart(fig_bc_pnl, use_container_width=True)

    # خلاصه سناریوها
    pnl_s1, pnl_s2, pnl_s3 = st.columns(3)
    pnl_at_spot   = float(bc_pnl_at_expiry([bc_spot], bc_strike, bc_prem_used, lot, bc_n_contracts,
                                             "call" if bc_is_call else "put",
                                             "long" if bc_is_long else "short")[0])
    pnl_at_strike = float(bc_pnl_at_expiry([bc_strike], bc_strike, bc_prem_used, lot, bc_n_contracts,
                                             "call" if bc_is_call else "put",
                                             "long" if bc_is_long else "short")[0])
    pnl_at_be     = float(bc_pnl_at_expiry([be_price], bc_strike, bc_prem_used, lot, bc_n_contracts,
                                             "call" if bc_is_call else "put",
                                             "long" if bc_is_long else "short")[0])

    with pnl_s1:
        st.metric("P&L در Spot فعلی", f"{bc_format_rial(pnl_at_spot)} ریال",
                  delta=f"{pnl_at_spot/(bc_prem_used*lot*bc_n_contracts+1)*100:.1f}%",
                  delta_color="normal" if pnl_at_spot>=0 else "inverse")
        st.caption(f"اگر قیمت تا سررسید تغییر نکند")
    with pnl_s2:
        st.metric("P&L در Strike", f"{bc_format_rial(pnl_at_strike)} ریال",
                  delta=f"{pnl_at_strike/(bc_prem_used*lot*bc_n_contracts+1)*100:.1f}%",
                  delta_color="normal" if pnl_at_strike>=0 else "inverse")
        st.caption(f"اگر قیمت به {bc_strike:,} ریال برسد")
    with pnl_s3:
        st.metric("P&L در Breakeven", f"{bc_format_rial(pnl_at_be)} ≈ 0",
                  delta=f"BE = {bc_format_rial(be_price)} ریال")
        st.caption(f"نقطه‌ای که زیان/سود صفر می‌شود")

    # ══ Theta Decay ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">⑤ Theta Decay — کاهش ارزش با گذر زمان</span></div>', unsafe_allow_html=True)

    days_arr_bc = np.arange(bc_dte, 0, -1)
    theta_vals_bc = []
    for d in days_arr_bc:
        p_t, *_ = bc_bs_price(bc_spot, bc_strike, d/365.0, bc_r_dec, bc_sig,
                               "call" if bc_is_call else "put")
        theta_vals_bc.append(p_t)

    fig_theta_bc = go.Figure()
    fig_theta_bc.add_trace(go.Scatter(
        x=days_arr_bc, y=theta_vals_bc,
        mode="lines", name="ارزش اختیار",
        line=dict(color=gold_bc, width=2.2),
        fill="tozeroy",
        fillcolor="rgba(200,168,75,0.12)" if is_dark else "rgba(138,106,26,0.08)",
        hovertemplate="روز تا سررسید: %{x}<br>ارزش: %{y:,.0f} ریال<extra></extra>",
    ))
    fig_theta_bc.add_hline(y=bc_prem_used, line_dash="dash",
                            line_color=blue_bc, line_width=1.2,
                            annotation_text=f"پرمیوم: {bc_prem_used:,.0f} ریال",
                            annotation_font_color=blue_bc, annotation_font_size=9)
    # نیمه عمر ارزش زمانی
    half_val = bc_prem_used * 0.5
    half_idx = np.argmin(np.abs(np.array(theta_vals_bc) - half_val))
    if half_idx < len(days_arr_bc):
        fig_theta_bc.add_vline(x=days_arr_bc[half_idx], line_dash="dot", line_color=red_bc,
                                line_width=1, annotation_text=f"نیمه عمر: {days_arr_bc[half_idx]}d",
                                annotation_font_color=red_bc, annotation_font_size=9)

    fig_theta_bc.update_layout(**get_plot_layout(
        "THETA DECAY — کاهش ارزش اختیار با گذر زمان",
        "روز تا سررسید", "ارزش اختیار (ریال)", 320
    ))
    fig_theta_bc.update_xaxes(autorange="reversed")
    fig_theta_bc.update_layout(yaxis=dict(tickformat=",.0f"))
    st.plotly_chart(fig_theta_bc, use_container_width=True)

    # ══ مقایسه Strike های مختلف ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">⑥ مقایسه Strike‌های مختلف — جدول زنجیره اختیار</span></div>', unsafe_allow_html=True)
    st.caption("قیمت‌گذاری BS برای استرایک‌های متداول اطراف قیمت فعلی — مشابه تابلوی Option Chain.")

    chain_strikes_pct = [-20, -15, -10, -5, 0, +5, +10, +15, +20]
    chain_strikes = [int(bc_spot * (1 + p/100)) for p in chain_strikes_pct]

    chain_rows = []
    for k in chain_strikes:
        cp_c, d_c, g_c, th_c, v_c = bc_bs_price(bc_spot, k, bc_T, bc_r_dec, bc_sig, "call")
        cp_p, d_p, g_p, th_p, v_p = bc_bs_price(bc_spot, k, bc_T, bc_r_dec, bc_sig, "put")
        m_pct = (k - bc_spot) / bc_spot * 100
        money = "ITM ✓" if (bc_is_call and k < bc_spot) or (not bc_is_call and k > bc_spot) else (
                "ATM ≈" if abs(m_pct) < 2 else "OTM")
        chain_rows.append({
            "Strike": f"{k:,}",
            "Moneyness": money,
            "Call قیمت": f"{cp_c:,.0f}",
            "Call Δ": f"{d_c:.3f}",
            "Put قیمت": f"{cp_p:,.0f}",
            "Put Δ": f"{d_p:.3f}",
            "Gamma": f"{g_c:.6f}",
            "Theta/روز": f"{th_c*cp_c:,.0f}",
        })

    df_chain = pd.DataFrame(chain_rows)
    st.dataframe(df_chain, use_container_width=True, hide_index=True)

    # نمودار Call vs Put قیمت در استرایک‌های مختلف
    call_prices = [float(bc_bs_price(bc_spot, k, bc_T, bc_r_dec, bc_sig, "call")[0]) for k in chain_strikes]
    put_prices  = [float(bc_bs_price(bc_spot, k, bc_T, bc_r_dec, bc_sig, "put")[0]) for k in chain_strikes]

    fig_chain = go.Figure()
    fig_chain.add_trace(go.Scatter(
        x=chain_strikes, y=call_prices, mode="lines+markers", name="Call",
        line=dict(color=green_bc, width=2), marker=dict(size=7, color=green_bc),
        hovertemplate="Strike: %{x:,}<br>Call: %{y:,.0f} ریال<extra></extra>",
    ))
    fig_chain.add_trace(go.Scatter(
        x=chain_strikes, y=put_prices, mode="lines+markers", name="Put",
        line=dict(color=red_bc, width=2), marker=dict(size=7, color=red_bc),
        hovertemplate="Strike: %{x:,}<br>Put: %{y:,.0f} ریال<extra></extra>",
    ))
    fig_chain.add_vline(x=bc_spot, line_dash="dot", line_color=blue_bc, line_width=1.5,
                         annotation_text=f"Spot {bc_spot:,}", annotation_font_color=blue_bc, annotation_font_size=9)
    fig_chain.update_layout(**get_plot_layout(
        "Option Chain — قیمت Call و Put در Strike‌های مختلف",
        "Strike (ریال)", "قیمت اختیار (ریال)", 360
    ))
    fig_chain.update_layout(xaxis=dict(tickformat=",.0f"), yaxis=dict(tickformat=",.0f"))
    st.plotly_chart(fig_chain, use_container_width=True)

    # ══ استراتژی‌های ترکیبی ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">⑦ استراتژی‌های ترکیبی بورس کالا</span></div>', unsafe_allow_html=True)

    strategy_bc = st.selectbox(
        "استراتژی",
        ["Bull Call Spread", "Bear Put Spread", "Long Straddle", "Long Strangle",
         "Covered Call (با دارایی پایه)", "Protective Put (بیمه دارایی)"],
        key="bc_strategy"
    )

    if strategy_bc == "Bull Call Spread":
        st.caption("خرید Call با Strike پایین + فروش Call با Strike بالا — هزینه کمتر، سود محدود")
        sc1, sc2 = st.columns(2)
        with sc1:
            k_low  = st.number_input("Strike پایین (خرید Call)", value=int(bc_spot*0.95), step=5_000, key="bcs_kl")
        with sc2:
            k_high = st.number_input("Strike بالا (فروش Call)", value=int(bc_spot*1.05), step=5_000, key="bcs_kh")

        p_buy,  *_ = bc_bs_price(bc_spot, k_low,  bc_T, bc_r_dec, bc_sig, "call")
        p_sell, *_ = bc_bs_price(bc_spot, k_high, bc_T, bc_r_dec, bc_sig, "call")
        net_debit = p_buy - p_sell
        max_profit_s = (k_high - k_low - net_debit) * lot * bc_n_contracts
        max_loss_s   = net_debit * lot * bc_n_contracts
        be_s = k_low + net_debit

        pnl_s = []
        for s in S_range_bc:
            long_val  = max(s - k_low, 0)
            short_val = max(s - k_high, 0)
            pnl_s.append((long_val - short_val - net_debit) * lot * bc_n_contracts)

        strat_info = [
            ("هزینه خالص (Net Debit)", f"{net_debit:,.0f} ریال/واحد", red_bc),
            ("حداکثر سود", f"{bc_format_rial(max_profit_s)} ریال", green_bc),
            ("حداکثر زیان", f"{bc_format_rial(max_loss_s)} ریال", red_bc),
            ("نقطه سربه‌سر", f"{be_s:,.0f} ریال", blue_bc),
            ("R/R نسبت", f"{abs(max_profit_s/max_loss_s):.2f}" if max_loss_s else "∞", gold_bc),
        ]

    elif strategy_bc == "Bear Put Spread":
        st.caption("خرید Put با Strike بالا + فروش Put با Strike پایین — برای انتظار ریزش قیمت")
        sc1, sc2 = st.columns(2)
        with sc1:
            k_high = st.number_input("Strike بالا (خرید Put)", value=int(bc_spot*1.05), step=5_000, key="bps_kh")
        with sc2:
            k_low  = st.number_input("Strike پایین (فروش Put)", value=int(bc_spot*0.95), step=5_000, key="bps_kl")

        p_buy,  *_ = bc_bs_price(bc_spot, k_high, bc_T, bc_r_dec, bc_sig, "put")
        p_sell, *_ = bc_bs_price(bc_spot, k_low,  bc_T, bc_r_dec, bc_sig, "put")
        net_debit = p_buy - p_sell
        max_profit_s = (k_high - k_low - net_debit) * lot * bc_n_contracts
        max_loss_s   = net_debit * lot * bc_n_contracts
        be_s = k_high - net_debit

        pnl_s = []
        for s in S_range_bc:
            long_val  = max(k_high - s, 0)
            short_val = max(k_low  - s, 0)
            pnl_s.append((long_val - short_val - net_debit) * lot * bc_n_contracts)

        strat_info = [
            ("هزینه خالص (Net Debit)", f"{net_debit:,.0f} ریال/واحد", red_bc),
            ("حداکثر سود", f"{bc_format_rial(max_profit_s)} ریال", green_bc),
            ("حداکثر زیان", f"{bc_format_rial(max_loss_s)} ریال", red_bc),
            ("نقطه سربه‌سر", f"{be_s:,.0f} ریال", blue_bc),
            ("R/R نسبت", f"{abs(max_profit_s/max_loss_s):.2f}" if max_loss_s else "∞", gold_bc),
        ]

    elif strategy_bc == "Long Straddle":
        st.caption("خرید همزمان Call و Put با یک Strike — برای انتظار نوسان شدید (صعود یا نزول)")

        p_call, *_ = bc_bs_price(bc_spot, bc_strike, bc_T, bc_r_dec, bc_sig, "call")
        p_put,  *_ = bc_bs_price(bc_spot, bc_strike, bc_T, bc_r_dec, bc_sig, "put")
        total_prem_s = p_call + p_put
        be_up_s  = bc_strike + total_prem_s
        be_dn_s  = bc_strike - total_prem_s
        max_loss_s = total_prem_s * lot * bc_n_contracts

        pnl_s = []
        for s in S_range_bc:
            call_val = max(s - bc_strike, 0)
            put_val  = max(bc_strike - s, 0)
            pnl_s.append((call_val + put_val - total_prem_s) * lot * bc_n_contracts)

        strat_info = [
            ("پرمیوم Call", f"{p_call:,.0f} ریال", green_bc),
            ("پرمیوم Put", f"{p_put:,.0f} ریال", red_bc),
            ("هزینه کل", f"{bc_format_rial(max_loss_s)} ریال", red_bc),
            ("Breakeven بالا", f"{be_up_s:,.0f} ریال", green_bc),
            ("Breakeven پایین", f"{be_dn_s:,.0f} ریال", green_bc),
            ("حداکثر زیان", f"{bc_format_rial(max_loss_s)} ریال", red_bc),
        ]

    elif strategy_bc == "Long Strangle":
        st.caption("خرید Call OTM + خرید Put OTM — هزینه کمتر از Straddle، نیاز به نوسان بیشتر")
        sc1, sc2 = st.columns(2)
        with sc1:
            k_put = st.number_input("Strike Put (پایین‌تر)", value=int(bc_spot*0.92), step=5_000, key="str_kp")
        with sc2:
            k_call = st.number_input("Strike Call (بالاتر)", value=int(bc_spot*1.08), step=5_000, key="str_kc")

        p_call, *_ = bc_bs_price(bc_spot, k_call, bc_T, bc_r_dec, bc_sig, "call")
        p_put,  *_ = bc_bs_price(bc_spot, k_put,  bc_T, bc_r_dec, bc_sig, "put")
        total_prem_s = p_call + p_put
        be_up_s  = k_call + total_prem_s
        be_dn_s  = k_put  - total_prem_s
        max_loss_s = total_prem_s * lot * bc_n_contracts

        pnl_s = []
        for s in S_range_bc:
            call_val = max(s - k_call, 0)
            put_val  = max(k_put - s, 0)
            pnl_s.append((call_val + put_val - total_prem_s) * lot * bc_n_contracts)

        strat_info = [
            ("پرمیوم Call OTM", f"{p_call:,.0f} ریال", green_bc),
            ("پرمیوم Put OTM", f"{p_put:,.0f} ریال", red_bc),
            ("هزینه کل", f"{bc_format_rial(max_loss_s)} ریال", red_bc),
            ("Breakeven بالا", f"{be_up_s:,.0f} ریال", green_bc),
            ("Breakeven پایین", f"{be_dn_s:,.0f} ریال", green_bc),
        ]

    elif strategy_bc == "Covered Call (با دارایی پایه)":
        st.caption("نگه‌داشتن دارایی پایه + فروش Call — درآمد پرمیوم، سود محدود")
        k_cc = st.number_input("Strike Call فروخته‌شده", value=int(bc_spot*1.05), step=5_000, key="cc_bourse_k")
        n_units = st.number_input("تعداد واحد دارایی پایه", value=lot*bc_n_contracts, step=lot, key="cc_bourse_n")

        p_cc, *_ = bc_bs_price(bc_spot, k_cc, bc_T, bc_r_dec, bc_sig, "call")
        pnl_s = []
        for s in S_range_bc:
            stock_pnl = (s - bc_spot) * n_units
            opt_pnl   = (p_cc - max(s - k_cc, 0)) * n_units
            pnl_s.append(stock_pnl + opt_pnl)

        max_profit_s = (k_cc - bc_spot + p_cc) * n_units
        be_s = bc_spot - p_cc
        strat_info = [
            ("پرمیوم دریافتی", f"{p_cc:,.0f} ریال/واحد", green_bc),
            ("کل پرمیوم", f"{bc_format_rial(p_cc*n_units)} ریال", green_bc),
            ("حداکثر سود", f"{bc_format_rial(max_profit_s)} ریال", green_bc),
            ("Breakeven", f"{be_s:,.0f} ریال", blue_bc),
            ("سقف سود (Strike)", f"{k_cc:,} ریال", gold_bc),
        ]

    else:  # Protective Put
        st.caption("نگه‌داشتن دارایی پایه + خرید Put — بیمه در برابر ریزش")
        k_pp = st.number_input("Strike Put خریداری‌شده", value=int(bc_spot*0.95), step=5_000, key="pp_bourse_k")
        n_units_pp = st.number_input("تعداد واحد دارایی", value=lot*bc_n_contracts, step=lot, key="pp_bourse_n")

        p_pp, *_ = bc_bs_price(bc_spot, k_pp, bc_T, bc_r_dec, bc_sig, "put")
        pnl_s = []
        for s in S_range_bc:
            stock_pnl = (s - bc_spot) * n_units_pp
            opt_pnl   = max(k_pp - s, 0) * n_units_pp - p_pp * n_units_pp
            pnl_s.append(stock_pnl + opt_pnl)

        max_loss_s  = (bc_spot - k_pp + p_pp) * n_units_pp
        be_pp_s     = bc_spot + p_pp
        strat_info = [
            ("هزینه بیمه (Put)", f"{p_pp:,.0f} ریال/واحد", red_bc),
            ("کل هزینه بیمه", f"{bc_format_rial(p_pp*n_units_pp)} ریال", red_bc),
            ("حداکثر زیان بیمه‌شده", f"{bc_format_rial(max_loss_s)} ریال", red_bc),
            ("Breakeven", f"{be_pp_s:,.0f} ریال", blue_bc),
            ("کف قیمت محافظت‌شده", f"{k_pp:,} ریال", green_bc),
        ]

    # نمودار استراتژی
    fig_strat = go.Figure()
    profit_arr_s = np.array(pnl_s)
    fig_strat.add_trace(go.Scatter(
        x=S_range_bc, y=np.where(profit_arr_s>=0, profit_arr_s, 0),
        fill="tozeroy",
        fillcolor="rgba(90,170,120,0.12)" if is_dark else "rgba(26,102,64,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig_strat.add_trace(go.Scatter(
        x=S_range_bc, y=np.where(profit_arr_s<0, profit_arr_s, 0),
        fill="tozeroy",
        fillcolor="rgba(204,85,85,0.12)" if is_dark else "rgba(138,32,32,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig_strat.add_trace(go.Scatter(
        x=S_range_bc, y=profit_arr_s, mode="lines",
        line=dict(color=blue_bc, width=2.5), name=strategy_bc,
        hovertemplate="قیمت: %{x:,.0f}<br>P&L: %{y:,.0f}<extra></extra>",
    ))
    fig_strat.add_vline(x=bc_spot, line_dash="dot", line_color=blue_bc, line_width=1.5,
                         annotation_text=f"Spot {bc_spot:,}", annotation_font_color=blue_bc, annotation_font_size=9)
    fig_strat.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
    fig_strat.update_layout(**get_plot_layout(
        f"{strategy_bc} — P&L در سررسید", "قیمت دارایی پایه (ریال)", "سود / زیان (ریال)", 400
    ))
    fig_strat.update_layout(xaxis=dict(tickformat=",.0f"), yaxis=dict(tickformat=",.0f"))

    strat_col1, strat_col2 = st.columns([2, 3])
    with strat_col1:
        st.markdown(f"""
        <div style="background:{card_bc};border:1px solid {border_bc};border-radius:4px;padding:0.9rem 1rem">
            <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;color:{muted_bc};
                text-transform:uppercase;margin-bottom:0.6rem">{strategy_bc}</div>
            {''.join([f"""
            <div style="display:flex;justify-content:space-between;padding:0.25rem 0;border-bottom:1px solid {border_bc}">
                <span style="font-size:0.68rem;color:{muted_bc};font-weight:600">{lbl}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:700;color:{color}">{val}</span>
            </div>""" for lbl, val, color in strat_info])}
        </div>
        """, unsafe_allow_html=True)
    with strat_col2:
        st.plotly_chart(fig_strat, use_container_width=True)

    # ══ مقایسه سررسیدها ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">⑧ مقایسه سررسیدهای مختلف</span></div>', unsafe_allow_html=True)
    st.caption("تأثیر تغییر زمان باقیمانده روی قیمت و Greeks اختیار — انتخاب سررسید بهینه.")

    exp_durations = [7, 14, 21, 30, 45, 60, 90]
    exp_rows = []
    for d in exp_durations:
        T_d = d / 365.0
        p_d, delta_d, gamma_d, theta_d, vega_d = bc_bs_price(
            bc_spot, bc_strike, T_d, bc_r_dec, bc_sig,
            "call" if bc_is_call else "put"
        )
        time_val = p_d - max(bc_spot - bc_strike, 0) if bc_is_call else p_d - max(bc_strike - bc_spot, 0)
        exp_rows.append({
            "DTE (روز)": d,
            "قیمت BS": f"{p_d:,.0f}",
            "ارزش زمانی": f"{max(time_val,0):,.0f}",
            "Delta": f"{delta_d:.4f}",
            "Gamma": f"{gamma_d:.6f}",
            "Theta روزانه": f"{theta_d*p_d:,.0f}",
            "Vega per 1%": f"{vega_d:.4f}",
            "هزینه کل (۱ قرارداد)": f"{bc_format_rial(p_d*lot)} ریال",
        })

    df_exp = pd.DataFrame(exp_rows)
    st.dataframe(df_exp, use_container_width=True, hide_index=True)

    # نمودار قیمت BS به ازای DTE
    prices_by_dte = [float(bc_bs_price(bc_spot, bc_strike, d/365.0, bc_r_dec, bc_sig,
                                        "call" if bc_is_call else "put")[0]) for d in exp_durations]
    deltas_by_dte = [float(bc_bs_price(bc_spot, bc_strike, d/365.0, bc_r_dec, bc_sig,
                                        "call" if bc_is_call else "put")[1]) for d in exp_durations]

    fig_dte = go.Figure()
    fig_dte.add_trace(go.Bar(
        x=[f"{d}d" for d in exp_durations],
        y=prices_by_dte,
        name="قیمت BS",
        marker_color=blue_bc,
        marker_line=dict(color="rgba(0,0,0,0.1)", width=0.5),
        text=[f"{p:,.0f}" for p in prices_by_dte],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono"),
        yaxis="y",
    ))
    fig_dte.add_trace(go.Scatter(
        x=[f"{d}d" for d in exp_durations],
        y=deltas_by_dte,
        name="Delta",
        mode="lines+markers",
        line=dict(color=gold_bc, width=2),
        marker=dict(size=7, color=gold_bc),
        yaxis="y2",
    ))
    fig_dte.update_layout(**get_plot_layout(
        "قیمت اختیار و Delta به ازای DTE‌های مختلف", "DTE", "قیمت (ریال)", 360
    ))
    fig_dte.update_layout(
        yaxis=dict(title="قیمت (ریال)", tickformat=",.0f"),
        yaxis2=dict(title="Delta", overlaying="y", side="right",
                    tickformat=".3f", range=[0, 1.1] if bc_is_call else [-1.1, 0],
                    showgrid=False),
        barmode="group",
    )
    st.plotly_chart(fig_dte, use_container_width=True)

    # ══ ماشین‌حساب بازده ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">⑨ ماشین‌حساب بازده سرمایه‌گذاری</span></div>', unsafe_allow_html=True)
    st.caption("با ورود قیمت خرید و تارگت، بازده و R/R محاسبه می‌شود.")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        bc_buy_price = st.number_input("قیمت خرید اختیار (ریال)", value=int(bc_price),
                                        step=1_000, key="bc_buy_opt")
        bc_n_buy = st.number_input("تعداد قرارداد", value=1, step=1, key="bc_n_buy_opt")
    with rc2:
        bc_target_spot = st.number_input("قیمت تارگت دارایی پایه (ریال)",
                                          value=int(bc_spot*1.10), step=5_000, key="bc_target_s")
        bc_target_days = st.number_input("روز تا هدف", value=int(bc_dte*0.7),
                                          min_value=1, max_value=bc_dte, step=1, key="bc_target_d")
    with rc3:
        bc_stop_spot = st.number_input("قیمت استاپ دارایی پایه (ریال)",
                                        value=int(bc_spot*0.95), step=5_000, key="bc_stop_s")

    # محاسبه ارزش اختیار در تارگت و استاپ
    T_target = max(bc_dte - bc_target_days, 1) / 365.0
    p_target, *_ = bc_bs_price(bc_target_spot, bc_strike, T_target, bc_r_dec, bc_sig,
                                 "call" if bc_is_call else "put")
    p_stop, *_ = bc_bs_price(bc_stop_spot, bc_strike, bc_T, bc_r_dec, bc_sig,
                               "call" if bc_is_call else "put")

    total_invest = bc_buy_price * lot * bc_n_buy
    pnl_target_r = (p_target - bc_buy_price) * lot * bc_n_buy
    pnl_stop_r   = (p_stop - bc_buy_price) * lot * bc_n_buy
    ret_target   = pnl_target_r / total_invest * 100 if total_invest > 0 else 0
    ret_stop     = pnl_stop_r / total_invest * 100 if total_invest > 0 else 0
    rr_ratio     = abs(pnl_target_r / pnl_stop_r) if pnl_stop_r != 0 else float("inf")

    rr_color = green_bc if rr_ratio >= 2 else (gold_bc if rr_ratio >= 1 else red_bc)

    rcc1, rcc2, rcc3, rcc4 = st.columns(4)
    with rcc1:
        st.metric("سرمایه‌گذاری کل", f"{bc_format_rial(total_invest)} ریال")
        st.caption(f"{bc_buy_price:,} × {lot:,} × {bc_n_buy} قرارداد")
    with rcc2:
        st.metric("سود در تارگت", f"{bc_format_rial(pnl_target_r)} ریال",
                  delta=f"{ret_target:+.1f}%", delta_color="normal")
        st.caption(f"ارزش اختیار در تارگت: {p_target:,.0f} ریال")
    with rcc3:
        st.metric("زیان در استاپ", f"{bc_format_rial(pnl_stop_r)} ریال",
                  delta=f"{ret_stop:+.1f}%", delta_color="inverse")
        st.caption(f"ارزش اختیار در استاپ: {p_stop:,.0f} ریال")
    with rcc4:
        st.metric("نسبت R/R", f"{rr_ratio:.2f}",
                  delta="✓ مناسب" if rr_ratio >= 2 else ("⚠ مرزی" if rr_ratio >= 1 else "✗ ضعیف"),
                  delta_color="normal" if rr_ratio >= 2 else ("off" if rr_ratio >= 1 else "inverse"))
        st.caption(f"سود/زیان = {rr_ratio:.2f} — حداقل ۲ توصیه می‌شود")

    # ── نمودار مقایسه سود/زیان اختیار vs دارایی پایه ──
    spot_pnl_direct = [(s - bc_spot) * lot * bc_n_buy for s in S_range_bc]
    opt_pnl_roi_line = []
    for s in S_range_bc:
        T_now = bc_T
        p_now, *_ = bc_bs_price(s, bc_strike, T_now, bc_r_dec, bc_sig, "call" if bc_is_call else "put")
        opt_pnl_roi_line.append((p_now - bc_buy_price) * lot * bc_n_buy)

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(
        x=S_range_bc, y=opt_pnl_roi_line, mode="lines",
        line=dict(color=blue_bc, width=2.5), name=f"{'Call' if bc_is_call else 'Put'} Option",
        hovertemplate="Spot: %{x:,.0f}<br>P&L اختیار: %{y:,.0f}<extra></extra>",
    ))
    fig_compare.add_trace(go.Scatter(
        x=S_range_bc, y=spot_pnl_direct, mode="lines",
        line=dict(color=muted_bc, width=1.5, dash="dash"), name="خرید مستقیم دارایی",
        hovertemplate="Spot: %{x:,.0f}<br>P&L مستقیم: %{y:,.0f}<extra></extra>",
    ))
    fig_compare.add_vline(x=bc_target_spot, line_dash="dot", line_color=green_bc, line_width=1.2,
                           annotation_text=f"تارگت {bc_target_spot:,}", annotation_font_color=green_bc, annotation_font_size=9)
    fig_compare.add_vline(x=bc_stop_spot, line_dash="dot", line_color=red_bc, line_width=1.2,
                           annotation_text=f"استاپ {bc_stop_spot:,}", annotation_font_color=red_bc, annotation_font_size=9)
    fig_compare.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
    fig_compare.update_layout(**get_plot_layout(
        "مقایسه: اختیار معامله vs خرید مستقیم دارایی پایه",
        "قیمت دارایی پایه (ریال)", "سود / زیان (ریال)", 380
    ))
    fig_compare.update_layout(xaxis=dict(tickformat=",.0f"), yaxis=dict(tickformat=",.0f"))
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── چک‌لیست قبل از معامله ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">⑩ چک‌لیست قبل از ثبت سفارش</span></div>', unsafe_allow_html=True)

    checks_bc = [
        (bc_dte >= 21, f"{'✓' if bc_dte>=21 else '✗'} زمان کافی تا سررسید — {bc_dte} روز {'(حداقل ۲۱ روز توصیه می‌شود)' if bc_dte<21 else ''}"),
        (bc_iv <= 80, f"{'✓' if bc_iv<=80 else '✗'} IV در محدوده منطقی — {bc_iv}% {'(IV بالای ۸۰٪ ریسک بالا)' if bc_iv>80 else ''}"),
        (abs(bc_delta) <= 0.6, f"{'✓' if abs(bc_delta)<=0.6 else '✗'} Delta مناسب — {bc_delta:.3f} {'(Delta بالا = ریسک ITM)' if abs(bc_delta)>0.6 else ''}"),
        (rr_ratio >= 1.5, f"{'✓' if rr_ratio>=1.5 else '✗'} نسبت R/R — {rr_ratio:.2f} {'(حداقل ۱.۵ توصیه می‌شود)' if rr_ratio<1.5 else ''}"),
        (total_invest > 0, f"{'✓' if total_invest>0 else '✗'} پارامترهای محاسبه کامل است"),
        (bc_n_contracts <= 10 or total_invest < 500_000_000, f"{'✓' if bc_n_contracts<=10 else '⚠'} {'حجم معامله منطقی' if bc_n_contracts<=10 else f'حجم بالا — {bc_n_contracts} قرارداد — مطمئن هستید؟'}"),
    ]

    passed_bc = sum(1 for ok, _ in checks_bc if ok)
    chk_color = green_bc if passed_bc >= 5 else (gold_bc if passed_bc >= 3 else red_bc)

    st.markdown(f"""
    <div style="background:{card_bc};border:1px solid {border_bc};border-radius:4px;padding:0.9rem 1.1rem">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.7rem">
            <span style="font-size:0.62rem;font-weight:700;letter-spacing:0.12em;
                color:{muted_bc};text-transform:uppercase">آمادگی معامله</span>
            <span style="font-family:'JetBrains Mono',monospace;font-weight:700;
                font-size:0.9rem;color:{chk_color}">{passed_bc}/{len(checks_bc)} ✓</span>
        </div>
        {"".join(f'''<div style="display:flex;align-items:flex-start;gap:8px;padding:0.28rem 0;border-bottom:1px solid {border_bc}">
            <span style="color:{green_bc if ok else red_bc};font-size:0.85rem;min-width:16px;margin-top:1px">{"✓" if ok else "✗"}</span>
            <span style="font-size:0.72rem;color:var(--{'silver' if ok else 'muted'});line-height:1.5">{txt}</span>
        </div>''' for ok, txt in checks_bc)}
        <div style="margin-top:0.6rem;padding-top:0.4rem;border-top:1px solid {border_bc};
            font-size:0.62rem;color:{muted_bc};line-height:1.7">
            💡 <strong>راهنما:</strong> اختیار ایده‌آل: OTM · IV ۲۰–۶۰٪ · Delta ۰.۲–۰.۴ · DTE ۲۱–۴۵ روز · R/R بالای ۲<br>
            📌 بورس کالای ایران: هر قرارداد = {lot:,} واحد · سررسید ماهانه · اعمال اروپایی
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── یادداشت تحلیلگر ──
    st.markdown('<div class="bp-section"><span class="bp-section-text">یادداشت تحلیلگر</span></div>', unsafe_allow_html=True)
    bc_note = st.text_area("یادداشت معامله (ذخیره در session)", placeholder="دلیل معامله، سطوح حمایت/مقاومت، کاتالیست...",
                            key="bc_note", height=80)
    if st.button("💾 ذخیره یادداشت این معامله", use_container_width=True, key="bc_save_note"):
        if "bc_trade_notes" not in st.session_state:
            st.session_state["bc_trade_notes"] = []
        note_entry = {
            "زمان": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "دارایی": ul["label"],
            "نوع": "Call" if bc_is_call else "Put",
            "Spot": bc_spot,
            "Strike": bc_strike,
            "DTE": bc_dte,
            "IV": bc_iv,
            "قیمت BS": round(bc_price),
            "یادداشت": bc_note,
        }
        st.session_state["bc_trade_notes"].append(note_entry)
        st.success("یادداشت ذخیره شد.")

    bc_notes = st.session_state.get("bc_trade_notes", [])
    if bc_notes:
        st.markdown('<div class="bp-section"><span class="bp-section-text">یادداشت‌های قبلی</span></div>', unsafe_allow_html=True)
        df_notes = pd.DataFrame(bc_notes)
        st.dataframe(df_notes, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# TAB IME LIVE — گواهی سپرده کالایی بورس کالای ایران (API زنده)
# ════════════════════════════════════════════════════════════════════
with tab_ime:
    import urllib.request as _ureq
    import json as _json_ime
    import time as _time_ime

    IME_API_KEY  = "B3dExgPIwrJFhn47YH9hxDrRnmAq6BQU"
    IME_BASE_URL = "https://api.ime.co.ir/api/v1"

    green_im = "#5aaa78" if is_dark else "#1a6640"
    gold_im  = "#c8a84b" if is_dark else "#8a6a1a"
    red_im   = "#cc5555" if is_dark else "#8a2020"
    blue_im  = "#5b9bd5"
    muted_im = "#888888" if is_dark else "#555550"
    card_im  = "#222222" if is_dark else "#e4e4e0"
    border_im= "rgba(180,180,180,0.1)" if is_dark else "rgba(60,60,55,0.15)"

    # ── نگاشت نماد → اطلاعات تکمیلی ──
    IME_SYMBOL_META = {
        "TLDY":  {"label":"صندوق طلای لوتوس",      "icon":"🪙","group":"طلا"},
        "GOLD":  {"label":"شمش طلا ۹۹۵+",           "icon":"🥇","group":"طلا"},
        "KHRB":  {"label":"صندوق کهربا ۱۰",          "icon":"🟡","group":"طلا"},
        "ZNAY":  {"label":"صندوق دنای زاگرس",        "icon":"🔶","group":"طلا"},
        "DRSH":  {"label":"صندوق درخشان",             "icon":"✨","group":"طلا"},
        "SILV":  {"label":"شمش نقره ۹۹۹.۹",          "icon":"🥈","group":"نقره"},
        "COPP":  {"label":"مس کاتد",                  "icon":"🟠","group":"فلزات"},
        "ZINC":  {"label":"شمش روی",                  "icon":"🔘","group":"فلزات"},
        "STLR":  {"label":"میلگرد",                   "icon":"⚙", "group":"فولاد"},
        "BITM":  {"label":"قیر",                      "icon":"🛢", "group":"نفتی"},
    }

    def ime_fmt(v, decimals=0):
        """فرمت عدد ریال با جداکننده"""
        try:
            if decimals == 0:
                return f"{int(v):,}"
            return f"{float(v):,.{decimals}f}"
        except Exception:
            return str(v)

    def ime_delta_color(v):
        try:
            f = float(v)
            return green_im if f > 0 else (red_im if f < 0 else muted_im)
        except Exception:
            return muted_im

    def ime_delta_sign(v):
        try:
            f = float(v)
            return f"+{f:.2f}" if f > 0 else f"{f:.2f}"
        except Exception:
            return str(v)

    # ── تابع فراخوانی API ──
    @st.cache_data(show_spinner=False, ttl=60)
    def ime_fetch(endpoint: str, params: dict | None = None) -> dict | None:
        """
        فراخوانی API بورس کالا.
        endpoint مثل: /commodity-deposit-certificate/last-changes
        """
        try:
            qs = f"?api_key={IME_API_KEY}"
            if params:
                for k, v in params.items():
                    qs += f"&{k}={v}"
            url = f"{IME_BASE_URL}{endpoint}{qs}"
            req = _ureq.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Portfolio360/3.2",
                    "X-Api-Key": IME_API_KEY,
                }
            )
            with _ureq.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
                return _json_ime.loads(raw)
        except Exception as e:
            return {"_error": str(e)}

    @st.cache_data(show_spinner=False, ttl=60)
    def ime_fetch_symbol(contract_code: str) -> dict | None:
        """دریافت داده یک نماد خاص"""
        return ime_fetch(f"/commodity-deposit-certificate/{contract_code}")

    @st.cache_data(show_spinner=False, ttl=300)
    def ime_fetch_history(contract_code: str, from_date: str, to_date: str) -> dict | None:
        """دریافت تاریخچه قیمت"""
        return ime_fetch(
            f"/commodity-deposit-certificate/{contract_code}/history",
            params={"from": from_date, "to": to_date}
        )

    def ime_parse_last_changes(data: dict) -> list[dict]:
        """
        پارس داده آرایه‌ای از last-changes API.
        ساختار: data["data"] = [ {...}, {...} ]
        """
        if not data or "_error" in data:
            return []
        rows = data.get("data", data.get("result", []))
        if isinstance(rows, dict):
            rows = [rows]
        return rows if isinstance(rows, list) else []

    # ══════════════════════════════════════════════════════════════
    # هدر تب
    # ══════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
        margin-bottom:1rem;padding-bottom:0.8rem;border-bottom:2px solid var(--border2)">
        <div style="display:flex;align-items:center;gap:12px">
            <span style="font-size:1.6rem">📡</span>
            <div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.88rem;
                    font-weight:700;letter-spacing:0.1em;color:var(--accent)">
                    IME LIVE — گواهی سپرده کالایی
                </div>
                <div style="font-size:0.62rem;color:{muted_im};letter-spacing:0.1em;margin-top:2px">
                    بورس کالای ایران · داده زنده از API رسمی · کلید: {IME_API_KEY[:8]}…
                </div>
            </div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
            color:{muted_im};text-align:right;line-height:1.8">
            نرخ تازه‌سازی: ۶۰ ثانیه<br>
            منبع: api.ime.co.ir
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══ بخش ۱ — تابلوی زنده همه نمادها ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">① تابلوی زنده — آخرین تغییرات همه گواهی‌ها</span></div>', unsafe_allow_html=True)

    ime_col_ctrl1, ime_col_ctrl2, ime_col_ctrl3 = st.columns([3, 1, 1])
    with ime_col_ctrl1:
        st.caption("داده از endpoint: `/commodity-deposit-certificate/last-changes` — هر ۶۰ ثانیه یک‌بار تازه می‌شود.")
    with ime_col_ctrl2:
        auto_refresh = st.checkbox("بروزرسانی خودکار", value=False, key="ime_auto")
    with ime_col_ctrl3:
        manual_refresh = st.button("🔄 بروزرسانی", use_container_width=True, key="ime_refresh")

    if manual_refresh:
        st.cache_data.clear()

    # فراخوانی API
    with st.spinner("در حال دریافت داده از بورس کالا…"):
        raw_all = ime_fetch("/commodity-deposit-certificate/last-changes")

    rows_all = ime_parse_last_changes(raw_all)

    if raw_all and "_error" in raw_all:
        err_msg = raw_all["_error"]
        st.markdown(f"""
        <div style="background:{card_im};border:1px solid {red_im};border-left:4px solid {red_im};
            border-radius:4px;padding:1rem 1.2rem;margin-bottom:1rem">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                font-weight:700;color:{red_im};margin-bottom:0.4rem">❌ خطا در اتصال به API</div>
            <div style="font-size:0.72rem;color:{muted_im};line-height:1.7">
                {err_msg}<br><br>
                <strong>راه‌حل‌های احتمالی:</strong><br>
                ① در صورت استفاده از Streamlit Cloud، دامنه <code>api.ime.co.ir</code> را در egress settings اضافه کنید.<br>
                ② یا برنامه را روی سرور ایرانی (VPS) اجرا کنید.<br>
                ③ برای تست، از داده نمونه زیر استفاده کنید.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── داده نمونه برای تست بدون اتصال ──
        use_demo = st.checkbox("استفاده از داده نمونه (Demo Mode)", value=True, key="ime_demo")
        if use_demo:
            rows_all = [
                {"commodity":"شمش طلا","contract_code":"CD1GOB0001",
                 "contract_description":"گواهی سپرده پیوسته شمش طلای +995",
                 "contract_size":10,"contract_currency":"ریال",
                 "date_y":"1404-07-01","py":12609320,
                 "pf_time":"12:00:00","pf":13230000,"pfc":620680,"pfp":4.92,
                 "pmax":13239780,"pmaxc":630460,"pmaxp":5.0,
                 "pmin":13130000,"pminc":520680,"pminp":4.13,
                 "pl_time":"16:43:27","pl":13239780,"plc":630460,"plp":5.0,
                 "tno":2714,"tvol":1198661,"tval":15869015001,"tval_unit":"هزار ریال",
                 "pd1":13239780,"pd2":13239770,"pd3":13230040,
                 "qd1":558699,"qd2":32,"qd3":203,
                 "po1":0,"po2":0,"po3":0,
                 "qo1":0,"qo2":0,"qo3":0,
                 "date_update":"1404-07-02","time_update":"16:43:56"},
                {"commodity":"شمش نقره","contract_code":"CD1SIB0001",
                 "contract_description":"گواهی سپرده پیوسته شمش نقره 999.9",
                 "contract_size":500,"contract_currency":"ریال",
                 "date_y":"1404-07-01","py":376500,
                 "pf_time":"12:00:05","pf":389200,"pfc":12700,"pfp":3.37,
                 "pmax":392000,"pmaxc":15500,"pmaxp":4.12,
                 "pmin":385000,"pminc":8500,"pminp":2.26,
                 "pl_time":"16:40:10","pl":390800,"plc":14300,"plp":3.80,
                 "tno":841,"tvol":320540,"tval":1251250,"tval_unit":"هزار ریال",
                 "pd1":390800,"pd2":390500,"pd3":390000,
                 "qd1":12000,"qd2":500,"qd3":200,
                 "po1":391000,"po2":391500,"po3":392000,
                 "qo1":2000,"qo2":500,"qo3":300,
                 "date_update":"1404-07-02","time_update":"16:43:56"},
                {"commodity":"مس کاتد","contract_code":"CD1CUB0001",
                 "contract_description":"گواهی سپرده مس کاتد",
                 "contract_size":1000,"contract_currency":"ریال",
                 "date_y":"1404-07-01","py":2210000,
                 "pf_time":"12:00:10","pf":2200000,"pfc":-10000,"pfp":-0.45,
                 "pmax":2215000,"pmaxc":5000,"pmaxp":0.23,
                 "pmin":2195000,"pminc":-15000,"pminp":-0.68,
                 "pl_time":"16:38:00","pl":2205000,"plc":-5000,"plp":-0.23,
                 "tno":290,"tvol":48700,"tval":1073835,"tval_unit":"هزار ریال",
                 "pd1":2205000,"pd2":2204000,"pd3":2203000,
                 "qd1":5000,"qd2":1000,"qd3":500,
                 "po1":2206000,"po2":2207000,"po3":2208000,
                 "qo1":3000,"qo2":2000,"qo3":1000,
                 "date_update":"1404-07-02","time_update":"16:43:56"},
                {"commodity":"شمش روی","contract_code":"CD1ZNB0001",
                 "contract_description":"گواهی سپرده شمش روی",
                 "contract_size":1000,"contract_currency":"ریال",
                 "date_y":"1404-07-01","py":480000,
                 "pf_time":"12:01:00","pf":478000,"pfc":-2000,"pfp":-0.42,
                 "pmax":482000,"pmaxc":2000,"pmaxp":0.42,
                 "pmin":476000,"pminc":-4000,"pminp":-0.83,
                 "pl_time":"16:35:00","pl":479100,"plc":-900,"plp":-0.19,
                 "tno":105,"tvol":22000,"tval":1054020,"tval_unit":"هزار ریال",
                 "pd1":479100,"pd2":479000,"pd3":478500,
                 "qd1":1000,"qd2":500,"qd3":200,
                 "po1":480000,"po2":481000,"po3":482000,
                 "qo1":500,"qo2":300,"qo3":200,
                 "date_update":"1404-07-02","time_update":"16:43:56"},
            ]
            st.info("📊 حالت Demo: داده نمونه برای نمایش رابط کاربری استفاده می‌شود.")
    else:
        use_demo = False

    if rows_all:
        # ── خلاصه آخرین بروزرسانی ──
        last_update = rows_all[0].get("date_update","—") + " " + rows_all[0].get("time_update","")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.8rem;
            padding:0.5rem 0.9rem;background:{card_im};border:1px solid {border_im};
            border-radius:4px">
            <span style="font-size:0.75rem">🕐</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                color:{muted_im}">آخرین بروزرسانی: <strong style="color:var(--accent)">{last_update}</strong></span>
            <span style="font-size:0.65rem;color:{muted_im}">|</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                color:{muted_im}">{len(rows_all)} نماد فعال</span>
            {"<span style='font-size:0.6rem;padding:2px 8px;background:rgba(90,170,120,0.15);border:1px solid "+green_im+";border-radius:10px;color:"+green_im+";font-family:monospace;font-size:0.62rem'>● LIVE</span>" if not use_demo else "<span style='font-size:0.6rem;padding:2px 8px;background:rgba(200,168,75,0.15);border:1px solid "+gold_im+";border-radius:10px;color:"+gold_im+";font-family:monospace;font-size:0.62rem'>◉ DEMO</span>"}
        </div>
        """, unsafe_allow_html=True)

        # ── کارت‌های خلاصه هر نماد ──
        n_cols = min(len(rows_all), 4)
        card_cols = st.columns(n_cols)
        for i, row in enumerate(rows_all[:8]):
            col_i = i % n_cols
            commodity = row.get("commodity", "—")
            pl   = row.get("pl", 0)
            plc  = row.get("plc", 0)
            plp  = row.get("plp", 0)
            py   = row.get("py", 0)
            tno  = row.get("tno", 0)
            tvol = row.get("tvol", 0)
            pmax = row.get("pmax", 0)
            pmin = row.get("pmin", 0)
            pl_time = row.get("pl_time", "—")
            clr = green_im if float(plc or 0) >= 0 else red_im
            sign = "▲" if float(plc or 0) >= 0 else "▼"

            with card_cols[col_i]:
                st.markdown(f"""
                <div style="background:{card_im};border:1px solid {border_im};
                    border-top:3px solid {clr};border-radius:4px;
                    padding:0.8rem 0.9rem;margin-bottom:0.6rem">
                    <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.12em;
                        color:{muted_im};text-transform:uppercase;margin-bottom:0.35rem">
                        {commodity}
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.15rem;
                        font-weight:700;color:var(--accent);line-height:1.1">
                        {ime_fmt(pl)}
                    </div>
                    <div style="font-size:0.62rem;color:{muted_im};margin-bottom:0.4rem">ریال</div>
                    <div style="display:flex;align-items:center;gap:6px;margin-bottom:0.35rem">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                            font-weight:700;color:{clr}">{sign} {ime_fmt(abs(float(plc or 0)))}
                        </span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                            font-weight:600;color:{clr}">({ime_delta_sign(plp)}%)</span>
                    </div>
                    <div style="border-top:1px solid {border_im};padding-top:0.35rem;
                        display:flex;flex-direction:column;gap:2px">
                        <div style="display:flex;justify-content:space-between">
                            <span style="font-size:0.6rem;color:{muted_im}">دیروز</span>
                            <span style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                                color:var(--accent)">{ime_fmt(py)}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between">
                            <span style="font-size:0.6rem;color:{muted_im}">بالا / پایین</span>
                            <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                                color:{muted_im}">{ime_fmt(pmax)} / {ime_fmt(pmin)}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between">
                            <span style="font-size:0.6rem;color:{muted_im}">معاملات / حجم</span>
                            <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                                color:{muted_im}">{ime_fmt(tno)} / {ime_fmt(tvol)}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between">
                            <span style="font-size:0.6rem;color:{muted_im}">آخرین قیمت</span>
                            <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                                color:{muted_im}">{pl_time}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── جدول کامل ──
        st.markdown('<div class="bp-section"><span class="bp-section-text">جدول کامل تابلوی معاملات</span></div>', unsafe_allow_html=True)

        table_rows = []
        for row in rows_all:
            plc_v = float(row.get("plc", 0) or 0)
            plp_v = float(row.get("plp", 0) or 0)
            tval_v = float(row.get("tval", 0) or 0)
            tval_unit = row.get("tval_unit", "هزار ریال")
            table_rows.append({
                "کالا":           row.get("commodity", "—"),
                "کد قرارداد":    row.get("contract_code", "—"),
                "آخرین قیمت":    ime_fmt(row.get("pl", 0)),
                "تغییر":         f"{'+' if plc_v>=0 else ''}{ime_fmt(plc_v)}",
                "تغییر %":       f"{'+' if plp_v>=0 else ''}{plp_v:.2f}%",
                "بالاترین":      ime_fmt(row.get("pmax", 0)),
                "پایین‌ترین":    ime_fmt(row.get("pmin", 0)),
                "قیمت دیروز":    ime_fmt(row.get("py", 0)),
                "تعداد معاملات": ime_fmt(row.get("tno", 0)),
                "حجم":           ime_fmt(row.get("tvol", 0)),
                "ارزش (هزار ریال)": ime_fmt(tval_v),
                "زمان":          row.get("pl_time","—"),
            })
        df_ime_table = pd.DataFrame(table_rows)
        st.dataframe(df_ime_table, use_container_width=True, hide_index=True)

        # ── نمودار مقایسه تغییرات ──
        st.markdown('<div class="bp-section"><span class="bp-section-text">نمودار تغییر درصدی نمادها</span></div>', unsafe_allow_html=True)

        names_chart  = [r.get("commodity","?") for r in rows_all]
        change_chart = [float(r.get("plp", 0) or 0) for r in rows_all]
        colors_chart = [green_im if v >= 0 else red_im for v in change_chart]

        fig_ime_bar = go.Figure()
        fig_ime_bar.add_trace(go.Bar(
            x=names_chart,
            y=change_chart,
            marker_color=colors_chart,
            marker_line=dict(color="rgba(0,0,0,0.15)", width=0.5),
            text=[f"{v:+.2f}%" for v in change_chart],
            textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono"),
            hovertemplate="%{x}<br>تغییر: %{y:+.2f}%<extra></extra>",
        ))
        fig_ime_bar.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
        fig_ime_bar.update_layout(**get_plot_layout(
            "تغییر درصدی نمادهای گواهی سپرده کالایی", "", "تغییر (%)", 360))
        fig_ime_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_ime_bar, use_container_width=True)

        # ── نمودار حجم معاملات ──
        st.markdown('<div class="bp-section"><span class="bp-section-text">حجم و ارزش معاملات</span></div>', unsafe_allow_html=True)

        vol_chart  = [float(r.get("tvol", 0) or 0) for r in rows_all]
        val_chart  = [float(r.get("tval", 0) or 0) / 1e6 for r in rows_all]  # میلیون ریال

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=names_chart, y=vol_chart,
            name="حجم معاملات",
            marker_color=blue_im, opacity=0.8,
            yaxis="y",
            hovertemplate="%{x}<br>حجم: %{y:,.0f}<extra></extra>",
        ))
        fig_vol.add_trace(go.Scatter(
            x=names_chart, y=val_chart,
            name="ارزش (میلیون ریال)",
            mode="lines+markers",
            line=dict(color=gold_im, width=2.5),
            marker=dict(size=9, color=gold_im),
            yaxis="y2",
            hovertemplate="%{x}<br>ارزش: %{y:,.1f} میلیون ریال<extra></extra>",
        ))
        fig_vol.update_layout(**get_plot_layout(
            "حجم و ارزش معاملات گواهی‌های کالایی", "", "حجم (واحد)", 380))
        fig_vol.update_layout(
            yaxis2=dict(
                title="ارزش (میلیون ریال)", overlaying="y", side="right",
                showgrid=False, tickformat=",.0f",
                tickfont=dict(size=9, color=gold_im),
            ),
            barmode="group",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # ══ بخش ۲ — جزئیات نماد انتخابی ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">② جزئیات نماد — عرضه و تقاضا + سفارش‌ها</span></div>', unsafe_allow_html=True)

    if rows_all:
        symbol_options = {r.get("contract_code","?"): r.get("commodity","?") for r in rows_all}
        sel_code = st.selectbox(
            "انتخاب نماد برای تحلیل جزئیات",
            list(symbol_options.keys()),
            format_func=lambda k: f"{symbol_options[k]}  ·  {k}",
            key="ime_sel_symbol"
        )

        # پیدا کردن ردیف انتخابی
        sel_row = next((r for r in rows_all if r.get("contract_code") == sel_code), None)

        if sel_row:
            c1, c2 = st.columns(2)

            # ── عرضه و تقاضا ──
            with c1:
                st.markdown(f"""
                <div style="background:{card_im};border:1px solid {border_im};
                    border-radius:4px;padding:0.9rem 1rem;margin-bottom:0.5rem">
                    <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.15em;
                        color:{muted_im};text-transform:uppercase;margin-bottom:0.6rem">
                        📋 عرضه و تقاضا — {sel_row.get("commodity","—")}
                    </div>
                    <!-- هدر -->
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
                        gap:4px;padding:0.3rem 0;border-bottom:2px solid {border_im};
                        margin-bottom:0.3rem">
                        <span style="font-size:0.6rem;font-weight:700;color:{green_im};text-align:right">حجم خرید</span>
                        <span style="font-size:0.6rem;font-weight:700;color:{green_im};text-align:right">قیمت خرید</span>
                        <span style="font-size:0.6rem;font-weight:700;color:{red_im}">قیمت فروش</span>
                        <span style="font-size:0.6rem;font-weight:700;color:{red_im}">حجم فروش</span>
                    </div>
                    <!-- ردیف‌ها -->
                    {''.join([f"""
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
                        gap:4px;padding:0.22rem 0;border-bottom:1px solid {border_im};
                        {'background:rgba(91,155,213,0.07)' if i==0 else ''}">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                            font-weight:{'700' if i==0 else '500'};color:{green_im};text-align:right">
                            {ime_fmt(sel_row.get(f"qd{i+1}",0))}
                        </span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                            font-weight:{'700' if i==0 else '500'};color:{green_im};text-align:right">
                            {ime_fmt(sel_row.get(f"pd{i+1}",0))}
                        </span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                            font-weight:{'700' if i==0 else '500'};color:{red_im}">
                            {ime_fmt(sel_row.get(f"po{i+1}",0)) if sel_row.get(f"po{i+1}",0) else "—"}
                        </span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                            font-weight:{'700' if i==0 else '500'};color:{red_im}">
                            {ime_fmt(sel_row.get(f"qo{i+1}",0)) if sel_row.get(f"qo{i+1}",0) else "—"}
                        </span>
                    </div>""" for i in range(3)])}
                    <!-- خلاصه Bid/Ask -->
                    <div style="margin-top:0.6rem;padding:0.4rem 0.6rem;
                        background:rgba(91,155,213,0.06);border-radius:3px;
                        display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <span style="font-size:0.58rem;color:{muted_im}">بهترین خرید: </span>
                            <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                font-weight:700;color:{green_im}">{ime_fmt(sel_row.get("pd1",0))}</span>
                        </div>
                        <div>
                            <span style="font-size:0.58rem;color:{muted_im}">اسپرد: </span>
                            <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                font-weight:700;color:{gold_im}">
                                {ime_fmt(abs(float(sel_row.get("po1",0) or 0) - float(sel_row.get("pd1",0) or 0)))}
                            </span>
                        </div>
                        <div>
                            <span style="font-size:0.58rem;color:{muted_im}">بهترین فروش: </span>
                            <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                font-weight:700;color:{red_im}">
                                {ime_fmt(sel_row.get("po1",0)) if sel_row.get("po1",0) else "—"}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── مشخصات قرارداد ──
            with c2:
                plp_v  = float(sel_row.get("plp", 0) or 0)
                pmaxp_v= float(sel_row.get("pmaxp",0) or 0)
                pminp_v= float(sel_row.get("pminp",0) or 0)
                tval_v = float(sel_row.get("tval", 0) or 0)
                tval_unit = sel_row.get("tval_unit","هزار ریال")

                # محاسبه Range روز
                pl_v   = float(sel_row.get("pl", 1) or 1)
                pmax_v = float(sel_row.get("pmax", 1) or 1)
                pmin_v = float(sel_row.get("pmin", 1) or 1)
                day_range_pct = (pmax_v - pmin_v) / pl_v * 100 if pl_v else 0

                details = [
                    ("کد قرارداد",       sel_row.get("contract_code","—"),  blue_im),
                    ("توضیح",             sel_row.get("contract_description","—")[:35], muted_im),
                    ("اندازه قرارداد",    f"{ime_fmt(sel_row.get('contract_size',0))} واحد", gold_im),
                    ("ارز",               sel_row.get("contract_currency","ریال"), muted_im),
                    ("تاریخ معامله",      sel_row.get("date_y","—"), muted_im),
                    ("قیمت پایانی دیروز",f"{ime_fmt(sel_row.get('py',0))} ریال", muted_im),
                    ("اولین قیمت",       f"{ime_fmt(sel_row.get('pf',0))} ریال ({ime_delta_sign(sel_row.get('pfp',0))}%)", blue_im),
                    ("بالاترین",         f"{ime_fmt(pmax_v)} ({ime_delta_sign(pmaxp_v)}%)", green_im),
                    ("پایین‌ترین",       f"{ime_fmt(pmin_v)} ({ime_delta_sign(pminp_v)}%)", red_im),
                    ("دامنه روز",         f"{day_range_pct:.2f}%", gold_im),
                    ("تعداد معاملات",     ime_fmt(sel_row.get("tno",0)), blue_im),
                    ("حجم معاملات",      ime_fmt(sel_row.get("tvol",0)), blue_im),
                    ("ارزش معاملات",     f"{ime_fmt(tval_v)} {tval_unit}", gold_im),
                ]
                st.markdown(f"""
                <div style="background:{card_im};border:1px solid {border_im};
                    border-radius:4px;padding:0.9rem 1rem">
                    <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.15em;
                        color:{muted_im};text-transform:uppercase;margin-bottom:0.6rem">
                        📌 مشخصات کامل قرارداد
                    </div>
                    {''.join([f"""
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;
                        padding:0.2rem 0;border-bottom:1px solid {border_im}">
                        <span style="font-size:0.63rem;color:{muted_im};font-weight:600;
                            min-width:110px">{lbl}</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                            font-weight:600;color:{color};text-align:right;flex:1">{val}</span>
                    </div>""" for lbl, val, color in details])}
                </div>
                """, unsafe_allow_html=True)

            # ── نمودار ميله Bid/Ask ──
            st.markdown('<div class="bp-section"><span class="bp-section-text">نمودار سفارش‌های عرضه و تقاضا</span></div>', unsafe_allow_html=True)

            bid_prices = [float(sel_row.get(f"pd{i}",0) or 0) for i in range(1,4)]
            bid_qtys   = [float(sel_row.get(f"qd{i}",0) or 0) for i in range(1,4)]
            ask_prices = [float(sel_row.get(f"po{i}",0) or 0) for i in range(1,4)]
            ask_qtys   = [float(sel_row.get(f"qo{i}",0) or 0) for i in range(1,4)]

            # فیلتر سفارش‌های صفر
            valid_bids = [(p, q) for p, q in zip(bid_prices, bid_qtys) if p > 0]
            valid_asks = [(p, q) for p, q in zip(ask_prices, ask_qtys) if p > 0]

            if valid_bids or valid_asks:
                fig_ob = go.Figure()
                if valid_bids:
                    fig_ob.add_trace(go.Bar(
                        x=[f"خرید {i+1}: {ime_fmt(p)}" for i,(p,q) in enumerate(valid_bids)],
                        y=[q for p,q in valid_bids],
                        name="خرید (Bid)",
                        marker_color=green_im,
                        marker_line=dict(color="rgba(0,0,0,0.1)", width=0.5),
                        text=[ime_fmt(q) for p,q in valid_bids],
                        textposition="outside",
                        textfont=dict(size=9, family="JetBrains Mono"),
                    ))
                if valid_asks:
                    fig_ob.add_trace(go.Bar(
                        x=[f"فروش {i+1}: {ime_fmt(p)}" for i,(p,q) in enumerate(valid_asks)],
                        y=[q for p,q in valid_asks],
                        name="فروش (Ask)",
                        marker_color=red_im,
                        marker_line=dict(color="rgba(0,0,0,0.1)", width=0.5),
                        text=[ime_fmt(q) for p,q in valid_asks],
                        textposition="outside",
                        textfont=dict(size=9, family="JetBrains Mono"),
                    ))
                fig_ob.update_layout(**get_plot_layout(
                    f"Order Book — {sel_row.get('commodity','')}", "", "حجم سفارش", 320))
                fig_ob.update_layout(barmode="group")
                st.plotly_chart(fig_ob, use_container_width=True)

    # ══ بخش ۳ — تاریخچه قیمت (API /history) ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">③ تاریخچه قیمت — نمودار شمعی</span></div>', unsafe_allow_html=True)
    st.caption("endpoint: `/commodity-deposit-certificate/{contract_code}/history` — بازه تاریخ شمسی")

    if rows_all:
        hist_col1, hist_col2, hist_col3 = st.columns([2, 1, 1])
        with hist_col1:
            hist_symbol = st.selectbox(
                "نماد تاریخچه",
                list(symbol_options.keys()),
                format_func=lambda k: f"{symbol_options[k]}  ·  {k}",
                key="ime_hist_sym"
            )
        with hist_col2:
            hist_from = st.text_input("از تاریخ (شمسی)", value="1404-01-01", key="ime_hist_from")
        with hist_col3:
            hist_to   = st.text_input("تا تاریخ (شمسی)", value="1404-07-02", key="ime_hist_to")

        if st.button("دریافت تاریخچه", use_container_width=True, key="ime_hist_fetch"):
            with st.spinner("در حال دریافت تاریخچه…"):
                raw_hist = ime_fetch_history(hist_symbol, hist_from, hist_to)
            st.session_state["ime_hist_raw"]  = raw_hist
            st.session_state["ime_hist_sym_lbl"] = symbol_options.get(hist_symbol, hist_symbol)

        raw_hist = st.session_state.get("ime_hist_raw")
        hist_lbl = st.session_state.get("ime_hist_sym_lbl", "")

        if raw_hist:
            if "_error" in raw_hist:
                st.warning(f"خطا در دریافت تاریخچه: {raw_hist['_error']}")
                # داده نمونه تاریخچه
                if use_demo:
                    import random
                    random.seed(42)
                    base = 10_000_000
                    hist_demo_rows = []
                    for day in range(120):
                        base = base * (1 + random.gauss(0.002, 0.015))
                        hist_demo_rows.append({
                            "date": f"1404-0{(day//30)+1:01d}-{(day%30)+1:02d}",
                            "open":  int(base * 0.99),
                            "high":  int(base * 1.012),
                            "low":   int(base * 0.988),
                            "close": int(base),
                            "volume": random.randint(500000, 2000000),
                        })
                    hist_rows = hist_demo_rows
                    st.info("داده نمونه تاریخچه برای نمایش رابط کاربری")
                else:
                    hist_rows = []
            else:
                hist_rows = raw_hist.get("data", raw_hist.get("result", []))
                if isinstance(hist_rows, dict):
                    hist_rows = [hist_rows]

            if hist_rows:
                # ── نمودار شمعی ──
                opens  = [float(r.get("open",  r.get("pf",  0)) or 0) for r in hist_rows]
                highs  = [float(r.get("high",  r.get("pmax",0)) or 0) for r in hist_rows]
                lows   = [float(r.get("low",   r.get("pmin",0)) or 0) for r in hist_rows]
                closes = [float(r.get("close", r.get("pl",  0)) or 0) for r in hist_rows]
                dates  = [r.get("date", r.get("date_y", str(i))) for i, r in enumerate(hist_rows)]
                vols   = [float(r.get("volume", r.get("tvol", 0)) or 0) for r in hist_rows]

                fig_candle = go.Figure()
                # شمعی
                fig_candle.add_trace(go.Candlestick(
                    x=dates, open=opens, high=highs, low=lows, close=closes,
                    name=hist_lbl,
                    increasing=dict(line=dict(color=green_im, width=1.5),
                                    fillcolor="rgba(90,170,120,0.7)"),
                    decreasing=dict(line=dict(color=red_im, width=1.5),
                                    fillcolor="rgba(204,85,85,0.7)"),
                ))
                # MA 20
                if len(closes) >= 20:
                    ma20 = pd.Series(closes).rolling(20).mean().values
                    fig_candle.add_trace(go.Scatter(
                        x=dates, y=ma20, mode="lines",
                        name="MA20", line=dict(color=blue_im, width=1.5, dash="dot"),
                    ))
                # MA 50
                if len(closes) >= 50:
                    ma50 = pd.Series(closes).rolling(50).mean().values
                    fig_candle.add_trace(go.Scatter(
                        x=dates, y=ma50, mode="lines",
                        name="MA50", line=dict(color=gold_im, width=1.5, dash="dash"),
                    ))

                fig_candle.update_layout(**get_plot_layout(
                    f"نمودار شمعی — {hist_lbl}", "تاریخ", "قیمت (ریال)", 500))
                fig_candle.update_layout(
                    xaxis_rangeslider_visible=False,
                    yaxis=dict(tickformat=",.0f"),
                )
                st.plotly_chart(fig_candle, use_container_width=True)

                # ── حجم ──
                vol_colors_h = []
                for i in range(len(closes)):
                    if i == 0:
                        vol_colors_h.append(blue_im)
                    else:
                        vol_colors_h.append(green_im if closes[i] >= closes[i-1] else red_im)

                fig_vol_h = go.Figure()
                fig_vol_h.add_trace(go.Bar(
                    x=dates, y=vols,
                    marker_color=vol_colors_h,
                    marker_line=dict(width=0),
                    name="حجم",
                    opacity=0.8,
                    hovertemplate="%{x}<br>حجم: %{y:,.0f}<extra></extra>",
                ))
                fig_vol_h.update_layout(**get_plot_layout(
                    "حجم معاملات تاریخی", "تاریخ", "حجم", 200))
                st.plotly_chart(fig_vol_h, use_container_width=True)

                # ── آمار تاریخچه ──
                closes_arr = np.array([c for c in closes if c > 0])
                if len(closes_arr) > 1:
                    returns_h = np.diff(closes_arr) / closes_arr[:-1]
                    ca, cb, cc, cd = st.columns(4)
                    with ca:
                        total_ret = (closes_arr[-1] / closes_arr[0] - 1) * 100
                        clr_tr = green_im if total_ret >= 0 else red_im
                        st.metric("بازده کل دوره", f"{total_ret:+.2f}%")
                        st.caption(f"{len(closes_arr)} روز معاملاتی")
                    with cb:
                        ann_vol_h = float(np.std(returns_h) * np.sqrt(252) * 100)
                        st.metric("نوسان سالانه‌شده", f"{ann_vol_h:.2f}%")
                        st.caption("انحراف معیار بازده × √252")
                    with cc:
                        max_h = float(np.max(closes_arr))
                        min_h = float(np.min(closes_arr))
                        mdd_h = (max_h - min_h) / max_h * 100
                        st.metric("دامنه نوسان کل", f"{mdd_h:.2f}%")
                        st.caption(f"بالا: {ime_fmt(max_h)} | پایین: {ime_fmt(min_h)}")
                    with cd:
                        pos_days = int(np.sum(returns_h > 0))
                        neg_days = int(np.sum(returns_h < 0))
                        st.metric("روزهای مثبت / منفی", f"{pos_days} / {neg_days}")
                        st.caption(f"نرخ موفقیت: {pos_days/(pos_days+neg_days)*100:.1f}%")

    else:
        st.info("ابتدا داده‌ها را از تابلوی زنده بارگزاری کنید.")

    # ══ بخش ۴ — مقایسه نمادها ══
    if rows_all and len(rows_all) >= 2:
        st.markdown('<div class="bp-section"><span class="bp-section-text">④ مقایسه نمادها — نمودار رادار</span></div>', unsafe_allow_html=True)
        st.caption("مقایسه شاخص‌های کلیدی نمادها: تغییر روز، دامنه نوسان، حجم، ارزش معاملات.")

        # نرمال‌سازی برای رادار
        def safe_float(v):
            try: return float(v or 0)
            except: return 0.0

        radar_metrics = {
            "تغییر %":       [abs(safe_float(r.get("plp",0))) for r in rows_all],
            "دامنه روز %":   [abs(safe_float(r.get("pmaxp",0)) - safe_float(r.get("pminp",0))) for r in rows_all],
            "حجم (norm)":    [safe_float(r.get("tvol",0)) for r in rows_all],
            "ارزش (norm)":   [safe_float(r.get("tval",0)) for r in rows_all],
            "تعداد معامله":  [safe_float(r.get("tno",0)) for r in rows_all],
        }
        # نرمال‌سازی 0–100
        def normalize(lst):
            mx = max(lst) if max(lst) > 0 else 1
            return [v/mx*100 for v in lst]

        norm_metrics = {k: normalize(v) for k, v in radar_metrics.items()}
        cats = list(norm_metrics.keys()) + [list(norm_metrics.keys())[0]]

        fig_radar_im = go.Figure()
        _colors_r = [blue_im, green_im, gold_im, red_im, "#9b72c8", "#48b8c0"]
        for i, row in enumerate(rows_all[:6]):
            vals = [norm_metrics[m][i] for m in norm_metrics]
            vals_closed = vals + [vals[0]]
            fig_radar_im.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats,
                fill="toself",
                fillcolor=f"rgba({','.join(str(int(int(_colors_r[i%6][j*2+1:j*2+3],16))) for j in range(3))},0.07)",
                line=dict(color=_colors_r[i%6], width=2),
                name=row.get("commodity","?"),
            ))

        _bg_r = "#161616" if is_dark else "#e8e8e4"
        _txt_r = "#c0c0c0" if is_dark else "#222220"
        _grid_r = "rgba(255,255,255,0.08)" if is_dark else "rgba(60,60,55,0.1)"
        fig_radar_im.update_layout(
            polar=dict(
                bgcolor=_bg_r,
                angularaxis=dict(tickfont=dict(size=9, color=_txt_r, family="JetBrains Mono"),
                                 linecolor=_grid_r, gridcolor=_grid_r),
                radialaxis=dict(visible=True, range=[0,100],
                               tickfont=dict(size=7, color=_txt_r), gridcolor=_grid_r,
                               linecolor=_grid_r, tickvals=[25,50,75,100],
                               ticktext=["25","50","75","100"]),
            ),
            paper_bgcolor=_bg_r, plot_bgcolor=_bg_r,
            font=dict(color=_txt_r, family="JetBrains Mono"),
            showlegend=True, height=380,
            margin=dict(l=60, r=60, t=40, b=40),
            title=dict(text="RADAR — مقایسه نمادها (نرمال‌شده)", font=dict(color=_txt_r, size=11), x=0.5),
            legend=dict(bgcolor="rgba(0,0,0,0.3)" if is_dark else "rgba(240,240,237,0.85)",
                        bordercolor="rgba(128,128,128,0.2)", borderwidth=1,
                        font=dict(color=_txt_r, size=9)),
        )
        st.plotly_chart(fig_radar_im, use_container_width=True)

    # ══ بخش ۵ — تحلیل‌گر هوشمند ══
    st.markdown('<div class="bp-section"><span class="bp-section-text">⑤ تحلیل‌گر هوشمند — سیگنال خودکار</span></div>', unsafe_allow_html=True)
    st.caption("بر اساس داده‌های API، سیگنال‌های خرید/فروش/خنثی صادر می‌کند.")

    if rows_all:
        signal_rows = []
        for row in rows_all:
            commodity = row.get("commodity","—")
            pl_v   = safe_float(row.get("pl", 0))
            plp_v  = safe_float(row.get("plp",0))
            tno_v  = safe_float(row.get("tno",0))
            tvol_v = safe_float(row.get("tvol",0))
            pmax_v = safe_float(row.get("pmax",0))
            pmin_v = safe_float(row.get("pmin",0))
            py_v   = safe_float(row.get("py", 0))
            pd1_v  = safe_float(row.get("pd1",0))
            po1_v  = safe_float(row.get("po1",0))

            # قوانین سیگنال ساده
            signals = []
            score = 0

            # ۱. تغییر روز
            if plp_v >= 4:
                signals.append("📈 صعود قوی (بالای ۴٪)")
                score += 2
            elif plp_v >= 1:
                signals.append("↗ صعود ملایم")
                score += 1
            elif plp_v <= -4:
                signals.append("📉 ریزش قوی (بیش از ۴٪)")
                score -= 2
            elif plp_v <= -1:
                signals.append("↘ ریزش ملایم")
                score -= 1

            # ۲. موقعیت نسبت به دامنه روز
            if pmax_v > 0 and pmin_v > 0:
                pos_in_range = (pl_v - pmin_v) / (pmax_v - pmin_v + 1e-9)
                if pos_in_range > 0.85:
                    signals.append("🔝 نزدیک به سقف روز")
                    score += 1
                elif pos_in_range < 0.15:
                    signals.append("⬇ نزدیک به کف روز")
                    score -= 1

            # ۳. فشار خرید vs فروش
            qd1_v = safe_float(row.get("qd1",0))
            qo1_v = safe_float(row.get("qo1",0))
            if qd1_v > 0 and qo1_v == 0:
                signals.append("💚 صف خرید — بدون عرضه")
                score += 2
            elif qo1_v > 0 and qd1_v == 0:
                signals.append("🔴 صف فروش — بدون تقاضا")
                score -= 2
            elif qd1_v > 0 and qo1_v > 0:
                ratio = qd1_v / (qo1_v + 1e-9)
                if ratio > 3:
                    signals.append(f"📗 فشار خرید قوی (D/S={ratio:.1f}x)")
                    score += 1
                elif ratio < 0.33:
                    signals.append(f"📛 فشار فروش قوی (D/S={ratio:.1f}x)")
                    score -= 1

            # حکم نهایی
            if score >= 3:
                verdict = "🟢 خرید"
                vcolor = green_im
            elif score >= 1:
                verdict = "💛 تمایل خرید"
                vcolor = gold_im
            elif score <= -3:
                verdict = "🔴 فروش"
                vcolor = red_im
            elif score <= -1:
                verdict = "🟠 تمایل فروش"
                vcolor = "#e8945a" if is_dark else "#7a3a00"
            else:
                verdict = "⚪ خنثی"
                vcolor = muted_im

            signal_rows.append({
                "کالا":    commodity,
                "آخرین":  ime_fmt(pl_v),
                "تغییر %": f"{plp_v:+.2f}%",
                "امتیاز":  score,
                "سیگنال":  verdict,
                "دلایل":   " | ".join(signals) if signals else "بدون سیگنال واضح",
            })

        df_signals = pd.DataFrame(signal_rows)

        # نمایش با رنگ‌بندی
        for _, srow in df_signals.iterrows():
            score_int = int(srow["امتیاز"])
            sc_color = green_im if score_int>=1 else (red_im if score_int<=-1 else muted_im)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
                padding:0.55rem 0.9rem;margin-bottom:0.35rem;
                background:{card_im};border:1px solid {border_im};
                border-left:4px solid {sc_color};border-radius:4px">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                    font-weight:700;color:var(--accent);min-width:80px">{srow['کالا']}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                    color:{muted_im};min-width:70px">{srow['آخرین']} ریال</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                    font-weight:700;color:{sc_color};min-width:60px">{srow['تغییر %']}</span>
                <span style="font-size:0.75rem;font-weight:700;min-width:100px">{srow['سیگنال']}</span>
                <span style="font-size:0.62rem;color:{muted_im};flex:1">{srow['دلایل']}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                    font-weight:700;color:{sc_color}">امتیاز: {score_int:+d}</span>
            </div>
            """, unsafe_allow_html=True)

        st.caption("⚠ این سیگنال‌ها صرفاً بر اساس داده لحظه‌ای API و قوانین ساده هستند. برای تصمیم‌گیری مالی، تحلیل جامع‌تری انجام دهید.")

    # ══ بخش ۶ — تنظیمات و endpoint تست ══
    with st.expander("⚙ تنظیمات API — تست endpoint های مختلف", expanded=False):
        st.markdown(f"""
        <div style="background:{card_im};border:1px solid {border_im};border-radius:4px;
            padding:0.8rem 1rem;margin-bottom:0.6rem;font-size:0.65rem;
            font-family:'JetBrains Mono',monospace;color:{muted_im};line-height:2">
            <strong style="color:var(--accent)">API Base:</strong> {IME_BASE_URL}<br>
            <strong style="color:var(--accent)">Key:</strong> {IME_API_KEY[:12]}…{IME_API_KEY[-4:]}<br>
            <strong style="color:var(--accent)">TTL Cache:</strong> 60 ثانیه (last-changes) | 300 ثانیه (history)
        </div>
        """, unsafe_allow_html=True)

        test_endpoint = st.text_input(
            "endpoint برای تست",
            value="/commodity-deposit-certificate/last-changes",
            key="ime_test_ep"
        )
        test_params = st.text_input(
            "پارامترها (JSON)",
            value="{}",
            placeholder='{"from": "1404-01-01", "to": "1404-07-02"}',
            key="ime_test_params"
        )

        if st.button("🧪 تست endpoint", use_container_width=True, key="ime_ep_test"):
            try:
                params_dict = _json_ime.loads(test_params) if test_params.strip() else {}
            except Exception:
                params_dict = {}
            with st.spinner("در حال فراخوانی API…"):
                result = ime_fetch(test_endpoint, params_dict if params_dict else None)
            if result:
                if "_error" in result:
                    st.error(f"خطا: {result['_error']}")
                else:
                    st.json(result)
            else:
                st.warning("پاسخی دریافت نشد.")

        # راهنمای endpoint ها
        st.markdown(f"""
        <div style="margin-top:0.5rem;font-size:0.62rem;color:{muted_im};line-height:2">
            <strong style="color:var(--accent)">Endpoints شناخته‌شده:</strong><br>
            <code>/commodity-deposit-certificate/last-changes</code> — همه نمادها<br>
            <code>/commodity-deposit-certificate/CD1GOB0001</code> — جزئیات یک نماد<br>
            <code>/commodity-deposit-certificate/CD1GOB0001/history?from=1404-01-01&to=1404-07-02</code> — تاریخچه
        </div>
        """, unsafe_allow_html=True)

    if auto_refresh:
        _time_ime.sleep(60)
        st.rerun()


saved_f = st.session_state.get("saved_risks", {})
risk_summary = ""
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
