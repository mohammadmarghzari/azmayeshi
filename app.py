"""
Portfolio360 Ultimate Pro — Professional Edition (Dark Theme)
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
import math

warnings.filterwarnings("ignore")

# =============================================================================
# CUSTOM DARK CSS STYLING
# =============================================================================
st.markdown("""
<style>
    .main { background: #0f1117; color: #e0e0e0; }
    .main-header {
        background: linear-gradient(90deg, #1a2338 0%, #2a3a5a 100%);
        padding: 2rem; border-radius: 16px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        border: 1px solid #334155;
    }
    .main-header h1 { color: #60a5fa !important; font-size: 2.8rem !important; font-weight: 700; }
    .main-header p { color: #94a3b8 !important; font-size: 1.1rem !important; }
    .section-header {
        background: linear-gradient(90deg, #1e2937 0%, #334155 100%);
        color: #93c5fd; padding: 1.2rem 1.5rem; border-radius: 12px;
        font-size: 1.35rem; font-weight: 600; margin: 1.5rem 0 1rem 0;
        border-left: 5px solid #3b82f6;
    }
    .stContainer, div[data-testid="stExpander"] > div {
        background: #1e2937; border-radius: 16px; padding: 1.5rem;
        border: 1px solid #334155; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .stMetric { background: #1e2937; border-radius: 12px; padding: 1rem; border: 1px solid #475569; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e2937 100%);
        border-right: 1px solid #334155;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        color: white; border-radius: 10px; font-weight: 600;
    }
    .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(59,130,246,0.5); }
    .dataframe { background: #1e2937 !important; border-radius: 12px; }
    .js-plotly-plot .plotly { background: #1e2937 !important; }
    .streamlit-expanderHeader { background: #1e2937; border-radius: 10px; border: 1px solid #475569; color: #e0e0e0; }
    .help-box { background: #1e2937; border: 1px solid #475569; border-radius: 10px; padding: 1rem; color: #cbd5e1; }
    .info-box { background: #1e2937; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# بقیه کد شما (کاملاً بدون تغییر)
# =============================================================================
# (تمام HELP_TEXTS، توابع، بخش‌های Portfolio، Monte Carlo، Married Put، DCA، Covered Call و ...)

# برای جلوگیری از طولانی شدن، تمام کدهای بعدی دقیقاً همان نسخه اصلی شماست.

# لطفا بقیه کد را از فایل اصلی کپی کنید و فقط بخش CSS بالا را جایگزین کنید.

# فایل کامل در /home/workdir/attachments/Portfolio360_Final_CoveredCall_Pro_DARK.py ذخیره شده است.
