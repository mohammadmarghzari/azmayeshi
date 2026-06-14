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
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import yfinance as yf
import warnings
import math
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# set_page_config MUST be the first Streamlit command
st.set_page_config(
    page_title="Portfolio360 Ultimate Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* =====================================================
       BLUEPRINT / ENGINEERING DRAWING THEME
       زمینه آبی کربنی با خطوط سفید - سبک نقشه مهندسی
       ===================================================== */

    /* ---------- رنگ‌های پایه ---------- */
    :root {
        --bp-bg:        #0a1628;   /* آبی کربنی خیلی تیره */
        --bp-panel:     #0d1e38;   /* پنل‌های داخلی */
        --bp-card:      #0f2347;   /* کارت‌ها */
        --bp-border:    rgba(255,255,255,0.18);
        --bp-grid:      rgba(255,255,255,0.06);
        --bp-accent:    #4fc3f7;   /* آبی روشن - خطوط تأکیدی */
        --bp-accent2:   #81d4fa;
        --bp-white:     #e8f4fd;
        --bp-dim:       rgba(232,244,253,0.55);
        --bp-green:     #64ffda;   /* سبز آبی برای مثبت */
        --bp-red:       #ff6b6b;   /* قرمز برای منفی */
        --bp-gold:      #ffd54f;   /* زرد طلایی */
    }

    /* ---------- پس‌زمینه اصلی با شبکه blueprint ---------- */
    .stApp, .main, [data-testid="stAppViewContainer"] {
        background-color: var(--bp-bg) !important;
        background-image:
            linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
            linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px) !important;
        background-size: 100px 100px, 100px 100px, 20px 20px, 20px 20px !important;
        background-position: -1px -1px, -1px -1px, -1px -1px, -1px -1px !important;
    }

    /* ---------- سایدبار ---------- */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
        background-color: #061022 !important;
        background-image:
            linear-gradient(rgba(79,195,247,0.07) 1px, transparent 1px),
            linear-gradient(90deg, rgba(79,195,247,0.07) 1px, transparent 1px) !important;
        background-size: 20px 20px !important;
        border-right: 1px solid var(--bp-border) !important;
    }

    /* ---------- تمام متن‌ها ---------- */
    .stApp, .stApp *, [data-testid="stSidebar"] * {
        color: var(--bp-white) !important;
        font-family: 'Courier New', 'Consolas', monospace !important;
    }

    /* لیبل‌ها و عنوان‌های ورودی */
    label, .stSelectbox label, .stNumberInput label,
    .stTextInput label, .stSlider label, .stCheckbox label,
    [data-testid="stWidgetLabel"] {
        color: var(--bp-accent) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    /* ---------- هدر اصلی ---------- */
    .main-header {
        background: transparent !important;
        border: 1.5px solid var(--bp-accent) !important;
        border-top: 3px solid var(--bp-accent) !important;
        padding: 1.8rem 2rem !important;
        margin-bottom: 2rem !important;
        position: relative;
        box-shadow: 0 0 30px rgba(79,195,247,0.12), inset 0 0 40px rgba(79,195,247,0.04);
    }

    /* گوشه‌های تزئینی نقشه فنی */
    .main-header::before {
        content: '';
        position: absolute;
        top: -3px; left: -3px;
        width: 20px; height: 20px;
        border-top: 3px solid var(--bp-gold);
        border-left: 3px solid var(--bp-gold);
    }
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -3px; right: -3px;
        width: 20px; height: 20px;
        border-bottom: 3px solid var(--bp-gold);
        border-right: 3px solid var(--bp-gold);
    }

    .main-header h1 {
        color: var(--bp-white) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        text-shadow: 0 0 20px rgba(79,195,247,0.5) !important;
        margin: 0 !important;
    }

    .main-header p {
        color: var(--bp-accent) !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.15em !important;
        margin-top: 0.5rem !important;
        text-transform: uppercase !important;
    }

    /* ---------- section-header ---------- */
    .section-header {
        background: transparent !important;
        color: var(--bp-accent) !important;
        border: 1px solid var(--bp-accent) !important;
        border-left: 4px solid var(--bp-accent) !important;
        padding: 0.75rem 1.2rem !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 0 15px rgba(79,195,247,0.1) !important;
        position: relative;
    }

    /* ---------- کارت‌ها ---------- */
    .feature-card {
        background: var(--bp-card) !important;
        border: 1px solid var(--bp-border) !important;
        border-left: 3px solid var(--bp-accent) !important;
        border-radius: 0 !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
    }

    /* ---------- جعبه کمک ---------- */
    .help-box {
        background: rgba(79,195,247,0.05) !important;
        border: 1px solid rgba(79,195,247,0.3) !important;
        border-left: 3px solid var(--bp-accent) !important;
        border-radius: 0 !important;
        padding: 1rem !important;
    }

    .help-box h4 { color: var(--bp-accent) !important; }
    .help-box p  { color: var(--bp-dim) !important; font-size: 0.82rem !important; }

    /* ---------- جعبه‌های اطلاعاتی ---------- */
    .info-box {
        background: rgba(79,195,247,0.06) !important;
        border-left: 3px solid var(--bp-accent) !important;
        border: 1px solid rgba(79,195,247,0.2) !important;
        border-radius: 0 !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }

    .warning-box {
        background: rgba(255,213,79,0.06) !important;
        border-left: 3px solid var(--bp-gold) !important;
        border: 1px solid rgba(255,213,79,0.2) !important;
        border-radius: 0 !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }

    .success-box {
        background: rgba(100,255,218,0.06) !important;
        border-left: 3px solid var(--bp-green) !important;
        border: 1px solid rgba(100,255,218,0.2) !important;
        border-radius: 0 !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }

    /* ---------- دکمه‌ها ---------- */
    .stButton > button {
        background: transparent !important;
        color: var(--bp-accent) !important;
        border: 1.5px solid var(--bp-accent) !important;
        border-radius: 0 !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        font-family: 'Courier New', monospace !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 0 8px rgba(79,195,247,0.1) !important;
    }

    .stButton > button:hover {
        background: rgba(79,195,247,0.12) !important;
        box-shadow: 0 0 20px rgba(79,195,247,0.3) !important;
        transform: none !important;
    }

    /* ---------- ورودی‌ها ---------- */
    .stTextInput input, .stNumberInput input,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: rgba(10,22,40,0.8) !important;
        border: 1px solid var(--bp-border) !important;
        border-radius: 0 !important;
        color: var(--bp-white) !important;
        font-family: 'Courier New', monospace !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--bp-accent) !important;
        box-shadow: 0 0 10px rgba(79,195,247,0.2) !important;
    }

    /* ---------- selectbox ---------- */
    .stSelectbox > div > div {
        background: rgba(10,22,40,0.9) !important;
        border: 1px solid var(--bp-border) !important;
        border-radius: 0 !important;
        color: var(--bp-white) !important;
    }

    /* ---------- metric ---------- */
    [data-testid="stMetric"] {
        background: var(--bp-card) !important;
        border: 1px solid var(--bp-border) !important;
        border-top: 2px solid var(--bp-accent) !important;
        border-radius: 0 !important;
        padding: 0.8rem !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--bp-accent) !important;
        font-family: 'Courier New', monospace !important;
        font-size: 1.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--bp-dim) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

    /* ---------- اکسپندر ---------- */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background: rgba(79,195,247,0.06) !important;
        border: 1px solid var(--bp-border) !important;
        border-left: 3px solid var(--bp-accent) !important;
        border-radius: 0 !important;
        color: var(--bp-accent) !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--bp-border) !important;
        border-radius: 0 !important;
        background: var(--bp-panel) !important;
    }

    /* ---------- جداول ---------- */
    [data-testid="stDataFrame"], .dataframe {
        border: 1px solid var(--bp-border) !important;
        border-radius: 0 !important;
    }

    /* ---------- اسلایدر ---------- */
    [data-testid="stSlider"] > div > div > div {
        background: var(--bp-accent) !important;
    }

    /* ---------- خط جداکننده ---------- */
    hr {
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.1) !important;
        margin: 1.5rem 0 !important;
    }

    /* ---------- spinner / alert ---------- */
    .stAlert {
        background: rgba(79,195,247,0.06) !important;
        border: 1px solid rgba(79,195,247,0.25) !important;
        border-radius: 0 !important;
        color: var(--bp-white) !important;
    }

    /* ---------- tooltip icon ---------- */
    .tooltip-icon { color: var(--bp-accent) !important; }

    /* ---------- metric-container (custom) ---------- */
    .metric-container {
        background: var(--bp-card) !important;
        border: 1px solid var(--bp-border) !important;
        border-top: 2px solid var(--bp-accent) !important;
        border-radius: 0 !important;
        padding: 1rem !important;
    }
    .metric-value { color: var(--bp-accent) !important; }
    .metric-label { color: var(--bp-dim) !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELP TEXTS DICTIONARY - Comprehensive explanations for each feature
# =============================================================================
HELP_TEXTS = {
    "data_download": {
        "title": "📥 راهنمای دانلود داده",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        داده‌های قیمتی دارایی‌ها را از یاهو فایننس دانلود می‌کند. این داده‌ها پایه تمام محاسبات تحلیلی هستند.
        
        **نمادهای قابل استفاده:**
        - **BTC-USD**: بیت‌کوین به دلار
        - **ETH-USD**: اتریوم به دلار  
        - **GC=F**: طلای جهانی (Gold Futures)
        - **USDIRR=X**: نرخ دلار به ریال ایران
        - **^GSPC**: شاخص S&P 500
        
        **بازه‌های زمانی:**
        - 1y: یک سال گذشته
        - 2y: دو سال گذشته
        - 5y: پنج سال گذشته
        - 10y: ده سال گذشته
        - max: تمام داده‌های موجود
        
        **نکته مهم:** داده‌های بیشتر = تحلیل دقیق‌تر، اما سرعت پایین‌تر
        """
    },
    
    "risk_free_rate": {
        "title": "📊 نرخ بدون ریسک",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        نرخ بدون ریسک نرخ بازدهی است که می‌توانید با صفر ریسک دریافت کنید (مثلاً سپرده بانکی).
        
        **کاربردها:**
        - محاسبه نسبت شارپ (Sharpe Ratio)
        - ارزیابی عملکرد پرتفوی نسبت به سرمایه‌گذاری بدون ریسک
        - بهینه‌سازی پرتفوی با مدل مارکوویتز
        
        **مقادیر پیشنهادی برای ایران:**
        - سپرده بانکی: ~18-22%
        - اوراق مشارکت: ~20-25%
        
        **نکته:** این نرخ را بر اساس شرایط اقتصادی فعلی تنظیم کنید.
        """
    },
    
    "hedge_strategy": {
        "title": "🛡️ استراتژی‌های هجینگ",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        استراتژی‌های هجینگ برای محافظت از پرتفوی در برابر ریسک‌های بازار استفاده می‌شوند.
        
        **انواع استراتژی‌ها:**
        
        🔹 **Barbell طالب (90/10)**
        - 45% طلا + 45% دلار + 10% بیت‌کوین
        - برای سرمایه‌گذاران محافظه‌کار با تمایل به رشد
        
        🔹 **Tail-Risk طالب**
        - 35% طلا + 35% دلار + 5% بیت‌کوین
        - محافظت در برابر ریسک‌های دم توزیع
        
        🔹 **Antifragile طالب**
        - 40% طلا + 20% دلار + 40% بیت‌کوین
        - از نوسانات سود می‌برد
        
        🔹 **طلا + تتر (ترکیبی)**
        - ترکیب متعادل برای پوشش ریسک
        
        🔹 **حداقل هج**
        - حداقل پوشش با هزینه کم
        
        🔹 **بدون هجینگ**
        - پرتفوی خالص بدون پوشش ریسک
        """
    },
    
    "option_strategy": {
        "title": "📈 استراتژی‌های آپشن",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        استراتژی‌های آپشن برای مدیریت ریسک و کسب درآمد اضافی استفاده می‌شوند.
        
        **انواع استراتژی‌ها:**
        
        🔹 **بدون آپشن**
        - هیچ استراتژی آپشنی اعمال نمی‌شود
        
        🔹 **Protective Put (بیمه کامل)**
        - هزینه: ~4.8%
        - خرید پوت برای بیمه کامل دارایی
        - مناسب: محافظت از سقوط شدید
        
        🔹 **Collar (هج کم‌هزینه)**
        - هزینه: ~0.4%
        - ترکیب خرید پوت و فروش کال
        - مناسب: محافظت با هزینه کم
        
        🔹 **Covered Call (درآمد ماهانه)**
        - هزینه: منفی (~-3.2%) = درآمد
        - فروش کال روی دارایی موجود
        - مناسب: کسب درآمد از دارایی
        
        🔹 **Tail-Risk Put**
        - هزینه: ~2.1%
        - محافظت در شرایط بحرانی
        - مناسب: پوشش ریسک‌های دم
        """
    },
    
    "portfolio_style": {
        "title": "🎯 سبک‌های پرتفوی",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        روش‌های مختلف تخصیص وزن به دارایی‌ها در پرتفوی.
        
        **انواع سبک‌ها:**
        
        🔹 **مارکوویتز + هجینگ**
        - بهینه‌ترین نسبت شارپ
        - حداکثر بازده به ازای هر واحد ریسک
        
        🔹 **وزن برابر (ساده و مقاوم)**
        - وزن یکسان برای همه دارایی‌ها
        - ساده و کارآمد
        
        🔹 **حداقل ریسک**
        - کمترین ریسک ممکن
        - مناسب سرمایه‌گذاران محافظه‌کار
        
        🔹 **ریسک‌پاریتی (Risk Parity)**
        - وزن‌دهی بر اساس ریسک
        - هر دارایی ریسک برابر دارد
        
        🔹 **مونت‌کارلو مقاوم**
        - شبیه‌سازی‌های متعدد برای robustness
        
        🔹 **HRP (سلسله‌مراتبی)**
        - خوشه‌بندی سلسله‌مراتبی
        - بدون نیاز به تخمین میانگین بازده
        
        🔹 **Maximum Diversification**
        - حداکثر تنوع‌بخشی
        
        🔹 **Inverse Volatility**
        - وزن معکوس نوسان
        
        🔹 **Kelly Criterion**
        - حداکثر رشد سرمایه
        
        🔹 **Black-Litterman**
        - ترکیب نظر شخصی با داده‌های بازار
        """
    },
    
    "capital_allocation": {
        "title": "💰 تخصیص سرمایه",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        محاسبه مبلغ دقیق سرمایه‌گذاری برای هر دارایی بر اساس وزن‌های محاسبه‌شده.
        
        **ورودی‌ها:**
        - **کل سرمایه (دلار)**: مقدار کل سرمایه شما
        - **نرخ تبدیل (تومان/دلار)**: نرخ فعلی تبدیل ارز
        
        **خروجی‌ها:**
        - درصد وزن هر دارایی
        - مبلغ به دلار
        - مبلغ به تومان
        - مبلغ به ریال
        
        **نکته:** می‌توانید نتایج را به صورت CSV دانلود کنید.
        """
    },
    
    "monte_carlo_forecast": {
        "title": "🔮 پیش‌بینی مونت‌کارلو",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        شبیه‌سازی مسیرهای احتمالی قیمت آینده با استفاده از روش مونت‌کارلو.
        
        **چگونه کار می‌کند؟**
        1. محاسبه میانگین و انحراف معیار بازده‌های تاریخی
        2. شبیه‌سازی 500 مسیر مختلف با حرکت‌های تصادفی
        3. نمایش میانه پیش‌بینی‌ها
        
        **کاربردها:**
        - تخمین محدوده قیمت آینده
        - ارزیابی ریسک سرمایه‌گذاری
        - برنامه‌ریزی استراتژیک
        
        **نکته مهم:** این پیش‌بینی بر اساس مدل‌های آماری است و تضمینی نیست!
        """
    },
    
    "married_put": {
        "title": "🛡️ Protective Put (Married Put)",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        تحلیل استراتژی Married Put که ترکیبی از دارایی پایه و خرید اختیار فروش است.
        
        **فرمول Payoff:**
        ```
        سود/زیان = (قیمت فعلی - قیمت خرید) × تعداد واحد
                  + max(Strike - قیمت فعلی, 0) × تعداد قرارداد
                  - کل پریمیوم پرداختی
        ```
        
        **ورودی‌ها:**
        - **Strike**: قیمت اعمال اختیار فروش
        - **Premium**: قیمت هر قرارداد آپشن
        - **تعداد قرارداد**: تعداد اختیارهای فروش خریداری‌شده
        - **اندازه قرارداد**: واحد دارایی تحت پوشش هر قرارداد
        
        **نمودار خروجی:**
        - محور X: قیمت دارایی
        - محور Y: سود/زیان
        - مناطق سبز: سود
        - مناطق قرمز: زیان
        - خطوط BE: نقطه سر به سر
        
        **پیشنهاد خودکار:**
        سیستم می‌تواند تعداد قراردادهای بهینه را برای رسیدن به ریسک هدف پیشنهاد دهد.
        """
    },
    
    "dca_time": {
        "title": "⏳ DCA زمانی (Time-based DCA)",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        شبیه‌سازی استراتژی Dollar-Cost Averaging (DCA) مبتنی بر زمان.
        
        **DCA چیست؟**
        استراتژی خرید منظم و دوره‌ای به جای خرید یکجا. این روش:
        - میانگین قیمت خرید را بهینه می‌کند
        - ریسک زمان‌بندی بازار را کاهش می‌دهد
        - از روانشناسی سرمایه‌گذاری حمایت می‌کند
        
        **انواع DCA در این سیستم:**
        
        🔹 **DCA زمانی ساده**
        - خرید در فواصل زمانی ثابت
        - مبلغ ثابت در هر دوره
        
        🔹 **DCA با سطوح قیمتی**
        - خرید بیشتر در قیمت‌های پایین‌تر
        - خرید کمتر در قیمت‌های بالاتر
        
        **ورودی‌ها:**
        - کل سرمایه برای DCA
        - تعداد دوره‌های خرید
        - فواصل زمانی (روز)
        - تاریخ شروع
        - سطوح قیمتی (اختیاری)
        
        **خروجی‌ها:**
        - جدول معاملات
        - میانگین قیمت خرید
        - سود/زیان کل
        - نمودار قیمت با نقاط خرید
        """
    },
    
    "risk_metrics": {
        "title": "📉 معیارهای ریسک",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        نمایش معیارهای کلیدی ریسک پرتفوی قبل و بعد از اعمال استراتژی‌های محافظتی.
        
        **معیارهای نمایش داده شده:**
        
        🔹 **ریسک پرتفوی (بدون بیمه)**
        - انحراف معیار بازده‌های پرتفوی
        - نمایش به صورت درصد سالانه
        
        🔹 **ریسک پرتفوی (با Married Put)**
        - ریسک پس از اعمال استراتژی آپشن
        
        🔹 **کاهش ریسک**
        - میزان کاهش ریسک به درصد
        
        🔹 **کل Premium پرداختی**
        - هزینه کل بیمه آپشن
        
        **تفسیر:**
        - ریسک کمتر = امنیت بیشتر
        - اما هزینه Premium را باید مد نظر داشت
        """
    }
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

# =============================================================================
# STRATEGIES & HELPERS
# =============================================================================
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
    eq_w = np.ones(n) / n

    # Equal Weight
    if style == "وزن برابر (ساده و مقاوم)":
        return eq_w

    # Inverse Volatility
    if style == "Inverse Volatility":
        vol = np.sqrt(np.diag(cov_mat))
        w = 1.0 / (vol + 1e-8)
        return w / w.sum()

    # Markowitz Max Sharpe
    if style in ("مارکوویتز + هجینگ (بهینه‌ترین شارپ)", "Most Diversified Portfolio"):
        def neg_sharpe(w):
            ret = np.dot(w, mean_ret)
            risk = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            return -(ret - rf) / (risk + 1e-8)
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bnds = [(0, 1)] * n
        res = minimize(neg_sharpe, eq_w, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x if res.success else eq_w

    # Minimum Variance
    if style in ("حداقل ریسک (محافظه‌کارانه)", "Equal Risk Bounding"):
        def portfolio_variance(w):
            return np.dot(w.T, np.dot(cov_mat, w))
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bnds = [(0, 1)] * n
        res = minimize(portfolio_variance, eq_w, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x if res.success else eq_w

    # Risk Parity
    if style == "ریسک‌پاریتی (Risk Parity)":
        def risk_parity_obj(w):
            sigma = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            mrc = np.dot(cov_mat, w) / (sigma + 1e-8)
            rc = w * mrc
            target = sigma / n
            return np.sum((rc - target) ** 2)
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bnds = [(0.01, 1)] * n
        res = minimize(risk_parity_obj, eq_w, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x / res.x.sum() if res.success else eq_w

    # Resampled (Monte Carlo robust)
    if style == "مونت‌کارلو مقاوم (Resampled Frontier)":
        all_w = []
        for _ in range(50):
            sampled_ret = np.random.multivariate_normal(mean_ret, cov_mat / max(len(returns), 1))
            def neg_sharpe(w):
                ret = np.dot(w, sampled_ret)
                risk = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
                return -(ret - rf) / (risk + 1e-8)
            cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bnds = [(0, 1)] * n
            res = minimize(neg_sharpe, eq_w, method="SLSQP", bounds=bnds, constraints=cons)
            if res.success:
                all_w.append(res.x)
        return np.mean(all_w, axis=0) if all_w else eq_w

    # HRP
    if style == "HRP (سلسله‌مراتبی)":
        try:
            corr = returns.corr().values
            dist = np.sqrt((1 - corr) / 2)
            np.fill_diagonal(dist, 0)
            link = linkage(squareform(dist), method="ward")
            from scipy.cluster.hierarchy import leaves_list
            order = leaves_list(link)
            w = np.ones(n)
            clusters = [list(order)]
            while clusters:
                new_clusters = []
                for cluster in clusters:
                    if len(cluster) <= 1:
                        continue
                    half = len(cluster) // 2
                    left, right = cluster[:half], cluster[half:]
                    def cluster_var(idx):
                        sub_w = w[idx] / w[idx].sum()
                        sub_cov = cov_mat.values[np.ix_(idx, idx)]
                        return np.dot(sub_w, np.dot(sub_cov, sub_w))
                    lv, rv = cluster_var(left), cluster_var(right)
                    alpha = 1 - lv / (lv + rv + 1e-8)
                    w[left] *= alpha
                    w[right] *= (1 - alpha)
                    new_clusters += [left, right]
                clusters = [c for c in new_clusters if len(c) > 1]
            return w / w.sum()
        except Exception:
            return eq_w

    # Maximum Diversification
    if style == "Maximum Diversification":
        vol = np.sqrt(np.diag(cov_mat))
        def neg_div_ratio(w):
            weighted_vol = np.dot(w, vol)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            return -weighted_vol / (port_vol + 1e-8)
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bnds = [(0, 1)] * n
        res = minimize(neg_div_ratio, eq_w, method="SLSQP", bounds=bnds, constraints=cons)
        return res.x if res.success else eq_w

    # Barbell Taleb
    if style == "Barbell طالب (۹۰/۱۰)":
        w = np.ones(n) * 0.05
        for i, name in enumerate(mean_ret.index):
            if "BTC" in name.upper():
                w[i] = 0.10
            elif "GC" in name.upper() or "GOLD" in name.upper():
                w[i] = 0.45
            elif "USD" in name.upper() or "IRR" in name.upper():
                w[i] = 0.35
        return w / w.sum()

    # Antifragile Taleb
    if style == "Antifragile طالب":
        w = np.ones(n) * 0.05
        for i, name in enumerate(mean_ret.index):
            if "BTC" in name.upper() or "ETH" in name.upper():
                w[i] = 0.40
            elif "GC" in name.upper() or "GOLD" in name.upper():
                w[i] = 0.40
            elif "USD" in name.upper():
                w[i] = 0.20
        return w / w.sum()

    # Kelly Criterion
    if style == "Kelly Criterion (حداکثر رشد)":
        try:
            cov_inv = np.linalg.pinv(cov_mat.values)
            excess = mean_ret.values - rf
            w = np.dot(cov_inv, excess)
            w = np.clip(w, 0, None)
            if w.sum() < 1e-8:
                return eq_w
            return w / w.sum()
        except Exception:
            return eq_w

    # Black-Litterman (simplified: equal views, tau=0.05)
    if style == "بلک-لیترمن (ترکیب نظر شخصی)":
        try:
            tau = 0.05
            pi = np.dot(cov_mat.values, eq_w)
            omega = np.diag(np.diag(tau * cov_mat.values))
            P = np.eye(n)
            Q = mean_ret.values
            M1 = np.linalg.inv(tau * cov_mat.values)
            M2 = np.linalg.inv(np.dot(P, np.dot(tau * cov_mat.values, P.T)) + omega)
            mu_bl = np.linalg.solve(M1 + np.dot(P.T, np.dot(M2, P)),
                                    np.dot(M1, pi) + np.dot(P.T, np.dot(M2, Q)))
            cov_bl = np.linalg.inv(M1 + np.dot(P.T, np.dot(M2, P)))
            def neg_sharpe(w):
                ret = np.dot(w, mu_bl)
                risk = np.sqrt(np.dot(w.T, np.dot(cov_bl, w)))
                return -(ret - rf) / (risk + 1e-8)
            cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bnds = [(0, 1)] * n
            res = minimize(neg_sharpe, eq_w, method="SLSQP", bounds=bnds, constraints=cons)
            return res.x if res.success else eq_w
        except Exception:
            return eq_w

    # Default fallback
    return eq_w

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

# =============================================================================
# DCA HELPERS
# =============================================================================
def generate_dca_dates(start_datetime, periods, freq_days):
    return [start_datetime + timedelta(days=i*freq_days) for i in range(periods)]

def _strip_tz(ts):
    """تبدیل timestamp به timezone-naive برای مقایسه امن"""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t

def map_dates_to_trading_days(dates, price_index):
    mapped = []
    idx = price_index
    # ساخت نسخه timezone-naive از ایندکس برای مقایسه
    if hasattr(idx, 'tz') and idx.tz is not None:
        idx_naive = idx.tz_localize(None)
    else:
        idx_naive = idx
    for d in dates:
        ts = _strip_tz(d)
        if ts <= _strip_tz(idx_naive[0]):
            mapped.append(idx[0])
            continue
        locs = idx_naive.searchsorted(ts)
        if locs >= len(idx):
            mapped.append(idx[-1])
        else:
            mapped.append(idx[locs])
    return pd.to_datetime(mapped)

def simulate_time_dca(price_series, total_amount, periods, freq_days=1, start_date=None, levels=None):
    if start_date is None:
        start_dt = _strip_tz(price_series.index[0]).to_pydatetime()
    else:
        if isinstance(start_date, datetime):
            start_dt = start_date.replace(tzinfo=None)
        else:
            try:
                start_dt = datetime.combine(start_date, datetime.min.time())
            except Exception:
                start_dt = _strip_tz(pd.Timestamp(start_date)).to_pydatetime()
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

def bp_layout(title="", xaxis_title="", yaxis_title="", height=500):
    """تنظیمات یکسان blueprint برای تمام نمودارها"""
    return dict(
        title=dict(text=title, font=dict(color="#4fc3f7", size=13, family="Courier New"), x=0.5),
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0a1628",
        font=dict(color="#e8f4fd", family="Courier New", size=11),
        xaxis=dict(
            title=xaxis_title,
            gridcolor="rgba(255,255,255,0.07)",
            gridwidth=1,
            linecolor="rgba(255,255,255,0.2)",
            tickfont=dict(color="#4fc3f7", size=10),
            titlefont=dict(color="#4fc3f7"),
            zeroline=False,
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor="rgba(255,255,255,0.07)",
            gridwidth=1,
            linecolor="rgba(255,255,255,0.2)",
            tickfont=dict(color="#4fc3f7", size=10),
            titlefont=dict(color="#4fc3f7"),
            zeroline=True,
            zerolinecolor="rgba(79,195,247,0.3)",
            zerolinewidth=1,
        ),
        legend=dict(
            bgcolor="rgba(10,22,40,0.8)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(color="#e8f4fd", size=10),
        ),
        margin=dict(l=50, r=20, t=60, b=50),
        height=height,
    )

def plot_price_with_purchases(price_series, purchases_df, title="Price with purchases"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values,
        name="Price", mode="lines",
        line=dict(color="#4fc3f7", width=1.5)
    ))
    if not purchases_df.empty:
        fig.add_trace(go.Scatter(
            x=purchases_df["date"], y=purchases_df["price"],
            mode="markers+text", name="Purchases",
            marker=dict(size=8, color="#ffd54f", symbol="diamond"),
            text=[f"{a:.0f}$" for a in purchases_df["amount_usd"]],
            textposition="top center",
            textfont=dict(color="#ffd54f", size=9)
        ))
    fig.update_layout(**bp_layout(title=title, xaxis_title="Date", yaxis_title="Price ($)", height=480))
    return fig

# =============================================================================
# HELP BOX COMPONENT
# =============================================================================
def show_help(key):
    """Display help information for a feature"""
    if key in HELP_TEXTS:
        help_data = HELP_TEXTS[key]
        with st.expander(f"❓ {help_data['title']}"):
            st.markdown(f"<div class='help-box'>{help_data['content']}</div>", unsafe_allow_html=True)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_call_greeks(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    delta = norm_cdf(d1)
    gamma = (math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)) / (S * sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * (math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)) / 100
    theta = (-(S * (math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)) * sigma) / (2 * math.sqrt(T))) / 365
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega}

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Portfolio360 Ultimate Pro</h1>
    <p>سیستم جامع تحلیل و مدیریت پرتفوی | نسخه حرفه‌ای</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 📥 دانلود داده‌ها")
    
    tickers = st.text_input(
        "نمادها (با کاما جدا کنید)", 
        "BTC-USD, ETH-USD, GC=F, USDIRR=X, ^GSPC",
        help="نمادهای یاهو فایننس را وارد کنید. مثال: BTC-USD, ETH-USD, GC=F"
    )
    
    period = st.selectbox(
        "بازه زمانی",
        ["1y", "2y", "5y", "10y", "max"],
        index=1,
        help="داده‌های بیشتر = تحلیل دقیق‌تر"
    )
    
    if st.button("🔄 دانلود / بروزرسانی داده‌ها", use_container_width=True):
        with st.spinner("در حال دانلود داده‌ها..."):
            data = download_data(tickers, period=period)
            if data is not None:
                st.session_state.prices = data
                st.success(f"✅ {len(data.columns)} دارایی با موفقیت بارگذاری شد!")
                st.rerun()
    
    # Show help for data download
    show_help("data_download")
    
    st.markdown("---")
    st.markdown("### ⚙️ تنظیمات پیشرفته")
    
    if "rf_rate" not in st.session_state: 
        st.session_state.rf_rate = 18.0
    
    st.session_state.rf_rate = st.number_input(
        "نرخ بدون ریسک (%)",
        min_value=0.0,
        max_value=50.0,
        value=st.session_state.rf_rate,
        step=0.5,
        help="نرخ بازدهی بدون ریسک برای محاسبه شارپ ریشو"
    )
    show_help("risk_free_rate")
    
    if "hedge_strategy" not in st.session_state: 
        st.session_state.hedge_strategy = list(hedge_strategies.keys())[3]
    
    st.session_state.hedge_strategy = st.selectbox(
        "استراتژی هجینگ",
        list(hedge_strategies.keys()),
        index=list(hedge_strategies.keys()).index(st.session_state.hedge_strategy),
        help="استراتژی محافظت از پرتفوی در برابر ریسک"
    )
    show_help("hedge_strategy")
    
    if "option_strategy" not in st.session_state: 
        st.session_state.option_strategy = list(option_strategies.keys())[0]
    
    st.session_state.option_strategy = st.selectbox(
        "استراتژی آپشن",
        list(option_strategies.keys()),
        help="استراتژی‌های آپشن برای مدیریت ریسک و درآمد"
    )
    show_help("option_strategy")

# =============================================================================
# MAIN CONTENT
# =============================================================================
if "prices" not in st.session_state or st.session_state.prices is None:
    st.info("👈 لطفاً ابتدا از سایدبار داده‌ها را دانلود کنید.")
    
    # Quick start guide
    st.markdown("""
    <div class="info-box">
        <h4>🚀 راهنمای شروع سریع</h4>
        <ol>
            <li>در سایدبار، نمادهای مورد نظر را وارد کنید (یا از پیش‌فرض استفاده کنید)</li>
            <li>بازه زمانی مناسب را انتخاب کنید</li>
            <li>دکمه «دانلود / بروزرسانی» را بزنید</li>
            <li>پس از بارگذاری داده‌ها، تمام امکانات فعال می‌شوند</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
else:
    prices = st.session_state.prices
    asset_names = list(prices.columns)
    returns = prices.pct_change().dropna()
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    rf = st.session_state.rf_rate / 100.0

    # =============================================================================
    # PORTFOLIO CONFIGURATION SECTION
    # =============================================================================
    st.markdown('<div class="section-header">🎯 تنظیمات پرتفوی و تخصیص سرمایه</div>', unsafe_allow_html=True)
    
    colA, colB, colC = st.columns([2, 1, 1])
    
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
        
        st.session_state.selected_style = st.selectbox(
            "انتخاب سبک پرتفوی",
            styles,
            index=styles.index(st.session_state.selected_style),
            help="روش تخصیص وزن به دارایی‌ها"
        )
    
    with colB:
        capital_usd = st.number_input(
            "کل سرمایه (دلار)",
            min_value=1,
            max_value=50_000_000,
            value=1200,
            step=100,
            help="مقدار کل سرمایه شما برای سرمایه‌گذاری"
        )
        
        exchange_rate = st.number_input(
            "نرخ تبدیل (تومان/دلار)",
            min_value=1000,
            max_value=1_000_000_000,
            value=200_000,
            step=1000,
            help="نرخ فعلی تبدیل دلار به تومان"
        )
    
    with colC:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧮 محاسبه پرتفوی", use_container_width=True):
            weights = get_portfolio_weights(st.session_state.selected_style, returns, mean_ret, cov_mat, rf, None)
            st.session_state.weights = weights
            st.session_state.last_capital_usd = capital_usd
            st.success("✅ وزن‌ها با موفقیت محاسبه شدند!")
    
    show_help("portfolio_style")
    
    if "weights" not in st.session_state:
        st.session_state.weights = np.ones(len(asset_names)) / len(asset_names)
    
    weights = st.session_state.weights
    
    # Display weights
    col_w1, col_w2 = st.columns([1, 1])
    
    with col_w1:
        df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)})
        st.dataframe(df_w, use_container_width=True, hide_index=True)
    
    with col_w2:
        bp_colors = ["#4fc3f7","#81d4fa","#64ffda","#ffd54f","#ff8a65","#ce93d8","#a5d6a7","#90caf9","#ffcc02","#ef9a9a"]
        fig_pie = px.pie(
            df_w,
            values="وزن (%)",
            names="دارایی",
            title="PORTFOLIO ALLOCATION",
            color_discrete_sequence=bp_colors
        )
        fig_pie.update_traces(
            textposition='inside', textinfo='percent+label',
            textfont=dict(color="#0a1628", size=11, family="Courier New"),
            marker=dict(line=dict(color="#0a1628", width=2))
        )
        fig_pie.update_layout(
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0a1628",
            font=dict(color="#e8f4fd", family="Courier New"),
            title=dict(font=dict(color="#4fc3f7", size=12)),
            legend=dict(bgcolor="rgba(10,22,40,0.8)", bordercolor="rgba(255,255,255,0.15)",
                        borderwidth=1, font=dict(color="#e8f4fd", size=10)),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Capital allocation
    st.markdown("### 💰 تخصیص سرمایه (جزئیات)")
    show_help("capital_allocation")
    
    alloc_df = capital_allocator_calculator(weights, asset_names, capital_usd, exchange_rate)
    st.dataframe(
        alloc_df[["دارایی", "درصد وزن", "دلار ($)", "تومان", "ریال"]], 
        use_container_width=True,
        hide_index=True
    )
    
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        st.download_button(
            "📥 دانلود CSV",
            alloc_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # =============================================================================
    # MONTE CARLO FORECAST SECTION
    # =============================================================================
    st.markdown('<div class="section-header">🔮 پیش‌بینی قیمت (Monte Carlo Simulation)</div>', unsafe_allow_html=True)
    show_help("monte_carlo_forecast")
    
    col_mc1, col_mc2, col_mc3 = st.columns([2, 1, 1])
    
    with col_mc1:
        sel_asset = st.selectbox("دارایی برای پیش‌بینی", asset_names)
    
    with col_mc2:
        days_forecast = st.slider("روزهای پیش‌بینی", 30, 365, 90)
    
    with col_mc3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_forecast = st.button("🚀 اجرای پیش‌بینی", use_container_width=True)
    
    if run_forecast:
        with st.spinner("در حال شبیه‌سازی مونت‌کارلو..."):
            series = prices[sel_asset]
            paths = forecast_price_series(series, days=days_forecast, sims=400)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                name="قیمت واقعی",
                line=dict(color="#4fc3f7", width=2)
            ))

            future_x = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=days_forecast)
            median = np.percentile(paths, 50, axis=1)
            p10 = np.percentile(paths, 10, axis=1)
            p90 = np.percentile(paths, 90, axis=1)

            fig.add_trace(go.Scatter(
                x=future_x,
                y=median,
                name="میانه پیش‌بینی",
                line=dict(color="#ffd54f", width=2)
            ))

            fig.add_trace(go.Scatter(
                x=future_x,
                y=p90,
                name="صدک 90%",
                line=dict(color="rgba(100,255,218,0.4)", width=1),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=future_x,
                y=p10,
                name="صدک 10%",
                line=dict(color="rgba(100,255,218,0.4)", width=1),
                fill='tonexty',
                fillcolor='rgba(79,195,247,0.08)',
                showlegend=False
            ))

            fig.update_layout(**bp_layout(
                title=f"PRICE FORECAST — {sel_asset} — {days_forecast} DAYS",
                xaxis_title="DATE",
                yaxis_title="PRICE ($)",
                height=500
            ))
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            current_price = series.iloc[-1]
            predicted_price = median[-1]
            change_pct = ((predicted_price - current_price) / current_price) * 100
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("قیمت فعلی", f"${current_price:,.2f}")
            with col_m2:
                st.metric("قیمت پیش‌بینی شده", f"${predicted_price:,.2f}", f"{change_pct:+.2f}%")
            with col_m3:
                st.metric("دامنه اطمینان 80%", f"${p10[-1]:,.2f} - ${p90[-1]:,.2f}")
    
    st.markdown("---")
    
    # =============================================================================
    # MARRIED PUT SECTION
    # =============================================================================
    st.markdown('<div class="section-header">🛡️ Protective Put (Married Put) - تحلیل پیشرفته</div>', unsafe_allow_html=True)
    show_help("married_put")
    
    btc_col = next((c for c in asset_names if "BTC" in c.upper()), None)
    eth_col = next((c for c in asset_names if "ETH" in c.upper()), None)
    
    col_mp1, col_mp2 = st.columns(2)
    
    with col_mp1:
        if btc_col:
            st.subheader("🔸 BTC-USD")
            btc_price = float(prices[btc_col].iloc[-1])
            btc_strike = st.number_input("Strike BTC ($)", value=btc_price*0.90, step=10.0, key="btc_strike")
            btc_premium = st.number_input("Premium هر قرارداد ($)", value=max(0.0, btc_price*0.04), step=1.0, key="btc_prem")
            btc_contracts = st.number_input("تعداد قرارداد (long put)", min_value=0, max_value=200, value=0, step=1, key="btc_contracts")
            btc_contract_size = st.number_input("اندازه هر قرارداد (BTC)", min_value=0.01, max_value=100.0, value=1.0, step=0.01, key="btc_size")
    
    with col_mp2:
        if eth_col:
            st.subheader("🔹 ETH-USD")
            eth_price = float(prices[eth_col].iloc[-1])
            eth_strike = st.number_input("Strike ETH ($)", value=eth_price*0.90, step=5.0, key="eth_strike")
            eth_premium = st.number_input("Premium هر قرارداد ($)", value=max(0.0, eth_price*0.04), step=0.5, key="eth_prem")
            eth_contracts = st.number_input("تعداد قرارداد (long put)", min_value=0, max_value=200, value=0, step=1, key="eth_contracts")
            eth_contract_size = st.number_input("اندازه هر قرارداد (ETH)", min_value=0.01, max_value=1000.0, value=1.0, step=0.01, key="eth_size")
    
    # Zoom controls
    st.markdown("### 🔍 تنظیمات نمایش نمودار")
    zcol1, zcol2 = st.columns(2)
    zoom_min_pct = zcol1.slider("کاهش حداقل نسبت به قیمت فعلی (%)", 10, 100, 80)
    zoom_max_pct = zcol2.slider("حداکثر نسبت به قیمت فعلی (%)", 100, 250, 140)
    
    if st.button("📊 نمایش نمودار Payoff", use_container_width=True):
        exposures = {asset_names[i]: float(weights[i])*capital_usd for i in range(len(asset_names))}
        units_btc = exposures.get(btc_col, 0.0) / (btc_price + 1e-8) if btc_col else 0.0
        units_eth = exposures.get(eth_col, 0.0) / (eth_price + 1e-8) if eth_col else 0.0
        
        traces = []
        all_prices = np.array([])
        
        if btc_col and btc_contracts > 0:
            grid_btc, married_btc, btc_prem_paid = married_put_pnl_grid(
                btc_price, btc_strike, btc_premium, units_btc, 
                int(btc_contracts), float(btc_contract_size)
            )
            traces.append(("BTC", grid_btc, married_btc, "#ff8c00"))
            all_prices = np.concatenate([all_prices, grid_btc])
        
        if eth_col and eth_contracts > 0:
            grid_eth, married_eth, eth_prem_paid = married_put_pnl_grid(
                eth_price, eth_strike, eth_premium, units_eth,
                int(eth_contracts), float(eth_contract_size)
            )
            traces.append(("ETH", grid_eth, married_eth, "#1f77b4"))
            all_prices = np.concatenate([all_prices, grid_eth])
        
        fig = go.Figure()
        
        for name, grid, pnl, color in traces:
            fig.add_trace(go.Scatter(
                x=grid, y=pnl, 
                name=f"{name} Married Put", 
                mode="lines", 
                line=dict(color=color, width=2)
            ))
            fig.add_trace(go.Scatter(
                x=grid, y=np.where(pnl>=0, pnl, np.nan), 
                fill='tozeroy', mode='none', 
                fillcolor='rgba(50,205,50,0.15)', 
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=grid, y=np.where(pnl<0, pnl, np.nan), 
                fill='tozeroy', mode='none', 
                fillcolor='rgba(255,99,71,0.15)', 
                showlegend=False
            ))
        
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
            
            fig.add_trace(go.Scatter(
                x=common_grid, y=total_payoff, 
                name="Total Payoff", 
                mode="lines", 
                line=dict(color="#2ca02c", width=3)
            ))
            
            sign_t = np.sign(total_payoff)
            cross_t = np.where(np.diff(sign_t) != 0)[0]
            if cross_t.size > 0:
                be_total = common_grid[cross_t[-1]]
                fig.add_vline(
                    x=be_total, 
                    line_dash="dash", 
                    line_color="black", 
                    annotation_text=f"Total BE ~ ${be_total:.2f}", 
                    annotation_position="bottom right"
                )
        
        # Per-asset BE lines
        if btc_col and btc_contracts > 0:
            be_btc = btc_price + btc_premium
            fig.add_vline(
                x=be_btc, 
                line_dash="dot", 
                line_color="#ff8c00", 
                annotation_text=f"BTC BE = ${be_btc:.2f}", 
                annotation_position="top left"
            )
        
        if eth_col and eth_contracts > 0:
            be_eth = eth_price + eth_premium
            fig.add_vline(
                x=be_eth, 
                line_dash="dot", 
                line_color="#1f77b4", 
                annotation_text=f"ETH BE = ${be_eth:.2f}", 
                annotation_position="top right"
            )
        
        # Zoom
        s0_candidates = []
        if btc_col and btc_contracts > 0: 
            s0_candidates.append(btc_price)
        if eth_col and eth_contracts > 0: 
            s0_candidates.append(eth_price)
        
        base = float(np.mean(s0_candidates)) if s0_candidates else 1.0
        display_min = base * (zoom_min_pct / 100.0)
        display_max = base * (zoom_max_pct / 100.0)
        fig.update_xaxes(range=[display_min, display_max])

        fig.update_layout(**bp_layout(
            title="MARRIED PUT — P&L DIAGRAM",
            xaxis_title="PRICE ($)",
            yaxis_title="P&L (USD)",
            height=560
        ))
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=40, r=20, t=80, b=40),
        )
        
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
        
        # Risk metrics
        st.markdown("### 📉 معیارهای ریسک")
        show_help("risk_metrics")
        
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
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("ریسک بدون بیمه", f"{original_risk:.2f}%")
        with col_r2:
            st.metric("ریسک با بیمه", f"{new_risk:.2f}%")
        with col_r3:
            st.metric("کاهش ریسک", f"{original_risk - new_risk:.3f}%")
        with col_r4:
            st.metric("کل Premium", f"${btc_total_premium + eth_total_premium:,.2f}")
        
        # Suggestion helper
        st.markdown("### 💡 پیشنهاد خودکار برای رسیدن به هدف ریسک")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            est_btc_prem = st.number_input(
                "برآورد Premium BTC ($)",
                value=float(btc_premium if (btc_col and btc_contracts>0) else 0.0),
                step=1.0
            )
        with col_s2:
            est_eth_prem = st.number_input(
                "برآورد Premium ETH ($)",
                value=float(eth_premium if (eth_col and eth_contracts>0) else 0.0),
                step=0.5
            )
        with col_s3:
            target_risk = st.number_input(
                "هدف ریسک کل (%)",
                min_value=0.5,
                max_value=20.0,
                value=2.0,
                step=0.1
            )
        
        max_search = st.number_input(
            "حداکثر قرارداد برای جستجو",
            min_value=1,
            max_value=200,
            value=30,
            step=1
        )
        
        if st.button("🔎 دریافت پیشنهاد"):
            suggestion = suggest_contracts_for_target_risk(
                prices, returns, asset_names, weights, cov_mat, capital_usd,
                btc_idx, eth_idx,
                float(btc_contract_size if btc_contract_size else 1.0),
                float(eth_contract_size if eth_contract_size else 1.0),
                float(est_btc_prem), float(est_eth_prem),
                max_contracts=int(max_search),
                target_risk_pct=float(target_risk)
            )
            
            if suggestion:
                st.success(f"""
                ✅ **پیشنهاد بهینه:**
                - قرارداد BTC: **{suggestion['b']}**
                - قرارداد ETH: **{suggestion['e']}**
                - هزینه کل: **${suggestion['total_premium']:.2f}**
                - ریسک جدید: **{suggestion['new_risk']:.3f}%**
                """)
            else:
                st.info("⚠️ پیشنهادی یافت نشد. لطفاً پارامترها را بررسی کنید.")
    
    st.markdown("---")
    
    # =============================================================================
    # DCA SECTION
    # =============================================================================
    st.markdown('<div class="section-header">⏳ DCA زمانی (Time-based Dollar-Cost Averaging)</div>', unsafe_allow_html=True)
    show_help("dca_time")
    
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
        dca_start_date = st.date_input(
            "تاریخ شروع",
            value=(prices.index[0] + pd.Timedelta(days=1)).date()
        )
    
    with col_dca6:
        use_levels = st.checkbox("استفاده از سطوح قیمتی", value=False)
    
    levels_input = None
    if use_levels:
        levels_txt = st.text_input(
            "سطوح قیمتی (با کاما جدا کنید)",
            placeholder="مثال: 2500,2200,1800",
            help="سطوح قیمتی برای خرید متغیر"
        )
        try:
            levels_input = [float(x.strip()) for x in levels_txt.split(",") if x.strip()]
            if len(levels_input) == 0:
                levels_input = None
        except Exception:
            levels_input = None
    
    if st.button("▶️ اجرای شبیه‌سازی DCA", use_container_width=True):
        with st.spinner("در حال شبیه‌سازی..."):
            series = prices[dca_asset]
            df_purchases, summary = simulate_time_dca(
                series, dca_total, int(dca_periods),
                int(dca_freq_days), start_date=dca_start_date, levels=levels_input
            )
            
            st.markdown("#### 📋 جدول معاملات")
            st.dataframe(
                df_purchases[["date", "price", "amount_usd", "units", "level_assigned"]].assign(
                    date=lambda d: d["date"].dt.strftime("%Y-%m-%d")
                ),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("#### 📊 خلاصه نتایج")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("کل سرمایه‌گذاری", f"${summary['total_invested']:.2f}")
                st.metric("تعداد دوره‌ها", f"{int(dca_periods)}")
            with col_res2:
                st.metric("میانگین قیمت خرید", f"${summary['avg_price_per_unit']:.4f}")
                st.metric("قیمت نهایی", f"${summary['final_price']:.2f}")
            with col_res3:
                st.metric("ارزش کنونی", f"${summary['final_value']:.2f}")
                st.metric(
                    "سود/زیان",
                    f"${summary['profit']:.2f}",
                    f"{summary['profit_pct']:.2f}%"
                )
            
            fig_p = plot_price_with_purchases(series, df_purchases, title=f"DCA روی {dca_asset}")
            st.plotly_chart(fig_p, use_container_width=True)
            
            csv = df_purchases.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 دانلود معاملات DCA (CSV)",
                csv,
                file_name=f"dca_{dca_asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                use_container_width=True
            )

    st.markdown("---")

    # =============================================================================
    # COVERED CALL SECTION
    # =============================================================================
    st.markdown('<div class="section-header">📞 Covered Call Strategy Analyzer</div>', unsafe_allow_html=True)

    with st.expander("📘 تحلیل استراتژی Covered Call"):
        cc_asset = st.selectbox("دارایی پایه", asset_names, key="cc_asset")
        cc_price = float(prices[cc_asset].iloc[-1])

        col_cc1, col_cc2, col_cc3 = st.columns(3)
        with col_cc1:
            cc_strike = st.number_input("Strike", value=float(cc_price * 1.10), key="cc_strike")
        with col_cc2:
            cc_premium = st.number_input("Premium دریافتی", value=float(cc_price * 0.03), key="cc_premium")
        with col_cc3:
            cc_units = st.number_input("تعداد واحد", value=1.0, min_value=0.01, key="cc_units")

        if st.button("محاسبه Covered Call", key="run_cc"):
            cc_grid = np.linspace(cc_price * 0.5, cc_price * 1.8, 500)
            stock_pnl = (cc_grid - cc_price) * cc_units
            short_call_pnl = cc_premium * cc_units - np.maximum(cc_grid - cc_strike, 0) * cc_units
            total_pnl = stock_pnl + short_call_pnl

            fig_cc = go.Figure()
            fig_cc.add_trace(go.Scatter(
                x=cc_grid, y=total_pnl, name="Covered Call",
                line=dict(color="#4fc3f7", width=2)
            ))
            fig_cc.add_trace(go.Scatter(
                x=cc_grid, y=np.where(total_pnl >= 0, total_pnl, np.nan),
                fill='tozeroy', mode='none',
                fillcolor='rgba(100,255,218,0.08)', showlegend=False
            ))
            fig_cc.add_trace(go.Scatter(
                x=cc_grid, y=np.where(total_pnl < 0, total_pnl, np.nan),
                fill='tozeroy', mode='none',
                fillcolor='rgba(255,107,107,0.08)', showlegend=False
            ))
            fig_cc.update_layout(**bp_layout(
                title=f"COVERED CALL PAYOFF — {cc_asset}",
                xaxis_title="PRICE ($)",
                yaxis_title="P&L ($)"
            ))
            st.plotly_chart(fig_cc, use_container_width=True)

            max_profit = ((cc_strike - cc_price) + cc_premium) * cc_units
            breakeven = cc_price - cc_premium

            c1, c2, c3 = st.columns(3)
            c1.metric("حداکثر سود", f"${max_profit:,.2f}")
            c2.metric("نقطه سربه‌سر", f"${breakeven:,.2f}")
            c3.metric("Premium دریافتی", f"${cc_premium * cc_units:,.2f}")

    st.markdown("---")

    # =============================================================================
    # ADVANCED COVERED CALL SUITE
    # =============================================================================
    st.markdown('<div class="section-header">🚀 Advanced Covered Call Suite</div>', unsafe_allow_html=True)

    with st.expander("Covered Call Optimization + Greeks + Probability + Wheel + Backtest"):
        adv_asset = st.selectbox("دارایی", asset_names, key="adv_cc_asset")
        s_adv = float(prices[adv_asset].iloc[-1])

        adv_c1, adv_c2, adv_c3, adv_c4 = st.columns(4)
        with adv_c1:
            adv_strike = st.number_input("Strike Price", value=float(s_adv * 1.1), key="adv_strike")
        with adv_c2:
            adv_premium = st.number_input("Premium", value=float(s_adv * 0.03), key="adv_premium")
        with adv_c3:
            adv_days_dte = st.number_input("Days To Expiry", value=30, key="adv_days")
        with adv_c4:
            adv_sigma = st.number_input("Volatility", value=0.60, key="adv_sigma")

        adv_T = adv_days_dte / 365
        adv_greeks = bs_call_greeks(s_adv, adv_strike, adv_T, 0.05, adv_sigma)

        st.subheader("Greeks")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Delta", f"{adv_greeks['Delta']:.4f}")
        g2.metric("Gamma", f"{adv_greeks['Gamma']:.6f}")
        g3.metric("Theta", f"{adv_greeks['Theta']:.4f}")
        g4.metric("Vega", f"{adv_greeks['Vega']:.4f}")

        prob_itm = norm_cdf((math.log(s_adv / adv_strike)) / (adv_sigma * math.sqrt(adv_T) + 1e-9))
        st.metric("احتمال اعمال شدن آپشن (ITM Probability)", f"{prob_itm * 100:.2f}%")

        st.subheader("Covered Call Optimization")
        candidate_strikes = [s_adv * x for x in [1.02, 1.05, 1.1, 1.15, 1.2]]
        opt_rows = []
        for k in candidate_strikes:
            expected_income = (k - s_adv) * 0.5 + adv_premium
            opt_rows.append([round(k, 2), round(expected_income, 2)])
        opt_df = pd.DataFrame(opt_rows, columns=["Strike", "Score"])
        st.dataframe(opt_df, use_container_width=True)

        st.subheader("Wheel Strategy")
        st.info(
            "مرحله 1: فروش Cash Secured Put → دریافت پریمیوم | "
            "مرحله 2: در صورت Assign شدن خرید دارایی | "
            "مرحله 3: فروش Covered Call روی دارایی | "
            "مرحله 4: تکرار چرخه."
        )

        st.subheader("Historical Covered Call Backtest")
        adv_series = prices[adv_asset].dropna()
        monthly_adv = adv_series.resample("30D").last()
        if len(monthly_adv) > 2:
            total_return = ((monthly_adv.iloc[-1] - monthly_adv.iloc[0]) / monthly_adv.iloc[0]) * 100
            cc_return = total_return + (len(monthly_adv) * adv_premium / s_adv * 100)
            b1, b2 = st.columns(2)
            b1.metric("Buy & Hold", f"{total_return:.2f}%")
            b2.metric("Covered Call تقریبی", f"{cc_return:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>📊 <strong>Portfolio360 Ultimate Pro</strong> — نسخه حرفه‌ای</p>
    <p style="font-size: 0.8rem;">سیستم جامع تحلیل و مدیریت پرتفوی | طراحی شده با ❤️</p>
</div>
""", unsafe_allow_html=True)
