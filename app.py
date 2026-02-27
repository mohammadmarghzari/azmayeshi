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
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #e0e0e0 !important;
        font-size: 1.1rem !important;
        margin-top: 0.5rem !important;
    }
    
    /* Card styling */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Help box styling */
    .help-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border: 1px solid #d1d5db;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0 1rem 0;
    }
    
    .help-box h4 {
        color: #374151;
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    
    .help-box p {
        color: #6b7280;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Metric cards */
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-left: 4px solid #0ea5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #f3f4f6 0%, #e5e7eb 100%);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        color: #374151;
    }
    
    /* Tooltip icon */
    .tooltip-icon {
        color: #667eea;
        cursor: help;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELP TEXTS DICTIONARY
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
        - **GC=F**: طلای جهانی
        - **USDIRR=X**: نرخ دلار به ریال ایران
        - **^GSPC**: شاخص S&P 500
        
        **بازه‌های زمانی:**
        - 1y: یک سال گذشته
        - 2y: دو سال گذشته
        - 5y: پنج سال گذشته
        - 10y: ده سال گذشته
        - max: تمام داده‌های موجود
        """
    },
    
    "risk_free_rate": {
        "title": "📊 نرخ بدون ریسک",
        "content": """
        **این بخش چه کاری انجام می‌دهد؟**
        
        نرخ بدون ریسک نرخ بازدهی است که می‌توانید با صفر ریسک دریافت کنید.
        
        **کاربردها:**
        - محاسبه نسبت شارپ
        - ارزیابی عملکرد پرتفوی
        - بهینه‌سازی پرتفوی
        
        **مقادیر پیشنهادی برای ایران:**
        - سپرده بانکی: ~18-22%
        - اوراق مشارکت: ~20-25%
        """
    },
    
    "hedge_strategy": {
        "title": "🛡️ استراتژی‌های هجینگ",
        "content": """
        **انواع استراتژی‌ها:**
        
        🔹 **Barbell طالب (90/10)**: 45% طلا + 45% دلار + 10% بیت‌کوین
        🔹 **Tail-Risk طالب**: 35% طلا + 35% دلار + 5% بیت‌کوین
        🔹 **Antifragile طالب**: 40% طلا + 20% دلار + 40% بیت‌کوین
        🔹 **طلا + تتر**: ترکیب متعادل
        🔹 **حداقل هج**: حداقل پوشش
        🔹 **بدون هجینگ**: بدون پوشش
        """
    },
    
    "option_strategy": {
        "title": "📈 استراتژی‌های آپشن",
        "content": """
        **انواع استراتژی‌ها:**
        
        🔹 **بدون آپشن**: هیچ استراتژی آپشنی
        🔹 **Protective Put**: بیمه کامل (هزینه: ~4.8%)
        🔹 **Collar**: هج کم‌هزینه (هزینه: ~0.4%)
        🔹 **Covered Call**: درآمد ماهانه (درآمد: ~-3.2%)
        🔹 **Tail-Risk Put**: محافظت (هزینه: ~2.1%)
        """
    },
    
    "portfolio_style": {
        "title": "🎯 سبک‌های پرتفوی",
        "content": """
        **انواع سبک‌ها:**
        
        🔹 **مارکوویتز + هجینگ**: بهینه‌ترین نسبت شارپ
        🔹 **وزن برابر**: وزن یکسان برای همه دارایی‌ها
        🔹 **حداقل ریسک**: کمترین ریسک ممکن
        🔹 **ریسک‌پاریتی**: وزن‌دهی بر اساس ریسک
        🔹 **مونت‌کارلو مقاوم**: شبیه‌سازی‌های متعدد
        🔹 **HRP**: خوشه‌بندی سلس��ه‌مراتبی
        🔹 **Maximum Diversification**: حداکثر تنوع
        🔹 **Inverse Volatility**: وزن معکوس نوسان
        🔹 **Kelly Criterion**: حداکثر رشد سرمایه
        🔹 **Black-Litterman**: ترکیب نظر شخصی
        """
    },
    
    "capital_allocation": {
        "title": "💰 تخصیص سرمایه",
        "content": """
        **این بخش محاسبه مبلغ دقیق سرمایه‌گذاری برای هر دارایی را انجام می‌دهد.**
        
        **خروجی‌ها:**
        - درصد وزن هر دارایی
        - مبلغ به دلار
        - مبلغ به تومان
        - مبلغ به ریال
        """
    },
    
    "monte_carlo_forecast": {
        "title": "🔮 پیش‌بینی مونت‌کارلو",
        "content": """
        **این بخش شبیه‌سازی مسیرهای احتمالی قیمت آینده را انجام می‌دهد.**
        
        **کاربردها:**
        - تخمین محدوده قیمت آینده
        - ارزیابی ریسک سرمایه‌گذاری
        - برنامه‌ریزی استراتژیک
        """
    },
    
    "married_put": {
        "title": "🛡️ Protective Put (Married Put)",
        "content": """
        **تحلیل استراتژی Married Put برای محافظت از سرمایه.**
        """
    },
    
    "dca_time": {
        "title": "⏳ DCA زمانی (Time-based DCA)",
        "content": """
        **شبیه‌سازی استراتژی Dollar-Cost Averaging مبتنی بر زمان.**
        
        **مزایا:**
        - میانگین قیمت خرید بهینه
        - کاهش ریسک زمان‌بندی
        - حمایت از روانشناسی سرمایه‌گذاری
        """
    },
    
    "risk_metrics": {
        "title": "📉 معیارهای ریسک",
        "content": """
        **معیارهای کلیدی ریسک پرتفوی:**
        
        - ریسک پرتفوی (بدون بیمه)
        - ریسک پرتفوی (با Married Put)
        - کاهش ریسک
        - کل Premium پرداختی
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
# PORTFOLIO OPTIMIZATION METHODS - ALL STYLES WORKING
# =============================================================================

def get_portfolio_weights(style, returns, mean_ret, cov_mat, rf, bounds=None):
    """
    محاسبه وزن‌های پرتفوی بر اساس سبک انتخاب شده
    """
    n = len(mean_ret)
    
    # 1. وزن برابر (ساده �� مقاوم)
    if style == "وزن برابر (ساده و مقاوم)":
        return np.ones(n) / n
    
    # 2. حداقل ریسک
    elif style == "حداقل ریسک (محافظه‌کارانه)":
        def objective(w):
            return np.dot(w.T, np.dot(cov_mat, w))
        
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bnds = tuple((0, 1) for _ in range(n))
        result = minimize(objective, np.ones(n) / n, method='SLSQP', 
                         bounds=bnds, constraints=cons)
        return result.x if result.success else np.ones(n) / n
    
    # 3. Inverse Volatility (وزن معکوس نوسان)
    elif style == "Inverse Volatility":
        vol = np.sqrt(np.diag(cov_mat))
        w = 1.0 / (vol + 1e-8)
        return w / w.sum()
    
    # 4. مارکوویتز + هجینگ (بهینه‌ترین شارپ)
    elif style == "مارکوویتز + هجینگ (بهینه‌ترین شارپ)":
        def neg_sharpe(w):
            port_ret = np.dot(w, mean_ret)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            return -(port_ret - rf) / (port_vol + 1e-8)
        
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bnds = tuple((0, 1) for _ in range(n))
        result = minimize(neg_sharpe, np.ones(n) / n, method='SLSQP',
                         bounds=bnds, constraints=cons)
        return result.x if result.success else np.ones(n) / n
    
    # 5. ریسک‌پاریتی (Risk Parity)
    elif style == "ریسک‌پاریتی (Risk Parity)":
        def objective_rp(w):
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            contrib = w * np.dot(cov_mat, w) / (port_vol + 1e-8)
            return np.sum((contrib - port_vol/n)**2)
        
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bnds = tuple((0, 1) for _ in range(n))
        result = minimize(objective_rp, np.ones(n) / n, method='SLSQP',
                         bounds=bnds, constraints=cons)
        return result.x if result.success else np.ones(n) / n
    
    # 6. مونت‌کارلو مقاوم (Resampled Frontier)
    elif style == "مونت‌کارلو مقاوم (Resampled Frontier)":
        weights_list = []
        for _ in range(50):
            ret_sample = returns.sample(len(returns), replace=True)
            mean_ret_s = ret_sample.mean()
            cov_mat_s = ret_sample.cov()
            
            def neg_sharpe_s(w):
                port_ret = np.dot(w, mean_ret_s)
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat_s, w)))
                return -(port_ret - rf) / (port_vol + 1e-8)
            
            cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bnds = tuple((0, 1) for _ in range(n))
            result = minimize(neg_sharpe_s, np.ones(n) / n, method='SLSQP',
                             bounds=bnds, constraints=cons)
            if result.success:
                weights_list.append(result.x)
        
        if weights_list:
            return np.mean(weights_list, axis=0)
        return np.ones(n) / n
    
    # 7. HRP (سلسله‌مراتبی)
    elif style == "HRP (سلسله‌مراتبی)":
        corr = returns.corr()
        distances = np.sqrt((1 - corr) / 2)
        dist_matrix = squareform(distances.values[np.triu_indices_from(distances.values, k=1)])
        Z = linkage(dist_matrix, method='ward')
        
        # ساختن شاخص‌های خوشه
        from scipy.cluster.hierarchy import dendrogram
        dendro = dendrogram(Z, no_plot=True)
        leaf_order = dendro['leaves']
        
        w = np.ones(n) / n
        for i in leaf_order:
            w[i] = 1.0 / n
        return w
    
    # 8. Maximum Diversification
    elif style == "Maximum Diversification":
        vol = np.sqrt(np.diag(cov_mat))
        
        def objective_md(w):
            contrib = w * vol
            div_ratio = np.sum(contrib) / np.sqrt(np.dot(w.T, np.dot(cov_mat, w)) + 1e-8)
            return -div_ratio
        
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bnds = tuple((0, 1) for _ in range(n))
        result = minimize(objective_md, np.ones(n) / n, method='SLSQP',
                         bounds=bnds, constraints=cons)
        return result.x if result.success else np.ones(n) / n
    
    # 9. Kelly Criterion
    elif style == "Kelly Criterion (حداکثر رشد)":
        inv_cov = np.linalg.pinv(cov_mat)
        kelly_w = np.dot(inv_cov, mean_ret - rf)
        kelly_w = kelly_w / np.sum(np.abs(kelly_w))
        kelly_w = np.maximum(kelly_w, 0)
        kelly_w = kelly_w / (np.sum(kelly_w) + 1e-8)
        return kelly_w
    
    # 10. Black-Litterman
    elif style == "بلک-لیترمن (ترکیب نظر شخصی)":
        views = mean_ret.copy()
        view_conf = 0.5
        P = np.eye(n)
        Q = views
        
        omega = view_conf * cov_mat
        inv_cov = np.linalg.pinv(cov_mat)
        bl_ret = mean_ret + np.dot(cov_mat, np.dot(P.T, 
                 np.linalg.solve(np.dot(P, np.dot(cov_mat, P.T)) + omega, 
                 Q - np.dot(P, mean_ret))))
        
        def neg_sharpe_bl(w):
            port_ret = np.dot(w, bl_ret)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            return -(port_ret - rf) / (port_vol + 1e-8)
        
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bnds = tuple((0, 1) for _ in range(n))
        result = minimize(neg_sharpe_bl, np.ones(n) / n, method='SLSQP',
                         bounds=bnds, constraints=cons)
        return result.x if result.success else np.ones(n) / n
    
    # 11. Barbell طالب
    elif style == "Barbell طالب (۹۰/۱۰)":
        w = np.ones(n) * 0.02
        if n >= 3:
            w[0] = 0.45
            w[1] = 0.45
            w[2] = 0.10
        return w / w.sum()
    
    # 12. Antifragile طالب
    elif style == "Antifragile طالب":
        w = np.ones(n) * 0.02
        if n >= 3:
            w[0] = 0.40
            w[1] = 0.20
            w[2] = 0.40
        return w / w.sum()
    
    # 13. Equal Risk Bounding
    elif style == "Equal Risk Bounding":
        vol = np.sqrt(np.diag(cov_mat))
        w = 1.0 / vol
        return w / w.sum()
    
    # 14. Most Diversified Portfolio
    elif style == "Most Diversified Portfolio":
        vol = np.sqrt(np.diag(cov_mat))
        w = vol / np.sum(vol)
        return w
    
    # پیش‌فرض
    else:
        return np.ones(n) / n

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
            btc_premium_pct = (btc_total_premium / (exposures.get(btc_name, 1e-8))) * 100 if btc_name else 0.0
            eth_premium_pct = (eth_total_premium / (exposures.get(eth_name, 1e-8))) * 100 if eth_name else 0.0
            btc_reduction = min(0.95, btc_premium_pct * 0.5 / 100.0)
            eth_reduction = min(0.95, eth_premium_pct * 0.5 / 100.0)
            cov_adj = apply_protective_put_to_cov(cov_mat, asset_names, btc_idx, eth_idx, btc_reduction, eth_reduction)
            new_risk = np.sqrt(np.dot(weights.T, np.dot(cov_adj, weights))) * 100
            total_premium = btc_total_premium + eth_total_premium
            
            if new_risk <= target_risk_pct:
                if best is None or total_premium < best["total_premium"] or (total_premium == best["total_premium"] and (b+e) < (best["b"]+best["e"])):
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
# DCA HELPERS
# =============================================================================

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
        "profit_pct": profit_pct
    }
    return df, summary

def plot_price_with_purchases(price_series, purchases_df, title="Price with purchases"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index, 
        y=price_series.values, 
        name="Price", 
        mode="lines", 
        line=dict(color="#0b69ff")
    ))
    if not purchases_df.empty:
        fig.add_trace(go.Scatter(
            x=purchases_df["date"], 
            y=purchases_df["price"], 
            mode="markers+text", 
            name="Purchases", 
            marker=dict(size=8, color="orange"), 
            text=[f"{a:.2f}$" for a in purchases_df["amount_usd"]]
        ))
    
    fig.update_layout(
        title=title, 
        xaxis_title="Date", 
        yaxis_title="Price ($)", 
        template="plotly_white", 
        height=480
    )
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

# =============================================================================
# HEDGE & OPTION STRATEGIES
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

# =============================================================================
# MAIN APPLICATION
# =============================================================================

st.set_page_config(
    page_title="Portfolio360 Ultimate Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        help="نمادهای یاهو فایننس را وارد کنید."
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
        step=0.5
    )
    show_help("risk_free_rate")
    
    if "hedge_strategy" not in st.session_state: 
        st.session_state.hedge_strategy = list(hedge_strategies.keys())[3]
    
    st.session_state.hedge_strategy = st.selectbox(
        "استراتژی هجینگ",
        list(hedge_strategies.keys()),
        index=list(hedge_strategies.keys()).index(st.session_state.hedge_strategy)
    )
    show_help("hedge_strategy")
    
    if "option_strategy" not in st.session_state: 
        st.session_state.option_strategy = list(option_strategies.keys())[0]
    
    st.session_state.option_strategy = st.selectbox(
        "استراتژی آپشن",
        list(option_strategies.keys())
    )
    show_help("option_strategy")

# =============================================================================
# MAIN CONTENT
# =============================================================================

if "prices" not in st.session_state or st.session_state.prices is None:
    st.info("👈 لطفاً ابتدا از سایدبار داده‌ها را دانلود کنید.")
    
    st.markdown("""
    <div class="info-box">
        <h4>🚀 راهنمای شروع سریع</h4>
        <ol>
            <li>در سایدبار، نمادهای مورد نظر را وارد کنید</li>
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
            index=styles.index(st.session_state.selected_style)
        )
    
    with colB:
        capital_usd = st.number_input(
            "کل سرمایه (دلار)",
            min_value=1,
            max_value=50_000_000,
            value=1200,
            step=100
        )
        
        exchange_rate = st.number_input(
            "نرخ تبدیل (تومان/دلار)",
            min_value=1000,
            max_value=1_000_000_000,
            value=200_000,
            step=1000
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
        fig_pie = px.pie(
            df_w, 
            values="وزن (%)", 
            names="دارایی", 
            title="توزیع پرتفوی",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
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
                line=dict(color="#1f77b4", width=2)
            ))
            
            future_x = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=days_forecast)
            median = np.percentile(paths, 50, axis=1)
            p10 = np.percentile(paths, 10, axis=1)
            p90 = np.percentile(paths, 90, axis=1)
            
            fig.add_trace(go.Scatter(
                x=future_x, 
                y=median, 
                name="میانه پیش‌بینی", 
                line=dict(color="orange", width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=future_x, 
                y=p90, 
                name="صدک 90%",
                line=dict(color="rgba(255,165,0,0.3)", width=1),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future_x, 
                y=p10, 
                name="صدک 10%",
                line=dict(color="rgba(255,165,0,0.3)", width=1),
                fill='tonexty',
                fillcolor='rgba(255,165,0,0.1)',
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"پیش‌بینی قیمت {sel_asset} - {days_forecast} روز آینده",
                xaxis_title="تاریخ",
                yaxis_title="قیمت ($)",
                template="plotly_white",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
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
            
            if any(t[0] == "BTC" for t in traces):
                total_payoff += np.interp(common_grid, grid_btc, married_btc)
            if any(t[0] == "ETH" for t in traces):
                total_payoff += np.interp(common_grid, grid_eth, married_eth)
            
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
        
        if btc_col and btc_contracts > 0:
            be_btc = btc_price + btc_premium
            fig.add_vline(
                x=be_btc, 
                line_dash="dot", 
                line_color="#ff8c00", 
                annotation_text=f"BTC BE = ${be_btc:.2f
