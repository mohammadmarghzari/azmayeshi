"""
Mr.option11 — Professional Edition
- 3 Portfolio Optimization Strategies
- Monte Carlo Forecast
- Capital Allocation
- Self-contained single-file Streamlit app
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
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
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .help-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border: 1px solid #d1d5db;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-left: 4px solid #0ea5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
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
        **نمادهای قابل استفاده:**
        - **BTC-USD**: بیت‌کوین به دلار
        - **ETH-USD**: اتریوم به دلار  
        - **GC=F**: طلای جهانی
        - **USDIRR=X**: نرخ دلار به ریال
        - **^GSPC**: شاخص S&P 500
        """
    },
    "portfolio_styles": {
        "title": "📚 راهنمای سبک‌های پرتفوی",
        "content": """
        **سبک‌های موجود:**
        - **حداکثر شارپ (مارکوویتز)**: بهینه‌سازی نسبت شارپ برای بهترین بازده نسبت به ریسک
        - **حداقل واریانس**: کمترین ریسک ممکن و بیشترین پایداری
        - **حداقل CVaR**: محافظت در برابر بدترین شرایط بازار (ریسک شرطی)
        """
    },
    "risk_free_rate": {
        "title": "📊 نرخ بدون ریسک",
        "content": "نرخ بازدهی بدون ریسک برای محاسبه شارپ. برای ایران: 18-25%"
    },
    "hedge_strategy": {
        "title": "🛡️ استراتژی‌های هجینگ",
        "content": "استراتژی‌های محافظت از پرتفوی در برابر ریسک‌های بازار"
    },
    "capital_allocation": {
        "title": "💰 تخصیص سرمایه",
        "content": "محاسبه مبلغ دقیق سرمایه‌گذاری برای هر دارایی"
    },
    "monte_carlo_forecast": {
        "title": "🔮 پیش‌بینی مونت‌کارلو",
        "content": "شبیه‌سازی مسیرهای احتمالی قیمت آینده برای هر دارایی"
    },
    "cvar_opt": {
        "title": "📉 CVaR Optimization",
        "content": """
        **Conditional Value at Risk (CVaR)**
        
        میانگین زیان در بدترین α% حالات. مثلاً CVaR 95% میانگین زیان در 5% بدترین روزهاست.
        
        **مزایا:**
        - مدل‌سازی بهتر ریسک دم
        - مناسب برای توزیع‌های غیرنرمال
        - قابلیت محاسبه آسان
        
        **کاربرد:** مدیریت ریسک در شرایط بحرانی
        """
    },
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

def forecast_price_series(price_series, days=63, sims=500):
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_ret) < 2:
        mu, sigma = 0.0, 0.01
    else:
        mu, sigma = log_ret.mean(), log_ret.std()
    last_price = price_series.iloc[-1]
    paths = np.zeros((days, sims))
    for i in range(sims):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * np.exp(mu + sigma * np.random.normal()))
        paths[:, i] = prices[1:]
    return paths

# =============================================================================
# PORTFOLIO OPTIMIZATION STRATEGIES
# =============================================================================

class PortfolioOptimizer:
    """کلاس بهینه‌سازی پرتفوی با 3 استراتژی"""
    
    def __init__(self, returns: pd.DataFrame, rf_rate: float = 0.0):
        self.returns = returns
        self.mean_ret = returns.mean() * 252
        self.cov_mat = returns.cov() * 252
        self.n = len(self.mean_ret)
        self.rf_rate = rf_rate
        self.asset_names = list(returns.columns)
    
    def _get_bounds(self, allow_short: bool = False):
        if allow_short:
            return [(-1, 1)] * self.n
        return [(0, 1)] * self.n
    
    def _constraint_sum_to_one(self, w):
        return np.sum(w) - 1
    
    def portfolio_volatility(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_mat.values, weights)))
    
    def portfolio_return(self, weights):
        return np.dot(weights, self.mean_ret)
    
    def sharpe_ratio(self, weights):
        p_ret = self.portfolio_return(weights)
        p_vol = self.portfolio_volatility(weights)
        return (p_ret - self.rf_rate) / p_vol if p_vol > 0 else 0
    
    def cvar(self, weights, alpha=0.95):
        """Conditional Value at Risk"""
        p_returns = self.returns @ weights
        var = np.percentile(p_returns, (1 - alpha) * 100)
        cvar = -np.mean(p_returns[p_returns <= var]) * 252
        return cvar
    
    # ==================== STRATEGIES ====================
    
    def equal_weight(self):
        """وزن برابر"""
        return np.ones(self.n) / self.n
    
    def min_variance(self, allow_short=False):
        """حداقل واریانس - دقیق"""
        bounds = self._get_bounds(allow_short)
        constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
        x0 = np.ones(self.n) / self.n
        result = minimize(self.portfolio_volatility, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints, options={'ftol': 1e-9, 'maxiter': 1000})
        return result.x if result.success else self.equal_weight()
    
    def max_sharpe(self, allow_short=False):
        """حداکثر شارپ - پرتقولیو مارکوویتز"""
        bounds = self._get_bounds(allow_short)
        constraints = [
            {'type': 'eq', 'fun': self._constraint_sum_to_one},
            {'type': 'ineq', 'fun': lambda w: self.portfolio_return(w) - self.rf_rate}
        ]
        x0 = np.ones(self.n) / self.n
        result = minimize(lambda w: -self.sharpe_ratio(w), x0, method='SLSQP',
                         bounds=bounds, constraints=constraints, options={'ftol': 1e-9, 'maxiter': 1000})
        return result.x if result.success else self.equal_weight()
    
    def min_cvar(self, alpha=0.95, allow_short=False):
        """حداقل CVaR - دقیق"""
        bounds = self._get_bounds(allow_short)
        constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
        x0 = np.ones(self.n) / self.n
        result = minimize(lambda w: self.cvar(w, alpha), x0, method='SLSQP',
                         bounds=bounds, constraints=constraints, options={'ftol': 1e-9, 'maxiter': 1000})
        return result.x if result.success else self.equal_weight()
    
    def get_weights(self, strategy: str):
        """دریافت وزن‌ها بر اساس استراتژی"""
        strategies = {
            "حداکثر شارپ (مارکوویتز)": self.max_sharpe,
            "حداقل واریانس": self.min_variance,
            "حداقل CVaR": self.min_cvar,
        }
        
        if strategy in strategies:
            return strategies[strategy]()
        return self.equal_weight()


# =============================================================================
# STRATEGIES CONFIG
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

PORTFOLIO_STYLES = [
    "حداکثر شارپ (مارکوویتز)",
    "حداقل واریانس",
    "حداقل CVaR",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def capital_allocator_calculator(weights, asset_names, total_usd, exchange_rate):
    allocation_data = []
    for i, asset in enumerate(asset_names):
        weight = float(weights[i])
        amount_usd = weight * total_usd
        amount_toman = amount_usd * exchange_rate
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
# HELP BOX COMPONENT
# =============================================================================
def show_help(key):
    if key in HELP_TEXTS:
        help_data = HELP_TEXTS[key]
        with st.expander(f"❓ {help_data['title']}"):
            st.markdown(f"<div class='help-box'>{help_data['content']}</div>", unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================
st.set_page_config(
    page_title="Portfolio360 Ultimate Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<div class="main-header">
    <h1>📊 Portfolio360 Ultimate Pro</h1>
    <p>سیستم جامع تحلیل و مدیریت پرتفوی | 3 استراتژی بهینه‌سازی</p>
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
        help="نمادهای یاهو فایننس"
    )
    
    period = st.selectbox("بازه زمانی", ["1y", "2y", "5y", "10y", "max"], index=1)
    
    if st.button("🔄 دانلود / بروزرسانی", use_container_width=True):
        with st.spinner("در حال دانلود..."):
            data = download_data(tickers, period=period)
            if data is not None:
                st.session_state.prices = data
                st.success(f"✅ {len(data.columns)} دارایی بارگذاری شد!")
                st.rerun()
    
    show_help("data_download")
    st.markdown("---")
    
    st.markdown("### ⚙️ تنظیمات")
    
    if "rf_rate" not in st.session_state:
        st.session_state.rf_rate = 18.0
    st.session_state.rf_rate = st.number_input(
        "نرخ بدون ریسک (%)", 0.0, 50.0, st.session_state.rf_rate, 0.5
    )
    show_help("risk_free_rate")
    
    if "hedge_strategy" not in st.session_state:
        st.session_state.hedge_strategy = list(hedge_strategies.keys())[3]
    st.session_state.hedge_strategy = st.selectbox(
        "استراتژی هجینگ", list(hedge_strategies.keys()),
        index=list(hedge_strategies.keys()).index(st.session_state.hedge_strategy)
    )
    show_help("hedge_strategy")
    
    if "option_strategy" not in st.session_state:
        st.session_state.option_strategy = list(option_strategies.keys())[0]
    st.session_state.option_strategy = st.selectbox(
        "استراتژی آپشن", list(option_strategies.keys())
    )

# =============================================================================
# MAIN CONTENT
# =============================================================================
if "prices" not in st.session_state or st.session_state.prices is None:
    st.info("👈 لطفاً ابتدا از سایدبار داده‌ها را دانلود کنید.")
    
    st.markdown("""
    <div class="info-box">
        <h4>🚀 راهنمای شروع سریع</h4>
        <ol>
            <li>نمادهای مورد نظر را وارد کنید</li>
            <li>بازه زمانی مناسب را انتخاب کنید</li>
            <li>دکمه «دانلود / بروزرسانی» را بزنید</li>
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
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns, rf_rate=rf)
    
    # =============================================================================
    # PORTFOLIO CONFIGURATION
    # =============================================================================
    st.markdown('<div class="section-header">🎯 تنظیمات پرتفوی و تخصیص سرمایه</div>', unsafe_allow_html=True)
    
    colA, colB, colC = st.columns([2, 1, 1])
    
    with colA:
        if "selected_style" not in st.session_state:
            st.session_state.selected_style = PORTFOLIO_STYLES[0]
        
        st.session_state.selected_style = st.selectbox(
            "انتخاب سبک پرتفوی",
            PORTFOLIO_STYLES,
            index=PORTFOLIO_STYLES.index(st.session_state.selected_style)
        )
        
        # Show specific help for selected strategy
        if "CVaR" in st.session_state.selected_style:
            show_help("cvar_opt")
        else:
            show_help("portfolio_styles")
    
    with colB:
        capital_usd = st.number_input("کل سرمایه ($)", 1, 50_000_000, 1200, 100)
        exchange_rate = st.number_input("نرخ تبدیل (تومان/$)", 1000, 1_000_000_000, 200_000, 1000)
    
    with colC:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧮 محاسبه پرتفوی", use_container_width=True):
            weights = optimizer.get_weights(st.session_state.selected_style)
            st.session_state.weights = weights
            st.session_state.last_capital_usd = capital_usd
            st.success("✅ وزن‌ها محاسبه شدند!")
    
    if "weights" not in st.session_state:
        st.session_state.weights = optimizer.equal_weight()
    
    weights = st.session_state.weights
    
    # Display portfolio metrics
    p_ret = optimizer.portfolio_return(weights)
    p_vol = optimizer.portfolio_volatility(weights)
    p_sharpe = (p_ret - rf) / p_vol if p_vol > 0 else 0
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("بازده سالانه", f"{p_ret*100:.2f}%")
    with col_m2:
        st.metric("نوسان سالانه", f"{p_vol*100:.2f}%")
    with col_m3:
        st.metric("نسبت شارپ", f"{p_sharpe:.3f}")
    with col_m4:
        st.metric("تعداد دارایی‌ها", len(asset_names))
    
    # Display weights
    col_w1, col_w2 = st.columns([1, 1])
    
    with col_w1:
        df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(weights*100, 2)})
        st.dataframe(df_w, use_container_width=True, hide_index=True)
    
    with col_w2:
        fig_pie = px.pie(df_w, values="وزن (%)", names="دارایی", title="توزیع پرتفوی",
                        color_discrete_sequence=px.colors.sequential.Viridis)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Capital allocation
    st.markdown("### 💰 تخصیص سرمایه")
    show_help("capital_allocation")
    
    alloc_df = capital_allocator_calculator(weights, asset_names, capital_usd, exchange_rate)
    st.dataframe(alloc_df[["دارایی", "درصد وزن", "دلار ($)", "تومان", "ریال"]],
                use_container_width=True, hide_index=True)
    
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        st.download_button("📥 دانلود CSV",
            alloc_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            use_container_width=True)
    
    st.markdown("---")
    
    # =============================================================================
    # MONTE CARLO FORECAST
    # =============================================================================
    st.markdown('<div class="section-header">🔮 پیش‌بینی قیمت (Monte Carlo)</div>', unsafe_allow_html=True)
    show_help("monte_carlo_forecast")
    
    col_mc1, col_mc2, col_mc3 = st.columns([2, 1, 1])
    
    with col_mc1:
        sel_asset = st.selectbox("دارایی", asset_names)
    with col_mc2:
        days_forecast = st.slider("روزها", 30, 365, 90)
    with col_mc3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_forecast = st.button("🚀 اجرا", use_container_width=True)
    
    if run_forecast:
        with st.spinner("در حال شبیه‌سازی..."):
            series = prices[sel_asset]
            paths = forecast_price_series(series, days=days_forecast, sims=400)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series.values,
                        name="واقعی", line=dict(color="#1f77b4", width=2)))
            
            future_x = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=days_forecast)
            median = np.percentile(paths, 50, axis=1)
            p10, p90 = np.percentile(paths, 10, axis=1), np.percentile(paths, 90, axis=1)
            
            fig.add_trace(go.Scatter(x=future_x, y=median, name="میانه",
                        line=dict(color="orange", width=2)))
            fig.add_trace(go.Scatter(x=future_x, y=p90, line=dict(color="rgba(255,165,0,0.3)"),
                        showlegend=False))
            fig.add_trace(go.Scatter(x=future_x, y=p10, fill='tonexty',
                        fillcolor='rgba(255,165,0,0.1)', showlegend=False))
            
            fig.update_layout(title=f"پیش‌بینی {sel_asset} - {days_forecast} روز",
                            xaxis_title="تاریخ", yaxis_title="قیمت ($)",
                            template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            current_price = series.iloc[-1]
            predicted_price = median[-1]
            change_pct = ((predicted_price - current_price) / current_price) * 100
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("قیمت فعلی", f"${current_price:,.2f}")
            with col_m2:
                st.metric("پیش‌بینی", f"${predicted_price:,.2f}", f"{change_pct:+.2f}%")
            with col_m3:
                st.metric("دامنه 80%", f"${p10[-1]:,.0f} - ${p90[-1]:,.0f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>📊 <strong>Mr.option11</strong> — 3 استراتژی بهینه‌سازی</p>
</div>
""", unsafe_allow_html=True)
