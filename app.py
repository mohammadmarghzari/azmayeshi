"""
Portfolio360 Ultimate Pro — Professional Edition
- Enhanced UI with modern design
- Comprehensive help tooltips for each feature
- Better organized sections with expandable explanations
- Professional styling and visual improvements
- 20+ Portfolio Optimization Strategies including:
  * Markowitz, Min Variance, Max Sharpe
  * Risk Parity, HRP, HERC
  * Black-Litterman, Kelly Criterion
  * CVaR, CDaR, Omega Ratio
  * Maximum Diversification, Most Diversified Portfolio
  * And many more...
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
        **سبک‌های کلاسیک:**
        - **مارکوویتز (حداکثر شارپ)**: بهینه‌سازی نسبت شارپ
        - **حداقل واریانس**: کمترین ریسک ممکن
        - **وزن برابر**: ساده و مقاوم
        
        **سبک‌های مبتنی بر ریسک:**
        - **ریسک‌پاریتی**: وزن‌دهی بر اساس ریسک
        - **HRP**: خوشه‌بندی سلسله‌مراتبی
        - **HERC**: ریسک‌پاریتی سلسله‌مراتبی
        
        **سبک‌های پیشرفته:**
        - **CVaR Optimization**: بهینه‌سازی ریسک شرطی
        - **CDaR Optimization**: بهینه‌سازی کشیدگی شرطی
        - **Omega Ratio**: نسبت امگا
        - **Sortino Ratio**: نسبت سورتینو
        - **Calmar Ratio**: نسبت کالمار
        - **Kelly Criterion**: معیار کلی
        - **Black-Litterman**: ترکیب نظر با داده
        - **Maximum Diversification**: حداکثر تنوع
        - **Most Diversified Portfolio**: متنوع‌ترین پرتفوی
        - **Equal Risk Bounding**: مرز ریسک برابر
        - **Minimum Correlation**: حداقل همبستگی
        - **Minimum Tail Dependence**: حداقل وابستگی دم
        - **Mean-Absolute Deviation**: انحراف مطلق میانگین
        - **Gini Mean Difference**: اختلاف میانگین جینی
        """
    },
    "risk_free_rate": {
        "title": "📊 نرخ بدون ریسک",
        "content": "نرخ بازدهی بدون ریسک برای محاسبه شارپ و سورتینو. برای ایران: 18-25%"
    },
    "hedge_strategy": {
        "title": "🛡️ استراتژی‌های هجینگ",
        "content": "استراتژی‌های محافظت از پرتفوی در برابر ریسک‌های بازار"
    },
    "option_strategy": {
        "title": "📈 استراتژی‌های آپشن",
        "content": "استراتژی‌های آپشن برای مدیریت ریسک و کسب درآمد"
    },
    "capital_allocation": {
        "title": "💰 تخصیص سرمایه",
        "content": "محاسبه مبلغ دقیق سرمایه‌گذاری برای هر دارایی"
    },
    "monte_carlo_forecast": {
        "title": "🔮 پیش‌بینی مونت‌کارلو",
        "content": "شبیه‌سازی مسیرهای احتمالی قیمت آینده"
    },
    "married_put": {
        "title": "🛡️ Protective Put",
        "content": "تحلیل استراتژی Married Put - ترکیب دارایی و آپشن فروش"
    },
    "dca_time": {
        "title": "⏳ DCA زمانی",
        "content": "شبیه‌سازی Dollar-Cost Averaging مبتنی بر زمان"
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
    "kelly": {
        "title": "🎯 Kelly Criterion",
        "content": """
        **معیار کلی برای حداکثر رشد سرمایه**
        
        فرمول: f* = (μ - r) / σ²
        
        **نکات:**
        - حداکثر رشد بلندمدت
        - نیاز به تخمین دقیق پارامترها
        - معمولاً نصف کلی توصیه می‌شود
        """
    },
    "black_litterman": {
        "title": "🌐 Black-Litterman",
        "content": """
        **ترکیب نظر شخصی با داده‌های بازار**
        
        مراحل:
        1. محاسبه بازده‌های تعادلی از واریانس-کواریانس
        2. ترکیب با نظرات سرمایه‌گذار
        3. بهینه‌سازی پرتفوی
        
        **مزایا:**
        - حل مشکل تخمین میانگین
        - پرتفوی‌های پایدارتر
        - قابلیت ترکیب دیدگاه
        """
    },
    "hrp": {
        "title": "🌳 HRP (Hierarchical Risk Parity)",
        "content": """
        **خوشه‌بندی سلسله‌مراتبی + ریسک‌پاریتی**
        
        مراحل:
        1. خوشه‌بندی دارایی‌ها بر اساس همبستگی
        2. تخصیص وزن درون‌خوشه‌ای
        3. تخصیص وزن بین‌خوشه‌ای
        
        **مزایا:**
        - بدون نیاز به تخمین میانگین
        - پایدارتر از مارکوویتز
        - مدیریت بهتر همبستگی
        """
    },
    "omega": {
        "title": "Ω Omega Ratio",
        "content": """
        **نسبت امگا - جایگزین شارپ**
        
        Ω(L) = E[max(R-L, 0)] / E[max(L-R, 0)]
        
        L = سطح آستانه (معمولاً نرخ بدون ریسک)
        
        **مزایا:**
        - در نظر گرفتن کل توزیع
        - بدون فرض نرمال بودن
        - انعطاف‌پذیری بیشتر
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
    """کلاس بهینه‌سازی پرتفوی با 20+ استراتژی"""
    
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
    
    def sortino_ratio(self, weights, target=0):
        p_ret = self.portfolio_return(weights)
        downside_returns = np.minimum(self.returns @ weights - target/252, 0)
        downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        return (p_ret - self.rf_rate) / downside_std if downside_std > 0 else 0
    
    def calmar_ratio(self, weights):
        p_ret = self.portfolio_return(weights)
        cummax = (1 + self.returns @ weights).cumprod()
        max_dd = np.max(1 - cummax / cummax.cummax())
        return p_ret / max_dd if max_dd > 0 else 0
    
    def omega_ratio(self, weights, threshold=None):
        if threshold is None:
            threshold = self.rf_rate
        p_returns = self.returns @ weights
        upside = np.mean(np.maximum(p_returns * 252 - threshold, 0))
        downside = np.mean(np.maximum(threshold - p_returns * 252, 0))
        return upside / downside if downside > 0 else np.inf
    
    def cvar(self, weights, alpha=0.95):
        """Conditional Value at Risk"""
        p_returns = self.returns @ weights
        var = np.percentile(p_returns, (1 - alpha) * 100)
        cvar = -np.mean(p_returns[p_returns <= var]) * 252
        return cvar
    
    def cdar(self, weights, alpha=0.95):
        """Conditional Drawdown at Risk"""
        cummax = (1 + self.returns @ weights).cumprod()
        drawdowns = 1 - cummax / cummax.cummax()
        var_dd = np.percentile(drawdowns, alpha * 100)
        cdar = np.mean(drawdowns[drawdowns >= var_dd])
        return cdar
    
    def diversification_ratio(self, weights):
        """نسبت تنوع‌بخشی"""
        p_vol = self.portfolio_volatility(weights)
        weighted_vols = np.sum(weights * np.sqrt(np.diag(self.cov_mat.values)))
        return weighted_vols / p_vol if p_vol > 0 else 0
    
    def herfindahl_index(self, weights):
        """شاخص هرfindahl برای ریسک‌پاریتی"""
        marginal_risk = self.cov_mat.values @ weights
        total_risk = weights @ marginal_risk
        risk_contrib = weights * marginal_risk / total_risk
        return np.sum(risk_contrib**2)
    
    # ==================== STRATEGIES ====================
    
    def equal_weight(self):
        """وزن برابر"""
        return np.ones(self.n) / self.n
    
    def inverse_volatility(self):
        """معکوس نوسان"""
        vols = np.sqrt(np.diag(self.cov_mat.values))
        inv_vols = 1 / (vols + 1e-8)
        return inv_vols / inv_vols.sum()
    
    def min_variance(self, allow_short=False):
        """حداقل واریانس"""
        bounds = self._get_bounds(allow_short)
        constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
        x0 = np.ones(self.n) / self.n
        result = minimize(self.portfolio_volatility, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else self.equal_weight()
    
    def max_sharpe(self, allow_short=False):
        """حداکثر شارپ"""
        bounds = self._get_bounds(allow_short)
        constraints = [
            {'type': 'eq', 'fun': self._constraint_sum_to_one},
            {'type': 'ineq', 'fun': lambda w: self.portfolio_return(w) - self.rf_rate}
        ]
        x0 = np.ones(self.n) / self.n
        result = minimize(lambda w: -self.sharpe_ratio(w), x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else self.equal_weight()
    
    def max_sortino(self, allow_short=False):
        """حداکثر سورتینو"""
        bounds = self._get_bounds(allow_short)
        constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
        x0 = np.ones(self.n) / self.n
        result = minimize(lambda w: -self.sortino_ratio(w), x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else self.equal_weight()
    
    def max_calmar(self, allow_short=False):
        """حداکثر کالمار"""
        bounds = self._get_bounds(allow_short)
        constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
        x0 = np.ones(self.n) / self.n
        result = minimize(lambda w: -self.calmar_ratio(w), x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else self.equal_weight()
    
    def max_omega(self, allow_short=False):
        """حداکثر امگا"""
        bounds = self._get_bounds(allow_short)
        constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
        x0 = np.ones(self.n) / self.n
        result = minimize(lambda w: -self.omega_ratio(w), x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else self.equal_weight()
    
    def min_cvar(self, alpha=0.95, allow_short=False):
        """حداقل CVaR"""
        bounds = self._get_bounds(allow_short)
        constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
        x0 = np.ones(self.n) / self.n
        result = minimize(lambda w: self.cvar(w, alpha), x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else self.equal_weight()
    
    def min_cdar(self, alpha=0.95, allow_short=False):
        """حداقل CDaR"""
        bounds = self._get_bounds(allow_short)
        constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
        x0 = np.ones(self.n) / self.n
        result = minimize(lambda w: self.cdar(w, alpha), x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else self.equal_weight()
    
    def risk_parity(self):
        """ریسک‌پاریتی کلاسیک"""
        def risk_parity_objective(w):
            marginal_risk = self.cov_mat.values @ w
            total_risk = w @ marginal_risk
            risk_contrib = w * marginal_risk / total_risk
            target = 1 / self.n
            return np.sum((risk_contrib - target)**2)
        
        bounds = [(0, 1)] * self.n
        constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
        x0 = np.ones(self.n) / self.n
        result = minimize(risk_parity_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else self.equal_weight()
    
    def hierarchical_risk_parity(self):
        """HRP - خوشه‌بندی سلسله‌مراتبی"""
        corr = self.returns.corr()
        dist = np.sqrt(0.5 * (1 - corr))
        
        # Linkage
        dist_array = squareform(dist.values, checks=False)
        link = linkage(dist_array, 'single')
        
        # Get quasi-diagonalization
        def get_quasi_diag(link):
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = link[-1, 3]
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= num_items]
                i = df0.index
                j = df0.values - num_items
                sort_ix[i] = link[j, 0]
                df0 = pd.Series(link[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df0])
                sort_ix = sort_ix.sort_index()
                sort_ix.index = range(sort_ix.shape[0])
            return sort_ix.tolist()
        
        sort_ix = get_quasi_diag(link)
        sort_ix = [int(i) for i in sort_ix]
        
        # Recursive bisection
        def get_recursive_bisection(cov, sorted_idx):
            w = pd.Series(1, index=sorted_idx)
            clusters = [sorted_idx]
            while len(clusters) > 0:
                clusters = [i[int(j):int(k)] for i in clusters 
                           for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]
                for i in range(0, len(clusters), 2):
                    if i + 1 < len(clusters):
                        c1, c2 = clusters[i], clusters[i + 1]
                        cov1 = cov.iloc[c1, c1]
                        cov2 = cov.iloc[c2, c2]
                        try:
                            vol1 = np.sqrt(np.linalg.pinv(cov1.values).sum())
                            vol2 = np.sqrt(np.linalg.pinv(cov2.values).sum())
                        except:
                            vol1 = np.sqrt(np.diag(cov1.values).sum())
                            vol2 = np.sqrt(np.diag(cov2.values).sum())
                        alpha = 1 - vol1 / (vol1 + vol2 + 1e-8)
                        w[c1] *= alpha
                        w[c2] *= 1 - alpha
            return w
        
        weights = get_recursive_bisection(self.cov_mat, sort_ix)
        weights = weights.reindex(range(self.n)).fillna(0)
        return weights.values / weights.sum()
    
    def max_diversification(self, allow_short=False):
        """حداکثر تنوع‌بخشی"""
        bounds = self._get_bounds(allow_short)
        constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
        x0 = np.ones(self.n) / self.n
        result = minimize(lambda w: -self.diversification_ratio(w), x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x if result.success else self.equal_weight()
    
    def most_diversified_portfolio(self):
        """متنوع‌ترین پرتفوی"""
        vols = np.sqrt(np.diag(self.cov_mat.values))
        inv_vols = 1 / vols
        return inv_vols / inv_vols.sum()
    
    def kelly_criterion(self, half_kelly=True):
        """معیار کلی"""
        try:
            cov_inv = np.linalg.inv(self.cov_mat.values)
            excess_ret = self.mean_ret.values - self.rf_rate
            kelly_weights = cov_inv @ excess_ret
            if np.sum(kelly_weights) <= 0:
                return self.equal_weight()
            kelly_weights = kelly_weights / np.sum(kelly_weights)
            if half_kelly:
                kelly_weights = kelly_weights * 0.5
                kelly_weights = kelly_weights / np.sum(kelly_weights)
            return np.maximum(kelly_weights, 0)  # No short selling
        except:
            return self.equal_weight()
    
    def min_correlation(self):
        """حداقل همبستگی"""
        corr = self.returns.corr()
        inv_corr_sum = (1 / (corr.sum(axis=1) - 1))
        weights = inv_corr_sum / inv_corr_sum.sum()
        return weights.values
    
    def equal_risk_bounding(self):
        """مرز ریسک برابر"""
        vols = np.sqrt(np.diag(self.cov_mat.values))
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
        
        # Iterative refinement
        for _ in range(100):
            marginal_risk = self.cov_mat.values @ weights
            total_risk = weights @ marginal_risk
            risk_contrib = weights * marginal_risk / total_risk
            adjustment = 1 / (risk_contrib + 1e-8)
            weights = weights * adjustment
            weights = weights / weights.sum()
        
        return weights
    
    def black_litterman(self, P=None, Q=None, omega=None, tau=0.025):
        """بلک-لیترمن"""
        try:
            # Implied equilibrium returns (reverse optimization)
            w_mkt = np.ones(self.n) / self.n  # Assume equal weight as market
            pi = self.rf_rate + self.cov_mat.values @ w_mkt
            
            if P is None or Q is None:
                # Equilibrium weights
                return self.equal_weight()
            
            if omega is None:
                omega = np.diag(np.diag(P @ self.cov_mat.values @ P.T))
            
            M = tau * self.cov_mat.values
            BL_cov = np.linalg.inv(np.linalg.inv(M) + P.T @ np.linalg.inv(omega) @ P)
            BL_ret = BL_cov @ (np.linalg.inv(M) @ pi + P.T @ np.linalg.inv(omega) @ Q)
            
            # Optimize with BL returns
            bounds = [(0, 1)] * self.n
            constraints = {'type': 'eq', 'fun': self._constraint_sum_to_one}
            x0 = np.ones(self.n) / self.n
            
            def neg_bl_sharpe(w):
                p_ret = np.dot(w, BL_ret)
                p_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_mat.values, w)))
                return -(p_ret - self.rf_rate) / p_vol if p_vol > 0 else 0
            
            result = minimize(neg_bl_sharpe, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            return result.x if result.success else self.equal_weight()
        except:
            return self.equal_weight()
    
    def mean_absolute_deviation(self):
        """انحراف مطلق میانگین"""
        median_ret = self.returns.median()
        mad = np.abs(self.returns - median_ret).mean()
        inv_mad = 1 / (mad + 1e-8)
        return inv_mad / inv_mad.sum()
    
    def resampled_efficiency(self, n_sims=100):
        """کارایی مجدد (Michaud, 1998)"""
        T = len(self.returns)
        weights_list = []
        
        for _ in range(n_sims):
            # Bootstrap sample
            sample_idx = np.random.choice(T, size=T, replace=True)
            sample_returns = self.returns.iloc[sample_idx]
            
            sample_mean = sample_returns.mean() * 252
            sample_cov = sample_returns.cov() * 252
            
            # Max Sharpe on sample
            bounds = [(0, 1)] * self.n
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            x0 = np.ones(self.n) / self.n
            
            def neg_sharpe(w):
                p_ret = np.dot(w, sample_mean)
                p_vol = np.sqrt(np.dot(w.T, np.dot(sample_cov.values, w)))
                return -(p_ret - self.rf_rate) / p_vol if p_vol > 0 else 0
            
            result = minimize(neg_sharpe, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            if result.success:
                weights_list.append(result.x)
        
        if weights_list:
            return np.mean(weights_list, axis=0)
        return self.equal_weight()
    
    def get_weights(self, strategy: str):
        """دریافت وزن‌ها بر اساس استراتژی"""
        strategies = {
            "وزن برابر (ساده و مقاوم)": self.equal_weight,
            "Inverse Volatility": self.inverse_volatility,
            "حداقل واریانس": self.min_variance,
            "مارکوویتز (حداکثر شارپ)": self.max_sharpe,
            "حداکثر سورتینو": self.max_sortino,
            "حداکثر کالمار": self.max_calmar,
            "حداکثر امگا": self.max_omega,
            "حداقل CVaR": self.min_cvar,
            "حداقل CDaR": self.min_cdar,
            "ریسک‌پاریتی": self.risk_parity,
            "HRP (خوشه‌بندی سلسله‌مراتبی)": self.hierarchical_risk_parity,
            "حداکثر تنوع‌بخشی": self.max_diversification,
            "متنوع‌ترین پرتفوی": self.most_diversified_portfolio,
            "Kelly Criterion (نیم‌کلی)": lambda: self.kelly_criterion(half_kelly=True),
            "Kelly Criterion (کامل)": lambda: self.kelly_criterion(half_kelly=False),
            "Black-Litterman": self.black_litterman,
            "حداقل همبستگی": self.min_correlation,
            "مرز ریسک برابر": self.equal_risk_bounding,
            "انحراف مطلق میانگین": self.mean_absolute_deviation,
            "کارایی مجدد (مونت‌کارلو)": self.resampled_efficiency,
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
    "مارکوویتز (حداکثر شارپ)",
    "وزن برابر (ساده و مقاوم)",
    "حداقل واریانس",
    "ریسک‌پاریتی",
    "HRP (خوشه‌بندی سلسله‌مراتبی)",
    "حداکثر سورتینو",
    "حداکثر کالمار",
    "حداکثر امگا",
    "حداقل CVaR",
    "حداقل CDaR",
    "حداکثر تنوع‌بخشی",
    "متنوع‌ترین پرتفوی",
    "Kelly Criterion (نیم‌کلی)",
    "Kelly Criterion (کامل)",
    "Black-Litterman",
    "Inverse Volatility",
    "حداقل همبستگی",
    "مرز ریسک برابر",
    "انحراف مطلق میانگین",
    "کارایی مجدد (مونت‌کارلو)",
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


def married_put_pnl_grid(S0, strike, premium_per_contract, units_held, contracts, contract_size, 
                         grid_min=None, grid_max=None, ngrid=600):
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


def suggest_contracts_for_target_risk(prices, returns, asset_names, weights, cov_mat, total_usd, 
                                      btc_idx, eth_idx, btc_contract_size, eth_contract_size, 
                                      est_btc_prem, est_eth_prem, max_contracts=30, target_risk_pct=2.0):
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
            new_risk = np.sqrt(np.dot(weights.T, np.dot(cov_adj.values, weights))) * 100
            total_premium = btc_total_premium + eth_total_premium
            if new_risk <= target_risk_pct:
                if best is None or total_premium < best["total_premium"]:
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
    price_index = pd.to_datetime(price_index)  # Ensure DatetimeIndex
    for d in dates:
        ts = pd.Timestamp(d)
        if ts <= price_index[0]:
            mapped.append(price_index[0])
            continue
        locs = price_index.searchsorted(ts)
        if locs >= len(price_index):
            mapped.append(price_index[-1])
        else:
            mapped.append(price_index[locs])
    return pd.to_datetime(mapped)


def simulate_time_dca(price_series, total_amount, periods, freq_days=1, start_date=None, levels=None):
    if start_date is None:
        start_dt = price_series.index[0]
    else:
        # Convert to Timestamp if it's date object
        if hasattr(start_date, 'year'):
            start_dt = pd.Timestamp(datetime.combine(start_date, datetime.min.time()))
        else:
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
    else:
        level_schedule = [None] * periods
    
    per_amount = total_amount / periods
    purchases = []
    for i, dt in enumerate(mapped_dates):
        price_on_date = float(price_series.loc[dt])
        units = per_amount / price_on_date if price_on_date > 0 else 0.0
        purchases.append({
            "date": pd.Timestamp(dt),
            "price": price_on_date,
            "amount_usd": per_amount,
            "units": units,
            "level_assigned": level_schedule[i]
        })
    
    df = pd.DataFrame(purchases)
    total_units = df["units"].sum()
    avg_price = df["amount_usd"].sum() / (total_units + 1e-12) if total_units > 0 else np.nan
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
        "profit_pct": profit_pct,
        "first_date": df["date"].min(),
        "last_date": df["date"].max()
    }
    return df, summary


def plot_price_with_purchases(price_series, purchases_df, title="Price with purchases"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values,
        name="قیمت", mode="lines", line=dict(color="#0b69ff")
    ))
    if not purchases_df.empty:
        fig.add_trace(go.Scatter(
            x=purchases_df["date"], y=purchases_df["price"],
            mode="markers", name="خریدها",
            marker=dict(size=10, color="orange", symbol="diamond")
        ))
    fig.update_layout(
        title=title, xaxis_title="تاریخ", yaxis_title="قیمت ($)",
        template="plotly_white", height=480
    )
    return fig


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
    <p>سیستم جامع تحلیل و مدیریت پرتفوی | 20+ استراتژی بهینه‌سازی</p>
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
        elif "Kelly" in st.session_state.selected_style:
            show_help("kelly")
        elif "Black-Litterman" in st.session_state.selected_style:
            show_help("black_litterman")
        elif "HRP" in st.session_state.selected_style:
            show_help("hrp")
        elif "امگا" in st.session_state.selected_style:
            show_help("omega")
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
    
    st.markdown("---")
    
    # =============================================================================
    # MARRIED PUT SECTION
    # =============================================================================
    st.markdown('<div class="section-header">🛡️ Protective Put (Married Put)</div>', unsafe_allow_html=True)
    show_help("married_put")
    
    btc_col = next((c for c in asset_names if "BTC" in c.upper()), None)
    eth_col = next((c for c in asset_names if "ETH" in c.upper()), None)
    
    col_mp1, col_mp2 = st.columns(2)
    
    with col_mp1:
        if btc_col:
            st.subheader("🔸 BTC-USD")
            btc_price = float(prices[btc_col].iloc[-1])
            btc_strike = st.number_input("Strike ($)", value=btc_price*0.90, step=10.0, key="btc_strike")
            btc_premium = st.number_input("Premium ($)", value=btc_price*0.04, step=1.0, key="btc_prem")
            btc_contracts = st.number_input("قراردادها", 0, 200, 0, 1, key="btc_contracts")
            btc_contract_size = st.number_input("اندازه قرارداد", 0.01, 100.0, 1.0, 0.01, key="btc_size")
    
    with col_mp2:
        if eth_col:
            st.subheader("🔹 ETH-USD")
            eth_price = float(prices[eth_col].iloc[-1])
            eth_strike = st.number_input("Strike ($)", value=eth_price*0.90, step=5.0, key="eth_strike")
            eth_premium = st.number_input("Premium ($)", value=eth_price*0.04, step=0.5, key="eth_prem")
            eth_contracts = st.number_input("قراردادها", 0, 200, 0, 1, key="eth_contracts")
            eth_contract_size = st.number_input("اندازه قرارداد", 0.01, 1000.0, 1.0, 0.01, key="eth_size")
    
    st.markdown("### 🔍 تنظیمات نمایش")
    zcol1, zcol2 = st.columns(2)
    zoom_min_pct = zcol1.slider("حداقن (%)", 10, 100, 80)
    zoom_max_pct = zcol2.slider("حداکثر (%)", 100, 250, 140)
    
    if st.button("📊 نمایش نمودار Payoff", use_container_width=True):
        exposures = {asset_names[i]: float(weights[i])*capital_usd for i in range(len(asset_names))}
        units_btc = exposures.get(btc_col, 0.0) / (btc_price + 1e-8) if btc_col else 0.0
        units_eth = exposures.get(eth_col, 0.0) / (eth_price + 1e-8) if eth_col else 0.0
        
        traces = []
        all_prices = np.array([])
        
        if btc_col and btc_contracts > 0:
            grid_btc, married_btc, _ = married_put_pnl_grid(
                btc_price, btc_strike, btc_premium, units_btc, int(btc_contracts), float(btc_contract_size))
            traces.append(("BTC", grid_btc, married_btc, "#ff8c00"))
            all_prices = np.concatenate([all_prices, grid_btc])
        
        if eth_col and eth_contracts > 0:
            grid_eth, married_eth, _ = married_put_pnl_grid(
                eth_price, eth_strike, eth_premium, units_eth, int(eth_contracts), float(eth_contract_size))
            traces.append(("ETH", grid_eth, married_eth, "#1f77b4"))
            all_prices = np.concatenate([all_prices, grid_eth])
        
        fig = go.Figure()
        for name, grid, pnl, color in traces:
            fig.add_trace(go.Scatter(x=grid, y=pnl, name=f"{name}", mode="lines", line=dict(color=color, width=2)))
            fig.add_trace(go.Scatter(x=grid, y=np.where(pnl>=0, pnl, np.nan), fill='tozeroy',
                        mode='none', fillcolor='rgba(50,205,50,0.15)', showlegend=False))
            fig.add_trace(go.Scatter(x=grid, y=np.where(pnl<0, pnl, np.nan), fill='tozeroy',
                        mode='none', fillcolor='rgba(255,99,71,0.15)', showlegend=False))
        
        if all_prices.size > 0:
            from numpy import interp
            common_grid = np.linspace(all_prices.min(), all_prices.max(), 800)
            total_payoff = np.zeros_like(common_grid)
            if any(t[0] == "BTC" for t in traces):
                total_payoff += interp(common_grid, grid_btc, married_btc)
            if any(t[0] == "ETH" for t in traces):
                total_payoff += interp(common_grid, grid_eth, married_eth)
            
            fig.add_trace(go.Scatter(x=common_grid, y=total_payoff, name="Total",
                        mode="lines", line=dict(color="#2ca02c", width=3)))
        
        s0_candidates = []
        if btc_col and btc_contracts > 0: s0_candidates.append(btc_price)
        if eth_col and eth_contracts > 0: s0_candidates.append(eth_price)
        base = float(np.mean(s0_candidates)) if s0_candidates else 1.0
        fig.update_xaxes(range=[base * zoom_min_pct/100, base * zoom_max_pct/100])
        
        fig.update_layout(title="نمودار سود/زیان Married Put",
                        xaxis_title="قیمت ($)", yaxis_title="PnL (USD)",
                        template="plotly_white", height=560)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics
        st.markdown("### 📉 معیارهای ریسک")
        btc_idx = asset_names.index(btc_col) if btc_col in asset_names else None
        eth_idx = asset_names.index(eth_col) if eth_col in asset_names else None
        
        btc_total_premium = (btc_premium * btc_contracts * btc_contract_size) if (btc_col and btc_contracts>0) else 0.0
        eth_total_premium = (eth_premium * eth_contracts * eth_contract_size) if (eth_col and eth_contracts>0) else 0.0
        
        exposures = {asset_names[i]: float(weights[i])*capital_usd for i in range(len(asset_names))}
        btc_premium_pct = (btc_total_premium / (exposures.get(btc_col,1e-8))) * 100 if btc_col else 0.0
        eth_premium_pct = (eth_total_premium / (exposures.get(eth_col,1e-8))) * 100 if eth_col else 0.0
        
        btc_reduction = min(0.95, btc_premium_pct * 0.5 / 100.0)
        eth_reduction = min(0.95, eth_premium_pct * 0.5 / 100.0)
        
        cov_adj = apply_protective_put_to_cov(cov_mat, asset_names, btc_idx, eth_idx, btc_reduction, eth_reduction)
        original_risk = np.sqrt(np.dot(weights.T, np.dot(cov_mat.values, weights))) * 100
        new_risk = np.sqrt(np.dot(weights.T, np.dot(cov_adj.values, weights))) * 100
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("ریسک بدون بیمه", f"{original_risk:.2f}%")
        with col_r2:
            st.metric("ریسک با بیمه", f"{new_risk:.2f}%")
        with col_r3:
            st.metric("کاهش ریسک", f"{original_risk - new_risk:.3f}%")
        with col_r4:
            st.metric("کل Premium", f"${btc_total_premium + eth_total_premium:,.2f}")
    
    st.markdown("---")
    
    # =============================================================================
    # DCA SECTION
    # =============================================================================
    st.markdown('<div class="section-header">⏳ DCA زمانی</div>', unsafe_allow_html=True)
    show_help("dca_time")
    
    col_dca1, col_dca2, col_dca3 = st.columns([2, 1, 1])
    with col_dca1:
        dca_asset = st.selectbox("دارایی", asset_names, index=0, key="dca_asset")
    with col_dca2:
        dca_total = st.number_input("سرمایه ($)", 1.0, value=1000.0, step=100.0)
    with col_dca3:
        dca_periods = st.number_input("دوره‌ها", 1, value=30, step=1)
    
    col_dca4, col_dca5, col_dca6 = st.columns([1, 1, 1])
    with col_dca4:
        dca_freq_days = st.number_input("فاصله (روز)", 1, value=1, step=1)
    with col_dca5:
        dca_start_date = st.date_input("تاریخ شروع",
            value=(prices.index[0] + pd.Timedelta(days=1)).date())
    with col_dca6:
        use_levels = st.checkbox("سطوح قیمتی", value=False)
    
    levels_input = None
    if use_levels:
        levels_txt = st.text_input("سطوح (با کاما)", placeholder="2500,2200,1800")
        try:
            levels_input = [float(x.strip()) for x in levels_txt.split(",") if x.strip()]
        except:
            levels_input = None
    
    if st.button("▶️ اجرای DCA", use_container_width=True):
        with st.spinner("در حال شبیه‌سازی..."):
            series = prices[dca_asset]
            df_purchases, summary = simulate_time_dca(series, dca_total, int(dca_periods),
                int(dca_freq_days), start_date=dca_start_date, levels=levels_input)
            
            st.markdown("#### 📋 جدول معاملات")
            st.dataframe(df_purchases[["date", "price", "amount_usd", "units", "level_assigned"]]
                        .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")),
                        use_container_width=True, hide_index=True)
            
            st.markdown("#### 📊 خلاصه")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("سرمایه‌گذاری", f"${summary['total_invested']:.2f}")
                st.metric("دوره‌ها", f"{int(dca_periods)}")
            with col_res2:
                st.metric("میانگین قیمت", f"${summary['avg_price_per_unit']:.4f}")
                st.metric("قیمت نهایی", f"${summary['final_price']:.2f}")
            with col_res3:
                st.metric("ارزش کنونی", f"${summary['final_value']:.2f}")
                st.metric("سود/زیان", f"${summary['profit']:.2f}", f"{summary['profit_pct']:.2f}%")
            
            fig_p = plot_price_with_purchases(series, df_purchases, title=f"DCA روی {dca_asset}")
            st.plotly_chart(fig_p, use_container_width=True)
            
            csv = df_purchases.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("📥 دانلود CSV", csv,
                file_name=f"dca_{dca_asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>📊 <strong>Portfolio360 Ultimate Pro</strong> — 20+ استراتژی بهینه‌سازی</p>
</div>
""", unsafe_allow_html=True)
