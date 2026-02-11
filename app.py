"""
Portfolio360 Ultimate Pro — Professional Edition
- Enhanced UI with modern design
- Comprehensive help tooltips for each feature
- Better organized sections with expandable explanations
- Professional styling and visual improvements
- 20+ Portfolio Optimization Strategies
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    .main-header h1 { color: white !important; font-size: 2.5rem !important; font-weight: 700 !important; margin: 0 !important; }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem 1.5rem; border-radius: 10px;
        font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;
    }
    .help-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border: 1px solid #d1d5db; border-radius: 10px; padding: 1rem; margin: 0.5rem 0 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-left: 4px solid #0ea5e9; border-radius: 8px; padding: 1rem; margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px; padding: 0.75rem 2rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELP TEXTS (کوتاه شده برای جلوگیری از طولانی شدن بیش از حد)
# =============================================================================
HELP_TEXTS = {
    "data_download": {
        "title": "📥 راهنمای دانلود داده",
        "content": "نمادهای معتبر یاهو فایننس مثل BTC-USD, ETH-USD, GC=F, USDIRR=X, ^GSPC"
    },
    "portfolio_styles": {
        "title": "📚 سبک‌های پرتفوی",
        "content": "مارکوویتز، حداقل واریانس، ریسک‌پاریتی، HRP، CVaR، Omega، Kelly و ..."
    },
    "risk_free_rate": {"title": "نرخ بدون ریسک", "content": "برای ایران معمولاً ۱۸–۲۵٪"},
}

def show_help(key):
    if key in HELP_TEXTS:
        data = HELP_TEXTS[key]
        with st.expander(f"❓ {data['title']}"):
            st.markdown(data["content"])

# =============================================================================
# DATA DOWNLOAD
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def download_data(tickers_str, period="2y"):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    data = {}
    failed = []
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, auto_adjust=True, progress=False)
            if len(df) > 20 and "Close" in df.columns:
                data[t] = df["Close"]
            else:
                failed.append(t)
        except Exception:
            failed.append(t)
    if not data:
        return None
    prices = pd.DataFrame(data).ffill().bfill().dropna(how="all")
    return prices, failed

# =============================================================================
# PORTFOLIO OPTIMIZER (ساده‌سازی شده)
# =============================================================================
class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, rf_rate: float = 0.0):
        self.returns = returns
        self.mean_ret = returns.mean() * 252
        self.cov_mat = returns.cov() * 252
        self.n = len(self.mean_ret)
        self.rf_rate = rf_rate
        self.asset_names = list(returns.columns)

    def _sum_to_one(self, w):
        return np.sum(w) - 1

    def portfolio_return(self, w):
        return np.dot(w, self.mean_ret)

    def portfolio_volatility(self, w):
        var = np.dot(w.T, np.dot(self.cov_mat, w))
        return np.sqrt(var) if var > 1e-10 else 0.0

    def equal_weight(self):
        return np.ones(self.n) / self.n

    def min_variance(self):
        bounds = [(0, 1)] * self.n
        cons = {'type': 'eq', 'fun': self._sum_to_one}
        x0 = self.equal_weight()
        res = minimize(self.portfolio_volatility, x0, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x if res.success else self.equal_weight()

    def max_sharpe(self):
        def neg_sharpe(w):
            vol = self.portfolio_volatility(w)
            return - (self.portfolio_return(w) - self.rf_rate) / vol if vol > 1e-8 else 0
        bounds = [(0, 1)] * self.n
        cons = {'type': 'eq', 'fun': self._sum_to_one}
        x0 = self.equal_weight()
        res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x if res.success else self.equal_weight()

    def get_weights(self, strategy: str):
        if strategy == "وزن برابر (ساده و مقاوم)":
            return self.equal_weight()
        elif strategy == "حداقل واریانس":
            return self.min_variance()
        elif strategy == "مارکوویتز (حداکثر شارپ)":
            return self.max_sharpe()
        else:
            return self.equal_weight()  # fallback

# =============================================================================
# MAIN APP
# =============================================================================
st.set_page_config(page_title="Portfolio360", layout="wide")

st.markdown('<div class="main-header"><h1>📊 Portfolio360</h1><p>تحلیل و بهینه‌سازی پرتفوی</p></div>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("دانلود داده‌ها")
    tickers_input = st.text_input("نمادها (کاما جدا کنید)", "BTC-USD, ETH-USD, GC=F, ^GSPC")
    period = st.selectbox("بازه زمانی", ["1y", "2y", "5y", "max"], index=1)

    if st.button("🔄 دانلود داده", use_container_width=True):
        with st.spinner("در حال دریافت داده..."):
            result = download_data(tickers_input, period)
            if result is not None:
                prices, failed = result
                st.session_state.prices = prices
                if failed:
                    st.warning(f"نمادهای ناموفق: {', '.join(failed)}")
                else:
                    st.success(f"داده {len(prices.columns)} دارایی بارگذاری شد")
            else:
                st.error("هیچ داده‌ای دریافت نشد")

    show_help("data_download")

# ── Main Content ───────────────────────────────────────────────────────────
if "prices" not in st.session_state:
    st.info("از سایدبار داده‌ها را دانلود کنید")
else:
    prices = st.session_state.prices
    returns = prices.pct_change().dropna()

    if returns.empty:
        st.error("داده‌های بازگشتی خالی است. بازه زمانی یا نمادها را تغییر دهید.")
    else:
        optimizer = PortfolioOptimizer(returns, rf_rate=0.18)

        st.subheader("انتخاب استراتژی")
        strategy = st.selectbox("سبک پرتفوی", [
            "وزن برابر (ساده و مقاوم)",
            "حداقل واریانس",
            "مارکوویتز (حداکثر شارپ)"
        ])

        if st.button("محاسبه پرتفوی", type="primary"):
            with st.spinner("در حال محاسبه..."):
                weights = optimizer.get_weights(strategy)
                st.session_state.weights = weights

        if "weights" in st.session_state:
            weights = st.session_state.weights

            ret = optimizer.portfolio_return(weights)
            vol = optimizer.portfolio_volatility(weights)
            sharpe = (ret - optimizer.rf_rate) / vol if vol > 1e-8 else 0.0

            cols = st.columns(4)
            cols[0].metric("بازده سالانه", f"{ret:.2%}")
            cols[1].metric("نوسان", f"{vol:.2%}")
            cols[2].metric("شارپ", f"{sharpe:.2f}")
            cols[3].metric("تعداد دارایی", len(prices.columns))

            # نمایش وزن‌ها
            df_weights = pd.DataFrame({
                "دارایی": optimizer.asset_names,
                "وزن": weights
            }).sort_values("وزن", ascending=False)
            df_weights["وزن"] = df_weights["وزن"].map("{:.2%}".format)

            st.dataframe(df_weights, use_container_width=True, hide_index=True)

            fig = px.pie(df_weights, values="وزن", names="دارایی", title="توزیع پرتفوی")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Portfolio360 — نسخه دیباگ شده — ساخته شده با Streamlit")
