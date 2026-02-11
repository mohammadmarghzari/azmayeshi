"""
Portfolio360 Mobile Pro
Persian | Mobile Friendly | Options & Hedging
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# ==================================================
# Page Config
# ==================================================
st.set_page_config(
    page_title="پرتفوی ۳۶۰",
    layout="wide"
)

# ==================================================
# Mobile + Persian CSS
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif;
    direction: rtl;
}

.block-container {
    padding: 0.8rem;
}

h1 { font-size: 1.6rem; }
h2 { font-size: 1.2rem; }
h3 { font-size: 1rem; }

.stButton>button {
    width: 100%;
    border-radius: 16px;
    padding: 0.6rem;
    font-weight: 600;
}

.card {
    background: #ffffff;
    border-radius: 20px;
    padding: 14px;
    margin-bottom: 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}

.metric {
    background: #f5f7fa;
    border-radius: 14px;
    padding: 10px;
    text-align: center;
    font-weight: 600;
}

small { color: #666; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# Header
# ==================================================
st.markdown("""
<h1 style="text-align:center;color:#0b9bd1">📊 پرتفوی ۳۶۰</h1>
<p style="text-align:center;color:#666;font-size:0.9rem">
نسخه موبایل‌پسند مدیریت سرمایه، اپشن و هجینگ
</p>
""", unsafe_allow_html=True)

# ==================================================
# Utils
# ==================================================
@st.cache_data(show_spinner=False)
def load_prices(tickers, period):
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    data = {}
    for s in symbols:
        try:
            df = yf.Ticker(s).history(period=period, auto_adjust=True)
            if "Close" in df.columns and len(df) > 20:
                data[s] = df["Close"]
        except Exception:
            pass
    if not data:
        return None
    return pd.DataFrame(data).ffill().bfill()

def portfolio_risk(weights, cov):
    return float(np.sqrt(weights.T @ cov @ weights) * 100)

# ==================================================
# Sidebar (Mobile Optimized)
# ==================================================
with st.sidebar:
    st.markdown("## 📥 داده بازار")

    tickers = st.text_input(
        "نمادها",
        "BTC-USD, ETH-USD, GC=F"
    )

    period = st.selectbox(
        "بازه زمانی",
        ["1y", "2y", "5y", "max"],
        index=1
    )

    if st.button("دانلود داده"):
        with st.spinner("در حال دریافت..."):
            prices = load_prices(tickers, period)
            if prices is None:
                st.error("داده‌ای دریافت نشد")
            else:
                st.session_state.prices = prices
                st.success("داده‌ها آماده شد")
                st.rerun()

# ==================================================
# Main
# ==================================================
if "prices" not in st.session_state:
    st.info("⬅️ ابتدا داده‌ها را دانلود کنید")
    st.stop()

prices = st.session_state.prices
assets = list(prices.columns)
returns = prices.pct_change().dropna()
cov = returns.cov() * 252

weights = np.ones(len(assets)) / len(assets)

# ==================================================
# Portfolio Card
# ==================================================
st.markdown("## 🧩 پرتفوی")

st.markdown('<div class="card">', unsafe_allow_html=True)
df_w = pd.DataFrame({
    "دارایی": assets,
    "وزن (%)": np.round(weights * 100, 2)
})
st.dataframe(df_w, use_container_width=True, height=220)
st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# Allocation Card
# ==================================================
st.markdown("## 💰 تخصیص سرمایه")

capital = st.number_input("کل سرمایه (دلار)", 100, 1_000_000, 5000, 500)
rate = st.number_input("نرخ دلار (تومان)", 100_000, 1_000_000, 600_000, 10_000)

alloc = []
for i, a in enumerate(assets):
    usd = capital * weights[i]
    alloc.append({
        "دارایی": a,
        "دلار": round(usd, 2),
        "تومان": f"{int(usd * rate):,}"
    })

st.markdown('<div class="card">', unsafe_allow_html=True)
st.dataframe(pd.DataFrame(alloc), use_container_width=True, height=220)
st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# Hedging & Options (Mobile Card)
# ==================================================
st.markdown("## 🛡️ اپشن و هجینگ")

st.markdown('<div class="card">', unsafe_allow_html=True)

hedge_type = st.selectbox(
    "انتخاب استراتژی هج",
    ["بدون هج", "Protective Put", "Collar"]
)

premium_pct = st.slider(
    "هزینه تقریبی آپشن (% از سرمایه)",
    0.0, 10.0, 3.0, 0.1
)

original_risk = portfolio_risk(weights, cov)
risk_reduction = premium_pct * 0.4
hedged_risk = max(original_risk - risk_reduction, 0.2)

c1, c2, c3 = st.columns(3)
c1.markdown(f"<div class='metric'>ریسک فعلی<br>{original_risk:.2f}%</div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric'>ریسک بعد از هج<br>{hedged_risk:.2f}%</div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric'>هزینه هج<br>{premium_pct:.1f}%</div>", unsafe_allow_html=True)

st.markdown("""
<small>
🔹 Protective Put: محافظت در ریزش شدید  
🔹 Collar: هج کم‌هزینه با محدودیت سود  
</small>
""")

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# Option Payoff (Visual)
# ==================================================
st.markdown("## 📈 نمودار سود/زیان آپشن")

st.markdown('<div class="card">', unsafe_allow_html=True)

spot = st.number_input("قیمت فعلی دارایی", value=100.0)
strike = st.number_input("قیمت اعمال (Strike)", value=90.0)
premium = st.number_input("Premium", value=4.0)

prices_grid = np.linspace(spot * 0.5, spot * 1.5, 200)
underlying_pnl = prices_grid - spot
put_pnl = np.maximum(strike - prices_grid, 0) - premium
total_pnl = underlying_pnl + put_pnl

fig = go.Figure()
fig.add_trace(go.Scatter(x=prices_grid, y=total_pnl, name="Payoff", line=dict(width=3)))
fig.add_hline(y=0, line_dash="dash")

fig.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=40, b=10),
    template="plotly_white",
    title="Married Put (هج با آپشن فروش)"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# Footer
# ==================================================
st.markdown("""
<p style="text-align:center;color:#777;font-size:0.8rem">
پرتفوی ۳۶۰ — نسخه موبایل‌پسند حرفه‌ای  
<br>
اپشن | هجینگ | مدیریت ریسک
</p>
""", unsafe_allow_html=True)
