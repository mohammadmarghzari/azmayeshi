import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf

# ==================== توابع کمکی ====================
def format_recovery_time(days):
    if days == 0 or np.isnan(days):
        return "بدون افت جدی"
    months = days / 21  # تقریباً 21 روز معاملاتی در ماه
    years = int(months // 12)
    months_remain = int(months % 12)
    if years > 0 and months_remain > 0:
        return f"{years} سال و {months_remain} ماه"
    elif years > 0:
        return f"{years} سال"
    elif months_remain > 0:
        return f"{months_remain} ماه"
    else:
        return "کمتر از ۱ ماه"

def calculate_recovery_time(returns_series):
    cum = (1 + returns_series).cumprod()
    peak = cum.cummax()
    drawdown = cum / peak - 1
    in_dd = False
    recoveries = []
    start = None
    for i in range(1, len(cum)):
        if drawdown.iloc[i] < 0:
            if not in_dd:
                in_dd = True
                start = i
        else:
            if in_dd:
                in_dd = False
                recoveries.append(i - start)
    return np.mean(recoveries) if recoveries else 0

def portfolio_var(returns_daily, weights, confidence=0.95):
    port_ret = returns_daily.dot(weights)
    return np.percentile(port_ret, (1-confidence)*100) * np.sqrt(252) * 100

def portfolio_cvar(returns_daily, weights, confidence=0.95):
    port_ret = returns_daily.dot(weights)
    var = np.percentile(port_ret, (1-confidence)*100)
    cvar = port_ret[port_ret <= var].mean() * np.sqrt(252) * 100
    return cvar

def max_drawdown(returns_daily, weights):
    port = (1 + returns_daily.dot(weights)).cumprod()
    peak = port.cummax()
    dd = (port / peak - 1).min() * 100
    return dd

# ==================== صفحه ====================
st.set_page_config(page_title="Portfolio360 Ultimate", layout="wide")
st.title("Portfolio360 Ultimate")
st.markdown("### کامل‌ترین ابزار تحلیل پرتفوی فارسی با زمان ریکاوری، VaR، دراوداون و ...")

# ==================== منبع داده ====================
st.sidebar.header("منبع داده")
source = st.sidebar.radio("از کجا داده بگیریم؟", ["Yahoo Finance", "آپلود CSV"])

if source == "Yahoo Finance":
    tickers = st.sidebar.text_input("نمادها (با کاما)", "BTC-USD,ETH-USD,GC=F,SI=F,^GSPC")
    period = st.sidebar.selectbox("بازه", ["1y","3y","5y","max"], index=2)
    if st.sidebar.button("دانلود داده"):
        with st.spinner("در حال دانلود..."):
            data = yf.download([t.strip() for t in tickers.split(",")], period=period)['Adj Close']
            prices = data.ffill().bfill()
else:
    uploaded = st.sidebar.file_uploader("آپلود CSV", type="csv", accept_multiple_files=True)
    if uploaded:
        dfs = []
        for f in uploaded:
            df = pd.read_csv(f)
            col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df = df[['Date', col]].copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.columns = [f.name.replace('.csv','')]
            dfs.append(df)
        prices = pd.concat(dfs, axis=1)

if 'prices' not in locals():
    st.stop()

prices = prices.dropna()
returns = prices.pct_change().dropna()
if len(returns) < 100:
    st.error("داده کافی نیست")
    st.stop()

asset_names = list(prices.columns)
n = len(asset_names)
annual_factor = 252
mean_ret = returns.mean() * annual_factor
cov_mat = returns.cov() * annual_factor

# ==================== توابع پرتفو ====================
def port_return(w): return np.dot(w, mean_ret) * 100
def port_risk(w): return np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * 100
def port_sharpe(w, rf=5): 
    r = port_return(w)
    risk = port_risk(w)
    return (r - rf) / risk if risk > 0 else -999

# ==================== تنظیمات کاربر ====================
st.sidebar.header("تنظیمات بهینه‌سازی")
goal = st.sidebar.selectbox("هدف", ["بیشترین شارپ", "کمترین ریسک", "ریسک هدف", "بازده هدف"])
if "هدف" in goal:
    target = st.sidebar.slider("مقدار هدف (%)", 5.0, 100.0, 30.0)

rf = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 20.0, 5.0)
max_btc = st.sidebar.slider("حداکثر وزن بیت‌کوین (%)", 0, 100, 30)
min_w = st.sidebar.slider("حداقل وزن هر دارایی (%)", 0, 40, 5)/100
max_w = st.sidebar.slider("حداکثر وزن هر دارایی (%)", 30, 100, 70)/100

# محدودیت وزن
bounds = []
for name in asset_names:
    if 'BTC' in name.upper():
        bounds.append((0, max_btc/100))
    else:
        bounds.append((min_w, max_w))

constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
x0 = np.ones(n) / n

# ==================== بهینه‌سازی ====================
res = None
if goal == "بیشترین شارپ":
    res = minimize(lambda w: -port_sharpe(w, rf), x0, bounds=bounds, constraints=constraints, method='SLSQP')
elif goal == "کمترین ریسک":
    res = minimize(port_risk, x0, bounds=bounds, constraints=constraints, method='SLSQP')
elif "ریسک هدف" in goal:
    cons = constraints + [{'type': 'eq', 'fun': lambda w: port_risk(w) - target}]
    res = minimize(lambda w: -port_sharpe(w, rf), x0, bounds=bounds, constraints=cons, method='SLSQP')
elif "بازده هدف" in goal:
    cons = constraints + [{'type': 'eq', 'fun': lambda w: port_return(w) - target}]
    res = minimize(port_risk, x0, bounds=bounds, constraints=cons, method='SLSQP')

# ==================== مونت کارلو ====================
@st.cache_data
def monte_carlo(n_ports=15000):
    results = np.zeros((3, n_ports))
    for i in range(n_ports):
        w = np.random.random(n)
        w = np.clip(w, min_w, max_w)
        if any('BTC' in a.upper() for a in asset_names):
            idx = next(i for i,a in enumerate(asset_names) if 'BTC' in a.upper())
            w[idx] = min(w[idx], max_btc/100)
        w /= w.sum()
        results[0,i] = port_return(w)
        results[1,i] = port_risk(w)
        results[2,i] = port_sharpe(w, rf)
    return results
mc = monte_carlo()

# ==================== نمودار مرز کارا ====================
st.markdown("---")
st.subheader("مرز کارا - رنگ = نسبت شارپ")
fig = go.Figure()
fig.add_trace(go.Scatter(x=mc[1], y=mc[0], mode='markers',
                         marker=dict(color=mc[2], colorscale='RdYlGn', size=6,
                                     colorbar=dict(title="نسبت شارپ")),
                         name="پرتفوهای تصادفی"))

if res and res.success:
    w_opt = res.x
    fig.add_trace(go.Scatter(x=[port_risk(w_opt)], y=[port_return(w_opt)],
                             mode='markers', marker=dict(color='red', size=20, symbol='star-diamond'),
                             name="پرتفوی بهینه شما"))
fig.update_layout(xaxis_title="ریسک سالیانه (%)", yaxis_title="بازده سالیانه (%)", height=650)
st.plotly_chart(fig, use_container_width=True)

# ==================== نتایج نهایی ====================
if res and res.success:
    w = res.x
    r = port_return(w)
    risk = port_risk(w)
    sharpe = port_sharpe(w, rf)
    var95 = -portfolio_var(returns, w, 0.95)
    cvar95 = -portfolio_cvar(returns, w, 0.95)
    dd = max_drawdown(returns, w)
    recovery_days = calculate_recovery_time(returns.dot(w))
    recovery_text = format_recovery_time(recovery_days)

    st.success("پرتفوی بهینه با موفقیت محاسبه شد!")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("بازده سالیانه", f"{r:.2f}%")
    c2.metric("ریسک سالیانه", f"{risk:.2f}%")
    c3.metric("نسبت شارپ", f"{sharpe:.3f}")
    c4.metric("زمان ریکاوری", recovery_text)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("VaR 95%", f"{var95:.2f}%")
    c6.metric("CVaR 95%", f"{cvar95:.2f}%")
    c7.metric("حداکثر دراوداون", f"{dd:.2f}%")
    c8.metric("تعداد دارایی", n)

    st.markdown("### وزن بهینه هر دارایی")
    df_w = pd.DataFrame({"دارایی": asset_names, "وزن (%)": np.round(w*100, 2)}).sort_values("وزن (%)", ascending=False)
    st.dataframe(df_w, use_container_width=True)

    fig_pie = px.pie(df_w, values="وزن (%)", names="دارایی", title="توزیع وزن دارایی‌ها")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### زمان ریکاوری هر دارایی")
    recs = []
    for col in returns.columns:
        days = calculate_recovery_time(returns[col])
        recs.append({"دارایی": col, "زمان ریکاوری": format_recovery_time(days)})
    st.dataframe(recs, use_container_width=True)

else:
    st.error("متأسفانه با این محدودیت‌ها پرتفویی پیدا نشد. محدودیت‌ها را شل کنید.")

st.balloons()
st.caption("Portfolio360 Ultimate - کامل‌ترین ابزار تحلیل پرتفوی فارسی | ۱۴۰۴")
