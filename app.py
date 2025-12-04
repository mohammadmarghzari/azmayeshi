import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf

# ==================== تابع تبدیل دوره به سال و ماه (دقیق و زیبا) ====================
def format_recovery_time(periods, period_type="روزانه"):
    if periods == 0 or np.isnan(periods):
        return "بدون افت جدی"
    
    # تبدیل تعداد دوره به ماه
    if "روز" in period_type:
        total_months = periods / 21  # تقریباً ۲۱ روز معاملاتی در ماه
    elif "ماه" in period_type:
        total_months = periods
    elif "سه‌ماه" in period_type:
        total_months = periods * 3
    elif "شش‌ماه" in period_type:
        total_months = periods * 6
    else:
        total_months = periods / 21
    
    years = int(total_months // 12)
    months = int(total_months % 12)
    
    parts = []
    if years > 0:
        parts.append(f"{years} سال")
    if months > 0:
        parts.append(f"{months} ماه")
    if not parts:
        return f"کمتر از ۱ ماه"
    
    return " و ".join(parts)

# ==================== محاسبه زمان ریکاوری (Recovery Time) ====================
def calculate_recovery_time(returns_series):
    if len(returns_series) < 10:
        return 0
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    in_drawdown = False
    recovery_periods = []
    start_idx = None
    
    for i in range(1, len(cum_returns)):
        if drawdown.iloc[i] < 0:
            if not in_drawdown:
                in_drawdown = True
                start_idx = i
        else:
            if in_drawdown:
                in_drawdown = False
                recovery_periods.append(i - start_idx)
    
    return np.mean(recovery_periods) if recovery_periods else 0

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Pro + زمان ریکاوری", layout="wide")
st.title("Portfolio360 Pro - تحلیل پرتفوی + زمان دقیق بازگشت به قله")
st.markdown("### مرز کارا • VaR • دراوداون • زمان ریکاوری • فارسی و حرفه‌ای")

# ==================== منبع داده ====================
st.sidebar.header("منبع داده")
data_source = st.sidebar.radio("داده‌ها را از کجا بگیریم؟", ["دانلود از Yahoo Finance", "آپلود فایل CSV"])

asset_names = []
prices_list = []

if data_source == "دانلود از Yahoo Finance":
    tickers = st.sidebar.text_input("نمادها (مثال: BTC-USD, GC=F, ^GSPC)", "BTC-USD,ETH-USD,GC=F,SI=F,^GSPC")
    period = st.sidebar.selectbox("بازه زمانی", ["1y","2y","3y","5y","max"], index=3)
    if st.sidebar.button("دانلود داده‌ها"):
        with st.spinner("در حال دانلود..."):
            data = yf.download([t.strip() for t in tickers.split(",")], period=period)['Adj Close']
            data = data.ffill().bfill()
            for col in data.columns:
                asset_names.append(col)
                prices_list.append(data[col])
            st.success(f"{len(asset_names)} دارایی دانلود شد!")

else:
    uploaded = st.sidebar.file_uploader("آپلود فایل CSV", type="csv", accept_multiple_files=True)
    if uploaded:
        for file in uploaded:
            df = pd.read_csv(file)
            name = file.name.replace('.csv','')
            price = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
            asset_names.append(name)
            prices_list.append(price)
        st.success("فایل‌ها آپلود شدند")

if not asset_names:
    st.stop()

# ==================== پردازش داده ====================
prices = pd.concat(prices_list, axis=1)
prices.columns = asset_names
prices = prices.dropna()
returns = prices.pct_change().dropna()

annual_factor = 252
mean_returns = returns.mean() * annual_factor
cov_matrix = returns.cov() * annual_factor

def portfolio_return(w): return np.dot(w, mean_returns) * 100
def portfolio_risk(w): return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * 100
def portfolio_sharpe(w, rf=5): 
    r = portfolio_return(w)
    risk = portfolio_risk(w)
    return (r - rf) / risk if risk > 0 else -99

# ==================== تنظیمات ====================
st.sidebar.header("تنظیمات بهینه‌سازی")
target_type = st.sidebar.selectbox("هدف پرتفو", ["بیشترین شارپ", "کمترین ریسک", "ریسک هدف", "بازده هدف"])
if "هدف" in target_type:
    target_val = st.sidebar.slider("مقدار هدف (%)", 5.0, 100.0, 30.0)

rf_rate = st.sidebar.number_input("نرخ بدون ریسک (%)", 0.0, 20.0, 5.0)
max_btc_weight = st.sidebar.slider("حداکثر وزن بیت‌کوین (%)", 0, 100, 25)

min_w = st.sidebar.slider("حداقل وزن (%)", 0, 40, 5) / 100
max_w = st.sidebar.slider("حداکثر وزن (%)", 30, 100, 60) / 100

# محدودیت وزن
bounds = []
for name in asset_names:
    if 'BTC' in name.upper():
        bounds.append((0, max_btc_weight/100))
    else:
        bounds.append((min_w, max_w))

constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
x0 = np.ones(len(asset_names)) / len(asset_names)

# بهینه‌سازی
res = None
if target_type == "بیشترین شارپ":
    res = minimize(lambda w: -portfolio_sharpe(w, rf_rate), x0, method='SLSQP', bounds=bounds, constraints=constraints)
elif target_type == "کمترین ریسک":
    res = minimize(portfolio_risk, x0, method='SLSQP', bounds=bounds, constraints=constraints)
elif "ریسک هدف" in target_type:
    cons = constraints + [{'type': 'eq', 'fun': lambda w: portfolio_risk(w) - target_val}]
    res = minimize(lambda w: -portfolio_sharpe(w, rf_rate), x0, method='SLSQP', bounds=bounds, constraints=cons)
elif "بازده هدف" in target_type:
    cons = constraints + [{'type': 'eq', 'fun': lambda w: portfolio_return(w) - target_val}]
    res = minimize(portfolio_risk, x0, method='SLSQP', bounds=bounds, constraints=cons)

# ==================== مونت کارلو + مرز کارا ====================
@st.cache_data
def monte_carlo(n=12000):
    results = np.zeros((3, n))
    for i in range(n):
        w = np.random.random(len(asset_names))
        w = np.clip(w, min_w, max_w)
        if any('BTC' in a.upper() for a in asset_names):
            btc_idx = next(i for i, a in enumerate(asset_names) if 'BTC' in a.upper())
            w[btc_idx] = min(w[btc_idx], max_btc_weight/100)
        w /= w.sum()
        results[0,i] = portfolio_return(w)
        results[1,i] = portfolio_risk(w)
        results[2,i] = portfolio_sharpe(w, rf_rate)
    return results

mc = monte_carlo()

# ==================== نمودار مرز کارا ====================
st.markdown("---")
st.subheader("مرز کارا پرتفوی - رنگ بر اساس نسبت شارپ")

fig = go.Figure()
fig.add_trace(go.Scatter(x=mc[1], y=mc[0], mode='markers',
                         marker=dict(color=mc[2], colorscale='RdYlGn', size=5,
                                     colorbar=dict(title="نسبت شارپ")),
                         name="پرتفوهای تصادفی"))

if res and res.success:
    w_opt = res.x
    fig.add_trace(go.Scatter(x=[portfolio_risk(w_opt)], y=[portfolio_return(w_opt)],
                             mode='markers', marker=dict(color='gold', size=18, symbol='star-diamond'),
                             name="پرتفوی بهینه شما"))

fig.update_layout(xaxis_title="ریسک سالیانه (%)", yaxis_title="بازده سالیانه (%)", height=650)
st.plotly_chart(fig, use_container_width=True)

# ==================== نتایج + زمان ریکاوری ====================
if res and res.success:
    w_opt = res.x
    ret = portfolio_return(w_opt)
    risk = portfolio_risk(w_opt)
    sharpe = portfolio_sharpe(w_opt, rf_rate)
    
    # زمان ریکاوری پرتفو
    port_returns = returns.dot(w_opt)
    recovery_days = calculate_recovery_time(port_returns)
    recovery_text = format_recovery_time(recovery_days, "روزانه")

    st.success("پرتفوی بهینه با موفقیت محاسبه شد!")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("بازده سالیانه", f"{ret:.2f}%")
    col2.metric("ریسک سالیانه", f"{risk:.2f}%")
    col3.metric("نسبت شارپ", f"{sharpe:.3f}")
    col4.metric("زمان ریکاوری میانگین", recovery_text)

    # جدول وزنی
    weights_df = pd.DataFrame({
        "دارایی": asset_names,
        "وزن (%)": np.round(w_opt * 100, 2)
    }).sort_values("وزن (%)", ascending=False)

    st.markdown("### تخصیص دارایی‌ها")
    st.dataframe(weights_df, use_container_width=True)

    # نمودار دایره‌ای
    fig_pie = px.pie(weights_df, values="وزن (%)", names="دارایی", title="توزیع وزن دارایی‌ها")
    st.plotly_chart(fig_pie, use_container_width=True)

    # زمان ریکاوری هر دارایی
    st.markdown("### زمان ریکاوری تاریخی هر دارایی")
    rec_list = []
    for name, col in zip(asset_names, returns.columns):
        rt_days = calculate_recovery_time(returns[col])
        rec_list.append({"دارایی": name, "زمان ریکاوری": format_recovery_time(rt_days, "روزانه")})
    st.dataframe(rec_list, use_container_width=True)

else:
    st.warning("پرتفو با این شرایط پیدا نشد. محدودیت‌ها را تغییر دهید.")

st.balloons()
st.caption("Portfolio360 Pro - با زمان ریکاوری دقیق | ساخته شده برای سرمایه‌گذاران حرفه‌ای ایرانی")
