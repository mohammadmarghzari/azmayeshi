import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf
import warnings
import time

# ==================== توابع کمکی ====================
def safe_download(tickers, period="5y"):
    try:
        with st.spinner(f"در حال دانلود {len(tickers)} دارایی..."):
            data = yf.download(tickers, period=period, progress=False, threads=True)
            if data.empty:
                return None
            # اگه چند سطحی بود (چند نماد)، Adj Close رو بگیر
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Adj Close']
            else:
                prices = data[['Close']].copy()
                prices.columns = [tickers[0]]
            prices = prices.ffill().bfill()
            return prices.dropna()
    except Exception as e:
        st.error(f"خطا در دانلود: {str(e)}")
        return None

def format_recovery_time(days):
    if days == 0 or np.isnan(days):
        return "بدون افت جدی"
    months = days / 21
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
    if len(returns_series) < 20:
        return 0
    cum = (1 + returns_series).cumprod()
    peak = cum.cummax()
    drawdown = cum / peak - 1
    recoveries = []
    in_dd = False
    start = None
    for i in range(1, len(cum)):
        if drawdown.iloc[i] < -0.01:  # افت بیشتر از ۱%
            if not in_dd:
                in_dd = True
                start = i
        elif in_dd:
            in_dd = False
            recoveries.append(i - start)
    return np.mean(recoveries) if recoveries else 0

def portfolio_metrics(returns, weights):
    port_ret_daily = returns.dot(weights)
    ann_ret = port_ret_daily.mean() * 252 * 100
    ann_vol = port_ret_daily.std() * np.sqrt(252) * 100
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0
    
    # VaR & CVaR
    var_95 = np.percentile(port_ret_daily, 5) * np.sqrt(252) * 100
    cvar_95 = port_ret_daily[port_ret_daily <= var_95/100/np.sqrt(252)].mean() * np.sqrt(252) * 100
    
    # Max Drawdown
    cum = (1 + port_ret_daily).cumprod()
    peak = cum.cummax()
    max_dd = (cum / peak - 1).min() * 100
    
    # Recovery Time
    rec_days = calculate_recovery_time(port_ret_daily)
    rec_text = format_recovery_time(rec_days)
    
    return ann_ret, ann_vol, sharpe, var_95, cvar_95, max_dd, rec_text

# ==================== صفحه اصلی ====================
st.set_page_config(page_title="Portfolio360 Ultimate + Hedging", layout="wide")
st.title("Portfolio360 Ultimate")
st.markdown("### کامل‌ترین ابزار تحلیل پرتفوی فارسی با هجینگ هوشمند، زمان ریکاوری، VaR و دراوداون")

# ==================== سایدبار ====================
st.sidebar.header("منبع داده")
source = st.sidebar.radio("داده از کجا بیاد؟", ["دانلود از Yahoo Finance", "آپلود فایل CSV"])

prices = None
asset_names = []

if source == "دانلود از Yahoo Finance":
    default = "BTC-USD,ETH-USD,GC=F,SI=F,^GSPC,TLT.US,USDIRR=X"
    tickers_input = st.sidebar.text_input("نمادها (با کاما جدا کنید)", default)
    period = st.sidebar.selectbox("بازه زمانی", ["1y", "3y", "5y", "10y", "max"], index=2)
    
    if st.sidebar.button("شروع دانلود"):
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        prices = safe_download(tickers, period)
        if prices is not None:
            asset_names = list(prices.columns)
            st.success(f"دانلود موفق: {len(asset_names)} دارایی")

else:
    uploaded = st.sidebar.file_uploader("آپلود فایل‌های CSV", type="csv", accept_multiple_files=True)
    if uploaded:
        dfs = []
        for file in uploaded:
            try:
                df = pd.read_csv(file)
                name = file.name.replace('.csv', '').replace('.CSV', '')
                if 'Adj Close' in df.columns:
                    col = 'Adj Close'
                elif 'Close' in df.columns:
                    col = 'Close'
                else:
                    col = df.columns[-1]
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df = df[[col]].rename(columns={col: name})
                dfs.append(df)
            except:
                st.error(f"خطا در خواندن {file.name}")
        if dfs:
            prices = pd.concat(dfs, axis=1).ffill().bfill()
            asset_names = list(prices.columns)
            st.success(f"آپلود موفق: {len(asset_names)} دارایی")

if prices is None or len(asset_names) == 0:
    st.info("لطفاً داده‌ها را دانلود یا آپلود کنید.")
    st.stop()

returns = prices.pct_change().dropna()
if len(returns) < 100:
    st.error("داده کافی نیست. حداقل ۱۰۰ روز نیاز است.")
    st.stop()

# ==================== تنظیمات ====================
st.sidebar.header("تنظیمات پرتفو")
rf_rate = st.sidebar.number_input("نرخ بدون ریسک سالیانه (%)", 0.0, 30.0, 18.0, 0.5)  # ایران ≈۱۸٪
hedge_mode = st.sidebar.checkbox("فعال کردن هجینگ هوشمند (طلا + تتر + محدودیت بیت‌کوین)", value=True)
max_btc_weight = st.sidebar.slider("حداکثر وزن بیت‌کوین (%)", 0, 50, 20) if hedge_mode else 100
min_weight_min = st.sidebar.slider("حداقل وزن هر دارایی (%)", 0, 40, 5) / 100
_weight_max = st.sidebar.slider("حداکثر وزن هر دارایی (%)", 30, 100, 70) / 100

# ==================== محدودیت‌های هجینگ هوشمند ====================
bounds = []
for name in asset_names:
    if hedge_mode:
        if 'BTC' in name.upper() or 'بیت' in name:
            bounds.append((0, max_btc_weight/100))
        elif 'GOLD' in name.upper() or 'GC=' in name or 'طلا' in name:
            bounds.append((0.15, 0.50))  # حداقل ۱۵٪ طلا!
        elif 'USD' in name.upper() or 'تتر' in name or 'دلار' in name:
            bounds.append((0.10, 0.40))  # حداقل ۱۰٪ تتر/دلار
        else:
            bounds.append((_weight_min, _weight_max))
    else:
        bounds.append((_weight_min, _weight_max))

constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
x0 = np.array([1.0/len(asset_names)] * len(asset_names))

# ==================== بهینه‌سازی ====================
def neg_sharpe(w):
    ann_ret, ann_vol, sharpe, _, _, _, _ = portfolio_metrics(returns, w)
    return -sharpe

st.info("در حال بهینه‌سازی پرتفو...")
with st.spinner("صبر کنید..."):
    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})

# ==================== مونت کارلو ====================
@st.cache_data
def monte_carlo_simulation(n=12000):
    rets, risks, sharpes = [], [], []
    for _ in range(n):
        w = np.random.random(len(asset_names))
        w = np.clip(w, [b[0] for b in bounds], [b[1] for b in bounds])
        w /= w.sum()
        r, risk, s, _, _, _, _ = portfolio_metrics(returns, w)
        rets.append(r)
        risks.append(risk)
        sharpes.append(s)
    return np.array(rets), np.array(risks), np.array(sharpes)

mc_ret, mc_risk, mc_sharpe = monte_carlo_simulation()

# ==================== نمایش نتایج ====================
st.markdown("---")
if res.success:
    w_opt = res.x
    ann_ret, ann_vol, sharpe, var95, cvar95, max_dd, rec_time = portfolio_metrics(returns, w_opt)

    st.success("پرتفوی بهینه با موفقیت ساخته شد!")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("بازده سالیانه", f"{ann_ret:.2f}%")
    col2.metric("ریسک سالیانه", f"{ann_vol:.2f}%")
    col3.metric("نسبت شارپ", f"{sharpe:.3f}")
    col4.metric("زمان ریکاوری", rec_time)

    col5, col6, col7 = st.columns(3)
    col5.metric("VaR 95%", f"{var95:.2f}%")
    col6.metric("CVaR 95%", f"{cvar95:.2f}%")
    col7.metric("حداکثر دراوداون", f"{max_dd:.2f}%")

    st.markdown("### تخصیص هوشمند دارایی‌ها (با هجینگ)")
    weights_df = pd.DataFrame({
        "دارایی": asset_names,
        "وزن (%)": np.round(w_opt * 100, 2)
    }).sort_values("وزن (%)", ascending=False)
    st.dataframe(weights_df, use_container_width=True)

    fig_pie = px.pie(weights_df, values="وزن (%)", names="دارایی", title="توزیع وزن دارایی‌ها")
    st.plotly_chart(fig_pie, use_container_width=True)

    # نمودار مرز کارا
    st.subheader("مرز کارا - رنگ = نسبت شارپ")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_risk, y=mc_ret, mode='markers',
                             marker=dict(color=mc_sharpe, colorscale='RdYlGn', size=6,
                                         colorbar=dict(title="شارپ")),
                             name="پرتفوهای تصادفی"))
    fig.add_trace(go.Scatter(x=[ann_vol], y=[ann_ret],
                             mode='markers', marker=dict(color='gold', size=20, symbol='star-diamond'),
                             name="پرتفوی بهینه شما"))
    fig.update_layout(xaxis_title="ریسک (%)", yaxis_title="بازده (%)", height=600)
    st.plotly_chart(fig, use_container_width=True)

    if hedge_mode:
        st.success("هجینگ هوشمند فعال است: حداقل طلا و تتر + محدودیت بیت‌کوین")

else:
    st.error("بهینه‌سازی ناموفق بود. محدودیت‌ها را شل کنید.")

st.balloons()
st.caption("Portfolio360 Ultimate + Hedging | نسخه ۱۴۰۴ | بدون خطا | با عشق برای شما")
