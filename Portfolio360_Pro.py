
# Portfolio360 Pro - Enhanced Version (Starter Pro Edition)
# Added: Efficient Frontier, Monte Carlo, VaR/CVaR, Correlation Heatmap,
# Risk Score, Dark Theme foundations, Better Metrics

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio360 Pro", layout="wide")

@st.cache_data
def load_data(tickers):
    data = yf.download(tickers, auto_adjust=True, progress=False)["Close"]
    return data.dropna()

def portfolio_stats(weights, returns, rf=0.0):
    mean = np.sum(returns.mean()*weights)*252
    vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    sharpe = (mean-rf)/vol if vol else 0
    return mean, vol, sharpe

def neg_sharpe(weights, returns, rf):
    return -portfolio_stats(weights, returns, rf)[2]

def var_cvar(series, alpha=0.95):
    q = np.percentile(series, (1-alpha)*100)
    cvar = series[series <= q].mean()
    return q, cvar

st.title("Portfolio360 Pro")

tickers = st.sidebar.text_input(
    "Tickers",
    "BTC-USD,ETH-USD,GC=F,^GSPC"
)

rf = st.sidebar.number_input("Risk Free Rate", value=0.05)

if st.sidebar.button("Run Analysis"):
    assets = [x.strip() for x in tickers.split(",")]
    prices = load_data(assets)

    returns = prices.pct_change().dropna()

    n = len(assets)
    bounds = tuple((0,1) for _ in range(n))
    cons = ({'type':'eq','fun':lambda x: np.sum(x)-1})
    init = np.ones(n)/n

    opt = minimize(
        neg_sharpe,
        init,
        args=(returns,rf),
        method='SLSQP',
        bounds=bounds,
        constraints=cons
    )

    w = opt.x

    ret, vol, sharpe = portfolio_stats(w, returns, rf)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Return", f"{ret*100:.2f}%")
    c2.metric("Volatility", f"{vol*100:.2f}%")
    c3.metric("Sharpe", f"{sharpe:.2f}")

    port_daily = returns.dot(w)

    var95, cvar95 = var_cvar(port_daily,0.95)

    risk_score = max(0,min(100,100-(vol*100*1.5)))

    c4.metric("Risk Score", f"{risk_score:.0f}/100")

    st.subheader("Optimal Weights")
    st.dataframe(
        pd.DataFrame({
            "Asset": assets,
            "Weight %": np.round(w*100,2)
        })
    )

    st.subheader("VaR / CVaR")
    st.write({
        "VaR95%": round(var95*100,2),
        "CVaR95%": round(cvar95*100,2)
    })

    st.subheader("Correlation Heatmap")
    corr = returns.corr()
    fig = px.imshow(corr,text_auto=True)
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Efficient Frontier")

    pts = []
    for _ in range(3000):
        rw = np.random.random(n)
        rw /= rw.sum()
        r,v,s = portfolio_stats(rw,returns,rf)
        pts.append([r,v,s])

    ef = pd.DataFrame(pts,columns=["Return","Risk","Sharpe"])

    fig2 = px.scatter(
        ef,
        x="Risk",
        y="Return",
        color="Sharpe",
        title="Efficient Frontier"
    )
    st.plotly_chart(fig2,use_container_width=True)

    st.subheader("Monte Carlo (1 Year)")

    sims = []
    mu = port_daily.mean()
    sigma = port_daily.std()

    for _ in range(5000):
        path = np.random.normal(mu,sigma,252)
        sims.append((1+path).prod())

    mc = pd.Series(sims)

    fig3 = px.histogram(mc,bins=60,title="Monte Carlo Distribution")
    st.plotly_chart(fig3,use_container_width=True)

    st.write({
        "Worst 5%": round(mc.quantile(0.05),2),
        "Median": round(mc.quantile(0.50),2),
        "Best 5%": round(mc.quantile(0.95),2)
    })
