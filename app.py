
# Portfolio360 Institutional Edition
# Extended architecture skeleton with:
# - Efficient Frontier
# - Monte Carlo
# - VaR / CVaR
# - Black-Scholes
# - Greeks
# - Stress Testing
# - Correlation Matrix
# - Risk Regime
# - Antifragility Score
# - Portfolio Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm

st.set_page_config(page_title="Portfolio360 Institutional", layout="wide")

# ---------- DATA ----------
@st.cache_data
def load_data(tickers):
    return yf.download(tickers, auto_adjust=True, progress=False)["Close"].dropna()

# ---------- PORTFOLIO ----------
def portfolio_stats(w, returns, rf):
    r = np.sum(returns.mean()*w)*252
    v = np.sqrt(np.dot(w.T, np.dot(returns.cov()*252, w)))
    s = (r-rf)/v if v else 0
    return r,v,s

def neg_sharpe(w, returns, rf):
    return -portfolio_stats(w, returns, rf)[2]

# ---------- RISK ----------
def var_cvar(series, alpha=0.95):
    q = np.percentile(series,(1-alpha)*100)
    cvar = series[series<=q].mean()
    return q,cvar

def max_drawdown(series):
    wealth=(1+series).cumprod()
    peak=wealth.cummax()
    dd=(wealth-peak)/peak
    return dd.min()

# ---------- OPTIONS ----------
def black_scholes_call(S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

def greeks_call(S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)

    delta=norm.cdf(d1)
    gamma=norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta=(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
           -r*K*np.exp(-r*T)*norm.cdf(d2))/365
    vega=S*norm.pdf(d1)*np.sqrt(T)/100
    rho=K*T*np.exp(-r*T)*norm.cdf(d2)/100

    return delta,gamma,theta,vega,rho

# ---------- UI ----------
st.title("Portfolio360 Institutional")

tab1,tab2,tab3,tab4 = st.tabs(
    ["Portfolio","Risk","Options","Stress Test"]
)

with st.sidebar:
    tickers = st.text_input(
        "Tickers",
        "BTC-USD,ETH-USD,GC=F,^GSPC"
    )
    rf = st.number_input("Risk Free",0.0,1.0,0.05)
    run = st.button("Analyze")

if run:

    assets=[x.strip() for x in tickers.split(",")]
    prices=load_data(assets)
    returns=prices.pct_change().dropna()

    n=len(assets)

    cons=({'type':'eq','fun':lambda x:np.sum(x)-1})
    bounds=tuple((0,1) for _ in range(n))

    opt=minimize(
        neg_sharpe,
        np.ones(n)/n,
        args=(returns,rf),
        method='SLSQP',
        bounds=bounds,
        constraints=cons
    )

    w=opt.x
    ret,vol,sharpe=portfolio_stats(w,returns,rf)

    with tab1:
        c1,c2,c3=st.columns(3)
        c1.metric("Return",f"{ret*100:.2f}%")
        c2.metric("Volatility",f"{vol*100:.2f}%")
        c3.metric("Sharpe",f"{sharpe:.2f}")

        st.dataframe(
            pd.DataFrame({
                "Asset":assets,
                "Weight %":np.round(w*100,2)
            })
        )

        pts=[]
        for _ in range(5000):
            rw=np.random.random(n)
            rw/=rw.sum()
            r,v,s=portfolio_stats(rw,returns,rf)
            pts.append([r,v,s])

        ef=pd.DataFrame(
            pts,
            columns=["Return","Risk","Sharpe"]
        )

        fig=px.scatter(
            ef,
            x="Risk",
            y="Return",
            color="Sharpe"
        )
        st.plotly_chart(fig,use_container_width=True)

    with tab2:

        port=returns.dot(w)

        var95,cvar95=var_cvar(port)

        st.metric("VaR95",f"{var95*100:.2f}%")
        st.metric("CVaR95",f"{cvar95*100:.2f}%")

        st.metric(
            "Max Drawdown",
            f"{max_drawdown(port)*100:.2f}%"
        )

        corr=returns.corr()

        fig2=px.imshow(
            corr,
            text_auto=True,
            title="Correlation Matrix"
        )
        st.plotly_chart(fig2,use_container_width=True)

        skew=port.skew()
        kurt=port.kurtosis()

        anti=max(
            0,
            min(
                100,
                50+(skew*10)-(kurt)
            )
        )

        st.metric(
            "Antifragility Score",
            f"{anti:.0f}/100"
        )

    with tab3:

        S=st.number_input("Spot",value=2000.0)
        K=st.number_input("Strike",value=2200.0)
        T=st.number_input("Years To Expiry",value=0.25)
        sigma=st.number_input("IV",value=0.55)

        price=black_scholes_call(
            S,K,T,rf,sigma
        )

        d,g,t,v,rho=greeks_call(
            S,K,T,rf,sigma
        )

        st.metric("Call Price",f"{price:.2f}")
        st.write({
            "Delta":round(d,4),
            "Gamma":round(g,6),
            "Theta":round(t,4),
            "Vega":round(v,4),
            "Rho":round(rho,4)
        })

    with tab4:

        shock=st.slider(
            "Market Shock %",
            -80,
            80,
            -20
        )

        stressed=((1+shock/100)*prices.iloc[-1])

        stress_df=pd.DataFrame({
            "Asset":assets,
            "Current":prices.iloc[-1].values,
            "Stressed":stressed.values
        })

        st.dataframe(stress_df)

        sims=[]

        mu=port.mean()
        sd=port.std()

        for _ in range(10000):
            path=np.random.normal(mu,sd,252)
            sims.append((1+path).prod())

        fig3=px.histogram(
            sims,
            nbins=80,
            title="Monte Carlo"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )
