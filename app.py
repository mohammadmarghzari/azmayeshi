
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Portfolio & Options Analyzer", layout="wide")

st.title("📈 Portfolio & Options Scenario Analyzer")

st.sidebar.header("Portfolio Inputs")

# Spot Assets
st.sidebar.subheader("Spot Assets")
n_spot = st.sidebar.number_input("Number of Spot Assets", 0, 50, 1)

spots = []
for i in range(n_spot):
    with st.sidebar.expander(f"Spot #{i+1}"):
        symbol = st.text_input(f"Symbol {i}", value="ETH", key=f"sym{i}")
        qty = st.number_input(f"Quantity {i}", value=0.0, key=f"qty{i}")
        price = st.number_input(f"Current Price {i}", value=2000.0, key=f"pr{i}")
        spots.append({"symbol": symbol, "qty": qty, "price": price})

# Debt
debt = st.sidebar.number_input("Debt (USDC/USD)", value=0.0)

# Options
st.sidebar.subheader("Options")
n_opt = st.sidebar.number_input("Number of Options", 0, 100, 1)

options = []
for i in range(n_opt):
    with st.sidebar.expander(f"Option #{i+1}"):
        underlying = st.text_input(f"Underlying {i}", value="ETH", key=f"u{i}")
        opt_type = st.selectbox(f"Type {i}", ["Call", "Put"], key=f"t{i}")
        strike = st.number_input(f"Strike {i}", value=2200.0, key=f"k{i}")
        premium = st.number_input(f"Premium Cost {i}", value=100.0, key=f"p{i}")
        qty = st.number_input(f"Contracts/Qty {i}", value=1.0, key=f"q{i}")
        expiry = st.text_input(f"Expiry {i}", value="2025-12-31", key=f"e{i}")
        options.append({
            "underlying": underlying,
            "type": opt_type,
            "strike": strike,
            "premium": premium,
            "qty": qty,
            "expiry": expiry
        })

st.sidebar.subheader("Scenario Range")
price_min = st.sidebar.number_input("Min Price", value=1000.0)
price_max = st.sidebar.number_input("Max Price", value=4000.0)
steps = st.sidebar.slider("Scenarios", 5, 50, 15)

# Current values
spot_value = sum(x["qty"] * x["price"] for x in spots)
option_cost = sum(x["premium"] for x in options)
net_value = spot_value - debt - option_cost

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot Value", f"${spot_value:,.2f}")
c2.metric("Debt", f"${debt:,.2f}")
c3.metric("Option Cost", f"${option_cost:,.2f}")
c4.metric("Net Value", f"${net_value:,.2f}")

prices = np.linspace(price_min, price_max, steps)

rows = []

for price in prices:
    scenario_spot = sum(x["qty"] * price for x in spots if x["symbol"].upper()=="ETH") \
                  + sum(x["qty"] * x["price"] for x in spots if x["symbol"].upper()!="ETH")

    option_value = 0
    for op in options:
        if op["type"] == "Call":
            option_value += max(price - op["strike"], 0) * op["qty"]
        else:
            option_value += max(op["strike"] - price, 0) * op["qty"]

    total = scenario_spot + option_value
    pnl = total - debt - option_cost

    rows.append({
        "Underlying Price": round(price,2),
        "Spot Value": round(scenario_spot,2),
        "Option Value": round(option_value,2),
        "Portfolio Value": round(total,2),
        "Net P/L": round(pnl,2)
    })

df = pd.DataFrame(rows)

st.subheader("Scenario Table")
st.dataframe(df, use_container_width=True)

st.subheader("Profit / Loss Curve")
fig = px.line(df, x="Underlying Price", y="Net P/L", markers=True)
st.plotly_chart(fig, use_container_width=True)

positive = df[df["Net P/L"] >= 0]
if len(positive):
    be = positive.iloc[0]["Underlying Price"]
    st.success(f"Approx Break-even Price: {be:,.2f}")
else:
    st.warning("No break-even found in selected range")

st.subheader("Portfolio Summary")
st.write(f"Spot Assets: {len(spots)}")
st.write(f"Options: {len(options)}")
st.write(f"Debt: ${debt:,.2f}")
