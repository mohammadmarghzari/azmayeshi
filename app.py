import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf
import warnings
import io
import base64
from datetime import datetime
warnings.filterwarnings("ignore")

# ==================== تم دارک/لایت ====================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def set_theme():
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
        .css-1d391kg {background-color: #0e1117; color: white;}
        .css-18e3th9 {background-color: #1f2c3a;}
        .css-1v3fvcr {color: white;}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .css-1d391kg {background-color: white; color: black;}
        </style>
        """, unsafe_allow_html=True)

set_theme()

# ==================== دانلود + ذخیره پرتفو + بک‌تست + اکسل + تلگرام ====================
# (کد اصلی قبلی تا calculate_portfolio همون قبلی هست — فقط بخش‌های جدید رو اضافه کردم)

# بعد از calculate_portfolio() این بخش جدید رو اضافه کن:

# ==================== بخش پریمیوم (ذخیره، اکسپورت، بک‌تست، تلگرام) ====================
if "portfolio_history" not in st.session_state:
    st.session_state.portfolio_history = {}

st.sidebar.markdown("---")
st.sidebar.subheader("ویژگی‌های پریمیوم")

# 1. ذخیره پرتفو
portfolio_name = st.sidebar.text_input("اسم پرتفو رو بنویس و ذخیره کن", placeholder="مثلاً: پرتفوی زمستان ۱۴۰۴")
if st.sidebar.button("ذخیره پرتفوی"):
    if portfolio_name and "weights" in locals():
        st.session_state.portfolio_history[portfolio_name] = {
            "weights": weights.tolist(),
            "names": asset_names,
            "return": ret,
            "risk": risk,
            "sharpe": sharpe,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.sidebar.success(f"پرتفوی «{portfolio_name}» ذخیره شد!")

# نمایش پرتفوهای ذخیره شده
if st.session_state.portfolio_history:
    st.sidebar.markdown("### پرتفوهای ذخیره‌شده")
    for name, data in st.session_state.portfolio_history.items():
        with st.sidebar.expander(f"{name} — {data['date']}"):
            st.write(f"بازده: {data['return']:.1f}% | ریسک: {data['risk']:.1f}% | شارپ: {data['sharpe']:.2f}")

# 2. اکسپورت به اکسل
if st.sidebar.button("اکسپورت به اکسل"):
    df_export = df_w.copy()
    df_export["تاریخ"] = datetime.now().strftime("%Y-%m-%d")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, sheet_name='پرتفوی', index=False)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Portfolio360_{datetime.now().strftime("%Y%m%d")}.xlsx">دانلود فایل اکسل</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)
    st.sidebar.success("فایل اکسل آماده دانلود است!")

# 3. بک‌تست ۵ ساله
if st.sidebar.button("بک‌تست ۵ ساله با نمودار رشد سرمایه"):
    with st.spinner("در حال محاسبه بک‌تست..."):
        full_prices = st.session_state.prices
        full_returns = full_prices.pct_change().dropna()
        cumulative = (1 + full_returns.dot(weights)).cumprod()
        cumulative = cumulative / cumulative.iloc[0] * 100
        
        fig_backtest = go.Figure()
        fig_backtest.add_trace(go.Scatter(y=cumulative, name="رشد سرمایه (با وزن فعلی)"))
        fig_backtest.add_trace(go.Scatter(y=np.ones(len(cumulative))*100, line=dict(dash="dash"), name="سرمایه اولیه"))
        fig_backtest.update_layout(title="بک‌تست ۵ ساله — رشد سرمایه با پرتفوی فعلی", height=500)
        st.plotly_chart(fig_backtest, use_container_width=True)

# 4. اعلان تلگرام
telegram_token = st.sidebar.text_input("توکن ربات تلگرام (اختیاری)", type="password")
chat_id = st.sidebar.text_input("چت آیدی تلگرام (اختیاری)")
if st.sidebar.button("ارسال پرتفو به تلگرام"):
    if telegram_token and chat_id and "weights" in locals():
        try:
            import requests
            text = f"""
            پرتفوی جدید — {datetime.now().strftime("%Y-%m-%d")}
            سبک: {st.session_state.selected_style}
            بازده: {ret:.1f}% | ریسک: {risk:.1f}% | شارپ: {sharpe:.2f}
            هجینگ: {st.session_state.hedge_type}
            """
            for name, w in zip(asset_names, weights):
                text += f"\n{name}: {w*100:.1f}%"
            url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            requests.post(url, data={"chat_id": chat_id, "text": text})
            st.sidebar.success("پرتفو با موفقیت به تلگرام ارسال شد!")
        except:
            st.sidebar.error("ارسال ناموفق — توکن یا چت آیدی اشتباهه")
    else:
        st.sidebar.warning("توکن و چت آیدی رو وارد کن!")

# 5. تغییر تم
if st.sidebar.button("تغییر تم (دارک/لایت)"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

# ==================== پایان ====================
st.balloons()
st.caption("Portfolio360 Ultimate Pro — اولین اپ سرمایه‌گذاری ایرانی با ذخیره پرتفو + اکسل + تلگرام + تم دارک | ۱۴۰۴ | با عشق برای ایران")
