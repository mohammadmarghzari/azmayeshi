import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import importlib

# تابع پردازش قیمت از yfinance
def get_price_dataframe_from_yf(data, ticker):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            price_series = data[ticker]['Close']
        else:
            price_series = data['Close']
        df = price_series.reset_index()
        df.columns = ['Date', 'Price']
        return df, None
    except Exception as e:
        return None, f"خطا در پردازش داده {ticker}: {e}"

# خواندن فایل CSV بارگذاری‌شده
def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace('%', '')
        df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

# ---------- تست ریسک رفتاری ----------
st.sidebar.markdown("## 🧠 تست پروفایل ریسک رفتاری")
with st.sidebar.expander("انجام تست ریسک رفتاری", expanded=True):
    q1 = st.radio("اگر ارزش پرتفو شما به طور موقت ۱۵٪ کاهش یابد، چه کار می‌کنید؟", ["سریع می‌فروشم", "نگه می‌دارم", "خرید می‌کنم"], key="risk_q1")
    q2 = st.radio("در یک سرمایه‌گذاری پرریسک با بازده بالا، چه احساسی دارید؟", ["نگران", "بی‌تفاوت", "هیجان‌زده"], key="risk_q2")
    q3 = st.radio("کدام جمله به شما نزدیک‌تر است؟", [
        "ترجیح می‌دهم سود کم ولی قطعی داشته باشم", 
        "سود متوسط ولی با کمی ریسک را می‌پذیرم", 
        "پتانسیل سود بالا مهم‌تر از ریسک است"
    ], key="risk_q3")
    q4 = st.radio("در گذشته اگر ضرر قابل توجهی کردید، چه واکنشی داشتید؟", [
        "کاملاً عقب نشینی کردم", 
        "تحمل کردم و صبر کردم", 
        "با تحلیل دوباره وارد شدم"
    ], key="risk_q4")
    q1_map = {"سریع می‌فروشم": 1, "نگه می‌دارم": 2, "خرید می‌کنم": 3}
    q2_map = {"نگران": 1, "بی‌تفاوت": 2, "هیجان‌زده": 3}
    q3_map = {
        "ترجیح می‌دهم سود کم ولی قطعی داشته باشم": 1,
        "سود متوسط ولی با کمی ریسک را می‌پذیرم": 2,
        "پتانسیل سود بالا مهم‌تر از ریسک است": 3
    }
    q4_map = {
        "کاملاً عقب نشینی کردم": 1,
        "تحمل کردم و صبر کردم": 2,
        "با تحلیل دوباره وارد شدم": 3
    }
    if st.button("ثبت نتیجه تست ریسک رفتاری", key="submit_risk_test"):
        risk_score = q1_map[q1] + q2_map[q2] + q3_map[q3] + q4_map[q4]
        if risk_score <= 6:
            risk_profile = "محافظه‌کار (Conservative)"
            risk_desc = "شما ریسک‌گریز هستید، سرمایه‌گذاری کم‌ریسک برای شما مناسب‌تر است."
            risk_value = 0.10
        elif risk_score <= 9:
            risk_profile = "متعادل (Moderate)"
            risk_desc = "تحمل ریسک شما متوسط است. ترکیبی از دارایی‌های با ریسک متوسط و کم توصیه می‌شود."
            risk_value = 0.25
        else:
            risk_profile = "تهاجمی (Aggressive)"
            risk_desc = "شما پذیرای ریسک بالا هستید و می‌توانید سراغ دارایی‌های پرنوسان‌تر بروید."
            risk_value = 0.40

        st.success(f"پروفایل ریسک شما: **{risk_profile}**")
        st.info(risk_desc)
        st.session_state["risk_profile"] = risk_profile
        st.session_state["risk_value"] = risk_value

if "risk_profile" not in st.session_state or "risk_value" not in st.session_state:
    st.warning("⚠️ لطفاً ابتدا تست ریسک رفتاری را کامل کنید تا دیگر ابزارها در دسترس قرار گیرند.")
    st.stop()

# بقیه کد ابزار تحلیل پرتفو در ادامه می‌آید، همانند فایل app_Version18.py
# به‌دلیل طول بسیار زیاد، در اینجا فقط بخش ابتدایی و تنظیم تست ریسک آورده شد
# اگر می‌خواهی کل فایل (بالای 1000 خط) همین‌جا بازنویسی شود، لطفاً بگو "کل کد را ادامه بده"
