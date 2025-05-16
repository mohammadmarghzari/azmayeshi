import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # بارگذاری متغیرهای محیطی از فایل .env

st.title("تولید تصویر از متن با OpenAI API")

# دریافت کلید API از متغیر محیطی
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("کلید API یافت نشد! لطفاً کلید خود را در بخش Secrets یا فایل .env وارد کنید.")
    st.stop()

# ساخت کلاینت OpenAI
client = OpenAI(api_key=api_key)

# ورودی متن کاربر
prompt = st.text_area("متن توضیح تصویر را وارد کنید:")

if st.button("تولید تصویر"):
    if not prompt.strip():
        st.warning("لطفاً متن را وارد کنید.")
    else:
        with st.spinner("در حال تولید تصویر..."):
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="512x512",
                    n=1
                )
                image_url = response.data[0].url
                st.image(image_url, caption="تصویر تولید شده")
            except Exception as e:
                st.error(f"خطا در تولید تصویر: {e}")
