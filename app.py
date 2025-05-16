import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی از فایل .env
load_dotenv()

# مقداردهی کلاینت
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("تولید تصویر با OpenAI API و Streamlit")

prompt = st.text_input("توضیح تصویری که می‌خواهید تولید شود:")

if st.button("تولید تصویر"):
    if not prompt:
        st.error("لطفاً ابتدا توضیح تصویر را وارد کنید.")
    else:
        try:
            response = client.images.generate(
                model="dall-e-3",  # مدل تولید تصویر، می‌تونی عوض کنی
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            image_url = response.data[0].url
            st.image(image_url, caption="تصویر تولید شده")
        except Exception as e:
            st.error(f"خطا در تولید تصویر: {e}")
