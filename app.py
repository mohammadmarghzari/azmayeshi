import streamlit as st
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="تولید تصویر با OpenAI", layout="centered")
st.title("🖼️ تولید تصویر با GPT-4 و DALL·E")

st.markdown("""
در این ابزار متن ساده خود را وارد کنید،  
یک پرامپت حرفه‌ای ساخته می‌شود،  
و بر اساس آن تصویر تولید می‌شود.
""")

user_input = st.text_area("متن توصیفی خود را اینجا وارد کنید:", height=100)

if st.button("تولید تصویر"):
    if not user_input.strip():
        st.error("لطفاً متن توصیفی را وارد کنید.")
    else:
        with st.spinner("در حال ساخت پرامپت و تولید تصویر..."):
            try:
                prompt_response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "شما یک متخصص تولید پرامپت برای مدل‌های تولید تصویر هستید."},
                        {"role": "user", "content": f"لطفاً یک پرامپت خلاقانه و دقیق برای تصویر بر اساس این متن بساز: {user_input}"}
                    ],
                    max_tokens=100,
                    temperature=0.8,
                )
                generated_prompt = prompt_response.choices[0].message['content'].strip()
                
                st.markdown(f"### پرامپت ساخته‌شده:\n\n`{generated_prompt}`")

                image_response = openai.Image.create(
                    prompt=generated_prompt,
                    n=1,
                    size="512x512"
                )
                image_url = image_response['data'][0]['url']

                st.image(image_url, caption="تصویر تولید شده توسط DALL·E", use_column_width=True)

            except Exception as e:
                st.error(f"خطا در تولید تصویر: {e}")
