import streamlit as st
import requests

st.set_page_config(page_title="AI Classifier", layout="centered")

st.title("🚀 AI Text Classifier")
st.markdown("Classify text into categories using Machine Learning")

# example suggestions
examples = [
    "Space rockets and NASA missions",
    "Computer graphics and images",
    "Politics and government policies",
    "Medical health and diseases"
]

option = st.selectbox("Try example:", ["--Select--"] + examples)

text = st.text_area("Enter your text", value=option if option != "--Select--" else "")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter some text first")
    else:
        with st.spinner("Analyzing..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/predict",
                    params={"text": text}
                )
                result = res.json()["prediction"]

                st.success(f"Prediction: **{result}**")

            except:
                st.error("API not running. Start FastAPI first.")

