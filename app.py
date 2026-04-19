import streamlit as st
import pickle

# load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.set_page_config(page_title="AI Classifier")

st.title("🚀 AI Text Classifier")

text = st.text_area("Enter text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter text first")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        st.success(f"Prediction: {pred}")

