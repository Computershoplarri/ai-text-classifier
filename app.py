import pickle
import streamlit as st

model = pickle.load(open("model.pkl", "rb"))

st.title("Support Ticket Classifier")

text = st.text_area("Enter ticket")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter text")
    else:
        pred = model.predict([text])[0]
        st.success(f"Category: {pred}")








