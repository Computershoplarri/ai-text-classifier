import pickle
import streamlit as st

# load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="AI Support Classifier", layout="centered")

st.title("🎫 Production AI Support Ticket Classifier")

text = st.text_area("Enter support ticket")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Classify"):
    if text.strip() == "":
        st.warning("Enter text first")
    else:
        pred = model.predict([text])[0]

        probs = model.predict_proba([text])[0]
        confidence = max(probs) * 100

        st.success(f"Category: {pred}")

        if confidence < 60:
            st.warning(f"Low confidence: {confidence:.2f}%")
        else:
            st.info(f"Confidence: {confidence:.2f}%")

        st.session_state.history.append((text, pred))

# history
if st.session_state.history:
    st.markdown("### Recent Predictions")
    for t, p in st.session_state.history[-5:][::-1]:
        st.write(f"{t} → {p}")








