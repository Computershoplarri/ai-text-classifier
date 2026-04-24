import pickle
import streamlit as st

# load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# page config
st.set_page_config(page_title="AI Support Ticket Classifier", layout="centered")

# sidebar
st.sidebar.title("⚙️ AI Tool")
st.sidebar.write("Support Ticket Classifier v1")

# main UI
st.title("🎫 AI Support Ticket Classifier")
st.markdown("Automatically categorize customer support messages using AI")

# example button
if st.button("✨ Try Example"):
    st.session_state.text = "My internet is not working properly since morning"

# input
text = st.text_area("✍️ Enter support ticket", height=150, key="text")

# history
if "history" not in st.session_state:
    st.session_state.history = []

# predict
if st.button("🔍 Classify Ticket"):
    if text.strip() == "":
        st.warning("⚠️ Please enter a support message")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)
        confidence = max(probs[0]) * 100

        st.success(f"✅ Category: {pred}")
        st.info(f"Confidence: {confidence:.2f}%")

        # save history
        st.session_state.history.append((text, pred))

# show history
if st.session_state.history:
    st.markdown("### 🕒 Recent Predictions")
    for item in st.session_state.history[-5:][::-1]:
        st.write(f"📌 {item[0]} → **{item[1]}**")






