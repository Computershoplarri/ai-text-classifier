import pickle
import streamlit as st

# load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# labels
labels = [
    "alt.atheism","comp.graphics","comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x",
    "misc.forsale","rec.autos","rec.motorcycles","rec.sport.baseball",
    "rec.sport.hockey","sci.crypt","sci.electronics","sci.med",
    "sci.space","soc.religion.christian","talk.politics.guns",
    "talk.politics.mideast","talk.politics.misc","talk.religion.misc"
]

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

        category = labels[pred]

        st.success(f"✅ Category: {category}")
        st.info(f"Confidence: {confidence:.2f}%")

        # save history
        st.session_state.history.append((text, category))

# show history
if st.session_state.history:
    st.markdown("### �






