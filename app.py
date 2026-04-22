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

# UI
st.title("🚀 AI Text Classifier")
st.markdown("### Classify any text into categories instantly")

text = st.text_area("✍️ Enter your text here", height=150)

if st.button("🔍 Predict Category"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        st.success(f"✅ Category: {labels[pred]}")





