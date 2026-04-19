from fastapi import FastAPI
import pickle
import os
from sklearn.datasets import fetch_20newsgroups

app = FastAPI()

base_dir = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(base_dir, "model/model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(base_dir, "model/vectorizer.pkl"), "rb"))

data = fetch_20newsgroups(subset="train")
labels = data.target_names

@app.get("/")
def home():
    return {"status": "API working"}

@app.post("/predict")
def predict(text: str):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return {"prediction": labels[pred]}
