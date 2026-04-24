from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# simple dataset
texts = [
    "I want refund",
    "app is not working",
    "I want to buy product",
    "win money click here"
]

labels = [
    "billing",
    "technical",
    "sales",
    "spam"
]

# ONE SIMPLE MODEL (no confusion)
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(texts, labels)

pickle.dump(model, open("model.pkl", "wb"))

print("DONE")





