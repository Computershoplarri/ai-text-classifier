from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import pickle

# -----------------------------
# DATASET (expandable structure)
# -----------------------------
texts = [
    # billing
    "I want a refund for my order",
    "I was charged twice",
    "Payment failed but money deducted",
    "Incorrect invoice generated",
    "Need refund immediately",
    "Billing issue with subscription",

    # technical
    "App is crashing on startup",
    "Unable to login",
    "Server error showing",
    "Internet not working properly",
    "Website is too slow",
    "Bug in dashboard",

    # sales
    "I want pricing details",
    "Do you offer discounts?",
    "I want to buy your product",
    "Tell me about your plans",
    "What is cost of service?",
    "Need subscription info",

    # spam
    "Win money now click link",
    "Claim your free prize",
    "Earn cash fast online",
    "Limited offer click here",
    "You are selected for reward",
    "Free gift waiting for you"
]

labels = [
    "billing","billing","billing","billing","billing","billing",
    "technical","technical","technical","technical","technical","technical",
    "sales","sales","sales","sales","sales","sales",
    "spam","spam","spam","spam","spam","spam"
]

# -----------------------------
# PRODUCTION MODEL PIPELINE
# -----------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000
)

base_model = LinearSVC()

# calibrated model gives probability (confidence)
model = CalibratedClassifierCV(base_model)

pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", model)
])

# train
pipeline.fit(texts, labels)

# save model
pickle.dump(pipeline, open("model.pkl", "wb"))

print("✅ Production model trained successfully")




